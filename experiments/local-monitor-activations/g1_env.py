#!/usr/bin/env python3
"""G1: environment and hardware verification for the local gpt-oss-20b monitor pass.

Writes results/raw/<slug>/g1_env_<ts>.json with, in order of TASK.md G1:
  1. GPU name / capability / VRAM, torch + CUDA versions      (hard checks: capability (12,0), CUDA >= 12.8)
  2. HF_HOME and free disk on the persistent volume
  3. resolved package versions vs the pins in track L's gpu/requirements.txt
  4. the four mirror payloads (lean + monitor-data for both *monitor* runs),
     cached under checkpoints/<slug>/data/api/, sha256 + size + counts, the
     server fingerprint and a local recompute of it, compared with manifest.json
     (recompute_fingerprint is a verbatim copy of data_export/recon.py's port of
     repro-api computeFingerprint). Also: whether the three L2 export files the
     runner needs are present under checkpoints/<slug>/data/export/.
  5. gpt-oss-20b load on the GPU: memory footprint, the classes and tensor
     dtypes inside two MoE expert modules, whether the quantizer took the
     dequantize path, a short greedy generation, and an eager-vs-flex_attention
     logit comparison (gpt-oss has _supports_sdpa=False; eager attention is
     O(L^2) memory so long prompts need flex_attention), plus peak memory for
     synthetic 4k / 16k-token prefills under flex_attention.

Pure stdlib + torch/transformers.  python experiments/<slug>/g1_env.py
"""
from __future__ import annotations

import datetime as dt
import hashlib
import importlib.metadata as md
import json
import os
import platform
import shutil
import socket
import subprocess
import sys
import time
import traceback
import urllib.request
from pathlib import Path

HERE = Path(__file__).resolve()
SLUG = HERE.parent.name
REPO = HERE.parents[2]
RAW = REPO / "results" / "raw" / SLUG
CKPT = REPO / "checkpoints" / SLUG
DATA_API = CKPT / "data" / "api"
DATA_EXPORT = CKPT / "data" / "export"
GPU_DIR = REPO / "experiments" / "logit-curve-replication" / "gpu"
MANIFEST = GPU_DIR / "manifest.json"
REQS = GPU_DIR / "requirements.txt"
MODEL_ID = "openai/gpt-oss-20b"
TS = dt.datetime.now(dt.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
OUT = RAW / f"g1_env_{TS}.json"

R: dict = {"slug": SLUG, "script": str(HERE.relative_to(REPO)), "timestamp": TS,
           "hostname": socket.gethostname(), "python": sys.version, "platform": platform.platform(),
           "checks": {}, "errors": {}}


def git_commit() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO, text=True).strip()
    except Exception:
        return "unknown"


def sha256_file(p: Path) -> str:
    h = hashlib.sha256()
    with p.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def save():
    RAW.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(R, indent=1, default=str))


def section(name):
    def deco(fn):
        def run():
            print(f"\n=== {name} ===", flush=True)
            t0 = time.time()
            try:
                fn()
            except Exception as e:  # keep going; the raw file records the failure
                R["errors"][name] = "".join(traceback.format_exception(e))
                print(f"!! {name} failed: {e!r}", flush=True)
            R.setdefault("seconds", {})[name] = round(time.time() - t0, 1)
            save()
        return run
    return deco


R["git_commit"] = git_commit()
manifest = json.loads(MANIFEST.read_text())
R["manifest"] = {"path": str(MANIFEST.relative_to(REPO)), "sha256": sha256_file(MANIFEST),
                 "git_commit": manifest["git_commit"], "n_actions": manifest["n_actions"],
                 "n_actions_by_side": manifest["n_actions_by_side"], "pilot_size": manifest["pilot_size"]}


# ── 1. hardware ──────────────────────────────────────────────────────────────
@section("hardware")
def _hardware():
    import torch
    smi = subprocess.check_output(["nvidia-smi", "--query-gpu=name,driver_version,memory.total,compute_cap",
                                   "--format=csv,noheader"], text=True).strip()
    cap = torch.cuda.get_device_capability()
    props = torch.cuda.get_device_properties(0)
    cuda = torch.version.cuda
    hw = {"nvidia_smi": smi, "torch": torch.__version__, "torch_cuda": cuda,
          "cudnn": torch.backends.cudnn.version(), "device_name": props.name,
          "capability": list(cap), "total_memory_bytes": props.total_memory,
          "total_memory_gib": round(props.total_memory / 2**30, 2), "n_gpus": torch.cuda.device_count(),
          "cpu_count": os.cpu_count()}
    R["hardware"] = hw
    major, minor = (int(x) for x in cuda.split(".")[:2])
    R["checks"]["capability_is_12_0"] = (tuple(cap) == (12, 0))
    R["checks"]["cuda_ge_12_8"] = (major, minor) >= (12, 8)
    print(json.dumps(hw, indent=1))


# ── 2. storage ───────────────────────────────────────────────────────────────
@section("storage")
def _storage():
    hf_home = os.environ.get("HF_HOME")
    du = shutil.disk_usage("/workspace")
    R["storage"] = {"HF_HOME": hf_home, "hf_home_on_workspace": bool(hf_home and hf_home.startswith("/workspace")),
                    "workspace_free_gib": round(du.free / 2**30, 1), "workspace_total_gib": round(du.total / 2**30, 1),
                    "checkpoints_dir": str(CKPT.relative_to(REPO)), "data_api_dir": str(DATA_API.relative_to(REPO)),
                    "data_export_dir": str(DATA_EXPORT.relative_to(REPO))}
    R["checks"]["hf_home_on_persistent_volume"] = R["storage"]["hf_home_on_workspace"]
    print(json.dumps(R["storage"], indent=1))


# ── 3. packages ──────────────────────────────────────────────────────────────
@section("packages")
def _packages():
    names = ["torch", "transformers", "tokenizers", "accelerate", "safetensors", "numpy", "triton",
             "kernels", "huggingface_hub", "nnsight", "flash_attn"]
    installed = {}
    for n in names:
        try:
            installed[n] = md.version(n)
        except md.PackageNotFoundError:
            installed[n] = None
    pins = {}
    for line in REQS.read_text().splitlines():
        line = line.split("#", 1)[0].strip()
        if "==" in line:
            k, v = line.split("==")
            pins[k.strip()] = v.strip()
    match = {k: (installed.get(k) or "").split("+")[0] == v for k, v in pins.items()}
    R["packages"] = {"installed": installed, "requirements_txt_pins": pins, "pin_match": match,
                     "requirements_txt": str(REQS.relative_to(REPO))}
    R["checks"]["all_requirements_pins_match"] = all(match.values())
    R["checks"]["kernels_installed"] = installed["kernels"] is not None
    print(json.dumps(R["packages"], indent=1))


# ── 4. mirror payloads ───────────────────────────────────────────────────────
def recompute_fingerprint(trajs: list[dict]) -> str:
    """Verbatim copy of experiments/logit-curve-replication/data_export/recon.py::recompute_fingerprint
    (port of repro-api/app/api/public/lean/[runId]/route.ts::computeFingerprint)."""
    ids = sorted(t.get("trajectory_id") or "" for t in trajs)
    st = sum(1 for t in trajs if t.get("side_task_success"))
    mt = sum(1 for t in trajs if t.get("main_task_success"))
    n_actions = 0
    sus_total = 0.0
    for t in trajs:
        for a in t.get("actions") or []:
            n_actions += 1
            sus_total += (a.get("monitor_response") or {}).get("sus_score") or 0
    ids_hash = hashlib.sha256("|".join(ids).encode()).hexdigest()
    sus_rounded = round(sus_total * 1e4) / 1e4
    if sus_rounded == int(sus_rounded):
        sus_str = str(int(sus_rounded))
    else:
        sus_str = repr(sus_rounded)
    s = (f"http:v3-eval|n_trajs={len(trajs)}|total_actions={n_actions}|st={st}|mt={mt}"
         f"|sus={sus_str}|ids={ids_hash}")
    return hashlib.sha256(s.encode()).hexdigest()


@section("mirror")
def _mirror():
    mirror = manifest["mirror"]
    DATA_API.mkdir(parents=True, exist_ok=True)
    out = {"mirror": mirror, "payloads": {}, "note": "only the four payloads TASK.md G1.4 names were fetched: "
           "lean + monitor-data for the two MONITOR run ids. The manifest's agent_fingerprint values were not "
           "re-checked on the pod (that would need lean/<agent_run_id>)."}
    for side, runs in manifest["runs"].items():
        rid = runs["monitor_run_id"]
        for endpoint in ("lean", "monitor-data"):
            p = DATA_API / f"{endpoint}_{rid}.json"
            fetched_now = False
            if not p.exists():
                url = f"{mirror}/{endpoint}/{rid}"
                print(f"  GET {url}", flush=True)
                t0 = time.time()
                with urllib.request.urlopen(url, timeout=1800) as r:
                    p.write_bytes(r.read())
                fetched_now = True
                print(f"    {p.stat().st_size} bytes in {time.time() - t0:.0f}s", flush=True)
            d = json.loads(p.read_text())
            trajs = d.get("trajectories") or []
            n_actions = sum(len(t.get("actions") or []) for t in trajs)
            n_scored = sum(1 for t in trajs for a in (t.get("actions") or []) if (a.get("monitor_response") or {}).get("sus_score") is not None)
            rec = {"side": side, "endpoint": endpoint, "run_id": rid, "path": str(p.relative_to(REPO)),
                   "fetched_now": fetched_now, "bytes": p.stat().st_size, "sha256": sha256_file(p),
                   "top_level_keys": sorted(d.keys()), "n_trajectories": len(trajs), "n_actions": n_actions,
                   "n_actions_with_sus_score": n_scored,
                   "trajectory_keys_example": sorted(trajs[0].keys()) if trajs else None,
                   "action_keys_example": sorted(trajs[0]["actions"][0].keys()) if trajs and trajs[0].get("actions") else None}
            if endpoint == "lean":
                rec["server_fingerprint"] = d.get("fingerprint")
                rec["recomputed_fingerprint"] = recompute_fingerprint(trajs)
                rec["manifest_monitor_fingerprint"] = runs["monitor_fingerprint"]
                rec["server_matches_manifest"] = rec["server_fingerprint"] == runs["monitor_fingerprint"]
                rec["recompute_matches_manifest"] = rec["recomputed_fingerprint"] == runs["monitor_fingerprint"]
            else:
                # monitor-data carries no fingerprint; recompute the same formula from its rows
                rec["recomputed_fingerprint"] = recompute_fingerprint(trajs)
                rec["recompute_matches_manifest"] = rec["recomputed_fingerprint"] == runs["monitor_fingerprint"]
            out["payloads"][f"{endpoint}_{side}"] = rec
            print(json.dumps({k: v for k, v in rec.items() if k not in ("trajectory_keys_example", "action_keys_example")}, indent=1), flush=True)
    R["mirror"] = out
    R["checks"]["lean_fingerprints_match_manifest"] = all(
        v["server_matches_manifest"] and v["recompute_matches_manifest"] for k, v in out["payloads"].items() if k.startswith("lean"))
    R["checks"]["monitor_data_recompute_matches_manifest"] = all(
        v["recompute_matches_manifest"] for k, v in out["payloads"].items() if k.startswith("monitor-data"))
    R["checks"]["lean_and_monitor_data_action_counts_agree"] = all(
        out["payloads"][f"lean_{s}"]["n_actions"] == out["payloads"][f"monitor-data_{s}"]["n_actions"] for s in manifest["runs"])

    # the three L2 export files run_monitor.py needs (not in git, not on the mirror)
    files = {}
    for name, c in manifest["data_files"].items():
        p = DATA_EXPORT / name
        rec = {"expected_sha256": c["sha256"], "expected_bytes": c["bytes"], "present": p.exists()}
        if p.exists():
            rec["sha256"] = sha256_file(p)
            rec["bytes"] = p.stat().st_size
            rec["sha256_matches_manifest"] = rec["sha256"] == c["sha256"]
        files[name] = rec
    R["data_files"] = {"dir": str(DATA_EXPORT.relative_to(REPO)), "files": files,
                       "note": "honest.jsonl / attack.jsonl / trajectories.jsonl are built by track L's data_export/export.py "
                               "from the mirror PLUS the 16 GB HF dump (tool arguments and task descriptions are not on the mirror). "
                               "They must be copied from the laptop; run_monitor.py refuses to start without them."}
    R["checks"]["export_data_files_present_and_match"] = all(f.get("sha256_matches_manifest") for f in files.values())
    print(json.dumps(R["data_files"], indent=1))


# ── 5. model ─────────────────────────────────────────────────────────────────
def tensor_attrs(mod, depth=2, prefix="") -> dict:
    """dtype/shape of every tensor-like attribute reachable from a module (params, buffers, plain attrs, wrappers)."""
    found = {}
    seen = set()

    def walk(obj, name, d):
        if id(obj) in seen or d < 0:
            return
        seen.add(id(obj))
        if hasattr(obj, "dtype") and hasattr(obj, "shape"):
            try:
                found[name] = {"type": type(obj).__name__, "dtype": str(obj.dtype), "shape": list(obj.shape),
                               "device": str(getattr(obj, "device", "?"))}
            except Exception:
                found[name] = {"type": type(obj).__name__}
            return
        if isinstance(obj, (int, float, str, bool, type(None))):
            return
        try:
            items = list(vars(obj).items())
        except TypeError:
            return
        for k, v in items:
            if k.startswith("_") and k not in ("_parameters", "_buffers"):
                continue
            if isinstance(v, dict):
                for kk, vv in v.items():
                    walk(vv, f"{name}.{kk}" if name else kk, d - 1)
            else:
                walk(v, f"{name}.{k}" if name else k, d - 1)

    walk(mod, prefix, depth)
    return found


@section("model")
def _model():
    import torch
    import transformers
    from huggingface_hub import snapshot_download
    from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

    snap = Path(snapshot_download(MODEL_ID, allow_patterns=["*.json", "*.safetensors", "*.jinja", "*.txt"]))
    st_files = sorted(snap.glob("*.safetensors"))
    cfg = AutoConfig.from_pretrained(MODEL_ID)
    lt = list(cfg.layer_types)
    m: dict = {"model_id": MODEL_ID, "snapshot": str(snap), "snapshot_commit": snap.name,
               "safetensors_files": [{"name": p.name, "bytes": p.resolve().stat().st_size} for p in st_files],
               "safetensors_total_gib": round(sum(p.resolve().stat().st_size for p in st_files) / 2**30, 2),
               "config": {"quantization_config": cfg.quantization_config if isinstance(cfg.quantization_config, dict) else cfg.quantization_config.to_dict(),
                          "num_hidden_layers": cfg.num_hidden_layers, "hidden_size": cfg.hidden_size,
                          "num_attention_heads": cfg.num_attention_heads, "num_key_value_heads": cfg.num_key_value_heads,
                          "head_dim": cfg.head_dim, "sliding_window": cfg.sliding_window,
                          "layer_types": lt, "n_sliding": lt.count("sliding_attention"), "n_full": lt.count("full_attention"),
                          "max_position_embeddings": cfg.max_position_embeddings,
                          "num_local_experts": cfg.num_local_experts, "num_experts_per_tok": cfg.num_experts_per_tok,
                          "vocab_size": cfg.vocab_size},
               "attention_support": {"_supports_sdpa": getattr(transformers.GptOssPreTrainedModel, "_supports_sdpa", None),
                                     "_supports_flash_attn": getattr(transformers.GptOssPreTrainedModel, "_supports_flash_attn", None),
                                     "_supports_flex_attn": getattr(transformers.GptOssPreTrainedModel, "_supports_flex_attn", None),
                                     "flash_attn_installed": R["packages"]["installed"].get("flash_attn") is not None}}
    R["model"] = m
    save()

    tok = AutoTokenizer.from_pretrained(MODEL_ID)
    ids10 = tok.encode("10", add_special_tokens=False)
    m["tokenizer"] = {"class": type(tok).__name__, "vocab_size": len(tok),
                      "digit_tokens": {s: tok.encode(s, add_special_tokens=False) for s in [str(i) for i in range(11)] + [" 10", " 1", "10</"]},
                      "'10'_is_single_token": len(ids10) == 1,
                      "eos_token": tok.eos_token, "eos_token_id": tok.eos_token_id, "pad_token_id": tok.pad_token_id,
                      "special": {s: tok.convert_tokens_to_ids(s) for s in ("<|return|>", "<|call|>", "<|start|>", "<|channel|>", "<|message|>", "<|end|>")}}
    # preview only — G3 settles the capture position on real prompts
    demo = tok.apply_chat_template([{"role": "user", "content": "hi"}], add_generation_prompt=True, tokenize=False)
    demo_ids = tok.encode(demo, add_special_tokens=False)
    m["tokenizer"]["chat_template_preview"] = {"text": demo, "n_tokens": len(demo_ids),
                                               "last_8": [{"id": i, "tok": tok.decode([i])} for i in demo_ids[-8:]]}
    print(json.dumps(m["tokenizer"], indent=1, default=str), flush=True)
    save()

    torch.cuda.reset_peak_memory_stats()
    t0 = time.time()
    model = AutoModelForCausalLM.from_pretrained(MODEL_ID, dtype=torch.bfloat16, device_map={"": 0}, attn_implementation="eager")
    model.eval()
    load_s = time.time() - t0
    torch.cuda.synchronize()
    hq = getattr(model, "hf_quantizer", None)
    qc = getattr(model.config, "quantization_config", None)
    experts = {}
    for li in (0, 12, 23):
        e = model.model.layers[li].mlp.experts
        experts[f"layer_{li}"] = {"class": type(e).__name__, "tensors": tensor_attrs(e, depth=3)}
    n_params = sum(p.numel() for p in model.parameters())
    dtype_hist: dict[str, int] = {}
    for p in model.parameters():
        dtype_hist[str(p.dtype)] = dtype_hist.get(str(p.dtype), 0) + p.numel()
    for _, b in model.named_buffers():
        dtype_hist["buffer:" + str(b.dtype)] = dtype_hist.get("buffer:" + str(b.dtype), 0) + b.numel()
    m["load"] = {"seconds": round(load_s, 1), "attn_implementation": model.config._attn_implementation,
                 "dtype_arg": "bfloat16", "device_map": {"": 0},
                 "get_memory_footprint_bytes": model.get_memory_footprint(),
                 "get_memory_footprint_gib": round(model.get_memory_footprint() / 2**30, 2),
                 "cuda_memory_allocated_gib": round(torch.cuda.memory_allocated() / 2**30, 2),
                 "cuda_max_memory_allocated_gib": round(torch.cuda.max_memory_allocated() / 2**30, 2),
                 "hf_quantizer_class": type(hq).__name__ if hq is not None else None,
                 "quantizer_dequantize_flag": getattr(getattr(hq, "quantization_config", None), "dequantize", None),
                 "config_quantization_config": qc if isinstance(qc, dict) else (qc.to_dict() if qc is not None else None),
                 "n_parameters_registered": n_params, "parameter_numel_by_dtype": dtype_hist,
                 "experts": experts, "n_decoder_layers": len(model.model.layers)}
    print(json.dumps({k: v for k, v in m["load"].items() if k != "experts"}, indent=1, default=str), flush=True)
    for k, v in experts.items():
        print(k, v["class"]); [print("   ", n, t) for n, t in v["tensors"].items()]
    fp_gib = m["load"]["get_memory_footprint_gib"]
    R["checks"]["model_footprint_under_20gib"] = fp_gib < 20      # bf16 dequantised would be ~40 GiB
    R["checks"]["quantizer_did_not_dequantize"] = m["load"]["quantizer_dequantize_flag"] is False
    R["checks"]["expert_weights_not_bf16"] = all(
        not any(t["dtype"] == "torch.bfloat16" and any(s > 2880 for s in t["shape"]) and "bias" not in n
                for n, t in v["tensors"].items()) for v in experts.values())
    save()

    # short generation, eager
    msgs = [{"role": "user", "content": "What is the capital of France? Answer in one word."}]
    enc = tok.apply_chat_template(msgs, add_generation_prompt=True, return_tensors="pt", return_dict=True)
    ids = enc["input_ids"].to(model.device)
    tests: dict = {"prompt_n_tokens": int(ids.shape[1])}
    with torch.no_grad():
        torch.cuda.reset_peak_memory_stats()
        out = model(ids)
        logits_eager = out.logits[0, -1].float().cpu()
        top = torch.topk(torch.softmax(logits_eager, -1), 5)
        tests["eager"] = {"top5_next_token": [{"tok": tok.decode([int(i)]), "p": round(float(p), 4)} for p, i in zip(top.values, top.indices)],
                          "peak_gib": round(torch.cuda.max_memory_allocated() / 2**30, 2)}
        t0 = time.time()
        g = model.generate(ids, max_new_tokens=64, do_sample=False)
        tests["eager"]["generate_64_seconds"] = round(time.time() - t0, 1)
        tests["eager"]["generated_text"] = tok.decode(g[0, ids.shape[1]:], skip_special_tokens=False)
    print(json.dumps(tests, indent=1), flush=True)
    save()

    # flex attention: same prompt, compare logits; then synthetic long prefills
    try:
        model.set_attn_implementation("flex_attention")
        tests["flex"] = {"attn_implementation": model.config._attn_implementation}
        with torch.no_grad():
            t0 = time.time()
            out = model(ids)
            tests["flex"]["first_call_seconds_incl_compile"] = round(time.time() - t0, 1)
            logits_flex = out.logits[0, -1].float().cpu()
            top = torch.topk(torch.softmax(logits_flex, -1), 5)
            tests["flex"]["top5_next_token"] = [{"tok": tok.decode([int(i)]), "p": round(float(p), 4)} for p, i in zip(top.values, top.indices)]
            tests["flex"]["max_abs_logit_diff_vs_eager"] = round(float((logits_flex - logits_eager).abs().max()), 4)
            tests["flex"]["argmax_same_as_eager"] = bool(logits_flex.argmax() == logits_eager.argmax())
            g = model.generate(ids, max_new_tokens=64, do_sample=False)
            tests["flex"]["generated_text"] = tok.decode(g[0, ids.shape[1]:], skip_special_tokens=False)
            tests["flex"]["generated_text_same_as_eager"] = tests["flex"]["generated_text"] == tests["eager"]["generated_text"]
            print(json.dumps(tests["flex"], indent=1), flush=True)
            save()
            tests["flex"]["synthetic_prefill"] = {}
            gen = torch.Generator().manual_seed(0)
            for L in (4096, 16384):
                torch.cuda.empty_cache()
                torch.cuda.reset_peak_memory_stats()
                x = torch.randint(0, 200000, (1, L), generator=gen).to(model.device)
                try:
                    torch.cuda.synchronize(); t0 = time.time()
                    o = model(x, use_cache=True)
                    torch.cuda.synchronize()
                    rec = {"seconds": round(time.time() - t0, 2), "peak_gib": round(torch.cuda.max_memory_allocated() / 2**30, 2),
                           "logits_finite": bool(torch.isfinite(o.logits[0, -1]).all())}
                    del o
                except torch.cuda.OutOfMemoryError as e:
                    rec = {"oom": True, "error": str(e)[:300]}
                del x
                torch.cuda.empty_cache()
                tests["flex"]["synthetic_prefill"][str(L)] = rec
                print(f"  flex prefill L={L}: {rec}", flush=True)
                save()
    except Exception as e:
        tests["flex_error"] = "".join(traceback.format_exception(e))
        print("!! flex attention failed:", repr(e), flush=True)
    m["tests"] = tests
    R["checks"]["eager_generation_produced_final_channel"] = "<|channel|>final<|message|>" in tests["eager"]["generated_text"]
    R["checks"]["flex_attention_works"] = "flex" in tests and tests["flex"].get("argmax_same_as_eager", False)
    save()
    del model
    torch.cuda.empty_cache()


if __name__ == "__main__":
    for step in (_hardware, _storage, _packages, _mirror, _model):
        step()
    R["finished"] = dt.datetime.now(dt.timezone.utc).isoformat()
    save()
    print("\n=== checks ===")
    print(json.dumps(R["checks"], indent=1))
    print("errors:", list(R["errors"]))
    print(f"raw -> {OUT.relative_to(REPO)}")
