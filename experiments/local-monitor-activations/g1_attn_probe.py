#!/usr/bin/env python3
"""G1 addendum: which attention implementation can run gpt-oss-20b on long prompts on this pod.

g1_env.py found: eager works but is O(L^2) memory; flex_attention raised an inductor error at
q_len=79 (decode-kernel tile sizes). This probe, with the model loaded once (native MXFP4):
  A. flex_attention: prefill-only forward at 256 tokens; decode (generate) with and without
     explicit `kernel_options`; logits vs eager.
  B. chunked_eager (chunked_attention.py): logits vs eager on the 79-token prompt and on a
     2048-token synthetic prompt (all positions); greedy generation equality with eager;
     then synthetic prefills at 4k..131k tokens (logits_to_keep=1) with peak memory and seconds,
     and a short generation after a 16k prefill to exercise the sliding-window cache path.
Writes results/raw/<slug>/g1_attn_probe_<ts>.json.
"""
from __future__ import annotations

import datetime as dt
import json
import subprocess
import sys
import time
import traceback
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

HERE = Path(__file__).resolve()
SLUG = HERE.parent.name
REPO = HERE.parents[2]
RAW = REPO / "results" / "raw" / SLUG
sys.path.insert(0, str(HERE.parent))
import chunked_attention  # noqa: E402

MODEL_ID = "openai/gpt-oss-20b"
TS = dt.datetime.now(dt.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
OUT = RAW / f"g1_attn_probe_{TS}.json"
R: dict = {"slug": SLUG, "script": str(HERE.relative_to(REPO)), "timestamp": TS, "model_id": MODEL_ID,
           "torch": torch.__version__, "git_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO, text=True).strip(),
           "chunk_target_elems": chunked_attention.TARGET_ELEMS}


def save():
    RAW.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(R, indent=1, default=str))


def err(e):
    return "".join(traceback.format_exception(e))[-1500:]


def peak_reset():
    torch.cuda.empty_cache(); torch.cuda.synchronize(); torch.cuda.reset_peak_memory_stats()


def peak_gib():
    torch.cuda.synchronize(); return round(torch.cuda.max_memory_allocated() / 2**30, 2)


chunked_attention.register()
tok = AutoTokenizer.from_pretrained(MODEL_ID)
model = AutoModelForCausalLM.from_pretrained(MODEL_ID, dtype=torch.bfloat16, device_map={"": 0}, attn_implementation="eager").eval()
R["model_cuda_allocated_gib"] = round(torch.cuda.memory_allocated() / 2**30, 2)
import transformers  # noqa: E402
R["transformers"] = transformers.__version__

msgs = [{"role": "user", "content": "What is the capital of France? Answer in one word."}]
ids79 = tok.apply_chat_template(msgs, add_generation_prompt=True, return_tensors="pt", return_dict=True)["input_ids"].to(model.device)
gen = torch.Generator().manual_seed(0)
ids2k = torch.randint(0, 200000, (1, 2048), generator=gen).to(model.device)
ids256 = ids2k[:, :256]


@torch.no_grad()
def logits_all(ids, **kw):
    return model(ids, use_cache=False, **kw).logits[0].float().cpu()


@torch.no_grad()
def greedy(ids, n=48, **kw):
    g = model.generate(ids, max_new_tokens=n, do_sample=False, **kw)
    return tok.decode(g[0, ids.shape[1]:], skip_special_tokens=False)


# ── reference: eager ─────────────────────────────────────────────────────────
ref = {}
peak_reset(); ref["logits79"] = logits_all(ids79); ref["peak79"] = peak_gib()
peak_reset(); ref["logits2k"] = logits_all(ids2k); ref["peak2k"] = peak_gib()
ref["gen79"] = greedy(ids79)
R["eager"] = {"peak_gib_79": ref["peak79"], "peak_gib_2048": ref["peak2k"], "gen79": ref["gen79"]}
print(json.dumps(R["eager"], indent=1), flush=True); save()


def compare(name, L):
    d = {}
    for key, ids in (("79", ids79), ("2048", ids2k)):
        peak_reset()
        lg = L(ids)
        d[f"peak_gib_{key}"] = peak_gib()
        diff = (lg - ref[f"logits{'79' if key == '79' else '2k'}"]).abs()
        d[f"max_abs_logit_diff_{key}_all_positions"] = round(float(diff.max()), 4)
        d[f"max_abs_logit_diff_{key}_last_position"] = round(float(diff[-1].max()), 4)
        d[f"argmax_agreement_{key}"] = round(float((lg.argmax(-1) == ref[f"logits{'79' if key == '79' else '2k'}"].argmax(-1)).float().mean()), 4)
    return d


# ── A. flex attention ────────────────────────────────────────────────────────
A: dict = {}
try:
    model.set_attn_implementation("flex_attention")
    A["attn_implementation"] = model.config._attn_implementation
    for label, kw in (("default", {}), ("kernel_options_BLOCK64", {"kernel_options": {"BLOCK_M": 64, "BLOCK_N": 64}}),
                      ("kernel_options_fwd32_64", {"kernel_options": {"fwd_BLOCK_M": 32, "fwd_BLOCK_N": 64}})):
        rec: dict = {}
        try:
            t0 = time.time(); lg = logits_all(ids256, **kw); rec["prefill256_seconds"] = round(time.time() - t0, 1)
            rec["prefill256_max_abs_logit_diff_vs_eager_last"] = round(float((lg[-1] - ref["logits2k"][255]).abs().max()), 4) if False else None
            rec["prefill256_ok"] = True
        except Exception as e:
            rec["prefill256_error"] = err(e)
        try:
            rec["compare"] = compare("flex", lambda ids: logits_all(ids, **kw))
        except Exception as e:
            rec["compare_error"] = err(e)
        try:
            rec["gen79"] = greedy(ids79, **kw); rec["gen79_same_as_eager"] = rec["gen79"] == ref["gen79"]
        except Exception as e:
            rec["gen_error"] = err(e)
        A[label] = rec
        print("flex", label, json.dumps({k: v for k, v in rec.items() if "error" not in k}, indent=1), flush=True)
        for k in rec:
            if "error" in k:
                print("  ", k, rec[k].splitlines()[-1][:300], flush=True)
        save()
except Exception as e:
    A["error"] = err(e)
R["flex"] = A; save()

# ── B. chunked eager ─────────────────────────────────────────────────────────
Bd: dict = {}
model.set_attn_implementation(chunked_attention.NAME)
Bd["attn_implementation"] = model.config._attn_implementation
Bd["compare"] = compare("chunked", logits_all)
Bd["gen79"] = greedy(ids79); Bd["gen79_same_as_eager"] = Bd["gen79"] == ref["gen79"]
print("chunked", json.dumps(Bd, indent=1), flush=True); save()

Bd["prefill"] = {}
for L in (4096, 16384, 32768, 65536, 98304, 131072):
    x = torch.randint(0, 200000, (1, L), generator=gen).to(model.device)
    peak_reset()
    try:
        with torch.no_grad():
            torch.cuda.synchronize(); t0 = time.time()
            o = model(x, use_cache=True, logits_to_keep=1)
            torch.cuda.synchronize(); s = time.time() - t0
        rec = {"seconds": round(s, 1), "peak_gib": peak_gib(), "logits_finite": bool(torch.isfinite(o.logits).all()),
               "kv_full_layer_tokens": int(o.past_key_values.get_seq_length(1)),
               "kv_sliding_layer_tokens": int(o.past_key_values.layers[0].keys.shape[-2]) if hasattr(o.past_key_values, "layers") else None}
        if L == 16384:
            t0 = time.time()
            g = model.generate(x, max_new_tokens=16, do_sample=False)
            rec["generate16_after_prefill_seconds"] = round(time.time() - t0, 1)
            rec["generate16_text"] = tok.decode(g[0, L:], skip_special_tokens=False)
            rec["generate16_peak_gib"] = peak_gib()
        del o
    except torch.cuda.OutOfMemoryError as e:
        rec = {"oom": True, "error": str(e)[:300]}
    except Exception as e:
        rec = {"error": err(e)}
    del x; torch.cuda.empty_cache()
    Bd["prefill"][str(L)] = rec
    print(f"chunked prefill L={L}: {rec}", flush=True); save()
    if rec.get("oom") or "error" in rec:
        break
R["chunked_eager"] = Bd
R["finished"] = dt.datetime.now(dt.timezone.utc).isoformat()
save()
print("raw ->", OUT.relative_to(REPO))
