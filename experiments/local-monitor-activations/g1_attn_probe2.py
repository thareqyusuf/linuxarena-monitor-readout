#!/usr/bin/env python3
"""G1 addendum 2: is chunked_eager within eager's own run-to-run noise, and does the multi-chunk
path (several query chunks, sliding-window band slicing with lo > 0) agree with the single-chunk one?

On two prompts (the 79-token chat prompt; 2048 synthetic random tokens), all logits, all positions:
  eager x2, chunked_eager single-chunk x2 (TARGET_ELEMS=2^28 -> one chunk at these lengths),
  chunked_eager multi-chunk (TARGET_ELEMS=2^20 -> 16-token chunks at 2048), flex_attention (fwd kernel opts).
Reports max / mean / p99 |logit diff| and argmax agreement of every run against eager run 1, and
eager2-vs-eager1 as the noise floor. Writes results/raw/<slug>/g1_attn_probe2_<ts>.json.
"""
from __future__ import annotations

import datetime as dt
import json
import subprocess
import sys
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
OUT = RAW / f"g1_attn_probe2_{TS}.json"
R: dict = {"slug": SLUG, "script": str(HERE.relative_to(REPO)), "timestamp": TS, "model_id": MODEL_ID,
           "torch": torch.__version__, "git_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO, text=True).strip()}

chunked_attention.register()
tok = AutoTokenizer.from_pretrained(MODEL_ID)
model = AutoModelForCausalLM.from_pretrained(MODEL_ID, dtype=torch.bfloat16, device_map={"": 0}, attn_implementation="eager").eval()
msgs = [{"role": "user", "content": "What is the capital of France? Answer in one word."}]
prompts = {"chat79": tok.apply_chat_template(msgs, add_generation_prompt=True, return_tensors="pt", return_dict=True)["input_ids"].to(model.device),
           "rand2048": torch.randint(0, 200000, (1, 2048), generator=torch.Generator().manual_seed(0)).to(model.device)}


@torch.no_grad()
def logits(ids, **kw):
    return model(ids, use_cache=False, **kw).logits[0].float().cpu()


def stats(a, ref):
    d = (a - ref).abs()
    per_pos = d.max(-1).values
    return {"max": round(float(d.max()), 4), "mean": round(float(d.mean()), 5), "p99": round(float(d.flatten().kthvalue(int(0.99 * d.numel())).values), 4),
            "last_pos_max": round(float(d[-1].max()), 4), "argmax_agreement": round(float((a.argmax(-1) == ref.argmax(-1)).float().mean()), 4),
            "n_positions_with_maxdiff_gt_1": int((per_pos > 1).sum()), "n_positions": int(per_pos.numel())}


runs = [("eager_1", "eager", 1 << 28, {}), ("eager_2", "eager", 1 << 28, {}),
        ("chunked_single_1", "chunked_eager", 1 << 28, {}), ("chunked_single_2", "chunked_eager", 1 << 28, {}),
        ("chunked_multi", "chunked_eager", 1 << 20, {}),
        ("flex_fwd32_64", "flex_attention", 1 << 28, {"kernel_options": {"fwd_BLOCK_M": 32, "fwd_BLOCK_N": 64}})]
R["runs"] = {name: {"attn": attn, "chunk_target_elems": te, "kwargs": kw} for name, attn, te, kw in runs}
R["results"] = {}
for pname, ids in prompts.items():
    out = {}
    res = {}
    for name, attn, te, kw in runs:
        try:
            model.set_attn_implementation(attn)
            chunked_attention.TARGET_ELEMS = te
            H, Q = 64, ids.shape[1]
            out[name] = logits(ids, **kw)
            res[name] = {"ok": True, "chunk_tokens_at_full_layer": max(16, min(Q, te // (H * Q))) if attn == "chunked_eager" else None}
        except Exception as e:
            res[name] = {"ok": False, "error": repr(e)[:400]}
    ref = out["eager_1"]
    for name in out:
        if name != "eager_1":
            res[name]["vs_eager_1"] = stats(out[name], ref)
    if "chunked_single_1" in out and "chunked_multi" in out:
        res["chunked_multi"]["vs_chunked_single_1"] = stats(out["chunked_multi"], out["chunked_single_1"])
    R["results"][pname] = res
    print(pname, json.dumps(res, indent=1), flush=True)
    RAW.mkdir(parents=True, exist_ok=True); OUT.write_text(json.dumps(R, indent=1))
# 131k prefill retry: probe 1 OOMed here with 22.0 GiB allocated + 6.2 GiB reserved-but-fragmented;
# this run is launched with PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True to see if that is enough.
import os, time
R["PYTORCH_CUDA_ALLOC_CONF"] = os.environ.get("PYTORCH_CUDA_ALLOC_CONF")
model.set_attn_implementation("chunked_eager"); chunked_attention.TARGET_ELEMS = 1 << 28
for L in (131072,):
    x = torch.randint(0, 200000, (1, L), generator=torch.Generator().manual_seed(1)).to(model.device)
    torch.cuda.empty_cache(); torch.cuda.synchronize(); torch.cuda.reset_peak_memory_stats()
    try:
        with torch.no_grad():
            t0 = time.time(); o = model(x, use_cache=True, logits_to_keep=1); torch.cuda.synchronize()
        rec = {"seconds": round(time.time() - t0, 1), "peak_gib": round(torch.cuda.max_memory_allocated() / 2**30, 2),
               "reserved_gib": round(torch.cuda.memory_reserved() / 2**30, 2), "logits_finite": bool(torch.isfinite(o.logits).all())}
        del o
    except torch.cuda.OutOfMemoryError as e:
        rec = {"oom": True, "error": str(e)[:400]}
    del x; torch.cuda.empty_cache()
    R.setdefault("prefill_expandable_segments", {})[str(L)] = rec
    print(f"chunked prefill (expandable_segments) L={L}: {rec}", flush=True)
R["finished"] = dt.datetime.now(dt.timezone.utc).isoformat()
OUT.write_text(json.dumps(R, indent=1))
print("raw ->", OUT.relative_to(REPO))
