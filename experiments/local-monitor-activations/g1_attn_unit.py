#!/usr/bin/env python3
"""G1 addendum 3: function-level test of chunked_attention vs transformers' eager_attention_forward
vs an fp64 reference of the same formula, on random q/k/v with gpt-oss shapes (64 q-heads, 8 kv-heads,
head_dim 64, sinks), prefill and decode, with and without the 128-token sliding window. No model, no MoE:
this isolates the attention arithmetic from the routing amplification seen in probe 2.
Writes results/raw/<slug>/g1_attn_unit_<ts>.json.
"""
from __future__ import annotations

import datetime as dt
import json
import subprocess
import sys
from pathlib import Path

import torch
from transformers.models.gpt_oss.modeling_gpt_oss import eager_attention_forward

HERE = Path(__file__).resolve()
SLUG = HERE.parent.name
REPO = HERE.parents[2]
RAW = REPO / "results" / "raw" / SLUG
sys.path.insert(0, str(HERE.parent))
import chunked_attention  # noqa: E402

TS = dt.datetime.now(dt.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
OUT = RAW / f"g1_attn_unit_{TS}.json"
dev = "cuda"
H, Hkv, D = 64, 8, 64
scaling = D ** -0.5


class Mod(torch.nn.Module):
    def __init__(self, sinks, sliding_window):
        super().__init__()
        self.sinks = torch.nn.Parameter(sinks)
        self.num_key_value_groups = H // Hkv
        self.sliding_window = sliding_window
        self.training = False


def additive_mask(q_pos, k_pos, window, dtype):
    allowed = k_pos[None, :] <= q_pos[:, None]
    if window is not None:
        allowed &= (q_pos[:, None] - k_pos[None, :]) < window
    m = torch.zeros(allowed.shape, dtype=dtype, device=dev)
    m.masked_fill_(~allowed, torch.finfo(dtype).min)
    return m[None, None]


def ref64(q, k, v, sinks, q_pos, k_pos, window):
    """Same formula in fp64: scores, mask, sink column, softmax, drop sink, PV."""
    q, k, v = q.double(), k.double(), v.double()
    k = k.repeat_interleave(H // Hkv, dim=1); v = v.repeat_interleave(H // Hkv, dim=1)
    s = torch.matmul(q, k.transpose(-1, -2)) * scaling
    allowed = (k_pos[None, :] <= q_pos[:, None])
    if window is not None:
        allowed &= (q_pos[:, None] - k_pos[None, :]) < window
    s = s.masked_fill(~allowed[None, None], float("-inf"))
    comb = torch.cat([s, sinks.double().view(1, H, 1, 1).expand(1, H, s.shape[2], 1)], -1)
    p = torch.softmax(comb, -1)[..., :-1]
    return torch.matmul(p, v).transpose(1, 2)


def stats(a, b):
    d = (a.double() - b.double()).abs()
    return {"max_abs": float(d.max()), "mean_abs": float(d.mean()), "frac_elems_diff_gt_1e-3": float((d > 1e-3).float().mean()),
            "max_abs_over_ref_scale": float(d.max() / b.double().abs().max())}


gen = torch.Generator(device=dev).manual_seed(0)
R: dict = {"slug": SLUG, "script": str(HERE.relative_to(REPO)), "timestamp": TS, "torch": torch.__version__,
           "git_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO, text=True).strip(),
           "shapes": {"H": H, "Hkv": Hkv, "D": D}, "cases": {}}
cases = [("prefill_79", 79, 79, None, 1 << 28), ("prefill_79_sw128", 79, 79, 128, 1 << 28),
         ("prefill_2048", 2048, 2048, None, 1 << 20), ("prefill_2048_sw128", 2048, 2048, 128, 1 << 20),
         ("prefill_4096_chunk16", 4096, 4096, None, 1 << 16), ("prefill_4096_sw128_chunk16", 4096, 4096, 128, 1 << 16),
         ("decode_q1_k300", 1, 300, None, 1 << 28), ("decode_q1_k128_sw128", 1, 128, 128, 1 << 28)]
cases = [(f"{n}__sinks_{sd}", Q, K, w, te, sd) for (n, Q, K, w, te) in cases for sd in ("fp32", "bf16")]
R["note"] = ("sinks_bf16 mirrors the loaded model (sinks are bf16 params, g1_env parameter_numel_by_dtype): eager then cats bf16 "
             "scores with bf16 sinks and runs the softmax in bf16; chunked_attention casts sinks to fp32 and softmaxes in fp32.")
for name, Q, K, window, te, sd in cases:
    q = torch.randn(1, H, Q, D, device=dev, generator=gen, dtype=torch.float32).to(torch.bfloat16)
    k = torch.randn(1, Hkv, K, D, device=dev, generator=gen, dtype=torch.float32).to(torch.bfloat16)
    v = torch.randn(1, Hkv, K, D, device=dev, generator=gen, dtype=torch.float32).to(torch.bfloat16)
    sinks = (torch.randn(H, device=dev, generator=gen) * 2)
    sinks = sinks.to(torch.bfloat16) if sd == "bf16" else sinks
    q_pos = torch.arange(K - Q, K, device=dev)          # queries are the last Q positions of K keys
    k_pos = torch.arange(K, device=dev)
    mod = Mod(sinks, window)
    mask = additive_mask(q_pos, k_pos, window, torch.bfloat16)
    with torch.no_grad():
        e, _ = eager_attention_forward(mod, q, k, v, mask, scaling=scaling, dropout=0.0)
        chunked_attention.TARGET_ELEMS = te
        c, _ = chunked_attention.chunked_eager_attention_forward(mod, q, k, v, None, scaling=scaling, sliding_window=window,
                                                                 s_aux=mod.sinks, cache_position=q_pos)
        r = ref64(q, k, v, sinks, q_pos, k_pos, window)
    n_chunks = -(-Q // max(16, min(Q, te // (H * (K if window is None else min(K, Q + window))))))
    rec = {"Q": Q, "K": K, "sliding_window": window, "sinks_dtype": sd, "chunked_n_chunks": n_chunks,
           "eager_vs_ref64": stats(e, r), "chunked_vs_ref64": stats(c, r), "chunked_vs_eager": stats(c, e),
           "chunked_eager_bitwise_equal": bool(torch.equal(c, e))}
    R["cases"][name] = rec
    print(name, json.dumps(rec), flush=True)
RAW.mkdir(parents=True, exist_ok=True)
OUT.write_text(json.dumps(R, indent=1))
print("raw ->", OUT.relative_to(REPO))
