"""Chunked eager attention for gpt-oss (attention sinks + sliding window) at O(L) memory.

transformers' `eager` path materialises the full [B, H, Q, K] score matrix per layer (64 heads x
25k x 25k x 2 bytes = 80 GB for a median prompt), gpt-oss has `_supports_sdpa = False`, and
flex_attention failed to compile on this pod (see g1_env raw). This function computes exactly
the same quantity as `modeling_gpt_oss.eager_attention_forward` — scores, additive causal /
sliding-window mask, sink column appended before the softmax, sink dropped after — but over
query chunks, so peak extra memory is ~chunk x K instead of Q x K. Sliding-window layers only
touch the 128-key band each chunk can see.

Registered as attention implementation "chunked_eager":

    import chunked_attention; chunked_attention.register()
    model = AutoModelForCausalLM.from_pretrained(..., attn_implementation="chunked_eager")

transformers builds no mask for an implementation it does not know (masking_utils returns
None), so the mask is derived here from `cache_position` (absolute query positions) and the
key length: keys are the contiguous run of positions ending at the last query position, which
holds for DynamicCache full layers and for the cropped sliding-window layers alike.
Batch size 1 only (no padding handling). Inference only (dropout ignored).
"""
from __future__ import annotations

import torch

NAME = "chunked_eager"
TARGET_ELEMS = 1 << 28          # ~1 GiB of fp32 scores per chunk (used only when CHUNK_SIZE is None)
CHUNK_SIZE: int | None = None   # fixed query-chunk length; set it (run_monitor --attn-chunk-size) so the
                                # reduction order is the same for every prompt in a run (G1: chunking changes
                                # logits by up to ~1.5 through MoE routing flips)


def chunked_eager_attention_forward(module, query, key, value, attention_mask=None, scaling=None,
                                    dropout=0.0, sliding_window=None, s_aux=None, cache_position=None,
                                    position_ids=None, **kwargs):
    B, H, Q, D = query.shape
    Hkv, K = key.shape[1], key.shape[2]
    assert B == 1, "chunked_eager handles batch size 1 only"
    groups = H // Hkv
    if scaling is None:
        scaling = D ** -0.5
    dev = query.device
    if cache_position is not None:
        q_pos = cache_position.to(dev)
    elif position_ids is not None:
        q_pos = position_ids[0].to(dev)
    else:
        q_pos = torch.arange(Q, device=dev)
    assert q_pos.shape[0] == Q
    k0 = int(q_pos[-1]) - K + 1                      # absolute position of key index 0
    sinks = s_aux if s_aux is not None else getattr(module, "sinks", None)
    sinks = sinks.to(torch.float32).view(1, H, 1, 1)

    out = torch.empty(B, H, Q, D, dtype=value.dtype, device=dev)
    if CHUNK_SIZE is not None:
        chunk = max(1, min(Q, int(CHUNK_SIZE)))
    else:
        chunk = max(16, min(Q, TARGET_ELEMS // max(1, H * (K if sliding_window is None else min(K, Q + sliding_window)))))
    for s in range(0, Q, chunk):
        e = min(Q, s + chunk)
        qp = q_pos[s:e]                                                    # [c]
        hi = int(qp[-1]) - k0 + 1                                          # causal cut (exclusive key index)
        lo = 0 if sliding_window is None else max(0, int(qp[0]) - sliding_window + 1 - k0)
        hi = max(min(hi, K), lo + 1)
        k = key[:, :, lo:hi]                                               # [B, Hkv, K', D]
        v = value[:, :, lo:hi]
        kp = k0 + torch.arange(lo, hi, device=dev)                         # [K']
        allowed = kp[None, :] <= qp[:, None]
        if sliding_window is not None:
            allowed &= (qp[:, None] - kp[None, :]) < sliding_window
        q = query[:, :, s:e].reshape(B, Hkv, groups, e - s, D)
        scores = torch.matmul(q, k.unsqueeze(2).transpose(-1, -2)) * scaling   # [B, Hkv, g, c, K'] (query dtype)
        scores = scores.reshape(B, H, e - s, hi - lo).float()
        scores = scores.masked_fill(~allowed[None, None], float("-inf"))
        combined = torch.cat([scores, sinks.expand(B, H, e - s, 1)], dim=-1)
        combined = combined - combined.max(dim=-1, keepdim=True).values
        probs = torch.softmax(combined, dim=-1)[..., :-1].to(v.dtype)       # drop the sink column
        probs = probs.reshape(B, Hkv, groups, e - s, hi - lo)
        out[:, :, s:e] = torch.matmul(probs, v.unsqueeze(2)).reshape(B, H, e - s, D)
        del scores, combined, probs
    return out.transpose(1, 2).contiguous(), None


def register():
    from transformers import AttentionInterface
    if NAME not in AttentionInterface._global_mapping:
        AttentionInterface.register(NAME, chunked_eager_attention_forward)
    return NAME
