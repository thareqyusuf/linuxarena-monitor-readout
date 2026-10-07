#!/usr/bin/env python
"""Check whether a model and its activation cache will fit before you commit.

    python preflight.py --model Qwen/Qwen3-8B --batch 16 --seq 512
    python preflight.py --model google/gemma-3-12b-it --batch 8 --seq 256 --load

Without --load this reads only the config off the Hub, so it costs a second
and no disk. With --load it actually pulls the weights and reports measured
VRAM, which is the number that counts.

Why this script exists: people size a GPU by weight bytes -- 9B params at bf16
is 18GB, so a 24GB card looks fine. Then they cache attention patterns and OOM,
because patterns scale with seq^2 and nothing about the weight count warns you.
"""

import argparse
import sys

GIB = 1024**3


def human(n_bytes: float) -> str:
    return f"{n_bytes / GIB:6.2f} GiB"


def cfg_get(cfg, *names, default=None):
    """Config field names vary across architectures; try several."""
    for n in names:
        v = getattr(cfg, n, None)
        if v is not None:
            return v
    text_cfg = getattr(cfg, "text_config", None)  # multimodal models nest it
    if text_cfg is not None:
        for n in names:
            v = getattr(text_cfg, n, None)
            if v is not None:
                return v
    return default


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, help="HuggingFace model id")
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--seq", type=int, default=256)
    ap.add_argument("--dtype", default="bfloat16", choices=["bfloat16", "float16", "float32"])
    ap.add_argument("--load", action="store_true", help="actually load the weights and measure")
    ap.add_argument("--backward", action="store_true", help="assume gradients (attribution patching)")
    args = ap.parse_args()

    import torch
    from transformers import AutoConfig

    bytes_per = {"bfloat16": 2, "float16": 2, "float32": 4}[args.dtype]

    if not torch.cuda.is_available():
        print("No CUDA device visible. Nothing to budget against.")
        return 1

    props = torch.cuda.get_device_properties(0)
    total = props.total_memory
    print(f"\nGPU: {props.name}   {human(total)} total\n")

    try:
        cfg = AutoConfig.from_pretrained(args.model, trust_remote_code=True)
    except Exception as e:  # gated repo, typo'd id, no network
        print(f"Could not read config for {args.model!r}:\n  {e}\n")
        print("If this is a 401/403, the repo is gated: accept the licence on the")
        print("model page while signed in, then set HF_TOKEN in /workspace/.env.")
        return 1

    n_layer = cfg_get(cfg, "num_hidden_layers", "n_layer", "num_layers")
    d_model = cfg_get(cfg, "hidden_size", "n_embd", "d_model")
    n_head = cfg_get(cfg, "num_attention_heads", "n_head")
    if None in (n_layer, d_model, n_head):
        print(f"Unrecognised config shape for {args.model}. Fields found: {cfg}")
        return 1

    B, S = args.batch, args.seq

    # One residual-stream tensor per layer.
    resid = n_layer * B * S * d_model * bytes_per
    # Full attention patterns, every layer, every head. The seq^2 term.
    patterns = n_layer * B * n_head * S * S * bytes_per
    # Rough stand-in for caching several hook points per layer (resid pre/mid/post,
    # attn out, mlp out). Order-of-magnitude, not exact.
    hooks = 5 * resid

    print(f"Model:   {args.model}")
    print(f"Shape:   {n_layer} layers, d_model {d_model}, {n_head} heads")
    print(f"Batch:   {B} x {S} tokens, {args.dtype}\n")

    if args.load:
        from transformers import AutoModelForCausalLM

        print("Loading weights (first run downloads; later runs hit HF_HOME cache)...")
        torch.cuda.reset_peak_memory_stats()
        model = AutoModelForCausalLM.from_pretrained(
            args.model,
            dtype=getattr(torch, args.dtype),
            device_map="cuda:0",
            trust_remote_code=True,
        )
        weights = torch.cuda.memory_allocated()
        del model
        print(f"  measured weights:        {human(weights)}\n")
    else:
        n_params = cfg_get(cfg, "num_parameters", default=None)
        if n_params is None:
            # 12 * n_layer * d_model^2 is the standard transformer estimate;
            # it ignores embeddings and is wrong for MoE and GQA models.
            n_params = 12 * n_layer * d_model**2
            print("  (parameter count estimated from config -- pass --load to measure)")
        weights = n_params * bytes_per
        print(f"  estimated weights:       {human(weights)}\n")

    print(f"  residual cache (all L):  {human(resid)}")
    print(f"  attention patterns:      {human(patterns)}   <- scales with seq^2")
    print(f"  ~5 hook points/layer:    {human(hooks)}")

    peak = weights + hooks + patterns
    if args.backward:
        peak *= 2.2
        print("  (x2.2 for backward pass)")

    print(f"\n  projected peak:          {human(peak)}  of {human(total)}")

    frac = peak / total
    if frac > 1.0:
        print("\n  VERDICT: will not fit. Options, roughly in order of preference:")
        print("    - shorter seq (patterns fall with the square)")
        print("    - smaller batch")
        print("    - cache fewer hook points, or stream them to disk")
        print("    - bigger card")
        print("    - quantise -- last resort; it adds an alternative explanation")
        print("      for every result you get")
        return 2
    if frac > 0.8:
        print("\n  VERDICT: tight. Fits on paper, likely OOMs once fragmentation")
        print("  and transient buffers land. Halve the batch.")
        return 0
    print("\n  VERDICT: comfortable.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
