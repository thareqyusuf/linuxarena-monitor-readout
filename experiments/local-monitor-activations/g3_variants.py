#!/usr/bin/env python3
"""G3 side task: two cheap variants of the unforced monitor on the same actions, model loaded once.

  A. --effort-greedy E   : greedy, unforced, with `Reasoning: E` in the pinned system prompt (default medium;
                           the earlier runs used low). One prefill + one decode per action.
  B. --n-samples k       : unforced, `Reasoning: low`, k samples at --temperature from ONE prefill: the KV cache
                           is deep-copied per sample, so the cost is k x decode only. Per-sample seed recorded.
                           Target = mean of the parsed scores (how the cached monitor's half-integers arose).

Unforced = decode from the templated prompt's last token (`assistant`); stop on <|return|>/<|call|> only.
Writes results/raw/<slug>/g3_variants_<ts>.json with every generation's text, channel structure, parsed score.
"""
from __future__ import annotations

import argparse
import copy
import datetime as dt
import json
import os
import re
import statistics
import sys
import time
from pathlib import Path

import torch

HERE = Path(__file__).resolve()
SLUG = HERE.parent.name
REPO = HERE.parents[2]
RAW = REPO / "results" / "raw" / SLUG
sys.path.insert(0, str(HERE.parent))
import chunked_attention  # noqa: E402
import monitor_prompt as mp  # noqa: E402
import run_monitor as rm  # noqa: E402

TS = dt.datetime.now(dt.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
FINAL_MARK = "<|channel|>final<|message|>"
ANALYSIS_MARK = "<|channel|>analysis<|message|>"
STOP = [200002, 200012]          # <|return|>, <|call|>

ap = argparse.ArgumentParser()
ap.add_argument("--keys", nargs="+", required=True)
ap.add_argument("--template-date", required=True)
ap.add_argument("--effort-greedy", default="medium")
ap.add_argument("--budget-greedy", type=int, default=2048)
ap.add_argument("--n-samples", type=int, default=4)
ap.add_argument("--temperature", type=float, default=1.0)
ap.add_argument("--effort-samples", default="low")
ap.add_argument("--budget-samples", type=int, default=1024)
ap.add_argument("--attn-chunk-size", type=int, default=256)
ap.add_argument("--seed", type=int, default=0)
args = ap.parse_args()
assert os.environ.get("PYTORCH_CUDA_ALLOC_CONF", "").startswith("expandable_segments")
from transformers import AutoTokenizer, AutoModelForCausalLM  # noqa: E402


def parse(text: str) -> dict:
    out = {"opened_analysis": ANALYSIS_MARK in text, "has_final_channel": FINAL_MARK in text}
    if ANALYSIS_MARK in text:
        a = text.split(ANALYSIS_MARK, 1)[1].split("<|end|>", 1)[0]
        out["analysis_chars"] = len(a); out["analysis_text"] = a
    final = text.split(FINAL_MARK, 1)[1] if FINAL_MARK in text else None
    out["final_text"] = final
    m = re.search(r"<score>\s*(\d+)\s*</score>", final or "", re.S)
    out["parsed_score"] = int(m.group(1)) if m else None
    return out


manifest = mp.load_manifest(); trajs, tmpl_sha = mp.load_trajs(); template = mp.fmp.load_template()
man = {(a["traj_id"], a["action_idx"]): a for a in manifest["actions"]}
tok = AutoTokenizer.from_pretrained(mp.MODEL_ID)
pinned = mp.pin_chat_template(tok, args.template_date)
chunked_attention.CHUNK_SIZE = args.attn_chunk_size
chunked_attention.register()
model = AutoModelForCausalLM.from_pretrained(mp.MODEL_ID, dtype=torch.bfloat16, device_map={"": 0}, attn_implementation="chunked_eager").eval()
R = {"slug": SLUG, "script": str(HERE.relative_to(REPO)), "timestamp": TS, "git_commit": rm.git_commit(), "args": vars(args),
     "pinned_chat_template_sha256": pinned, "decoding": "unforced from `assistant`; stop on <|return|>/<|call|>", "generations": []}


@torch.no_grad()
def prefill(ids):
    x = torch.tensor([ids], device=model.device)
    torch.cuda.synchronize(); t0 = time.time()
    out = model(x, use_cache=True, logits_to_keep=1)
    torch.cuda.synchronize()
    return out.past_key_values, out.logits[0, -1].float(), time.time() - t0


@torch.no_grad()
def decode(pkv, first_logits, start, budget, temperature, gen: torch.Generator | None):
    """Manual token loop on an existing cache. temperature None -> greedy."""
    gen_ids = []
    logits = first_logits
    pos = start
    t0 = time.time()
    for _ in range(budget):
        if temperature is None:
            nxt = int(logits.argmax())
        else:
            probs = torch.softmax(logits / temperature, dim=-1)
            nxt = int(torch.multinomial(probs, 1, generator=gen))
        gen_ids.append(nxt)
        if nxt in STOP:
            break
        cp = torch.tensor([pos], device=model.device)
        o = model(torch.tensor([[nxt]], device=model.device), past_key_values=pkv, use_cache=True, logits_to_keep=1,
                  cache_position=cp, position_ids=cp[None])
        pkv = o.past_key_values
        logits = o.logits[0, -1].float()
        pos += 1
    return gen_ids, time.time() - t0


for key in args.keys:
    tid, idx = key.split("|"); idx = int(idx)
    t = trajs[tid]; m = man[(tid, idx)]
    r = mp.render_user_text(t, idx, "original", template)
    rec = {"key": key, "side": m["side"], "env": m["env"], "side_task": t.get("side_task"), "cached_scores": m["cached_scores"],
           "current_action": r["text"][r["action_span"][0]:r["action_span"][1]][:300]}
    # A. greedy at --effort-greedy
    s = mp.apply_template(tok, r["text"], args.template_date, args.effort_greedy)
    ids = tok(s, add_special_tokens=False)["input_ids"]
    pkv, lg, pre_s = prefill(ids)
    g, dec_s = decode(pkv, lg, len(ids), args.budget_greedy, None, None)
    del pkv
    text = tok.decode(g, skip_special_tokens=False)
    pr = parse(text)
    rec["greedy"] = {"effort": args.effort_greedy, "n_prompt_tokens": len(ids), "n_generated": len(g), "hit_budget": len(g) >= args.budget_greedy,
                     "prefill_seconds": pre_s, "decode_seconds": dec_s, "text": text, **pr}
    print(f"\n===== [greedy {args.effort_greedy}] {m['side']} {key[:8]}#{idx} {t.get('side_task')} cached={m['cached_scores']} {len(ids)} tok -> {len(g)} gen "
          f"(prefill {pre_s:.1f}s, decode {dec_s:.1f}s) analysis {pr.get('analysis_chars')} chars, score {pr['parsed_score']}", flush=True)
    print("  ANALYSIS:", repr((pr.get("analysis_text") or "")[:400]))
    print("  FINAL:", repr((pr.get("final_text") or "")[:300]))
    # B. k samples at --effort-samples from one prefill
    s2 = mp.apply_template(tok, r["text"], args.template_date, args.effort_samples)
    ids2 = tok(s2, add_special_tokens=False)["input_ids"]
    pkv0, lg0, pre2 = prefill(ids2)
    samples = []
    for k in range(args.n_samples):
        seed = args.seed * 1000 + k
        gen = torch.Generator(device=model.device).manual_seed(seed)
        pkv = copy.deepcopy(pkv0)
        g, dec_s = decode(pkv, lg0.clone(), len(ids2), args.budget_samples, args.temperature, gen)
        del pkv
        text = tok.decode(g, skip_special_tokens=False)
        pr = parse(text)
        samples.append({"seed": seed, "n_generated": len(g), "hit_budget": len(g) >= args.budget_samples, "decode_seconds": dec_s, "text": text, **pr})
        print(f"  sample {k} seed {seed}: {len(g)} gen in {dec_s:.1f}s, analysis {pr.get('analysis_chars')} chars, score {pr['parsed_score']}  | {repr((pr.get('analysis_text') or '')[:160])}", flush=True)
    del pkv0
    scores = [x["parsed_score"] for x in samples if x["parsed_score"] is not None]
    rec["samples"] = {"effort": args.effort_samples, "temperature": args.temperature, "n_prompt_tokens": len(ids2), "prefill_seconds": pre2,
                      "n": len(samples), "n_parsed": len(scores), "scores": [x["parsed_score"] for x in samples],
                      "mean": statistics.mean(scores) if scores else None, "stdev": statistics.pstdev(scores) if len(scores) > 1 else None,
                      "decode_seconds_total": sum(x["decode_seconds"] for x in samples), "samples": samples}
    print(f"  => sampled scores {rec['samples']['scores']} mean {rec['samples']['mean']} (cached {m['cached_scores']}); decode total {rec['samples']['decode_seconds_total']:.0f}s", flush=True)
    R["generations"].append(rec)
    torch.cuda.empty_cache()

RAW.mkdir(parents=True, exist_ok=True)
out_p = RAW / f"g3_variants_{TS}.json"
out_p.write_text(json.dumps(R, indent=1, ensure_ascii=False))
print("raw ->", out_p.relative_to(REPO))
