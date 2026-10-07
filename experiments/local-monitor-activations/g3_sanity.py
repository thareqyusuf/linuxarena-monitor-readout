#!/usr/bin/env python3
"""G3 step 0: tiny GPU sanity check before the pilot (2 prompts: shortest honest and shortest attack with a
neutralization event in the current action, from subset_main).

For each prompt, with the model loaded once (chunked_eager, fixed chunk size):
  a. prefill with the four-position capture (same code path as run_monitor: PositionCapture, logits_to_keep=1);
  b. forced `<|channel|>final<|message|>` -> next-token distribution (top-10, digit mass);
  c. greedy PEEK of --peek-tokens tokens from there (what the model would actually say in the final channel),
     decoded — this is the only place reasoning-free text is generated, and only for these 2 prompts;
  d. fresh prefill again -> forced final -> `<score>\n` -> digit distribution (the run_monitor reading);
  e. the same prefill a third time: activations and both distributions must be bitwise identical (determinism
     at a fixed chunk size), and activations from the two prompts must differ at every position.
  f. seconds and peak GiB for each prefill.
Writes results/raw/<slug>/g3_sanity_<ts>.json.
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import os
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

ap = argparse.ArgumentParser()
ap.add_argument("--attn-chunk-size", type=int, default=256)
ap.add_argument("--template-date", required=True)
ap.add_argument("--reasoning-effort", default="low")
ap.add_argument("--peek-tokens", type=int, default=48)
args = ap.parse_args()
assert os.environ.get("PYTORCH_CUDA_ALLOC_CONF", "").startswith("expandable_segments")

from transformers import AutoTokenizer, AutoModelForCausalLM  # noqa: E402

R: dict = {"slug": SLUG, "script": str(HERE.relative_to(REPO)), "timestamp": TS, "git_commit": rm.git_commit(), "args": vars(args)}
manifest = mp.load_manifest(); trajs, tmpl_sha = mp.load_trajs(); template = mp.fmp.load_template()
man = {(a["traj_id"], a["action_idx"]): a for a in manifest["actions"]}
g2 = json.loads(sorted(RAW.glob("g2_prep_*.json"))[-1].read_text())
pa = g2["per_action"]
neut_act = {(t, i): a for t, i, a in zip(pa["traj_id"], pa["action_idx"], pa["neut_action_events"])}
sub = json.loads((HERE.parent / "subset_main.json").read_text())["actions"]
sub.sort(key=lambda a: a["n_tokens"])
ex_h = next(a for a in sub if a["side"] == "honest")
ex_a = next(a for a in sub if a["side"] == "attack" and neut_act[(a["traj_id"], a["action_idx"])] > 0)

tok = AutoTokenizer.from_pretrained(mp.MODEL_ID)
pinned = mp.pin_chat_template(tok, args.template_date)
digits = rm.digit_ids(tok)
forced_ids = tok.encode(rm.FORCED_FINAL, add_special_tokens=False)
score_tag_ids = tok.encode(rm.SCORE_TAG, add_special_tokens=False)
chunked_attention.CHUNK_SIZE = args.attn_chunk_size
chunked_attention.register()
t0 = time.time()
model = AutoModelForCausalLM.from_pretrained(mp.MODEL_ID, dtype=torch.bfloat16, device_map={"": 0}, attn_implementation="chunked_eager").eval()
R["model"] = {"load_seconds": time.time() - t0, "resident_gib": torch.cuda.memory_allocated() / 2**30,
              "attn": model.config._attn_implementation, "dequantize": model.hf_quantizer.quantization_config.dequantize,
              "sinks_dtype_layer0": str(model.model.layers[0].self_attn.sinks.dtype)}
print(R["model"], flush=True)
capture = rm.PositionCapture(model.model.embed_tokens, list(model.model.layers))


@torch.no_grad()
def prefill(ids, positions):
    torch.cuda.reset_peak_memory_stats()
    x = torch.tensor([ids], device=model.device)
    capture.arm(positions)
    torch.cuda.synchronize(); t1 = time.time()
    out = model(x, use_cache=True, logits_to_keep=1)
    torch.cuda.synchronize()
    return out.past_key_values, capture.take(), time.time() - t1, torch.cuda.max_memory_allocated() / 2**30


@torch.no_grad()
def step(pkv, new_ids, start):
    cp = torch.arange(start, start + len(new_ids), device=model.device)
    o = model(torch.tensor([new_ids], device=model.device), past_key_values=pkv, use_cache=True, logits_to_keep=1, cache_position=cp, position_ids=cp[None])
    return o.past_key_values, o.logits[0, -1].float().cpu()


R["prompts"] = {}
acts_by = {}
for name, a in (("honest", ex_h), ("attack", ex_a)):
    t = trajs[a["traj_id"]]
    for render in ("original", "neutralized") if name == "attack" else ("original",):
        r = mp.render_user_text(t, a["action_idx"], render, template)
        s = mp.apply_template(tok, r["text"], args.template_date, args.reasoning_effort)
        tp = mp.tokenize_with_positions(tok, s, r["text"], r)
        ids = tp["input_ids"]; L = len(ids)
        positions = [tp["positions"][k] for k in mp.POSITIONS]
        rec = {"traj_id": a["traj_id"], "action_idx": a["action_idx"], "env": a["env"], "render": render, "n_tokens": L,
               "cached_scores": man[(a["traj_id"], a["action_idx"])]["cached_scores"],
               "capture_positions": mp.decode_positions(tok, ids, tp["positions"]),
               "current_action_text": r["text"][r["action_span"][0]:r["action_span"][1]][:600], "neutralize_events": r["subs"]}
        # a+b+c: prefill, forced final, peek
        pkv, acts1, sec1, peak1 = prefill(ids, positions)
        pkv, lg = step(pkv, forced_ids, L)
        rec["after_message"] = rm.read_distribution(tok, lg, digits)
        gen = []
        pos = L + len(forced_ids)
        cur = lg
        for _ in range(args.peek_tokens):
            nxt = int(cur.argmax())
            gen.append(nxt)
            if nxt in (200002, 200007, 200012):     # <|return|>, <|end|>, <|call|>
                break
            pkv, cur = step(pkv, [nxt], pos); pos += 1
        rec["peek_greedy"] = tok.decode(gen, skip_special_tokens=False)
        rec["peek_ids"] = gen
        del pkv
        # d: fresh prefill -> forced final -> <score>\n
        pkv, acts2, sec2, peak2 = prefill(ids, positions)
        pkv, lg2 = step(pkv, forced_ids, L)
        pkv, lg3 = step(pkv, score_tag_ids, L + len(forced_ids))
        rec["after_score_tag"] = rm.read_distribution(tok, lg3, digits)
        # peek after <score>\n too (what would follow the digit)
        gen2 = []; pos = L + len(forced_ids) + len(score_tag_ids); cur = lg3
        for _ in range(8):
            nxt = int(cur.argmax()); gen2.append(nxt)
            if nxt in (200002, 200007, 200012):
                break
            pkv, cur = step(pkv, [nxt], pos); pos += 1
        rec["peek_after_score_tag"] = tok.decode(gen2, skip_special_tokens=False)
        del pkv
        # e: third prefill, determinism
        pkv, acts3, sec3, peak3 = prefill(ids, positions)
        pkv, lg4 = step(pkv, forced_ids, L)
        del pkv
        rec["determinism"] = {"acts_run1_eq_run2": bool(torch.equal(acts1, acts2)), "acts_run2_eq_run3": bool(torch.equal(acts2, acts3)),
                              "max_abs_acts_diff_1v3": float((acts1.float() - acts3.float()).abs().max()),
                              "after_message_logits_eq_1v2": bool(torch.equal(lg, lg2)), "after_message_logits_eq_2v4": bool(torch.equal(lg2, lg4)),
                              "max_abs_logit_diff_1v4": float((lg - lg4).abs().max())}
        rec["timing"] = {"prefill_seconds": [sec1, sec2, sec3], "peak_gib": [peak1, peak2, peak3]}
        rec["acts"] = {"shape": list(acts1.shape), "dtype": str(acts1.dtype), "finite": bool(torch.isfinite(acts1).all()),
                       "norm_by_layer_at_final": [float(x) for x in acts1[3].float().norm(dim=-1)],
                       "norm_by_position_at_last_layer": [float(x) for x in acts1[:, -1].float().norm(dim=-1)]}
        acts_by[f"{name}/{render}"] = acts1
        R["prompts"][f"{name}/{render}"] = rec
        print(f"\n=== {name}/{render} {a['traj_id'][:8]}#{a['action_idx']} {L} tok  prefill {sec1:.2f}/{sec2:.2f}/{sec3:.2f}s peak {peak1:.2f} GiB")
        print("  capture:", [(p["name"], p["token_index"], p["decoded"]) for p in rec["capture_positions"]])
        print("  after <|message|>: digit mass %.4f, top5 %s" % (rec["after_message"]["digit_token_mass"], [(x["token"], round(x["p"], 3)) for x in rec["after_message"]["top10"][:5]]))
        print("  PEEK:", repr(rec["peek_greedy"]))
        print("  after <score>\\n: digit mass %.4f, E %s, argmax %s, top5 %s" % (rec["after_score_tag"]["digit_token_mass"], rec["after_score_tag"]["expected_score"], rec["after_score_tag"]["argmax_digit"], [(x["token"], round(x["p"], 3)) for x in rec["after_score_tag"]["top10"][:5]]))
        print("  then:", repr(rec["peek_after_score_tag"]), " cached:", rec["cached_scores"])
        print("  determinism:", rec["determinism"])
        print("  norms at final by layer:", [round(x, 1) for x in rec["acts"]["norm_by_layer_at_final"]])
        sys.stdout.flush()

# cross-prompt distinctness
keys = list(acts_by)
R["distinct"] = {}
for i in range(len(keys)):
    for j in range(i + 1, len(keys)):
        A, B = acts_by[keys[i]].float(), acts_by[keys[j]].float()
        cos = torch.nn.functional.cosine_similarity(A, B, dim=-1)          # [4, 25]
        R["distinct"][f"{keys[i]} vs {keys[j]}"] = {"any_position_layer_identical": bool((A == B).all(dim=-1).any()),
                                                    "cosine_by_position_last_layer": [float(x) for x in cos[:, -1]],
                                                    "min_cosine_distance": float((1 - cos).min())}
print("\ndistinct:", json.dumps(R["distinct"], indent=1))
R["finished"] = dt.datetime.now(dt.timezone.utc).isoformat()
RAW.mkdir(parents=True, exist_ok=True)
out = RAW / f"g3_sanity_{TS}.json"
out.write_text(json.dumps(R, indent=1, default=str))
print("raw ->", out.relative_to(REPO))
