#!/usr/bin/env python3
"""G3 addendum: free generation. For each chosen pilot action and each --mode:
  forced_final : prefill the pinned prompt + `<|channel|>final<|message|>`, then greedy-decode with nothing else forced
  free         : prefill the pinned prompt only (last token `assistant`); the model opens its own channel(s);
                 stop only on <|return|> / <|call|> (NOT <|end|>, which closes the analysis message)
Records the raw text, which channels were opened, the parsed <score> from the final channel, and the digit
distribution at the digit the model itself wrote. Writes results/raw/<slug>/g3_peek_<ts>.json.
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import re
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
ap.add_argument("--keys", nargs="+", required=True, help="traj_id|action_idx")
ap.add_argument("--max-new-tokens", type=int, default=200, help="budget for forced_final")
ap.add_argument("--max-new-tokens-free", type=int, default=1024)
ap.add_argument("--modes", nargs="+", default=["forced_final"], choices=["forced_final", "free"])
ap.add_argument("--attn-chunk-size", type=int, default=256)
ap.add_argument("--template-date", required=True)
ap.add_argument("--reasoning-effort", default="low")
args = ap.parse_args()
assert os.environ.get("PYTORCH_CUDA_ALLOC_CONF", "").startswith("expandable_segments")
from transformers import AutoTokenizer, AutoModelForCausalLM  # noqa: E402

manifest = mp.load_manifest(); trajs, tmpl_sha = mp.load_trajs(); template = mp.fmp.load_template()
man = {(a["traj_id"], a["action_idx"]): a for a in manifest["actions"]}
tok = AutoTokenizer.from_pretrained(mp.MODEL_ID)
pinned = mp.pin_chat_template(tok, args.template_date)
digits = rm.digit_ids(tok)
forced_ids = tok.encode(rm.FORCED_FINAL, add_special_tokens=False)
chunked_attention.CHUNK_SIZE = args.attn_chunk_size
chunked_attention.register()
model = AutoModelForCausalLM.from_pretrained(mp.MODEL_ID, dtype=torch.bfloat16, device_map={"": 0}, attn_implementation="chunked_eager").eval()
STOP = {"forced_final": [200002, 200007, 200012], "free": [200002, 200012]}     # <|return|>, (<|end|>), <|call|>
R = {"slug": SLUG, "script": str(HERE.relative_to(REPO)), "timestamp": TS, "git_commit": rm.git_commit(), "args": vars(args),
     "forced_prefix": rm.FORCED_FINAL, "decoding": "greedy; forced_final stops on <|return|>/<|end|>/<|call|>, free stops on <|return|>/<|call|>", "generations": []}
FINAL_MARK = "<|channel|>final<|message|>"
ANALYSIS_MARK = "<|channel|>analysis<|message|>"


def parse(text):
    out = {"opened_analysis": text.startswith(ANALYSIS_MARK) or ANALYSIS_MARK in text, "has_final_channel": FINAL_MARK in text}
    if ANALYSIS_MARK in text:
        a = text.split(ANALYSIS_MARK, 1)[1].split("<|end|>", 1)[0]
        out["analysis_text"] = a; out["analysis_chars"] = len(a)
    final = text.split(FINAL_MARK, 1)[1] if FINAL_MARK in text else None
    out["final_text"] = final
    mtch = re.search(r"<score>\s*(\d+)\s*</score>", final or "", re.S)
    out["parsed_score"] = int(mtch.group(1)) if mtch else None
    out["score_char_start_in_text"] = (text.index(FINAL_MARK) + len(FINAL_MARK) + mtch.start(1)) if mtch else None
    return out


for key in args.keys:
    tid, idx = key.split("|"); idx = int(idx)
    t = trajs[tid]; m = man[(tid, idx)]
    r = mp.render_user_text(t, idx, "original", template)
    s = mp.apply_template(tok, r["text"], args.template_date, args.reasoning_effort)
    ids = tok(s, add_special_tokens=False)["input_ids"]
    rec = {"key": key, "side": m["side"], "env": m["env"], "side_task": t.get("side_task"), "cached_scores": m["cached_scores"], "n_prompt_tokens": len(ids),
           "current_action": r["text"][r["action_span"][0]:r["action_span"][1]][:500], "modes": {}}
    for mode in args.modes:
        prefix = forced_ids if mode == "forced_final" else []
        budget = args.max_new_tokens if mode == "forced_final" else args.max_new_tokens_free
        x = torch.tensor([ids + prefix], device=model.device)
        t0 = time.time()
        with torch.no_grad():
            out = model.generate(x, max_new_tokens=budget, do_sample=False, eos_token_id=STOP[mode], pad_token_id=tok.pad_token_id or 200002,
                                 output_scores=True, return_dict_in_generate=True)
        gen = out.sequences[0, x.shape[1]:].tolist()
        text = tok.decode(gen, skip_special_tokens=False)
        full = (rm.FORCED_FINAL if mode == "forced_final" else "") + text
        sec = time.time() - t0
        pr = parse(full)
        mr = {"n_generated": len(gen), "hit_budget": len(gen) >= budget, "budget": budget, "seconds": sec,
              "stopped_on": tok.decode([gen[-1]]) if gen and gen[-1] in STOP[mode] else None, "text": text, **pr}
        if pr["parsed_score"] is not None:
            # generated-token index of the digit: char offset within `text` (strip the forced prefix chars)
            c = pr["score_char_start_in_text"] - (len(rm.FORCED_FINAL) if mode == "forced_final" else 0)
            pos = None
            for i in range(len(gen)):
                if len(tok.decode(gen[: i + 1], skip_special_tokens=False)) > c:
                    pos = i; break
            if pos is not None:
                mr["own_digit_token"] = tok.decode([gen[pos]])
                d = rm.read_distribution(tok, out.scores[pos][0], digits); d.pop("top10", None); mr["own_digit_dist"] = d
        rec["modes"][mode] = mr
        print(f"\n===== [{mode}] {m['side']} {key[:8]}#{idx} env={m['env']} side_task={t.get('side_task')} cached={m['cached_scores']} {len(ids)} tok -> {len(gen)} gen in {sec:.1f}s; "
              f"analysis={pr['opened_analysis']} ({pr.get('analysis_chars')} chars) final={pr['has_final_channel']} score={pr['parsed_score']} p(own digit)={mr.get('own_digit_dist', {}).get('argmax_digit_p')}")
        print("  CURRENT ACTION:", rec["current_action"][:200].replace("\n", "\n    "))
        print("  GENERATION:\n" + "\n".join("    | " + l for l in text.splitlines()))
        sys.stdout.flush()
    R["generations"].append(rec)
RAW.mkdir(parents=True, exist_ok=True)
out_p = RAW / f"g3_peek_{TS}.json"
out_p.write_text(json.dumps(R, indent=1, ensure_ascii=False))
print("raw ->", out_p.relative_to(REPO))
