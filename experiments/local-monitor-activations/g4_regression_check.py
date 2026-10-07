#!/usr/bin/env python3
"""G4.2 regression test of the rev-3 runner: out/consistent_high (7-position) vs out/consistent_high_pass1 (4-position).

For every key present in both: the 4 prompt-side activation rows (reordered from the old order) must be bitwise equal,
expected_score / digit_dist must be equal, score and generation text must agree (greedy is deterministic).
Also summarises the new generation-side capture (capture_ok, clean, pre_score digit mass) over the new run. CPU only.
Writes results/raw/<slug>/g4_consistent_high_regression_<ts>.json.
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import statistics
import sys
from pathlib import Path

import torch

HERE = Path(__file__).resolve()
SLUG = HERE.parent.name
REPO = HERE.parents[2]
RAW = REPO / "results" / "raw" / SLUG
OUT = REPO / "checkpoints" / SLUG / "out"
sys.path.insert(0, str(HERE.parent))
import run_monitor as rm  # noqa: E402

OLD_ORDER = ["transcript_end", "action_end", "instruction_end", "final"]


def load_dir(d: Path):
    recs = {rm.key_of_rec(r): r for r in (json.loads(l) for l in (d / "scores.jsonl").open()) if r.get("status") == "ok"}
    idx = json.loads((d / "acts_index.json").read_text())
    acts = {}
    for shard in idx["shards"]:
        sh = torch.load(d / shard)
        acts.update({k: v for k, v in sh.items() if k != "_meta"})
    return recs, acts, idx


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--new", type=Path, default=OUT / "consistent_high")
    ap.add_argument("--old", type=Path, default=OUT / "consistent_high_pass1")
    ap.add_argument("--tag", default="consistent_high")
    args = ap.parse_args()
    ts = dt.datetime.now(dt.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    nrec, nacts, nidx = load_dir(args.new)
    orec, oacts, oidx = load_dir(args.old)
    assert oidx["position_order"] == OLD_ORDER and nidx["position_order"] == list(rm.POSITIONS)
    common = sorted(set(nrec) & set(orec))
    rows = []
    for k in common:
        oa = oacts[k]; na = nacts[k]
        reord = torch.stack([oa[OLD_ORDER.index(rm.PROMPT_POS_SOURCE[n])] for n in rm.PROMPT_POSITIONS])
        d = (na[:4].float() - reord.float()).abs()
        o, n = orec[k], nrec[k]
        rows.append({"key": k, "acts_bitwise_equal": bool(torch.equal(na[:4], reord)), "max_abs_diff": float(d.max()),
                     "expected_score_equal": o["expected_score"] == n["expected_score"], "digit_dist_equal": o["digit_dist"] == n["digit_dist"],
                     "score_old": o["score"], "score_new": n["score"], "score_equal": o["score"] == n["score"],
                     "generation_equal": o["generation"] == n["generation"],
                     "positions_equal": [p["token_index"] for p in o["capture_positions"]] ==
                                        [{p["name"]: p["token_index"] for p in n["capture_positions"]}[nm] for nm in OLD_ORDER[:3]] +
                                        [{p["name"]: p["token_index"] for p in n["capture_positions"]}["prompt_end"]],
                     "capture_ok_new": n.get("capture_ok"), "capture_clean_new": n.get("capture_clean")})
    def frac(f): return sum(bool(f(r)) for r in rows) / len(rows) if rows else None
    newr = list(nrec.values())
    R = {"slug": SLUG, "script": str(HERE.relative_to(REPO)), "timestamp": ts, "git_commit": rm.git_commit(),
         "new_dir": str(args.new.relative_to(REPO)), "old_dir": str(args.old.relative_to(REPO)),
         "n_new": len(nrec), "n_old": len(orec), "n_common": len(common),
         "n_only_new": len(set(nrec) - set(orec)), "n_only_old": len(set(orec) - set(nrec)),
         "n_mismatch_acts": sum(not r["acts_bitwise_equal"] for r in rows),
         "n_mismatch_expected_score": sum(not r["expected_score_equal"] for r in rows),
         "n_mismatch_score": sum(not r["score_equal"] for r in rows),
         "n_mismatch_generation": sum(not r["generation_equal"] for r in rows),
         "n_mismatch_positions": sum(not r["positions_equal"] for r in rows),
         "score_agreement": f"{sum(r['score_equal'] for r in rows)}/{len(rows)}",
         "max_abs_diff_over_all": max((r["max_abs_diff"] for r in rows), default=None),
         "new_run_capture": {"frac_capture_ok": statistics.mean(bool(r.get("capture_ok")) for r in newr) if newr else None,
                             "frac_capture_clean": statistics.mean(bool(r.get("capture_clean")) for r in newr) if newr else None,
                             "n_hit_budget": sum(bool(r.get("hit_budget")) for r in newr),
                             "min_digit_token_mass_post": min((r["digit_token_mass_post"] for r in newr if r.get("digit_token_mass_post") is not None), default=None),
                             "n_argmax_post_ne_score": sum(1 for r in newr if r.get("capture_ok") and not r.get("argmax_digit_post_matches_score")),
                             "pre_score_decoded_counts": dict(__import__("collections").Counter(
                                 next((p["decoded"] for p in r["capture_positions"] if p["name"] == "pre_score"), None) for r in newr)),
                             "seconds_total": sum(r["seconds"] for r in newr)},
         "mismatches": [r for r in rows if not (r["acts_bitwise_equal"] and r["expected_score_equal"] and r["score_equal"] and r["generation_equal"])],
         "rows": rows}
    RAW.mkdir(parents=True, exist_ok=True)
    out = RAW / f"g4_{args.tag}_regression_{ts}.json"
    out.write_text(json.dumps(R, indent=1))
    print(json.dumps({k: v for k, v in R.items() if k not in ("rows", "mismatches")}, indent=1))
    print("n mismatches:", len(R["mismatches"]), "raw ->", out.relative_to(REPO))


if __name__ == "__main__":
    main()
