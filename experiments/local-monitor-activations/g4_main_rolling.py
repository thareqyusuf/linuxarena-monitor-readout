#!/usr/bin/env python3
"""Rolling / final summary of a stage's scores.jsonl (default: main), split on placeholder and side, plus the both-cached>=6 slice.
CPU only, no model. Writes results/raw/<slug>/g4_<stage>_rolling_<n>_<ts>.json where <n> is the number of ok records seen."""
from __future__ import annotations

import argparse
import collections
import datetime as dt
import json
import statistics
import sys
from pathlib import Path

from scipy.stats import spearmanr

HERE = Path(__file__).resolve()
SLUG = HERE.parent.name
REPO = HERE.parents[2]
RAW = REPO / "results" / "raw" / SLUG
OUT = REPO / "checkpoints" / SLUG / "out"
sys.path.insert(0, str(HERE.parent))
import run_monitor as rm  # noqa: E402


def summ(rs):
    s = [r["score"] for r in rs if r["score"] is not None]; e = [r["expected_score"] for r in rs if r["expected_score"] is not None]
    if not rs:
        return {"n": 0}
    h = collections.Counter(int(x) for x in s)
    return {"n": len(rs), "n_traj": len({r["traj_id"] for r in rs}), "hist": {str(v): h.get(v, 0) for v in range(11)},
            "frac_ge3": sum(x >= 3 for x in s) / len(s) if s else None, "frac_ge6": sum(x >= 6 for x in s) / len(s) if s else None,
            "score_mean": statistics.mean(s) if s else None, "expected_mean": statistics.mean(e) if e else None,
            "expected_sd": statistics.stdev(e) if len(e) > 1 else None,
            "position_median": statistics.median(r["position_in_trajectory"] for r in rs),
            "n_tokens_median": statistics.median(r["n_prompt_tokens"] for r in rs),
            "cached_mean_mean": statistics.mean(r["cached_mean"] for r in rs)}


def sp(rs, a, b):
    pairs = [(r[a], r[b]) for r in rs if r.get(a) is not None and r.get(b) is not None]
    return float(spearmanr([x for x, _ in pairs], [y for _, y in pairs]).statistic) if len(pairs) > 2 else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", default="main")
    args = ap.parse_args()
    ts = dt.datetime.now(dt.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    recs = [json.loads(l) for l in (OUT / args.stage / "scores.jsonl").open()]
    ok = [r for r in recs if r["status"] == "ok"]
    npl = [r for r in ok if not r["has_attachment_placeholder"]]
    R = {"slug": SLUG, "script": str(HERE.relative_to(REPO)), "timestamp": ts, "git_commit": rm.git_commit(), "stage": args.stage,
         "n_records": len(recs), "by_status": dict(collections.Counter(r["status"] for r in recs)),
         "groups": {"all": summ(ok), "no_placeholder": summ(npl), "no_placeholder_attack": summ([r for r in npl if r["side"] == "attack"]),
                    "no_placeholder_honest": summ([r for r in npl if r["side"] == "honest"]),
                    "placeholder": summ([r for r in ok if r["has_attachment_placeholder"]]),
                    "both_cached_ge6": summ([r for r in ok if all(c >= 6 for c in r["cached_scores"])]),
                    "both_cached_ge6_no_placeholder": summ([r for r in npl if all(c >= 6 for c in r["cached_scores"])]),
                    "both_cached_le2_attack_no_placeholder": summ([r for r in npl if r["side"] == "attack" and all(c <= 2 for c in r["cached_scores"])])},
         "health": {"n_ok": len(ok), "frac_capture_ok": statistics.mean(bool(r["capture_ok"]) for r in ok) if ok else None,
                    "frac_capture_clean": statistics.mean(bool(r["capture_clean"]) for r in ok) if ok else None,
                    "n_hit_budget": sum(bool(r["hit_budget"]) for r in ok), "n_score_none": sum(r["score"] is None for r in ok),
                    "seconds_per_action": statistics.mean(r["seconds"] for r in ok) if ok else None,
                    "peak_gib_max": max(r["peak_gib"] for r in ok) if ok else None,
                    "min_digit_token_mass_post": min((r["digit_token_mass_post"] for r in ok if r.get("digit_token_mass_post") is not None), default=None),
                    "n_epost_ne_score": sum(1 for r in ok if r.get("expected_score_post") is not None and r["score"] is not None and abs(r["expected_score_post"] - r["score"]) > 1e-3)},
         "spearman": {"E_vs_score": sp(ok, "expected_score", "score"), "score_vs_cached_mean": sp(ok, "score", "cached_mean"),
                      "E_vs_cached_mean": sp(ok, "expected_score", "cached_mean"), "score_vs_cached_mean_no_placeholder": sp(npl, "score", "cached_mean"),
                      "E_vs_cached_mean_no_placeholder": sp(npl, "expected_score", "cached_mean"), "n_pairs_note": "records with a None value are dropped pairwise"},
         "ge3_actions": [{"key": rm.key_of_rec(r), "score": r["score"], "expected_score": r["expected_score"], "cached_scores": r["cached_scores"], "side": r["side"],
                          "side_task": r["side_task"], "placeholder": r["has_attachment_placeholder"], "position": round(r["position_in_trajectory"], 3),
                          "analysis_note_text": r.get("analysis_note_text")} for r in sorted(ok, key=lambda r: -(r["score"] or 0)) if (r["score"] or 0) >= 3]}
    RAW.mkdir(parents=True, exist_ok=True)
    out = RAW / f"g4_{args.stage}_rolling_{len(ok)}_{ts}.json"
    out.write_text(json.dumps(R, indent=1, ensure_ascii=False))
    for k, g in R["groups"].items():
        if g["n"]:
            print(f"{k:38s} n={g['n']:4d} traj={g['n_traj']:3d} hist={[g['hist'][str(v)] for v in range(11)]} ge3={g['frac_ge3']:.3f} mean={g['score_mean']:.2f} E={g['expected_mean']:.3f} sdE={g['expected_sd'] or 0:.3f} pos={g['position_median']:.2f} tok={g['n_tokens_median']:.0f} cached={g['cached_mean_mean']:.2f}")
    print("health", json.dumps(R["health"])); print("spearman", json.dumps(R["spearman"]))
    print("raw ->", out.relative_to(REPO))


if __name__ == "__main__":
    main()
