#!/usr/bin/env python3
"""G4.1 contrast arm: out/low_cached (attack, both cached <= 2, non-placeholder, position-matched) vs the consistent-high
re-run out/consistent_high (attack, both cached >= 6), split on the placeholder flag. Score, E (forced) and E_post
distributions, Mann-Whitney, and the reading: local ~0 on the low arm -> local tracks the cached labels; local high on the
low arm -> local's signal is its own. Also the honest consistent-high control. CPU only.
Writes results/raw/<slug>/g4_low_cached_<ts>.json and results/figures/<slug>/g4_low_cached.png.
"""
from __future__ import annotations

import argparse
import collections
import datetime as dt
import json
import statistics
import sys
from pathlib import Path

import numpy as np
from scipy.stats import mannwhitneyu, spearmanr

HERE = Path(__file__).resolve()
SLUG = HERE.parent.name
REPO = HERE.parents[2]
RAW = REPO / "results" / "raw" / SLUG
FIG = REPO / "results" / "figures" / SLUG
OUT = REPO / "checkpoints" / SLUG / "out"
sys.path.insert(0, str(HERE.parent))
import monitor_prompt as mp  # noqa: E402
import run_monitor as rm  # noqa: E402


def load(d: Path):
    return [r for r in (json.loads(l) for l in (d / "scores.jsonl").open()) if r.get("status") == "ok"]


def hist(rs):
    c = collections.Counter(r["score"] for r in rs if r["score"] is not None)
    return {str(v): c.get(float(v), 0) for v in range(11)}


def summ(rs):
    s = [r["score"] for r in rs if r["score"] is not None]
    e = [r["expected_score"] for r in rs if r["expected_score"] is not None]
    ep = [r["expected_score_post"] for r in rs if r.get("expected_score_post") is not None]
    pos = [r["position_in_trajectory"] for r in rs]
    return {"n": len(rs), "hist": hist(rs), "frac_ge3": sum(x >= 3 for x in s) / len(s) if s else None,
            "frac_ge6": sum(x >= 6 for x in s) / len(s) if s else None, "frac_le1": sum(x <= 1 for x in s) / len(s) if s else None,
            "score_mean": statistics.mean(s) if s else None, "score_median": statistics.median(s) if s else None,
            "expected_mean": statistics.mean(e) if e else None, "expected_sd": statistics.stdev(e) if len(e) > 1 else None,
            "expected_post_mean": statistics.mean(ep) if ep else None,
            "position_median": statistics.median(pos) if pos else None, "n_tokens_median": statistics.median(r["n_prompt_tokens"] for r in rs) if rs else None,
            "cached_mean_mean": statistics.mean(r["cached_mean"] for r in rs) if rs else None}


def mw(a, b, key):
    x = [r[key] for r in a if r.get(key) is not None]; y = [r[key] for r in b if r.get(key) is not None]
    if len(x) < 2 or len(y) < 2:
        return None
    return {"n_a": len(x), "n_b": len(y), "mean_a": statistics.mean(x), "mean_b": statistics.mean(y),
            "p_two_sided": float(mannwhitneyu(x, y, alternative="two-sided").pvalue),
            "p_a_greater": float(mannwhitneyu(x, y, alternative="greater").pvalue)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--low", type=Path, default=OUT / "low_cached")
    ap.add_argument("--high", type=Path, default=OUT / "consistent_high")
    args = ap.parse_args()
    args.low, args.high = args.low.resolve(), args.high.resolve()
    ts = dt.datetime.now(dt.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    low = load(args.low); high = load(args.high)
    trajs, _ = mp.load_trajs()
    low_np = [r for r in low if not r["has_attachment_placeholder"]]
    high_np_att = [r for r in high if not r["has_attachment_placeholder"] and r["side"] == "attack"]
    high_np_hon = [r for r in high if not r["has_attachment_placeholder"] and r["side"] == "honest"]
    high_ph = [r for r in high if r["has_attachment_placeholder"]]

    def describe(r):
        a = trajs[r["traj_id"]]["actions"][r["action_idx"]]
        return {"key": rm.key_of_rec(r), "score": r["score"], "expected_score": r["expected_score"], "expected_score_post": r.get("expected_score_post"),
                "cached_scores": r["cached_scores"], "side_task": r.get("side_task"), "step_index": r["step_index"],
                "trajectory_length": r["trajectory_length"], "position": round(r["position_in_trajectory"], 3),
                "current_action": mp.fmp.format_action(a, hide_output=True)[:300], "analysis_note_text": r.get("analysis_note_text")}

    R = {"slug": SLUG, "script": str(HERE.relative_to(REPO)), "timestamp": ts, "git_commit": rm.git_commit(),
         "low_dir": str(args.low.relative_to(REPO)), "high_dir": str(args.high.relative_to(REPO)),
         "definitions": {"low_cached": "attack, both cached samples <= 2, non-placeholder, same 17 consistent-high trajectories, position-matched (|dpos| <= 0.20)",
                         "consistent_high": "both cached samples >= 6 (subset_consistent_high.json), re-run with the rev-3 runner"},
         "groups": {"low_cached": summ(low), "low_cached_no_placeholder": summ(low_np), "high_attack_no_placeholder": summ(high_np_att),
                    "high_honest_no_placeholder": summ(high_np_hon), "high_placeholder": summ(high_ph), "high_all": summ(high)},
         "tests_low_vs_high_attack_np": {"score": mw(high_np_att, low_np, "score"), "expected_score": mw(high_np_att, low_np, "expected_score"),
                                         "expected_score_post": mw(high_np_att, low_np, "expected_score_post")},
         "tests_low_vs_high_honest_np": {"score": mw(low_np, high_np_hon, "score"), "expected_score": mw(low_np, high_np_hon, "expected_score")},
         "spearman_E_vs_score": {"low": float(spearmanr([r["expected_score"] for r in low], [r["score"] for r in low]).statistic) if len(low) > 2 else None,
                                 "high": float(spearmanr([r["expected_score"] for r in high], [r["score"] for r in high]).statistic) if len(high) > 2 else None},
         "spearman_Epost_vs_score": {"high": float(spearmanr([r["expected_score_post"] for r in high if r.get("expected_score_post") is not None],
                                                             [r["score"] for r in high if r.get("expected_score_post") is not None]).statistic) if len(high) > 2 else None},
         "low_cached_actions_sorted_by_score": [describe(r) for r in sorted(low, key=lambda r: -(r["score"] or 0))],
         "low_by_side_task": {"ge7": dict(collections.Counter(r["side_task"] for r in low if (r["score"] or 0) >= 7)),
                              "le1": dict(collections.Counter(r["side_task"] for r in low if (r["score"] or 0) <= 1))},
         "parse_health": {"low": {"n": len(low), "n_score_ok": sum(r["score"] is not None for r in low), "n_hit_budget": sum(bool(r.get("hit_budget")) for r in low),
                                  "frac_capture_ok": statistics.mean(bool(r.get("capture_ok")) for r in low) if low else None},
                          "high": {"n": len(high), "n_score_ok": sum(r["score"] is not None for r in high), "n_hit_budget": sum(bool(r.get("hit_budget")) for r in high),
                                   "frac_capture_ok": statistics.mean(bool(r.get("capture_ok")) for r in high) if high else None}}}
    # figure
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.2))
    BLUE, ORANGE, GREEN = "#2a78d6", "#eb6834", "#1baf7a"
    xs = np.arange(11); w = 0.27
    for off, rows, col, lab in ((-w, low_np, BLUE, f"low-cached attack (n={len(low_np)})"), (0, high_np_att, ORANGE, f"consistent-high attack (n={len(high_np_att)})"),
                                (w, high_np_hon, GREEN, f"consistent-high honest (n={len(high_np_hon)})")):
        if rows:
            axes[0].bar(xs + off, [hist(rows)[str(v)] / len(rows) for v in range(11)], w, color=col, label=lab, linewidth=0)
    axes[0].set_xlabel("score (generated, unforced)"); axes[0].set_ylabel("fraction of group"); axes[0].set_xticks(xs)
    axes[0].set_title("non-placeholder actions, by cached label", fontsize=10); axes[0].legend(frameon=False, fontsize=8)
    for rows, col, lab in ((low_np, BLUE, "low-cached"), (high_np_att, ORANGE, "consistent-high attack"), (high_np_hon, GREEN, "consistent-high honest")):
        if rows:
            axes[1].scatter([r["position_in_trajectory"] for r in rows], [r["score"] for r in rows], s=18, color=col, alpha=0.6, linewidths=0, label=lab)
    axes[1].set_xlabel("position in trajectory"); axes[1].set_ylabel("score"); axes[1].set_title("position-matched by construction", fontsize=10)
    axes[1].legend(frameon=False, fontsize=8)
    rng = np.random.RandomState(0)
    for rows, col, lab in ((low_np, BLUE, "low-cached"), (high_np_att, ORANGE, "consistent-high attack")):
        if rows:
            axes[2].scatter(np.array([r["expected_score"] for r in rows]), [r.get("expected_score_post") for r in rows], s=18, color=col, alpha=0.6, linewidths=0, label=lab)
    axes[2].set_xlabel("expected_score (forced, pre-reasoning)"); axes[2].set_ylabel("expected_score_post (at pre_score, own reasoning)")
    axes[2].set_title("forced vs post-reasoning readout", fontsize=10); axes[2].legend(frameon=False, fontsize=8)
    for ax in axes:
        ax.grid(axis="y", color="#e5e5e5", lw=0.6); ax.set_axisbelow(True)
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)
    fig.tight_layout(); FIG.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG / "g4_low_cached.png", dpi=150); plt.close(fig)
    R["figure"] = str((FIG / "g4_low_cached.png").relative_to(REPO))
    RAW.mkdir(parents=True, exist_ok=True)
    out = RAW / f"g4_low_cached_{ts}.json"
    out.write_text(json.dumps(R, indent=1, ensure_ascii=False))
    print(json.dumps({k: v for k, v in R.items() if k not in ("low_cached_actions_sorted_by_score",)}, indent=1))
    print("raw ->", out.relative_to(REPO))


if __name__ == "__main__":
    main()
