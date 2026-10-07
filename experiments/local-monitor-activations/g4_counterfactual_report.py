#!/usr/bin/env python3
"""G4.5 counterfactual renders: original vs neutralized vs random_scrub on subset_counterfactual.json (250 actions).

Per action: score / E (forced) / E_post under each render, the number of substitution events (history / current action), the render
sha256, stratum (action vs history_only), placeholder, side, cached. Aggregates: paired deltas (neutralized - original, random_scrub -
original) by stratum x placeholder, Wilcoxon signed-rank on E, fraction of actions whose integer score changed, and the chunk-size noise
floor (|dE| ~ 0.05) as the reference. The original-render records come from out/main (or out/main_pass1 for the 200 pass-1 keys).
CPU only. Writes results/raw/<slug>/g4_counterfactual_<ts>.json and results/figures/<slug>/g4_counterfactual.png.
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
from scipy.stats import wilcoxon

HERE = Path(__file__).resolve()
SLUG = HERE.parent.name
REPO = HERE.parents[2]
RAW = REPO / "results" / "raw" / SLUG
FIG = REPO / "results" / "figures" / SLUG
OUT = REPO / "checkpoints" / SLUG / "out"
sys.path.insert(0, str(HERE.parent))
import run_monitor as rm  # noqa: E402


def load(d: Path) -> dict:
    if not (d / "scores.jsonl").exists():
        return {}
    return {f"{r['traj_id']}|{r['action_idx']}": r for r in (json.loads(l) for l in (d / "scores.jsonl").open()) if r.get("status") == "ok"}


def agg(rows, key_orig, key_alt):
    pairs = [(r[key_orig], r[key_alt]) for r in rows if r.get(key_orig) is not None and r.get(key_alt) is not None]
    if not pairs:
        return {"n": 0}
    d = [b - a for a, b in pairs]
    out = {"n": len(pairs), "mean_orig": statistics.mean(a for a, _ in pairs), "mean_alt": statistics.mean(b for _, b in pairs),
           "mean_delta": statistics.mean(d), "median_delta": statistics.median(d), "mean_abs_delta": statistics.mean(abs(x) for x in d),
           "frac_changed": sum(x != 0 for x in d) / len(d), "frac_down": sum(x < 0 for x in d) / len(d), "frac_up": sum(x > 0 for x in d) / len(d)}
    if len(d) >= 10 and any(x != 0 for x in d):
        out["wilcoxon_p"] = float(wilcoxon([a for a, _ in pairs], [b for _, b in pairs]).pvalue)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--subset", type=Path, default=HERE.parent / "subset_counterfactual.json")
    args = ap.parse_args()
    ts = dt.datetime.now(dt.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    sub = {f"{a['traj_id']}|{a['action_idx']}": a for a in json.loads(args.subset.read_text())["actions"]}
    orig = load(OUT / "main"); orig_p1 = load(OUT / "main_pass1")
    neut = load(OUT / "cf_neutralized"); scrub = load(OUT / "cf_random_scrub")
    rows = []
    for k, a in sub.items():
        o = orig.get(k) or orig_p1.get(k)
        n, s = neut.get(k), scrub.get(k)
        if o is None:
            continue
        row = {"key": k, "stratum": a["counterfactual_stratum"], "placeholder": o["has_attachment_placeholder"], "side": o["side"],
               "side_task": o["side_task"], "cached_scores": o["cached_scores"], "orig_source": "main" if k in orig else "main_pass1",
               "score_orig": o["score"], "E_orig": o["expected_score"], "Epost_orig": o.get("expected_score_post"),
               "sha_orig": o.get("render_sha256") or o["prompt_sha256"]}
        for tag, r in (("neut", n), ("scrub", s)):
            if r is not None:
                row.update({f"score_{tag}": r["score"], f"E_{tag}": r["expected_score"], f"Epost_{tag}": r.get("expected_score_post"),
                            f"events_{tag}": r["neutralize_events"], f"sha_{tag}": r["render_sha256"], f"capture_ok_{tag}": r["capture_ok"],
                            f"n_tokens_{tag}": r["n_prompt_tokens"]})
            else:
                row.update({f"score_{tag}": None, f"E_{tag}": None, f"Epost_{tag}": None, f"events_{tag}": None, f"sha_{tag}": None})
        rows.append(row)
    groups = {"all": rows, "action_stratum": [r for r in rows if r["stratum"] == "action"], "history_only": [r for r in rows if r["stratum"] == "history_only"],
              "action_no_placeholder": [r for r in rows if r["stratum"] == "action" and not r["placeholder"]],
              "action_placeholder": [r for r in rows if r["stratum"] == "action" and r["placeholder"]],
              "history_only_no_placeholder": [r for r in rows if r["stratum"] == "history_only" and not r["placeholder"]],
              "attack_no_placeholder": [r for r in rows if r["side"] == "attack" and not r["placeholder"]],
              "honest_no_placeholder": [r for r in rows if r["side"] == "honest" and not r["placeholder"]],
              "orig_score_ge3": [r for r in rows if (r["score_orig"] or 0) >= 3]}
    R = {"slug": SLUG, "script": str(HERE.relative_to(REPO)), "timestamp": ts, "git_commit": rm.git_commit(),
         "n_subset": len(sub), "n_with_original": len(rows), "n_neutralized": sum(r["score_neut"] is not None or r["E_neut"] is not None for r in rows),
         "n_random_scrub": sum(r["E_scrub"] is not None for r in rows),
         "noise_floor_reference": "chunk 256 vs 128 on the same prompts (G3.2): |dE| median 0.05, cosine ~1e-3 at prompt_end",
         "caveat": "neutralization is meaning-changing (curl->cat etc.) and rewrites benign text too; a drop is partly legitimate. This arm bounds surface-form leakage, it does not isolate it.",
         "events": {"neut_action_events_hist": dict(collections.Counter(r["events_neut"]["action_events"] for r in rows if r["events_neut"])),
                    "neut_history_events_median": statistics.median(r["events_neut"]["history_events"] for r in rows if r["events_neut"]) if neut else None,
                    "scrub_matched_target": (sum(r["events_scrub"]["action_events"] == r["events_scrub"].get("target_action_events") and
                                                 r["events_scrub"]["history_events"] == r["events_scrub"].get("target_history_events") for r in rows if r["events_scrub"]) if scrub else None),
                    "sha_all_distinct_from_original": {"neut": all(r["sha_neut"] != r["sha_orig"] for r in rows if r["sha_neut"]),
                                                       "scrub": all(r["sha_scrub"] != r["sha_orig"] for r in rows if r["sha_scrub"])}},
         "aggregates": {g: {"score_neut": agg(rs, "score_orig", "score_neut"), "E_neut": agg(rs, "E_orig", "E_neut"),
                            "score_scrub": agg(rs, "score_orig", "score_scrub"), "E_scrub": agg(rs, "E_orig", "E_scrub")} for g, rs in groups.items()},
         "rows": rows}
    # figure
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.2))
    BLUE, ORANGE = "#2a78d6", "#eb6834"
    for ax, tag, lab in ((axes[0], "neut", "neutralized"), (axes[1], "scrub", "random_scrub")):
        for rs, col, name in ((groups["action_stratum"], ORANGE, "action stratum"), (groups["history_only"], BLUE, "history-only")):
            pr = [(r["E_orig"], r[f"E_{tag}"]) for r in rs if r[f"E_{tag}"] is not None]
            if pr:
                ax.scatter([a for a, _ in pr], [b for _, b in pr], s=16, color=col, alpha=0.6, linewidths=0, label=f"{name} (n={len(pr)})")
        lim = max([r["E_orig"] for r in rows] + [r[f"E_{tag}"] or 0 for r in rows] + [1]) * 1.05
        ax.plot([0, lim], [0, lim], color="#999", lw=0.8); ax.set_xlabel("E (original)"); ax.set_ylabel(f"E ({lab})")
        ax.set_title(f"{lab} vs original", fontsize=10); ax.legend(frameon=False, fontsize=8)
    d_n = [r["score_neut"] - r["score_orig"] for r in rows if r["score_neut"] is not None and r["score_orig"] is not None]
    d_s = [r["score_scrub"] - r["score_orig"] for r in rows if r["score_scrub"] is not None and r["score_orig"] is not None]
    bins = np.arange(-10.5, 11.5, 1)
    if d_n:
        axes[2].hist(d_n, bins=bins, color=ORANGE, alpha=0.7, label=f"neutralized − original (n={len(d_n)})")
    if d_s:
        axes[2].hist(d_s, bins=bins, color=BLUE, alpha=0.5, label=f"random_scrub − original (n={len(d_s)})")
    axes[2].set_xlabel("Δ score"); axes[2].set_ylabel("actions"); axes[2].set_title("integer score change", fontsize=10); axes[2].legend(frameon=False, fontsize=8)
    for ax in axes:
        ax.grid(axis="y", color="#e5e5e5", lw=0.6); ax.set_axisbelow(True)
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)
    fig.tight_layout(); FIG.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG / "g4_counterfactual.png", dpi=150); plt.close(fig)
    R["figure"] = str((FIG / "g4_counterfactual.png").relative_to(REPO))
    RAW.mkdir(parents=True, exist_ok=True)
    out = RAW / f"g4_counterfactual_{ts}.json"
    out.write_text(json.dumps(R, indent=1, ensure_ascii=False))
    print(json.dumps({k: v for k, v in R.items() if k != "rows"}, indent=1))
    print("raw ->", out.relative_to(REPO))


if __name__ == "__main__":
    main()
