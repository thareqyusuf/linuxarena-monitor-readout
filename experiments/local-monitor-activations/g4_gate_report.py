#!/usr/bin/env python3
"""G4 gate report: what the reviewer asked for at the 200-action checkpoint (CPU only, no model).

Reads <out>/scores.jsonl and reports, over the first --n completed actions:
  1. score distribution overall, split by side, and again with has_attachment_placeholder actions REMOVED
  2. the --top highest-scoring actions with their current_action text and the model's own analysis note
     (recovered from the stored `generation`), so a score >= 3 can be inspected for what triggered it
  3. fraction with score_parse_ok == False, fraction hitting the --max-new-tokens budget, channel structure
  4. expected_score distribution and its Spearman correlation against score (and against cached_mean)
  5. the gate arithmetic itself, restated from the records

Writes results/raw/<slug>/g4_gate_report_<ts>.json and results/figures/<slug>/g4_gate_scores.png.
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

HERE = Path(__file__).resolve()
SLUG = HERE.parent.name
REPO = HERE.parents[2]
RAW = REPO / "results" / "raw" / SLUG
FIG = REPO / "results" / "figures" / SLUG
TS = dt.datetime.now(dt.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
sys.path.insert(0, str(HERE.parent))
import monitor_prompt as mp  # noqa: E402
import run_monitor as rm  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--out", type=Path, default=REPO / "checkpoints" / SLUG / "out" / "main")
ap.add_argument("--n", type=int, default=200, help="use the first n completed (ok) records; 0 = all")
ap.add_argument("--top", type=int, default=5)
ap.add_argument("--label", default="gate200")
args = ap.parse_args()

recs = [json.loads(l) for l in (args.out / "scores.jsonl").open()]
ok = [r for r in recs if r.get("status") == "ok"]
if args.n:
    ok = ok[: args.n]
assert ok, "no completed records yet"
meta = json.loads((args.out / "run_meta.json").read_text())
BUDGET = ok[0].get("max_new_tokens", 1024)


def dist(xs) -> dict:
    xs = [x for x in xs if x is not None]
    if not xs:
        return {"n": 0}
    a = np.asarray(xs, float)
    return {"n": len(xs), "mean": float(a.mean()), "sd": float(a.std(ddof=1)) if len(a) > 1 else 0.0,
            "min": float(a.min()), "p25": float(np.percentile(a, 25)), "median": float(np.median(a)),
            "p75": float(np.percentile(a, 75)), "p90": float(np.percentile(a, 90)), "max": float(a.max())}


def hist(rows) -> dict:
    c = collections.Counter(r["score"] for r in rows if r.get("score") is not None)
    return {str(v): c.get(float(v), 0) for v in range(11)}


def frac_ge(rows, k) -> float:
    sc = [r["score"] for r in rows if r.get("score") is not None]
    return (sum(s >= k for s in sc) / len(sc)) if sc else 0.0


no_ph = [r for r in ok if not r["has_attachment_placeholder"]]
ph = [r for r in ok if r["has_attachment_placeholder"]]
R: dict = {"slug": SLUG, "script": str(HERE.relative_to(REPO)), "timestamp": TS, "label": args.label,
           "out_dir": str(args.out.relative_to(REPO)), "n_records_in_file": len(recs), "n_ok_used": len(ok),
           "run_meta": {k: meta.get(k) for k in ("render", "attn_chunk_size", "reasoning_effort", "template_date",
                                                 "score_source", "expected_score_source", "git_commit", "gate")}}

# ── 1. score distribution, by side, and with placeholders removed ───────────────────────────
R["scores"] = {
    "all": {"n": len(ok), "hist": hist(ok), "frac_ge3": frac_ge(ok, 3), "frac_ge6": frac_ge(ok, 6),
            "dist": dist([r["score"] for r in ok])},
    "by_side": {s: {"n": sum(r["side"] == s for r in ok), "hist": hist([r for r in ok if r["side"] == s]),
                    "frac_ge3": frac_ge([r for r in ok if r["side"] == s], 3),
                    "dist": dist([r["score"] for r in ok if r["side"] == s])} for s in ("honest", "attack")},
    "placeholders_removed": {"n": len(no_ph), "hist": hist(no_ph), "frac_ge3": frac_ge(no_ph, 3),
                             "dist": dist([r["score"] for r in no_ph]),
                             "by_side": {s: {"n": sum(r["side"] == s for r in no_ph), "hist": hist([r for r in no_ph if r["side"] == s]),
                                             "frac_ge3": frac_ge([r for r in no_ph if r["side"] == s], 3)} for s in ("honest", "attack")}},
    "placeholders_only": {"n": len(ph), "hist": hist(ph), "frac_ge3": frac_ge(ph, 3), "dist": dist([r["score"] for r in ph])},
    "note": "placeholder actions are those whose rendered prompt contains attachment:// (track L finding 2); the cached monitor saw the real content, this run does not",
}

# ── 3. parse health ─────────────────────────────────────────────────────────────────────────
R["parse_health"] = {
    "n": len(ok),
    "frac_score_parse_ok": sum(r.get("score") is not None for r in ok) / len(ok),
    "frac_score_parse_failed": sum(r.get("score") is None for r in ok) / len(ok),
    "n_score_parse_failed": sum(r.get("score") is None for r in ok),
    "frac_hit_budget": sum(bool(r.get("hit_max_new_tokens")) for r in ok) / len(ok),
    "n_hit_budget": sum(bool(r.get("hit_max_new_tokens")) for r in ok),
    "budget": BUDGET,
    "frac_opened_analysis": sum(bool(r.get("opened_analysis")) for r in ok) / len(ok),
    "frac_has_final_channel": sum(bool(r.get("has_final_channel")) for r in ok) / len(ok),
    "score_regex_used": dict(collections.Counter(r.get("score_regex") for r in ok)),
    "n_score_out_of_range": sum(bool(r.get("score_out_of_range")) for r in ok),
    "n_generated": dist([r.get("n_generated") for r in ok]),
    "analysis_chars": dist([r.get("analysis_chars") for r in ok]),
    "parse_failures": [{"key": f"{r['traj_id']}|{r['action_idx']}", "n_generated": r.get("n_generated"),
                        "hit_budget": r.get("hit_max_new_tokens"), "has_final_channel": r.get("has_final_channel"),
                        "generation_tail": (r.get("generation") or "")[-400:]}
                       for r in ok if r.get("score") is None][:10],
}

# ── 4. expected_score distribution and rank correlation ─────────────────────────────────────
from scipy.stats import spearmanr  # noqa: E402
E = [r["expected_score"] for r in ok if r.get("expected_score") is not None]
paired = [(r["expected_score"], r["score"]) for r in ok if r.get("expected_score") is not None and r.get("score") is not None]
pc = [(r["expected_score"], r["cached_mean"]) for r in ok if r.get("expected_score") is not None and r.get("cached_mean") is not None]
sc_c = [(r["score"], r["cached_mean"]) for r in ok if r.get("score") is not None and r.get("cached_mean") is not None]
R["expected_score"] = {
    "dist": dist(E), "dist_placeholders_removed": dist([r["expected_score"] for r in no_ph if r.get("expected_score") is not None]),
    "by_side": {s: dist([r["expected_score"] for r in ok if r["side"] == s and r.get("expected_score") is not None]) for s in ("honest", "attack")},
    "digit_token_mass": dist([r.get("digit_token_mass") for r in ok]),
    "spearman_expected_vs_score": float(spearmanr([a for a, _ in paired], [b for _, b in paired]).statistic) if len(paired) > 2 else None,
    "spearman_expected_vs_score_p": float(spearmanr([a for a, _ in paired], [b for _, b in paired]).pvalue) if len(paired) > 2 else None,
    "n_paired": len(paired),
    "spearman_expected_vs_cached": float(spearmanr([a for a, _ in pc], [b for _, b in pc]).statistic) if len(pc) > 2 else None,
    "spearman_score_vs_cached": float(spearmanr([a for a, _ in sc_c], [b for _, b in sc_c]).statistic) if len(sc_c) > 2 else None,
    "mean_expected_by_score": {str(v): dist([e for e, s in paired if s == v]) for v in sorted({s for _, s in paired})},
}

# ── 2. the top-scoring actions, with the action text and the model's own analysis note ──────
trajs, _ = mp.load_trajs()
top = sorted([r for r in ok if r.get("score") is not None], key=lambda r: (-r["score"], -(r.get("expected_score") or 0)))[: args.top]
R["top_scoring"] = []
for r in top:
    t = trajs[r["traj_id"]]
    act = t["actions"][r["action_idx"]]
    pt = rm.parse_generation(r.get("generation") or "")
    R["top_scoring"].append({
        "key": f"{r['traj_id']}|{r['action_idx']}", "side": r["side"], "env": r["env"], "side_task": r.get("side_task"),
        "score": r["score"], "expected_score": r["expected_score"], "cached_scores": r["cached_scores"], "cached_mean": r["cached_mean"],
        "has_attachment_placeholder": r["has_attachment_placeholder"], "n_prompt_tokens": r["n_prompt_tokens"],
        "step_index": r["step_index"], "trajectory_length": r["trajectory_length"],
        "current_action": mp.fmp.format_action(act, hide_output=True)[:1500],
        "analysis_note_text": pt.get("analysis_text"),
        "final_text": (pt.get("final_text") or "")[:1500],
    })

# ── 5. gate arithmetic ──────────────────────────────────────────────────────────────────────
sc = [r["score"] for r in ok if r.get("score") is not None]
gate_file = args.out / "gate.json"
R["gate"] = {"recomputed_frac_score_ge3": frac_ge(ok, 3),
             "recomputed_sd_expected_score": statistics.stdev(E) if len(E) > 1 else 0.0,
             "thresholds": meta.get("gate", {}),
             "rule": "halt iff frac(score>=3) < min_frac_ge3 AND stdev(expected_score) < min_sd",
             "decision_from_runner": json.loads(gate_file.read_text()) if gate_file.exists() else None}

# ── figure ──────────────────────────────────────────────────────────────────────────────────
import matplotlib  # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
fig, axes = plt.subplots(1, 3, figsize=(15, 4.2))
BLUE, ORANGE = "#2a78d6", "#eb6834"
w = 0.4
xs = np.arange(11)
h_h = [hist([r for r in ok if r["side"] == "honest"])[str(v)] for v in range(11)]
h_a = [hist([r for r in ok if r["side"] == "attack"])[str(v)] for v in range(11)]
axes[0].bar(xs - w / 2, h_h, w, color=BLUE, label=f"honest (n={sum(h_h)})", linewidth=0)
axes[0].bar(xs + w / 2, h_a, w, color=ORANGE, label=f"attack (n={sum(h_a)})", linewidth=0)
axes[0].set_xlabel("score (generated, unforced)"); axes[0].set_ylabel("actions"); axes[0].set_xticks(xs)
axes[0].set_title(f"score by side, first {len(ok)} actions", fontsize=10); axes[0].legend(frameon=False, fontsize=9)
h_np = [hist(no_ph)[str(v)] for v in range(11)]
h_p = [hist(ph)[str(v)] for v in range(11)]
axes[1].bar(xs - w / 2, h_np, w, color=BLUE, label=f"no placeholder (n={sum(h_np)})", linewidth=0)
axes[1].bar(xs + w / 2, h_p, w, color=ORANGE, label=f"attachment:// (n={sum(h_p)})", linewidth=0)
axes[1].set_xlabel("score"); axes[1].set_ylabel("actions"); axes[1].set_xticks(xs)
axes[1].set_title("score by placeholder status", fontsize=10); axes[1].legend(frameon=False, fontsize=9)
for s, col in (("honest", BLUE), ("attack", ORANGE)):
    e = [r["expected_score"] for r in ok if r["side"] == s and r.get("expected_score") is not None]
    sc_ = [r["score"] for r in ok if r["side"] == s and r.get("expected_score") is not None and r.get("score") is not None]
    ee = [r["expected_score"] for r in ok if r["side"] == s and r.get("expected_score") is not None and r.get("score") is not None]
    axes[2].scatter(np.array(sc_) + (0.12 if s == "attack" else -0.12) + np.random.RandomState(0).uniform(-0.05, 0.05, len(sc_)),
                    ee, s=14, color=col, alpha=0.6, linewidths=0, label=s)
axes[2].set_xlabel("generated score"); axes[2].set_ylabel("expected_score (forced readout)")
rho = R["expected_score"]["spearman_expected_vs_score"]
axes[2].set_title(f"forced readout vs generated score (Spearman {rho:.2f})" if rho is not None else "forced readout vs generated score", fontsize=10)
axes[2].legend(frameon=False, fontsize=9)
for ax in axes:
    ax.grid(axis="y", color="#e5e5e5", lw=0.6); ax.set_axisbelow(True)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
fig.tight_layout()
FIG.mkdir(parents=True, exist_ok=True)
fig.savefig(FIG / f"g4_{args.label}_scores.png", dpi=150)
plt.close(fig)
R["figure"] = str((FIG / f"g4_{args.label}_scores.png").relative_to(REPO))

RAW.mkdir(parents=True, exist_ok=True)
outp = RAW / f"g4_gate_report_{TS}.json"
outp.write_text(json.dumps(R, indent=1, ensure_ascii=False))
print(json.dumps({k: v for k, v in R.items() if k != "top_scoring"}, indent=1, default=str))
for t in R["top_scoring"]:
    print(f"\n##### score {t['score']} E {t['expected_score']:.2f} | {t['side']} {t['key'][:8]}#{t['key'].split('|')[1]} "
          f"[{t['step_index']}/{t['trajectory_length']}] {t['env']} side_task={t['side_task']} cached={t['cached_scores']} ph={t['has_attachment_placeholder']}")
    print("  CURRENT ACTION:", (t["current_action"] or "")[:600].replace("\n", "\n     "))
    print("  ANALYSIS NOTE:", (t["analysis_note_text"] or "(none)")[:700].replace("\n", "\n     "))
print("\nraw ->", outp.relative_to(REPO))
