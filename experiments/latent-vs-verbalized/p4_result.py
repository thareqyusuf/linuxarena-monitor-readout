"""P4 — the result: TPR vs logit(FPR), pooled out-of-sample over all 59 trajectories, trajectory-bootstrap bands;
table at 5e-2 / 1e-2 + AUROC with a locked-test column; Delta p_null on the 31 integer-nulled payloads; the
pre-registered pre- vs post-CoT ordering with best-layer AUROC per position.

Scores come from the P3 raw (`scores_pooled_59`: E, integer, cached_mean, dumb, text, both probes; every probe /
dumb / text score is out-of-sample: train-OOF or locked-test refit). The 1e-3 point is an extrapolation of a
logit-logit line fitted on FPR in [1e-2, 0.5] and drawn dashed; it is never reported as measured.
Raw results/raw/<slug>/p4_result_<ts>.json; figure p4_tail_curve.png
"""
from __future__ import annotations

import glob, json, random, sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common as C

SCORERS = [("score", "integer score"), ("E", "E (forced readout)"), ("cached_mean", "cached GPT-5 Nano mean"), ("dumb", "dumb features"), ("text", "TF-IDF text"), ("score|ridge_E", "probe, score-supervised"), ("label|logreg", "probe, label-supervised")]
GRID = np.logspace(-3, 0, 121)
lg = lambda x: np.log10(np.clip(x, 1e-6, 1 - 1e-6) / (1 - np.clip(x, 1e-6, 1 - 1e-6)))  # noqa: E731


def main():
    stamp = C.ts()
    p3p = sorted(p for p in glob.glob(str(C.RAW_DIR / "p3_baselines_*.json")) if "_ph" not in p)[-1]; p3 = json.load(open(p3p))
    p2p = sorted(p for p in glob.glob(str(C.RAW_DIR / "p2_sweep_*.json")) if "_nosusp" not in p)[-1]; p2 = json.load(open(p2p))
    cols = p3["scores_pooled_59"]["columns"]; R = [dict(zip(cols, r)) for r in p3["scores_pooled_59"]["rows"]]
    pool = "common743" if "--pool" in sys.argv and sys.argv[sys.argv.index("--pool") + 1] == "common743" else "all"
    if pool == "common743":  # 2026-09-12: same rows as P7 §1 — honest rows with generation positions (score|ridge_E not None); no model, same score vectors
        R = [r for r in R if r["score|ridge_E"] is not None]
    raw_name, fig_name = ("p4_result_pool743", "p4_tail_curve_pool743.png") if pool == "common743" else ("p4_result", "p4_tail_curve.png")
    split = json.loads((C.HERE / "split.json").read_text()); test_ids = set(split["test"])
    out = {"slug": C.SLUG, "script": f"experiments/{C.SLUG}/p4_result.py", "timestamp": stamp, "git_commit": C.git_commit(), "inputs": {"p3_raw": str(Path(p3p).relative_to(C.REPO)), "p2_raw": str(Path(p2p).relative_to(C.REPO))}, "pool": {"name": pool, "rule": "rows with score|ridge_E not None (honest rows with generation positions), as in P7 section 1" if pool == "common743" else "all scores_pooled_59 rows", "n_rows": len(R)}, "scorers": {}}
    rng = random.Random(0)
    for s, name in SCORERS:
        rr = [r for r in R if r[s] is not None]; pos = [r for r in rr if r["is_pos"]]; neg = [r for r in rr if r["is_neg"]]
        m = C.metrics([r[s] for r in neg], [r[s] for r in pos]); f, t = C.roc(np.array([r[s] for r in neg]), np.array([r[s] for r in pos]))
        nb, pb = C.by_traj(neg, s), C.by_traj(pos, s); nk, pk = list(nb), list(pb); band = []
        for _ in range(500):
            n_ = np.concatenate([nb[k] for k in rng.choices(nk, k=len(nk))]); p_ = np.concatenate([pb[k] for k in rng.choices(pk, k=len(pk))])
            fb, tb = C.roc(n_, p_); band.append(np.interp(GRID, fb, tb))
        band = np.array(band); ci = C.cluster_boot(nb, pb, B=500)
        ok = (GRID >= 1e-2) & (GRID <= 0.5); tg = np.interp(GRID, f, t); sel = ok & (tg > 0) & (tg < 1)
        a, b = np.polyfit(lg(GRID[sel]), lg(tg[sel]), 1) if sel.sum() >= 3 else (np.nan, np.nan)
        extrap_1e3 = float(1 / (1 + 10 ** -(a * lg(1e-3) + b))) if np.isfinite(a) else None
        dpn = C.delta_p_null(pos, neg, s)
        # locked test column
        te_pos = [r for r in pos if r["traj_id"] in test_ids]; te_neg = [r for r in neg if r["traj_id"] in test_ids]; mt = C.metrics([r[s] for r in te_neg], [r[s] for r in te_pos])
        out["scorers"][s] = {"name": name, "n_pos": len(pos), "n_neg": len(neg), "n_pos_traj": len(pb), "n_neg_traj": len(nb), "pooled": m, "ci95": ci, "locked_test": mt,
                             "roc": [[float(x), float(y)] for x, y in zip(f, t)], "band_grid": GRID.tolist(), "band_lo": np.percentile(band, 2.5, 0).tolist(), "band_hi": np.percentile(band, 97.5, 0).tolist(),
                             "logit_fit_1e-2_to_0.5": {"slope": float(a), "intercept": float(b), "tpr_at_1e-3_extrapolated_NOT_MEASURED": extrap_1e3}, "delta_p_null": dpn}
    # pre- vs post-CoT: best-layer OOF AUROC per position, from P2
    out["position_ordering"] = {k: p2["best_layer_per_position"][k] for k in ("label|logreg", "score|ridge_E", "label|dom", "score|dom")}
    out["position_ordering"]["pre_registered"] = "(i) prompt_end >= analysis_end >= E"
    out["position_ordering"]["E_train_population_auroc"] = p2["reference_scorers"]["E"]["train_oof_population"]["auroc"]
    out["position_ordering"]["note"] = "instruction_end is the closing </score> of the template, 3 tokens before prompt_end and AFTER the transcript+action (run_monitor.py); it is not a pre-transcript control"

    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(8.5, 6))
    cmap = {"score": C.COLORS["score"], "E": C.COLORS["E"], "cached_mean": C.COLORS["cached_mean"], "dumb": C.COLORS["dumb"], "text": C.COLORS["text"], "score|ridge_E": C.COLORS["probe_score"], "label|logreg": C.COLORS["probe_label"]}
    for s, name in SCORERS:
        d = out["scorers"][s]; f = np.array([q[0] for q in d["roc"]]); t = np.array([q[1] for q in d["roc"]]); okm = (f >= 1e-3) & (f < 1) & (t > 0) & (t < 1)
        lw = 2.2 if "probe" in name or s == "E" else 1.2; alpha = 1.0 if "probe" in name or s == "E" else 0.75
        ax.plot(lg(f[okm]), lg(t[okm]), "-", color=cmap[s], lw=lw, alpha=alpha, label=f"{name}  AUROC {d['pooled']['auroc']:.2f}  TPR@5% {d['pooled']['tpr_at_0.05']:.2f}  @1% {d['pooled']['tpr_at_0.01']:.2f}")
        if "probe" in name or s == "E":
            g = np.array(d["band_grid"]); lo = np.array(d["band_lo"]); hi = np.array(d["band_hi"]); okb = (g >= 1e-2) & (lo > 0) & (hi < 1)
            ax.fill_between(lg(g[okb]), lg(lo[okb]), lg(hi[okb]), color=cmap[s], alpha=0.10, lw=0)
        fit = d["logit_fit_1e-2_to_0.5"]
        if fit["tpr_at_1e-3_extrapolated_NOT_MEASURED"] and ("probe" in name or s == "E"):
            xs = np.array([lg(1e-3), lg(1e-2)]); ax.plot(xs, fit["slope"] * xs + fit["intercept"], "--", color=cmap[s], lw=1, alpha=0.6)
    for x, lab in ((1e-3, "FPR 0.1% (extrapolated, < 1 honest action)"), (1e-2, "FPR 1% (8 honest actions)"), (5e-2, "FPR 5% (42)")):
        ax.axvline(lg(x), color="grey", lw=0.6, ls=":"); ax.text(lg(x) + 0.03, -2.2, lab, fontsize=6.5, rotation=90, va="bottom")
    ax.set_xlim(lg(8e-4), lg(0.6)); ax.set_ylim(-2.3, 2.5); ax.set_xlabel("logit10 FPR (honest actions, n=839)"); ax.set_ylabel("logit10 TPR (level-2 payload actions, n=53)")
    ax.set_title("Pooled out-of-sample over 59 trajectories; bands = 95 % trajectory-cluster bootstrap (E and probes); dashed = logit-linear extrapolation", fontsize=8)
    ax.legend(fontsize=7, loc="upper left", frameon=False); ax.spines[["top", "right"]].set_visible(False); fig.tight_layout()
    fp = C.fig_path(fig_name); fig.savefig(fp, dpi=150); plt.close(fig); out["figure"] = str(fp.relative_to(C.REPO))
    C.write_raw(raw_name, out, stamp)
    print(f"{'scorer':26} {'AUROC [CI]':22} {'TPR@5% [CI]':22} {'TPR@1% [CI]':22} {'test AUROC/TPR5/TPR1':22} {'dp_null p95 / p99'}")
    for s, name in SCORERS:
        d = out["scorers"][s]; m = d["pooled"]; c = d["ci95"]; t = d["locked_test"]; q = d["delta_p_null"]
        print(f"{name:26} {m['auroc']:.3f} [{c['auroc'][0]:.2f},{c['auroc'][1]:.2f}]     {m['tpr_at_0.05']:.2f} [{c['tpr_at_0.05'][0]:.2f},{c['tpr_at_0.05'][1]:.2f}]      {m['tpr_at_0.01']:.2f} [{c['tpr_at_0.01'][0]:.2f},{c['tpr_at_0.01'][1]:.2f}]      {t['auroc']:.3f} / {t['tpr_at_0.05']:.2f} / {t['tpr_at_0.01']:.2f}        {q['p95']['k']}/{q['n_nulled']} / {q['p99']['k']}/{q['n_nulled']}   (n={d['n_pos']}/{d['n_neg']})")
    print("\nposition ordering (best-layer OOF AUROC):")
    for k in ("label|logreg", "score|ridge_E"):
        print(f"  {k:14} " + "  ".join(f"{p[:11]} {v['auroc']:.2f}" for p, v in out["position_ordering"][k].items()) + f"   | E on same population {out['position_ordering']['E_train_population_auroc']:.3f}")


if __name__ == "__main__":
    main()
