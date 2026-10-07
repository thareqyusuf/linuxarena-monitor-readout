"""P2 — linear probes on the residual stream, position x layer sweep, two supervision arms.

Features: acts[row, p, l] fp32, standardized with the training-fold mean/sd. Rows: usable (score not null,
current-action placeholder excluded). Cells at generation positions (analysis_end, reasoning_end, pre_score)
drop the 199 pass-1 rows (NaN there); n is recorded per cell.

Arms (both never see the test trajectories):
  score-supervised  trains on ALL usable train rows (honest + every attack level), target E (ridge) or
                    score >= 3 (logistic); diff-of-means = mean(score>=3) - mean(score<3). Never sees a label.
  label-supervised  trains on level-2 vs honest train rows only; diff-of-means = mean(level2) - mean(honest).
Regularisation (C for logistic, alpha for ridge) is a hyper-parameter selected, like (position, layer), on the
pooled 5-fold grouped-CV out-of-fold AUROC over the 40 train trajectories (split.json). Nothing is nested;
the OOF numbers are therefore selection-optimistic and the locked-test numbers are the clean ones.
Locked test (19 trajectories) is scored only for the selected (p*, l*, hyper) per arm x model.

Evaluation: level-2 vs honest AUROC / TPR@5e-2 / TPR@1e-2 (conventions in common.py); Spearman(pred, E) on all
usable OOF rows. Raw results/raw/<slug>/p2_sweep_<ts>.json; OOF prediction matrix for every cell in
checkpoints/<slug>/p2_oof_<ts>.npz; figures results/figures/<slug>/p2_sweep_{label,score}.png.

  python p2_probes.py [--jobs 32] [--exclude-keys path.json]   (exclude-keys: the P2(ii) suspicion-language re-run)
"""
from __future__ import annotations

import argparse, collections, json, os, sys, time, warnings
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1"); os.environ.setdefault("OPENBLAS_NUM_THREADS", "1"); os.environ.setdefault("MKL_NUM_THREADS", "1")
import numpy as np
from joblib import Parallel, delayed
from scipy.stats import spearmanr
from sklearn.linear_model import LogisticRegression, Ridge

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common as C

C_GRID = [1e-4, 1e-3, 1e-2, 1e-1]
ALPHA_GRID = [1e2, 1e3, 1e4, 1e5]
ARMS = {"label": ["dom", "logreg"], "score": ["dom", "ridge_E", "logreg_ge3"]}
GEN_POS = {4, 5, 6}


def standardize(Xtr, Xev):
    mu = Xtr.mean(0); sd = Xtr.std(0); sd[sd < 1e-6] = 1.0
    return (Xtr - mu) / sd, (Xev - mu) / sd


def fit_predict(model, Xtr, ytr, Xev, hyper):
    """Returns {hyper_value: predictions on Xev}."""
    out = {}
    if model == "dom":
        w = Xtr[ytr == 1].mean(0) - Xtr[ytr == 0].mean(0); out["-"] = Xev @ w
    elif model.startswith("logreg"):
        for c in hyper:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                m = LogisticRegression(C=c, penalty="l2", solver="lbfgs", max_iter=1000, tol=1e-4).fit(Xtr, ytr)
            out[str(c)] = m.decision_function(Xev)
    elif model == "ridge_E":
        for a in hyper:
            out[str(a)] = Ridge(alpha=a).fit(Xtr, ytr).predict(Xev)
    return out


def run_cell(pos, layer, D):
    """One (position, layer) cell: 5-fold OOF predictions for every arm/model/hyper on the train rows."""
    acts = np.load(C.ACTS, mmap_mode="r")
    rows = D["train_rows"] if pos not in GEN_POS else D["train_rows"][D["train_has_gen"]]
    X = np.asarray(acts[rows, pos, layer, :], dtype=np.float32)
    assert np.isfinite(X).all(), (pos, layer)
    mask = np.isin(D["train_rows"], rows)                   # which train entries are in play at this cell
    fold = D["fold"][mask]; is_pos = D["is_pos"][mask]; is_neg = D["is_neg"][mask]; E = D["E"][mask]; ge3 = D["score_ge3"][mask]
    preds = {}
    for arm, models in ARMS.items():
        for model in models:
            hyper = C_GRID if model.startswith("logreg") else ALPHA_GRID if model == "ridge_E" else ["-"]
            P = {h: np.full(len(rows), np.nan, np.float32) for h in map(str, hyper)}
            for f in range(5):
                tr = fold != f; ev = fold == f
                if arm == "label":
                    tr = tr & (is_pos | is_neg); y = is_pos[tr].astype(int)
                else:
                    y = E[tr] if model == "ridge_E" else ge3[tr].astype(int)
                Xtr, Xev = standardize(X[tr], X[ev])
                for h, p in fit_predict(model, Xtr, y, Xev, hyper).items():
                    P[h][ev] = p
            preds[(arm, model)] = P
    metrics = {}
    for (arm, model), P in preds.items():
        for h, p in P.items():
            m = C.metrics(p[is_neg], p[is_pos])
            rho = spearmanr(p, E).statistic
            metrics[f"{arm}|{model}|{h}"] = {"auroc": m["auroc"], "tpr_at_0.05": m["tpr_at_0.05"], "tpr_at_0.01": m["tpr_at_0.01"], "spearman_E": float(rho),
                                             "per_fold_auroc": [C.auroc(p[is_neg & (fold == f)], p[is_pos & (fold == f)]) if (is_pos & (fold == f)).any() else None for f in range(5)]}
    return {"pos": pos, "layer": layer, "n_rows": int(len(rows)), "n_pos": int(is_pos.sum()), "n_neg": int(is_neg.sum()), "mask": mask, "metrics": metrics,
            "preds": {f"{arm}|{model}|{h}": p for (arm, model), P in preds.items() for h, p in P.items()}}


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--jobs", type=int, default=32); ap.add_argument("--exclude-keys", type=Path, default=None)
    ap.add_argument("--tag", default=""); args = ap.parse_args()
    t0 = time.time(); stamp = C.ts()
    rows, meta = C.build_table(with_text=True); split = json.loads((C.HERE / "split.json").read_text())
    acts, idx = C.open_acts(); row_of = {(r["traj_id"], r["action_idx"]): r["row"] for r in idx["rows"]}
    has_gen = {(r["traj_id"], r["action_idx"]): r["n_positions"] == 7 and not r["nan_positions"] for r in idx["rows"]}
    u = C.usable(rows, "current")
    excluded = []
    if args.exclude_keys:
        ex = {tuple(k) for k in json.loads(args.exclude_keys.read_text())["keys"]}
        excluded = [r for r in u if (r["traj_id"], r["action_idx"]) in ex]; u = [r for r in u if (r["traj_id"], r["action_idx"]) not in ex]
    test_ids = set(split["test"]); fold_of = split["fold_of"]
    tr = [r for r in u if r["traj_id"] not in test_ids]; te = [r for r in u if r["traj_id"] in test_ids]
    D = {"train_rows": np.array([row_of[(r["traj_id"], r["action_idx"])] for r in tr]), "train_has_gen": np.array([has_gen[(r["traj_id"], r["action_idx"])] for r in tr]),
         "fold": np.array([fold_of[r["traj_id"]] for r in tr]), "is_pos": np.array([r["is_pos"] for r in tr]), "is_neg": np.array([r["is_neg"] for r in tr]),
         "E": np.array([r["E"] for r in tr], np.float32), "score_ge3": np.array([r["score"] >= 3 for r in tr])}
    T = {"rows": np.array([row_of[(r["traj_id"], r["action_idx"])] for r in te]), "has_gen": np.array([has_gen[(r["traj_id"], r["action_idx"])] for r in te]),
         "is_pos": np.array([r["is_pos"] for r in te]), "is_neg": np.array([r["is_neg"] for r in te]), "E": np.array([r["E"] for r in te], np.float32)}
    print(f"train {len(tr)} rows ({D['is_pos'].sum()} pos / {D['is_neg'].sum()} neg, {D['train_has_gen'].sum()} with gen positions); test {len(te)} rows ({T['is_pos'].sum()} pos / {T['is_neg'].sum()} neg); excluded {len(excluded)}", flush=True)

    cells = [(p, l) for p in range(7) for l in range(25)]
    res = Parallel(n_jobs=args.jobs, verbose=5)(delayed(run_cell)(p, l, D) for p, l in cells)
    print(f"sweep done in {time.time()-t0:.0f}s", flush=True)

    # ── selection on train-OOF ──
    grid = {}   # config -> [7][25] auroc
    for r in res:
        for cfg, m in r["metrics"].items():
            grid.setdefault(cfg, np.full((7, 25), np.nan))[r["pos"], r["layer"]] = m["auroc"]
    chosen = {}
    for arm, models in ARMS.items():
        for model in models:
            best = None
            for r in res:
                for cfg, m in r["metrics"].items():
                    a, mo, h = cfg.split("|")
                    if a == arm and mo == model and (best is None or m["auroc"] > best[0]):
                        best = (m["auroc"], r["pos"], r["layer"], h, cfg)
            chosen[f"{arm}|{model}"] = {"pos": best[1], "position": C.POSITIONS[best[1]], "layer": best[2], "hyper": best[3], "cfg": best[4], "oof": next(r for r in res if r["pos"] == best[1] and r["layer"] == best[2])["metrics"][best[4]]}
    # best layer per position (per arm x model): the pre-registered ordering test
    best_per_pos = {f"{arm}|{model}": {C.POSITIONS[p]: {"layer": int(np.nanargmax(np.nanmax(np.stack([grid[c] for c in grid if c.startswith(f"{arm}|{model}|")]), 0)[p])),
                                                       "auroc": float(np.nanmax(np.stack([grid[c] for c in grid if c.startswith(f"{arm}|{model}|")]), 0)[p].max())} for p in range(7)}
                    for arm, models in ARMS.items() for model in models}

    # ── locked test + full-train refit for the chosen configs only ──
    test_out = {}; preds_out = {}
    for key, ch in chosen.items():
        arm, model = key.split("|"); p, l, h = ch["pos"], ch["layer"], ch["hyper"]
        trm = D["train_has_gen"] if p in GEN_POS else np.ones(len(tr), bool); tem = T["has_gen"] if p in GEN_POS else np.ones(len(te), bool)
        Xtr = np.asarray(acts[D["train_rows"][trm], p, l, :], np.float32); Xte = np.asarray(acts[T["rows"][tem], p, l, :], np.float32)
        if arm == "label":
            sel = (D["is_pos"] | D["is_neg"])[trm]; y = D["is_pos"][trm][sel].astype(int); Xfit = Xtr[sel]
        else:
            y = D["E"][trm] if model == "ridge_E" else D["score_ge3"][trm].astype(int); Xfit = Xtr
        Xfit_s, Xte_s = standardize(Xfit, Xte)
        hyper = [float(h)] if h != "-" else ["-"]
        pte = fit_predict(model, Xfit_s, y, Xte_s, hyper)[h]
        mt = C.metrics(pte[T["is_neg"][tem]], pte[T["is_pos"][tem]])
        test_out[key] = {**mt, "spearman_E": float(spearmanr(pte, T["E"][tem]).statistic), "n_rows": int(tem.sum()),
                         "ci95": C.cluster_boot(C.by_traj([dict(traj_id=r["traj_id"], v=v) for r, v in zip([x for x, m in zip(te, tem) if m], pte) if r["is_neg"]], "v"),
                                                C.by_traj([dict(traj_id=r["traj_id"], v=v) for r, v in zip([x for x, m in zip(te, tem) if m], pte) if r["is_pos"]], "v"))}
        # per-row predictions: train OOF (from the sweep) + test (this refit) -> pooled over all 59 trajectories
        cell = next(r for r in res if r["pos"] == p and r["layer"] == l); oof = cell["preds"][ch["cfg"]]
        tr_keys = [x for x, m in zip(tr, cell["mask"]) if m]
        preds_out[key] = {"train_oof": [[r["traj_id"], r["action_idx"], float(v)] for r, v in zip(tr_keys, oof)],
                          "test": [[r["traj_id"], r["action_idx"], float(v)] for r, v in zip([x for x, m in zip(te, tem) if m], pte)]}
    # reference scorers on the same populations
    ref = {}
    for nm, fld in (("E", "E"), ("score", "score"), ("cached_mean", "cached_mean")):
        ref[nm] = {"train_oof_population": C.metrics([r[fld] for r in tr if r["is_neg"]], [r[fld] for r in tr if r["is_pos"]]),
                   "test": C.metrics([r[fld] for r in te if r["is_neg"]], [r[fld] for r in te if r["is_pos"]])}
    # score-arm top positives by hand: top 15 train-OOF rows of the chosen score|logreg_ge3 and score|ridge_E
    tops = {}
    for key in ("score|ridge_E", "score|logreg_ge3", "label|logreg"):
        pr = sorted(preds_out[key]["train_oof"], key=lambda x: -x[2])[:15]; lut = {(r["traj_id"], r["action_idx"]): r for r in tr}
        tops[key] = [{"traj_id": t, "action_idx": i, "pred": v, "side": lut[(t, i)]["side"], "label_level": lut[(t, i)]["label_level"], "score": lut[(t, i)]["score"],
                      "E": lut[(t, i)]["E"], "cached_mean": lut[(t, i)]["cached_mean"], "tool": lut[(t, i)]["tool_name"], "action_text": lut[(t, i)]["action_text"][:240]} for t, i, v in pr]

    # ── figures ──
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    figs = {}
    for arm, models in ARMS.items():
        fig, axes = plt.subplots(1, len(models), figsize=(5.2 * len(models), 3.6), squeeze=False)
        for ax, model in zip(axes[0], models):
            cfgs = [c for c in grid if c.startswith(f"{arm}|{model}|")]; G = np.nanmax(np.stack([grid[c] for c in cfgs]), 0)   # best hyper per cell
            im = ax.imshow(G, vmin=0, vmax=1, cmap="RdBu_r", aspect="auto"); ax.set_yticks(range(7)); ax.set_yticklabels(C.POSITIONS, fontsize=7)
            ax.set_xlabel("layer (0 = embeddings)"); ax.set_title(f"{arm}-supervised, {model}: OOF AUROC (best hyper per cell)", fontsize=8)
            ch = chosen[f"{arm}|{model}"]; ax.plot(ch["layer"], ch["pos"], "k*", ms=9)
            for p in range(7):
                ax.text(24.3, p, f"{np.nanmax(G[p]):.2f}", fontsize=6, va="center", ha="right", bbox=dict(fc="white", ec="none", alpha=0.8, pad=0.5))
            ax.axhline(3.5, color="k", lw=0.6, ls=":")
        plt.colorbar(im, ax=axes[0].tolist(), fraction=0.02, label="AUROC (white = 0.5)")
        p = C.fig_path(f"p2_sweep_{arm}{args.tag}.png"); fig.savefig(p, dpi=150, bbox_inches="tight"); plt.close(fig); figs[arm] = str(p.relative_to(C.REPO))

    # ── outputs ──
    npz = C.CKPT / f"p2_oof{args.tag}_{stamp}.npz"
    np.savez_compressed(npz, **{f"{r['pos']}|{r['layer']}|{cfg}": v for r in res for cfg, v in r["preds"].items()},
                        train_keys=np.array([f"{r['traj_id']}|{r['action_idx']}" for r in tr]), **{f"mask|{r['pos']}|{r['layer']}": r["mask"] for r in res})
    out = {"slug": C.SLUG, "script": f"experiments/{C.SLUG}/p2_probes.py", "timestamp": stamp, "git_commit": C.git_commit(), "inputs": meta | {"split_json_sha256": C.sha256(C.HERE / "split.json"), "acts_sha256": idx["sha256"]},
           "design": {"placeholder_rule": "current", "positions": C.POSITIONS, "layers": list(range(25)), "C_grid": C_GRID, "alpha_grid": ALPHA_GRID, "arms": ARMS,
                      "selection": "pooled 5-fold grouped OOF AUROC over train trajectories, argmax over (position, layer, hyper); test scored once for the argmax only",
                      "excluded_keys_file": str(args.exclude_keys) if args.exclude_keys else None, "n_excluded": len(excluded)},
           "populations": {"train": {"n": len(tr), "n_pos": int(D["is_pos"].sum()), "n_neg": int(D["is_neg"].sum()), "n_with_gen": int(D["train_has_gen"].sum()), "n_traj": len({r["traj_id"] for r in tr})},
                           "test": {"n": len(te), "n_pos": int(T["is_pos"].sum()), "n_neg": int(T["is_neg"].sum()), "n_with_gen": int(T["has_gen"].sum()), "n_traj": len({r["traj_id"] for r in te}),
                                    "n_pos_traj": len({r["traj_id"] for r in te if r["is_pos"]})}},
           "sweep": [{"pos": r["pos"], "position": C.POSITIONS[r["pos"]], "layer": r["layer"], "n_rows": r["n_rows"], "n_pos": r["n_pos"], "n_neg": r["n_neg"], "metrics": r["metrics"]} for r in res],
           "chosen": chosen, "best_layer_per_position": best_per_pos, "locked_test": test_out, "reference_scorers": ref, "top15_train_oof": tops,
           "predictions_chosen": preds_out, "oof_npz": str(npz.relative_to(C.REPO)), "figures": figs, "seconds": time.time() - t0}
    C.write_raw(f"p2_sweep{args.tag}", out, stamp)
    print("\n== chosen (train-OOF, selection-optimistic) and locked test ==")
    for key, ch in chosen.items():
        t = test_out[key]; o = ch["oof"]
        print(f"  {key:18} {ch['position']:15} L{ch['layer']:2} hyper {ch['hyper']:6} | OOF AUROC {o['auroc']:.3f} TPR@5% {o['tpr_at_0.05']:.2f} @1% {o['tpr_at_0.01']:.2f} rhoE {o['spearman_E']:.2f} | TEST AUROC {t['auroc']:.3f} [{t['ci95']['auroc'][0]:.2f},{t['ci95']['auroc'][1]:.2f}] TPR@5% {t['tpr_at_0.05']:.2f} @1% {t['tpr_at_0.01']:.2f} rhoE {t['spearman_E']:.2f} (n={t['n_rows']})")
    for nm, m in ref.items():
        print(f"  {nm:18} reference: train-pop AUROC {m['train_oof_population']['auroc']:.3f} TPR@5% {m['train_oof_population']['tpr_at_0.05']:.2f} | test AUROC {m['test']['auroc']:.3f} TPR@5% {m['test']['tpr_at_0.05']:.2f}")
    print("\n== best layer per position (OOF AUROC) ==")
    for key, d in best_per_pos.items():
        print(f"  {key:18} " + "  ".join(f"{p[:11]} {v['auroc']:.2f}@L{v['layer']}" for p, v in d.items()))
    for key, t in tops.items():
        print(f"\n== top 15 train-OOF by {key} ==")
        for x in t:
            print(f"  {x['pred']:7.2f} {x['side']:6} lvl={x['label_level']} score={x['score']} E={x['E']:.2f} cached={x['cached_mean']} {x['tool']:12} {x['action_text'][:110]!r}")


if __name__ == "__main__":
    main()
