"""P3 — baselines for the two probes chosen in P2 (label|logreg and score|ridge_E), same split, same metric.

  1 random directions      200 random unit vectors at (p*, l*), scored on the train population; probe's percentile
  2 shuffled labels        50x: positives re-assigned across train ATTACK trajectories (per-trajectory count and
                           relative position preserved, trajectory identity broken), refit at (p*, l*, C*), OOF AUROC
                           against the permuted labels. Label arm only (the score arm never sees labels).
  3 dumb features          logistic on [step_index, position_in_trajectory, n_prompt_tokens, n_context_tokens,
                           command_length, has_sudo, has_chmod, has_curl, has_nc]
  4 text ceiling           TF-IDF char 3-5-grams on the rendered current action (output hidden) + logistic
  5 dumb + probe           logistic on dumb features + the probe's out-of-sample score (stacking, grouped CV)
  6 leave-one-environment-out   10 env folds over ALL 59 trajectories (honest in every fold); n positives per env
  7 matched negatives      per level-2 action up to 5 honest actions matched on length decile x position decile x tool
  8 residualise            probe score regressed on the dumb features, residual evaluated
  9 leave-one-category-out 6 side_task_category folds, honest split 6 ways
 10 --placeholder-rule none re-runs 1-9 with placeholders included (appendix)

Probe scores for 7-8 and the pooled curves are the out-of-sample ones: train-OOF from the P2 sweep + locked-test
predictions from the full-train refit (p2 raw `predictions_chosen`). Dumb/text get the same treatment here.
Raw results/raw/<slug>/p3_baselines[_tag]_<ts>.json; figure p3_baselines[_tag].png
"""
from __future__ import annotations

import argparse, collections, glob, json, os, random, sys, time, warnings
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "8")
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LinearRegression, LogisticRegression, Ridge

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common as C
from p2_probes import C_GRID, ALPHA_GRID, GEN_POS, standardize

DUMB = ["step_index", "position_in_trajectory", "n_prompt_tokens", "n_context_tokens", "command_length", "has_sudo", "has_chmod", "has_curl", "has_nc"]
PROBES = ["label|logreg", "score|ridge_E"]
B = 500


def logreg(C_):
    return LogisticRegression(C=C_, penalty="l2", solver="lbfgs", max_iter=2000, tol=1e-4)


def ev(rows, pred, boot=True):
    """level-2 vs honest metrics on rows (list of dicts) with pred aligned; cluster-bootstrap CI."""
    pos = [(r, p) for r, p in zip(rows, pred) if r["is_pos"]]; neg = [(r, p) for r, p in zip(rows, pred) if r["is_neg"]]
    if not pos or not neg:
        return {"n_pos": len(pos), "n_neg": len(neg), "auroc": None}
    m = C.metrics([p for _, p in neg], [p for _, p in pos])
    m["n_pos_traj"] = len({r["traj_id"] for r, _ in pos}); m["n_neg_traj"] = len({r["traj_id"] for r, _ in neg})
    if boot:
        m["ci95"] = C.cluster_boot(C.by_traj([dict(traj_id=r["traj_id"], v=p) for r, p in neg], "v"), C.by_traj([dict(traj_id=r["traj_id"], v=p) for r, p in pos], "v"), B=B)
    return m


def grouped_oof(fit_predict, rows, fold, hyper_grid, target):
    """Generic pooled grouped-CV OOF with hyper chosen by pooled OOF AUROC (same convention as P2). Returns (pred, hyper, auroc)."""
    best = None
    for h in hyper_grid:
        pred = np.full(len(rows), np.nan)
        for f in sorted(set(fold)):
            tr = fold != f; ev_ = fold == f
            pred[ev_] = fit_predict(tr, ev_, h, target)
        a = ev(rows, pred, boot=False)["auroc"]
        if best is None or a > best[2]:
            best = (pred, h, a)
    return best


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--placeholder-rule", default="current"); ap.add_argument("--p2-raw", default=None); args = ap.parse_args()
    tag = "" if args.placeholder_rule == "current" else f"_ph{args.placeholder_rule}"
    t0 = time.time(); stamp = C.ts(); rng = np.random.RandomState(0)
    p2p = Path(args.p2_raw) if args.p2_raw else sorted(p for p in glob.glob(str(C.RAW_DIR / "p2_sweep_*.json")) if "_nosusp" not in p and "_ph" not in p)[-1]
    p2 = json.load(open(p2p)); chosen = {k: p2["chosen"][k] for k in PROBES}
    rows, meta = C.build_table(with_text=True); split = json.loads((C.HERE / "split.json").read_text())
    acts, idx = C.open_acts(); row_of = {(r["traj_id"], r["action_idx"]): r["row"] for r in idx["rows"]}
    has_gen = {(r["traj_id"], r["action_idx"]): r["n_positions"] == 7 and not r["nan_positions"] for r in idx["rows"]}
    u = C.usable(rows, args.placeholder_rule); test_ids = set(split["test"]); fold_of = split["fold_of"]
    tr = [r for r in u if r["traj_id"] not in test_ids]; te = [r for r in u if r["traj_id"] in test_ids]
    key = lambda r: (r["traj_id"], r["action_idx"])  # noqa: E731
    out = {"slug": C.SLUG, "script": f"experiments/{C.SLUG}/p3_baselines.py", "timestamp": stamp, "git_commit": C.git_commit(), "placeholder_rule": args.placeholder_rule,
           "inputs": meta | {"p2_raw": str(Path(p2p).relative_to(C.REPO)), "split_json_sha256": C.sha256(C.HERE / "split.json"), "acts_sha256": idx["sha256"]}, "chosen": chosen,
           "populations": {"train": {"n": len(tr), "n_pos": sum(r["is_pos"] for r in tr), "n_neg": sum(r["is_neg"] for r in tr)}, "test": {"n": len(te), "n_pos": sum(r["is_pos"] for r in te), "n_neg": sum(r["is_neg"] for r in te)}}}

    # ── probe out-of-sample scores (train-OOF + test) from P2; under a different placeholder rule the extra rows have no OOS score -> refit here
    probe_scores = {}
    for pk, ch in chosen.items():
        d = {tuple(x[:2]): x[2] for x in p2["predictions_chosen"][pk]["train_oof"]} | {tuple(x[:2]): x[2] for x in p2["predictions_chosen"][pk]["test"]}
        probe_scores[pk] = d
    def fit_probe_cell(pk, rows_tr, rows_ev, hyper=None):
        """Fit the chosen probe (arm/model/cell/hyper) on rows_tr, score rows_ev. Rows without gen positions are dropped when the cell is a gen position."""
        ch = chosen[pk]; p, l = ch["pos"], ch["layer"]; h = float(ch["hyper"]) if hyper is None else hyper
        rtr = [r for r in rows_tr if p not in GEN_POS or has_gen[key(r)]]; rev = [r for r in rows_ev if p not in GEN_POS or has_gen[key(r)]]
        Xtr = np.asarray(acts[[row_of[key(r)] for r in rtr], p, l, :], np.float32); Xev = np.asarray(acts[[row_of[key(r)] for r in rev], p, l, :], np.float32)
        if pk.startswith("label"):
            sel = np.array([r["is_pos"] or r["is_neg"] for r in rtr]); y = np.array([r["is_pos"] for r in rtr])[sel].astype(int); Xtr = Xtr[sel]
        else:
            y = np.array([r["E"] for r in rtr], np.float32)
        Xs, Xe = standardize(Xtr, Xev)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            m = logreg(h).fit(Xs, y) if pk.startswith("label") else Ridge(alpha=h).fit(Xs, y)
        pred = m.decision_function(Xe) if pk.startswith("label") else m.predict(Xe)
        return rev, pred
    if args.placeholder_rule != "current":   # rows not scored in P2 (placeholders) get an out-of-sample score by refit with the same folds / full-train
        for pk in PROBES:
            miss_tr = [r for r in tr if key(r) not in probe_scores[pk]]; miss_te = [r for r in te if key(r) not in probe_scores[pk]]
            fold = np.array([fold_of[r["traj_id"]] for r in tr])
            for f in range(5):
                rev, pred = fit_probe_cell(pk, [r for r, ff in zip(tr, fold) if ff != f], [r for r in miss_tr if fold_of[r["traj_id"]] == f])
                probe_scores[pk].update({key(r): float(v) for r, v in zip(rev, pred)})
            rev, pred = fit_probe_cell(pk, tr, miss_te); probe_scores[pk].update({key(r): float(v) for r, v in zip(rev, pred)})
    pooled = {pk: [r for r in u if key(r) in probe_scores[pk]] for pk in PROBES}
    out["probe_pooled_oos"] = {pk: {"n": len(pooled[pk]), "train_oof": ev([r for r in pooled[pk] if r["traj_id"] not in test_ids], [probe_scores[pk][key(r)] for r in pooled[pk] if r["traj_id"] not in test_ids]),
                                    "test": ev([r for r in pooled[pk] if r["traj_id"] in test_ids], [probe_scores[pk][key(r)] for r in pooled[pk] if r["traj_id"] in test_ids]),
                                    "pooled_59": ev(pooled[pk], [probe_scores[pk][key(r)] for r in pooled[pk]])} for pk in PROBES}
    out["reference_pooled_59"] = {nm: ev(u, [r[f] for r in u]) for nm, f in (("E", "E"), ("score", "score"), ("cached_mean", "cached_mean"))}
    print("pooled-59 AUROC:", {pk: round(v["pooled_59"]["auroc"], 3) for pk, v in out["probe_pooled_oos"].items()}, {k: round(v["auroc"], 3) for k, v in out["reference_pooled_59"].items()}, flush=True)

    # ── 1 random directions
    rd = {}
    for pk, ch in chosen.items():
        p, l = ch["pos"], ch["layer"]; rr = [r for r in tr if p not in GEN_POS or has_gen[key(r)]]
        X = np.asarray(acts[[row_of[key(r)] for r in rr], p, l, :], np.float32); X, _ = standardize(X, X[:1])
        W = rng.randn(200, X.shape[1]); W /= np.linalg.norm(W, axis=1, keepdims=True); S = X @ W.T
        aucs = [ev(rr, S[:, j], boot=False)["auroc"] for j in range(200)]; sym = [max(a, 1 - a) for a in aucs]
        probe_auc = ch["oof"]["auroc"]
        rd[pk] = {"cell": [C.POSITIONS[p], l], "n_rows": len(rr), "aucs": aucs, "mean": float(np.mean(aucs)), "sd": float(np.std(aucs)), "max": float(np.max(aucs)), "max_sign_free": float(np.max(sym)),
                  "probe_oof_auroc": probe_auc, "probe_percentile": float(np.mean([a < probe_auc for a in aucs]) * 100), "probe_percentile_sign_free": float(np.mean([a < probe_auc for a in sym]) * 100)}
        print(f"random dirs {pk}: mean {rd[pk]['mean']:.3f} sd {rd[pk]['sd']:.3f} max {rd[pk]['max']:.3f} (sign-free max {rd[pk]['max_sign_free']:.3f}); probe {probe_auc:.3f} pct {rd[pk]['probe_percentile']:.1f}", flush=True)
    out["1_random_directions"] = rd

    # ── 2 shuffled labels (label arm)
    pk = "label|logreg"; ch = chosen[pk]; p, l = ch["pos"], ch["layer"]
    rr = [r for r in tr if (r["is_pos"] or r["is_neg"] or r["side"] == "attack") and (p not in GEN_POS or has_gen[key(r)])]
    X = np.asarray(acts[[row_of[key(r)] for r in rr], p, l, :], np.float32); fold = np.array([fold_of[r["traj_id"]] for r in rr])
    att_traj = sorted({r["traj_id"] for r in rr if r["side"] == "attack"}); by_t = collections.defaultdict(list)
    for i, r in enumerate(rr):
        by_t[r["traj_id"]].append(i)
    relpos = {t: [rr[i]["position_in_trajectory"] for i in ix if rr[i]["is_pos"]] for t, ix in by_t.items() if t in att_traj}
    prng = random.Random(0); null = []
    for rep in range(50):
        perm = att_traj[:]; prng.shuffle(perm); ypos = np.zeros(len(rr), bool)
        for t, src in zip(att_traj, perm):
            ix = by_t[t]; pos_here = np.array([rr[i]["position_in_trajectory"] for i in ix])
            for q in relpos[src]:
                j = ix[int(np.argmin(np.abs(pos_here - q)))]; ypos[j] = True   # nearest action by relative position; may coincide -> fewer positives
        yneg = np.array([r["is_neg"] for r in rr]); pred = np.full(len(rr), np.nan)
        for f in range(5):
            trm = (fold != f) & (ypos | yneg); evm = fold == f
            Xs, Xe = standardize(X[trm], X[evm])
            with warnings.catch_warnings():
                warnings.simplefilter("ignore"); pred[evm] = logreg(float(ch["hyper"])).fit(Xs, ypos[trm].astype(int)).decision_function(Xe)
        a = C.auroc(pred[yneg], pred[ypos]); null.append({"auroc": a, "n_pos": int(ypos.sum()), "n_pos_traj": len({rr[i]["traj_id"] for i in np.where(ypos)[0]})})
    out["2_shuffled_labels"] = {"probe": pk, "cell": [C.POSITIONS[p], l], "C": ch["hyper"], "reps": null, "mean": float(np.mean([n["auroc"] for n in null])), "sd": float(np.std([n["auroc"] for n in null])),
                                "max": float(np.max([n["auroc"] for n in null])), "probe_oof_auroc": ch["oof"]["auroc"], "probe_percentile": float(np.mean([n["auroc"] < ch["oof"]["auroc"] for n in null]) * 100),
                                "rule": "positives re-assigned across train attack trajectories preserving per-trajectory count and relative position; honest unchanged"}
    print(f"shuffled labels: null mean {out['2_shuffled_labels']['mean']:.3f} sd {out['2_shuffled_labels']['sd']:.3f} max {out['2_shuffled_labels']['max']:.3f}; probe {ch['oof']['auroc']:.3f} pct {out['2_shuffled_labels']['probe_percentile']:.0f}", flush=True)

    # ── 3 dumb, 4 text, on the same folds; train on level-2 vs honest; test via full-train refit
    fold_tr = np.array([fold_of[r["traj_id"]] for r in tr]); ytr = np.array([r["is_pos"] for r in tr]); mtr = np.array([r["is_pos"] or r["is_neg"] for r in tr])
    Dtr = np.array([[float(r[c] or 0) for c in DUMB] for r in tr]); Dte = np.array([[float(r[c] or 0) for c in DUMB] for r in te])
    def fp_dumb(trm, evm, h, _):
        trm = trm & mtr; Xs, Xe = standardize(Dtr[trm], Dtr[evm])
        return logreg(h).fit(Xs, ytr[trm].astype(int)).decision_function(Xe)
    pred_d, h_d, _ = grouped_oof(fp_dumb, tr, fold_tr, [1e-2, 1e-1, 1, 10], None)
    Xs, Xe = standardize(Dtr[mtr], Dte); pred_d_te = logreg(h_d).fit(Xs, ytr[mtr].astype(int)).decision_function(Xe)
    Ttr = [r["action_text"] for r in tr]; Tte = [r["action_text"] for r in te]
    def fp_text(trm, evm, h, _):
        trm = trm & mtr; vec = TfidfVectorizer(analyzer="char", ngram_range=(3, 5), min_df=2, sublinear_tf=True, max_features=300000)
        Xt = vec.fit_transform([t for t, m in zip(Ttr, trm) if m]); return logreg(h).fit(Xt, ytr[trm].astype(int)).decision_function(vec.transform([t for t, m in zip(Ttr, evm) if m]))
    pred_t, h_t, _ = grouped_oof(fp_text, tr, fold_tr, [0.1, 1, 10, 100], None)
    vec = TfidfVectorizer(analyzer="char", ngram_range=(3, 5), min_df=2, sublinear_tf=True, max_features=300000); Xt = vec.fit_transform([t for t, m in zip(Ttr, mtr) if m])
    pred_t_te = logreg(h_t).fit(Xt, ytr[mtr].astype(int)).decision_function(vec.transform(Tte))
    base_scores = {"dumb": {key(r): float(v) for r, v in zip(tr, pred_d)} | {key(r): float(v) for r, v in zip(te, pred_d_te)},
                   "text": {key(r): float(v) for r, v in zip(tr, pred_t)} | {key(r): float(v) for r, v in zip(te, pred_t_te)}}
    out["3_dumb"] = {"features": DUMB, "C": h_d, "train_oof": ev(tr, pred_d), "test": ev(te, pred_d_te), "pooled_59": ev(u, [base_scores["dumb"][key(r)] for r in u])}
    out["4_text"] = {"features": "tfidf char 3-5 grams, min_df 2, sublinear tf, current action rendered with output hidden", "C": h_t, "train_oof": ev(tr, pred_t), "test": ev(te, pred_t_te), "pooled_59": ev(u, [base_scores["text"][key(r)] for r in u])}
    print(f"dumb: OOF {out['3_dumb']['train_oof']['auroc']:.3f} test {out['3_dumb']['test']['auroc']:.3f} pooled {out['3_dumb']['pooled_59']['auroc']:.3f} | text: OOF {out['4_text']['train_oof']['auroc']:.3f} test {out['4_text']['test']['auroc']:.3f} pooled {out['4_text']['pooled_59']['auroc']:.3f}", flush=True)

    # ── 5 dumb + probe (stacking on out-of-sample probe scores), 8 residualise
    out["5_dumb_plus_probe"] = {}; out["8_residualised"] = {}
    for pk in PROBES:
        rr = [r for r in tr if key(r) in probe_scores[pk]]; rte = [r for r in te if key(r) in probe_scores[pk]]
        f_ = np.array([fold_of[r["traj_id"]] for r in rr]); y_ = np.array([r["is_pos"] for r in rr]); m_ = np.array([r["is_pos"] or r["is_neg"] for r in rr])
        Z = np.array([[float(r[c] or 0) for c in DUMB] + [probe_scores[pk][key(r)]] for r in rr]); Zte = np.array([[float(r[c] or 0) for c in DUMB] + [probe_scores[pk][key(r)]] for r in rte])
        def fp_stack(trm, evm, h, _):
            trm = trm & m_; Xs, Xe = standardize(Z[trm], Z[evm]); return logreg(h).fit(Xs, y_[trm].astype(int)).decision_function(Xe)
        pred_s, h_s, _ = grouped_oof(fp_stack, rr, f_, [1e-2, 1e-1, 1, 10], None)
        Xs, Xe = standardize(Z[m_], Zte); pred_s_te = logreg(h_s).fit(Xs, y_[m_].astype(int)).decision_function(Xe)
        out["5_dumb_plus_probe"][pk] = {"C": h_s, "train_oof": ev(rr, pred_s), "test": ev(rte, pred_s_te), "dumb_alone_train_oof_auroc": out["3_dumb"]["train_oof"]["auroc"], "dumb_alone_test_auroc": out["3_dumb"]["test"]["auroc"]}
        # residualise on the pooled-59 population: regress probe score on dumb features (honest+level2 rows), evaluate residual
        pr = pooled[pk]; Dp = np.array([[float(r[c] or 0) for c in DUMB] for r in pr]); sp = np.array([probe_scores[pk][key(r)] for r in pr])
        Ds, _ = standardize(Dp, Dp[:1]); resid = sp - LinearRegression().fit(Ds, sp).predict(Ds)
        out["8_residualised"][pk] = {"pooled_59_residual": ev(pr, resid), "pooled_59_raw": out["probe_pooled_oos"][pk]["pooled_59"], "r2_dumb_on_probe": float(LinearRegression().fit(Ds, sp).score(Ds, sp))}
        print(f"{pk}: dumb+probe OOF {out['5_dumb_plus_probe'][pk]['train_oof']['auroc']:.3f} test {out['5_dumb_plus_probe'][pk]['test']['auroc']:.3f} | residual pooled AUROC {out['8_residualised'][pk]['pooled_59_residual']['auroc']:.3f} (raw {out['8_residualised'][pk]['pooled_59_raw']['auroc']:.3f}, R2 {out['8_residualised'][pk]['r2_dumb_on_probe']:.2f})", flush=True)

    # ── 6 LOEO and 9 LOCO over all 59 trajectories: probe (refit at chosen cell/hyper), text, dumb
    def leave_one_out(group_of, name):
        groups = sorted({group_of(r) for r in u}); res = {"groups": {}, "pooled": {}}
        pooled_pred = {k: {} for k in PROBES + ["text", "dumb"]}
        for g in groups:
            rtr = [r for r in u if group_of(r) != g]; rev = [r for r in u if group_of(r) == g]
            for pk in PROBES:
                rv, pr_ = fit_probe_cell(pk, rtr, rev); pooled_pred[pk].update({key(r): float(v) for r, v in zip(rv, pr_)})
            m2 = np.array([r["is_pos"] or r["is_neg"] for r in rtr]); y2 = np.array([r["is_pos"] for r in rtr])[m2].astype(int)
            vec = TfidfVectorizer(analyzer="char", ngram_range=(3, 5), min_df=2, sublinear_tf=True, max_features=300000)
            Xt = vec.fit_transform([r["action_text"] for r, m in zip(rtr, m2) if m]); pt = logreg(h_t).fit(Xt, y2).decision_function(vec.transform([r["action_text"] for r in rev]))
            pooled_pred["text"].update({key(r): float(v) for r, v in zip(rev, pt)})
            D2 = np.array([[float(r[c] or 0) for c in DUMB] for r in rtr]); D3 = np.array([[float(r[c] or 0) for c in DUMB] for r in rev]); Xs, Xe = standardize(D2[m2], D3)
            pd_ = logreg(h_d).fit(Xs, y2).decision_function(Xe); pooled_pred["dumb"].update({key(r): float(v) for r, v in zip(rev, pd_)})
            res["groups"][g] = {"n_pos": sum(r["is_pos"] for r in rev), "n_neg": sum(r["is_neg"] for r in rev), "n_pos_traj": len({r["traj_id"] for r in rev if r["is_pos"]}),
                                **{k: ev([r for r in rev if key(r) in pooled_pred[k]], [pooled_pred[k][key(r)] for r in rev if key(r) in pooled_pred[k]], boot=False) for k in pooled_pred}}
            res["groups"][g]["interpretable"] = res["groups"][g]["n_pos"] >= 3
        for k in pooled_pred:
            rr_ = [r for r in u if key(r) in pooled_pred[k]]; res["pooled"][k] = ev(rr_, [pooled_pred[k][key(r)] for r in rr_])
        res["pooled"]["E"] = ev(u, [r["E"] for r in u], boot=False)
        print(f"{name}: pooled AUROC " + " ".join(f"{k} {v['auroc']:.3f}" for k, v in res["pooled"].items()) + " | per-group " + " ".join(f"{g}:{v['n_pos']}pos:{'{:.2f}'.format(v[PROBES[0]]['auroc']) if v[PROBES[0]]['auroc'] is not None else '-'}" for g, v in res["groups"].items()), flush=True)
        return res
    out["6_leave_one_env_out"] = leave_one_out(lambda r: r["env"], "LOEO")
    out["9_leave_one_category_out"] = leave_one_out(lambda r: r["side_task_category"] if r["side"] == "attack" else f"honest_{hash(r['traj_id']) % 6}", "LOCO")
    # honest trajectories split 6 ways deterministically by traj id so every category fold holds out some honest
    hon_groups = {}
    hon_ids = sorted({r["traj_id"] for r in u if r["side"] == "honest"}); cats = sorted({r["side_task_category"] for r in u if r["side"] == "attack"})
    for i, t in enumerate(hon_ids): hon_groups[t] = cats[i % len(cats)]
    out["9_leave_one_category_out"] = leave_one_out(lambda r: r["side_task_category"] if r["side"] == "attack" else hon_groups[r["traj_id"]], "LOCO")

    # ── 7 matched negatives on the pooled-59 population
    hon = [r for r in u if r["is_neg"]]; l2 = [r for r in u if r["is_pos"]]
    ldec = np.quantile([r["n_action_chars"] for r in u], np.linspace(0, 1, 11)[1:-1]); pdec = np.quantile([r["position_in_trajectory"] for r in u], np.linspace(0, 1, 11)[1:-1])
    strat = lambda r: (int(np.searchsorted(ldec, r["n_action_chars"])), int(np.searchsorted(pdec, r["position_in_trajectory"])), r["tool_name"])  # noqa: E731
    pool = collections.defaultdict(list)
    for r in hon: pool[strat(r)].append(r)
    mrng = random.Random(0); matched = []; unmatched = 0; used = set()
    for r in l2:
        cands = [h for h in pool[strat(r)] if key(h) not in used]
        if not cands:   # relax tool, then position decile +-1
            cands = [h for h in hon if key(h) not in used and strat(h)[:2] == strat(r)[:2]] or [h for h in hon if key(h) not in used and strat(h)[0] == strat(r)[0] and abs(strat(h)[1] - strat(r)[1]) <= 1 and strat(h)[2] == strat(r)[2]]
        if not cands: unmatched += 1; continue
        pick = mrng.sample(cands, min(5, len(cands))); used.update(key(h) for h in pick); matched += pick
    msub = l2 + matched
    out["7_matched_negatives"] = {"rule": "per level-2 action up to 5 honest actions, without replacement, matched on action-length decile x position decile x tool (fallbacks: drop tool; position decile +-1)",
                                  "n_pos": len(l2), "n_matched_neg": len(matched), "n_pos_unmatched": unmatched, "n_matched_neg_traj": len({r["traj_id"] for r in matched}),
                                  "median_chars": {"pos": float(np.median([r["n_action_chars"] for r in l2])), "matched_neg": float(np.median([r["n_action_chars"] for r in matched])), "all_honest": float(np.median([r["n_action_chars"] for r in hon]))},
                                  "median_pos_in_traj": {"pos": float(np.median([r["position_in_trajectory"] for r in l2])), "matched_neg": float(np.median([r["position_in_trajectory"] for r in matched]))},
                                  "scorers": {**{pk: ev([r for r in msub if key(r) in probe_scores[pk]], [probe_scores[pk][key(r)] for r in msub if key(r) in probe_scores[pk]]) for pk in PROBES},
                                              **{k: ev(msub, [base_scores[k][key(r)] for r in msub]) for k in ("dumb", "text")}, **{k: ev(msub, [r[k] for r in msub]) for k in ("E", "score", "cached_mean")}}}
    print("matched negatives:", {k: round(v["auroc"], 3) for k, v in out["7_matched_negatives"]["scorers"].items()}, f"(n_neg {len(matched)}, unmatched pos {unmatched})", flush=True)
    out["scores_pooled_59"] = {"columns": ["traj_id", "action_idx", "is_pos", "is_neg", "E", "score", "cached_mean", *PROBES, "dumb", "text"],
                               "rows": [[r["traj_id"], r["action_idx"], r["is_pos"], r["is_neg"], r["E"], r["score"], r["cached_mean"], *[probe_scores[pk].get(key(r)) for pk in PROBES], base_scores["dumb"][key(r)], base_scores["text"][key(r)]] for r in u]}

    # ── figure: one bar per method, pooled-59 AUROC with cluster-bootstrap CI, plus test-only
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    bars = [("E", out["reference_pooled_59"]["E"], C.COLORS["E"]), ("integer", out["reference_pooled_59"]["score"], C.COLORS["score"]), ("cached", out["reference_pooled_59"]["cached_mean"], C.COLORS["cached_mean"]),
            ("dumb", out["3_dumb"]["pooled_59"], C.COLORS["dumb"]), ("text", out["4_text"]["pooled_59"], C.COLORS["text"]),
            ("probe label", out["probe_pooled_oos"]["label|logreg"]["pooled_59"], C.COLORS["probe_label"]), ("probe score", out["probe_pooled_oos"]["score|ridge_E"]["pooled_59"], C.COLORS["probe_score"]),
            ("probe label\nresid.", out["8_residualised"]["label|logreg"]["pooled_59_residual"], C.COLORS["probe_label"]), ("probe score\nresid.", out["8_residualised"]["score|ridge_E"]["pooled_59_residual"], C.COLORS["probe_score"]),
            ("probe label\nmatched", out["7_matched_negatives"]["scorers"]["label|logreg"], C.COLORS["probe_label"]), ("probe score\nmatched", out["7_matched_negatives"]["scorers"]["score|ridge_E"], C.COLORS["probe_score"]),
            ("E matched", out["7_matched_negatives"]["scorers"]["E"], C.COLORS["E"]), ("text matched", out["7_matched_negatives"]["scorers"]["text"], C.COLORS["text"]),
            ("probe label\nLOEO", out["6_leave_one_env_out"]["pooled"]["label|logreg"], C.COLORS["probe_label"]), ("probe score\nLOEO", out["6_leave_one_env_out"]["pooled"]["score|ridge_E"], C.COLORS["probe_score"]), ("text LOEO", out["6_leave_one_env_out"]["pooled"]["text"], C.COLORS["text"])]
    fig, ax = plt.subplots(figsize=(13, 4))
    for i, (nm, m, col) in enumerate(bars):
        ci = m.get("ci95", {}).get("auroc", [m["auroc"], m["auroc"]])
        ax.bar(i, m["auroc"], 0.7, color=col, alpha=0.9); ax.errorbar(i, m["auroc"], yerr=[[m["auroc"] - ci[0]], [ci[1] - m["auroc"]]], color="k", lw=0.8, capsize=2)
        ax.text(i, 0.31, f"{m['auroc']:.2f}", ha="center", fontsize=6.5, color="white" if m["auroc"] > 0.4 else "k")
    ax.axhline(0.5, color="grey", lw=0.6, ls=":"); ax.axhline(out["reference_pooled_59"]["E"]["auroc"], color=C.COLORS["E"], lw=0.8, ls="--")
    ax.set_xticks(range(len(bars))); ax.set_xticklabels([b[0] for b in bars], fontsize=7); ax.set_ylim(0.3, 1.0); ax.set_ylabel("AUROC, level-2 vs honest")
    ax.set_title(f"P3 baselines, pooled out-of-sample over 59 trajectories (placeholder rule: {args.placeholder_rule}); bars = 95 % trajectory-cluster bootstrap", fontsize=9)
    ax.spines[["top", "right"]].set_visible(False); fig.tight_layout(); fp = C.fig_path(f"p3_baselines{tag}.png"); fig.savefig(fp, dpi=150); plt.close(fig); out["figure"] = str(fp.relative_to(C.REPO))
    out["seconds"] = time.time() - t0
    C.write_raw(f"p3_baselines{tag}", out, stamp)


if __name__ == "__main__":
    main()
