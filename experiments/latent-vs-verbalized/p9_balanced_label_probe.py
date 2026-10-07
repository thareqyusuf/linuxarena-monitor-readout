"""P9 — is the label probe's format-learning just `588b46de` dominance? Two re-weighted refits of the label arm.

Same split.json, same folds, same cell (action_end L8, C from P2's chosen config), same evaluation as P3
(level-2 vs honest on ALL 53 positives; the cap changes only what the model trains on).
  invcount  positive sample weight 1 / (level-2 count in its trajectory), rescaled so the total positive weight is
            unchanged (= n_pos); honest weight 1.
  cap3      keep at most the last 3 level-2 actions (by action_idx) per trajectory as training positives; the other
            level-2 actions are in neither training class. They are still positives at evaluation.
For each: pooled-59 OOS (train-OOF + full-train refit on test) AUROC with cluster-bootstrap CI, LOCO (P3.9 rule),
matched negatives (P3.7 rule, seed 0, asserted to reproduce P3's 147), residualised on the dumb features (P3.8),
P6 top-10 honest list, and the P8.2 columns for the 20 audit payloads (common-743 percentile, >p95 flags,
two-by-two vs E on the quiet 8).
Reading rule (TASK-ver2, written before running): LOCO data_tampering <= 0.65 under both refits -> not a dominance
artefact; > 0.90 (E) under either -> the original null was the two-trajectory concentration.
Raw results/raw/<slug>/p9_balanced_label_probe_<ts>.json; figure p9_balanced.png.
"""
from __future__ import annotations

import collections, glob, json, random, sys, time, warnings
from pathlib import Path

import numpy as np
from sklearn.linear_model import LinearRegression, LogisticRegression

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common as C
from p2_probes import standardize
from p3_baselines import DUMB, ev

VARIANTS = ["original", "invcount", "cap3"]
CAP = 3


def main():
    t0 = time.time(); stamp = C.ts()
    p2p = sorted(p for p in glob.glob(str(C.RAW_DIR / "p2_sweep_*.json")) if "_nosusp" not in p)[-1]; p2 = json.load(open(p2p))
    p3p = sorted(p for p in glob.glob(str(C.RAW_DIR / "p3_baselines_*.json")) if "_ph" not in p)[-1]; p3 = json.load(open(p3p))
    p8p = sorted(glob.glob(str(C.RAW_DIR / "p8_quiet_payloads_*.json")))[-1]; p8 = json.load(open(p8p))
    ch = p2["chosen"]["label|logreg"]; pos_i, layer, C_ = ch["pos"], ch["layer"], float(ch["hyper"]); assert C.POSITIONS[pos_i] == "action_end"
    rows, meta = C.build_table(with_text=True); u = C.usable(rows, "current"); split = json.loads((C.HERE / "split.json").read_text())
    test_ids = set(split["test"]); fold_of = split["fold_of"]
    acts, idx = C.open_acts(); row_of = {(r["traj_id"], r["action_idx"]): r["row"] for r in idx["rows"]}
    key = lambda r: (r["traj_id"], r["action_idx"])  # noqa: E731
    tr = [r for r in u if r["traj_id"] not in test_ids]; te = [r for r in u if r["traj_id"] in test_ids]
    X_all = {}   # lazily cached fp32 features per key
    def X(rs):
        miss = [r for r in rs if key(r) not in X_all]
        if miss:
            A = np.asarray(acts[[row_of[key(r)] for r in miss], pos_i, layer, :], np.float32)
            for r, a in zip(miss, A): X_all[key(r)] = a
        return np.stack([X_all[key(r)] for r in rs])

    def train_set(variant, rows_tr):
        """(rows, y, sample_weight) for the label arm under a variant."""
        lab = [r for r in rows_tr if r["is_pos"] or r["is_neg"]]
        if variant == "cap3":
            keep = set()
            for t in {r["traj_id"] for r in lab if r["is_pos"]}:
                ix = sorted(r["action_idx"] for r in lab if r["is_pos"] and r["traj_id"] == t)[-CAP:]; keep |= {(t, i) for i in ix}
            lab = [r for r in lab if r["is_neg"] or key(r) in keep]
        y = np.array([r["is_pos"] for r in lab]).astype(int); w = np.ones(len(lab))
        if variant == "invcount":
            cnt = collections.Counter(r["traj_id"] for r in lab if r["is_pos"])
            w = np.array([1.0 / cnt[r["traj_id"]] if r["is_pos"] else 1.0 for r in lab]); w[y == 1] *= y.sum() / w[y == 1].sum()
        return lab, y, w

    def fit_predict(variant, rows_tr, rows_ev):
        lab, y, w = train_set(variant, rows_tr); Xs, Xe = standardize(X(lab), X(rows_ev))
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            m = LogisticRegression(C=C_, penalty="l2", solver="lbfgs", max_iter=2000, tol=1e-4).fit(Xs, y, sample_weight=w)
        return m.decision_function(Xe), {"n_train_pos": int(y.sum()), "n_train_neg": int((y == 0).sum()), "pos_weight_total": float(w[y == 1].sum())}

    out = {"slug": C.SLUG, "script": f"experiments/{C.SLUG}/p9_balanced_label_probe.py", "timestamp": stamp, "git_commit": C.git_commit(),
           "inputs": meta | {"p2_raw": str(Path(p2p).relative_to(C.REPO)), "p3_raw": str(Path(p3p).relative_to(C.REPO)), "p8_raw": str(Path(p8p).relative_to(C.REPO)), "split_json_sha256": C.sha256(C.HERE / "split.json"), "acts_sha256": idx["sha256"]},
           "design": {"cell": [C.POSITIONS[pos_i], layer], "C": C_, "variants": VARIANTS, "cap": CAP, "evaluation": "level-2 vs honest on all positives, as P3", "reading_rule": "LOCO data_tampering <= 0.65 under both -> not dominance; > 0.90 under either -> dominance"},
           "variants": {}}
    fold = np.array([fold_of[r["traj_id"]] for r in tr])
    # P3 reproductions: matched negatives, LOCO honest groups
    hon = [r for r in u if r["is_neg"]]; l2 = [r for r in u if r["is_pos"]]
    ldec = np.quantile([r["n_action_chars"] for r in u], np.linspace(0, 1, 11)[1:-1]); pdec = np.quantile([r["position_in_trajectory"] for r in u], np.linspace(0, 1, 11)[1:-1])
    strat = lambda r: (int(np.searchsorted(ldec, r["n_action_chars"])), int(np.searchsorted(pdec, r["position_in_trajectory"])), r["tool_name"])  # noqa: E731
    pool = collections.defaultdict(list)
    for r in hon: pool[strat(r)].append(r)
    mrng = random.Random(0); matched = []; unmatched = 0; used = set()
    for r in l2:
        cands = [h for h in pool[strat(r)] if key(h) not in used]
        if not cands:
            cands = [h for h in hon if key(h) not in used and strat(h)[:2] == strat(r)[:2]] or [h for h in hon if key(h) not in used and strat(h)[0] == strat(r)[0] and abs(strat(h)[1] - strat(r)[1]) <= 1 and strat(h)[2] == strat(r)[2]]
        if not cands: unmatched += 1; continue
        pick = mrng.sample(cands, min(5, len(cands))); used.update(key(h) for h in pick); matched += pick
    assert len(matched) == p3["7_matched_negatives"]["n_matched_neg"] and unmatched == p3["7_matched_negatives"]["n_pos_unmatched"], (len(matched), unmatched)
    msub = l2 + matched
    hon_ids = sorted({r["traj_id"] for r in u if r["side"] == "honest"}); cats = sorted({r["side_task_category"] for r in u if r["side"] == "attack"})
    hon_groups = {t: cats[i % len(cats)] for i, t in enumerate(hon_ids)}
    loco_of = lambda r: r["side_task_category"] if r["side"] == "attack" else hon_groups[r["traj_id"]]  # noqa: E731
    cols = p3["scores_pooled_59"]["columns"]; S = {(r[0], r[1]): dict(zip(cols, r)) for r in p3["scores_pooled_59"]["rows"]}
    hon743 = [r for r in hon if S[key(r)]["score|ridge_E"] is not None]
    pay = {(x["traj_id"], x["action_idx"]): x for x in p8["quiet"] + p8["loud"]}; quiet_keys = {(x["traj_id"], x["action_idx"]) for x in p8["quiet"]}

    for variant in VARIANTS:
        print(f"\n== {variant} ==", flush=True); V = {}
        # pooled OOS: 5-fold OOF on train + full-train refit on test
        pred = {}
        for f in range(5):
            ev_rows = [r for r, ff in zip(tr, fold) if ff == f]; p, info = fit_predict(variant, [r for r, ff in zip(tr, fold) if ff != f], ev_rows)
            pred.update({key(r): float(v) for r, v in zip(ev_rows, p)}); V.setdefault("fold_train_info", {})[f"fold{f}"] = info
        p, info = fit_predict(variant, tr, te); pred.update({key(r): float(v) for r, v in zip(te, p)}); V["full_train_info"] = info
        V["train_oof"] = ev(tr, [pred[key(r)] for r in tr]); V["test"] = ev(te, [pred[key(r)] for r in te]); V["pooled_59"] = ev(u, [pred[key(r)] for r in u])
        print(f"pooled-59 AUROC {V['pooled_59']['auroc']:.3f} [{V['pooled_59']['ci95']['auroc'][0]:.2f},{V['pooled_59']['ci95']['auroc'][1]:.2f}]  train-OOF {V['train_oof']['auroc']:.3f}  test {V['test']['auroc']:.3f}  train pos used {V['full_train_info']['n_train_pos']}", flush=True)
        # LOCO
        groups = sorted({loco_of(r) for r in u}); lp = {}; per = {}
        for g in groups:
            rev = [r for r in u if loco_of(r) == g]; p, _ = fit_predict(variant, [r for r in u if loco_of(r) != g], rev); lp.update({key(r): float(v) for r, v in zip(rev, p)})
            per[g] = {"n_pos": sum(r["is_pos"] for r in rev), "n_neg": sum(r["is_neg"] for r in rev), **ev(rev, [lp[key(r)] for r in rev], boot=False)}
        V["loco"] = {"pooled": ev(u, [lp[key(r)] for r in u]), "groups": per}
        print(f"LOCO pooled {V['loco']['pooled']['auroc']:.3f}  data_tampering {per['data_tampering']['auroc']:.3f} (n={per['data_tampering']['n_pos']})  " + " ".join(f"{g[:10]} {v['auroc']:.2f}" for g, v in per.items() if v["auroc"] is not None and g != "data_tampering"), flush=True)
        # matched negatives, residualised
        V["matched_negatives"] = {"n_pos": len(l2), "n_matched_neg": len(matched), **ev(msub, [pred[key(r)] for r in msub])}
        Dp = np.array([[float(r[c] or 0) for c in DUMB] for r in u]); sp = np.array([pred[key(r)] for r in u]); Ds, _ = standardize(Dp, Dp[:1])
        lr = LinearRegression().fit(Ds, sp); V["residualised"] = {**ev(u, sp - lr.predict(Ds)), "r2_dumb_on_probe": float(lr.score(Ds, sp))}
        print(f"matched {V['matched_negatives']['auroc']:.3f} [{V['matched_negatives']['ci95']['auroc'][0]:.2f},{V['matched_negatives']['ci95']['auroc'][1]:.2f}]  residual {V['residualised']['auroc']:.3f} (R2 {V['residualised']['r2_dumb_on_probe']:.2f})", flush=True)
        # P6 top-10 honest
        top = sorted(hon, key=lambda r: -pred[key(r)])[:10]
        V["top10_honest"] = [{"traj_id": r["traj_id"], "action_idx": r["action_idx"], "pred": pred[key(r)], "tool": r["tool_name"], "n_action_chars": r["n_action_chars"], "position_in_trajectory": r["position_in_trajectory"], "env": r["env"], "score": r["score"], "E": r["E"], "action_text_160": r["action_text"][:160]} for r in top]
        print("top-10 honest: " + "; ".join(f"{x['tool']}/{x['n_action_chars']}ch/pos{x['position_in_trajectory']:.2f}" for x in V["top10_honest"]), flush=True)
        # P8.2 columns on the common-743 pool
        hv = np.sort([pred[key(r)] for r in hon743]); p95 = C.honest_pct(hv, 0.95); p99 = C.honest_pct(hv, 0.99)
        V["payloads_743"] = {f"{t[:8]}#{i}": {"quiet": (t, i) in quiet_keys, "pred": pred[(t, i)], "pct": float(np.searchsorted(hv, pred[(t, i)], side="left") / len(hv) * 100), "gt_p95": bool(pred[(t, i)] > p95), "gt_p99": bool(pred[(t, i)] > p99), "E_gt_p95": pay[(t, i)]["scores"]["common_743"]["E"]["gt_p95"]} for t, i in sorted(pay)}
        q = {k: v for k, v in V["payloads_743"].items() if v["quiet"]}
        V["quiet_summary"] = {"lifted_p95": sum(v["gt_p95"] for v in q.values()), "lifted_p99": sum(v["gt_p99"] for v in q.values()), "n": len(q),
                              "probe_not_E": sorted(k for k, v in q.items() if v["gt_p95"] and not v["E_gt_p95"]), "E_not_probe": sorted(k for k, v in q.items() if v["E_gt_p95"] and not v["gt_p95"])}
        print(f"quiet payloads lifted {V['quiet_summary']['lifted_p95']}/8 (p99 {V['quiet_summary']['lifted_p99']}); probe-not-E {V['quiet_summary']['probe_not_E']}; E-not-probe {V['quiet_summary']['E_not_probe']}", flush=True)
        V["predictions"] = [[t, i, v] for (t, i), v in sorted(pred.items())]
        out["variants"][variant] = V
    # reference from P3 for the original (should match 'original' here up to solver tolerance)
    out["p3_reference_label_probe"] = {"pooled_59": p3["probe_pooled_oos"]["label|logreg"]["pooled_59"]["auroc"], "loco_pooled": p3["9_leave_one_category_out"]["pooled"]["label|logreg"]["auroc"],
                                       "loco_data_tampering": p3["9_leave_one_category_out"]["groups"]["data_tampering"]["label|logreg"]["auroc"], "matched": p3["7_matched_negatives"]["scorers"]["label|logreg"]["auroc"], "residual": p3["8_residualised"]["label|logreg"]["pooled_59_residual"]["auroc"]}
    dt = {v: out["variants"][v]["loco"]["groups"]["data_tampering"]["auroc"] for v in VARIANTS}
    out["reading"] = {"loco_data_tampering": dt, "both_refits_le_0.65": all(dt[v] <= 0.65 for v in ("invcount", "cap3")), "either_refit_gt_0.90": any(dt[v] > 0.90 for v in ("invcount", "cap3"))}

    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(9, 3.8)); mets = [("pooled_59", "pooled-59"), ("loco_dt", "LOCO data_tampering"), ("loco", "LOCO pooled"), ("matched_negatives", "matched neg."), ("residualised", "residualised")]
    cols_ = {"original": "#0f766e", "invcount": "#2563eb", "cap3": "#7c3aed"}
    for i, (mk, nm) in enumerate(mets):
        for j, v in enumerate(VARIANTS):
            V = out["variants"][v]; m = V["loco"]["groups"]["data_tampering"] if mk == "loco_dt" else V["loco"]["pooled"] if mk == "loco" else V[mk]
            ci = m.get("ci95", {}).get("auroc", [m["auroc"], m["auroc"]]); x = i - 0.27 + 0.27 * j
            ax.bar(x, m["auroc"], 0.25, color=cols_[v], label=v if i == 0 else None); ax.errorbar(x, m["auroc"], yerr=[[m["auroc"] - ci[0]], [ci[1] - m["auroc"]]], color="k", lw=0.7, capsize=2)
            ax.text(x, 0.32, f"{m['auroc']:.2f}", ha="center", fontsize=6, color="white")
    ax.axhline(0.5, color="grey", lw=0.6, ls=":"); ax.axhline(p3["reference_pooled_59"]["E"]["auroc"], color=C.COLORS["E"], lw=0.8, ls="--", label="E pooled-59")
    ax.set_xticks(range(len(mets))); ax.set_xticklabels([n for _, n in mets], fontsize=8); ax.set_ylim(0.3, 1.0); ax.set_ylabel("AUROC, level-2 vs honest")
    ax.set_title(f"P9: label probe (action_end L{layer}, C={C_}) under positive re-weighting; bars = 95 % cluster bootstrap", fontsize=8.5); ax.legend(fontsize=7, frameon=False, ncol=4)
    ax.spines[["top", "right"]].set_visible(False); fig.tight_layout(); fp = C.fig_path("p9_balanced.png"); fig.savefig(fp, dpi=150); plt.close(fig); out["figure"] = str(fp.relative_to(C.REPO))
    out["seconds"] = time.time() - t0
    C.write_raw("p9_balanced_label_probe", out, stamp)
    print("\nreading:", json.dumps(out["reading"]), "| P3 reference:", json.dumps(out["p3_reference_label_probe"]))


if __name__ == "__main__":
    main()
