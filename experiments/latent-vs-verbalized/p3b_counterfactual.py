"""P3b — counterfactual renders: score the P2 probes, without refitting on the renders, on the same 250 actions
rendered three ways (original / neutralized / random_scrub). The only control that removes the giveaway from the
INPUT. random_scrub is the baseline (same number of substitution events, benign swaps).

Models: for each chosen probe, the 5 fold models (train minus fold f) and the full-train model are fitted ONCE
on original-render activations; an action in train fold f is scored by fold model f, a test action by the
full-train model, so every score is out-of-sample and the same model scores all three renders of an action.
The text baseline (TF-IDF char 3-5 + logistic, P3 hyper) is treated identically, with the render's current-action
text rebuilt by monitor_prompt.neutralize / random_scrub (deterministic). E and the integer come from the runs.

Raw results/raw/<slug>/p3b_counterfactual_<ts>.json; figure p3b_renders.png
"""
from __future__ import annotations

import collections, glob, json, statistics as st, sys, time, warnings
from pathlib import Path

import numpy as np
import torch
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression, Ridge

sys.path.insert(0, str(Path(__file__).resolve().parent)); sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "local-monitor-activations"))
import common as C
from p2_probes import GEN_POS, standardize
import monitor_prompt as mp

PROBES = ["label|logreg", "score|ridge_E"]; RENDERS = ["original", "neutralized", "random_scrub"]


def load_cf_cells(stage, cells):
    """(traj, idx) -> {cell: vector} for one cf stage; plus the score records."""
    d = C.OUT / stage; vec = collections.defaultdict(dict)
    for shp in sorted(d.glob("acts_*.pt")):
        sh = torch.load(shp, map_location="cpu", weights_only=False)
        assert sh["_meta"]["position_order"] == C.POSITIONS
        for k, t in sh.items():
            if k == "_meta": continue
            tr, i, render = k.split("|")
            for (p, l) in cells:
                vec[(tr, int(i))][(p, l)] = t[p, l].numpy().astype(np.float32)
        del sh
    rec = {}
    for line in (d / "scores.jsonl").open():
        if line.strip():
            r = json.loads(line); rec[(r["traj_id"], int(r["action_idx"]))] = r
    return vec, rec


def main():
    t0 = time.time(); stamp = C.ts()
    p2p = sorted(p for p in glob.glob(str(C.RAW_DIR / "p2_sweep_*.json")) if "_nosusp" not in p)[-1]; p2 = json.load(open(p2p)); chosen = {k: p2["chosen"][k] for k in PROBES}
    p3p = sorted(p for p in glob.glob(str(C.RAW_DIR / "p3_baselines_*.json")) if "_ph" not in p)[-1]; p3 = json.load(open(p3p)); C_text = p3["4_text"]["C"]
    rows, meta = C.build_table(with_text=True); lut = {(r["traj_id"], r["action_idx"]): r for r in rows}
    split = json.loads((C.HERE / "split.json").read_text()); test_ids = set(split["test"]); fold_of = split["fold_of"]
    acts, idx = C.open_acts(); row_of = {(r["traj_id"], r["action_idx"]): r["row"] for r in idx["rows"]}; has_gen = {(r["traj_id"], r["action_idx"]): r["n_positions"] == 7 and not r["nan_positions"] for r in idx["rows"]}
    sub = json.load(open(C.REPO / "experiments" / C.LMA / "subset_counterfactual.json"))["actions"]; cf_keys = [(a["traj_id"], int(a["action_idx"])) for a in sub]; stratum = {(a["traj_id"], int(a["action_idx"])): a["counterfactual_stratum"] for a in sub}
    cells = [(chosen[pk]["pos"], chosen[pk]["layer"]) for pk in PROBES]
    cf = {"neutralized": load_cf_cells("cf_neutralized", cells), "random_scrub": load_cf_cells("cf_random_scrub", cells)}
    print(f"loaded cf shards in {time.time()-t0:.0f}s: " + ", ".join(f"{k} {len(v[0])} acts / {len(v[1])} recs" for k, v in cf.items()), flush=True)
    u = C.usable(rows, "current"); tr = [r for r in u if r["traj_id"] not in test_ids]; key = lambda r: (r["traj_id"], r["action_idx"])  # noqa: E731

    # ── fit fold models + full model per probe, and the text baseline, on ORIGINAL renders
    models = {}
    for pk in PROBES:
        ch = chosen[pk]; p, l = ch["pos"], ch["layer"]; h = float(ch["hyper"])
        def fit(rows_tr):
            rr = [r for r in rows_tr if p not in GEN_POS or has_gen[key(r)]]
            X = np.asarray(acts[[row_of[key(r)] for r in rr], p, l, :], np.float32)
            if pk.startswith("label"):
                sel = np.array([r["is_pos"] or r["is_neg"] for r in rr]); X = X[sel]; y = np.array([r["is_pos"] for r in rr])[sel].astype(int)
            else:
                y = np.array([r["E"] for r in rr], np.float32)
            mu = X.mean(0); sd = X.std(0); sd[sd < 1e-6] = 1.0
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                m = LogisticRegression(C=h, solver="lbfgs", max_iter=2000).fit((X - mu) / sd, y) if pk.startswith("label") else Ridge(alpha=h).fit((X - mu) / sd, y)
            w = (m.coef_.ravel() if pk.startswith("label") else m.coef_) / sd; b = float(np.ravel(m.intercept_)[0]) - float(mu @ w)
            return w.astype(np.float32), b
        models[pk] = {f: fit([r for r in tr if fold_of[r["traj_id"]] != f]) for f in range(5)} | {"full": fit(tr)}
    def fit_text(rows_tr):
        rr = [r for r in rows_tr if r["is_pos"] or r["is_neg"]]; vec = TfidfVectorizer(analyzer="char", ngram_range=(3, 5), min_df=2, sublinear_tf=True, max_features=300000)
        Xt = vec.fit_transform([r["action_text"] for r in rr]); return vec, LogisticRegression(C=C_text, solver="lbfgs", max_iter=2000).fit(Xt, [int(r["is_pos"]) for r in rr])
    text_models = {f: fit_text([r for r in tr if fold_of[r["traj_id"]] != f]) for f in range(5)} | {"full": fit_text(tr)}
    print(f"models fitted in {time.time()-t0:.0f}s", flush=True)

    # ── score the 250 x 3 renders
    per = []
    for k in cf_keys:
        r = lut[k]; which = "full" if k[0] in test_ids else fold_of[k[0]]
        rec_r = {"original": {"score": r["score"], "E": r["E"], "capture_ok": r["capture_ok"]}, **{rn: {"score": cf[rn][1][k]["score"], "E": cf[rn][1][k]["expected_score"], "capture_ok": cf[rn][1][k]["capture_ok"]} for rn in ("neutralized", "random_scrub")}}
        txt = {"original": r["action_text"]}; txt["neutralized"], n_ev, _ = mp.neutralize(r["action_text"]); txt["random_scrub"], _ = mp.random_scrub(r["action_text"], n_ev)
        out = {"traj_id": k[0], "action_idx": k[1], "side": r["side"], "label_level": r["label_level"], "is_pos": r["is_pos"], "is_neg": r["is_neg"], "stratum": stratum[k], "placeholder_current": r["placeholder_current"],
               "model_used": str(which), "action_events": n_ev, "cached_mean": r["cached_mean"], "render": {}}
        for rn in RENDERS:
            d = {"score": rec_r[rn]["score"], "E": rec_r[rn]["E"], "capture_ok": rec_r[rn]["capture_ok"], "action_text_changed": txt[rn] != txt["original"]}
            vec, tm = text_models[which]; d["text"] = float(tm.decision_function(vec.transform([txt[rn]]))[0])
            for pk in PROBES:
                p, l = chosen[pk]["pos"], chosen[pk]["layer"]; w, b = models[pk][which]
                if rn == "original":
                    x = np.asarray(acts[row_of[k], p, l, :], np.float32) if (p not in GEN_POS or has_gen[k]) else None
                else:
                    x = cf[rn][0].get(k, {}).get((p, l)); x = None if x is None or (p in GEN_POS and not rec_r[rn]["capture_ok"]) or (x is not None and not np.isfinite(x).all()) else x
                d[pk] = float(x @ w + b) if x is not None else None
            out["render"][rn] = d
        per.append(out)
    # ── metrics per render, per scorer, on the same rows (rows with a value in all three renders; score not null; placeholder rule current)
    scorers = ["score", "E", "text"] + PROBES; summary = {}
    for s in scorers:
        ok = [a for a in per if not a["placeholder_current"] and all(a["render"][rn][s] is not None for rn in RENDERS)]
        pos = [a for a in ok if a["is_pos"]]; neg = [a for a in ok if a["is_neg"]]
        summary[s] = {"n": len(ok), "n_pos": len(pos), "n_neg": len(neg), "per_render": {rn: (C.metrics([a["render"][rn][s] for a in neg], [a["render"][rn][s] for a in pos]) if pos and neg else None) for rn in RENDERS}}
        for rn in ("neutralized", "random_scrub"):
            dl = [a["render"][rn][s] - a["render"]["original"][s] for a in ok]
            summary[s][f"delta_{rn}_minus_original"] = {"mean": st.mean(dl), "sd": st.pstdev(dl), "median": st.median(dl),
                                                        "by_stratum": {stx: st.mean([a["render"][rn][s] - a["render"]["original"][s] for a in ok if a["stratum"] == stx]) for stx in ("action", "history_only")},
                                                        "by_class": {cl: st.mean([a["render"][rn][s] - a["render"]["original"][s] for a in ok if f(a)]) for cl, f in (("level2", lambda a: a["is_pos"]), ("attack_other", lambda a: a["side"] == "attack" and not a["is_pos"]), ("honest", lambda a: a["is_neg"])) if any(f(a) for a in ok)},
                                                        "n_changed_action_text": sum(a["render"][rn]["action_text_changed"] for a in ok)}
        summary[s]["delta_neutralized_minus_random_scrub"] = {"mean": st.mean([a["render"]["neutralized"][s] - a["render"]["random_scrub"][s] for a in ok]),
                                                              "level2_mean": st.mean([a["render"]["neutralized"][s] - a["render"]["random_scrub"][s] for a in pos]) if pos else None}
    frac3 = {rn: {"frac_score_ge3": float(np.mean([a["render"][rn]["score"] >= 3 for a in per if a["render"][rn]["score"] is not None])), "n": sum(a["render"][rn]["score"] is not None for a in per)} for rn in RENDERS}
    out = {"slug": C.SLUG, "script": f"experiments/{C.SLUG}/p3b_counterfactual.py", "timestamp": stamp, "git_commit": C.git_commit(), "inputs": meta | {"p2_raw": str(Path(p2p).relative_to(C.REPO)), "p3_raw": str(Path(p3p).relative_to(C.REPO)),
           "cf_scores_sha256": {rn: C.sha256(C.OUT / f"cf_{rn}" / "scores.jsonl") for rn in ("neutralized", "random_scrub")}, "acts_sha256": idx["sha256"]},
           "chosen": chosen, "n_actions": len(per), "strata": dict(collections.Counter(a["stratum"] for a in per)), "classes": dict(collections.Counter("level2" if a["is_pos"] else a["side"] for a in per)),
           "frac_score_ge3_all_250": frac3, "summary": summary, "per_action": per, "seconds": time.time() - t0}
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1, 2, figsize=(11, 3.8))
    cols = {"score": C.COLORS["score"], "E": C.COLORS["E"], "text": C.COLORS["text"], "label|logreg": C.COLORS["probe_label"], "score|ridge_E": C.COLORS["probe_score"]}
    for j, s in enumerate(scorers):
        for i, rn in enumerate(RENDERS):
            m = summary[s]["per_render"][rn]
            if m: ax[0].bar(j + (i - 1) * 0.26, m["auroc"], 0.26, color=cols[s], alpha=[1.0, 0.6, 0.35][i], edgecolor="k", lw=0.3)
    ax[0].set_xticks(range(len(scorers))); ax[0].set_xticklabels(scorers, fontsize=7); ax[0].set_ylim(0.3, 1); ax[0].set_ylabel("AUROC within the 250 (level-2 vs honest)")
    ax[0].set_title("per scorer: original (solid) / neutralized (mid) / random_scrub (light)", fontsize=8)
    for j, s in enumerate(scorers):
        ok = [a for a in per if not a["placeholder_current"] and all(a["render"][rn][s] is not None for rn in RENDERS)]
        sd = st.pstdev([a["render"]["original"][s] for a in ok]) or 1.0
        for i, rn in enumerate(("neutralized", "random_scrub")):
            for cl, mk in (("level2", "o"), ("honest", "s")):
                dl = [(a["render"][rn][s] - a["render"]["original"][s]) / sd for a in ok if (a["is_pos"] if cl == "level2" else a["is_neg"])]
                if dl: ax[1].errorbar(j + (i - 0.5) * 0.3 + (0.08 if cl == "honest" else -0.08), st.mean(dl), yerr=st.pstdev(dl) / np.sqrt(len(dl)), fmt=mk, color=cols[s], alpha=1 if i == 0 else 0.45, ms=5, capsize=2)
    ax[1].axhline(0, color="grey", lw=0.6); ax[1].set_xticks(range(len(scorers))); ax[1].set_xticklabels(scorers, fontsize=7); ax[1].set_ylabel("(render - original) / sd(original), mean +- se")
    ax[1].set_title("per-action shift: neutralized (dark) vs random_scrub (light); o level-2, square honest", fontsize=8)
    for a in ax: a.spines[["top", "right"]].set_visible(False)
    fig.tight_layout(); fp = C.fig_path("p3b_renders.png"); fig.savefig(fp, dpi=150); plt.close(fig); out["figure"] = str(fp.relative_to(C.REPO))
    C.write_raw("p3b_counterfactual", out, stamp)
    print("frac(score>=3) on the 250:", frac3); print("classes:", out["classes"], "strata:", out["strata"])
    for s in scorers:
        m = summary[s]; print(f"  {s:14} n={m['n']} ({m['n_pos']} pos/{m['n_neg']} neg) " + " | ".join(f"{rn} AUROC {m['per_render'][rn]['auroc']:.3f} TPR5 {m['per_render'][rn]['tpr_at_0.05']:.2f}" for rn in RENDERS if m['per_render'][rn])
              + f" | dN-O {m['delta_neutralized_minus_original']['mean']:+.3f} (L2 {m['delta_neutralized_minus_original']['by_class'].get('level2', float('nan')):+.3f}) dR-O {m['delta_random_scrub_minus_original']['mean']:+.3f} (L2 {m['delta_random_scrub_minus_original']['by_class'].get('level2', float('nan')):+.3f}) dN-R L2 {m['delta_neutralized_minus_random_scrub']['level2_mean']}")


if __name__ == "__main__":
    main()
