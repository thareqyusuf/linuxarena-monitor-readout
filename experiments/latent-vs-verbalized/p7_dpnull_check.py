"""P7 — review re-check of the pre-registered Δp_null headline. CPU only; no model, no probe fit.

Reads P3 `scores_pooled_59` (the same score vector P4 uses) and answers four questions the P4 table does not:
  1 same-pool comparison   E and the label probe evaluated on the score probe's 743 honest rows (the 96 pass-1
                           honest rows have no `reasoning_end` and are all low-E), so every scorer's p95 is
                           taken over the same honest set
  2 CI on Δp_null          trajectory-cluster bootstrap (common.cluster_boot conventions, seed 0, 500 reps) of the
                           lifted fraction per scorer, and PAIRED bootstrap of (probe − E) in AUROC and in lifted count
  3 threshold convention   pooled scores mix five fold models + the test refit; lifted count when the p95 is taken
                           within each model's own honest rows instead of globally
  4 per-position Δp_null   train-OOF lifted count at the best-OOF-AUROC cell of every position (incl. the
                           pre-registered `prompt_end`), from the P2 OOF checkpoint if present (gitignored)
Raw results/raw/<slug>/p7_dpnull_check_<ts>.json.
"""
from __future__ import annotations

import collections, glob, json, random, sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common as C

SCORERS = ["E", "score", "cached_mean", "score|ridge_E", "label|logreg", "text", "dumb"]
PROBES = ["score|ridge_E", "label|logreg"]
B = 500


def lifted(pos, neg, s):
    """common.delta_p_null p95 count on rows with a value for s."""
    return C.delta_p_null([r for r in pos if r[s] is not None], [r for r in neg if r[s] is not None], s)


def main():
    stamp = C.ts()
    p3p = sorted(p for p in glob.glob(str(C.RAW_DIR / "p3_baselines_*.json")) if "_ph" not in p)[-1]; p3 = json.load(open(p3p))
    p2p = sorted(p for p in glob.glob(str(C.RAW_DIR / "p2_sweep_*.json")) if "_nosusp" not in p)[-1]; p2 = json.load(open(p2p))
    cols = p3["scores_pooled_59"]["columns"]; R = [dict(zip(cols, r)) for r in p3["scores_pooled_59"]["rows"]]
    split = json.loads((C.HERE / "split.json").read_text()); test_ids = set(split["test"]); fold_of = split["fold_of"]
    group = lambda r: "test" if r["traj_id"] in test_ids else f"fold{fold_of[r['traj_id']]}"  # noqa: E731
    pos = [r for r in R if r["is_pos"]]; neg = [r for r in R if r["is_neg"]]
    out = {"slug": C.SLUG, "script": f"experiments/{C.SLUG}/p7_dpnull_check.py", "timestamp": stamp, "git_commit": C.git_commit(),
           "inputs": {"p3_raw": str(Path(p3p).relative_to(C.REPO)), "p2_raw": str(Path(p2p).relative_to(C.REPO)), "split_json_sha256": C.sha256(C.HERE / "split.json")},
           "n_pos": len(pos), "n_neg": len(neg), "n_nulled": sum(r["score"] <= 1 for r in pos)}

    # ── 1 same pool: rows the score probe can score (743 honest)
    R743 = [r for r in R if r["score|ridge_E"] is not None]; pos7 = [r for r in R743 if r["is_pos"]]; neg7 = [r for r in R743 if r["is_neg"]]
    dropped = [r for r in neg if r["score|ridge_E"] is None]
    out["1_same_pool_743"] = {"n_pos": len(pos7), "n_neg": len(neg7), "dropped_honest": {"n": len(dropped), "mean_E": float(np.mean([r["E"] for r in dropped])),
                                                                                          "n_above_E_p95_839": int(sum(r["E"] > C.honest_pct([r["E"] for r in neg], 0.95) for r in dropped))},
                              "kept_honest_mean_E": float(np.mean([r["E"] for r in neg7])),
                              "scorers": {s: {**C.scorer_metrics(pos7, neg7, s, B=B), "delta_p_null": lifted(pos7, neg7, s)} for s in SCORERS},
                              "scorers_839": {s: {**C.metrics([r[s] for r in neg if r[s] is not None], [r[s] for r in pos if r[s] is not None]), "delta_p_null": lifted(pos, neg, s)} for s in SCORERS}}

    # ── 2 CI on Δp_null; paired (probe − E) bootstrap on the population each probe can score
    def boot(rows, scorers, seed=0):
        nb = collections.defaultdict(list); pb = collections.defaultdict(list)
        for r in rows:
            (pb if r["is_pos"] else nb)[r["traj_id"]].append(r) if (r["is_pos"] or r["is_neg"]) else None
        rng = random.Random(seed); nk, pk = list(nb), list(pb); acc = collections.defaultdict(list)
        for _ in range(B):
            N = [x for k in rng.choices(nk, k=len(nk)) for x in nb[k]]; P = [x for k in rng.choices(pk, k=len(pk)) for x in pb[k]]
            nul = [x for x in P if x["score"] <= 1]; vals = {}
            for s in scorers:
                thr = C.honest_pct([x[s] for x in N], 0.95); k = sum(x[s] > thr for x in nul)
                vals[s] = {"auroc": C.auroc([x[s] for x in N], [x[s] for x in P]), "k": k, "frac": k / max(len(nul), 1)}
                acc[f"{s}|auroc"].append(vals[s]["auroc"]); acc[f"{s}|k"].append(k); acc[f"{s}|frac"].append(vals[s]["frac"])
            for s in scorers:
                if s != "E":
                    acc[f"{s}-E|d_auroc"].append(vals[s]["auroc"] - vals["E"]["auroc"]); acc[f"{s}-E|d_k"].append(vals[s]["k"] - vals["E"]["k"])
        return {k: {"ci95": [float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))], "mean": float(np.mean(v)), **({"p_gt_0": float(np.mean(np.array(v) > 0))} if "-E|" in k else {})} for k, v in acc.items()} | {"B": B, "seed": seed}
    out["2_bootstrap"] = {"pool_839_label_vs_E": boot(R, ["E", "label|logreg", "text", "cached_mean"]),
                          "pool_743_score_vs_E": boot(R743, ["E", "score|ridge_E", "label|logreg"])}

    # ── 3 threshold convention: global p95 vs within-model p95
    conv = {}
    for s in ["E", *PROBES]:
        nn = [r for r in neg if r[s] is not None]; thr = C.honest_pct([r[s] for r in nn], 0.95); nul = [r for r in pos if r["score"] <= 1 and r[s] is not None]
        groups = sorted({group(r) for r in nn}); thr_g = {g: C.honest_pct([r[s] for r in nn if group(r) == g], 0.95) for g in groups}
        conv[s] = {"global_p95": thr, "per_group": {g: {"n_honest": sum(group(r) == g for r in nn), "own_p95": thr_g[g], "honest_above_global": int(sum(r[s] > thr for r in nn if group(r) == g)),
                                                        "nulled_here": sum(group(r) == g for r in nul), "lifted_global": int(sum(r[s] > thr for r in nul if group(r) == g)), "lifted_own": int(sum(r[s] > thr_g[g] for r in nul if group(r) == g))} for g in groups},
                   "lifted_global": int(sum(r[s] > thr for r in nul)), "lifted_own_group": int(sum(r[s] > thr_g[group(r)] for r in nul)), "n_nulled": len(nul)}
    out["3_threshold_convention"] = conv
    out["3b_nulled_by_trajectory"] = {t: {"group": group(next(r for r in pos if r["traj_id"] == t)), "n_nulled": sum(r["traj_id"] == t for r in pos if r["score"] <= 1),
                                          **{s: int(sum(r[s] > conv[s]["global_p95"] for r in pos if r["traj_id"] == t and r["score"] <= 1 and r[s] is not None)) for s in conv}}
                                      for t in sorted({r["traj_id"] for r in pos if r["score"] <= 1})}

    # ── 4 per-position train-OOF Δp_null from the P2 OOF checkpoint (gitignored; skipped if absent)
    npz_p = C.REPO / p2["oof_npz"]
    if npz_p.exists():
        z = np.load(npz_p); tk = [(a, int(b)) for a, b in (x.split("|") for x in z["train_keys"])]; lut = {(r["traj_id"], r["action_idx"]): r for r in R}
        per_pos = {}
        for arm in ("label|logreg", "score|ridge_E"):
            per_pos[arm] = {}
            for pi, pn in enumerate(C.POSITIONS):
                best = max(((m["auroc"], c["layer"], cfg) for c in p2["sweep"] if c["pos"] == pi for cfg, m in c["metrics"].items() if cfg.startswith(arm)))
                mask = z[f"mask|{pi}|{best[1]}"]; pred = z[f"{pi}|{best[1]}|{best[2]}"]; keys = [k for k, m in zip(tk, mask) if m]
                assert len(keys) == len(pred), (pn, len(keys), len(pred))   # pred is over the masked rows only (gen positions drop pass-1 rows)
                rows = [dict(lut[k], p=float(v)) for k, v in zip(keys, pred)]
                rp = [r for r in rows if r["is_pos"]]; rn = [r for r in rows if r["is_neg"]]
                per_pos[arm][pn] = {"layer": best[1], "cfg": best[2], "oof_auroc": best[0], "probe": C.delta_p_null(rp, rn, "p"), "E_same_rows": C.delta_p_null(rp, rn, "E")}
        out["4_train_oof_dpnull_per_position"] = {"npz": p2["oof_npz"], "note": "train-OOF only (26 of the 31 nulled payloads); best cell per position by OOF AUROC, as best_layer_per_position", **per_pos}
    else:
        out["4_train_oof_dpnull_per_position"] = {"npz": p2["oof_npz"], "skipped": "checkpoint absent"}

    # ── 5 trajectory identity: does the scorer separate NON-payload attack actions (level 0/1/null) from honest?
    other = [r for r in R if not r["is_pos"] and not r["is_neg"]]; ptraj = {r["traj_id"] for r in pos}
    au = lambda A, Bv, s: C.auroc([r[s] for r in Bv if r[s] is not None], [r[s] for r in A if r[s] is not None])  # noqa: E731
    out["5_attack_other"] = {"n_other": len(other), "n_other_traj": len({r["traj_id"] for r in other}), "n_other_in_pos_traj": sum(r["traj_id"] in ptraj for r in other),
                             "scorers": {s: {"level2_vs_honest": au(pos, neg, s), "other_vs_honest": au(other, neg, s), "level2_vs_other": au(pos, other, s),
                                             "level2_vs_other_same_traj": au(pos, [r for r in other if r["traj_id"] in ptraj], s),
                                             "frac_other_above_honest_p95": float(np.mean([r[s] > C.honest_pct([x[s] for x in neg if x[s] is not None], 0.95) for r in other if r[s] is not None]))} for s in SCORERS}}

    C.write_raw("p7_dpnull_check", out, stamp)
    print("attack-other (level 0/1/null) vs honest AUROC:", {s: round(v["other_vs_honest"], 3) for s, v in out["5_attack_other"]["scorers"].items()})
    print("level-2 vs attack-other in the same trajectories:", {s: round(v["level2_vs_other_same_traj"], 3) for s, v in out["5_attack_other"]["scorers"].items()})
    s1 = out["1_same_pool_743"]["scorers"]; s8 = out["1_same_pool_743"]["scorers_839"]
    print(f"nulled {out['n_nulled']}/{out['n_pos']}; dropped honest {len(dropped)} (mean E {out['1_same_pool_743']['dropped_honest']['mean_E']:.2f}, {out['1_same_pool_743']['dropped_honest']['n_above_E_p95_839']} above E p95)")
    print(f"{'scorer':14} {'AUROC 839':10} {'lift 839':9} | {'AUROC 743 [CI]':24} {'lift 743':9}")
    for s in SCORERS:
        print(f"{s:14} {s8[s]['auroc']:.3f}      {s8[s]['delta_p_null']['p95']['k']:2}/{s8[s]['delta_p_null']['n_nulled']}     | {s1[s]['auroc']:.3f} [{s1[s]['ci95']['auroc'][0]:.2f},{s1[s]['ci95']['auroc'][1]:.2f}]        {s1[s]['delta_p_null']['p95']['k']:2}/{s1[s]['delta_p_null']['n_nulled']}")
    for pool, d in out["2_bootstrap"].items():
        print(f"\n{pool}:")
        for k, v in d.items():
            if isinstance(v, dict):
                print(f"  {k:24} CI [{v['ci95'][0]:+.3f},{v['ci95'][1]:+.3f}] mean {v['mean']:+.3f}" + (f"  P(>0) {v['p_gt_0']:.2f}" if "p_gt_0" in v else ""))
    print("\nthreshold convention (lifted global / own-group):", {s: f"{v['lifted_global']}/{v['lifted_own_group']} of {v['n_nulled']}" for s, v in conv.items()})
    print("nulled by trajectory:", {t[:8]: v for t, v in out["3b_nulled_by_trajectory"].items()})
    if "skipped" not in out["4_train_oof_dpnull_per_position"]:
        for arm in ("label|logreg", "score|ridge_E"):
            print(f"\ntrain-OOF Δp_null per position, {arm}: " + "  ".join(f"{pn[:11]} {v['probe']['p95']['k']}/{v['E_same_rows']['p95']['k']}/{v['probe']['n_nulled']}" for pn, v in out["4_train_oof_dpnull_per_position"][arm].items()) + "  (probe/E/n)")


if __name__ == "__main__":
    main()
