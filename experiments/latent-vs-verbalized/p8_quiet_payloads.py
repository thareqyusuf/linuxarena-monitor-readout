"""P8 — the quiet payloads: existence test on the 8 audit-verdict payloads the signature label carries at level 0.

CPU only. Scores come from P3 `scores_pooled_59` (the P4 vector: integer, E, cached_mean, dumb, text, label probe
action_end L8, score probe reasoning_end L14) — the same path P7 §5 used for the non-payload attack rows. The
pre-registered `prompt_end` label cell comes from the P2 OOF checkpoint for train-trajectory rows; test-trajectory
rows have no OOF prediction there, so ONE full-train refit at that cell (same recipe as P2's locked test: level-2 vs
honest train rows, standardised, logistic C from the sweep) scores them. That is the only fit in this script and it
is flagged in the raw (`prompt_end_test_refit`).

Honest pools: common 743 (rows with generation positions) for every scorer — primary; 839 appendix. Percentile =
share of the honest pool strictly below the value; p95/p99 flags use common.honest_pct (strict >, as delta_p_null).
Own-model thresholds (P7 §3): the p95 of the honest rows scored by the same model (fold k / test refit).

Raw results/raw/<slug>/p8_quiet_payloads_<ts>.json; figure results/figures/<slug>/p8_quiet_payloads.png.
"""
from __future__ import annotations

import collections, glob, json, subprocess, sys, warnings
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common as C
from p2_probes import standardize

SCORERS = ["score", "E", "cached_mean", "text", "dumb", "label|logreg", "label|logreg@prompt_end", "score|ridge_E"]
MODEL_SCORED = {"text", "dumb", "label|logreg", "label|logreg@prompt_end", "score|ridge_E"}   # scale differs per fold model
A4_DIR = "results/raw/action-level-attack-label"
A4_FILES = {"a4_audit": "a4_audit_20260911T085753Z.json", "a4b_audit": "a4b_audit_20260911T103200Z.json",
            "a4_draw": "a4_draw_20260911T074044Z.json", "a4b_draw": "a4b_draw_20260911T090845Z.json"}


def label_branch_raw(name: str) -> dict:
    cache = C.CKPT / f"labelbranch_{name}"
    if not cache.exists():
        cache.write_text(subprocess.run(["git", "show", f"{C.LABELS_BRANCH}:{A4_DIR}/{name}"], cwd=C.REPO, capture_output=True, text=True, check=True).stdout)
    return json.loads(cache.read_text())


def main():
    stamp = C.ts()
    rows, meta = C.build_table(with_text=True); u = C.usable(rows, "current")
    split = json.loads((C.HERE / "split.json").read_text()); test_ids = set(split["test"]); fold_of = split["fold_of"]; folds = {int(k): set(v) for k, v in split["folds"].items()}
    p3p = sorted(p for p in glob.glob(str(C.RAW_DIR / "p3_baselines_*.json")) if "_ph" not in p)[-1]; p3 = json.load(open(p3p))
    p2p = sorted(p for p in glob.glob(str(C.RAW_DIR / "p2_sweep_*.json")) if "_nosusp" not in p)[-1]; p2 = json.load(open(p2p))
    cols = p3["scores_pooled_59"]["columns"]; S = {(r[0], r[1]): dict(zip(cols, r)) for r in p3["scores_pooled_59"]["rows"]}
    key = lambda r: (r["traj_id"], r["action_idx"])  # noqa: E731
    group = lambda r: "test" if r["traj_id"] in test_ids else f"fold{fold_of[r['traj_id']]}"  # noqa: E731

    # ── audit inputs: stratum + objective per audited action (never re-open the audit files)
    a4 = {k: label_branch_raw(v) for k, v in A4_FILES.items()}
    stratum = {(r["traj_id"], int(r["action_idx"])): "random_a4" for r in a4["a4_audit"]["rows"]}
    stratum.update({(r["traj_id"], int(r["action_idx"])): r["stratum"] for r in a4["a4b_audit"]["rows"]})
    objective = {(r["traj_id"], int(r["action_idx"])): r["objective"] for d in ("a4_draw", "a4b_draw") for r in a4[d]["draw"]}
    audit_meta = {k: {"sha256": None} for k in A4_FILES}   # content pinned by branch+filename; sha recorded below
    for k, v in A4_FILES.items():
        audit_meta[k] = {"path": f"{A4_DIR}/{v}", "branch": C.LABELS_BRANCH, "sha256": C.sha256(C.CKPT / f"labelbranch_{v}")}

    # ── P8.1 assemble
    pay = [r for r in rows if r["audit_verdict"] == "payload"]; prep = [r for r in rows if r["audit_verdict"] == "prep"]
    dropped = [key(r) for r in pay + prep if r["score"] is None or r["placeholder_current"]]
    pay = [r for r in pay if key(r) not in dropped]; prep = [r for r in prep if key(r) not in dropped]
    quiet = [r for r in pay if r["label_level"] == 0]; loud = [r for r in pay if r["label_level"] == 2]
    assert len(pay) == 20 and len(quiet) == 8 and len(loud) == 12, (len(pay), len(quiet), len(loud))
    print(f"payload {len(pay)} in {len({r['traj_id'] for r in pay})} traj: quiet {len(quiet)} / loud {len(loud)}; prep {len(prep)}; dropped {dropped}")

    # ── prompt_end label cell: train OOF from the checkpoint, test rows by one full-train refit
    pe = C.POSITIONS.index("prompt_end")
    best = max(((m["auroc"], c["layer"], cfg) for c in p2["sweep"] if c["pos"] == pe for cfg, m in c["metrics"].items() if cfg.startswith("label|logreg")))
    pe_layer, pe_cfg = best[1], best[2]; pe_C = float(pe_cfg.split("|")[-1])
    z = np.load(C.REPO / p2["oof_npz"]); tk = [(a, int(b)) for a, b in (x.split("|") for x in z["train_keys"])]
    mask = z[f"mask|{pe}|{pe_layer}"]; pred = z[f"{pe}|{pe_layer}|{pe_cfg}"]; keys_m = [k for k, m in zip(tk, mask) if m]; assert len(keys_m) == len(pred)
    pe_score = {k: float(v) for k, v in zip(keys_m, pred)}
    acts, idx = C.open_acts(); row_of = {(r["traj_id"], r["action_idx"]): r["row"] for r in idx["rows"]}
    tr = [r for r in u if r["traj_id"] not in test_ids and (r["is_pos"] or r["is_neg"])]; te = [r for r in u if r["traj_id"] in test_ids]
    Xtr = np.asarray(acts[[row_of[key(r)] for r in tr], pe, pe_layer, :], np.float32); Xte = np.asarray(acts[[row_of[key(r)] for r in te], pe, pe_layer, :], np.float32)
    Xs, Xe = standardize(Xtr, Xte)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        m = LogisticRegression(C=pe_C, penalty="l2", solver="lbfgs", max_iter=1000, tol=1e-4).fit(Xs, np.array([r["is_pos"] for r in tr]).astype(int))
    pe_score.update({key(r): float(v) for r, v in zip(te, m.decision_function(Xe))})
    for r in u:
        S[key(r)]["label|logreg@prompt_end"] = pe_score.get(key(r))
    te_pos = [r for r in te if r["is_pos"]]; te_neg = [r for r in te if r["is_neg"]]
    pe_test_metrics = C.metrics([pe_score[key(r)] for r in te_neg], [pe_score[key(r)] for r in te_pos])
    print(f"prompt_end cell L{pe_layer} C={pe_C}: train-OOF AUROC {best[0]:.3f}; test refit AUROC {pe_test_metrics['auroc']:.3f} (n={len(te)})")

    # ── blindness check
    blind = []
    for r in pay:
        g = group(r); train_traj = set(split["train"]) if g == "test" else set(split["train"]) - folds[int(g[-1])]
        blind.append({"key": key(r), "scoring_model": "test_refit" if g == "test" else g, "traj_in_training": r["traj_id"] in train_traj, "in_label_class": bool(r["is_pos"] or r["is_neg"])})
    assert not any(b["traj_in_training"] for b in blind), blind
    print("blindness: all 20 scored by a model whose training trajectories exclude their own:", all(not b["traj_in_training"] for b in blind))

    # ── honest pools and thresholds
    hon = [r for r in u if r["is_neg"]]; hon743 = [r for r in hon if S[key(r)]["score|ridge_E"] is not None]
    pools = {"common_743": hon743, "all_839": hon}
    thr = {}
    for pn, pool in pools.items():
        thr[pn] = {}
        for s in SCORERS:
            v = np.array([S[key(r)][s] for r in pool if S[key(r)][s] is not None], float)
            own = {}
            for g in sorted({group(r) for r in pool}):
                vg = np.array([S[key(r)][s] for r in pool if group(r) == g and S[key(r)][s] is not None], float)
                own[g] = {"n": int(len(vg)), "p95": C.honest_pct(vg, 0.95), "p99": C.honest_pct(vg, 0.99)}
            thr[pn][s] = {"n": int(len(v)), "p95": C.honest_pct(v, 0.95), "p99": C.honest_pct(v, 0.99), "sorted": np.sort(v), "own": own}

    def score_row(r, pn):
        out = {}
        for s in SCORERS:
            x = S[key(r)][s]; t = thr[pn][s]
            if x is None:
                out[s] = None; continue
            g = group(r); o = t["own"].get(g)
            out[s] = {"value": float(x), "pct": float(np.searchsorted(t["sorted"], x, side="left") / len(t["sorted"]) * 100), "gt_p95": bool(x > t["p95"]), "gt_p99": bool(x > t["p99"]),
                      "own_model_gt_p95": bool(x > o["p95"]) if (s in MODEL_SCORED and o) else None, "own_model_p95": o["p95"] if (s in MODEL_SCORED and o) else None}
        return out

    def rec(r):
        return {"traj_id": r["traj_id"], "action_idx": r["action_idx"], "group": group(r), "scoring_model": "test_refit" if group(r) == "test" else group(r),
                "stratum": stratum.get(key(r)), "side_task": r["side_task"], "category": r["side_task_category"], "objective": objective.get(key(r)),
                "step_index": r["step_index"], "trajectory_length": r["trajectory_length"], "position_in_trajectory": r["position_in_trajectory"], "tool": r["tool_name"],
                "command_length": r["command_length"], "n_action_chars": r["n_action_chars"], "label_level": r["label_level"], "audit_verdict": r["audit_verdict"],
                "score": r["score"], "E": r["E"], "cached_scores": r["cached_scores"], "cached_mean": r["cached_mean"], "label_cached_high": r["label_cached_high"],
                "scores": {pn: score_row(r, pn) for pn in pools}, "action_text_300": r["action_text"][:300], "analysis_note_text": r["analysis_note_text"]}

    Q = [rec(r) for r in sorted(quiet, key=lambda r: (r["traj_id"], r["action_idx"]))]; L = [rec(r) for r in sorted(loud, key=lambda r: (r["traj_id"], r["action_idx"]))]
    P = [rec(r) for r in sorted(prep, key=lambda r: (r["traj_id"], r["action_idx"]))]

    def summary(recs, pn):
        out = {}
        for s in SCORERS:
            have = [x for x in recs if x["scores"][pn][s] is not None]
            out[s] = {"n": len(have), "gt_p95": sum(x["scores"][pn][s]["gt_p95"] for x in have), "gt_p99": sum(x["scores"][pn][s]["gt_p99"] for x in have),
                      "own_model_gt_p95": sum(bool(x["scores"][pn][s]["own_model_gt_p95"]) for x in have) if s in MODEL_SCORED else None,
                      "by_stratum_gt_p95": dict(collections.Counter(x["stratum"] for x in have if x["scores"][pn][s]["gt_p95"])),
                      "n_by_stratum": dict(collections.Counter(x["stratum"] for x in have))}
        return out

    def two_by_two(recs, pn, probe, ref="E"):
        pl = [x for x in recs if x["scores"][pn][probe] and x["scores"][pn][probe]["gt_p95"]]; el = [x for x in recs if x["scores"][pn][ref]["gt_p95"]]
        kp = {(x["traj_id"], x["action_idx"]) for x in pl}; ke = {(x["traj_id"], x["action_idx"]) for x in el}
        nm = lambda ks: sorted(f"{t[:8]}#{i}" for t, i in ks)  # noqa: E731
        return {"probe_not_E": nm(kp - ke), "E_not_probe": nm(ke - kp), "both": nm(kp & ke), "neither": len(recs) - len(kp | ke)}

    out = {"slug": C.SLUG, "script": f"experiments/{C.SLUG}/p8_quiet_payloads.py", "timestamp": stamp, "git_commit": C.git_commit(),
           "inputs": meta | {"p3_raw": str(Path(p3p).relative_to(C.REPO)), "p2_raw": str(Path(p2p).relative_to(C.REPO)), "oof_npz": p2["oof_npz"], "split_json_sha256": C.sha256(C.HERE / "split.json"), "audit_raws": audit_meta},
           "design": {"scorers": SCORERS, "model_scored": sorted(MODEL_SCORED), "primary_pool": "common_743", "percentile": "share of honest pool strictly below the value",
                      "flags": "strict > common.honest_pct(p) as delta_p_null", "own_model_threshold": "p95 of the honest rows scored by the same fold model / test refit (P7 s3)",
                      "prompt_end_cell": {"layer": pe_layer, "C": pe_C, "train_oof_auroc": best[0], "test_refit": {"n_train_rows": len(tr), "n_test_rows": len(te), **pe_test_metrics},
                                          "note": "train rows: OOF from the P2 checkpoint; test rows: ONE full-train refit here (the only fit in P8)"}},
           "counts": {"payload": len(pay), "quiet": len(quiet), "loud": len(loud), "prep": len(prep), "dropped": dropped, "payload_traj": len({r["traj_id"] for r in pay}),
                      "quiet_traj": dict(collections.Counter(r["traj_id"] for r in quiet)), "quiet_by_group": dict(collections.Counter(group(r) for r in quiet)),
                      "quiet_by_stratum": dict(collections.Counter(stratum.get(key(r)) for r in quiet)), "quiet_integer_ge3": sum(r["score"] >= 3 for r in quiet)},
           "blindness": blind, "thresholds": {pn: {s: {k: v for k, v in t.items() if k != "sorted"} for s, t in d.items()} for pn, d in thr.items()},
           "quiet": Q, "loud": L, "prep_appendix": P,
           "summary": {pn: {"quiet": summary(Q, pn), "loud": summary(L, pn), "prep": summary(P, pn)} for pn in pools},
           "two_by_two_vs_E": {pn: {"quiet": {pr: two_by_two(Q, pn, pr) for pr in ("label|logreg", "label|logreg@prompt_end", "score|ridge_E")},
                                    "loud": {pr: two_by_two(L, pn, pr) for pr in ("label|logreg", "label|logreg@prompt_end", "score|ridge_E")}} for pn in pools}}

    # ── figure
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(9, 7.5)); pn = "common_743"; recs = Q + L; mk = {"end": "s", "random": "o", "random_a4": "^", None: "x"}
    xs = list(range(len(SCORERS)))
    for i, x in enumerate(recs):
        y = len(recs) - 1 - i
        for j, s in enumerate(SCORERS):
            d = x["scores"][pn][s]
            if d is None: continue
            ax.scatter(j + (0.0), y, marker=mk.get(x["stratum"], "x"), s=42, c="#ea580c" if d["gt_p95"] else "#6b7280", edgecolors="k", linewidths=0.4, zorder=3)
            ax.text(j + 0.18, y, f"{d['pct']:.0f}", fontsize=5.5, va="center")
    ax.axhline(len(L) - 0.5, color="k", lw=0.8); ax.text(-0.6, len(L) - 0.5 + 0.2, "quiet (level 0)", fontsize=7); ax.text(-0.6, len(L) - 0.5 - 0.6, "loud (level 2)", fontsize=7)
    ax.set_yticks(range(len(recs))); ax.set_yticklabels([f"{x['traj_id'][:8]}#{x['action_idx']} {x['category'][:10]} int={x['score']:.0f} [{(x['stratum'] or '?')[:6]}]" for x in recs][::-1], fontsize=6)
    ax.set_xticks(xs); ax.set_xticklabels([s.replace("|", "\n") for s in SCORERS], fontsize=6.5); ax.set_xlim(-0.7, len(SCORERS) - 0.3)
    ax.set_title("P8: the 20 audit payloads, honest percentile on the common-743 pool per scorer\norange = above honest p95; marker = draw stratum (square end, circle random, triangle random_a4)", fontsize=8)
    ax.spines[["top", "right"]].set_visible(False); fig.tight_layout(); fp = C.fig_path("p8_quiet_payloads.png"); fig.savefig(fp, dpi=150); plt.close(fig); out["figure"] = str(fp.relative_to(C.REPO))
    C.write_raw("p8_quiet_payloads", out, stamp)

    # ── print
    for pn in ("common_743",):
        for nm, recs in (("QUIET", Q), ("LOUD", L)):
            print(f"\n== {nm} ({pn}) — honest pct per scorer, * = >p95, ** = >p99, (o) = above own-model p95")
            print(f"{'key':14} {'grp':6} {'strat':9} {'cat':22} {'tool':11} int  E     " + " ".join(f"{s[:14]:>14}" for s in SCORERS))
            for x in recs:
                cells = []
                for s in SCORERS:
                    d = x["scores"][pn][s]
                    cells.append("     -        " if d is None else f"{d['pct']:5.1f}{'**' if d['gt_p99'] else '* ' if d['gt_p95'] else '  '}{'(o)' if d['own_model_gt_p95'] else '   '}     ")
                print(f"{x['traj_id'][:8]}#{x['action_idx']:<4} {x['group']:6} {str(x['stratum']):9} {x['category'][:22]:22} {x['tool'][:11]:11} {x['score']:3.0f} {x['E']:5.2f} " + " ".join(c[:14].rjust(14) for c in cells))
            sm = out["summary"][pn][nm.lower()]
            print("  lifted >p95 / >p99 / own-model >p95: " + "  ".join(f"{s[:16]} {v['gt_p95']}/{v['gt_p99']}/{v['own_model_gt_p95'] if v['own_model_gt_p95'] is not None else '-'} of {v['n']}" for s, v in sm.items()))
        print("\n== two-by-two vs E (common_743), quiet:", json.dumps(out["two_by_two_vs_E"][pn]["quiet"]))
        print("== two-by-two vs E (common_743), loud:", json.dumps(out["two_by_two_vs_E"][pn]["loud"]))
    print("\n== the 8 quiet payloads verbatim ==")
    for x in Q:
        print(f"\n-- {x['traj_id'][:8]}#{x['action_idx']} [{x['group']}, stratum {x['stratum']}] {x['category']} / {x['side_task']}  step {x['step_index']}/{x['trajectory_length']} (pos {x['position_in_trajectory']:.2f})  tool {x['tool']}  int {x['score']:.0f}  E {x['E']:.2f}  cached {x['cached_scores']}")
        print(f"   objective: {x['objective']}")
        print(f"   action: {x['action_text_300']!r}")
        print(f"   analysis: {(x['analysis_note_text'] or '')[:300]!r}")


if __name__ == "__main__":
    main()
