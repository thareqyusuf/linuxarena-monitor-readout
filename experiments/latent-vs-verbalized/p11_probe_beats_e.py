"""P11 — every row where a probe beats E, and whether the boring explanation reaches it. Pool-wide version of the
P8 `30e71655#26` paragraph.

Population: the P3 pooled-59 vector restricted to rows every scorer can score (generation positions present):
1,722 rows = 743 honest + 53 level-2 + 926 attack-other. Percentile = share of the 743 honest strictly below the
value, per scorer. Lifted = above the honest p95 (common.honest_pct, strict >).

For each probe P in {label|logreg (action_end L8), score|ridge_E (reasoning_end L14), label|logreg@prompt_end (L4,
pre-registered; train rows from the P2 OOF checkpoint, test rows by the same one full-train refit P8 used)}:
  beats_E  = lifted(P) and not lifted(E)                          (the Δp_null-style event)
  wide     = pct(P) − pct(E) >= 40                                 (the 30e71655#26 shape: 100 vs 38)
Each beats_E row is tagged with the boring alternatives that also reach it:
  integer_caught  emitted integer >= 6      surface  text or dumb lifted      cached_caught  cached_mean lifted
and 'unexplained' when none does. Counts by class (honest = the probe's false positives; level-2; attack-other; and
the audit verdict where one exists). Reverse direction (E lifted, probe not) reported for symmetry.

Also the E-vs-integer check: rows with integer >= 6 whose E sits below the honest median (the reasoning caught it, the
pre-CoT readout did not), by class, with the analysis text; and the full integer-band x E-band cross-tab.

Raw results/raw/<slug>/p11_probe_beats_e_<ts>.json; figure results/figures/<slug>/p11_probe_beats_e.png.
"""
from __future__ import annotations

import collections, glob, json, sys, warnings
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common as C
from p2_probes import standardize

PROBES = ["label|logreg", "score|ridge_E", "label|logreg@prompt_end"]
BASE = ["score", "E", "cached_mean", "text", "dumb"]
WIDE = 40


def main():
    stamp = C.ts()
    rows, meta = C.build_table(with_text=True); u = C.usable(rows, "current")
    split = json.loads((C.HERE / "split.json").read_text()); test_ids = set(split["test"])
    p3p = sorted(p for p in glob.glob(str(C.RAW_DIR / "p3_baselines_*.json")) if "_ph" not in p)[-1]; p3 = json.load(open(p3p))
    p2p = sorted(p for p in glob.glob(str(C.RAW_DIR / "p2_sweep_*.json")) if "_nosusp" not in p)[-1]; p2 = json.load(open(p2p))
    p8p = sorted(glob.glob(str(C.RAW_DIR / "p8_quiet_payloads_*.json")))[-1]; p8 = json.load(open(p8p))
    cols = p3["scores_pooled_59"]["columns"]; S = {(r[0], r[1]): dict(zip(cols, r)) for r in p3["scores_pooled_59"]["rows"]}
    key = lambda r: (r["traj_id"], r["action_idx"])  # noqa: E731

    # prompt_end label cell for every row: OOF (train) + the P8 refit recipe (test)
    pe = C.POSITIONS.index("prompt_end"); pe_layer, pe_C = p8["design"]["prompt_end_cell"]["layer"], p8["design"]["prompt_end_cell"]["C"]
    pe_cfg = f"label|logreg|{pe_C}"
    z = np.load(C.REPO / p2["oof_npz"]); tk = [(a, int(b)) for a, b in (x.split("|") for x in z["train_keys"])]
    mask = z[f"mask|{pe}|{pe_layer}"]; pred = z[f"{pe}|{pe_layer}|{pe_cfg}"]; keys_m = [k for k, m in zip(tk, mask) if m]; assert len(keys_m) == len(pred)
    pe_score = {k: float(v) for k, v in zip(keys_m, pred)}
    acts, idx = C.open_acts(); row_of = {(r["traj_id"], r["action_idx"]): r["row"] for r in idx["rows"]}
    tr = [r for r in u if r["traj_id"] not in test_ids and (r["is_pos"] or r["is_neg"])]; te = [r for r in u if r["traj_id"] in test_ids]
    Xs, Xe = standardize(np.asarray(acts[[row_of[key(r)] for r in tr], pe, pe_layer, :], np.float32), np.asarray(acts[[row_of[key(r)] for r in te], pe, pe_layer, :], np.float32))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        m = LogisticRegression(C=pe_C, penalty="l2", solver="lbfgs", max_iter=1000, tol=1e-4).fit(Xs, np.array([r["is_pos"] for r in tr]).astype(int))
    pe_score.update({key(r): float(v) for r, v in zip(te, m.decision_function(Xe))})
    # sanity: P8's stored payload values must reproduce
    for x in p8["quiet"] + p8["loud"]:
        assert abs(pe_score[(x["traj_id"], x["action_idx"])] - x["scores"]["common_743"]["label|logreg@prompt_end"]["value"]) < 1e-6
    for r in u:
        S[key(r)]["label|logreg@prompt_end"] = pe_score.get(key(r))

    # population and percentiles
    P = [r for r in u if all(S[key(r)][s] is not None for s in PROBES + BASE)]
    hon = [r for r in P if r["is_neg"]]
    sorted_h = {s: np.sort([S[key(r)][s] for r in hon]) for s in PROBES + BASE}
    thr = {s: {"p50": C.honest_pct(sorted_h[s], 0.50), "p95": C.honest_pct(sorted_h[s], 0.95)} for s in sorted_h}
    pct = lambda s, x: float(np.searchsorted(sorted_h[s], x, side="left") / len(sorted_h[s]) * 100)  # noqa: E731
    lifted = lambda s, x: x > thr[s]["p95"]  # noqa: E731
    def cls(r):
        return "honest" if r["is_neg"] else "level2" if r["is_pos"] else "attack_other"
    audit = {(x["traj_id"], x["action_idx"]): ("quiet" if x in p8["quiet"] else "loud") for x in p8["quiet"] + p8["loud"]} | {(x["traj_id"], x["action_idx"]): "prep" for x in p8["prep_appendix"]}

    def row_rec(r, probe):
        v = S[key(r)]; d = {"traj_id": r["traj_id"], "action_idx": r["action_idx"], "class": cls(r), "audit": audit.get(key(r)), "label_level": r["label_level"], "category": r["side_task_category"],
                            "group": "test" if r["traj_id"] in test_ids else f"fold{split['fold_of'][r['traj_id']]}", "tool": r["tool_name"], "n_action_chars": r["n_action_chars"], "position_in_trajectory": r["position_in_trajectory"],
                            "integer": v["score"], "E": v["E"], "cached_mean": v["cached_mean"],
                            "pct": {s: pct(s, v[s]) for s in PROBES + BASE}, "lifted": {s: lifted(s, v[s]) for s in PROBES + BASE}}
        d["gap"] = d["pct"][probe] - d["pct"]["E"]
        d["explained_by"] = [t for t, ok in (("integer_caught", v["score"] >= 6), ("surface", d["lifted"]["text"] or d["lifted"]["dumb"]), ("cached_caught", d["lifted"]["cached_mean"])) if ok]
        d["unexplained"] = not d["explained_by"]; d["E_near_line"] = d["pct"]["E"] >= 90; d["action_text_200"] = r["action_text"][:200]; d["analysis_200"] = (r["analysis_note_text"] or "")[:200]
        return d

    out = {"slug": C.SLUG, "script": f"experiments/{C.SLUG}/p11_probe_beats_e.py", "timestamp": stamp, "git_commit": C.git_commit(),
           "inputs": meta | {"p3_raw": str(Path(p3p).relative_to(C.REPO)), "p2_raw": str(Path(p2p).relative_to(C.REPO)), "p8_raw": str(Path(p8p).relative_to(C.REPO)), "oof_npz": p2["oof_npz"], "split_json_sha256": C.sha256(C.HERE / "split.json")},
           "design": {"population": {"n": len(P), "honest": len(hon), "level2": sum(r["is_pos"] for r in P), "attack_other": sum(cls(r) == "attack_other" for r in P)}, "wide_gap_points": WIDE,
                      "thresholds": thr, "prompt_end_cell": {"layer": pe_layer, "C": pe_C, "test_rows_refit": len(te)}, "explained_by": "integer_caught: integer>=6; surface: text or dumb above honest p95; cached_caught: cached_mean above honest p95"},
           "probes": {},
           "rows_all": {"columns": ["traj_id", "action_idx", "class", "audit", "label_level", "integer", *PROBES, *[b for b in BASE if b != "score"]],
                        "note": "raw scorer values for every row in the population; pct = searchsorted(sorted honest values, x, 'left')/n_honest; lifted = x > common.honest_pct(honest, .95)",
                        "rows": [[r["traj_id"], r["action_idx"], cls(r), audit.get(key(r)), r["label_level"], S[key(r)]["score"], *[S[key(r)][s_] for s_ in PROBES], *[S[key(r)][b] for b in BASE if b != "score"]] for r in P]}}
    for probe in PROBES:
        beats = [row_rec(r, probe) for r in P if lifted(probe, S[key(r)][probe]) and not lifted("E", S[key(r)]["E"])]
        rev = [row_rec(r, probe) for r in P if lifted("E", S[key(r)]["E"]) and not lifted(probe, S[key(r)][probe])]
        both = sum(lifted(probe, S[key(r)][probe]) and lifted("E", S[key(r)]["E"]) for r in P)
        def tab(recs):
            t = {}
            for c in ("honest", "level2", "attack_other"):
                rr = [d for d in recs if d["class"] == c]
                t[c] = {"n": len(rr), "wide": sum(d["gap"] >= WIDE for d in rr), "integer_caught": sum("integer_caught" in d["explained_by"] for d in rr), "surface": sum("surface" in d["explained_by"] for d in rr),
                        "cached_caught": sum("cached_caught" in d["explained_by"] for d in rr), "unexplained": sum(d["unexplained"] for d in rr), "unexplained_and_wide": sum(d["unexplained"] and d["gap"] >= WIDE for d in rr), "unexplained_and_E_near_line": sum(d["unexplained"] and d["E_near_line"] for d in rr),
                        "audit": dict(collections.Counter(d["audit"] for d in rr if d["audit"]))}
            return t
        out["probes"][probe] = {"n_lifted": int(sum(lifted(probe, S[key(r)][probe]) for r in P)), "n_E_lifted": int(sum(lifted("E", S[key(r)]["E"]) for r in P)), "n_both": int(both),
                                "beats_E": {"summary": tab(beats), "rows": sorted(beats, key=lambda d: -d["gap"])}, "E_beats_probe": {"summary": tab(rev), "rows": sorted(rev, key=lambda d: d["gap"])}}
    # E vs integer
    Eb = lambda x: "lt_p50" if x <= thr["E"]["p50"] else "p50_p95" if x <= thr["E"]["p95"] else "gt_p95"  # noqa: E731
    Ib = lambda k: "0-1" if k <= 1 else "2-5" if k <= 5 else "6-10"  # noqa: E731
    xt = collections.defaultdict(lambda: collections.Counter())
    for r in P: xt[f"{cls(r)}|int{Ib(S[key(r)]['score'])}"][Eb(S[key(r)]["E"])] += 1
    hi_lo = [row_rec(r, "label|logreg") for r in P if S[key(r)]["score"] >= 6 and S[key(r)]["E"] <= thr["E"]["p50"]]
    out["E_vs_integer"] = {"crosstab_class_x_integer_band_x_E_band": {k: dict(v) for k, v in xt.items()},
                           "integer_ge6_and_E_below_honest_p50": {"n": len(hi_lo), "by_class": dict(collections.Counter(d["class"] for d in hi_lo)), "by_traj": dict(collections.Counter(d["traj_id"][:8] for d in hi_lo)),
                                                                   "n_integer_ge6_total": int(sum(S[key(r)]["score"] >= 6 for r in P)), "rows": sorted(hi_lo, key=lambda d: d["pct"]["E"])}}

    # ── figure: A/B scatter per probe, C strip for level-2 / audit rows, D integer vs E, S stacked summary
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    fig = plt.figure(figsize=(15, 10)); gs = fig.add_gridspec(2, 3, height_ratios=[1, 1]); col = {"honest": C.COLORS["honest"], "level2": C.COLORS["level2"], "attack_other": C.COLORS["attack_other"]}
    for j, probe in enumerate(["label|logreg", "score|ridge_E"]):
        ax = fig.add_subplot(gs[0, j])
        for c, z_, a_, s_ in (("honest", 1, 0.35, 8), ("attack_other", 2, 0.4, 8), ("level2", 3, 0.9, 24)):
            rr = [r for r in P if cls(r) == c]; ax.scatter([pct("E", S[key(r)]["E"]) for r in rr], [pct(probe, S[key(r)][probe]) for r in rr], s=s_, c=col[c], alpha=a_, zorder=z_, label=f"{c} (n={len(rr)})", edgecolors="none")
        qr = [r for r in P if audit.get(key(r)) == "quiet"]; ax.scatter([pct("E", S[key(r)]["E"]) for r in qr], [pct(probe, S[key(r)][probe]) for r in qr], s=80, facecolors="none", edgecolors="k", linewidths=1.0, zorder=4, label="audit quiet payload (8)")
        for d in out["probes"][probe]["beats_E"]["rows"]:
            if (d["class"] == "level2" and d["gap"] >= WIDE) or d["audit"] == "quiet":
                off = (6, -14) if d["audit"] == "quiet" else (6, -26)
                ax.annotate(f"{d['traj_id'][:8]}#{d['action_idx']}  int={d['integer']:.0f}  text={d['pct']['text']:.0f}  dumb={d['pct']['dumb']:.0f}", (d["pct"]["E"], d["pct"][probe]), fontsize=6.5, xytext=off, textcoords="offset points", arrowprops=dict(arrowstyle="-", lw=0.5))
        ax.axvline(95, color="k", lw=0.6, ls=":"); ax.axhline(95, color="k", lw=0.6, ls=":"); ax.fill_between([0, 95], 95, 100.5, color="#fde68a", alpha=0.35, zorder=0)
        ax.set_xlabel("E, honest percentile"); ax.set_ylabel(f"{probe}, honest percentile"); ax.set_xlim(-1, 101); ax.set_ylim(-1, 101)
        sm = out["probes"][probe]["beats_E"]["summary"]
        ax.set_title(f"({'AB'[j]}) {probe}\nprobe > p95 & E < p95 (shaded): {sum(v['n'] for v in sm.values())} rows = honest {sm['honest']['n']} (FPs)\n+ level-2 {sm['level2']['n']} ({sm['level2']['unexplained']} unexplained, all E >= p90) + attack-other {sm['attack_other']['n']} ({sm['attack_other']['unexplained']} unexplained)", fontsize=7.5)
        ax.legend(fontsize=6.5, loc="lower right", frameon=False); ax.spines[["top", "right"]].set_visible(False)
    # D integer vs E
    ax = fig.add_subplot(gs[0, 2]); rng = np.random.RandomState(0)
    for c, a_, s_ in (("honest", 0.3, 8), ("attack_other", 0.4, 8), ("level2", 0.9, 24)):
        rr = [r for r in P if cls(r) == c]; ax.scatter([S[key(r)]["score"] + rng.uniform(-0.25, 0.25) for r in rr], [pct("E", S[key(r)]["E"]) for r in rr], s=s_, c=col[c], alpha=a_, edgecolors="none", label=c)
    ax.axhline(50, color="k", lw=0.6, ls=":"); ax.axhline(95, color="k", lw=0.6, ls=":"); ax.axvline(5.5, color="k", lw=0.6, ls=":"); ax.fill_between([5.5, 10.5], -1, 50, color="#fecaca", alpha=0.4, zorder=0)
    hl = out["E_vs_integer"]["integer_ge6_and_E_below_honest_p50"]
    for d in hl["rows"]:
        ax.annotate(f"{d['traj_id'][:8]}#{d['action_idx']}", (d["integer"], d["pct"]["E"]), fontsize=6, xytext=(-40, 4), textcoords="offset points")
    ax.set_title(f"(D) emitted integer vs E\nshaded: integer >= 6 but E below honest median\n{hl['n']} of {hl['n_integer_ge6_total']} integer>=6 rows (level-2 {hl['by_class'].get('level2', 0)}, attack-other {hl['by_class'].get('attack_other', 0)})", fontsize=7.5)
    ax.set_xlabel("emitted integer (jittered)"); ax.set_ylabel("E, honest percentile"); ax.legend(fontsize=6.5, frameon=False, loc="center left"); ax.spines[["top", "right"]].set_visible(False)
    # C strip: level-2 and audit-quiet rows where either chosen probe beats E
    ax = fig.add_subplot(gs[1, :2]); keys_c = sorted({(d["traj_id"], d["action_idx"]) for pr in ("label|logreg", "score|ridge_E") for d in out["probes"][pr]["beats_E"]["rows"] if d["class"] == "level2" or d["audit"] == "quiet"})
    recs = [row_rec(next(r for r in P if key(r) == k), "label|logreg") for k in keys_c]; recs.sort(key=lambda d: -(d["pct"]["label|logreg"] - d["pct"]["E"])); yy = np.arange(len(recs))[::-1]
    for y, d in zip(yy, recs):
        lo = min(d["pct"]["E"], d["pct"]["label|logreg"], d["pct"]["score|ridge_E"]); hi = max(d["pct"]["E"], d["pct"]["label|logreg"], d["pct"]["score|ridge_E"]); ax.plot([lo, hi], [y, y], color="#d1d5db", lw=1.2, zorder=1)
        ax.scatter(d["pct"]["E"], y, marker="o", c=C.COLORS["E"], s=30, zorder=3); ax.scatter(d["pct"]["label|logreg"], y, marker="D", c=C.COLORS["probe_label"], s=26, zorder=3); ax.scatter(d["pct"]["score|ridge_E"], y, marker="s", c=C.COLORS["probe_score"], s=26, zorder=3)
        ax.scatter(d["pct"]["text"], y, marker="|", c=C.COLORS["text"], s=90, zorder=2); ax.scatter(d["pct"]["dumb"], y, marker="|", c=C.COLORS["dumb"], s=90, zorder=2); ax.scatter(d["pct"]["cached_mean"], y, marker="_", c=C.COLORS["cached_mean"], s=90, zorder=2)
        tags = "  ".join(t for t in ("int>=6" if "integer_caught" in d["explained_by"] else "", "surface" if "surface" in d["explained_by"] else "", "cached" if "cached_caught" in d["explained_by"] else "")) or ("unexplained, E at p%d" % d["pct"]["E"])
        ax.text(102, y, f"int {d['integer']:.0f}   {tags}", fontsize=6.5, va="center")
    ax.set_yticks(yy); ax.set_yticklabels([f"{d['traj_id'][:8]}#{d['action_idx']} {d['category'][:12]} {d['tool'][:11]} {d['n_action_chars']}ch" + ("  [audit quiet]" if d["audit"] == "quiet" else "") for d in recs], fontsize=6.5)
    ax.axvline(95, color="k", lw=0.6, ls=":"); ax.set_xlim(0, 135); ax.set_xlabel("honest percentile:   ● E    ◆ label probe    ■ score probe    | text    | dumb    — cached_mean")
    ax.set_title(f"(C) the {len(recs)} level-2 / audit-quiet rows where a probe is above p95 and E is not — with the surface baselines on the same row", fontsize=8); ax.spines[["top", "right"]].set_visible(False)
    # S stacked summary
    ax = fig.add_subplot(gs[1, 2]); cats_ = [("label|logreg", "level2"), ("label|logreg", "attack_other"), ("score|ridge_E", "level2"), ("score|ridge_E", "attack_other")]
    parts = [("integer_caught", "#b91c1c", "integer >= 6"), ("surface_only", "#7c3aed", "surface (text/dumb) reaches it"), ("cached_only", "#2563eb", "only cached_mean reaches it"), ("unexplained_near", "#fbbf24", "unexplained, E >= p90"), ("unexplained_far", "#111827", "unexplained, E < p90")]
    for i, (pr, c) in enumerate(cats_):
        rr = [d for d in out["probes"][pr]["beats_E"]["rows"] if d["class"] == c]; bottom = 0
        cnt = {"integer_caught": sum("integer_caught" in d["explained_by"] for d in rr), "surface_only": sum("integer_caught" not in d["explained_by"] and "surface" in d["explained_by"] for d in rr),
               "cached_only": sum(d["explained_by"] == ["cached_caught"] for d in rr), "unexplained_near": sum(d["unexplained"] and d["E_near_line"] for d in rr), "unexplained_far": sum(d["unexplained"] and not d["E_near_line"] for d in rr)}
        for k_, col_, lab_ in parts:
            ax.bar(i, cnt[k_], 0.7, bottom=bottom, color=col_, label=lab_ if i == 0 else None); bottom += cnt[k_]
        ax.text(i, bottom + 1, str(len(rr)), ha="center", fontsize=7)
    ax.set_xticks(range(4)); ax.set_xticklabels([f"{pr.split('|')[0]} probe\n{c}" for pr, c in cats_], fontsize=7); ax.set_ylabel("rows where probe > p95 and E < p95")
    ax.set_title("(S) what else reaches each probe-beats-E row", fontsize=8); ax.legend(fontsize=6, frameon=False, loc="upper right"); ax.spines[["top", "right"]].set_visible(False)
    fig.suptitle(f"P11: where a probe beats E — pool of {len(P)} rows ({len(hon)} honest, 53 level-2, {len(P) - len(hon) - 53} attack-other); percentiles over the common honest pool", fontsize=9); fig.tight_layout()
    fp = C.fig_path("p11_probe_beats_e.png"); fig.savefig(fp, dpi=150); plt.close(fig); out["figure"] = str(fp.relative_to(C.REPO))
    C.write_raw("p11_probe_beats_e", out, stamp)

    for probe in PROBES:
        d = out["probes"][probe]; print(f"\n== {probe}: lifted {d['n_lifted']} | E lifted {d['n_E_lifted']} | both {d['n_both']} ==")
        for c, v in d["beats_E"]["summary"].items():
            print(f"  beats_E {c:13} n {v['n']:3}  wide {v['wide']:3}  int>=6 {v['integer_caught']:3}  surface {v['surface']:3}  cached {v['cached_caught']:3}  UNEXPLAINED {v['unexplained']:3} (wide {v['unexplained_and_wide']}, E>=p90 {v['unexplained_and_E_near_line']})  audit {v['audit']}")
        for c, v in d["E_beats_probe"]["summary"].items():
            print(f"  E_beats  {c:13} n {v['n']:3}  audit {v['audit']}")
        print("  non-honest beats_E rows (gap desc):")
        for x in d["beats_E"]["rows"]:
            if x["class"] != "honest":
                print(f"    {x['traj_id'][:8]}#{x['action_idx']:<3} {x['class']:12} {str(x['audit']):5} lvl {x['label_level']} int {x['integer']:.0f}  probe {x['pct'][probe]:5.1f} E {x['pct']['E']:5.1f} gap {x['gap']:+5.1f} | text {x['pct']['text']:5.1f} dumb {x['pct']['dumb']:5.1f} cached {x['pct']['cached_mean']:5.1f} | {x['explained_by'] or 'UNEXPLAINED'} | {x['tool']} {x['n_action_chars']}ch pos {x['position_in_trajectory']:.2f}")
    hl = out["E_vs_integer"]["integer_ge6_and_E_below_honest_p50"]; print(f"\n== integer >= 6 and E below honest p50: {hl['n']} of {hl['n_integer_ge6_total']}; by class {hl['by_class']}; by traj {hl['by_traj']} ==")
    for x in hl["rows"]:
        print(f"    {x['traj_id'][:8]}#{x['action_idx']:<3} {x['class']:12} {str(x['audit']):5} int {x['integer']:.0f} E {x['E']:.2f} (pct {x['pct']['E']:.0f}) cached {x['cached_mean']} text {x['pct']['text']:.0f} | {x['analysis_200'][:150]!r}")
    print("\ncrosstab:", json.dumps(out["E_vs_integer"]["crosstab_class_x_integer_band_x_E_band"]))


if __name__ == "__main__":
    main()
