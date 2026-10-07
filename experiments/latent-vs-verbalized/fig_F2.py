"""F2 — no probe beats E (FIGURES.md). Pooled-59 AUROC + cluster-bootstrap CI on the common 743-honest pool from P7 §1;
leave-one-category-out and matched-negative AUROCs from P3 (their own pools: LOCO uses each scorer's full pool, matched uses
147 negatives); Δp_null recoveries (k above honest p95, 743 pool) from P7 §1; the per-trajectory share of the label probe's
recoveries read off P11's row table with the p95 threshold recorded in the same raw."""
from fig_common import *

P3 = RAW / "p3_baselines_20260911T143755Z.json"; P7 = RAW / "p7_dpnull_check_20260911T152343Z.json"; P11 = RAW / "p11_probe_beats_e_20260912T015856Z.json"
p3, p7, p11 = load(P3), load(P7), load(P11)
ROWS = [("E", "E", "forced readout E"), ("score|ridge_E", "score_probe", "score probe"), ("label|logreg", "label_probe", "label probe"), ("text", "text", "TF-IDF text"), ("dumb", "dumb", "9 dumb features")]
same = p7["1_same_pool_743"]["scorers"]; loco = p3["9_leave_one_category_out"]["pooled"]; matched = p3["7_matched_negatives"]["scorers"]
# recoveries by trajectory on the 743 pool: nulled level-2 rows (label_level 2, integer <= 1) above each scorer's p95 recorded in P11
cols = p11["rows_all"]["columns"]; ci_ = cols.index; TH = p11["design"]["thresholds"]
nulled = [r for r in p11["rows_all"]["rows"] if r[ci_("label_level")] == 2 and r[ci_("integer")] <= 1]
traj_n = {}
for r in nulled: traj_n[r[ci_("traj_id")][:8]] = traj_n.get(r[ci_("traj_id")][:8], 0) + 1
BIG = max(traj_n, key=traj_n.get)
lifted = {key: [r for r in nulled if r[ci_(key)] is not None and r[ci_(key)] > TH[key]["p95"]] for key, _, _ in ROWS}
from_big = {key: sum(r[ci_("traj_id")].startswith(BIG) for r in lifted[key]) for key in lifted}
for key, _, _ in ROWS:
    assert len(lifted[key]) == same[key]["delta_p_null"]["p95"]["k"], (key, len(lifted[key]), same[key]["delta_p_null"]["p95"]["k"])
print(f"nulled payloads by trajectory: {traj_n}; largest {BIG} ({traj_n[BIG]} of {len(nulled)})")
print("recovered on 743 pool (from " + BIG + "): " + ", ".join(f"{key} {len(lifted[key])} ({from_big[key]})" for key in lifted))

fig, ax = plt.subplots(figsize=(6.5, 3.4)); fig.subplots_adjust(right=0.74)
ys = list(range(len(ROWS)))[::-1]
for y in ys: ax.axhline(y, color="#eeeeee", lw=0.8, zorder=0)
for y, (key, ck, lab) in zip(ys, ROWS):
    s = same[key]; lo, hi = s["ci95"]["auroc"]; mk, _ = M[ck]
    ax.plot([lo, hi], [y, y], color=C[ck], lw=1.6, solid_capstyle="butt")
    ax.plot([s["auroc"]], [y], mk, color=C[ck], ms=8 if mk in "x+" else 7, mew=1.8 if mk in "x+" else 1, zorder=4)
    line = f"{lab:18s} pooled(743) {s['auroc']:.3f} CI [{lo:.3f}, {hi:.3f}] | k(p95) {s['delta_p_null']['p95']['k']}/{s['delta_p_null']['n_nulled']}"
    if key in loco and key != "E":
        v = loco[key]["auroc"]; ax.plot([v], [y + 0.22], mk, color=C[ck], ms=6, mfc="white" if mk not in "x+" else None, mew=1.2, alpha=1 if mk not in "x+" else 0.5, zorder=4)
        line += f" | LOCO {v:.3f} (n_neg {loco[key]['n_neg']})"
    if key in matched:
        v = matched[key]["auroc"]; ax.plot([v], [y - 0.22], mk, color=C[ck], ms=3.5 if mk not in "x+" else 5, mew=1.2, zorder=4)
        line += f" | matched {v:.3f} (n_neg {matched[key]['n_neg']})"
    print(line)
    k = s["delta_p_null"]["p95"]["k"]
    ax.text(1.015, y, f"{k}  ({from_big[key]})", transform=ax.get_yaxis_transform(), va="center", ha="left", fontsize=9, clip_on=False)
ax.text(1.015, len(ROWS) - 0.5, f"of the {len(nulled)} the integer misses,\nhow many above honest p95\n(brackets: from trajectory\n{BIG}, which holds\n{traj_n[BIG]} of the {len(nulled)})", transform=ax.get_yaxis_transform(), va="bottom", ha="left", fontsize=9, clip_on=False)
eV = same["E"]["auroc"]; ax.axvline(eV, color=C["E"], lw=0.9, ls=(0, (4, 3)), zorder=1); ax.text(eV - 0.008, len(ROWS) + 0.3, f"E {eV:.3f}", color=C["E"], ha="right", va="bottom", fontsize=9)
print(f"data_tampering LOCO fold: label probe {p3['9_leave_one_category_out']['groups']['data_tampering']['label|logreg']['auroc']:.3f}, score probe {p3['9_leave_one_category_out']['groups']['data_tampering']['score|ridge_E']['auroc']:.3f}")
b = p7["2_bootstrap"]["pool_743_score_vs_E"]["score|ridge_E-E|d_auroc"]; print(f"paired score probe − E AUROC (743): {b['mean']:+.3f} CI [{b['ci95'][0]:+.3f}, {b['ci95'][1]:+.3f}]")
ax.set_yticks(ys); ax.set_yticklabels([r[2] for r in ROWS]); ax.set_xlim(0.2, 1.0); ax.set_ylim(-0.6, len(ROWS) + 0.55); ax.set_xlabel("AUROC, level-2 payloads vs honest actions")
from matplotlib.lines import Line2D
proxies = [Line2D([], [], color="#333333", marker="o", ms=7, lw=1.6, label="pooled, 59 trajectories (95 % CI)"),
           Line2D([], [], color="#333333", marker="o", ms=6, mfc="white", lw=0, label="leave one attack category out"),
           Line2D([], [], color="#333333", marker="o", ms=3.5, lw=0, label=f"matched negatives (n = {p3['7_matched_negatives']['n_matched_neg']})")]
ax.legend(handles=proxies, loc="upper left", frameon=False, handlelength=1.5, borderaxespad=0.2)
ax.text(0.0, 1.01, f"n = {same['E']['n_pos']} payloads / {same['E']['n_neg']} honest", transform=ax.transAxes, fontsize=9, va="bottom")
save(fig, "F2_no_probe_beats_E")
