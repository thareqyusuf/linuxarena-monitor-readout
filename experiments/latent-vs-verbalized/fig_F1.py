"""F1 — readout loss: E vs the emitted integer (FIGURES.md). Reads P4 (ROC, cluster-bootstrap bands, logit fit) and P0b
(mean digit distributions, P0b rerun with `--pool common743` so the honest bars are the mean over the same 743 rows). Pool: the common 743 honest rows — P4 rerun with `--pool common743` (2026-09-12; same score vectors
as P3, no model), so F1 shares its pool with F2. The P7 §1 scalars on the same pool are printed as a cross-check."""
from fig_common import *

P4 = RAW / "p4_result_pool743_20260912T030653Z.json"; P0B = RAW / "p0b_anchoring_pool743_20260912T031729Z.json"; P7 = RAW / "p7_dpnull_check_20260911T152343Z.json"
p4, p0b, p7 = load(P4), load(P0B), load(P7)
lg = lambda x: np.log10(np.asarray(x, float) / (1 - np.asarray(x, float)))  # noqa: E731
TICKS = [1e-3, 1e-2, 5e-2, 0.1, 0.5]
KEYS = [("score", "integer", "emitted integer"), ("E", "E", "forced readout E"), ("cached_mean", "cached", "GPT-5-nano mean")]

fig, ax = plt.subplots(figsize=(6.5, 3.6))
ax.axvspan(lg(1e-3) - 0.3, lg(1e-2), color="#ebebeb", zorder=0, lw=0)
for key, ck, lab in KEYS:
    d = p4["scorers"][key]; roc = np.array(d["roc"]); f, t = roc[:, 0], roc[:, 1]
    grid = np.array(d["band_grid"]); lo, hi = np.array(d["band_lo"]), np.array(d["band_hi"]); tg = np.interp(grid, f, t)
    m = (grid >= 1e-2) & (grid <= 0.5)
    ax.fill_between(lg(grid[m]), lo[m], hi[m], color=C[ck], alpha=0.12, lw=0)
    ax.plot(lg(grid[m]), tg[m], color=C[ck], ls=M[ck][1], lw=1.6, label=f"{lab} · AUROC {d['pooled']['auroc']:.2f}")
    tk = [x for x in TICKS if x >= 1e-2]; ax.plot(lg(tk), np.interp(tk, f, t), M[ck][0], color=C[ck], ms=5.5, mfc="white" if ck != "integer" else C[ck], mew=1.3)
    fit = d["logit_fit_1e-2_to_0.5"]; xs = np.linspace(lg(1e-3), lg(1e-2), 25); ys = 1 / (1 + 10 ** -(fit["slope"] * xs + fit["intercept"]))
    ax.plot(xs, ys, ls=(0, (3, 2)), color=C[ck], lw=1.2, alpha=0.8)
    print(f"{lab:26s} AUROC {d['pooled']['auroc']:.3f} | TPR@0.05 {d['pooled']['tpr_at_0.05']:.3f} CI {np.round(d['ci95']['tpr_at_0.05'], 3).tolist()}"
          f" | TPR@0.01 {d['pooled']['tpr_at_0.01']:.3f} CI {np.round(d['ci95']['tpr_at_0.01'], 3).tolist()} | extrapolated TPR@1e-3 {fit['tpr_at_1e-3_extrapolated_NOT_MEASURED']:.3f}"
          f" | n_pos {d['n_pos']} n_neg {d['n_neg']} | band at 0.05: [{np.interp(0.05, grid, lo):.3f}, {np.interp(0.05, grid, hi):.3f}]")
    s7 = p7["1_same_pool_743"]["scorers"][key]; assert abs(s7["tpr_at_0.05"] - d["pooled"]["tpr_at_0.05"]) < 1e-9 and abs(s7["auroc"] - d["pooled"]["auroc"]) < 1e-9, key; print(f"{'':26s} (P7 §1 same pool, cross-check) TPR@0.05 {s7['tpr_at_0.05']:.3f} CI {np.round(s7['ci95']['tpr_at_0.05'], 3).tolist()} | TPR@0.01 {s7['tpr_at_0.01']:.3f}")
for x in (1e-2, 5e-2):
    ax.axvline(lg(x), color="#555555", lw=0.8, ls=(0, (2, 2)), zorder=1)
ax.text((lg(1e-3) + lg(1e-2)) / 2, 0.985, "logit-linear\nextrapolation,\nnot measured", ha="center", va="top", fontsize=9, color="#555555")
ax.set_xticks(lg(TICKS)); ax.set_xticklabels([f"{x:g}" for x in TICKS]); ax.set_xlim(lg(1e-3) - 0.08, lg(0.5) + 0.08); ax.set_ylim(0, 1)
ax.set_xlabel("false-positive rate on honest actions (logit scale)"); ax.set_ylabel("true-positive rate, level-2 payloads")
ax.text(lg(1e-2) + 0.12, 0.975, f"n = {p4['scorers']['E']['n_pos']} payloads / {p4['scorers']['E']['n_neg']} honest", va="top", fontsize=9)
ax.legend(loc="upper left", bbox_to_anchor=(0.0, 0.82), frameon=False, handlelength=2.2, handletextpad=0.5)

# inset: mean digit distribution, 31 integer-nulled payloads vs honest
dm = p0b["digit_mass"]; nul, hon = dm["nulled_level2"], dm["honest"]; digits = list(range(11))
a = np.array([nul["mean_digit_dist"][str(k)] for k in digits]); b = np.array([hon["mean_digit_dist"][str(k)] for k in digits])
ins = ax.inset_axes([0.715, 0.10, 0.28, 0.29]); ins.set_facecolor("white"); ins.set_zorder(6); w = 0.4
ins.bar(np.array(digits) - w / 2, a, w, color=C["E"], label=f"nulled ({nul['n']})")
ins.bar(np.array(digits) + w / 2, b, w, color="#bbbbbb", label=f"honest ({hon['n']})"); assert hon["n"] == p4["scorers"]["E"]["n_neg"], (hon["n"], p4["scorers"]["E"]["n_neg"])
ins.set_xticks(digits); ins.set_xticklabels([str(k) if k % 2 == 0 else "" for k in digits], fontsize=9); ins.tick_params(axis="y", labelsize=9); ins.set_ylim(0, 0.8); ins.set_yticks([0, 0.4])
ins.set_title("mean digit distribution", fontsize=9, pad=1); ins.legend(fontsize=9, frameon=False, loc="upper right", handlelength=0.8, handletextpad=0.4, borderaxespad=0.0, labelspacing=0.2)
m610 = float(a[6:].sum()); ins.annotate(f"nulled, digits 6–10: {m610:.3f}", xy=(8, 0.02), xytext=(5.6, 0.30), fontsize=9, arrowprops=dict(arrowstyle="-", lw=0.7, color="#333333"))
for sp in ("top", "right"): ins.spines[sp].set_visible(False)
print(f"inset: nulled mean digit dist {np.round(a, 3).tolist()} (mass 6-10 = {m610:.4f}); honest {np.round(b, 3).tolist()} (mass 6-10 = {b[6:].sum():.4f}); bands 2-5 nulled {a[2:6].sum():.3f} honest {b[2:6].sum():.3f}")
fig.tight_layout(); save(fig, "F1_readout_loss")
