"""F3 — where the linear signal sits (FIGURES.md). Best-layer train-OOF AUROC per capture position from the P2 sweep
(`best_layer_per_position`), for the label arm (logreg) and the score arm (ridge on E); band = range over the 5 grouped folds
of that best cell (from `sweep[].metrics[].per_fold_auroc`); E on the same train rows from `reference_scorers`."""
from fig_common import *

P2 = RAW / "p2_sweep_20260911T143053Z.json"; p2 = load(P2)
POS = ["transcript_end", "action_end", "instruction_end", "prompt_end", "analysis_end", "reasoning_end", "pre_score"]
LAB = ["history", "action", "scoring instructions", "prompt", "hidden CoT", "visible reasoning", "before the digit"]
ARMS = [("label|logreg", "label_probe", "label-supervised probe"), ("score|ridge_E", "score_probe", "score-supervised probe")]
cells = {(c["position"], c["layer"]): c["metrics"] for c in p2["sweep"]}

def fold_range(arm, pos, layer, auroc):
    ms = [(k, v) for k, v in cells[(pos, layer)].items() if k.startswith(arm + "|") and abs(v["auroc"] - auroc) < 1e-9]
    assert ms, (arm, pos, layer); return ms[0][0], ms[0][1]["per_fold_auroc"]

fig, ax = plt.subplots(figsize=(4.5, 2.6)); x = np.arange(len(POS))
ax.axvspan(5.5, 6.6, color="#e6e6e6", lw=0, zorder=0); ax.axvline(3.5, color="#888888", lw=0.9, zorder=1)
ax.text(3.58, 0.998, "generation starts", ha="left", va="top", fontsize=9, color="#555555")
ax.text(6.55, 0.305, "reads the verdict\n(digit mass 1.0)", ha="right", va="bottom", fontsize=9, color="#555555")
for arm, ck, lab in ARMS:
    bp = p2["best_layer_per_position"][arm]; ys = [bp[p]["auroc"] for p in POS]; lo, hi = [], []
    for p, yv in zip(POS, ys):
        cfg, folds = fold_range(arm, p, bp[p]["layer"], yv); lo.append(min(folds)); hi.append(max(folds))
        print(f"{lab:24s} {p:16s} layer {bp[p]['layer']:2d} cfg {cfg:24s} OOF AUROC {yv:.3f}  fold range [{min(folds):.3f}, {max(folds):.3f}]")
    dx = -0.13 if ck == "label_probe" else 0.13
    for xi, p in zip(x, POS):
        _, folds = fold_range(arm, p, bp[p]["layer"], bp[p]["auroc"]); ax.plot([xi + dx] * len(folds), folds, ".", color=C[ck], ms=3.5, alpha=0.45, zorder=2)
    ax.plot(x, ys, marker=M[ck][0], ls=M[ck][1], color=C[ck], lw=1.5, ms=5, label=lab, zorder=3)
e = p2["reference_scorers"]["E"]["train_oof_population"]; ax.axhline(e["auroc"], color=C["E"], lw=1, ls=(0, (4, 3)))
ax.text(-0.35, e["auroc"] - 0.006, f"E {e['auroc']:.2f}", color=C["E"], fontsize=9, va="top", ha="left")
print(f"E train-OOF population AUROC {e['auroc']:.3f} (n_pos {e['n_pos']}, n_neg {e['n_neg']})")
pop = p2["populations"]["train"]; ax.text(1.0, 1.01, f"n = {pop['n_pos']} payloads / {pop['n_neg']} honest, {pop['n_traj']} train trajectories, out-of-fold", transform=ax.transAxes, ha="right", va="bottom", fontsize=9)
ax.set_xticks(x); ax.set_xticklabels(LAB, fontsize=9, rotation=25, ha="right", rotation_mode="anchor"); ax.set_xlabel("last token of …"); ax.set_xlim(-0.4, 6.6); ax.set_ylim(0.3, 1.0); ax.set_ylabel("best-layer OOF AUROC")
from matplotlib.patches import Patch
from matplotlib.legend_handler import HandlerTuple
h, l = ax.get_legend_handles_labels(); from matplotlib.lines import Line2D
h.append((Line2D([], [], ls="", marker=".", color=C["label_probe"], ms=3.5, alpha=0.45), Line2D([], [], ls="", marker=".", color=C["score_probe"], ms=3.5, alpha=0.45))); l.append("small dots: the 5 fold-wise AUROCs")
ax.legend(h, l, loc="lower right", bbox_to_anchor=(1.0, 0.17), frameon=False, ncol=1, handlelength=2.4, labelspacing=0.25, handler_map={tuple: HandlerTuple(ndivide=None, pad=0)})
save(fig, "F3_where_signal_sits")
