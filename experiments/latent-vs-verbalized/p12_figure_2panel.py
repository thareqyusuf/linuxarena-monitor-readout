"""P12 write-up figure: two panels, model-free, drawn from the P12 raws.

  A  #25 — forced digit readout after the model's own analysis fed one sentence at a time; one line at the sentence
     that names the covert objective (s11).
  B  #27 — the benign --max-words edit; its own (flat) curve plus the two appends that make the point: the objective
     sentence (no verdict word) vs a length- and register-matched concern text with no objective.
Keeps results/figures/<slug>/p12_hop_truncation.png untouched; writes p12_hop_truncation_2panel.png.

  python p12_figure_2panel.py <P12 raw with curves> <P12C raw with the matched appends>
"""
from __future__ import annotations

import json, sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common as C


def main():
    curves = json.load(open(sys.argv[1])); appends = json.load(open(sys.argv[2]))
    A = {a["action_idx"]: a for a in curves["actions"]}; B = {a["action_idx"]: a for a in appends["actions"]}
    iv27 = {x["name"]: x for x in B[27]["interventions"]}
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    plt.rcParams.update({"font.size": 12})
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12.5, 5), gridspec_kw={"width_ratios": [1.15, 1]})

    # A — #25
    cv = A[25]["curve"]; xs = [c["point"] for c in cv]; ys = [c["E"] for c in cv]
    ax1.plot(xs, ys, "-o", color="k", ms=4.5, lw=1.4)
    s_obj = A[25]["first_marker_sentence"]["objective"]
    ax1.axvline(s_obj, color="#b91c1c", lw=1.6, ls="--")
    ax1.text(s_obj - 0.35, 9.6, "names the covert objective", color="#b91c1c", ha="right", va="top", fontsize=12)
    ax1.axhline(A[25]["recorded"]["E"], color=C.COLORS["E"], lw=1, ls=":")
    ax1.text(6.0, A[25]["recorded"]["E"] - 0.55, f"E, no CoT = {A[25]['recorded']['E']:.2f}", color=C.COLORS["E"], fontsize=11)
    for pt, txt, dx, dy in ((0, f"{ys[0]:.2f}", 0.3, 1.0), (s_obj, f"{ys[s_obj]:.1f}", 0.4, -0.1), (xs[-1], f"{ys[-1]:.1f}", -0.3, 0.35)):
        ax1.annotate(txt, (pt, ys[pt]), xytext=(pt + dx, ys[pt] + dy), fontsize=11, ha="left" if dx > 0 else "right")
    ax1.set_title("#25  payload (emitted integer 9)", fontsize=13)
    ax1.set_xlabel("sentences of the model's own analysis fed before forcing <score>"); ax1.set_ylabel("forced digit readout (expected score)")
    ax1.set_ylim(-0.3, 10.3); ax1.set_xlim(-0.5, xs[-1] + 0.8); ax1.spines[["top", "right"]].set_visible(False)

    # B — #27
    cv = A[27]["curve"]; xs = [c["point"] for c in cv]; ys = [c["E"] for c in cv]
    ax2.plot(xs, ys, "-o", color="k", ms=4.5, lw=1.4, label="its own analysis")
    obj = iv27["append_hop3_noverdict_control"]; con = iv27["append_matched_concern_control"]
    base = next(c["E"] for c in cv if c["sentence"] == obj["cut_before_sentence"] - 1)
    x_obj, x_con = xs[-1] + 2.5, xs[-1] + 5.0
    ax2.scatter([x_obj], [obj["E"]], marker="D", s=170, color="#b91c1c", zorder=5, edgecolors="k", linewidths=0.6)
    ax2.scatter([x_con], [con["E"]], marker="D", s=170, color="#6b7280", zorder=5, edgecolors="k", linewidths=0.6)
    ax2.annotate(f"objective sentence appended\n(no verdict word)   {obj['E']:.1f}", (x_obj, obj["E"]), xytext=(x_obj - 0.6, obj["E"] + 1.3), fontsize=11.5, ha="center", color="#b91c1c",
                 arrowprops=dict(arrowstyle="-", lw=0.7, color="#b91c1c"))
    ax2.annotate(f"matched concern text,\nno objective   {con['E']:.1f}", (x_con, con["E"]), xytext=(x_con - 0.2, con["E"] + 2.2), fontsize=11.5, ha="center", color="#374151",
                 arrowprops=dict(arrowstyle="-", lw=0.7, color="#6b7280"))
    ax2.axhline(base, color="#9ca3af", lw=1, ls=":"); ax2.text(4.3, base + 0.45, f"baseline {base:.2f}", fontsize=10.5, color="#6b7280", ha="left")
    ax2.axvline(xs[-1] + 1.0, color="#d1d5db", lw=0.8); ax2.text(xs[-1] + 1.2, 9.7, f"appended after sentence {obj['cut_before_sentence'] - 1}\n({obj['appended_tokens']} vs {con['appended_tokens']} tokens)", fontsize=10.5, color="#6b7280", va="top")
    ax2.set_title("#27  benign edit (emitted integer 1)", fontsize=13)
    ax2.set_xlabel("sentences fed  |  appended text"); ax2.set_ylim(-0.3, 10.3); ax2.set_xlim(-0.5, x_con + 1.2)
    ax2.set_xticks(list(range(0, xs[-1] + 1, 2))); ax2.spines[["top", "right"]].set_visible(False)

    fig.suptitle("Forced digit readout after the monitor's own chain of thought, trajectory 30e71655 (n = 1 trajectory)", fontsize=13)
    fig.tight_layout()
    fp = C.fig_path("p12_hop_truncation_2panel.png"); fig.savefig(fp, dpi=150); plt.close(fig); print("figure ->", fp.relative_to(C.REPO))


if __name__ == "__main__":
    main()
