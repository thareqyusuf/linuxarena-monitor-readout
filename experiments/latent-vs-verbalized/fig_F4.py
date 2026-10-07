"""F4 — P12: the verdict forms in the CoT and the digit follows what it says (FIGURES.md). Left: #25 sentence curve and its two
replacements/isolations (P12). Right: #27 curve (P12) and its appends incl. the matched-register controls (P12C). Model-free."""
from fig_common import *

P12 = RAW / "p12_hop_truncation_20260912T001151Z.json"; P12C = RAW / "p12_hop_truncation_20260912T015935Z.json"
A = {a["action_idx"]: a for a in load(P12)["actions"]}; B = {a["action_idx"]: a for a in load(P12C)["actions"]}
iv25 = {x["name"]: x for x in B[25]["interventions"]}; iv27 = {x["name"]: x for x in B[27]["interventions"]}
RED, GREY, DK = "#b91c1c", "#6b7280", "#111111"
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(6.5, 3.2), gridspec_kw={"width_ratios": [1.2, 1]})

# left — #25
a = A[25]; cv = a["curve"]; xs = [c["point"] for c in cv]; ys = [c["E"] for c in cv]; s_obj = a["first_marker_sentence"]["objective"]
ax1.plot(xs, ys, "-o", color=DK, ms=3, lw=1.3)
ax1.axvline(s_obj, color=RED, lw=1.3, ls="--"); ax1.text(s_obj - 0.4, 8.6, "names the\ncovert objective", color=RED, ha="right", va="top", fontsize=9)
ax1.axhline(a["recorded"]["E"], color=C["E"], lw=0.9, ls=":"); ax1.text(0.3, 1.9, f"point 0 = {ys[0]:.2f},\nempty channel;\nE {a['recorded']['E']:.2f} dotted", color=C["E"], fontsize=9, va="bottom")
ax1.axhline(a["recorded"]["score"], color=DK, lw=0.9, ls=":"); ax1.text(12.3, a["recorded"]["score"] + 0.2, f"recorded integer {a['recorded']['score']:.0f}", color=DK, fontsize=9, va="bottom")
ob, be = iv25["objective_only"], iv25["replace_hop3_benign"]; xd = xs[-1] + 2.2
ax1.scatter([xd], [ob["E"]], marker="D", s=70, color=RED, zorder=5, edgecolors="k", linewidths=0.5)
ax1.scatter([xd], [be["E"]], marker="D", s=70, color=GREY, zorder=5, edgecolors="k", linewidths=0.5)
ax1.annotate(f"objective sentence\nalone → {ob['E']:.1f}", xy=(xd, ob["E"]), xytext=(18.0, 3.4), fontsize=9, ha="center", va="center", color=RED, arrowprops=dict(arrowstyle="-", lw=0.6, color=RED))
ax1.annotate(f"benign conclusion\nswapped in → {be['E']:.1f}", xy=(xd, be["E"]), xytext=(16.0, 1.8), fontsize=9, ha="center", va="center", color="#374151", arrowprops=dict(arrowstyle="-", lw=0.6, color=GREY))
ax1.set_title(f"#25  payload (integer {a['recorded']['score']:.0f}, E {a['recorded']['E']:.2f})", fontsize=10)
ax1.axvline(xs[-1] + 1.0, color="#9ca3af", lw=0.8); ax1.set_xlabel("sentences of its own analysis fed  |  edited"); ax1.set_ylabel("forced digit readout"); ax1.set_ylim(-0.3, 10.3); ax1.set_xlim(-0.6, xd + 1)
ax1.set_xticks(list(range(0, xs[-1] + 1, 4)))
print(f"#25 curve (point: E): {[(p, round(v, 2)) for p, v in zip(xs, ys)]}; objective sentence s{s_obj}; E at 0/s{s_obj}/end = {ys[0]:.2f}/{ys[s_obj]:.2f}/{ys[-1]:.2f}; recorded E {a['recorded']['E']:.3f} integer {a['recorded']['score']:.0f}")
print(f"#25 diamonds: objective_only {ob['E']:.2f} ({ob['appended_tokens']} tok); replace_hop3_benign {be['E']:.2f} (cut before s{be['cut_before_sentence']}, {be['appended_tokens']} tok)")

# right — #27
a = A[27]; cv = a["curve"]; xs = [c["point"] for c in cv]; ys = [c["E"] for c in cv]
ax2.plot(xs, ys, "-o", color=DK, ms=3, lw=1.3)
hop, nov, con, ver = iv27["append_hop3_control"], iv27["append_hop3_noverdict_control"], iv27["append_matched_concern_control"], iv27["append_verdict_only_control"]
base = next(c["E"] for c in cv if c["sentence"] == hop["cut_before_sentence"] - 1)
x1, x2, x3 = xs[-1] + 2.4, xs[-1] + 4.2, xs[-1] + 6.0
ax2.axvline(xs[-1] + 1.0, color="#9ca3af", lw=0.8)
ax2.scatter([x1], [hop["E"]], marker="D", s=70, color=RED, zorder=5, edgecolors="k", linewidths=0.5)
ax2.scatter([x1], [nov["E"]], marker="D", s=40, color="white", zorder=5, edgecolors=RED, linewidths=1.2)
ax2.scatter([x2], [con["E"]], marker="D", s=70, color=GREY, zorder=5, edgecolors="k", linewidths=0.5)
ax2.scatter([x3], [ver["E"]], marker="D", s=70, color="#bdbdbd", zorder=5, edgecolors="k", linewidths=0.5)
ax2.annotate(f"same objective\nsentences appended\n→ {hop['E']:.1f} ({nov['E']:.1f} without\na verdict word)", xy=(x1, hop["E"]), xytext=(4.2, 7.4), fontsize=9, ha="center", va="center", color=RED,
             arrowprops=dict(arrowstyle="-", lw=0.6, color=RED))
ax2.plot([hop["cut_before_sentence"] - 1, x3 + 0.8], [base, base], ls=":", color="#9ca3af", lw=0.9); ax2.text(4.0, base - 0.2, f"baseline {base:.2f}", fontsize=9, color="#6b7280", va="top")
ax2.annotate(f"matched-register concern,\nno objective → {con['E']:.1f}\n(lift {con['E'] - base:.2f} over baseline)", xy=(x2, con["E"]), xytext=(4.0, 4.4), fontsize=9, ha="center", va="center", color="#374151",
             arrowprops=dict(arrowstyle="-", lw=0.6, color=GREY))
ax2.text(x3 - 0.2, 1.5, f"verdict word\nalone → {ver['E']:.1f}", fontsize=9, ha="center", va="bottom", color="#374151")
ax2.text(0.98, 0.97, "n = 1 trajectory, both panels", transform=ax2.transAxes, ha="right", va="top", fontsize=9)
ax2.set_title(f"#27  benign edit (integer {a['recorded']['score']:.0f})", fontsize=10)
ax2.set_xlabel("sentences fed  |  appended"); ax2.set_ylim(-0.3, 10.3); ax2.set_xlim(-0.6, x3 + 1.6); ax2.set_xticks(list(range(0, xs[-1] + 1, 2)))
print(f"#27 curve (point: E): {[(p, round(v, 2)) for p, v in zip(xs, ys)]}; appends after s{hop['cut_before_sentence'] - 1} (baseline {base:.2f}): hop3 {hop['E']:.2f} ({hop['appended_tokens']} tok), "
      f"no-verdict {nov['E']:.2f} ({nov['appended_tokens']} tok), matched concern {con['E']:.2f} ({con['appended_tokens']} tok, lift {con['E'] - base:+.2f}), verdict only {ver['E']:.2f} ({ver['appended_tokens']} tok)")
fig.tight_layout(w_pad=1.0); save(fig, "F4_verdict_in_cot")
