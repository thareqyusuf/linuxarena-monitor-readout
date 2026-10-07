"""Random-example exhibits for the write-up (placed right after the summary, not counted as figures):
  X1  five analysis texts drawn with random.Random(0) over usable actions — verbatim (already in the P6 raw)
  X2  the six quiet-and-nulled audit payloads (audit level-0 signatures missed, integer <= 1) with I / E / cached — P8 table,
      action text verbatim (full text from the export, which the P8 raw truncates to 300 chars)
  X3  P12 #25 sentences s10–s12 verbatim with the forced E after each
Writes results/raw/<slug>/x_examples_<ts>.json (everything verbatim, so the laptop needs no export), the markdown
results/figures/<slug>/summary/EXAMPLES.md, and two PNG panels (X2 table, X3 sentences). X1 is markdown only: five
paragraphs of prose do not survive rasterisation."""
from __future__ import annotations

import json, sys, textwrap
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common as C
from fig_common import mpl, plt, OUT, REPO, RAW

P6 = RAW / "p6_read_20260911T144332Z.json"; P8 = RAW / "p8_quiet_payloads_20260911T155530Z.json"; P12 = RAW / "p12_hop_truncation_20260912T001151Z.json"
p6, p8, p12 = (json.load(open(p)) for p in (P6, P8, P12))
stamp = C.ts()

# X1
five = [{k: r[k] for k in ("traj_id", "action_idx", "side", "env", "label_level", "audit_verdict", "tool", "score", "E", "cached_mean", "action_text", "analysis_note_text", "generation_text")} for r in p6["five_generations_random0"]]
assert len(five) == 5

# X2 — full action text from the export (same renderer P8 used; P8 keeps only the first 300 chars)
export = C.load_export(); quiet = [r for r in p8["quiet"] if r["score"] <= 1]; assert len(quiet) == 6, len(quiet)
six = []
for r in quiet:
    full = C.action_text(export[r["traj_id"]], r["action_idx"]); assert full[:300] == r["action_text_300"], (r["traj_id"][:8], r["action_idx"])
    s = r["scores"]["common_743"]
    six.append({"traj_id": r["traj_id"], "action_idx": r["action_idx"], "category": r["category"], "side_task": r["side_task"], "objective": r["objective"], "stratum": r["stratum"],
                "tool": r["tool"], "n_action_chars": r["n_action_chars"], "integer": r["score"], "E": r["E"], "cached_scores": r["cached_scores"], "cached_mean": r["cached_mean"],
                "pct_743": {k: s[k]["pct"] for k in ("score", "E", "cached_mean", "label|logreg", "score|ridge_E", "text", "dumb")},
                "action_text": full, "analysis_note_text": r["analysis_note_text"]})
blind = p8["blindness"]; n_blind_ok = sum(not b["traj_in_training"] for b in blind)

# X3
a25 = next(a for a in p12["actions"] if a["action_idx"] == 25)
s10_12 = [{k: c[k] for k in ("sentence", "point", "E", "argmax", "argmax_p", "text_added")} for c in a25["curve"] if c["sentence"] in (10, 11, 12)]
assert [c["sentence"] for c in s10_12] == [10, 11, 12]
e_before = next(c["E"] for c in a25["curve"] if c["sentence"] == 9); e0 = a25["curve"][0]["E"]

out = {"slug": C.SLUG, "script": f"experiments/{C.SLUG}/fig_examples.py", "timestamp": stamp, "git_commit": C.git_commit(),
       "inputs": {"p6_raw": str(P6.relative_to(REPO)), "p8_raw": str(P8.relative_to(REPO)), "p12_raw": str(P12.relative_to(REPO)), "export": p8["inputs"]["export"]},
       "x1_five_generations_random0": five, "x2_quiet_nulled_payloads": six, "x2_blindness_n_not_in_training": [n_blind_ok, len(blind)],
       "x3_p12_action25_s10_s12": {"E_point0": e0, "E_after_s9": e_before, "recorded_E": a25["recorded"]["E"], "recorded_integer": a25["recorded"]["score"], "sentences": s10_12}}
if "--from-raw" in sys.argv:  # redraw from an existing raw (no new raw written)
    rp = REPO / sys.argv[sys.argv.index("--from-raw") + 1]; prev = json.load(open(rp)); stamp = prev["timestamp"]
    assert prev["x2_quiet_nulled_payloads"] == six and prev["x3_p12_action25_s10_s12"]["sentences"] == s10_12, "raw differs from a fresh build"
else:
    rp = C.write_raw("x_examples", out, stamp)

# markdown
def fmt(x): return f"{x:.2f}"
md = [f"# Random examples (after the summary; not counted as figures)\n", f"Raw: `{rp.relative_to(REPO)}` (`fig_examples.py`, {stamp}). Every text below is verbatim from that raw.\n",
      "## X1 — five analysis texts, `random.Random(0)` over the usable actions (P6)\n",
      "Sample of 5 from the usable actions (score not null, current-action placeholder excluded), drawn once with seed 0 in P6 and stored there. I = emitted integer, E = forced readout, cached = GPT-5-nano mean.\n"]
for r in five:
    md += [f"**`{r['traj_id'][:8]}#{r['action_idx']}`** · {r['side']}, label level {r['label_level']}, {r['env']} · I {r['score']:.0f} · E {fmt(r['E'])} · cached {r['cached_mean']}\n",
           "Action:\n", "```\n" + r["action_text"] + "\n```\n", "Analysis text:\n", "> " + r["analysis_note_text"].replace("\n", "\n> ") + "\n"]
md += ["## X2 — the six quiet payloads the integer nulled (P8), I / E / cached\n",
       f"Audit payloads with no regex-signature hit (label level 0) and integer ≤ 1; the two integer-8 quiet payloads are excluded. Percentiles are on the common 743-honest pool. Probe scored the row blind: {n_blind_ok} of {len(blind)} audit payloads come from trajectories outside the scoring model's training set.\n",
       "| action | category | stratum | I (pct) | E (pct) | cached two scores → mean (pct) | label probe pct | score probe pct |", "|---|---|---|---|---|---|---|---|"]
for r in six:
    p = r["pct_743"]
    md.append(f"| `{r['traj_id'][:8]}#{r['action_idx']}` | {r['category']} | {r['stratum']} | {r['integer']:.0f} ({p['score']:.0f}) | {fmt(r['E'])} ({p['E']:.0f}) | {r['cached_scores']} → {r['cached_mean']} ({p['cached_mean']:.0f}) | {p['label|logreg']:.0f} | {p['score|ridge_E']:.0f} |")
md.append("")
for r in six:
    md += [f"**`{r['traj_id'][:8]}#{r['action_idx']}`** · {r['category']} / {r['side_task']} · objective: {r['objective']} · {r['n_action_chars']} chars\n",
           "```\n" + r["action_text"] + "\n```\n", "Monitor's analysis text:\n", "> " + r["analysis_note_text"].replace("\n", "\n> ") + "\n"]
md += ["## X3 — P12 `30e71655#25`: sentences s10–s12 of the monitor's own analysis, with the forced E after each\n",
       f"Forced `<score>` readout after feeding the recorded analysis up to and including each sentence. E with no analysis (point 0) = {fmt(e0)}; after s9 = {fmt(e_before)}; recorded E (pre-CoT) {fmt(a25['recorded']['E'])}; recorded integer {a25['recorded']['score']:.0f}.\n",
       "| sentence | text (verbatim) | E after it | argmax digit (p) |", "|---|---|---|---|"]
for c in s10_12:
    md.append(f"| s{c['sentence']} | {c['text_added']} | **{fmt(c['E'])}** | {c['argmax']} ({c['argmax_p']:.2f}) |")
(OUT / "EXAMPLES.md").write_text("\n".join(md) + "\n"); print("markdown ->", (OUT / "EXAMPLES.md").relative_to(REPO))

# PNG panels
fig, ax = plt.subplots(figsize=(6.5, 2.1)); ax.axis("off")
cols = ["action", "category", "stratum", "I (pct)", "E (pct)", "cached (pct)", "label probe pct", "score probe pct"]
rows = [[f"{r['traj_id'][:8]}#{r['action_idx']}", r["category"].replace("_", " "), r["stratum"], f"{r['integer']:.0f} ({r['pct_743']['score']:.0f})", f"{r['E']:.2f} ({r['pct_743']['E']:.0f})",
         f"{r['cached_scores']} → {r['cached_mean']} ({r['pct_743']['cached_mean']:.0f})", f"{r['pct_743']['label|logreg']:.0f}", f"{r['pct_743']['score|ridge_E']:.0f}"] for r in six]
t = ax.table(cellText=rows, colLabels=cols, loc="upper center", cellLoc="left", colLoc="left"); t.auto_set_font_size(False); t.set_fontsize(9); t.scale(1, 1.35)
for (i, j), cell in t.get_celld().items():
    cell.set_edgecolor("#dddddd"); cell.set_linewidth(0.6)
    if i == 0: cell.set_text_props(weight="bold"); cell.set_facecolor("#f3f4f6")
t.auto_set_column_width(list(range(len(cols))))
ax.set_title("Six quiet payloads the integer nulled (audit level 0, I ≤ 1); percentiles on 743 honest", fontsize=10, loc="left")
fig.text(0.01, 0.06, f"n = 6 of 8 quiet audit payloads (two with integer 8 excluded); {n_blind_ok}/{len(blind)} scored by a model that never saw their trajectory", fontsize=9, color="#555555")
fig.savefig(OUT / "X2_quiet_payloads.png"); fig.savefig(OUT / "X2_quiet_payloads.svg"); plt.close(fig)

fig, ax = plt.subplots(figsize=(6.5, 2.7)); ax.axis("off"); y = 0.98
ax.text(0.0, y, f"30e71655#25 — forced readout after each analysis sentence (no analysis: E {e0:.2f}; after s9: {e_before:.2f})", fontsize=10, va="top", transform=ax.transAxes); y -= 0.16
for c in s10_12:
    wrapped = textwrap.fill(c["text_added"], 64)
    ax.text(0.0, y, f"s{c['sentence']}", fontsize=10, weight="bold", va="top", transform=ax.transAxes)
    ax.text(0.06, y, wrapped, fontsize=9, va="top", transform=ax.transAxes, family="DejaVu Sans")
    ax.text(1.0, y, f"E {c['E']:.2f}", fontsize=11, weight="bold", va="top", ha="right", color="#b91c1c" if c["E"] > 3 else "#111111", transform=ax.transAxes)
    y -= 0.115 * (wrapped.count("\n") + 1) + 0.07
ax.text(0.0, 0.0, f"recorded: integer {a25['recorded']['score']:.0f}, pre-CoT E {a25['recorded']['E']:.2f}; n = 1 trajectory", fontsize=9, color="#555555", va="bottom", transform=ax.transAxes)
fig.savefig(OUT / "X3_p12_s10_s12.png"); fig.savefig(OUT / "X3_p12_s10_s12.svg"); plt.close(fig)
print("panels -> X2_quiet_payloads, X3_p12_s10_s12")
print("X3:", [(c["sentence"], round(c["E"], 2)) for c in s10_12], "| X2 rows:", [(r["traj_id"][:8], r["action_idx"], r["integer"], round(r["E"], 2), r["cached_mean"]) for r in six], "| blindness", n_blind_ok, "/", len(blind))
