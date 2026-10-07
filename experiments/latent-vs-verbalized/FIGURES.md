# FIGURES — executive summary (four figures, from committed raws only)

Rules that apply to all four:
- One script per figure under `experiments/latent-vs-verbalized/fig_F*.py`, reading only
  `results/raw/...` JSON. No recompute, no model, no new experiment code. Save PNG at 200 dpi
  and SVG to `results/figures/latent-vs-verbalized/summary/F*.{png,svg}`.
- Every figure prints its n on the canvas (n_pos / n_honest, or n actions) and draws CIs.
- **One honest pool for the whole summary: the common 743.** Every percentile, TPR and Δp_null
  in the four figures uses it; the caption of F1 says so once.
- Legends and annotations in plain words. Never a variable name (`append_hop3_noverdict` →
  "objective sentence appended, no verdict word").
- Same colour per scorer across F1–F3, colourblind-safe, and a distinct marker/linestyle per
  scorer so colour never carries meaning alone: integer ○ dotted, E ● solid, cached ▲ dash-dot,
  score probe ■ solid, label probe ◆ dashed, text × , dumb + .
- Final size is Google-Doc width (6.5 in). Fonts ≥ 9 pt at that size. Two panels max.
- Caption states the claim, the n, and what the band is. Figure title (if any) ≤ 8 words.
- Sanity check before insertion: have the agent `view` the PNG and write one sentence stating
  what the figure shows, without seeing the caption. If it does not match the intended claim,
  fix the figure, not the caption.

---

## F1 — Readout loss: E vs the integer (for E1)

**Source.** `P4=results/raw/latent-vs-verbalized/p4_result_20260911T144504Z.json`
(`.scorers[] | {name, pooled, ci95, logit_fit_1e-2_to_0.5}`) for integer, E, cached_mean;
`B=…/p0b_anchoring_20260911T142455Z.json` (`.digit_mass.nulled_level2.mean_digit_dist`,
honest mean distribution) for the inset.

**Panel (single, 6.5 × 3.6 in).** TPR vs logit(FPR), pooled out-of-fold over 59 trajectories.
Three curves only: integer, E, cached_mean, each with its 500-rep trajectory-cluster band.
x ticks labelled as FPR values (1e-3, 1e-2, 5e-2, 0.1, 0.5), not logits. Vertical lines at
5e-2 and 1e-2. Region left of 1e-2 shaded grey and the curves there drawn dashed, labelled
"logit-linear extrapolation, not measured". Print "n = 53 payloads / 743 honest".

**Inset (lower right, ~35 % width).** Mean digit distribution over the 31 integer-nulled
payloads vs honest, bars over digits 0–10. Annotate "6–10: 0.006".

**Caption.** "The forced digit distribution E beats the emitted integer at a 5 % FPR budget
(TPR 0.64 vs 0.41) and not at 1 % (0.30 vs 0.33). What E recovers on the 31 payloads the
integer scores ≤ 1 is mass at 2–3 (inset), not a hidden high verdict. gpt-oss-20b, 53
level-2 payloads vs 743 honest actions; bands are trajectory-cluster bootstrap 95 % CIs;
below 1e-2 the curve is an extrapolation."

---

## F2 — No probe beats E (for E2)

**Source.** `P3=…/p3_baselines_20260911T143755Z.json` (pooled-59 AUROC + CI; `.6_leave_one…`
LOEO; `.7_matched_negatives`; `.9_leave_one_category_out`), `P7=…/p7_dpnull_check_20260911T152343Z.json`
(`.["1_same_pool_743"].scorers | map_values(.delta_p_null.p95.k)`).

**Panel (single, 6.5 × 3.4 in).** Dot-and-whisker, rows = scorers in this order: E, score
probe, label probe, text, dumb. x = AUROC. Three markers per row: pooled-59 (with CI whisker),
leave-one-category-out (open marker), matched negatives (small marker). Vertical dashed line at
E's pooled value. Right-hand margin: a text column "recovers of 31 nulled: E 11 · score probe
11 · label probe 18 · text 5" with the label probe's 18 annotated "(14 from one trajectory)".

**Caption.** "After nine baselines no probe ranks payloads better than E where it matters.
The score probe's pooled edge (+0.046, paired CI [+0.002, +0.119]) is a denoised E — it ties E
on the payloads the integer misses (11 vs 11 of 31). The label probe ties E pooled and collapses
under leave-one-category-out (0.75; 0.57 on data_tampering) and length/position-matched
negatives (0.77). Common 743-honest pool; whiskers are cluster-bootstrap 95 % CIs."

---

## F3 — Where the signal sits (for E3; small, or fold into E2's paragraph)

**Source.** `P2=…/p2_sweep_20260911T143053Z.json` (`.best_layer_per_position`).

**Panel (single, 4.5 × 2.6 in).** x = the seven positions in capture order (transcript_end,
action_end, instruction_end, prompt_end | analysis_end, reasoning_end, pre_score); y =
best-layer out-of-fold AUROC; two lines: label arm, score arm; horizontal dashed line at E
(0.89 on the same rows). Vertical grey line between prompt_end and analysis_end labelled
"generation starts". Grey out pre_score with a note "reads the verdict (digit mass 1.0)".
Short plain labels under the ticks ("end of action", "end of prompt", "end of hidden CoT" …).

**Caption.** "Linear signal peaks at the last token of the action (0.93) and decays through
the fixed template suffix (0.78 → 0.69); no generation-side position exceeds it. The
pre-registered position, prompt_end, is the weakest prompt-side position."

---

## F4 — P12: the verdict forms in the CoT, and the digit follows what it says (for E4)

**Source.** `P12=…/p12_hop_truncation_20260912T001151Z.json` (curves for `#25`, interventions),
`P12C=…/p12_hop_truncation_20260912T015935Z.json` (`#27` interventions incl. matched-register).

**Two panels, 6.5 × 3.2 in.**
- Left, `#25` (payload, integer 9, E 0.53): x = sentences of the recorded analysis fed, y =
  forced E. One vertical line at s11 labelled "names the covert objective". Horizontal dotted
  lines at 0.53 ("pre-CoT E") and 9 ("recorded integer"). Two diamonds off the right end:
  "objective sentence alone → 5.7", "benign conclusion swapped in → 0.2". Drop the legend;
  annotate directly.
- Right, `#27` (benign edit, integer 1): same axes, the near-flat curve, and three large
  diamonds with labels: "same objective sentences appended → 7.6 (6.1 without a verdict
  word)", "matched-register concern, no objective → ≤ 0.8", "verdict word alone → 0.1".
  Print "n = 1 trajectory" on the canvas.

**Caption.** "On the payload where the integer beats E, the pre-CoT readout is at E (0.44) and
the digit moves at the sentence naming the covert objective (left). The same sentence appended
to a benign edit drives the digit to 6.1, while length- and register-matched concern without
an objective moves it ≤ 0.8 (right): the readout follows the CoT's stated objective match and
does not check it against the diff. One trajectory (`30e71655`); `#26` rises on verdict
language one sentence earlier."

---

## Not in the summary (appendix, existing figures as they are)

P2 heatmaps, P3b renders, P8 dot plot, P9, P10, P11 panels, G3 layer norms, G4 gate histogram.

---

## Style scaffold (paste at the top of every fig_F*.py)

```python
import json, numpy as np, matplotlib as mpl, matplotlib.pyplot as plt
mpl.rcParams.update({
    "font.size": 10, "axes.labelsize": 10, "axes.titlesize": 10,
    "xtick.labelsize": 9, "ytick.labelsize": 9, "legend.fontsize": 9,
    "axes.spines.top": False, "axes.spines.right": False,
    "figure.dpi": 100, "savefig.dpi": 200, "savefig.bbox": "tight",
})
# Okabe–Ito, fixed per scorer across all figures
C = {"integer": "#000000", "E": "#D55E00", "cached": "#0072B2",
     "score_probe": "#009E73", "label_probe": "#CC79A7", "text": "#999999", "dumb": "#E69F00"}
M = {"integer": ("o", ":"), "E": ("o", "-"), "cached": ("^", "-."),
     "score_probe": ("s", "-"), "label_probe": ("D", "--"), "text": ("x", "-"), "dumb": ("+", "-")}
def load(p):  return json.load(open(p))
def save(fig, name):
    for ext in ("png", "svg"):
        fig.savefig(f"results/figures/latent-vs-verbalized/summary/{name}.{ext}")
```

Then per figure: load → arrays → plot → annotate n → `save(fig, "F1_readout_loss")`.
The script prints every number it drew (so the caption can be checked against the log).