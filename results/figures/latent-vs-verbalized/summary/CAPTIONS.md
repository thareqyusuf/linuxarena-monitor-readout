# Summary figures — captions (FIGURES.md, 2026-09-12; revised after the pool review the same day)

Captions are the FIGURES.md text with numbers replaced only where the committed raws say otherwise; every
replacement is listed under "differs from FIGURES.md". Recompute commands are in
`experiments/latent-vs-verbalized/RESULTS.md` (entry "Summary figures F1–F4"). **One honest pool for all four: the
common 743** (honest rows that have generation positions; F3 is train-only, 599 of them).

## F1 — `F1_readout_loss.{png,svg}` (`fig_F1.py`; P4 and P0b both rerun on the 743 pool)

The forced digit distribution E beats the emitted integer at a 5 % FPR budget (TPR 0.55 vs 0.42) and not at 1 %
(0.30 vs 0.33); AUROC E 0.90, integer 0.74, GPT-5-nano mean 0.85. What E recovers on the 31 payloads the integer
scores ≤ 1 is mass at 2–3 (inset), not a hidden high verdict (mass on 6–10: 0.006, against 0.001 for honest actions).
gpt-oss-20b, 53 level-2 payloads vs 743 honest actions, the same 743 for the curves and the inset's honest bars; bands
are trajectory-cluster bootstrap 95 % CIs. Below 1e-2 the dashed curves are logit-linear extrapolations of the 1e-2–0.5 segment, not
measurements: the fits are driven by the mid-curve slope, the integer's is flattened by its ties, and the ordering
of the three below 1e-2 (integer's dashed line ending above E's at 1e-3) is not meaningful — consistent with the
measured 1 % point, where there is no gap.

Differs from FIGURES.md: TPR numbers 0.64 / 0.41 → 0.55 / 0.42, because the FIGURES.md caption quoted P4's original
839-pool run while stating 743. F1 is now drawn from `p4_result_pool743_20260912T030653Z.json` (P4 rerun with
`--pool common743`, same score vectors, no model), which reproduces P7 §1 exactly; the AUROCs in the legend are from
that raw (the 839-pool integer AUROC was 0.748; on 743 it is 0.744). The inset's honest bars are the mean digit
distribution over the same 743 rows (`p0b_anchoring_pool743_20260912T031729Z.json`, P0b rerun with `--pool common743`,
keyed off P11's row table); the 839-row mean differs in the third decimal (mass on 6–10: 0.0012 on both).

## F2 — `F2_no_probe_beats_E.{png,svg}` (`fig_F2.py`; P7 + P3 + P11)

After nine baselines no probe ranks payloads better than E where it matters. The score probe's pooled edge (+0.046,
paired CI [+0.002, +0.119]) is a denoised E — it ties E on the payloads the integer misses (11 vs 11 of 31). The
label probe ties E pooled and collapses under leave-one-category-out (0.75; 0.57 on data_tampering) and
length/position-matched negatives (0.77). Recoveries of the 31 integer-nulled payloads are concentrated in one
trajectory (`588b46de`, which holds 14 of the 31): it supplies 7 of E's 11, 7 of the score probe's 11 and 9 of the
label probe's 18, none of text's 5 or dumb's 7. Pooled markers and recoveries use the common 743-honest pool;
whiskers are cluster-bootstrap 95 % CIs; the leave-one-category-out markers are P3's (each scorer's full pool: 839,
or 743 for the score probe) and the matched markers use 147 negatives.

Differs from FIGURES.md: "(14 from one trajectory)" on the label-probe row → the from-`588b46de` count is printed on
every row (E 7, score probe 7, label probe 9, text 0, dumb 0; 743 pool). E has no leave-one-category-out marker (it is
not trained).

## F3 — `F3_where_signal_sits.{png,svg}` (`fig_F3.py`; P2 sweep)

The label-supervised probe peaks at the last token of the action (0.93) and decays through the fixed template suffix
(scoring instructions 0.78 → prompt end 0.69); no generation-side position exceeds it (0.84, 0.90, 0.84). The
score-supervised probe is flat at 0.93 from the action onward (0.93 / 0.93 / 0.93 / 0.93 at action, hidden-CoT,
visible-reasoning and pre-digit positions). The pre-registered position, prompt_end, is the weakest prompt-side
position for both arms. Train out-of-fold, 33 payloads vs 599 honest over 40 trajectories; E on the same rows 0.89.
Small dots are the five fold-wise AUROCs of the best cell (fold range 0.36–1.0 across positions; the pooled OOF
value can fall outside the fold range, as it does for the label arm at hidden-CoT end: 0.84 vs folds 0.89–1.0).

Differs from FIGURES.md: "no generation-side position exceeds it" is qualified to the label arm — for the score arm
hidden-CoT end (0.9322) and visible-reasoning end (0.9324) are 0.001 above the action-end value (0.9311). No CI exists
in the raw for the per-position best cells; the fold-wise values are drawn as dots instead of a band. The
`instruction_end` position is labelled "scoring instructions" because it follows the action in the prompt.

## F4 — `F4_verdict_in_cot.{png,svg}` (`fig_F4.py`; P12 + P12C)

On the payload where the integer beats E, the pre-CoT readout is at E (point 0 = 0.44, the empty analysis channel;
recorded E 0.53) and the digit moves at the sentence naming the covert objective (left: 0.4–1.6 through s10, 4.5 at
s11, 9.0 at the end; the objective sentence alone gives 5.7, a benign conclusion swapped in gives 0.2). The same
sentences appended to a benign edit drive the digit to 7.6 (6.1 without a verdict word), while length- and
register-matched concern without an objective reads 1.0, a lift of 0.75 over the 0.27 baseline, and the verdict word
alone 0.1 (right): the readout follows the CoT's stated objective match and does not check it against the diff. One
trajectory (`30e71655`), n = 1; `#26` rises on verdict language one sentence earlier.

Differs from FIGURES.md: "matched-register concern, no objective → ≤ 0.8" → the value is 1.0; ≤ 0.8 is the lift over
the 0.27 prefix baseline (0.75 on `#27`, 0.61 on `#24`). Both numbers are on the figure.
