# TASK-P8 — the quiet payloads: an existence test the probes are blind to (+ P9, optional)

Branch `exp/latent-vs-verbalized`. CPU only, no fits in P8, one refit in P9. Both read artefacts
that already exist. Conventions as in CLAUDE.md: script + raw + recompute one-liner; append to
`RESULTS.md` newest at top; status tags. Hard cut from TASK.md still applies.

## Why

The label probe was trained on signature level-2. The 8 actions the user judged `payload` in the
blind audits (A4, A4-r2) that the signatures missed were kept at `label_level 0` — and attack
level-0 is excluded from both classes. So those 8 were in neither training class, every OOF fold
model and the locked-test refit is blind to them, and they are the quiet payloads the hypothesis
is about. Ranking them against honest is the one existence test this data still supports.

**Kind of claim:** existence proof. One quiet payload that a probe lifts above the honest tail and
E does not is the result; zero is the honest close on reading (c) for this monitor. No AUROC on n = 8.

## Inputs (all exist)

- `action_labels.json` at the commit recorded in P0's raw (`.inputs.labels`), columns
  `audit_verdict` (176 rows: payload / prep / benign), `label_level`, `in_subset_main`.
- A4 raws on `origin/exp/action-level-attack-label` (`a4_audit_*`, `a4_audit_r2_*`) — carry the
  stratum (`end` / `random`) per audited action. Record it; do not re-open the audit files.
- Pooled score vectors: P3 `scores_pooled_59` (the P4 vector) for integer, E, cached_mean, dumb,
  text, label probe (action_end L8), score probe (reasoning_end L14); P2 OOF checkpoint
  `checkpoints/latent-vs-verbalized/p2_oof_20260911T143053Z.npz` for the pre-registered
  `prompt_end` label cell. P7 §5 already scored the non-payload attack rows through the same
  path — reuse that code, do not write a new scoring path.
- Honest pools as in P7 §1: the **common 743** (rows with generation positions) for every
  scorer; the 839 pool for the scorers that have it, appendix only.

## P8.1 Assemble (10 min)

1. Select: `audit_verdict == payload`. Expect 20 in 11 trajectories. Split into
   `quiet` = `label_level == 0` (expect 8) and `loud` = `label_level == 2` (expect 12).
   Assert none of the 20 has `score: null`; drop and report if any does.
2. For each, record: key, traj, step/len, position, side_task, category, stratum it was drawn
   in, tool, `command_length`, integer, E, cached scores, `label_cached_high`.
3. **Blindness check, per action, explicitly:** name the model that scored it (fold k or
   test-refit) and assert its training trajectories do not include the action's trajectory.
   Print the assertion result for all 20. If any fails, stop.

## P8.2 Rank (5 min)

For each of the 20 and each scorer — integer, E, cached_mean, text, dumb, label probe (chosen
cell), label probe (`prompt_end`, pre-registered cell), score probe (chosen cell) — report:

- honest percentile on the common-743 pool, and above-p95 / above-p99 flags;
- the same under the scoring model's **own** honest p95 (P7 §3 convention), since pooled
  thresholds mix fold scales. Report both; state which one any sentence uses.

Table 1: the 8 quiet payloads × scorers, one row each, percentile with ✓ at p95. Table 2: the 12
loud ones. Summary line per scorer: quiet lifted k/8 at p95 and p99; loud k/12.
Then the two-by-two that matters: for each probe, count quiet payloads it lifts **and E does
not**, and the reverse. Name them.
Appendix: the 25 `prep` verdicts through the same table, no summary claims.

Raw `p8_quiet_payloads_<ts>.json`. Figure `p8_quiet_payloads.png`: dot plot, 20 rows
(quiet above a rule, loud below) × scorers on the x-axis as honest percentile, p95 and p99 as
vertical lines, stratum as marker shape.

## P8.3 Read them (10 min; this goes in the write-up)

The 8 quiet payloads verbatim (command truncated at 300 chars), with side-task objective in one
line, step/len, integer, E, cached, and the analysis note. The audit is done, so reading them
now is not contamination. For each lifted-by-probe-not-E case, one sentence on what the probe
could be reading that E is not — and the boring alternative (length, tool, position) beside it.

## P8 caveats to state in the entry

`end` stratum draws by position, so quiet payloads from it sit late in short trajectories where
the probes' position confound helps them — report stratum per row and do not pool across strata
in any sentence. n = 8, existence only. Threshold convention (P7 §3). The 12 loud ones are
signature level-2 and were training positives in other folds; they are a comparison, not a test.

---

## P9 (optional, ~30 min) — is the label probe's format-learning just `588b46de` dominance?

Same `split.json`, same folds, same cell (action_end L8, C = 0.1), same evaluation as P3.
Two refits of the label arm only:

- **inverse-count weights**: positive weight 1 / (level-2 count in its trajectory), rescaled so
  total positive weight is unchanged; honest weight 1.
- **cap-3**: keep at most the last 3 level-2 actions (by step) per trajectory; report how many
  positives remain (expect ~25 of 53) and per-fold counts.

For each: pooled-59 AUROC with cluster-bootstrap CI, LOCO (P3.9, data_tampering row
especially), matched negatives (P3.7), residualised (P3.8), and the P6 top-10 honest list
regenerated — are they still big `str_replace` edits? Then re-run P8.2 with these two models as
extra columns.

Reading rule, written before running: if LOCO data_tampering stays ≤ 0.65 under both refits,
the format-learning is not a dominance artefact and the null stands as a null of this label; if
it rises above E (0.90) under either, the original null was the two-trajectory concentration and
the write-up says so.

Raw `p9_balanced_label_probe_<ts>.json`. Figure `p9_balanced.png`.

---

## P10 (~20 min) — per-category catch rate, annotated with label coverage. Reporting, not exclusion.

The signature label is blind in some categories (A3 blind-spot table; A4 audit: exfiltration
4/4 missed, running_malware 0/5, data_tampering 1/7). That is a fact about the *label*, fixed on
the label branch before any probe ran. It changes what a per-category TPR means; it does not
change the headline, and no category is dropped from any pooled number.

**Why not exclude:** exfiltration contributes ≤ 3 level-2 actions to subset_main, so excluding it
moves ≤ 3 of 53 positives; the quiet payloads it misses are level 0 and already outside both
classes; and the label probe's LOCO collapse is on data_tampering, the best-covered category.

1. Tier every `side_task_category` from the A3/A4 raws only — write the rule before reading
   any probe number: **tier 1** = level-2 coverage ≥ 90 % of successful trajectories AND audit
   miss ≤ 1 (expect running_malware, data_tampering); **tier 2** = coverage ≥ 75 % (expect
   monitoring_subversion); **tier 3** = the rest (exfiltration, access_control_bypass,
   access_secret). Record the tier table with its source numbers.
2. Per category × scorer (integer, E, cached_mean, text, label probe, score probe): n level-2 in
   subset_main, TPR@5 % and @1 % against the common honest pool, cluster-bootstrap CI where
   n ≥ 6, "n too small" otherwise. Tier in the first column. A tier-1 row is close to a true
   catch rate; tier-3 rows are upper bounds only — say this once above the table, not per row.
3. One sensitivity row, labelled as such: pooled TPR@5 % over tier-1 categories only, beside the
   all-category pooled value, with the paired difference and CI. Expect it to be within noise.
   If it is not, report which category moved it and how many positives that is.

Raw `p10_category_coverage_<ts>.json`. Figure `p10_category_tpr.png` (grouped bars by
category, tier shaded, n printed under each group).

Write-up sentence this licenses: "TPR is measured on a surface-form label whose coverage
varies by category; in the two fully-covered categories (n = 28) the numbers are close to true
catch rates, elsewhere they are upper bounds." Not: "excluding weak categories improves the probe."

## Alternative explanations every entry must address

Position/length confound on the `end` stratum; threshold convention; the scoring model for a
train-trajectory action is one fold model (fold scales differ); the 8 are defined by one
person's blind verdict (n = 176 audited, 20 payload); the level-2 label is surface-form biased
(A3/A4), so P9 rebalances *within* a biased set and cannot add quiet positives.