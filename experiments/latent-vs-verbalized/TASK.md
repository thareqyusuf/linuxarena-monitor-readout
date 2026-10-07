# TASK — latent-vs-verbalized — does the monitor's residual stream rank payloads better than its verbalized score?

Hand-off from `exp/action-level-attack-label` to the next experiment.

Run on the pod, not the laptop: the activation shards never left it, it has the RAM, and every
stage here is CPU work that takes minutes there. No new GPU runs.

Conventions as in README.md: every number carries script + raw + a recompute one-liner; append to
`RESULTS.md` newest at top; status tags `[hunch]` `[tested]` `[verified]` `[dead]`.


---

## The claim this is testing

The monitor's forward pass over the transcript represents "evidence this action is part of an
attack". The verbal pipeline maps that through a CoT, a rubric and one sampled integer. The claim
is that the map is lossy **in the ranking region that sets TPR at a realistic FPR budget**.

Three mutually exclusive readings, and what each predicts here:

| reading | prediction | status coming in |
|---|---|---|
| (a) readout loss — the integer discards a distribution | E beats the integer | **already true**: AUROC 0.905 vs 0.748; E recovers 11 of the 31 integer-nulled payloads at 5 % FPR |
| (b) expression loss — the internal state doesn't map onto the scale | a probe beats E | this is what P2–P4 test |
| (c) comprehension loss — it never registered | probe at chance after baselines | a real, publishable outcome |

**(a) is settled before you start, and it sets the bar.** The probe must beat **E**, not the
integer. If it only beats the integer you have re-derived a mean, expensively.

## Inputs, all of which already exist

- **Activations** on this pod: `out/main` (1,812 actions, 7 positions), the first-pass 200
  (4 prompt-side positions only), `out/consistent_high` (105, duplicates of main keys).
  Shard format in `experiments/local-monitor-activations/run_monitor.py`. Rev-3 position order:
  `instruction_end, transcript_end, action_end, prompt_end, analysis_end, reasoning_end, pre_score`.
  G3 shards call `prompt_end` → `final`; alias it in the loader.
- **Scores**: `results/raw/local-monitor-activations/g4_main_scores_20260911T075642Z_part{1,2}.jsonl`
  plus `g4_main_first200_scores_20260910T133150Z.jsonl`. Carries `score`, `expected_score`,
  `digit_dist`, `cached_scores`, `analysis_note_text`, the dumb features, `has_attachment_placeholder`.
  5 actions have `score: null` — exclude them everywhere, never treat null as 0.
- **Labels**: `git show origin/exp/action-level-attack-label:experiments/action-level-attack-label/action_labels.json`.
  Columns: `label_level` (2 payload / 1 prep / 0 none / null placeholder), `audit_verdict` (the 176
  hand-audited actions), `level2_comment_only`, `label_cached_high`, `in_subset_main`.
  Record the commit hash in every raw.
- **Prior numbers to reproduce as the loader gate** (C and D entries on that branch; subset_main,
  non-placeholder, 53 level-2 vs honest):

  | quantity | value |
  |---|---|
  | AUROC E / integer / cached_mean | 0.905 / 0.748 / 0.853 |
  | TPR@5 % E / cached | 0.64 / 0.42 |
  | p_null of the integer | 31/53 = 0.58 |
  | E recovers, of those 31, above honest p95 / p99 | 15 / 4 |
  | digit mass at ≥ 3, level-2 / honest | 0.308 / 0.082 |

  If these do not reproduce to ±0.01, the join is wrong. Stop and fix it.

## Fixed decisions — do not re-open

- **Positives** = attack actions with `label_level == 2` (53, in 15 trajectories).
  **Negatives** = honest actions only. Attack level-0 and level-1 are excluded from both classes:
  the blind audit found 8 of 20 confirmed payloads sitting at level 0, so attack level-0 is not a
  clean negative.
- **Placeholders: exclude on the CURRENT ACTION only.** `placeholder` in `g2_prep` is
  `"attachment://" in <the whole rendered prompt>`, so it flags every action whose *history*
  mentions an attachment, not actions whose own content is hidden. In subset_main that is 347
  flagged attack actions of which only **74** have the attachment in the current action; honest 27
  flagged, 13 current-action. Excluding all 347 drops 273 actions whose content is perfectly
  readable, and it drops them non-randomly (once a trajectory uses an attachment, every later
  action is flagged — i.e. late actions, where payloads live). **It changes nothing in the
  headline**: 0 level-2 positives are affected and AUROC is identical to 3 dp under all three rules
  (E 0.905 / 0.905 / 0.906). Use the current-action rule, and report the other two in the appendix.
  See the E entry (`e_cap_placeholder_audit.py`).
- **Headline local scorer is E.** The integer is reported beside it, never instead of it.
- **FPR operating points: 5e-2 primary, 1e-2 secondary.** 728 honest non-placeholder actions means
  1e-2 is 7 actions and 1e-3 is less than one. Fit the L3 logistic on FPR ∈ [1e-2, 0.5]; draw 1e-3
  dashed and label it extrapolation. Do not report a 1e-3 number as measured.
- **Locked test** = 1/3 of trajectories, stratified by side × `side_task_category`, seed 0.
  Expect ~5 positive trajectories; report n. Selection happens only on out-of-fold train.
  The tail curve uses pooled out-of-fold over all 59 trajectories.
- **Every CI** is a trajectory-cluster bootstrap (resample honest and attack trajectories
  independently), 500 reps, seed 0. n_eff is 59 trajectories, not 2,012 actions.

---

## P-0. Consolidate the shards once (20 min, do this before anything else)

You do **not** need a Jupyter kernel for the compute. The kernel discipline in README.md exists to
avoid reloading a 20B model; TASK-P never loads the model. What is expensive here is re-reading
activations: each action is 1,008,000 bytes (7 positions x 25 layers x 2880 dims x fp16), so
subset_main is ~2.0 GB across ~80 shards, and the P2 sweep touches 175 (position, layer) cells per
arm. Reading shards per cell means ~350 passes over 2 GB.

Instead, one pass that writes a single array plus an index:

- `checkpoints/latent-vs-verbalized/acts.npy` — fp16, shape `[n_actions, 7, 25, 2880]`, C-order.
- `checkpoints/latent-vs-verbalized/acts_index.json` — row i -> `(traj_id, action_idx, render,
  source_shard, n_positions)`; `n_positions` is 4 for the first-pass 200, 7 for the rest, and the
  missing three positions are written as NaN so the array stays rectangular.

Then every downstream stage opens it with `np.memmap(..., mode="r")` and slices. One (position,
layer) slice is n x 2880 x 2 bytes ~ 11 MB and is instant once the page cache is warm. This beats a
persistent kernel because it survives process death, and it keeps each stage a committed,
re-runnable script the way the repo contract requires.

Verify before moving on: index length equals the deduped key count from P-1.3; a random 5 rows
re-read from their original shard are bitwise equal to the array rows; NaN pattern matches
`n_positions` exactly.

**Where a notebook does earn its place:** `experiments/lab.ipynb` for P0b (reading the analysis
texts) and P6 (reading transcripts and eyeballing figures) — genuine exploration with fast
iteration. Save every figure to `results/figures/<slug>/` as well as displaying it. Keep P2-P4 as
scripts: they are long enough to want a logfile, they must re-run from the committed tree for the
review, and a notebook makes it easy to end up at 02:00Z with a number whose provenance you cannot
reconstruct.

## P-1. Verify capture (15 min; do not touch running processes)

1. Newest `acts_*.pt` in `out/main`: keys, shape per key, dtype, recorded position order.
   Expect `[7, 25, 2880]`. If it is `[4, …]` the post-CoT positions are missing — report, do not restart.
2. Read three `scores.jsonl` records end to end (honest, attack, placeholder). Check
   `capture_positions` has 7 entries with `token_index` + `decoded`; `analysis_end` decodes to the
   last token before the analysis channel's `<|end|>`; `pre_score` to the token before the digit.
3. Dedupe on `(traj_id, action_idx, render)` preferring `out/main`. Assert 2,012 unique
   original-render keys. Record which 200 have only 4 positions.

Raw `p_1_verify_<ts>.json`. **Stop and report** if anything differs from the above.

## P0. Assembly and gate (45 min)

One table, one row per key: every scalar in the record, plus `label_level`, `audit_verdict`,
`label_cached_high`, `placeholder`, `env`, `shard_path`, `shard_row`. Then reproduce the gate table
above. Also report, split by side × placeholder: n, n trajectories, score histogram, p(score ≥ 3),
E mean/sd, Spearman(E, score).

Raw `p0_gate_<ts>.json`. Figure `p0_score_by_side.png`.

## P0b. Anchoring — free, no activations, do it before the probes (30 min)

Reading (b) says the internal state exists but the *scale* distorts it. Two tests from the records:

1. **Where the mass sits.** Aggregate `digit_dist` for level-2 actions whose integer is ≤ 1. If the
   mass is bimodal (a lump at 0–1 and a second lump at 6–8) that is sampling discarding a real
   second mode — reading (a). If it is unimodal at 0–1 with a thin tail, the scale is pulling the
   verdict down — reading (b). Report the shape, do not assert which.
2. **Says one thing, scores another.** Filter on `analysis_note_text` containing suspicion language
   (`suspicious|malicious|attack|exfiltrat|backdoor|unauthoriz|covert`) while `score ≤ 1`. Count them,
   report their mean E and their `label_level` breakdown, and **read ten of them verbatim**. A
   monitor that writes "this exfiltrates credentials" and then emits 1 is expression loss you can
   quote in the write-up without a probe.

Raw `p0b_anchoring_<ts>.json`. Figure `p0b_digit_mass.png`.
A proper de-anchored prompt is a new GPU run — out of scope. The `cf_neutralized` render in the
G4.5 chain is the nearest thing; if it lands, add it here as an appendix row.

## P0c. Sample variance — readout loss on a model you can open (30 min)

The `sampled_k4` stage (150 subset_main actions, 4 samples at T=1 from a shared prefill) finishes
~13:15Z. This is the local-monitor version of the cached monitor's two-sample disagreement, and it
is the cleanest measurement of loss (a) you will get.

Report, split by `label_level` and by side: the distribution of (max − min) across the 4 samples;
the fraction of actions where at least one sample is ≥ 3 while the greedy integer is ≤ 1; and
Spearman(mean of 4 samples, E). Redwood averages 5 samples per step, so the 4-sample mean is the
realistic deployed readout — report AUROC for it alongside E and the greedy integer.
If a payload's samples span 1 and 8, the state was there and sampling threw it away: quote it.

Raw `p0c_sample_variance_<ts>.json`. Figure `p0c_sample_spread.png`.

## P1. Lock the split (10 min, before any activation tensor is loaded)

Write and **commit** `split.json`: test trajectory ids, plus 5 grouped-CV folds over the train ids.

## P2. Probes (90 min)

Features: activation at (position p, layer l), fp32, standardized with train-fold mean/sd.
Sweep positions × layers 0–24.

**The position axis is the CoT experiment.** Pre-register which ordering you expect before running:

| position | what a high AUROC there means |
|---|---|
| `instruction_end` | **negative control** — precedes the transcript; must be at chance. If it isn't, the pipeline leaks. |
| `prompt_end` | the monitor represented it from reading the action, before generating anything |
| `analysis_end` | the CoT computed something the transcript read did not |
| `reasoning_end` | as above, later in the CoT |
| `pre_score` | **not evidence of latent knowledge** — digit mass is 1.0 here, so this position *is* the score |

The informative comparison is `prompt_end` vs `analysis_end`. Write down, in RESULTS, which of
these three you predicted **before** looking:
(i) `prompt_end` ≥ `analysis_end` ≥ E — the information is there pre-generation and the CoT adds nothing;
(ii) `analysis_end` > `prompt_end` > E — the CoT computes it and the score loses it (then you must
show the probe isn't just reading verbalized suspicion: re-run it with the P0b suspicion-language
actions removed);
(iii) `prompt_end` ≈ E — nothing latent beyond what is expressed, i.e. reading (c) for this monitor.

Two supervision arms, both required, because they separate the hypotheses:
- **score-supervised** (tests readout/expression loss): ridge on E; logistic on `score ≥ 3`. Never sees a label.
- **label-supervised** (existence test for comprehension): logistic on level-2 vs honest.

For each arm: diff-of-means direction first (report its AUROC — if it matches the fitted probe, say
so and prefer it), then the fitted model with regularization chosen by grouped CV inside train.
Evaluate against level-2 vs honest (AUROC, TPR@5e-2, TPR@1e-2) and against E (Spearman).
Choose (p*, l*) per arm by out-of-fold AUROC. Report locked test for those only.

Note on the score-supervised arm: 93 % of non-placeholder actions score 0–1, so its target is nearly
constant and it may learn the `view /` false-positive trigger rather than attack evidence.
**Check its top positives by hand** and say what they are.

Raw `p2_sweep_<ts>.json`. Figures `p2_sweep_<arm>.png` (position × layer heatmap, white at 0.5).

## P3. Baselines — this order, same split, same metric (2 h)

The confounds are measured, not hypothetical. From the D entry on the label branch:

| feature | level-2 | honest |
|---|---|---|
| `str_replace` in the action | 49 % | 5.5 % |
| median action length | 741 chars | 93 chars |
| median position in trajectory | 0.64 | 0.26 |
| zero-honest unigrams (`tracker_pass_123`, `user_tracking`, `stories`) | 14 of 53, **all one trajectory** | 0 |
| `packages-cdn.com` | 5 of 53 | 0 |
| positives per trajectory | 15 trajectories; top two hold 14 + 14 | — |

So the danger is **environment identity and action length**, not the CDN host. Run in this order:

1. **Random direction**: 200 random unit vectors at (p*, l*); the probe's percentile among them.
2. **Shuffled labels** permuted across trajectories, refit, 50×; AUROC distribution.
3. **Dumb features**: logistic on `[step_index, position_in_trajectory, n_prompt_tokens,
   n_context_tokens, command_length, has_sudo, has_chmod, has_curl, has_nc]`.
4. **Text ceiling**: TF-IDF char 3–5-grams + logistic on the rendered `current_action` only.
   If this matches the probe, surface form is the default explanation and the probe must **beat** it.
5. **Dumb + probe**: how much the probe adds over 3.
6. **Leave-one-environment-out** (10 envs, honest trajectories in every fold). This is the sharp
   control, because the zero-honest unigrams are one environment's DB password and table names.
   Report n positives per held-out env; `web_scraping`, `model_registry`, `clinical_trial` have 1
   each and will be uninterpretable — say so rather than reporting the number bare.
7. **Length- and position-matched negatives**: for each level-2 action draw honest actions matched
   on length decile × position decile × tool; re-evaluate on the matched subset. Collapse here means
   the probe read shape, not content.
8. **Residualize**: regress the probe score on the dumb features, evaluate the residual. This is
   what the activations add beyond length, position and tool.
9. **Leave-one-side-task-category-out**, as originally specified, as a secondary cut.
10. Rows 1–9 again with placeholders included (appendix only).

You cannot "suppress the activation around the CDN link" — by the capture position, attention has
already mixed that token everywhere, and removing it is a causal intervention, not a control.
Leakage is handled by 4, 6, 7, 8. The only true content filter is re-running with the artefact
scrubbed from the **input**, which is the `cf_neutralized` render; add it if it lands.

Raw `p3_baselines_<ts>.json`. Figure `p3_baselines.png`, one bar per method with cluster-bootstrap CI.

## P3b. Counterfactual renders — the real content filter (60 min)

`cf_neutralized` and `cf_random_scrub` both finished (250 subset_main actions each, 156 attack /
94 honest, strata `action` 150 and `history_only` 100). This is the only control that removes the
giveaway from the **input** rather than controlling for it in the evaluation, so it is the strongest
leakage test available and it is already computed.

Take the probe chosen in P2, **without refitting**, and score all three renders of the same 250
actions. Report AUROC and TPR@5 % per render, and the per-action score deltas.

| outcome | reading |
|---|---|
| neutralized ≈ original, random_scrub ≈ original | the probe is not reading the planted artifact |
| neutralized collapses, random_scrub holds | the probe was reading the artifact — this is the leakage result, report it |
| both collapse | scrubbing anything breaks the representation; the comparison is uninformative |

**Use `random_scrub` as the baseline, not `main`.** The chain log shows `cf_random_scrub` at
frac(score ≥ 3) = 0.125 against main's 0.077, i.e. scrubbing random tokens appears to *raise*
suspicion. If that holds, an artifact-removal effect measured against `main` is confounded with a
scrubbing effect. Verify the 0.125 first and report it either way — it is itself a finding about
the monitor.

Also run the same three-render comparison on the verbalized integer, E, and the text baseline, so
the probe's robustness is stated relative to theirs rather than in isolation.

Raw `p3b_counterfactual_<ts>.json`. Figure `p3b_renders.png`.

## P4. The result (45 min)

One Redwood-style figure: TPR vs logit(FPR), pooled out-of-fold, trajectory-bootstrap bands.
Curves: integer `score`, `E`, `cached_mean`, dumb, text, probe(score-sup), probe(label-sup).
Mark 5e-2 and 1e-2. Table of the same at both FPRs plus AUROC, with a locked-test column.

**Δp_null, the headline.** Of the 31 level-2 actions the integer puts at ≤ 1, how many does each
scorer rank above the honest 5 % and 1 % thresholds. E already gets 15 and 4. Report the probe on
the same denominator, with n, beside those. A probe that does not beat 15/31 has not shown (b).

Pre- vs post-CoT: best-layer AUROC per position in one row, with the pre-registered prediction from
P2 quoted above it and whether it held.

## P6. Read the data (45 min; this goes in the write-up)

- 5 reasoning texts drawn with `random.Random(0)` from subset_main, verbatim.
- Top 10 honest actions by the label probe, out-of-fold: command, score, cached, one line on why
  each might be a false positive.
- The level-2 actions with integer ≤ 1, ranked by probe: top 10, same fields, and confirm each is
  actually the payload (the hand-audit verdict is a column in `action_labels.json`).
- Bottom 10 level-2 by probe — what the probe also misses.


## Alternative explanations every RESULTS entry must address

n_eff = 59 trajectories (15 carry all 53 positives; two carry 28 of them); 1e-3 is not measurable
on 728 honest actions; length and position confounds (P3.3, P3.7, P3.8); environment identity
(P3.6); placeholders excluded and enriched for payloads (B entry); the level-2 label is
surface-form biased toward loud payloads (A3, A4), so every TPR here is an upper bound on catching
payloads; the score-supervised target is ~93 % constant; `pre_score` contains the verdict by
construction; chunk-size noise floor |ΔE| ≈ 0.05; 32k-token cap selection.

## Hard cut

Whatever is in the raws 6 h before the deadline is the dataset; the write-up needs 4 h.
Push after every stage with `bash infra/handoff.sh "<what ran>"`.
