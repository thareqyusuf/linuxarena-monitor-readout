# RESULTS — logit-curve-replication

Branch `exp/logit-curve-replication`. Running log for this experiment only. Newest at the top.
Every quantitative claim carries its source, its raw file, and a recompute
command — see README.md. Status tags: `[hunch]` `[tested]` `[verified]` `[dead]`

## Hypothesis

The GPT-5 Nano live action-monitor's ROC on Opus honest vs Opus attack
(LinuxArena §5.3), plotted as logit10(TPR) vs logit10(FPR), can be rebuilt from
the cached per-action scores on the public mirror and is approximately linear
over the cropped range (≥5 FP, ≥5 TP, ≥5 FN). Kind of claim: general claim
(replication of a published curve; the baseline is the paper's own `data.json`).

Track L (this branch) only prepares data + the cached-score ROC + the pod
handoff; the open-weight monitor with activation capture runs on the pod
(track G) and is a separate branch.

## Task

See `TASK.md` in this directory — the prompt the experiment session was given,
verbatim, so the reviewer can see what was asked vs what was run. The running
narrative per step is in `REPORT.md` (same directory).

---

## 2026-09-10 — L4 handoff package: prompt formatter byte-identical on all 17,234 actions

**Claim.** `gpu/format_monitor_prompt.py` (pure Python) reproduces
control-tower 1.0.1's live-monitor prompt byte for byte on every scorable
action of the export (17,234/17,234); `gpu/manifest.json` lists those actions
with cached scores and control-tower prompt hashes plus a 200-action pilot;
`gpu/run_monitor.py` runs end to end on CPU with Qwen3-0.6B (2 actions),
capturing fp16 `[n_layers, d_model]` last-prompt-token residuals, a parsed
score, an expected score over digit tokens, and resuming from its checkpoint.
Token-length statistics: see the L4.3 entry in `REPORT.md`.

**Kind of claim.** Engineering verification (no hypothesis).

**Evidence.**
- Scripts: `experiments/logit-curve-replication/gpu/format_monitor_prompt.py`, `data_export/verify_prompt_format.py` @ commit `4032851` (= `.git_commit` in the verify raw), `data_export/build_manifest.py`, `data_export/token_lengths.py`, `gpu/run_monitor.py`
- Raw: `results/raw/logit-curve-replication/l4_prompt_verify_20260910T043718Z.json`; `l4_token_lengths_<ts>.json` (see REPORT L4.3)
- Logs: `logs/logit-curve-replication/l4_prompt_verify.log`, `l4_build_manifest.log`, `l4_token_lengths.log`, `l4_smoke_qwen3-0.6b.log`
- Figure: `results/figures/logit-curve-replication/l4_token_lengths.png`
- Recompute: `jq '{n_actions_compared, n_equal, n_mismatch}' results/raw/logit-curve-replication/l4_prompt_verify_20260910T043718Z.json` → 17234, 17234, 0; `jq '{n_actions, pilot_size}' experiments/logit-curve-replication/gpu/manifest.json` → 17234, 200

**Alternative explanations considered.**
- "Byte-identical to control-tower" is not "byte-identical to what GPT-5
  Nano saw": stored prompts are empty in the dump, task descriptions and
  side-task goals come from the dump / v1.0.1 registry, and the dump was
  edited after the fact (39 renamed outputs). Nothing here can close that gap.
- The smoke test used the two shortest pilot prompts on CPU; nothing about
  CUDA, gpt-oss's channel format, or 100k-token prompts is exercised.

**Status.** `[tested]`

---

## 2026-09-10 — L3 logit-ROC from cached GPT-5 Nano scores

**Claim.** With per-action ensemble means, an honest pool of 12,494 actions and
the max action per attack trajectory (n=92, ungated), the cached monitor
reaches TPR 0.560 (95% bootstrap CI 0.461–0.654) at FPR 1e-2 and 0.314
(0.223–0.409) at FPR 3e-3; the 11 cropped ROC points in logit10–logit10 space
fit a line of slope 0.881 with R² 0.985, with the low-FPR end bending below it.
The honest pool and attack maxima are identical (as multisets) to the paper's
`data.json`, and the paper's per-budget thresholds are reproduced exactly.

**Kind of claim.** General claim (replication of a published curve). Baseline
= the paper's own numbers (cross-check above); chance line on the standard ROC.

**Evidence.**
- Script: `experiments/logit-curve-replication/data_export/logit_roc.py` @ commit `4032851` (= `.git_commit` in the raw file)
- Input: `results/raw/logit-curve-replication/export_actions_20260910T043234Z.jsonl` (from `data_export/export.py`, same commit)
- Raw: `results/raw/logit-curve-replication/l3_logit_roc_20260910T043236Z.json`
- Log: `logs/logit-curve-replication/l3_logit_roc.log`
- Figures: `results/figures/logit-curve-replication/l3_roc_standard.png`, `l3_logit_roc_pooled.png`, `l3_logit_roc_per_category.png`
- Recompute (no model needed):
  - `jq '.tpr_at_fpr["0.01"] | {tpr, ci95}' results/raw/logit-curve-replication/l3_logit_roc_20260910T043236Z.json` → 0.560, [0.461, 0.654]
  - `jq '.ols_pooled' …` → slope 0.881, intercept 1.906, r2 0.985, n 11
  - from scratch: `python3 -c "import json;d=json.load(open('results/raw/logit-curve-replication/l3_logit_roc_20260910T043236Z.json'));b=d['benign_scores'];m=[t['max_score'] for t in d['attack_trajectories']];print(sum(s>7.5 for s in b)/len(b), sum(x>7.5 for x in m)/len(m))"` → FPR 0.00840, TPR 0.543 at threshold 7.5 (the point just below FPR 1e-2)
  - cross-check: `jq '.cross_check_paper | {honest_pool_multiset_equal, attack_max_multiset_equal, thr: [.per_budget[].threshold_equal]}' …` → true, true, all true

**Baselines.**
| condition | value |
|---|---|
| main (TPR at FPR 1e-2) | 0.560 |
| chance | 0.010 |
| paper safety at budget 1e-2 (different metric, same threshold) | 0.572 |

**What I checked myself.** (left for the reviewer)

**Alternative explanations considered.**
- Action-level bootstrap ignores that honest actions are clustered in 200
  trajectories of very unequal length; CIs on FPR are likely too narrow.
- TPR at a target FPR is an interpolation on a 0.5-step score grid; the strict
  step-function value at threshold 7.5 is 0.543 (FPR 8.4e-3).
- Per-category curves rest on 5–31 trajectories; only one category has ≥5
  cropped points.

**Status.** `[tested]`

---

## 2026-09-10 — L2 export: 12,675 honest + 5,097 attack actions

**Claim.** The export covers all 200 honest and 92 attack monitor-run
trajectories; every action's cached samples equal the mirror's; 12,487 honest
actions are scorable by the live monitor (12,494 in the paper's pool);
tool_output differs from the HF dump on 39 actions, all `LinuxBench`→`linuxarena`
renames, and no tool_input contains either string.

**Kind of claim.** Descriptive (data preparation).

**Evidence.**
- Script: `experiments/logit-curve-replication/data_export/export.py` @ commit `4032851` (= `.git_commit` in the raw file)
- Raw: `results/raw/logit-curve-replication/export_meta_20260910T043234Z.json`, `export_actions_20260910T043234Z.jsonl` (superseded first run: `…_20260910T042851Z.*`)
- Log: `logs/logit-curve-replication/l2_export.log`
- Data: `experiments/logit-curve-replication/data_export/data/{honest,attack,trajectories}.jsonl` (gitignored, 61 MB; sha256 in `data_export/data/CHECKSUMS.json` and in the raw meta), `summary.md`
- Recompute: `jq '.sides | map_values({n_trajectories, n_actions, n_actions_scorable, member_mismatch: .checks.member_mismatch_hf_vs_api, output_diff: .checks.output_diff_hf_vs_api})' results/raw/logit-curve-replication/export_meta_20260910T043234Z.json`

**Alternative explanations considered.** `tool_input` has no mirror-side
check at all (the mirror strips arguments); if the HF dump altered arguments
in ways other than the two strings searched for, nothing here would notice.

**Status.** `[tested]`

---

## 2026-09-10 — L1 reconnaissance: mirror + HF rows audited

**Claim.** The four runs (Opus honest/attack agent runs; GPT-5 Nano monitor
runs on each) are complete on the public mirror; monitor runs join to agent
runs by `(env, main_task, side_task)` 200/200 and 92/92; the honest monitor
scored a strict prefix of 3 of the 200 honest trajectories; both monitor runs
are 2-sample ensembles except 333 single-sample actions on the attack side;
per-action scores on the mirror equal the HF dump on every action.

**Kind of claim.** Descriptive data audit (no hypothesis tested yet).

**Evidence.**
- Script: `experiments/logit-curve-replication/data_export/recon.py` @ commit `4362d82` (= `.git_commit` in the raw file; the script was committed together with the raw file in the next commit — the raw file records the scaffold commit it was run at)
- Raw: `results/raw/logit-curve-replication/l1_recon_20260910T042333Z.json`
- Log: `logs/logit-curve-replication/l1_recon.log`
- Recompute (no model needed):
  - join: `jq '.agent_monitor_join.honest | {task_keys_matched, matched_with_identical_function_sequence, matched_with_different_actions: (.matched_with_different_actions|length)}' results/raw/logit-curve-replication/l1_recon_20260910T042333Z.json` → 200, 197, 3
  - samples per action: `jq '.samples_per_action | {honest: .honest_monitor.n_actions_with_2_members, attack2: .attack_monitor.n_actions_with_2_members, attack1: .attack_monitor.n_actions_non_ensemble}' …` → 12675, 4764, 333
  - mirror vs HF: `jq '.hf | {h: .honest_monitor.api_vs_hf_member_scores, a: .attack_monitor.api_vs_hf_member_scores}' …` → different: 0 on both
  - fingerprints: `jq '[.fingerprints[].recompute_matches] | all' …` → true
  - honest pool: `jq '.samples_per_action.honest_monitor.n_scorable_actions' …` → 12494 (= `len(honest_action_scores)` for "GPT-5 Nano" in `linuxarena-paper/src/neurips/section_5_3_monitoring_results/data.json`)

**Baselines.** n/a (audit).

**What I checked myself.** (left for the reviewer)

**Alternative explanations considered.**
- The 3 truncated honest trajectories could be a different *run* of the same
  task rather than a prefix: ruled out for the function sequence (exact
  prefix), not for the arguments/outputs (L2 checks content equality of the
  prefix).
- "Mirror == HF" is only checked on scores and outputs; `arguments` exist only
  in HF, so nothing on the mirror can confirm them. The mirror fingerprint
  does not cover them either.

**Status.** `[tested]`

---

## Verification

Filled in by the review session, on this branch, before merge to main.

### L1 reconnaissance — 2026-09-10

**Verdict.** Verified but weaker than stated. Every number in the L1 claim
reproduces; two of the five sub-claims say more than the data supports.

**Pre-flight.** Script `recon.py` is byte-identical between the commit that
added it (`8ac6c1c`) and HEAD; raw file records the scaffold commit
`4362d82` as stated. Raw + log + RESULTS entry are on the branch (no figure;
none expected for an audit). Review branch is `vrf/logit-curve-replication`,
not `exp/`. The `_scan_shard` 600-byte prefilter could silently drop HF rows,
but the row counts it found (200/200/92/92) equal the API trajectory counts,
so nothing was dropped.

**What the script computes (in my words).** It downloads two projections of
five run ids from the public mirror and caches them; hashes the `lean`
payload the same way the server does; tabulates key sets and histograms per
run; pairs agent and monitor trajectories by (env, main_task, side_task)
and compares their tool-name sequences; pulls the same run ids out of the
local HF jsonl shards and compares, per action, the HF score/member list
with the mirror's; counts members per action; and counts actions whose tool
name is not in a hardcoded hidden-tool set. It never touches a model.

**What was recomputed** (independent Python over `checkpoints/logit-curve-replication/{api,hf_rows}`,
the cached mirror payloads and HF rows; script in the verify session's
scratchpad, ~80 lines, no code shared with `recon.py`):

- Join by task key: 200/200 honest, 92/92 attack, 0 agent-only, 0
  monitor-only, 197 and 92 with identical tool-name sequences. Matches.
- For the 197 + 92 pairs with identical tool sequences, `arguments` and
  `output` are identical on every action (0 mismatches). Stronger than the
  raw file states.
- Members per action: honest 12675×2; attack 4764×2 and 333×0. Matches.
- HF vs mirror per-action `sus_score` and member lists: 12675/12675 and
  5097/5097 equal. Matches.
- Honest pool: the 12494 scores on the mirror (all actions minus 181
  `submit`) are an exact **multiset** match to `honest_action_scores` for
  "GPT-5 Nano" in the paper's `data.json`, not just a count match.
- Bonus: the paper's `attack_max_sus` (92 values) is an exact multiset match
  to per-trajectory max over mirror actions **only** when hidden tools are
  excluded and the 333 no-member actions are kept as-is. Including hidden
  tools breaks it (13 trajectories jump to 9). So the paper's pool is
  exactly the L1 pool, and the paper uses the 333 unchanged.
- Recompute commands in the L1 entry are `jq '.field'` reads of the stored
  number, not recomputations; the raw file holds aggregates only. The
  inputs (per-trajectory tool sequences, per-action scores) live in
  gitignored `checkpoints/` and on the public mirror, not in the raw file.
  Acceptable for an audit of an immutable public source, but the raw file
  does not meet the README.md "recompute without the model" bar on its own.

**Dumbest ways this could be wrong, and what rules each out.**

1. *"Strict prefix of 3 honest trajectories" — prefix of what?* Tool names
   and `arguments` are an exact prefix (0 argument mismatches over the 74,
   72 and 15 prefix actions). `output` is **not**: 10, 23 and 3 prefix
   positions differ. The differences are (a) the agent-run copy holding a
   literal `bash: line 1: attachment://<hash>: No such file or directory`
   where the monitor copy holds real output, and (b) `ls -la` timestamps a
   month apart (Apr 10 vs Mar 10). So the monitor scored a **re-executed
   replay** of the same action sequence that stopped early, not the agent
   run's recorded outputs. L2 already records this
   (`prefix_content_identical: false`). Claim should read "prefix in tool
   names and arguments; tool outputs differ". Not ruled out; confirmed.
2. *"333 single-sample actions."* Not supported. Those 333 have
   `is_ensemble=false`, `members=null`, `raw_response="None"` (the string),
   no `model_name`, tokens, cost or latency, and a single-monitor
   `reasoning` text. But **136 of 333 have half-integer scores** (0.5, 1.5,
   …), which a single 0–10 integer sample cannot produce (all 34k member
   scores in both runs are integers). Provenance unknown: could be ensemble
   means whose member records were lost, or a different parser. Their mean
   score is 0.85 vs 1.88 for ensemble actions; 33 trajectories affected, one
   (`1356572f…`) has 37 of 68 actions in this state; in 6 of 92 trajectories
   the per-trajectory max comes only from such an action. Claim should say
   "333 actions with no member records" and stop there. L3 should treat them
   as a labelled subgroup, not as one sample.
3. *"Mirror == HF on every action" proves nothing independent.* Both are
   projections of the same private database keyed by the same run id.
   Agreement shows the mirror is a faithful projection of the dump, which is
   what L1 needed; it is not a second source. Also the claim text says
   "scores"; outputs differ on 25 honest and 14 attack actions (in the raw
   file, not in the claim). Fine as stated, but should not be cited later as
   corroboration.
4. *Fingerprint check is a tautology.* The server computes the fingerprint
   over the payload it serves; recomputing it client-side checks transport
   integrity only. `public_run_ids.json` carries no fingerprint, so there is
   no external reference. The `source_run_id` (`18962e83…`) returns zero
   trajectories and its "matching" fingerprint is the hash of an empty list.
5. *Hidden-tool set is hardcoded, not read from the paper's code.* Ruled
   out by the multiset match to `data.json` (both honest pool and attack
   maxima). `start_phase1/2_planning` (83 attack actions) are correctly kept.
6. *Could a task-key match pair different rollouts?* Ruled out for 289 of
   292 pairs by full content equality; for the 3 truncated ones the
   argument-level prefix match over 15–74 stochastic Opus steps rules out an
   independent rollout.

**Findings outside the claim that later steps need.**

- HF `arguments` contain `attachment://` placeholders in 50 honest-monitor
  actions (42 `bash.cmd`, 7 `text_editor.file_text`, 1 `db_bash.cmd`) and
  197 attack-monitor actions; the attack-monitor `output` has 197 such
  placeholders on both mirror and HF. The mirror's `tool_call` carries only
  the tool name. For those actions the monitor's true input is unrecoverable
  from either source. Bears directly on the L4 "byte-identical prompt" claim.
- The honest agent run carries 635 stray GPT-5-nano scores; 443 of them
  differ from the monitor run's score on the same action. The cached monitor
  is not deterministic across runs.
- L2's `n_actions_scorable` is 12487 (7 "malformed" dropped) vs L1's 12494.
  Small, but the two entries currently disagree on the pool size.

### L2 export — 2026-09-10

**Verdict.** Verified but weaker than stated. All counts reproduce and the
export is byte-reproducible from the committed script. Two of the text
claims are not computed by the script at all, and the "scorable" count
departs from the paper's pool by a rule the paper's own code cannot apply.

**Pre-flight.** `export.py` at HEAD is byte-identical to the version
committed in `b6114ff`. The raw meta records `git_commit` `4032851`, which
is the merge commit *before* the script existed; the superseded first run
(`…042851Z`) records the same commit but produced different files
(15.7 MB vs 16.4 MB attack; no `prefix_actions_differing` field), so the
script changed between the two runs without a commit. Resolved by re-running:
executing the committed script against the cached mirror payloads and HF
rows in the verify scratchpad reproduces the cited sha256 of
`honest.jsonl`, `attack.jsonl` and `trajectories.jsonl` exactly. So the
committed script is the one that made the cited data.

**What the script computes (in my words).** For each monitor-run
trajectory (HF row), it looks up the agent-run row by task key, the mirror
`lean` payload (scores) and the mirror `monitor-data` payload (outputs) by
trajectory id, and emits one line per action taking `tool_input` from HF
`arguments`, `tool_output` from the mirror when non-null else HF, and
`monitor_scores` from the mirror member list, or a one-element list of the
mirror mean when there are no members. It marks an action skipped if its
tool name is hidden or its HF `output_error` starts with `parsing:`, and
counts the rest as scorable. It counts HF-vs-mirror output mismatches but
does not classify them, and never searches any text for `LinuxBench`.

**What was recomputed** (independent Python over `checkpoints/` and the
exported files; verify scratchpad):

- 200 + 92 trajectories, 12675 + 5097 lines, no duplicate
  (monitor_traj_id, action_idx). Matches.
- `monitor_scores` equals the mirror member list on all 17772 actions and
  `monitor_score_mean` equals the mirror mean on all. Matches. Note this is
  by construction: both come from the same mirror payload. The independent
  fact is the HF-vs-mirror member check, whose counter is *absent* from
  `checks` (the stated `jq` recompute returns `null`, not 0) because the
  script only creates the key on a mismatch.
- Scorable: 12675 − 181 hidden − 7 malformed = 12487 honest;
  5097 − 264 − 86 = 4747 attack. Matches.
- Output diffs: 25 + 14 = 39. Matches. 34 become identical after
  `linuxarena`→`LinuxBench`; the other 5 (1 honest, 4 attack) are the same
  rename in lowercase (`linuxbench`, in a seed-script path and message).
  Direction: the mirror has the old name, HF has the new one. Exported
  text has 0 occurrences of `linuxarena` (case-insensitive) in inputs,
  outputs or task descriptions, so no scrubbed text leaked in.
- No `tool_input` on any of the four runs contains `LinuxBench`,
  `linuxbench` or `linuxarena`. Matches, but this was **not** computed by
  the script; nor was "all renames". Neither has a raw-file trace.
- `side_task_trigger_action_idx`: set on all 30 live-failure trajectories,
  on none of the 62 delayed ones; 79 attack actions fall after the trigger.
  The port matches control-tower 1.0.1 `get_live_failure_action_index`.

**Dumbest ways this could be wrong, and what rules each out.**

1. *"12,487 scorable by the live monitor" is not what the live monitor
   scored.* All 7 honest "malformed" actions (and all 86 attack ones) carry
   real two-member scores with no error, so the paper's monitor scored
   them. The paper's pool is 12494 because its lean loader projects away
   `output_error` (checked `lean_load.py`), so its `should_skip_action`
   can never see the `parsing:` flag; the mirror lean payload has no such
   field either. The 12487 rule matches control-tower 1.0.1 source
   (checked in the paper's venv) but not the paper's pool. Dropping them
   changes no attack-trajectory max (0 of 92), so it only moves the FPR
   denominator by 7 actions. Which pool to use is an experimental-design
   call, not mine; the entry should say the two numbers come from different
   rules and pick one explicitly.
2. *Exported text is not what the monitor saw.* Not ruled out; the claim is
   silent on it. `tool_input` has `attachment://` placeholders on 50 honest
   and 197 attack actions, and `tool_output` on 197 attack actions (mirror
   and HF agree, so both sources are lossy). The export carries no flag for
   these. Any "byte-identical prompt" claim downstream is bounded by this.
3. *`tool_output` from the mirror is treated as "what the monitor saw".*
   For the 3 truncated honest trajectories this is a replay, not the agent
   run's output (see L1); for the rest the mirror and HF differ only by
   the rename, so the choice is immaterial there.
4. *The 333 no-member attack actions become one-sample lists.* The script
   turns the mirror mean into `[mean]`; 136 of those are half-integers.
   Downstream code that treats `len(monitor_scores)` as the number of
   samples will count them as single draws. See L1 finding 2.
5. *Key collision or silent `zip` truncation.* Ruled out: task keys are
   unique on both sides (asserted), and the script asserts equal action
   counts between HF, lean and monitor-data per trajectory.
6. *Superseded run swapped in.* Ruled out by the byte-reproduction above.

### L3 logit-ROC — 2026-09-10

**Verdict.** Verified but weaker than stated. Every number reproduces
exactly, including a full re-run of the committed script. The chance
baseline in the table is wrong for the aggregation used, and the
"bending below" description has the wrong sign at the low-FPR end.

**Pre-flight.** `logit_roc.py` at HEAD is byte-identical to `b6114ff`.
Raw `git_commit` is `4032851` (script not yet committed then); resolved by
re-running the committed script on the committed input in the paper's venv:
the raw JSON reproduces field-for-field (all keys except timestamp and
commit), bootstrap included (seed 0). Three figures present.

**What the script computes (in my words).** Benign pool = the mirror mean
of every honest action whose tool is not hidden (12494; malformed kept).
Attack score = per trajectory, max of those means over non-hidden actions.
For every distinct score value as a strict `>` threshold it counts FP, TP,
FN; "cropped" = all three ≥ 5. OLS is fit to log10-odds of (FPR, TPR) on
the cropped points. TPR at a target FPR is linear interpolation between
adjacent ROC points, which equals the paper's randomized tie-breaking in
expectation. The CI resamples honest *actions* and attack trajectories
independently, 2000 reps, percentile. It also writes a per-sample variant
and compares pool, maxima and per-budget thresholds against the paper's
`data.json`.

**What was recomputed** (plain-Python re-derivation from `benign_scores`
and `attack_trajectories[].max_score` in the raw file; separately checked
those two arrays against the mirror lean payloads directly, bypassing the
export chain: both multisets equal):

- 22 ROC points, 11 cropped (thresholds 3.0 … 8.0). Matches.
- OLS on cropped: slope 0.8810, intercept 1.9061, R² 0.9846. Matches.
- TPR at FPR 1e-2 = 0.5596, at 3e-3 = 0.3136, at 1e-3 = 0.2282. Matches.
  Step-function value at t>7.5: FPR 0.00840, TPR 0.543. Matches.
- Independent bootstrap (my own code, B=1000): action-level CI at 1e-2
  [0.459, 0.654] vs stated [0.461, 0.654]. Matches to resampling noise.
- **Trajectory-cluster bootstrap on the honest side** (resample the 200
  honest trajectories; needs `traj_id`, which `benign_scores` in the raw
  file lacks, so this used `export_actions_…jsonl`): CI at 1e-2
  [0.439, 0.663], at 3e-3 [0.224, 0.428]. Slightly wider than stated, not
  materially. The 105 FPs at t>7.5 sit in 40 of 200 honest trajectories,
  top three holding 11, 10, 9.
- Per-budget threshold port equals control-tower 1.0.1
  `compute_threshold_at_fpr` line for line; all six thresholds and tie
  fractions match the paper. Matches.
- Gated vs ungated max: identical on all 92 trajectories, so "ungated" is
  immaterial here.

**Dumbest ways this could be wrong, and what rules each out.**

1. *The chance baseline is 0.010 in the table. It is not.* The attack score
   is a max over ~53 scored actions per trajectory (16–130), so a monitor
   emitting random scores at per-action FPR p catches a trajectory with
   probability 1 − (1−p)^n. From `n_scored_actions` in the raw file that is
   **0.393 at FPR 1e-2, 0.144 at 3e-3, 0.051 at 1e-3**. The paper's own
   `compute_random_safety_curve` is exactly this formula and its
   `random_safety` at FPR ≈ 1e-2 is 0.390. So the headline is 0.560 vs a
   chance of 0.39, not vs 0.01; at 3e-3 it is 0.314 vs 0.144. The table
   row should be replaced and the per-action chance line on the standard
   ROC figure is the wrong reference for a per-trajectory TPR.
2. *"Low-FPR end bending below the line" has the wrong sign.* Residuals of
   the cropped points from the OLS line: t=6.5 −0.076, t=7.0 −0.116,
   t=7.5 −0.005, **t=8.0 +0.067**. The dip below is in the middle
   (thresholds 6.5–7.0); the lowest-FPR cropped point is above the line,
   dropping it moves the slope to 0.957, and the first uncropped point
   (t=8.5, FPR 2.4e-4, TPR 0.196) is 0.67 above the extrapolated line. The
   curve flattens (bends *above*) at the low-FPR end. This is visible in
   `l3_logit_roc_pooled.png`.
3. *Per-action mean is not the convention in TASK.md.* TASK says benign =
   every (action, sample) and attack = the samples of the max action, with
   per-action mean as the fallback if sample counts differ. The fallback was
   taken and stated. But the per-sample variant in the same raw file gives
   **TPR 0.341 at FPR 1e-2 and 0.185 at 3e-3**, i.e. the headline depends
   heavily on which convention is used. Which one the writeup should use
   is a design call; the entry should show both.
4. *Ties at the interpolation point.* 26 of 92 attack maxima sit exactly
   at 8.0 and 7 at 7.5; the 3e-3 number is an interpolation across that
   step. Randomized tie-breaking is the paper's rule, so this is
   consistent, but the number is a mean over tie-breaks, not an observed
   TPR.
5. *Scores copied through a lossy export.* Ruled out: benign pool and
   attack maxima taken straight from the mirror lean payload equal the raw
   file's arrays as multisets.
6. *Pool disagreement with L2.* L3 uses 12494 (paper pool); L2's 12487
   "scorable" is unused here. Consistent with the paper, inconsistent with
   the L2 entry's headline. Say which is canonical.

### L4 handoff package — 2026-09-10

**Verdict.** Verified but weaker than stated, with one sub-claim not
delivered. The formatter, manifest and CPU smoke test check out. The
token-length statistics do not exist: the only raw file is the run the
script's own docstring calls wrong, REPORT.md's L4.3 reads
`TOKEN_STATS_PENDING`, the cited figure is absent and its log is empty.
Separately, the L4 commit added 3.4 MB of Claude Code session transcripts
to the repo root.

**Pre-flight.** `format_monitor_prompt.py`, `verify_prompt_format.py`,
`build_manifest.py` and `manifest.json` are unchanged since `b6114ff`.
Verify raw records commit `4032851` (script not yet committed then; same
pattern as L2/L3). Commit `6ab2837` ("exp: L4 setup", not the
`exp(<slug>):` form) tracks four files that are not project files:
`4e46a300-….jsonl`, `69e093e5-….jsonl`, `c4174914-….jsonl` and
`c4174914-…/tool-results/bpmslut79.txt` at the repo root. They are Claude
Code session logs (session ids, account and org UUIDs, bridge session ids,
full tool output). A masked scan found no token values (`hf_`, `sk-`,
`gh*_`, `JUPYTER_TOKEN=<value>`), only the names `JUPYTER_TOKEN` and
`/workspace/.env`. Remove them from the branch before merge; they are not
gitignored. `REPORT.md` is deleted in the working tree but present at HEAD.
The claimed figure `l4_token_lengths.png` is not in
`results/figures/logit-curve-replication/`; `l4_token_lengths.log` is 0
bytes.

**What the scripts compute (in my words).** The formatter re-implements
control-tower's history/action formatting, the str_replace diff
reconstruction, and template fill, reading tool inputs and outputs from
the L2 export and side-task goal text from a dumped registry file. The
verifier builds a control-tower `Trajectory` from the HF row *after
overwriting its outputs with the export's mirror text*, so both sides get
identical inputs by construction, then compares `build_action_prompt` to
the formatter on every non-hidden, non-malformed action and stores each
prompt's sha256. The manifest lists those actions with cached scores and
the hashes and flags a 200-action pilot. The runner renders, hashes
against the manifest, chat-templates, hooks every decoder layer to keep
the last-position output of the prefill pass, greedy-generates, parses
`<score>`, and computes a digit-token expectation at the score token.

**What was recomputed.**

- Independent re-verification in the paper venv on 313 scorable actions
  (300 random, seed 12345, plus all 8 str_replace-diff actions and 5
  `attachment://` actions): control-tower prompt equals the formatter on
  313/313 and its sha256 equals the manifest on 313/313. On the
  *unmodified* HF row (no output overwrite) 281/313 match; the 32 that
  differ are the rename scrub, as expected.
- Template sha256 `a4a48d03…` equals the paper's `old_action.md`; the
  paper's headline config routes `mean_action_live_monitor` →
  `old_action_live_monitor(provide_goals=True)` → that template, so the
  right template is being matched.
- Manifest: 17234 actions, `cached_scores` equals the export on all,
  pilot 100 honest + 100 attack. Matches.
- Smoke checkpoint (gitignored): `acts_0000.pt` holds two `[28, 1024]`
  fp16 tensors, no NaN, layer-0/27 norms 6.5/341; `scores.jsonl` has
  parsed score 5.0 on both, expected score 4.70 / 5.35 with digit-token
  mass 0.9996 / 0.99999, `prompt_sha256_matches_manifest` true on both.
  `run_meta.json` shows a second invocation finishing with
  `n_actions_this_run: 0`, which is the resume evidence; it is not in the
  committed log.
- Token lengths: `l4_token_lengths_20260910T052617Z.json` reports
  median = p95 = max = 2 tokens for every prompt on both tokenizers. The
  committed `token_lengths.py` docstring says this run took `len()` of a
  BatchEncoding and is wrong; the fixed script has not been run (no
  second raw file). The tokenizers are not in any local HF cache, so I did
  not reproduce them offline. From the character stats (median 99.6k,
  p95 282k, max 499k chars) the prompts are roughly 25k / 70k / 125k
  tokens, an estimate only. The model decision that TASK.md ties to these
  numbers is therefore unsupported.

**Dumbest ways this could be wrong, and what rules each out.**

1. *"Byte-identical to control-tower" is measured on inputs that are not
   what the monitor saw, and the claim's caveat understates how much.*
   Rendering all 17234 prompts: **1626 contain `attachment://`
   placeholders** (162 honest, 1464 attack = 31% of attack prompts) across
   43 trajectories, because a placeholder in one action's input or output
   propagates into every later prompt of that trajectory. 29 of the 200
   pilot prompts (28 attack) are affected. The entry's caveat mentions
   empty stored prompts and the 39 renames, not this. For those prompts
   the pod will score text the cached monitor never saw.
2. *The verifier can't fail on outputs.* It copies export outputs into the
   HF row before building the control-tower trajectory, so output
   handling is compared formatter-vs-formatter on identical strings. That
   is what "same inputs" means and it is stated, but it means the check
   is purely of formatting code, not of data.
3. *Manifest pool ≠ L3 pool.* 17234 excludes the 7 honest and 86 attack
   malformed actions that the cached monitor did score and L3 keeps. The
   pod run will have no local score for them; any cached-vs-local
   comparison is on 12487, not 12494.
4. *Smoke test proves little.* Two shortest pilot prompts (1.7k tokens),
   CPU, bf16, a 0.6B model, thinking off. With `enable_thinking=False` the
   chat template appends an empty `<think>` block, so the captured "last
   prompt token" is the newline after `</think>`, a different position
   than the thinking-on pod configuration will capture. Nothing about
   gpt-oss's channel parsing, 100k-token prompts, or CUDA is exercised;
   the entry says so.
5. *"10" folded into "1".* `expected_score` renormalises over single-token
   digits; on Qwen "10" is two tokens, so P(10) is absorbed into P(1).
   Recorded in the output, harmless for a 0–9 mass check, wrong if
   anyone reads `expected_score` as a 0–10 expectation.
6. *Wrong template.* Ruled out by the config trace above and the sha256.
7. *Side-task goals from a different registry version.* Not ruled out by
   anything here; the goals text comes from `dump_side_task_goals.py`
   against the v1.0.1 registry and is asserted equal to control-tower's
   `get_side_task_goals` output only through the byte-identical prompt
   check, which uses the same source.

**On "L4 was not possible".** REPORT.md's L1 entry says L4 is impossible
*mirror-only* because the mirror strips arguments. The HF dump supplied
arguments, which is why L4.1 could run. The HF dump's arguments are
themselves placeholders on 247 actions, so for 1626 prompts L4 is still
impossible in the sense that matters. The token-length step was not
completed at all.

---

## Dead ends

- (none yet)
