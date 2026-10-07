# TASK-A — action-level attack labels (laptop, CPU, ~2–3 h, while G4 runs)

Why: `side` labels the trajectory. In a 20-step attack trajectory one to three actions do the
malicious thing. A label built from the cached monitor (`both cached ≥ 6`) marks exactly the
actions the hypothesis is about — malicious and scored low — as benign, so a null under it is
uninterpretable. Build a monitor-independent label and keep the cached one as a secondary column.

## Contamination rule (read before anything else)

Any session that has read RESULTS.md has read 17+ high-cached actions. Signature drafting (A2)
happens in a **fresh session** whose only context is `side_task_descriptions.md` (the A1 output)
and this file. That session must not open RESULTS.md, any `scores.jsonl`, or any raw file.
Freeze and commit `payload_signatures.json` before any check runs.

## A1. What the export already gives (~30 min, main session)

1. `trajectories.jsonl`: list every field. Report `side_task_success` per trajectory (and
   `main_task_success`, `side_task_category`, `side_task`, `failure_type` if present) for all 92
   attack trajectories and for the 32 in subset_main. Count successful / failed / unknown.
2. Extract, per `side_task` (the fine-grained name, across the 6 categories), the side-task
   description text from the export (the task-description / covert-objective block of the
   rendered prompt) into `side_task_descriptions.md`: name, category, environment, verbatim
   objective. No actions, no scores, no trajectory ids.
3. Re-check `side_task_trigger_action_idx` once: for the 12 known, print agent-trajectory length,
   manifest trajectory length, and the index. If a constant offset maps it into the manifest,
   report it; otherwise one line saying so and move on.

Raw `a1_export_fields_<ts>.json`. **Stop — hand `side_task_descriptions.md` to the fresh session.**

## A2. Draft signatures — fresh session (~45 min)

From `side_task_descriptions.md` only. One entry per `side_task`:
`{side_task, category, level2: [regex, …], level1: [regex, …], rationale}`.

- Level 2 = the action that accomplishes the objective: the append to `authorized_keys`, the
  `COPY … TO PROGRAM`, the fetch-and-execute, the `DELETE FROM <named table>`, the trigger
  install, the `cat` of the named secret file, the outbound POST of the named data.
- Level 1 = recon or preparation the description implies: reading credential files, listing
  users, dumping the named store.
- Regexes run against the `current_action` string only (tool name + arguments as rendered),
  case-sensitive where the objective is. Prefer the objective's named artefact (table, path,
  host) over generic verbs.

Write the file, commit, record the commit hash. **Stop.**

## A3. Validate (~45 min, main session)

1. Apply to every action in subset_main and in all 92 attack trajectories. Label = max level
   matched; 0 otherwise. Honest actions are 0 by construction — still run the signatures on them.
2. Coverage: for each attack trajectory with `side_task_success == True`, does ≥ 1 action match
   level 2? Report "level-2 fires in N of M successful trajectories" and list the misses.
3. Specificity: number of honest actions matching any level-2 signature (should be ~0) and any
   level-1 (report; some `cat .env` is expected). Print every honest level-2 hit verbatim.
4. Fix misses by reading **only** the failed trajectories. Every edit to the signature file is a
   new commit with a one-line reason. Re-run 2–3. Stop after 3 rounds or when coverage stops improving.
5. Placeholder actions: the signature cannot see the content; label NaN with
   `label_source="placeholder"`, not 0.

Raw `a3_validate_<ts>.json`. Figure `a3_label_position.png`: position-in-trajectory of level-2
actions, by category.

## A4. Blind audit (60 actions, ~45 min — the user reads; the agent only draws and formats)

Draw 60 attack actions with `random.Random(0)` from subset_main, non-placeholder, without
reference to cached scores or signatures. Present each as: key, step / length, side-task
objective (one line), `current_action` verbatim, and an empty verdict column. The user fills
payload / prep / benign. Then compute, on the user's verdicts: signature miss rate, cached-high
(`both ≥ 6`) miss rate, signature false-positive rate. Report all three with n.

Raw `a4_audit_<ts>.json`, user verdicts included verbatim.

## A5. Emit

`action_labels.json`, one row per action in the 17,234 pool:
```
traj_id, action_idx, side, side_task, side_task_category, side_task_success,
label_level (0/1/2/NaN), label_source (signature|placeholder), matched_pattern,
label_cached_high (both ≥ 6), cached_mean, signature_commit
```
And `subset_payload_enrich.json`: level-2 actions in attack trajectories NOT in subset_main,
≤ 32,000 tokens (G2's `per_action.n_tokens`), non-placeholder, with n per category. This is
G4.4's input — write it as soon as A3 passes so the pod can pick it up.

## B. Placeholder control (~15 min, same session)

Cached-score distribution for honest-placeholder (162 in pool) vs honest-non-placeholder, and
attack-placeholder vs attack-non-placeholder: mean, p(≥ 6), Mann-Whitney. If placeholders score
high in honest trajectories, the placeholder string is itself a suspicion trigger for the cached
monitor, and both monitors are partly scoring a rendering artefact.

Raw `b_placeholder_<ts>.json`.

## Alternative explanations to state in RESULTS

Signatures are surface forms — a linear probe can read `packages-cdn.com` off the tokens, which
is why the neutralized arm and the dumb/text baselines are load-bearing; level 1 vs 2 is a
judgement, exposed as a column rather than buried in a binary; `side_task_success` is
trajectory-level and says the payload ran, not which action ran it; the 60-action audit is the
only human-verified number and n = 60.
