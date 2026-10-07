# TASK-G4' — relaunch: main capture + CoT tier + contrast arms + integrity

Supersedes the G4 stage of TASK-ver2.md and the separate G5'/G6'. G1–G3 stand; read the newest
eight RESULTS.md entries before writing code. Nothing is running. `out/main` has 200 actions from
the halted pass 1 and `out/consistent_high` has 105, both at the OLD 4-position capture. Keep
them: they are the regression reference for this run.

## What the earlier passes established (given, not to be re-derived)

1. Ascending-length order == ascending step index. The first 200 actions were opening steps
   (median position 0.025, 1/200 consistent-high). The gate measured obedience to "default to
   LOW", not discrimination. Any partial dataset from this run must be a representative sample,
   so the main pass runs in seeded random order with no halt gate.
2. On the 105 consistent-high actions the local monitor is bimodal (non-placeholder: 22/54 ≥ 7,
   23/54 ≤ 1). Spearman(E, score) = 0.755 there vs 0.251 on opening steps. The signal exists and
   lives late in trajectories.
3. Forced-final readout without reasoning is anchored flat (E 0.3–1.5); after the model's own
   reasoning p(digit) = 1.0 at its own digit. Post-reasoning *readout* is degenerate; post-reasoning
   *activations* are not captured yet — that is what this run adds.
4. Reasoning=medium and k=4 sampling do not change ranking (n = 10). Not in the main pass.
5. 55% of consistent-high attack actions are `attachment://` placeholders. Every table splits on the flag.
6. Prefill is bitwise deterministic at fixed chunk size; forced readout on a deepcopy of the KV
   cache is bitwise identical to a standalone run. Chunk 256, fp32 sinks,
   `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`, `kernels==0.16.1`.

## Decisions already made — implement, don't ask

- `--gate-at 0`. Keep the rolling stats (frac score ≥ 3, sd E) in the log every 200 actions as
  information only. Never halt on them.
- **Template date stays `2026-09-10`**, `Reasoning: low`. Changing the date changes every prompt
  and breaks the bitwise check against the old shards for a non-bug reason.
- One mode, unchanged: unforced greedy generation (budget 1,024) for `score`; forced readout on a
  cache deepcopy for `expected_score`. Add generation-time capture; touch nothing else in the
  generation path.
- Capture positions become 7, in this order, recorded in every shard:
  - prompt side: `instruction_end`, `transcript_end`, `action_end`, `prompt_end`
    (== old `final`; write both names in the record)
  - generation side: `analysis_end` = last token of the analysis channel (the token before its
    `<|end|>`); `reasoning_end` = last token before `</reasoning>` in the final channel;
    `pre_score` = the `\n` after `<score>`, i.e. the position whose next-token distribution is the digit.
  - At `pre_score` also record `digit_dist_post`, `digit_token_mass_post`, `expected_score_post`
    from the logits of that step (free).
  - If a generation has no parseable `<|end|>`, `</reasoning>` or `<score>`, store NaN rows for the
    missing positions and set `capture_ok=false`. Do not fail the action.
- Run order, one chained script, one tmux session, each stage resumable, each stage its own dir:
  `subset_low_cached` (40) → `subset_consistent_high` (105, re-run) → `subset_main` remainder in
  `random.Random(0)` shuffle → `subset_payload_enrich` (only if the file exists when main finishes)
  → `subset_counterfactual` `neutralized` → `random_scrub` → optional k=4 on `subset_sampled`.
- Write `run_order.json` (stage, key, index, seed) and commit it before launch.
- Checkpoint every 25 actions; log ETA every 25; `run_meta.json` per stage with git commit, chunk
  size, template date, positions list.
- Rename old dirs to `out/main_pass1`, `out/consistent_high_pass1`. Delete nothing.

## Budget

~24 h wall clock. Projections: checks 0.5 h, low_cached 0.25 h, consistent_high 0.75 h, main
~12 h, enrichment 1–1.5 h, neutralized 1.8 h, random_scrub 1.8 h, k=4 2.8 h (optional),
integrity 0.5 h. Hard rule: 6 h before the user's cut-off, launch no new stage, run G6' on what
exists, pull `out/` to the laptop. A partial main pass in random order is a usable dataset; a
partial counterfactual render is not.

---

## G4.0 Pre-launch checks (~30 min; 3 actions from `out/consistent_high_pass1`: one honest, one attack payload, one placeholder)

1. Prompt-side rows: the 4 activation rows and `expected_score` bitwise equal the old shard for
   the same key. Assert; print max|Δ| (must be 0).
2. Generation determinism: generated text with capture hooks == generated text without hooks,
   token for token.
3. Position decoding: for each generation-side position print the decoded token with 5 tokens
   before and after. `analysis_end` precedes `<|end|>`; `reasoning_end` precedes `</reasoning>`;
   `pre_score` is followed by the digit that `score` parses to. `digit_token_mass_post` ≥ 0.99
   and argmax == parsed score.
4. Resume: kill after 2 actions, relaunch, assert no duplicate keys and no recompute of done keys.
5. Peak VRAM on the longest action (~32k) with hooks on; must be ≤ 24 GiB.
6. Shape `[7, 25, 2880]` fp16, no NaN/Inf on the 3, position-order list present in the shard.

Raw `g4_prelaunch_<ts>.json`. **Stop — show the six checks. Do not launch on any failure.**

## G4.1 low_cached (40) — contrast arm (~15 min)

Attack, both cached ≤ 2, non-placeholder, position-matched to consistent-high. Report score and E
distribution against the consistent-high re-run: local ~0 here too → local tracks the cached
labels; local high here → local's signal is its own. Raw `g4_low_cached_<ts>.json`.

## G4.2 consistent_high re-run (105, ~45 min)

Same records as pass 1 plus the 3 new positions. When done, run the G4.0.1 bitwise check over
all 105 prompt-side rows against `out/consistent_high_pass1`. Report n_mismatch (must be 0) and
score agreement with pass 1 (must be 105/105; greedy is deterministic). This is the regression
test of the new runner.

## G4.3 main pass — `subset_main.json` remainder, random order (~12 h)

Log every 25: done, s/action over the last 100, ETA, rolling frac(score ≥ 3), sd(E), frac hit
budget, frac capture_ok. If frac hit budget > 2% or capture_ok < 98% at 200, write it in the log
and continue.

## G6'a integrity on main (pod CPU or laptop, while later stages run; ~15 min)

File count; every shard loads; shard keys == `scores.jsonl` keys; line count == subset size −
documented failures; sha256 manifest of `out/main`; one tensor per stage opened, position order
printed. Then pull `out/` to the laptop (rsync over ssh — no object-storage remote exists; the
pod has 64 CPUs, so Track P may also run there while the GPU continues). Only after this does
Track P start on main.

## G4.4 payload enrichment (only if `subset_payload_enrich.json` exists; else skip and say so)

Built by TASK-A: every action in the 60 attack trajectories NOT in subset_main whose current
action matches a frozen level-2 payload signature, ≤ 32,000 tokens, non-placeholder. Adds
positives from trajectories the probe never sees otherwise. Same settings, dir `out/enrich`.

## G4.5 counterfactual renders — `subset_counterfactual.json`, `neutralized` then `random_scrub`

Same mode, same positions. Per action record n substitution events (history / current action)
and the render sha256. Carry G2's caveat into RESULTS: the substitutions are meaning-changing
(`curl→cat`), so a score drop is partly legitimate; this arm bounds surface-form leakage of the
probe, it does not isolate it.

## G4.6 optional k=4 on `subset_sampled.json` (2.8 h)

Only if every stage above is done and > 4 h remain before cut-off.

## G6'b final integrity + transfer (~30 min)

sha256 manifest of every `out/*` dir on both sides; `scores.jsonl` line counts per stage vs
subset size; every per-stage `.jsonl` committed under `results/raw/`; git commit hash recorded.
Only then terminate the pod.

## Output record schema (additions to the TASK-ver2 schema)

```
generation_text, analysis_note_text, n_generated_tokens, hit_budget, capture_ok,
expected_score_post, digit_dist_post, digit_token_mass_post,
capture_positions: 7 × {name, phase: prompt|generation, token_index, token_id, decoded},
run_stage, run_order_index
```

Activations `acts_XXXX.pt` keyed by `(traj_id, action_idx, render)`, shape `[7, 25, 2880]`
fp16, embedding output as layer 0, position-order list stored in the file.

## Not in scope

No probes, no ROC, no comparison figures (TASK-P). No reasoning=medium. No changes to
`monitor_prompt.py` rendering.

## Alternative explanations every RESULTS entry must address

The 32k cap and its selection bias (early actions kept; late honest actions dropped);
placeholders (split every table); 17,234 vs 12,494; MXFP4 in use (assert at load); capture
positions (print decoded tokens); chunk-size noise floor (|ΔE| ≈ 0.05, cosine ~1e-3 at
`prompt_end`); neutralization removes surface form, not evidence, and rewrites benign text too;
trajectory-level correlation (59 trajectories; every split by trajectory); and, new: the
generation-side positions contain the model's own verbalized verdict (analysis notes end
"Score N"; p(digit) = 1.0 at `pre_score`), so a probe there is not evidence of latent knowledge
beyond the score — `prompt_end` vs `analysis_end` is the informative pair.