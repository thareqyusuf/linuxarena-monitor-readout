# TASK-G' — compressed run to G4

Supersedes G2–G6 in the original TASK.md. G1 stands; read its RESULTS.md entry before writing code. Everything under "What G1 established" is a finding from a previous session on this pod — given, not to be re-derived.

## What G1 established

1. **Attention.** Neither shipped path works: eager is O(L²) memory (80 GB at 25k tokens), flex_attention can't decode. Use `chunked_eager` from `experiments/local-monitor-activations/chunked_attention.py`, registered via `AttentionInterface`. Bitwise-equal to eager at the function level with fp32 sinks; closer to fp64 than eager with the model's bf16 sinks. Keep fp32 sinks. Always set `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`.
2. **Prefill cost** (batch 1, model resident 12.83 GiB): 4k→1.7s/16.7GiB, 16k→8.5s/17.6GiB, 33k→27.6s/18.9GiB, 65k→108s/21.4GiB, 98k→271s/24.2GiB, 131k→513s/28.0GiB. Quadratic. This is the binding constraint on the run.
3. **Noise floor is not zero.** Different chunk sizes give up to 1.49 logit difference at 2048 tokens (reduction order × MoE routing flips). Fix chunk size across the run and record it everywhere.
4. **Tokenizer.** `"10"` is a single token (id 702); digits 0–9 are ids 15–24. `expected_score` covers 0–10 exactly with the existing `digit_token_ids`. Original constraint 5 is moot.
5. **Template.** Ends `…<|end|><|start|>assistant` (last token `assistant`, id 173781). No injected empty reasoning block — the L4 capture-position failure cannot recur. The template injects `Current date:` and `Reasoning:` into the system message.
6. **Data.** Manifest pool is 17,234 actions (12,487 honest + 4,747 attack), not the paper's 12,494. 1,626 prompts contain `attachment://` placeholders (162 honest, 1,464 attack). 333 attack actions have no member records. The cached monitor is non-deterministic (443/635 stray scores disagree with themselves). Export files are on the pod; verify sha256 against the manifest before anything else.

## Decisions already made — implement, don't ask

- Thin runner at `experiments/local-monitor-activations/run_monitor.py`, importing track L's modules. Track L's files at the L4 commit stay untouched.
- CLI: `--attn chunked_eager --attn-chunk-size N --max-prompt-tokens N --render {original,neutralized,random_scrub} --reasoning-effort low --template-date YYYY-MM-DD --subset <file>`.
- **Pin the template date** to a fixed literal passed in, recorded per action. Pin `Reasoning:` explicitly. Neither may float.
- Softmax/sinks stay fp32.
- No Jupyter MCP. Plain scripts, tmux, logs to disk.
- `kernels==0.16.1` must be installed or the quantizer silently dequantizes to bf16 and OOMs.
- Mode is forced-final only: one forward pass over the prompt, capture activations, then force `<|channel|>final<|message|>` and read the digit-token distribution at the next position. No reasoning tokens generated.

## Budget

~36 h wall clock, of which ~20–22 h is GPU. Wall time while a run is unattended does not count against my application hours — keep the pod busy, don't rush.

---

## G2. Prep — lengths, capture positions, subset (~1 h, no GPU except step 4)

1. Verify the three export files' sha256 against `manifest.json`. Stop on mismatch.
2. Render all 17,234 prompts, apply the chat template with the pinned date and `add_generation_prompt=True`, tokenize. Length = `len(input_ids)`. Assert on 10 examples that length ≈ chars/4.
3. Report median/p90/p95/p99/max, count over 128k, split by side and by `has_attachment_placeholder`. Figure: histogram with the chosen cap marked.
4. Print the **last 12 tokens** of one honest and one attack templated prompt, ids and decoded strings. Confirm the last is `assistant`. Then locate and record the character and token offsets of the other three capture positions (end of action under review, end of transcript block, end of instruction block) for both examples, decoded.
5. **Choose the cap and subset, write `subset_main.json`:**
   - Cap: largest round number ≤ 32,000 tokens such that projected prefill for the subset fits budget. Show the projection arithmetic.
   - Sample **by trajectory, not by action**. ~2,000 actions from ~35 trajectories, weighted toward attack (~20 attack / ~15 honest). Include all of a chosen trajectory's actions under the cap.
   - Attack trajectories must cover all four side tasks with ≥3 trajectories each (leave-one-side-task-out).
   - Report honest/attack ratio before and after the cap, and the placeholder rate in the subset. The cap over-represents attack because attack trajectories are shorter — state this as a known selection bias.
6. **Neutralization rule.** Write a deterministic substitution table by hand (no LLM rewriting). Emit `subset_counterfactual.json` (~250 actions, stratified by side and cached-score bucket) plus, if timing allows, a matched `random_scrub` render that removes an equal quantity of non-attack-indicative text. Print 10 original/neutralized diffs for me to read.
7. Projected wall-clock from the G1 prefill curve (Σ t(Lᵢ), fitted quadratic).

**Raw:** `g2_prep_<ts>.json`. **Figures:** `g2_lengths.png`. **Stop — show me the cap, the subset composition, and the neutralization diffs.**

## G3. Pilot (30 actions, ~30 min)

1. 30 actions spanning the length range, both sides. Report seconds per action by length decile, peak VRAM, activation payload size per action.
2. **Noise floor:** re-run 20 at a different `--attn-chunk-size`. Report distribution of |Δscore|, |Δexpected_score|, and cosine distance between capture vectors at each of the four positions. This number is the floor for interpreting any probe-vs-score gap.
3. Activation health: shapes, dtype, no NaN/Inf, per-layer norm profile per position, and that two different actions give different vectors at every position.
4. Run 10 of them under `--render neutralized` and confirm the prompt actually changed and still parses.
5. Recompute N for the main run from measured seconds. If more than 20% off the G2 projection, say so and re-pick.

**Raw:** `g3_pilot_<ts>.json`. **Figure:** `g3_layer_norms.png`. **Stop.**

## G4. Capture run (~14–18 h, unattended)

`--render original --subset subset_main.json`, ascending prompt-length order. Then `--render neutralized --subset subset_counterfactual.json`, then `random_scrub` if time remains.

Checkpoint every 25 actions, resumable, rsync `out/` to object storage every 100. Log to `logs/local-monitor-activations/g4_run.log`. If the pod dies, a partial short-prompt-only dataset must still be complete and usable.

**Raw:** `g4_run_<ts>.json` — completion counts, failures with reasons, wall-clock, cost, checksums. **Stop.**

## Output record schema

```
traj_id, action_idx, side, side_task, render,
prompt_sha256, prompt_sha256_matches_manifest, has_attachment_placeholder,
n_prompt_tokens, score, expected_score, digit_token_mass, cached_scores,
step_index, trajectory_length, position_in_trajectory, n_context_tokens,
command_length, has_sudo, has_chmod, has_curl, has_nc,
capture_positions: [{name, token_index, token_id, decoded}],
seconds, attn_impl, attn_chunk_size, softmax_dtype, reasoning_effort,
template_date, max_prompt_tokens, git_commit
```

Activations to `acts_XXXX.pt` keyed by `(traj_id, action_idx, render)`, shape `[n_positions, n_layers+1, d_model]` fp16, embedding output as layer 0, position order recorded in the file.

## Not in scope

No probe training, no ROC, no comparison figures.

## Alternative explanations every RESULTS entry must address

The length cap and its selection bias; the 1,626 placeholder prompts; the 17,234 vs 12,494 pool gap; whether MXFP4 was actually used; the capture positions; the chunk-size noise floor; whether the neutralization removed real evidence rather than surface form; and the fact that trajectory-level correlation makes action-level splits invalid downstream.