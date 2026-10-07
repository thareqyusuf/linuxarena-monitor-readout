# TASK.md — Track G: local monitor pass with activation capture

Branch: `exp/local-monitor-activations`. Machine: rented GPU pod, **RTX 5090 (32 GB, sm_120)**.
Depends on track L's handoff (`gpu/format_monitor_prompt.py`, `gpu/manifest.json`, `gpu/run_monitor.py`,
`gpu/requirements.txt`) at the L4 commit.

The pod has **no** control-tower, no inspect-ai, no repo clone beyond the scripts repo. It fetches its
own copy of the mirror payloads. It never calls a paid API.

Follow `README.md`: every quantitative claim gets a raw file under `results/raw/local-monitor-activations/`,
a log under `logs/local-monitor-activations/`, a recompute command that does not need the model, and a
RESULTS.md entry with status tag and "alternative explanations considered". Stop after each numbered
step and wait for review.

---

## What track L established that constrains this run

Read these before writing code; they are not optional context.

1. **The manifest pool is 17,234 actions** (12,487 honest + 4,747 attack). This is *not* the paper's
   L3 pool of 12,494 honest actions: the 7 honest and 86 attack "malformed" actions the cached monitor
   did score are excluded. Any cached-vs-local comparison is on the intersection, not on L3's pool.
   Record this in every comparison rather than silently reconciling it.
2. **1,626 of the 17,234 rendered prompts contain `attachment://` placeholders** (162 honest,
   1,464 attack = 31% of attack prompts, 43 trajectories). A placeholder in one action propagates into
   every later prompt of that trajectory. For those prompts the local monitor scores text the cached
   monitor never saw. **29 of the 200 pilot actions (28 attack) are affected.** Every output record must
   carry a boolean `has_attachment_placeholder` so downstream analysis can split on it. Do not drop them
   at this stage.
3. **Token lengths are unmeasured.** The L4 raw file reporting median = p95 = max = 2 tokens is a bug
   (`len()` of a BatchEncoding). The only available estimate is from character counts: median ≈ 99.6k
   chars, p95 ≈ 282k, max ≈ 499k → very roughly 25k / 70k / 125k tokens. **The max may exceed
   gpt-oss-20b's 128k context.** Measuring this correctly is step G2 and it gates everything after.
4. **The L4 smoke test captured the wrong position for a thinking model.** With
   `enable_thinking=False` the Qwen template appended an empty `<think>` block, so the captured
   "last prompt token" was the newline after `</think>`. The pod must define the capture position
   explicitly against gpt-oss's harmony template and record the decoded token, not assume `[-1]`.
5. **`expected_score` folds "10" into "1"** on tokenizers that split two-digit numbers. gpt-oss's
   tokenizer must be checked; if "10" is multi-token, either compute the two-token joint properly or
   name the field `expected_score_single_digit` and document the ceiling.
6. **333 attack actions have no member records** and 136 of those carry half-integer scores. They are
   in the manifest with a one-element `cached_scores` list. Do not treat `len(cached_scores)` as a
   sample count.
7. **The cached monitor is not deterministic**: 443 of 635 stray GPT-5-nano scores in the honest agent
   run disagree with the monitor run on the same action. Local-vs-cached score differences are
   therefore expected and are not evidence of a formatting bug on their own.

---

## G1. Environment and hardware verification

1. `nvidia-smi`; `python -c "import torch; print(torch.__version__, torch.version.cuda, torch.cuda.get_device_capability(), torch.cuda.get_device_properties(0).total_memory)"`.
   Require capability `(12, 0)` and CUDA ≥ 12.8. If the image is older, stop — Blackwell kernels will
   not build and the run will be silently slow or crash.
2. Persistent volume if the provider has one; `export HF_HOME=<volume>/hf`.
3. `git clone <scripts repo>`; `pip install -r gpu/requirements.txt`. Record the resolved versions of
   `torch`, `transformers`, `accelerate`, `safetensors` in the raw file — MXFP4 support on sm_120 is
   version-sensitive and this is the single most likely source of a silent fallback.
4. Fetch the four mirror payloads (2 `lean`, 2 `monitor-data`) to `<volume>/data/`. Verify each
   content fingerprint equals `manifest.json`. Fetch nothing else.
5. Download `openai/gpt-oss-20b`. Confirm it loaded in **native MXFP4**, not dequantized bf16:
   log `model.get_memory_footprint()` and the dtype of a couple of MoE expert weights. bf16 would be
   ~42 GB and will OOM on 32 GB, so a successful load is weak evidence, but log it explicitly.

**Raw:** `g1_env_<ts>.json` — GPU name, capability, VRAM, package versions, model footprint, fingerprints.
**Stop.**

## G2. Token-length measurement (gates the model choice)

Redo what L4.3 failed to do, correctly.

1. Render all 17,234 prompts with `gpu/format_monitor_prompt.py`.
2. Apply the gpt-oss chat template and tokenize. Length = `len(input_ids)` of the templated prompt, not
   `len(BatchEncoding)`. Assert on a handful of examples that the number is plausible against the
   character count (roughly chars/4).
3. Report median, p90, p95, p99, max; the count over 128k; the distribution split by honest/attack and
   by `has_attachment_placeholder`; and a histogram figure.
4. KV-cache estimate at p95 and max for gpt-oss-20b at batch 1, accounting for its alternating
   dense / sliding-window layers. Compare against `32 GB − model footprint`.

**Decision rule, applied in the report:**
- max ≤ 128k and p95 KV fits → proceed with gpt-oss-20b, batch 1, prompts processed in length order.
- a small tail exceeds 128k → proceed, but list the exact `(traj_id, action_idx)` that will be dropped
  or truncated, and state which. Truncation changes the information set relative to the paper and must
  be flagged per-action in the output, not applied silently.
- a large fraction exceeds context or KV does not fit → **stop and report.** Do not silently switch
  models; the fallback (`Qwen/Qwen3-14B` bf16, ~30 GB, leaves almost no KV headroom) is worse and
  leaves the blog's monitor list.

**Raw:** `g2_token_lengths_<ts>.json`. **Figure:** `g2_token_lengths.png`.
**Recompute:** `jq '{median, p95, max, n_over_128k}' …`.
**Stop.**

## G3. Capture-position definition

Before any batch run, settle exactly which hidden state is stored.

1. Render one honest and one attack prompt. Apply the chat template with `add_generation_prompt=True`.
   Decode and print the **last 12 tokens** with their ids. Record them in the raw file.
2. The capture position is the final token of the templated prompt — i.e. the position whose forward
   pass is the last one before the model emits its first generated token. For gpt-oss this is inside the
   harmony assistant header, before the `analysis` channel opens. State the decoded token explicitly.
3. Confirm no template-injected empty reasoning block sits between the transcript and the capture
   position (the L4 failure mode). If one does, capture instead at the last token of the *user* content
   and document both positions.
4. Store all layers: hidden states `[n_layers+1, d_model]` fp16 (include the embedding output as layer 0
   and say so). Expected 25 × 2880 → ~144 KB per action → ~2.5 GB total.

**Raw:** `g3_capture_position_<ts>.json` — token ids, decoded strings, chosen index, tensor shape.
**Stop.**

## G4. Pilot (200 actions)

`python gpu/run_monitor.py --model openai/gpt-oss-20b --subset pilot --out <volume>/out/`

Per action the runner writes: `traj_id`, `action_idx`, `side`, `prompt_sha256`,
`prompt_sha256_matches_manifest`, `has_attachment_placeholder`, `n_prompt_tokens`, `truncated`,
`score` (parsed from the harmony **final** channel), `expected_score` (+ digit-token mass and the "10"
caveat), `n_generated_tokens`, `stop_reason`, `response_text`, `seconds`. Activations to
`acts_XXXX.pt` keyed by `(traj_id, action_idx)`. Checkpoint every 50 in the pilot; resumable.

Report:
1. **Throughput**: seconds per action, split by prompt-length decile; peak VRAM; projected wall-clock
   and cost for all 17,234.
2. **Parse health**: how many responses yielded a score; how many hit `max_new_tokens` without closing
   the final channel (gpt-oss reasons in `analysis` first — if a large fraction truncates, raise
   `--max-new-tokens` and say so). Start at 2048.
3. **Score comparison against cached GPT-5 Nano on the same 200 actions**: Spearman, mean difference,
   and the two histograms. Split by `has_attachment_placeholder` — the 29 affected actions are the
   ones where disagreement is expected for a known reason. This is a sanity check on the input, not a
   result; a different model will not track gpt-5-nano closely, and the cached monitor is itself
   non-deterministic (finding 7).
4. **Activation health**: shapes, dtype, no NaN/Inf, per-layer norm profile, and that two different
   actions give different vectors (guards against a hook capturing a constant).

**Raw:** `g4_pilot_<ts>.json`. **Figures:** `g4_score_hist.png`, `g4_layer_norms.png`.
**Stop — review before the full run.**

## G5. Full run

`--subset all`. Process in ascending prompt-length order. `rsync`/`rclone` `<volume>/out/` to object
storage every 500 actions; a reclaimed pod must not cost more than 500 actions of work.

**Raw:** `g5_run_<ts>.json` — completion counts, failures with reasons, total tokens, wall-clock, cost,
final checksums. **Log:** stream to `logs/local-monitor-activations/g5_run.log`.
**Stop.**

## G6. Return and integrity check

1. Copy `out/` to the laptop under `experiments/local-monitor-activations/data/`.
2. Verify: file count, sha256 manifest, one activation tensor loads and matches its pod-side checksum,
   `scores.jsonl` line count = 17,234 minus documented failures.
3. Only then terminate the pod.

**Raw:** `g6_transfer_<ts>.json` (checksums both sides).

---

## Not in scope

No probe training, no ROC recomputation with local scores, no comparison figures beyond the pilot
sanity check. Those are laptop work on a later branch, once G6's integrity check passes.

## RESULTS.md entry requirements

Each of G2, G4, G5 gets an entry. State the kind of claim (G2 and G4 are engineering verification /
descriptive; none of them test the research hypothesis). Under "alternative explanations considered",
at minimum address: the 1,626 placeholder prompts, the 17,234 vs 12,494 pool gap, whether MXFP4 was
actually used, whether the capture position is the intended one, and any truncated prompts.