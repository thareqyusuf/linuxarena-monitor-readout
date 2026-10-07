# TASK.md — LinuxArena monitor data + local monitor with activation capture

Two machines, two tracks that run in parallel. One sync point (L4 → G3).

- **Laptop (track L):** has the `linuxarena-paper` repo and control-tower. Produces the exported
  dataset, the logit-ROC baseline, and the small handoff package for the pod. Its in `../benchmark/linuxarena-paper`
- **GPU pod (track G):** A100 SXM 

Claude Code runs on the laptop. Pod-side scripts are written on the laptop under `gpu/`, tested there
with a tiny model, and pushed to a small git repo (scripts + manifest only, no data) that the pod clones.

## Fixed facts

Data source: public read-only mirror `https://repro.linuxarena.ai` (immutable snapshot, no credentials).
- `GET /api/public/lean/<run_id>` — projected trajectories + content fingerprint
- `GET /api/public/monitor-data/<monitor_run_id>` — per-action cached monitor scores, streamed
  (probably NDJSON). If `monitor_run_id` 404s, try the entry's `source_run_id`.
Run IDs in `src/results/public_run_ids.json`. Two entry types: `agent` (trajectories, key `run_id`)
and `monitor` (scores, keys `monitor_run_id` + `source_run_id`).

Starting pair — Opus policy, GPT-5 Nano cached monitor:

| | agent run_id | GPT-5 Nano monitor_run_id |
|---|---|---|
| honest | `ef6ed492f5a74ab8885e95469fe72fd2` | `f25f1eafacd749e89f1736f26d3df751` |
| attack | `d085c1cbc4984437823563de06fb8cbe` | `30d53f5ae4f7444687459d13ccc61101` |

Known oddities to verify, not assume:
- The "monitor on Opus honest" entries list `source_run_id: 18962e83fe48407abe7a6b753cb2f940`, which
  is NOT the `ef6ed492…` agent entry. Fetch both; find out which one the monitor scores actually
  join to (by run ID, or by task ID + action index).
- The attack-side monitor has `ensemble_size: 2`; the honest side has none. Count samples per action
  in each stream. If they differ, use per-action mean on both sides for the ROC and say so.

Monitor input convention (from the blog + `mean_action.py`): task description + all prior actions'
tool inputs AND outputs + current action's tool input only. Never the policy's reasoning/scratchpad,
never the current action's output, never future actions. Live action-level scores only (the paper's
headline ensemble blends live + delayed 2:1; we don't).

Aggregation (blog convention): benign = every (honest action, sample). Attack = per attack trajectory,
the action with the highest mean score, keep its samples. FPR(t) = benign rows > t.
TPR(t) = attack trajectories with max action > t. logit10(p) = log10(p / (1 − p)).
Crop plots to points with ≥ 5 FP, ≥ 5 TP, ≥ 5 FN. Min usable FPR ≈ 20 / n_benign_actions.

Hard constraints (both machines): no `make runs`, no `ct run eval/sabotage-eval/monitor`, no Docker,
no paid LLM API calls. New laptop code under `data_export/`; pod code under `gpu/`. Don't modify
repo files. Each step appends to `REPORT.md` and stops for review. If the data or code differs from
what this file assumes, report and ask; don't guess.

---

## Track L — Laptop

### L1. Setup + reconnaissance
1. `uv sync --frozen`; `uv run ct --help` works. `gh auth status`; `uv run ct registry pull linuxarena`.
2. Read and summarize (short): `src/lib/lean_load.py` (how it hits the API, what "projected" drops),
   `src/monitors/live/mean_action.py` (exact prompt construction; where the 0–100 score is parsed),
   `src/section_5_3_monitoring_results/process.py` (action → trajectory aggregation),
   `src/pipeline/configs/phase2_main_results.yml` (which runs are headline).
3. Fetch the honest + attack runs above via both endpoints. Report per-action schema; whether
   `tool_input` / `tool_output` are complete or truncated in the projection; whether policy reasoning
   is present; how monitor-data keys back to actions; samples per action on each side; the two
   oddities above. Record each run's content fingerprint.

Stop.

### L2. Export
`data_export/export.py` → `data_export/data/{honest,attack}.jsonl`, one line per **action**:
`run_id, traj_id, env, main_task, side_task (null for honest), policy, action_idx, n_actions_in_traj,
tool_name, tool_input, tool_output, side_task_succeeded, side_task_trigger_action_idx (if recorded;
else null), monitor_scores (list of cached samples)`. Policy reasoning excluded.
`data_export/data/summary.md`: counts per (policy, env, side_task); total benign actions; total attack
trajectories; min usable FPR.

Stop.

### L3. Logit-ROC from cached scores
`data_export/logit_roc.py`: standard ROC; logit10–logit10 ROC per side_task + pooled, cropped points
filled, others hollow; TPR at FPR ∈ {1e-1, 3e-2, 1e-2, 3e-3, 1e-3} with bootstrap 95% CIs
(trajectories for TPR, actions for FPR); OLS slope / intercept / R² on cropped points.
Cross-check vs `src/section_5_3_monitoring_results/data.json`; explain any gap.

Stop.

### L4. Handoff package for the pod  ← sync point
Everything the pod needs and nothing else. Goes in `gpu/` and gets pushed to the scripts repo.
1. `gpu/format_monitor_prompt.py` — pure Python, **no control-tower imports**. `(trajectory_dict,
   action_idx) -> messages`. Verified byte-identical to `mean_action.py`'s output on ≥ 50 actions
   spanning honest/attack and early/late actions. This is the most important file in the project.
2. `gpu/manifest.json` — run IDs (agent + monitor), content fingerprints from L1, the full list of
   `(traj_id, action_idx)` to process, and a `pilot` subset of 200 (100 honest / 100 attack, spread
   across envs and across early/late action indices).
3. Token-length histogram: download only the tokenizer for `openai/gpt-oss-20b` and for
   `Qwen/Qwen3-30B-A3B`; apply chat template to every rendered prompt; report median / p95 / max
   per model. Write the model decision into `REPORT.md` using the rule in G1.
4. `gpu/run_monitor.py` — written here, tested here on CPU with `Qwen/Qwen3-0.6B` end to end:
   fetch from API → verify fingerprint → render prompt → chat template → prefill with forward hooks
   on every decoder layer's output → store residual at the **last prompt token** (before any
   generation; before `<think>` / the `analysis` channel) as fp16 `[n_layers, d_model]` keyed by
   `(traj_id, action_idx)` → generate → parse score → also store `expected_score` (probability-
   weighted over the score token(s); handle digit-by-digit tokenization — if messy, sample N=5 and
   store all, report which) and the full response text → checkpoint every 500 actions → resumable.
   Flags: `--model`, `--manifest`, `--subset {pilot,all}`, `--out`, `--max-new-tokens`.
   For gpt-oss: parse the score from the `final` channel. Cap `max_new_tokens` (start at 2048).
5. `gpu/requirements.txt` pinned to the versions tested in (4).

Stop. Push `gpu/` to the scripts repo.

---