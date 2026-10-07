# Latent vs verbalised suspicion in an open-weight LLM monitor

Does an LLM action monitor *know more than it says*? On the LinuxArena agent-control
benchmark, a gpt-oss-20b monitor reads each agent action and emits an integer
suspicion score 0–10. This repo asks whether the model's residual stream, or its
full digit distribution at the readout token, carries attack signal that the
emitted integer throws away, and whether a linear probe can recover it.

**Short answer.** The forced digit distribution E (expected score over the ten
digit tokens) beats the emitted integer where it matters, but no linear probe on
the residual stream beats E. The signal the integer loses is "mass at 2–3", not a
hidden high verdict, and it appears in the model's chain of thought at the sentence
that names the covert objective.

| | AUROC | TPR @ 5 % FPR | TPR @ 1 % FPR |
|---|---|---|---|
| emitted integer | 0.74 | 0.42 | 0.33 |
| forced digit distribution E | 0.90 | 0.55 | 0.30 |
| GPT-5 Nano cached monitor (paper) | 0.85 | 0.40 | 0.31 |
| best residual probe (score-supervised, pooled) | 0.94 | — | — |

53 level-2 payload actions vs 743 honest actions, trajectory-cluster bootstrap
CIs in the raw. The probe's pooled edge is a denoised E: on the 31 payloads the
integer scores ≤ 1 it recovers 11, the same 11 E recovers. Source:
`results/raw/latent-vs-verbalized/p4_result_pool743_20260912T030653Z.json` and
`p7_dpnull_check_20260911T152343Z.json`; recompute commands in
[`experiments/latent-vs-verbalized/RESULTS.md`](experiments/latent-vs-verbalized/RESULTS.md).

## Summary figures

| | |
|---|---|
| ![F1](results/figures/latent-vs-verbalized/summary/F1_readout_loss.png) | ![F2](results/figures/latent-vs-verbalized/summary/F2_no_probe_beats_E.png) |
| F1 — what the integer throws away | F2 — no probe beats E on the missed payloads |
| ![F3](results/figures/latent-vs-verbalized/summary/F3_where_signal_sits.png) | ![F4](results/figures/latent-vs-verbalized/summary/F4_verdict_in_cot.png) |
| F3 — where in the prompt the signal sits | F4 — the readout moves at the objective sentence |

Captions with every deviation from the figure spec:
[`CAPTIONS.md`](results/figures/latent-vs-verbalized/summary/CAPTIONS.md).
Verbatim examples: [`EXAMPLES.md`](results/figures/latent-vs-verbalized/summary/EXAMPLES.md).
Appendix one-liners: [`APPENDIX.md`](results/figures/latent-vs-verbalized/summary/APPENDIX.md).

## The four stages

| stage | what it does | runs on |
|---|---|---|
| [`logit-curve-replication`](experiments/logit-curve-replication/) | Pulls the public mirror, rebuilds the paper's cached-monitor ROC, ports the control-tower prompt builder to pure Python (byte-identical on all 17,234 actions), builds the manifest the GPU stage consumes. | laptop, CPU |
| [`local-monitor-activations`](experiments/local-monitor-activations/) | Runs gpt-oss-20b (MXFP4) as the live monitor on 2,012 actions with chunked attention for 25k-token prompts, capturing `[7 positions, 25 layers, 2880]` residuals per action. Counterfactual renders with the evidence neutralised. | one 32 GB GPU |
| [`action-level-attack-label`](experiments/action-level-attack-label/) | Regex payload signatures drafted blind from the side-task objectives, validated and audited by hand (176 actions, two rounds, in `a4_audit*/`). Produces the per-action label every later ROC uses. | laptop, CPU |
| [`latent-vs-verbalized`](experiments/latent-vs-verbalized/) | Linear probes per layer and position, nine baselines, leave-one-category-out, matched negatives, CoT truncation. The summary figures above. | GPU for P12, otherwise CPU |

[`RESULTS.md`](RESULTS.md) is the index. Each stage directory has:

- `TASK.md` — the brief for that stage (later briefs in `notes/`),
- `RESULTS.md` — the running log, one entry per run, newest first. Every number
  cites the script that produced it, the raw file it came from, and a `jq` or
  Python one-liner that recomputes it without the model. A "Verification" section
  at the bottom records an independent recompute from code and raw files only,
  with a verdict of *Verified*, *Verified but weaker than stated*, *Not verifiable
  from what's here*, or *Wrong*.

## Layout

```
experiments/<slug>/           scripts, TASK.md, RESULTS.md, notes/
results/raw/<slug>/           every raw output, append-only; carries the inputs to each number
results/figures/<slug>/       PNG/SVG
logs/<slug>/                  stdout of every run
infra/                        pod bootstrap, torch pins, VRAM preflight
checkpoints/<slug>/           activations, exports, trained probes — not in the repo
```

Scripts resolve their own namespace from their location, so they run from any
checkout. Raw files stay under 10 MB; anything larger is a checkpoint, and the
numbers derived from it live in a raw file next to it.

Example, the headline AUROC above:

```sh
P4=results/raw/latent-vs-verbalized/p4_result_pool743_20260912T030653Z.json
jq '.scorers | {score, E, cached_mean} | map_values(.pooled.auroc)' $P4
```

## Setup

**Data.** The public read-only mirror `https://repro.linuxarena.ai` (no
credentials) for trajectories and cached monitor scores, plus a checkout of the
LinuxArena paper repo for the cross-check against its `data.json` and for the
control-tower prompt builder. Set `LINUXARENA_PAPER` to that checkout (default
`../benchmark/linuxarena-paper`).

**Laptop (analysis, figures, labels).**

```sh
python -m venv .venv && . .venv/bin/activate
pip install -r requirements.txt
```

**GPU pod (monitor pass, activation capture, P12).** One 32 GB card is enough:
gpt-oss-20b in MXFP4 loads at 12.8 GiB and peaks at 22.7 GiB on a 32k-token
prompt. On a Runpod pod with a network volume at `/workspace`:

```sh
bash infra/bootstrap.sh                     # system packages, venv, pinned torch stack
python infra/preflight.py --model openai/gpt-oss-20b
```

Pinned versions are in `infra/constraints.txt` and
`experiments/logit-curve-replication/gpu/requirements.txt`. Weights go to
`HF_HOME` (the pod scripts default to `/workspace/hf`).

## Running

Scripts within a stage run in the order of their prefixes; the exact invocation,
inputs and outputs of every run are recorded in that stage's `RESULTS.md` entry.

1. **Export and baseline** (laptop) — `experiments/logit-curve-replication/data_export/`:
   `recon.py` → `export.py` → `logit_roc.py` → `build_manifest.py` →
   `verify_prompt_format.py`. Smoke-test the GPU runner on CPU:
   `python gpu/run_monitor.py --model Qwen/Qwen3-0.6B --subset pilot --limit 2`.
2. **Monitor pass** (pod) — `experiments/local-monitor-activations/`:
   `g1_env.py` and `g1_attn_*.py` check the environment and the chunked attention;
   `g2_prep.py` builds the capped subset; `g3_make_pilot.py` + `g3_pilot.sh` is the
   pilot; `g4_make_run_order.py` → `g4_prelaunch.py` → `g4_chain.sh <cutoff>` runs
   the main pass and the contrast and counterfactual arms (about 11 h for the
   main pass); `g4_*_report.py` and `g6_integrity.py` read the outputs.
3. **Action-level label** (laptop) — `experiments/action-level-attack-label/`:
   `a1_export_fields.py` → draft `payload_signatures.json` from the objective
   text alone → `a3_validate.py` → `a4_draw.py` → write a verdict into each
   `a4_audit/NN.md` by hand → `a4_score.py` (same for `a4b_*` and `a4_audit_r2/`)
   → `a5_emit.py` → `b_placeholder.py` → `c_roc.py` → `d_expression_gap.py`.
4. **Probes and figures** — `experiments/latent-vs-verbalized/`:
   `p0_consolidate.py` (needs the stage-2 activation checkpoints) → `p0_gate.py`
   → `p1_split.py` → `p2_probes.py` → `p3_baselines.py` → `p4_result.py` →
   `p6`–`p11` → `p12_hop_truncation.py` (pod) → `fig_F1.py` … `fig_F4.py`,
   `fig_examples.py`. The `fig_*` and `p4`/`p7`/`p11` scripts read only committed
   raw files and run anywhere.

Without a GPU and without the checkpoints, every number in every `RESULTS.md`
still recomputes from `results/raw/`; what cannot be re-run is the probe fitting
and the monitor pass themselves.

## Provenance

This repo is a cleaned snapshot of the working repository
`thareqyusuf/judge-llm-l-probe-experiment-on-linux-arena` (`main` at `b51477d`,
with the branch `exp/latent-vs-verbalized` at `103ea55` folded in). History was
not carried over, so commit hashes quoted inside raw files and `RESULTS.md`
entries refer to that original repository.
