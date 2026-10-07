#!/usr/bin/env bash
# G4 driver (TASK-ver2 rev.2 + the 2026-09-10 revisions): main pass, then the two counterfactual renders.
# Run inside tmux:  tmux new-window -t mi -n g4 "bash experiments/local-monitor-activations/g4_run.sh"
# The main pass evaluates the 200-action gate and exits 3 if it halts; the counterfactual renders are then skipped.
set -uo pipefail
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO"
SLUG=local-monitor-activations
LOG="logs/$SLUG/g4_run.log"
OUT="checkpoints/$SLUG/out"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True HF_HOME="${HF_HOME:-/workspace/hf}"
COMMON=(--attn chunked_eager --attn-chunk-size 256 --max-prompt-tokens 32000 --reasoning-effort low
        --template-date 2026-09-10 --max-new-tokens 1024 --checkpoint-every 25)
{
  echo "== G4 start $(date -u +%Y%m%dT%H%M%SZ) commit $(git rev-parse --short HEAD)"
  echo "== pass 1/3: subset_main.json, render original, gate at 200"
  python experiments/$SLUG/run_monitor.py "${COMMON[@]}" --render original \
      --subset experiments/$SLUG/subset_main.json --out "$OUT/main" \
      --gate-at 200 --gate-min-frac-ge3 0.05 --gate-min-sd 0.2
  rc=$?
  if [ $rc -eq 3 ]; then echo "== HALTED by the gate; passes 2 and 3 skipped. $(date -u +%Y%m%dT%H%M%SZ)"; exit 3; fi
  if [ $rc -ne 0 ]; then echo "== pass 1 failed rc=$rc; stopping. $(date -u +%Y%m%dT%H%M%SZ)"; exit $rc; fi
  echo "== pass 2/3: subset_counterfactual.json, render neutralized"
  python experiments/$SLUG/run_monitor.py "${COMMON[@]}" --render neutralized \
      --subset experiments/$SLUG/subset_counterfactual.json --out "$OUT/cf_neutralized" --gate-at 0 || echo "== pass 2 rc=$?"
  echo "== pass 3/3: subset_counterfactual.json, render random_scrub"
  python experiments/$SLUG/run_monitor.py "${COMMON[@]}" --render random_scrub \
      --subset experiments/$SLUG/subset_counterfactual.json --out "$OUT/cf_random_scrub" --gate-at 0 || echo "== pass 3 rc=$?"
  echo "== G4 done $(date -u +%Y%m%dT%H%M%SZ)"
} 2>&1 | grep --line-buffered -v -E "Loading weights|Fetching|Warning: You are sending" | tee -a "$LOG"
