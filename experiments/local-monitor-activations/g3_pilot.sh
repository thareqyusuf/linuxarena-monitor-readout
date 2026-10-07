#!/usr/bin/env bash
# G3 pilot driver: three runs in sequence, each loads the model once (~35 s). Run inside tmux:
#   tmux new-window -t mi -n g3 "bash experiments/local-monitor-activations/g3_pilot.sh"
set -euo pipefail
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO"
SLUG=local-monitor-activations
TS="$(date -u +%Y%m%dT%H%M%SZ)"
LOG="logs/$SLUG/g3_pilot_$TS.log"
OUT="checkpoints/$SLUG/out"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True HF_HOME="${HF_HOME:-/workspace/hf}"
COMMON=(--attn chunked_eager --max-prompt-tokens 32000 --reasoning-effort low --template-date 2026-09-10 --checkpoint-every 10)
{
  echo "== g3 pilot $TS  commit $(git rev-parse --short HEAD)"
  echo "== run 1/3: 30 actions, original, chunk 256"
  python experiments/$SLUG/run_monitor.py "${COMMON[@]}" --attn-chunk-size 256 --render original \
      --subset experiments/$SLUG/subset_pilot.json --out "$OUT/pilot_c256"
  echo "== run 2/3: 20 actions, original, chunk 128 (noise floor)"
  python experiments/$SLUG/run_monitor.py "${COMMON[@]}" --attn-chunk-size 128 --render original \
      --subset experiments/$SLUG/subset_pilot_noise.json --out "$OUT/pilot_c128"
  echo "== run 3/3: 10 actions, neutralized, chunk 256"
  python experiments/$SLUG/run_monitor.py "${COMMON[@]}" --attn-chunk-size 256 --render neutralized \
      --subset experiments/$SLUG/subset_pilot_neut.json --out "$OUT/pilot_neut_c256"
  echo "== done $(date -u +%Y%m%dT%H%M%SZ)"
} 2>&1 | grep --line-buffered -v -E "Loading weights|Fetching" | tee "$LOG"
