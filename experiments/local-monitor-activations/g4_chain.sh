#!/usr/bin/env bash
# G4' chained driver (TASK-ver3). One tmux window, every stage resumable, every stage its own out dir.
#   tmux new-window -t mi -n g4 "bash experiments/local-monitor-activations/g4_chain.sh 2026-09-11T19:00:00Z"
# Arg 1 = the user's cut-off (UTC ISO). Rule: within 6 h of the cut-off no new stage is launched (exit 99);
# the optional k=4 stage needs > 4 h remaining. Any non-zero stage exit (130 = interrupted by signal) stops the chain;
# relaunching the chain resumes every stage (done keys are skipped by the runner).
set -uo pipefail
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO"
SLUG=local-monitor-activations
EXP="experiments/$SLUG"
OUT="checkpoints/$SLUG/out"
LOG="logs/$SLUG/g4_chain.log"
CUTOFF="${1:?usage: g4_chain.sh <cutoff UTC ISO, e.g. 2026-09-11T19:00:00Z>}"
CUTOFF_EPOCH=$(date -u -d "$CUTOFF" +%s) || { echo "bad cutoff $CUTOFF"; exit 2; }
NO_NEW_STAGE_AFTER=$((CUTOFF_EPOCH - 6 * 3600))
K4_NEEDS_AFTER=$((CUTOFF_EPOCH - 4 * 3600))
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True HF_HOME="${HF_HOME:-/workspace/hf}"
COMMON=(--attn chunked_eager --attn-chunk-size 256 --max-prompt-tokens 32000 --reasoning-effort low
        --template-date 2026-09-10 --max-new-tokens 1024 --checkpoint-every 25 --gate-at 0)
ts() { date -u +%Y-%m-%dT%H:%M:%SZ; }

run_stage() {   # run_stage <stage> <run_order.json> [extra runner args]
  local name="$1" ro="$2"; shift 2
  local now; now=$(date -u +%s)
  if (( now > NO_NEW_STAGE_AFTER )); then
    echo "== $(ts) stage $name NOT launched: within 6 h of the cut-off $CUTOFF (TASK-ver3 hard rule)"; return 99
  fi
  echo "== $(ts) stage $name start (commit $(git rev-parse --short HEAD), run order $(sha256sum "$ro" | cut -c1-16))"
  python "$EXP/run_monitor.py" "${COMMON[@]}" --run-order "$ro" --stage "$name" --out "$OUT/$name" "$@" 2>&1 \
    | grep --line-buffered -v -E "Loading weights|Fetching|Warning: You are sending"
  local rc=${PIPESTATUS[0]}
  echo "== $(ts) stage $name exit $rc"
  return "$rc"
}

{
  echo "== $(ts) G4' chain start; cut-off $CUTOFF; no new stage after $(date -u -d @$NO_NEW_STAGE_AFTER +%Y-%m-%dT%H:%M:%SZ)"
  RO="$EXP/run_order.json"
  run_stage low_cached "$RO"      || exit $?
  run_stage consistent_high "$RO" || exit $?
  run_stage main "$RO"            || exit $?
  if [ -f "$EXP/subset_payload_enrich.json" ]; then
    ROE="$EXP/run_order_enrich.json"
    [ -f "$ROE" ] || python "$EXP/g4_make_run_order.py" --enrich-out "$ROE"
    run_stage enrich "$ROE"       || exit $?
  else
    echo "== $(ts) G4.4 payload enrichment SKIPPED: $EXP/subset_payload_enrich.json does not exist"
  fi
  run_stage cf_neutralized "$RO"  || exit $?
  run_stage cf_random_scrub "$RO" || exit $?
  now=$(date -u +%s)
  if (( now < K4_NEEDS_AFTER )); then
    run_stage sampled_k4 "$RO"    || exit $?
  else
    echo "== $(ts) G4.6 optional k=4 SKIPPED: fewer than 4 h remain before the cut-off $CUTOFF"
  fi
  echo "== $(ts) G4' chain done"
} 2>&1 | tee -a "$LOG"
exit "${PIPESTATUS[0]}"
