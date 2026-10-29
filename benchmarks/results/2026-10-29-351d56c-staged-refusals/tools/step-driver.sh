#!/bin/bash
# Queues $SCRIPT in the machine lock, on AC power only (the release check's part-4 pattern):
# acgate.sh waits for AC before the lock wait, inlock.sh refuses a lock taken on battery (exit 75),
# and $SCRIPT itself exits 75 when the power drops between stages; each 75 queues again.
set -uo pipefail
SCRIPT=${1:?usage: step-driver.sh <script>}
OUT=~/qe-test/xp/runs.noindex/v150-staged-refusals-2026-10-27
T=~/qe-test/xp/runs.noindex/v150-qe-release-check/tools
ML=~/qe-test/xp/tools/machlock.py
export ACGATE_LOG=$OUT/logs/acgate.log
log() { echo "== $(date '+%F %T') $*"; }
( while :; do echo "$(date +%s) $(pmset -g batt | head -2 | tr '\n' ' ' | sed -E "s/Now drawing from '([^']*)'.*InternalBattery-0 \(id=[0-9]+\)[[:space:]]*/\1; /")"; sleep 20; done ) >> "$OUT/power.log" 2>&1 &
PLOG=$!
trap 'kill $PLOG 2>/dev/null' EXIT
while :; do
  log "waiting for AC power and the machine lock"
  "$T/acgate.sh" "$ML" bench -- "$T/inlock.sh" "$OUT/tools/$SCRIPT"
  rc=$?
  [ $rc -eq 75 ] || break
  sleep 60
done
log "$(basename "$SCRIPT") ended rc=$rc"
