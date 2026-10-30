#!/bin/bash
# Step 6 (inside `machlock.py bench`, behind acgate.sh and inlock.sh; step-driver.sh retries it while it
# exits 75): does the /simplify refactor 8ebe73c time as the measured fix 351d56c did? Both frozen
# (3ead2d56bc84 = 8ebe73c, edd6aa2db786 = 351d56c), same pw.x copy; per input and layout they alternate in
# gpu mode, twice each, as step2.sh's timings did. Resumable: state/<stage>.done; exit 75 on battery
# before a stage.
set -uo pipefail
OUT=/Users/grantheileman/qe-test/xp/runs.noindex/v150-staged-refusals-2026-10-27
FIXWT=/Users/grantheileman/Dev/Claude/apple-bottom/.claude/worktrees/staged-refusals
RUN=$FIXWT/benchmarks/qe/run_campaign.sh
NEW=/Users/grantheileman/qe-test/ab-frozen/3ead2d56bc84/pw.x
OLD=/Users/grantheileman/qe-test/ab-frozen/edd6aa2db786/pw.x
export PSEUDO_DIR=/Users/grantheileman/qe-test/val1-o3-2026-10-14/pseudo
ST=$OUT/state; mkdir -p "$ST"
log() { echo "== $(date '+%F %T') $*"; }
plan() { printf '%s\n' "# label input np omp reps modes args" "$2" > "$OUT/plans/$1.txt"; echo "$OUT/plans/$1.txt"; }
todo() {   # <stage>: whether it still has to run; on battery, exit 75
  [ -f "$ST/$1.done" ] && return 1
  pmset -g batt | head -1 | grep -q "'AC Power'" || { log "on battery before $1: releasing the lock"; exit 75; }
  log "stage $1"; return 0
}
done_() { touch "$ST/$1.done"; }
run() { todo "$1" || return 0; PW_X="$2" "$RUN" "$(plan "$1" "$3")" "$OUT" && done_ "$1"; }
log "step6 (re)start; $(pmset -g batt | head -2 | tr '\n' ' ')"
for inp in "B_si64_160b_k222 1" "A_si64_500b_k222 8" "B_si64_160b_k222 8"; do
  read -r I NP <<< "$inp"
  ARGS=""; [ "$NP" = 8 ] && ARGS="-nk 4"
  T=${I:0:1}$NP
  for r in 1 2; do
    run "t6-$T-new-$r" "$NEW" "t6-8ebe73c-np$NP $I $NP 1 1 gpu $ARGS"
    run "t6-$T-old-$r" "$OLD" "t6-351d56c-np$NP $I $NP 1 1 gpu $ARGS"
  done
done
log "step6 done; $(pmset -g batt | head -1)"
