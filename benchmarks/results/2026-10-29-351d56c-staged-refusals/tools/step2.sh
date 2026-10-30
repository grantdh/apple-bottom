#!/bin/bash
# Step 2, run inside `machlock.py bench` behind acgate.sh and inlock.sh (step2-driver.sh retries it
# while it exits 75). QE develop 86c214974 (pw.x.ab150's objects), inputs B at 1 x 1 and A, B at
# 8 x 1 -nk 4:
#   diag     the counter commit 2da6a89 (1.5.0 = 415cb5d + counter): build, its tests, freeze, one
#            AB_MODE=gpu run of B 1x1 with AB_TRACE and AB_TRACE_HASH (the refusal breakdown)
#   fix      fix/v1.5-staged-refusals HEAD (on release/v1.5.0 882d2d8): make test; only if it
#            passes: freeze, a hashed B 1x1 run compared call for call with diag's (and, if they
#            differ, 1.5.0's own, for determinism), then timings, 1.5.0 (755a46613bf2) and the
#            fix alternating in gpu mode, twice each, then one cpu run: B 1x1, A 8x1, B 8x1
#   cause    A 8x1 in gpu mode on the host path (AB_HOST_PACK=1), with AB_PREFETCH=0, and with an
#            experiment-only serial Gauss-3M library: what raised the GPU kernel time per call
# Every stage leaves state/<stage>.done, and a rerun skips the stages done. Before each stage the
# power is checked: on battery the script exits 75, the driver releases the lock, waits for AC
# power and queues again.
set -uo pipefail
OUT=<runs>
DIAG=<worktree>-diag
FIXWT=<worktree>
SER=<worktree>-serial3m
RUN=$FIXWT/benchmarks/qe/run_campaign.sh
BASE=<ab-frozen>/755a46613bf2/pw.x
export PSEUDO_DIR=~/qe-test/val1-o3-2026-10-14/pseudo
ST=$OUT/state; mkdir -p "$ST" "$OUT/traces"
log() { echo "== $(date '+%F %T') $*"; }
plan() { printf '%s\n' "# label input np omp reps modes args" "$2" > "$OUT/plans/$1.txt"; echo "$OUT/plans/$1.txt"; }
todo() {   # <stage>: whether it still has to run; on battery, exit 75
  [ -f "$ST/$1.done" ] && return 1
  pmset -g batt | head -1 | grep -q "'AC Power'" || { log "on battery before $1: releasing the lock"; exit 75; }
  log "stage $1"; return 0
}
done_() { touch "$ST/$1.done"; }
run() {    # <stage> <pw.x> <plan line>: one run_campaign.sh plan as its own stage
  todo "$1" || return 0
  local t                                     # a traced run starts from an empty trace directory
  t=$(sed -nE 's/.*AB_TRACE=([^ ]*).*/\1/p' <<< "$3")
  if [ -n "$t" ]; then rm -rf "$(dirname "$t")"; mkdir -p "$(dirname "$t")"; fi
  PW_X="$2" "$RUN" "$(plan "$1" "$3")" "$OUT" && done_ "$1"
}
log "step2 (re)start; $(pmset -g batt | head -2 | tr '\n' ' ')"

if todo diag-build; then
  cd "$DIAG" || exit 1
  log "diag $(git rev-parse --short HEAD), $(git status --porcelain --untracked-files=no | wc -l | tr -d ' ') modified"
  make -j8 lib build/test_trace build/test_blas_abi build/test_staging build/test_qe_golden > "$OUT/logs/diag-build.log" 2>&1 \
    || { log "diag build failed: $(tail -3 "$OUT/logs/diag-build.log")"; exit 1; }
  for t in test_trace "test_blas_abi test_exit_reports test_exponent_range test_zgemm_imbalance test_profile_path" test_staging test_qe_golden; do
    n=${t%% *}; ./build/$t > "$OUT/logs/diag-$n.log" 2>&1
    log "diag $t rc=$?: $(grep -aE 'Results|passed' "$OUT/logs/diag-$n.log" | tail -1)"
  done
  "$OUT/tools/freeze.sh" "$DIAG" "counter commit 2da6a89 on 415cb5d: the refusal breakdown" > "$ST/diag.path" || exit 1
  log "diag frozen $(cat "$ST/diag.path")"; done_ diag-build
fi
DIAGPW=$(cat "$ST/diag.path")/pw.x
run diag-run "$DIAGPW" "diag-2da6a89 B_si64_160b_k222 1 1 1 gpu AB_TRACE=$OUT/traces/diag-2da6a89/trace AB_TRACE_HASH=1"

if todo fix-test; then
  cd "$FIXWT" || exit 1
  SHA=$(git rev-parse --short HEAD)
  log "fix make test at $SHA, $(git status --porcelain --untracked-files=no | wc -l | tr -d ' ') modified"
  if make -j8 test > "$OUT/logs/fix-$SHA-make-test.log" 2>&1; then
    "$OUT/tools/freeze.sh" "$FIXWT" "fix $SHA on 882d2d8: the staged path keeps scaled lines and 4M" > "$ST/fix.path" || exit 1
    echo pass > "$ST/fix-test.result"
    log "fix make test passed; frozen $(cat "$ST/fix.path")"
  else
    echo fail > "$ST/fix-test.result"
    log "fix make test FAILED: $(grep -aE 'FAIL|rror' "$OUT/logs/fix-$SHA-make-test.log" | head -5 | tr '\n' ' ')"
  fi
  echo "$SHA" > "$ST/fix.sha"; done_ fix-test
fi
if [ "$(cat "$ST/fix-test.result")" != pass ]; then log "the fix failed make test: no fix runs"; exit 0; fi
FIXPW=$(cat "$ST/fix.path")/pw.x; SHA=$(cat "$ST/fix.sha")

run fix-run "$FIXPW" "fix-$SHA B_si64_160b_k222 1 1 1 gpu AB_TRACE=$OUT/traces/fix-$SHA/trace AB_TRACE_HASH=1"
if todo compare; then
  if python3 -I "$OUT/tools/compare_traces.py" "$OUT/traces/diag-2da6a89" "$OUT/traces/fix-$SHA" > "$OUT/logs/compare-diag-fix.txt" 2>&1; then
    log "fix vs diag: $(tail -1 "$OUT/logs/compare-diag-fix.txt")"
  else
    log "fix vs diag DIFFER: $(head -3 "$OUT/logs/compare-diag-fix.txt" | tr '\n' ' ')"
    run base-run "$BASE" "base-755a B_si64_160b_k222 1 1 1 gpu AB_TRACE=$OUT/traces/base-755a/trace AB_TRACE_HASH=1"
    python3 -I "$OUT/tools/compare_traces.py" "$OUT/traces/diag-2da6a89" "$OUT/traces/base-755a" > "$OUT/logs/compare-diag-base.txt" 2>&1
    log "diag vs 1.5.0: $(tail -1 "$OUT/logs/compare-diag-base.txt")"
  fi
  done_ compare
fi

for inp in "B_si64_160b_k222 1" "A_si64_500b_k222 8" "B_si64_160b_k222 8"; do
  read -r I NP <<< "$inp"
  ARGS=""; [ "$NP" = 8 ] && ARGS="-nk 4"
  T=${I:0:1}$NP
  for r in 1 2; do
    run "$T-base-$r" "$BASE"  "time-755a-np$NP $I $NP 1 1 gpu $ARGS"
    run "$T-fix-$r"  "$FIXPW" "time-$SHA-np$NP $I $NP 1 1 gpu $ARGS"
  done
  run "$T-cpu" "$FIXPW" "time-$SHA-np$NP $I $NP 1 1 cpu $ARGS"
done

if todo ser-build; then
  cd "$SER" || exit 1
  make -j8 lib > "$OUT/logs/serial3m-build.log" 2>&1 || { log "serial3m build failed"; exit 1; }
  "$OUT/tools/freeze.sh" "$SER" "EXPERIMENT ONLY, never for release: 415cb5d with Gauss-3M's products in a serial encoder" > "$ST/ser.path" || exit 1
  log "serial3m frozen $(cat "$ST/ser.path")"; done_ ser-build
fi
SERPW=$(cat "$ST/ser.path")/pw.x
A="A_si64_500b_k222 8 1 1 gpu -nk 4"
run cause-host      "$BASE"  "cause-755a-hostpack $A AB_HOST_PACK=1"
run cause-host-pf0  "$BASE"  "cause-755a-hostpack-pf0 $A AB_HOST_PACK=1 AB_PREFETCH=0"
run cause-ser       "$SERPW" "cause-serial3m-hostpack $A AB_HOST_PACK=1"
run cause-ser-pf0   "$SERPW" "cause-serial3m-hostpack-pf0 $A AB_HOST_PACK=1 AB_PREFETCH=0"
log "step2 done; $(pmset -g batt | head -1)"
