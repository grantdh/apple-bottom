#!/bin/bash
# Step 4 (inside `machlock.py bench`): the /simplify refactor of the fix, 8ebe73c, which changes no result
# by design. Correctness only, no timings, so it runs on battery too; each stage logs the power.
#   r-test, r-ci  make test and make ci-local on the fix's worktree at 8ebe73c.
#   ver2-*        an EXPERIMENT-ONLY build (8ebe73c + AB_VERIFY_STAGED in blas_wrapper.c, uncommitted,
#                 never for release; frozen as d34c064c0e21): every call the staged path keeps is redone
#                 on the host path from the same C and compared bit for bit. QE B 1x1, A 8x1 and B 8x1
#                 in gpu mode, as step3v.sh did for 351d56c.
# Resumable as step3v.sh: state/<stage>.done.
set -uo pipefail
OUT=<runs>
FIXWT=<worktree>
RUN=$FIXWT/benchmarks/qe/run_campaign.sh
VPW=<ab-frozen>/d34c064c0e21/pw.x
export PSEUDO_DIR=~/qe-test/val1-o3-2026-10-14/pseudo
ST=$OUT/state; mkdir -p "$ST"
power() { pmset -g batt | head -2 | tr '\n' ' ' | sed -E "s/Now drawing from '([^']*)'.*InternalBattery-0 \(id=[0-9]+\)[[:space:]]*/\1; /"; }
log() { echo "== $(date '+%F %T') $*"; }
todo() { [ -f "$ST/$1.done" ] && return 1; log "stage $1; $(power)"; return 0; }
done_() { touch "$ST/$1.done"; }
plan() { printf '%s\n' "# label input np omp reps modes args" "$2" > "$OUT/plans/$1.txt"; echo "$OUT/plans/$1.txt"; }
run() { todo "$1" || return 0; PW_X="$2" "$RUN" "$(plan "$1" "$3")" "$OUT" && done_ "$1"; }
log "step4 (re)start at $(git -C "$FIXWT" rev-parse --short HEAD), $(git -C "$FIXWT" status --porcelain --untracked-files=no | wc -l | tr -d ' ') modified; $(power)"

if todo r-test; then
  if make -C "$FIXWT" test > "$OUT/logs/refactor-make-test.log" 2>&1; then
    log "make test passed: $(grep -aE 'Results:|Golden bits|test_staging|passed' "$OUT/logs/refactor-make-test.log" | tail -6 | tr '\n' ' ')"
  else
    log "make test FAILED: $(grep -aE 'FAIL|rror' "$OUT/logs/refactor-make-test.log" | head -5 | tr '\n' ' ')"
  fi
  done_ r-test
fi
if todo r-ci; then
  if make -C "$FIXWT" ci-local > "$OUT/logs/refactor-ci-local.log" 2>&1; then
    log "ci-local passed: $(grep -aE 'passed|PASS' "$OUT/logs/refactor-ci-local.log" | tail -3 | tr '\n' ' ')"
  else
    log "ci-local FAILED: $(grep -aE 'FAIL|rror' "$OUT/logs/refactor-ci-local.log" | head -5 | tr '\n' ' ')"
  fi
  done_ r-ci
fi
run ver2-B1 "$VPW" "verify-8ebe73c B_si64_160b_k222 1 1 1 gpu AB_VERIFY_STAGED=1"
run ver2-A8 "$VPW" "verify-8ebe73c A_si64_500b_k222 8 1 1 gpu -nk 4 AB_VERIFY_STAGED=1"
run ver2-B8 "$VPW" "verify-8ebe73c B_si64_160b_k222 8 1 1 gpu -nk 4 AB_VERIFY_STAGED=1"
for d in "$OUT"/runs/verify-8ebe73c/*/gpu_r*; do
  # Searched, not anchored: ranks' lines interleave in stderr.
  log "verify $(basename "$(dirname "$d")") $(basename "$d"): $(grep -aoh 'AB_VERIFY_STAGED\[[0-9]*\] [a-z]* kept_calls=[0-9]* differ_calls=[0-9]* differ_elems=[0-9]*' "$d/err.txt" | awk '{c[$2]+=substr($3,12); d[$2]+=substr($4,14); e[$2]+=substr($5,14); n[$2]++} END {for (k in c) printf "%s ranks=%d kept=%d differ_calls=%d differ_elems=%d; ", k, n[k], c[k], d[k], e[k]}') differing-call lines: $(grep -ac 'elements differ' "$d/err.txt")"
done
log "step4 done; $(power)"
