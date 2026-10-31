#!/bin/bash
# Step 3 (inside `machlock.py bench`, behind acgate.sh and inlock.sh; step-driver.sh retries it while
# it exits 75): bit identity on real QE data, and the fix's ci-local.
#   verify   an EXPERIMENT-ONLY build (the fix 351d56c + AB_VERIFY_STAGED in blas_wrapper.c,
#            uncommitted, never for release): every call the staged path keeps is redone on the host
#            path from the same C and compared bit for bit. QE B 1x1, A 8x1 and B 8x1 in gpu mode.
#   ci       make ci-local on the fix's worktree (351d56c).
# Resumable as step2.sh: state/<stage>.done, exit 75 on battery between stages.
set -uo pipefail
OUT=<runs>
VER=<worktree>-verify
FIXWT=<worktree>
RUN=$FIXWT/benchmarks/qe/run_campaign.sh
export PSEUDO_DIR=~/qe-test/val1-o3-2026-10-14/pseudo
ST=$OUT/state; mkdir -p "$ST"
log() { echo "== $(date '+%F %T') $*"; }
plan() { printf '%s\n' "# label input np omp reps modes args" "$2" > "$OUT/plans/$1.txt"; echo "$OUT/plans/$1.txt"; }
todo() {
  [ -f "$ST/$1.done" ] && return 1
  pmset -g batt | head -1 | grep -q "'AC Power'" || { log "on battery before $1: releasing the lock"; exit 75; }
  log "stage $1"; return 0
}
done_() { touch "$ST/$1.done"; }
run() { todo "$1" || return 0; PW_X="$2" "$RUN" "$(plan "$1" "$3")" "$OUT" && done_ "$1"; }
log "step3v (re)start; $(pmset -g batt | head -2 | tr '\n' ' ')"

if todo ver-build; then
  cd "$VER" || exit 1
  make -j8 lib > "$OUT/logs/verify-build.log" 2>&1 || { log "verify build failed: $(tail -3 "$OUT/logs/verify-build.log")"; exit 1; }
  "$OUT/tools/freeze.sh" "$VER" "EXPERIMENT ONLY, never for release: fix 351d56c + AB_VERIFY_STAGED (redo each kept staged call on the host path and compare)" > "$ST/ver.path" || exit 1
  log "verify frozen $(cat "$ST/ver.path")"; done_ ver-build
fi
VPW=$(cat "$ST/ver.path")/pw.x
run ver-B1 "$VPW" "verify-351d56c B_si64_160b_k222 1 1 1 gpu AB_VERIFY_STAGED=1"
run ver-A8 "$VPW" "verify-351d56c A_si64_500b_k222 8 1 1 gpu -nk 4 AB_VERIFY_STAGED=1"
run ver-B8 "$VPW" "verify-351d56c B_si64_160b_k222 8 1 1 gpu -nk 4 AB_VERIFY_STAGED=1"
for d in "$OUT"/runs/verify-351d56c/*/gpu_r*; do
  log "verify $(basename "$(dirname "$d")") $(basename "$d"): $(grep -ah '^AB_VERIFY_STAGED\[' "$d/err.txt" | awk '{c[$2]+=substr($3,12); d[$2]+=substr($4,14); e[$2]+=substr($5,14)} END {for (k in c) printf "%s kept=%d differ_calls=%d differ_elems=%d; ", k, c[k], d[k], e[k]}')"
done
if todo ci; then
  cd "$FIXWT" || exit 1
  log "ci-local at $(git rev-parse --short HEAD), $(git status --porcelain --untracked-files=no | wc -l | tr -d ' ') modified"
  if make ci-local > "$OUT/logs/fix-ci-local.log" 2>&1; then log "ci-local passed: $(grep -aE 'passed|PASS' "$OUT/logs/fix-ci-local.log" | tail -3 | tr '\n' ' ')"
  else log "ci-local FAILED: $(grep -aE 'FAIL|rror' "$OUT/logs/fix-ci-local.log" | head -5 | tr '\n' ' ')"; fi
  done_ ci
fi
log "step3v done; $(pmset -g batt | head -1)"
