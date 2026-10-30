#!/bin/bash
# Step 5 (inside `machlock.py bench`): step 4's verify run of 8ebe73c at A 8x1 found 2 of 1,789 kept ZGEMMs
# whose staged C differs from the host path's redo in every element, and one gpu_failed; no run before had
# either. Same check, alternating the refactor and the measured fix, both EXPERIMENT-ONLY builds with
# tools/verify_more.py (which path the redo took, both results' error against FP64 cblas, Metal
# command-buffer errors, GPU-vs-host scale mismatches): e397e91002ad = 8ebe73c, 67511d46c83c = 351d56c.
# A 8x1 twice each, B 8x1 once each. Correctness only (no timings): not AC-gated, the power is logged.
set -uo pipefail
OUT=<runs>
FIXWT=<worktree>
RUN=$FIXWT/benchmarks/qe/run_campaign.sh
NEW=<ab-frozen>/e397e91002ad/pw.x
OLD=<ab-frozen>/67511d46c83c/pw.x
export PSEUDO_DIR=~/qe-test/val1-o3-2026-10-14/pseudo
ST=$OUT/state; mkdir -p "$ST"
power() { pmset -g batt | head -2 | tr '\n' ' ' | sed -E "s/Now drawing from '([^']*)'.*InternalBattery-0 \(id=[0-9]+\)[[:space:]]*/\1; /"; }
log() { echo "== $(date '+%F %T') $*"; }
todo() { [ -f "$ST/$1.done" ] && return 1; log "stage $1; $(power)"; return 0; }
done_() { touch "$ST/$1.done"; }
plan() { printf '%s\n' "# label input np omp reps modes args" "$2" > "$OUT/plans/$1.txt"; echo "$OUT/plans/$1.txt"; }
run() { todo "$1" || return 0; PW_X="$2" "$RUN" "$(plan "$1" "$3")" "$OUT" && done_ "$1"; }
log "step5 (re)start; $(power)"
run v5-A8-new1 "$NEW" "verify5-8ebe73c A_si64_500b_k222 8 1 1 gpu -nk 4 AB_VERIFY_STAGED=1"
run v5-A8-old1 "$OLD" "verify5-351d56c A_si64_500b_k222 8 1 1 gpu -nk 4 AB_VERIFY_STAGED=1"
run v5-A8-new2 "$NEW" "verify5-8ebe73c A_si64_500b_k222 8 1 1 gpu -nk 4 AB_VERIFY_STAGED=1"
run v5-A8-old2 "$OLD" "verify5-351d56c A_si64_500b_k222 8 1 1 gpu -nk 4 AB_VERIFY_STAGED=1"
run v5-B8-new1 "$NEW" "verify5-8ebe73c B_si64_160b_k222 8 1 1 gpu -nk 4 AB_VERIFY_STAGED=1"
run v5-B8-old1 "$OLD" "verify5-351d56c B_si64_160b_k222 8 1 1 gpu -nk 4 AB_VERIFY_STAGED=1"
for d in "$OUT"/runs/verify5-*/*/gpu_r*; do
  # Searched, not anchored: ranks' lines interleave in stderr.
  log "$(basename "$(dirname "$(dirname "$d")")") $(basename "$(dirname "$d")") $(basename "$d"): $(grep -aoh 'AB_VERIFY_STAGED\[[0-9]*\] [a-z]* kept_calls=[0-9]* differ_calls=[0-9]* differ_elems=[0-9]*' "$d/err.txt" | awk '{c[$2]+=substr($3,12); d[$2]+=substr($4,14); n[$2]++} END {for (k in c) printf "%s ranks=%d kept=%d differ_calls=%d; ", k, n[k], c[k], d[k]}') cmdbuf_errors=$(grep -ac 'AB_CMDBUF_ERROR' "$d/err.txt") scale_mismatches=$(grep -ac 'AB_VERIFY_SCALES' "$d/err.txt")"
  grep -aoh 'AB_VERIFY_STAGED [dz]gemm [A-Z][A-Z] M=.*staged_record=-*[0-9]*' "$d/err.txt" | sed 's/^/     /'
  grep -aoh 'AB_CMDBUF_ERROR pid=[0-9]* code=[0-9]* [^A]*' "$d/err.txt" | sort | uniq -c | sed 's/^/     /'
done
log "step5 done; $(power)"
