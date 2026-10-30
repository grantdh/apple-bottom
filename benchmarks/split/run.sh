#!/bin/bash
# Usage: benchmarks/split/run.sh <outdir> [suite ...]
#
# Runs build/bench_split's cases, suite by suite, into <outdir>/<suite>/ (benchmarks/split/README.md):
#   smoke          D2048, Z2048 at 1x8; UM160, RK160 at 4x2 (about 30 s)
#   sq1x8          DGEMM and ZGEMM squares 2048-8192, 1 process x 8 Accelerate threads (about 10 min)
#   qe4x2, qe1x8   the QE-shaped updates and overlaps at 4 processes x 2 threads, then 1 x 8 (3 + 2 min)
#   qelike4x2      UN1000, UN512, RK500 with QE-like data (rows decaying over 9 decades), 4x2
#   qelike4x2-low  the same updates at GPU shares 0.06-0.32
#   noreg4x2       UN1000, UM160, RK500 with copied instead of registered operands
#   power4x2       UN1000, RK500 with 6 s runs per engine for wall power
#   sustain        80 s back-to-back runs of DGEMM and ZGEMM 8192^2 at 1x8, at the best share of
#                  sq1x8's case (measured first if sq1x8 has not run)
# Default: every suite but smoke, in that order (about 35 min).
#
# Run on an otherwise idle Mac on AC power. Every case checks `pmset` first, and on battery the script
# stops with exit status 75. A case already complete in <outdir> is skipped, and one that did not both
# start and end on AC power is moved aside, so running the script again resumes.
set -u
here=$(cd "$(dirname "$0")" && pwd)
bin=${BENCH_SPLIT:-$here/../../build/bench_split}
[ -x "$bin" ] || { echo "no $bin: run 'make build/bench_split' first" >&2; exit 2; }
[ $# -ge 1 ] || { sed -n 2,18p "$0" >&2; exit 2; }
out=$1; shift
suites=${*:-sq1x8 qe4x2 qe1x8 qelike4x2 qelike4x2-low noreg4x2 power4x2 sustain}
env_common=(AB_MODE=gpu AB_MIN_GPU_DIM=0 AB_STAGING_MB=8192 AB_REPORT=1)

onac() { pmset -g batt | head -1 | grep -q "'AC Power'"; }
note() { echo "$(date '+%F %T') $2: $(pmset -g batt | head -1) | load $(sysctl -n vm.loadavg)" >> "$1/power.log"; }
# go <dir> <prefix> <marker> <label> <command ...>: run one case into <dir>/<prefix>.out, unless that
# already holds a <marker> line; afterwards it must hold one and have started and ended on AC power,
# else the case's files are moved aside and the script stops
go() {
    local d=$1 pre=$2 mark=$3 label=$4 rc; shift 4
    grep -q "^$mark" "$d/$pre.out" 2>/dev/null && return 0
    mkdir -p "$d"
    note "$d" "before $label"
    onac || { note "$d" "not on AC power, stopping before $label"; exit 75; }
    "$@" > "$d/$pre.out" 2>&1
    rc=$?
    note "$d" "after $label rc=$rc"
    if [ $rc -ne 0 ] || ! grep -q "^$mark" "$d/$pre.out" ||
       ! grep -q "^start: Now drawing from 'AC Power'" "$d/$pre.out" || ! grep -q "^end: Now drawing from 'AC Power'" "$d/$pre.out"; then
        local aside="$d/incomplete-$(date +%Y%m%d-%H%M%S)"
        mkdir -p "$aside" && mv "$d/$pre".* "$aside/" 2>/dev/null
        note "$d" "moved $pre.* to $aside"
        onac || exit 75
        exit 1
    fi
}
# one <suite> <np> <veclib threads> <data> <case> [VAR=value ...]
one() {
    local sub=$1 np=$2 vl=$3 data=$4 c=$5; shift 5
    local pre=$c.np$np.d$data
    go "$out/$sub" "$pre" SUMMARY "$c np=$np veclib=$vl data=$data" \
        env "$@" "${env_common[@]}" VECLIB_MAXIMUM_THREADS="$vl" AB_TRACE="$out/$sub/$pre.trace" "$bin" run "$c" "$np" "$out/$sub" "$data"
}
# sus <case>: sustained runs at the GPU part g of sq1x8's fixed block for the same case
sus() {
    local c=$1 g
    one sq1x8 1 8 0 "$c"
    g=$(sed -nE 's/^SUMMARY .* g=([0-9]+) .*/\1/p' "$out/sq1x8/$c.np1.d0.out")
    go "$out/sustain" "$c.sustain" SUSTAIN "sustain $c g=$g" \
        env "${env_common[@]}" VECLIB_MAXIMUM_THREADS=8 AB_TRACE="$out/sustain/$c.sustain.trace" "$bin" sustain "$c" "$out/sustain" "$g" 80 30 20
}

QE="UN1000 UN512 UM160 UM320 UM1000 RK500 RK160"
for s in $suites; do
    case $s in
    smoke)         one smoke 1 8 0 D2048; one smoke 1 8 0 Z2048; one smoke 4 2 0 UM160; one smoke 4 2 0 RK160 ;;
    sq1x8)         for c in D2048 D4096 D8192 Z2048 Z4096 Z8192; do one sq1x8 1 8 0 "$c"; done ;;
    qe4x2)         for c in $QE; do one qe4x2 4 2 0 "$c"; done ;;
    qe1x8)         for c in $QE; do one qe1x8 1 8 0 "$c"; done ;;
    qelike4x2)     for c in UN1000 UN512 RK500; do one qelike4x2 4 2 1 "$c"; done ;;
    qelike4x2-low) for c in UN512L UN1000L; do one qelike4x2-low 4 2 1 "$c"; done ;;
    noreg4x2)      for c in UN1000 UM160 RK500; do one noreg4x2 4 2 0 "$c" SPLIT_NOREG=1; done ;;
    power4x2)      for c in UN1000 RK500; do one power4x2 4 2 0 "$c" SPLIT_POWER_S=6 SPLIT_IDLE_S=6; done ;;
    sustain)       sus D8192; sus Z8192 ;;
    *)             echo "unknown suite: $s" >&2; exit 2 ;;
    esac
done
