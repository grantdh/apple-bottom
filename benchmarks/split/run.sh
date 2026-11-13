#!/bin/bash
# Usage: benchmarks/split/run.sh <outdir> [suite ...]
#
# Runs build/bench_split's cases, suite by suite, into <outdir>/<suite>/ (benchmarks/split/README.md):
#   smoke          D2048, Z2048 at 1x8; UM160, RK160 at 4x2 (about 30 s)
#   sq1x8          DGEMM and ZGEMM squares 2048-8192, 1 process x 8 Accelerate threads (about 10 min)
#   qe4x2, qe1x8   the QE-shaped updates and overlaps at 4 processes x 2 threads, then 1 x 8 (3 + 2 min)
#   qe8x1          pw.x's calls at 8 ranks x 1 thread, -nk 4 (input A: zUN.k1, zRK.t1, zRK.t2; B: zUM.k2,
#                  zRK.t1), at 8 processes x 1 Accelerate thread
#   qelike4x2      UN1000, UN512, RK500 with QE-like data (rows decaying over 9 decades), 4x2
#   qelike4x2-low  the same updates at GPU shares 0.06-0.32
#   noreg4x2       UN1000, UM160, RK500 with copied instead of registered operands
#   power4x2       UN1000, RK500 with 6 s runs per engine for wall power
#   sustain        80 s back-to-back runs of DGEMM and ZGEMM 8192^2 at 1x8, at the best share of
#                  sq1x8's case (measured first if sq1x8 has not run)
#   g3-1x8-s8192, g3-1x8-s2048, g3-4x2-s8192, g3-4x2-s2048 (or g3 for all four)
#                  G3 (library seam against user code): the squares and the QE-shaped cases with copied
#                  operands (SPLIT_NOREG=1) at 1x8 and 4x2, AB_STAGING_MB 8192 or 2048; no Z8192 at 4x2;
#                  6 rounds of 2 repeats, so the arms' order is balanced and intervals can be by round;
#                  no AB_TRACE or AB_REPORT, which profile the user split's GPU slices but not the seam
# Default: every suite but smoke and the G3 suites, in that order (about 60 min, 27 of them the sustained
# runs; from the 2026-10-28 timings with the arms, openers, gaps and warm-up added).
#
# Run on an otherwise idle Mac on AC power from an adapter of at least SPLIT_MIN_W watts (default 90).
# Before every case the script also wants the battery at SPLIT_MIN_BATT percent or more (default 95)
# and neither charging nor finishing its charge: it waits up to SPLIT_BATT_WAIT_S seconds (default
# 900) for that, and checks AC and the adapter again after waiting. It stops with exit status 75 when a
# check fails (after the wait, for the battery). A case already complete in <outdir> is skipped. A case
# that fails is moved aside, so running the script again resumes: with exit status 75 if its report
# does not start and end on AC power from an adapter of SPLIT_MIN_W watts or more (or the power is not
# right now), else 1.
# SPLIT_DRY=1 makes the same checks (without waiting) and prints each case's command instead.
set -u
here=$(cd "$(dirname "$0")" && pwd)
bin=${BENCH_SPLIT:-$here/../../build/bench_split}
[ -x "$bin" ] || { echo "no $bin: run 'make build/bench_split' first" >&2; exit 2; }
[ $# -ge 1 ] || { awk 'NR > 1 && /^set -u/ { exit } NR > 1' "$0" >&2; exit 2; }
out=$1; shift
suites=
for s in ${*:-sq1x8 qe4x2 qe1x8 qe8x1 qelike4x2 qelike4x2-low noreg4x2 power4x2 sustain}; do
    [ "$s" = g3 ] && suites="$suites g3-1x8-s8192 g3-1x8-s2048 g3-4x2-s8192 g3-4x2-s2048" || suites="$suites $s"
done
env_common=(AB_MODE=gpu AB_MIN_GPU_DIM=0 AB_STAGING_MB=8192 AB_REPORT=1)
min_w=${SPLIT_MIN_W:-90}
min_batt=${SPLIT_MIN_BATT:-95}
batt_wait=${SPLIT_BATT_WAIT_S:-900}
dry=${SPLIT_DRY:-}

onac() { pmset -g batt | head -1 | grep -q "'AC Power'"; }
adapter_w() { pmset -g adapter | sed -nE 's/^ *Wattage = ([0-9]+)W.*/\1/p' | head -1; }
adapter_ok() { local w; w=$(adapter_w); [ -n "$w" ] && [ "$w" -ge "$min_w" ]; }
batt_line() { pmset -g batt | sed -n 2p; }
batt_ok() {   # at least min_batt percent, not charging and not finishing its charge (as the AB_SPLIT pilot)
    local b; b=$(batt_line)
    ! echo "$b" | grep -qE '; (charging|finishing charge);' && [ "$(echo "$b" | grep -oE '[0-9]+%' | tr -d %)" -ge "$min_batt" ] 2>/dev/null
}
ac_ok() {   # ac_ok <what>: on AC power from an adapter of at least min_w watts, else say so on stderr
    onac || { echo "not on AC power, stopping $1: exit 75" >&2; return 1; }
    adapter_ok || { echo "adapter $(adapter_w) W, under $min_w W, stopping $1: exit 75" >&2; return 1; }
}
power_ok() {   # power_ok <what>: ac_ok and the battery as batt_ok wants (waiting up to batt_wait s for it,
               # except in a dry run), else say so on stderr; ac_ok again after a wait, which unplugging
               # the adapter can end (the battery then reads discharging)
    ac_ok "$1" || return 1
    local t=0
    while ! batt_ok; do
        [ -z "$dry" ] && [ $t -lt "$batt_wait" ] || { echo "battery [$(batt_line)], stopping $1: exit 75" >&2; return 1; }
        sleep 30; t=$((t + 30))
    done
    [ $t -eq 0 ] || ac_ok "$1"
}
report_power_ok() {   # report_power_ok <file>: its start and end lines say AC power from an adapter of min_w W or more
    local w
    for w in start end; do
        grep -q "^$w: Now drawing from 'AC Power'" "$1" || return 1
        [ "$(sed -nE "s/^$w: .*\| adapter ([0-9]+) W.*/\1/p" "$1")" -ge "$min_w" ] 2>/dev/null || return 1
    done
}
note() { echo "$(date '+%F %T') $2: $(pmset -g batt | tr '\n' ' ') | adapter $(adapter_w) W | load $(sysctl -n vm.loadavg)" >> "$1/power.log"; }
power_ok "before the first case" || exit 75
# go <dir> <prefix> <marker> <label> <command ...>: run one case into <dir>/<prefix>.out, unless that
# already holds a <marker> line; afterwards it must hold one and have started and ended on AC power from
# an adapter of min_w W or more, else the case's files are moved aside and the script stops
go() {
    local d=$1 pre=$2 mark=$3 label=$4 rc; shift 4
    grep -q "^$mark" "$d/$pre.out" 2>/dev/null && return 0
    if [ -n "$dry" ]; then power_ok "before $label" || exit 75; echo "$label: $*"; return 0; fi
    mkdir -p "$d"
    power_ok "before $label" || { note "$d" "power check failed, stopping before $label"; exit 75; }
    note "$d" "before $label"
    "$@" > "$d/$pre.out" 2>&1
    rc=$?
    note "$d" "after $label rc=$rc"
    local power=ok
    report_power_ok "$d/$pre.out" || power=bad
    if [ $rc -ne 0 ] || ! grep -q "^$mark" "$d/$pre.out" || [ $power = bad ]; then
        local aside="$d/incomplete-$(date +%Y%m%d-%H%M%S)"
        mkdir -p "$aside" && mv "$d/$pre".* "$aside/" 2>/dev/null
        note "$d" "moved $pre.* to $aside"
        # 75: a power problem (the report's start or end line, or now), so a driver can wait and rerun;
        # 1: the case itself failed
        { [ $power = ok ] && ac_ok "after $label"; } || exit 75
        exit 1
    fi
}
# one <suite> <np> <veclib threads> <data> <case> [VAR=value ...]: the VAR=value settings come last,
# so they override env_common and the trace file (env keeps the last value of a name; AB_TRACE= with
# no value turns the trace off)
one() {
    local sub=$1 np=$2 vl=$3 data=$4 c=$5; shift 5
    local pre=$c.np$np.d$data
    go "$out/$sub" "$pre" SUMMARY "$c np=$np veclib=$vl data=$data" \
        env "${env_common[@]}" VECLIB_MAXIMUM_THREADS="$vl" AB_TRACE="$out/$sub/$pre.trace" "$@" "$bin" run "$c" "$np" "$out/$sub" "$data"
}
# sus <case>: sustained runs at the GPU part g of sq1x8's fixed block for the same case
sus() {
    local c=$1 g
    one sq1x8 1 8 0 "$c"
    g=$(sed -nE 's/^SUMMARY .* g=([0-9]+) .*/\1/p' "$out/sq1x8/$c.np1.d0.out" 2>/dev/null)
    go "$out/sustain" "$c.sustain" SUSTAIN "sustain $c g=$g" \
        env "${env_common[@]}" VECLIB_MAXIMUM_THREADS=8 AB_TRACE="$out/sustain/$c.sustain.trace" "$bin" sustain "$c" "$out/sustain" "$g" 80 30 20
}

QE="UN1000 UN512 UM160 UM320 UM1000 RK500 RK160"
QE8="A8_UN512 A8_RK512 A8_RK1000 B8_UM1152 B8_RK1152"
SQ="D2048 D4096 D8192 Z2048 Z4096 Z8192"
SQ4X2="D2048 D4096 D8192 Z2048 Z4096"   # Z8192 at 4x2 does not fit in 64 GB (README, Memory)
for s in $suites; do
    case $s in
    smoke)         one smoke 1 8 0 D2048; one smoke 1 8 0 Z2048; one smoke 4 2 0 UM160; one smoke 4 2 0 RK160 ;;
    sq1x8)         for c in $SQ; do one sq1x8 1 8 0 "$c"; done ;;
    qe4x2)         for c in $QE; do one qe4x2 4 2 0 "$c"; done ;;
    qe1x8)         for c in $QE; do one qe1x8 1 8 0 "$c"; done ;;
    qe8x1)         for c in $QE8; do one qe8x1 8 1 0 "$c"; done ;;
    qelike4x2)     for c in UN1000 UN512 RK500; do one qelike4x2 4 2 1 "$c"; done ;;
    qelike4x2-low) for c in UN512L UN1000L; do one qelike4x2-low 4 2 1 "$c"; done ;;
    noreg4x2)      for c in UN1000 UM160 RK500; do one noreg4x2 4 2 0 "$c" SPLIT_NOREG=1; done ;;
    power4x2)      for c in UN1000 RK500; do one power4x2 4 2 0 "$c" SPLIT_POWER_S=6 SPLIT_IDLE_S=6; done ;;
    sustain)       sus D8192; sus Z8192 ;;
    g3-1x8-s8192|g3-1x8-s2048|g3-4x2-s8192|g3-4x2-s2048)
        case $s in g3-1x8-*) np=1 vl=8 cs="$SQ $QE" ;; *) np=4 vl=2 cs="$SQ4X2 $QE" ;; esac
        for c in $cs; do one "$s" "$np" "$vl" 0 "$c" SPLIT_NOREG=1 AB_STAGING_MB="${s##*-s}" SPLIT_ROUNDS=6 SPLIT_REPS=2 AB_TRACE= AB_REPORT=; done ;;
    *)             echo "unknown suite: $s" >&2; exit 2 ;;
    esac
done
