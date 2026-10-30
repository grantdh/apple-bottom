#!/bin/bash
# CPU cores in use per mode: bench_split built with bench_split_cores.patch, which adds the process's CPU
# time (getrusage, all threads) over each timed call of the fixed block and prints it as CORES lines.
# usage: cores.sh <absolute path to the patched bench_split> <output directory>
# Run it with no other benchmark on the machine, on AC power.
set -u
B=$1 OUT=$2
mkdir -p "$OUT"
echo "start $(date '+%F %T') $(pmset -g batt | head -1) load $(sysctl -n vm.loadavg)" > "$OUT/log.txt"
run() {  # run <case> <processes> <Accelerate threads per process>
    env VECLIB_MAXIMUM_THREADS="$3" AB_MODE=gpu AB_MIN_GPU_DIM=0 AB_STAGING_MB=8192 AB_REPORT=1 \
        "$B" run "$1" "$2" "$OUT" 0 > "$OUT/$1.np$2.out" 2>&1
    echo "$1 np=$2 threads=$3 rc=$? load $(sysctl -n vm.loadavg)" >> "$OUT/log.txt"
}
run D2048 1 8
run Z2048 1 8
run RK500 4 2
echo "end $(date '+%F %T') $(pmset -g batt | head -1)" >> "$OUT/log.txt"
grep -h '^CORES' "$OUT"/*.out
