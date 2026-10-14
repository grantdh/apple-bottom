# Reproducing apple-bottom measurements

The performance numbers in the README and VAL-1 each have a dated directory
under `benchmarks/results/` that names the machine, OS, toolchain and
library build they came from:

| What | Command | Results |
|---|---|---|
| Native-API GEMM throughput | `make build/bench_paper && ./build/bench_paper --csv` (three runs) | `benchmarks/results/2026-10-14-3253a22-bench-paper/` |
| BLAS-ABI GPU vs Accelerate, and the `AB_MODE=auto` table | `make calibrate` (about 15 minutes) | `benchmarks/results/2026-10-14-52f1880-calibrate/` |
| Quantum ESPRESSO validation (VAL-1) | `benchmarks/qe/run_campaign.sh` (see `benchmarks/qe/README.md`) | `benchmarks/results/2026-10-14-1dbdb52-qe-si64/` |

Environment of those runs: MacBook Pro (Mac14,6), Apple M2 Max, 38-core
GPU, 8P+4E CPU cores, 64 GB; macOS 27.0.1 (26A434); macOS SDK 26.0, Apple
clang 17.0.0. Run on AC power under `caffeinate -is`, with nothing else
using the GPU, and record `ab_build_info()` with the results (it prints
the library version and commit, `-dirty` if tracked files had changed).

## Build

```bash
make clean && make test   # 138 tests
make ci-local             # 156 tests + the V-2 convergence study
make bench                # builds and runs the bench_* binaries
```

## CSV emission to timestamped directory

```bash
DATE=$(date -u +%Y-%m-%d)
SHA=$(git rev-parse --short HEAD)
mkdir -p benchmarks/results/${DATE}-${SHA}
./build/bench_dgemm -o benchmarks/results/${DATE}-${SHA}/dgemm.csv
./build/bench_zgemm -o benchmarks/results/${DATE}-${SHA}/zgemm.csv
./build/bench_sgemm -o benchmarks/results/${DATE}-${SHA}/sgemm.csv
```

## Powermetrics capture (two-terminal pattern)

```bash
mkdir -p docs/vv/powermetrics
DATE=$(date -u +%Y-%m-%d)

# Terminal A (500ms sampling, 240 samples = 2 min):
sudo powermetrics --samplers gpu_power -i 500 -n 240 \
  | tee docs/vv/powermetrics/${DATE}-<bench-name>.txt

# Terminal B (start immediately after first sample prints):
./build/<bench-name>
# For sustained-boost characterization, loop to keep the GPU warm:
# for i in $(seq 1 10); do ./build/bench_dgemm; done
```

## Quick residency histogram

```bash
FILE=docs/vv/powermetrics/<capture>.txt
grep "GPU HW active residency" $FILE | grep -cE "1398 MHz:[ ]*[5-9][0-9]%"
# Returns number of samples at ≥50% residency at 1398 MHz
```
