# Reproducing apple-bottom measurements

The performance numbers in the README and VAL-1 each have a dated directory
under `benchmarks/results/` that names the machine, OS, toolchain and
library build they came from:

| What | Command | Results |
|---|---|---|
| Native-API GEMM throughput | `make build/bench_paper && ./build/bench_paper --csv` (three runs) | `benchmarks/results/2026-10-14-3253a22-bench-paper/` |
| BLAS-ABI GPU vs Accelerate, and the `AB_MODE=auto` table | `make calibrate` (about 15 minutes) | `benchmarks/results/2026-10-14-52f1880-calibrate/` |
| Quantum ESPRESSO validation (VAL-1) | `benchmarks/qe/run_campaign.sh` (see `benchmarks/qe/README.md`) | `benchmarks/results/2026-10-14-1dbdb52-qe-si64/` |
| Quantum ESPRESSO, 1.4.0 release check (QE compiled `-O3`, 1 and 8 ranks) | `benchmarks/qe/run_campaign.sh` | `benchmarks/results/2026-10-15-c5a1fa6-qe-si64-opt/` |
| Register prefetch: kernel GPU times, bit checks, BLAS-ABI timings | the folder's `pfprobe.m` (build and run lines in its header); `bench_rect_dgemm` / `bench_rect_zgemm` over its `shapes/qe_e2e.txt` | `benchmarks/results/2026-10-24-15ef7b9-prefetch/` |
| Quantum ESPRESSO, 1.5.0 release check (1.5.0 and 1.4.0 on one QE build) | `benchmarks/qe/run_campaign.sh` with the folder's `plans/` | `benchmarks/results/2026-10-28-415cb5d-qe-si64-release-check/` |
| One GEMM split between Accelerate and the GPU | `make build/bench_split`, then `benchmarks/split/run.sh <out>` (about 35 minutes; see `benchmarks/split/README.md`) | `benchmarks/results/2026-10-28-415cb5d-hybrid-split/` |
| Quantum ESPRESSO, why GPU calls left the staged path and the fix | `benchmarks/qe/run_campaign.sh` with the folder's `plans/` and `tools/` | `benchmarks/results/2026-10-29-351d56c-staged-refusals/` |

Environment of the 2026-10-14 runs: MacBook Pro (Mac14,6), Apple M2 Max, 38-core
GPU, 8P+4E CPU cores, 64 GB; macOS 27.0.1 (26A434); macOS SDK 26.0, Apple
clang 17.0.0. The later folders ran on the same machine and macOS and name
their own library builds. Run on AC power under `caffeinate -is`, with nothing else
using the GPU, and record `ab_build_info()` with the results (it prints
the library version and commit, `-dirty` if tracked files had changed).

## Build

```bash
make clean && make test   # every test but the chaos ones (counts: tests/README.md)
make ci-local             # every test + the V-2 convergence study
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
