# Reproducing apple-bottom benchmarks

## Environment

- Hardware: Apple M2 Max (38-core GPU)
- macOS: 26.2 (build 25C56)
- Xcode: 26.0.1 (build 17A400)
- apple-bottom: commit `0805de5` (HEAD at time of capture)

## Build

```bash
make clean && make test   # expect 113/113 passing
make bench                # builds and runs all bench_* binaries
                          # (bench_dgemm, bench_sgemm, bench_zgemm,
                          #  bench_pool, bench_dsyrk, bench_zherk,
                          #  bench_async)
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
