# Native-API GEMM throughput, 2026-10-14

`build/bench_paper --csv`, three runs back to back under `caffeinate -is`
(1-minute load 1.1–1.3 at each start), library built at 3253a22
(`ab_build_info()`: `1.4.0 (3253a22fa37c-dirty)`; the uncommitted files
were documentation only, nothing under `src/` or `include/`).

Machine: MacBook Pro (Mac14,6), Apple M2 Max, 38-core GPU, 8P+4E CPU, 64 GB,
macOS 27.0.1 (26A434), AC power.

**What is timed.** Matrices are uploaded once; each size runs one warm-up
call, then the mean of 3–10 calls of `ab_dgemm` / `ab_zgemm` on the
resident matrices (no conversion or transfer), against the same number of
Accelerate `cblas_dgemm` / `cblas_zgemm` calls on the CPU. ZGEMM rates
count 8N³ operations for both; the GPU's Gauss three-multiplication form
performs 25% fewer. For calls with host arrays through the BLAS ABI see
`../2026-10-14-52f1880-calibrate/`.

| File | |
|---|---|
| `run1.txt`–`run3.txt` | the full output of each run (all tables, CSV lines interleaved) |
| `square_gemm_median.csv` | Table 1 (square DGEMM/ZGEMM): median over the three runs of each rate and of the GPU/Accelerate ratio, with its range |
