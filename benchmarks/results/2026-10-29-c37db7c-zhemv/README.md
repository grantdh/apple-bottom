# ZHEMV in one read of the triangle, five runs, M2 Max, 2026-10-29

`build/bench_gemv --routines zhemv --reps 9 --warmup 2`
(`benchmarks/bench_gemv.c`) five times, each run a new process, back to back
in one acquisition of `machlock.py bench` (exclusive use of the machine),
10:07:50–10:09:22. The library was built from a clean checkout of c37db7c,
the tip of `perf/gemv-fastconv` (`ab_build_info()`: `1.4.0 (c37db7c23314)`).
This repeats `bench_gemv.*` of `../2026-10-19-1bc503a-zhemv/`, which was one
run: same program, sizes and options, and the same kernel. 8e7140b gave GEMV
the ZHEMV kernel's element conversion and renamed it, and found
`ab_dev_zhemv`'s results bit for bit unchanged; the errors below are that
run's to every printed digit.

Machine: MacBook Pro (Mac14,6), Apple M2 Max, 38-core GPU, 8P+4E CPU, 64 GB,
macOS 27.0.1 (26A434), macOS SDK 26.0, Apple clang 17.0.0. AC power
throughout: checked inside the lock before and after every run, and AC in
all ten samples taken every 10 s (battery at 67%, charging). `pmset -g
therm` recorded no thermal or performance warning at the start or the end.
1-minute load 4.9 at the start (the lock waits for it to fall below 5 after
the previous job) and 2.9–4.5 during the runs.

| File | |
|---|---|
| `bench_gemv_run1.*` … `bench_gemv_run5.*` | the five runs in order, 18–19 s each |
| `machine.txt` | machine, power and load at the start and the end, the library's SHA-256 |
| `batch.log` | each run's start and end with power and load; `pmset -g batt` and `pmset -g therm` at the start and the end |
| `power.log` | power source, battery and load every 10 s |

Each run times y = A x (alpha 1, beta 0, lda = N, unit increments) at
N = 4096, 8192 and 16384, `'U'` and `'L'`, four ways (as in
`../2026-10-19-317bb91-gemv/README.md`): resident (`ab_dev_zhemv` on a
matrix registered once), staged (`ab_zhemv_blas` with the GPU path forced,
so each call also maps A), Accelerate (`cblas_zhemv`) and plain C (the
fastest of 4, 8 and 12 threads). Each is the median of 9 calls after 2
warm-up calls (GPU modes: and at least 0.5 s of calls). GB/s counts one
triangle, N (N + 1) / 2 × 16 bytes.

Median of the five runs, their range in brackets. Ratios are Accelerate's
time over the GPU's, taken within each run:

| N | uplo | resident GB/s | resident ms | Accelerate ms | Accelerate GB/s | resident / Accelerate | staged GB/s | staged / Accelerate | plain C GB/s |
|---:|:-:|---:|---:|---:|---:|---:|---:|---:|---:|
| 4096 | U | 161 (124–164) | 0.836 (0.818–1.082) | 15.2 (15.1–15.3) | 8.84 (8.78–8.87) | 18.1 (14.1–18.5) | 34.5 (33.1–34.6) | 3.90 (3.74–3.92) | 71 (62–72) |
| 4096 | L | 159 (151–161) | 0.845 (0.833–0.888) | 14.7 (14.6–15.2) | 9.11 (8.81–9.17) | 17.3 (16.5–18.3) | 34.5 (34.3–37.4) | 3.77 (3.77–4.25) | 59 (53–76) |
| 8192 | U | 273 (272–280) | 1.969 (1.921–1.977) | 64.9 (64.3–65.6) | 8.27 (8.18–8.35) | 33.2 (32.6–33.7) | 38.7 (38.5–40.9) | 4.70 (4.61–4.94) | 86 (86–89) |
| 8192 | L | 261 (258–263) | 2.061 (2.045–2.080) | 64.5 (64.4–65.4) | 8.33 (8.21–8.33) | 31.5 (31.0–32.0) | 38.3 (37.3–38.7) | 4.61 (4.47–4.66) | 92 (91–92) |
| 16384 | U | 312 (312–316) | 6.875 (6.801–6.889) | 246.7 (245.8–247.4) | 8.71 (8.68–8.74) | 35.8 (35.7–36.3) | 37.8 (37.1–38.1) | 4.33 (4.26–4.39) | 100 (97–106) |
| 16384 | L | 305 (303–305) | 7.042 (7.039–7.087) | 244.9 (243.9–245.3) | 8.77 (8.75–8.81) | 34.8 (34.4–34.8) | 37.6 (37.3–38.0) | 4.29 (4.23–4.34) | 102 (101–103) |

- **Resident:** the medians run from 159 to 312 GB/s, 17.3–35.8× Accelerate.
  The lowest single value is the batch's first GPU measurement: run 1,
  `'U'` at 4096, 124 GB/s (1.08 ms per call), where the other four runs
  gave 159–164; probably the GPU's clock still rising after `bench_gemv`'s
  0.5 s warm-up. Without it, every run lies in 151–316 GB/s and 16.5–36.3×
  Accelerate.
- **Accuracy:** each mode's `err` is the same in all five runs and as on
  2026-10-19: 2.5–3.2×10⁻¹⁶ relative to |A||x| for the GPU (resident and
  staged give the same bits), 4.6×10⁻¹⁷ to 1.3×10⁻¹⁶ for Accelerate.
- **Against the run of 2026-10-19**
  (`../2026-10-19-1bc503a-zhemv/bench_gemv.csv`): its resident rates, 162 /
  156, 274 / 264 and 316 / 305 GB/s (`'U'` / `'L'` at 4096, 8192, 16384),
  lie within these ranges but for 8192 `'L'` (264, now 258–263). Its
  ratios to Accelerate, 17.8–35.7×, lie within them but for 4096 `'U'`
  (19.1×, now 14.1–18.5×) and 16384 `'L'` (35.4×, now 34.4–34.8×):
  Accelerate's times differ by up to 4% between the two sessions, either
  way (4096 `'U'` 15.8 ms then, 15.1–15.3 now; 16384 `'U'` 242.7 then,
  245.8–247.4 now). Staged calls at 16384 take 57 ms now against 75 ms
  then, with the same GPU kernel, so the difference is in mapping A's
  4.3 GB for each call; staged against Accelerate there is 4.2–4.4×, then
  3.2–3.3×. Plain C moved the most (8192 `'L'` 76 GB/s then, 91–92 now;
  4096 `'L'` 53–76 within this batch).
