# ZHEMV in one read of the triangle, M2 Max, 2026-10-19

`build/bench_gemv --routines zhemv` (`benchmarks/bench_gemv.c`) under
`machlock.py bench` (exclusive use of the machine), library built from the
working tree on 1bc503a that became the single-read ZHEMV commit
(`ab_build_info()`: `1.4.0 (1bc503a2e4d3-dirty)`). The `*_twopass` files
are the same benchmark linked against the library at 1bc503a, whose ZHEMV
made two passes (rows, then columns) and so read each off-diagonal element
twice, run right after the new one for a comparison under the same
conditions.

Machine: MacBook Pro (Mac14,6), Apple M2 Max, 38-core GPU, 8P+4E CPU, 64 GB,
macOS 27.0.1 (26A434), macOS SDK 26.0, Apple clang 17.0.0, AC power.

All files come from one locked run, 10:02–10:05, load 1.9–2.9:

| File | |
|---|---|
| `bench_gemv.txt`, `bench_gemv.csv`; `*_twopass.*` | `--reps 9`: 4096, 8192 and 16384, four ways (as in `2026-10-19-317bb91-gemv/README.md`) |
| `sweep_square_run1.*`, `sweep_square.*` | `--sweep`: staged GPU against Accelerate, square 256–16384, twice: the data behind the `AB_MODE=auto` ZHEMV row |
| `bench_gemv_small.*`; `*_twopass.*` | `--sizes 256,512,1024,1536,2048,3072 --reps 21` |
| `bench_gemv_tiny_alternating.csv` | `--sizes 128,256,384 --reps 51`, single-read and two-pass alternately, three rounds (`round`, `kernel` columns added) |

All calls compute y = A x (alpha 1, beta 0) with lda = N and unit
increments; GB/s counts one triangle, N (N + 1) / 2 × 16 bytes.

**Resident** (`ab_dev_zhemv` on a registered matrix), GB/s, 'U' / 'L':

| N | single read | two passes | speedup |
|---:|---:|---:|---:|
| 4096 | 162 / 156 | 113 / 113 | 1.44 / 1.38× |
| 8192 | 274 / 264 | 137 / 135 | 2.00 / 1.95× |
| 16384 | 316 / 305 | 155 / 162 | 2.04 / 1.88× |

From 1024 to 3072 the single read is 1.20–1.41× faster
(`bench_gemv_small*`). At 128–512 both take 0.16–0.25 ms per call, the
submit and wait, and either can be ahead by 10–30 µs from one round to the
next (`bench_gemv_tiny_alternating.csv`, and 512 in `bench_gemv_small*`):
no size gains from the two passes, so that kernel is gone. Error against
the compensated reference: 2.5–3.2×10⁻¹⁶ relative to |A||x|, against
4.0–9.7×10⁻¹⁶ for the two passes.

**Staged against Accelerate** (each call maps A): 2.35–4.51× from 2048 to
16384 in both runs, 1.07–1.14× at 1536, below 1 from 1024 down. The
`AB_MODE=auto` row (ZHEMV for 2048 ≤ N ≤ 16384) stands as it was ('U' at
1536 reached 1.10× in one run only); `tests/test_blas2.c` checks it
against both sweep files.

**What the kernel does.** A simdgroup takes a panel of 32 R rows (lane l
holds rows l, l + 32, ...) and a chunk of columns, reads each stored
element once, adds a x_j to its row's sum (kept in registers) and conj(a)
x_i to its column's, which is summed over the lane's R rows and then
across the lanes through threadgroup memory, four columns at a time. R = 1
below N = 4608 and 3 from there: at 4096 R = 1 was 5% faster (more work
items), at 5120 R = 3 6% faster (a third as many cross-lane additions per
element). Chunks hold about 4096 (panel, chunk) work items per call. What
shaped it, measured while tuning:
- The FP64-to-double-float conversion, not memory, bounded the first
  version (removing it doubled the speed). `f64_to_dd_fast` gives the same
  bits as `f64_to_dd` wherever the high word is finite, with two exact
  multiplies in place of `ldexp` and branches: on this GPU both flush FP32
  subnormal results, and arithmetic reads subnormal inputs as zero (134
  million inputs compared, every exponent, ties and specials). Elements
  beyond FP32's range, Inf and NaN become NaN, which sends the call to
  Accelerate as before, without the selects that kept Inf and NaN apart:
  3–11% faster at 8192 and 16384.
- Registers limit occupancy (R = 3: 512 threads per threadgroup, R = 4:
  384). Four rows per lane were no faster at 16384 and slower at 8192;
  eight columns per group of column sums (twice the threadgroup memory)
  were slower everywhere.

The partial results, per (panel, column) and per (chunk, row), take about
N²/8 bytes plus 9 MB from N = 4608 (45 MB at 16384); up to 64 MB stay
allocated per calling thread between calls, beyond that (N above about
21000) they are allocated for each call.
