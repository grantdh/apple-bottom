# A ZGEMM's FP64 epilogue at the edge of FP64's range: 2026-11-09

`benchmarks/probe_zgemm_epilogue.c` on 1.5.1, on 56216ee (`release/v1.6.0` before the fix) and on
f0c325b (`fix/zgemm-epilogue-range`, whose compiled code later commits on that branch keep, but for
the fallback message's text). The calls behind CHANGELOG 1.6.0's "A ZGEMM through the BLAS ABI whose
FP64 epilogue could overflow runs on Accelerate".

| | |
|---|---|
| Machine | MacBook Pro (Mac14,6), Apple M2 Max, 38-core GPU, 64 GB, macOS 27.0.1 |
| Libraries | Each `make clean && make -j8 lib` in a fresh clone with 0 modified tracked files, Apple clang 17.0.0, frozen under `~/qe-test/ab-frozen/`: 1.5.1 (93b1f2d) as `ab6faf15d432`, 56216ee as `366876ac76be`, f0c325b as `1064751262e6` (the first 12 hex digits of each dylib's SHA-256) |
| Runs | under machlock's exclusive bench lock; `overflow` and `bits` with `AB_MODE=gpu AB_MIN_GPU_DIM=0` (`*_staged.txt`) and with `AB_HOST_PACK=1` as well (`*_host.txt`); `time` with `AB_MODE=gpu`, on AC (140 W adapter) with load1 under 5 |

## The overflow probe (`overflow_*.txt`)

32 × 32 × 32 calls through `ab_zgemm_blas` and `ab_dgemm_blas`, each followed by Accelerate on the
same inputs, then the reproduction through `ab_dev_zgemm`. Each line gives the path, whether the
staged path kept the call, how far the GEMM counters moved, how many parts of C are not finite,
whether C is Accelerate's bit for bit, and C[0] from both.

- **The reproduction** (A at 2⁶⁰⁰, B at 2⁵⁰⁰, alpha 2⁻²⁰⁰, beta 0; C = 2⁹⁰⁵): NaN + NaN i in all
  1,024 entries with path `gpu` on 1.5.1 and 56216ee, staged and on the host path. Accelerate returns
  Inf + NaN i (its own product overflows). f0c325b returns Accelerate's bits with path `cpu`, and the
  GEMM counters do not move.
- **Inf alpha, Inf beta** on ordinary data: NaN + NaN i on 1.5.1 and 56216ee where Accelerate gives
  Inf + NaN i; Accelerate's bits on f0c325b.
- **The rest of the ZGEMM rows** (alpha 1 with the same product, beta 1 or 2 with C at 2¹⁰²³, alpha
  2⁸ with a product near 2¹⁰²⁰): the same bits as Accelerate on every library, with path `gpu` on
  1.5.1 and 56216ee and `cpu` on f0c325b. The alpha = beta = 1 row with C at 2¹⁰²³ and a small product
  is finite: the bound refuses it although the GPU's result was right.
- **Calls inside the bound** (alpha 1 + i with a product near 2¹⁰¹⁶, an ordinary call) stay on the
  GPU with Accelerate's bits on all three.
- **DGEMM** applies alpha and beta in DD before its one scale-back, the same on all three: the
  reproduction as a DGEMM returns the finite 2⁹⁰⁵ on the GPU (Accelerate: Inf), and a DGEMM whose
  result itself overflows returns ±Inf, Accelerate's bits in these rows.
- **`ab_dev_zgemm`**, which has no Accelerate fallback and is unchanged: NaN + NaN i (alpha 2⁻²⁰⁰) and
  Inf + 0i (alpha 1) with `AB_OK` on all three.

## Bits of calls the bound passes (`bits_*.txt`)

100 ordinary ZGEMMs (5 shapes, 4 TRANS pairs, 5 scalar sets covering the GPU epilogue's alpha = 1,
beta 0 or 1 and the host epilogue's other alpha and beta), one FNV-1a hash of C per call. 56216ee
and f0c325b give the same 100 lines, staged and on the host path.

## Cost (`timing.tsv`, `timing_cn.tsv`)

`ab_zgemm_blas` on uniform data (`time` mode): 3 warm-up and 15 timed calls per process, the median
per process. Each round runs 56216ee and f0c325b once on every shape, alternating which goes first.
Order-balanced ratio = sqrt(median of fix/base when 56216ee ran first × median when f0c325b ran first).
On AC (140 W adapter), load1 4.46 at the start and 3.06 at the end.

| Call | 56216ee | f0c325b | Ratio, order-balanced | Rounds |
|---|---|---|---|---|
| 'N','N' 9139 × 500 × 512, alpha 0.75 − 0.5i, beta 0 | 39.22 ms | 39.36 ms | 1.0022 | 8 |
| the same, beta 0.5 + 0.25i | 44.79 ms | 44.50 ms | 0.9982 | 8 |
| 'N','N' 2048³, alpha 0.75 − 0.5i, beta 0 | 112.18 ms | 111.84 ms | 0.9986 | 8 |
| 'C','N' 512 × 512 × 9139, alpha 1, beta 0 | 29.78 ms | 29.57 ms | 0.9935 | 8 |
| the same, again (`timing_cn.tsv`) | 32.20 ms | 32.06 ms | 0.9940 | 12 |

C's hash is the same for both libraries in every line. Two runs earlier the same evening (22:15 and
22:19), from a scratch copy of this mode with another generator and hash, gave 1.0009, 0.9998 and
1.0013 on the first three calls (8 rounds), and on the 'C','N' call 1.0078 over 8 rounds, with a
run-order effect (1.026 when 56216ee ran first, 0.990 when f0c325b did), then 1.0012 over 12 rounds.
