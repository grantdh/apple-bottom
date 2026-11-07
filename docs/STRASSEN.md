# Strassen-Winograd GEMM (opt-in)

apple-bottom's double-float GEMM kernel already runs within about 10% of the
M2 Max GPU's instruction-issue ceiling, so a large product only gets faster by
doing less arithmetic. One level of Winograd's variant of Strassen's algorithm
multiplies two matrices with 7 products of half-size blocks instead of 8,
plus 15 additions of blocks. That removes an eighth of the multiplications,
at the price of a weaker error bound: it is normwise, not componentwise.

Measured on an M2 Max at 8b48826 (November 6, 2026), it makes resident DGEMMs
1.07× faster at N = 4096 and 1.13× at 8192, and ZGEMMs 1.06–1.13× from 4096 up.
At 2048 it loses, 0.93× and 0.89× ([Speed](#speed)). At 274e778, on 1.4.0's
kernels, it used about 10% less energy per product; that was not repeated. On
data of uniform scale and mixed sign the error is 1.9× the classical DD GEMM's.
Where rows or columns differ widely in scale, the small elements of C have no
correct digits.

It is off by default. With it off, every result is bitwise what it was.

## Using it

| | |
|---|---|
| `ab_dgemm_strassen(alpha, A, B, beta, C)` | what `ab_dgemm_scaled` computes, C = alpha A B + beta C |
| `ab_zgemm_strassen(Ar, Ai, Br, Bi, Cr, Ci)` | what `ab_zgemm` computes: Gauss's three-multiplication form, or four real products where the parts are imbalanced, chosen as `ab_zgemm` chooses; each real product by Strassen-Winograd |
| `AB_STRASSEN=1` | `ab_dgemm`, `ab_dgemm_scaled`, `ab_zgemm` and `ab_zgemm_ex` take Strassen-Winograd products when M, N and K are all at least `AB_STRASSEN_MIN`, as do the BLAS-ABI GEMMs (`dgemm_`, `zgemm_`) that the routing sends to the GPU, on the staged path and on the host path, and the device-API GEMMs (`ab_dev_dgemm`, `ab_dev_zgemm`). |
| `AB_STRASSEN_MIN=<n>` | that threshold, default 4096 ([why](#threshold)) |

Both variables are read once per process. Never routed: the batched,
asynchronous and `ab_batch_*` GEMMs, and the library's own uses of its GEMMs,
whose accuracy is stated for classical products: DSYRK, ZHERK, DTRSM's
updates and the residuals of `ab_dgesv_mpir`. The session GEMMs
(`ab_session_dgemm`, `ab_session_zgemm`) call `ab_dgemm` and `ab_zgemm` and
are routed with them. The two entry points work for any shape: an odd
dimension is padded with zeros as the blocks are read and cropped as C is
written.

**What a call keeps.** Strassen-Winograd takes the place of each real
product inside the GEMM that would have made it, so everything the GEMM
does around the product stays as it is:
- the power-of-two scale of each matrix (data outside FP32's exponent
  range, 1.5.0). The blocks of A share A's scale and those of B share B's,
  so every sum adds values on one scale and every product carries A's plus
  B's, as the classical product does. Scaled data gives the unscaled data's
  result times the scales, bit for bit;
- the choice between Gauss-3M and 4M for ZGEMM (1.5.0): per matrix in the
  native API, per row and column in the BLAS ABI and the device API;
- the BLAS ABI's staged path, with its conversions, line scales and
  epilogue on the GPU, and its guard and fallbacks.

**Lines with scales of their own.** The BLAS ABI scales each row of op(A)
and column of op(B) whose largest entry lies outside 2^±20 by its own power
of two (1.5.0). The classical product never adds one line to another, so
that is exact for it. Strassen-Winograd's block sums do: S2 = A21 + A22 − A11
adds row i of A to row M/2 + i. Rows on scales 2^30 apart would put the
larger one's rounding error into the other's elements, about 2^30 times the
normwise bound. So a BLAS-ABI call with any line scaled takes classical
products, on the staged path (which learns the scales on the GPU and waits
for them before it encodes the products) and on the host path alike.

**Memory.** Each real product needs a workspace of two ⌈M/2⌉ × ⌈K/2⌉
blocks, two ⌈K/2⌉ × ⌈N/2⌉ and three ⌈M/2⌉ × ⌈N/2⌉: 1.75 N² double-float
values, 14 bytes per element of a square C, so 940 MB at N = 8192. A ZGEMM's
three or four products run one after another and share one workspace, next
to the Gauss-3M or 4M temporaries the classical ZGEMM also takes (4 N² for a
square N): 5.75 N² in all, 3.1 GB at 8192. The workspace comes from the same
pool as those temporaries (`AB_STAGING_MB`, default 2048 MB), which keeps
buffers between calls because a new Metal buffer is slow the first time the
GPU uses it. The pool grows a buffer by an eighth more than a call asks for,
and what it holds beyond the cap is freed when a call returns, so a ZGEMM at
8192 gets new buffers on every call unless `AB_STAGING_MB` is about 3500 or
more.

**Threads.** Each call takes its own buffers from the pool, so Strassen
calls on different threads run side by side, as the classical ones do.

## How it works

With the blocks of A (M × K), B (K × N) and C numbered as quadrants,

```
S1 = A21 + A22   T1 = B12 − B11   P1 = A11 B11   P5 = S1 T1   U2 = P1 + P6
S2 = S1 − A11    T2 = B22 − T1    P2 = A12 B21   P6 = S2 T2   U3 = U2 + P7
S3 = A11 − A21   T3 = B22 − B12   P3 = S4 B22    P7 = S3 T3   U4 = U2 + P5
S4 = A12 − S2    T4 = T2 − B21    P4 = A22 T4
C11 = P1 + P2    C12 = U4 + P3    C21 = U3 − P4  C22 = U3 + P5
```

(Boyer, Dumas, Pernet and Zhou, ISSAC 2009, Algorithm 1). `src/strassen.m`
encodes one product in one serial compute encoder, inside the command buffer
of the GEMM it serves (`src/strassen.h`), so each step sees the writes of the
ones before it, and the GEMM's steps after it see C whole:

- **The 7 products** run on the library's own GEMM kernels, dispatched exactly
  as the classical GEMMs dispatch their products, so any change to the kernels
  or their dispatch, `AB_PRECISION=fast` included, applies to them too. They
  need contiguous operands: the sums S and T are written that way, and P1–P4's
  quadrant operands are copied (6 copies).
- **Everything else** is one kernel, `dd_block_combine`: Z = X, X + Y or
  X − Y on blocks of row-major matrices with offsets and leading dimensions of
  their own, the block reading as zero beyond its matrix. It forms the S and T
  sums straight from A's and B's quadrants, and the last four additions write
  C's quadrants through the GEMM epilogue (alpha U + beta C, as `dd_dgemm_ab`
  applies it). Those are the 15 additions; with the copies, 21 passes over
  blocks.
- **Order.** Products are formed in the order P1, P2, P5, P6, P7, P3, P4 and
  folded into the U sums as soon as possible, so that three product blocks are
  live at a time, not seven.

The DD arithmetic is the library's: the MSL in `strassen.m` repeats
`apple_bottom.m`'s error-free transformations, addition and multiplication,
compiled without fast math as the library's shaders are.

## Speed

**Setup.**
- Machine and power: M2 Max (38-core GPU), macOS 27.0.1, AC power from a 140 W
  adapter.
- Build: `bench_strassen speed` at 8b48826.
- Method: matrices resident on the GPU (native API, no conversion or
  transfer timed); median of 5 calls (3 for ZGEMM 8192) after a 0.5 s
  warm-up of each method. Classical and Strassen calls alternate back to
  back; Accelerate (`cblas_*gemm`, FP64, all CPU cores) runs on its own.
- Passes: two for the squares. The tables give the second, with the first's
  speedup in brackets. The first pass of DGEMM 6144 and 8192 is left out:
  something else on the machine slowed it, and Accelerate's 8192 median read
  2742 ms against 1772 ms. The rectangular shapes ran one pass.
- [Data](../benchmarks/results/2026-11-06-8b48826-pretag/strassen/). The runs at
  274e778, on 1.4.0's kernels and BLAS ABI, are in
  [its folder](../benchmarks/results/2026-10-20-274e778-strassen/).

DGEMM, square, wall time per call:

| N | classical DD | Strassen | Strassen speedup | Accelerate | Strassen vs Accelerate |
|---:|---:|---:|---:|---:|---:|
| 2048 | 25.1 ms | 26.9 ms | 0.93× (0.93×) | 28.5 ms | 1.06× |
| 2560 | 48.1 ms | 45.6 ms | 1.05× (1.05×) | 52.8 ms | 1.16× |
| 3072 | 83.0 ms | 78.3 ms | 1.06× (1.06×) | 90.5 ms | 1.16× |
| 3584 | 128.5 ms | 124.3 ms | 1.03× (1.04×) | 142.7 ms | 1.15× |
| 4096 | 191.8 ms | 178.9 ms | **1.07×** (1.07×) | 224.6 ms | 1.26× |
| 6144 | 643.1 ms | 590.6 ms | **1.09×** | 726.7 ms | 1.23× |
| 8192 | 1695 ms | 1505 ms | **1.13×** | 1772 ms | 1.18× |

At 8192 the classical DD GEMM runs at 649 GFLOP/s. Strassen's effective rate
there (2N³ per call) is 731 GFLOP/s. The ceiling is 8/7 = 1.143×.

DGEMM, rectangular and odd, M × K × N:

| shape | classical DD | Strassen | Strassen speedup | Accelerate |
|---|---:|---:|---:|---:|
| 8192 × 4096 × 8192 | 757.8 ms | 685.5 ms | 1.11× | 875.0 ms |
| 4096 × 8192 × 4096 | 382.9 ms | 358.4 ms | 1.07× | 457.1 ms |
| 8192 × 2048 × 8192 | 378.5 ms | 345.9 ms | 1.09× | 432.9 ms |
| 16384 × 4096 × 4096 (tall-skinny kernel) | 896.3 ms | 802.6 ms | 1.12× | 845.5 ms |
| 4096 × 4096 × 16384 | 798.2 ms | 702.3 ms | 1.14× | 904.2 ms |
| 4097³ | 198.6 ms | 194.1 ms | 1.02× | 245.0 ms |
| 2048 × 32768 × 2048 | 412.4 ms | 423.0 ms | **0.97×** | 466.1 ms |

The classical median at 4096 × 4096 × 16384 sits above its fastest call
(758.2 ms). Fastest against fastest (Strassen 688.1 ms) gives 1.10×.

ZGEMM, square, wall time per call and the speedup in command-buffer GPU time
alone:

| N | classical DD | Strassen | speedup, wall | speedup, GPU time | Accelerate | Strassen vs Accelerate |
|---:|---:|---:|---:|---:|---:|---:|
| 2048 | 73.1 ms | 81.7 ms | 0.89× (0.89×) | 0.90× | 111.6 ms | 1.37× |
| 3072 | 244.9 ms | 236.8 ms | 1.03× (1.03×) | 1.03× | 368.7 ms | 1.56× |
| 4096 | 612.2 ms | 542.2 ms | 1.13× (1.06×) | 1.13× | 881.1 ms | 1.63× |
| 6144 | 2116.3 ms | 1891.6 ms | 1.12× (1.10×) | 1.13× | 2919.7 ms | 1.54× |
| 8192 | 4616 ms | 4166 ms | 1.11× (1.12×) | 1.11× | 6933 ms | 1.66× |

- **4096.** The second pass's classical calls ran in two states (median
  612.2 ms, fastest 574.5 ms). The first pass's 1.06× is the steadier figure.
- **Wall against GPU time.** At 274e778 the classical ZGEMM allocated its
  temporaries afresh on every call and its wall-time speedup ran ahead of the
  GPU-time one (1.15× against 1.11× at 6144). Since 1.5.0 it keeps them, and
  the two columns agree.
- **8192.** Neither method's buffers fit the default pool, so a call shows
  56–84 ms between commit and the GPU starting. With `AB_STAGING_MB=4000` that
  falls to 0.2 ms for the classical product and 11 ms for Strassen. The speedup
  is then 1.12×, and the two passes differ by more than the time saved.

**Through the BLAS ABI** (`ab_*gemm_blas`, column-major FP64 in and out,
`AB_MODE=gpu`, the staged path; median of 5 calls for DGEMM and 3 for ZGEMM):
the speedup is smaller than the native one in the tables above, except at DGEMM
4096. Outside DGEMM 4096 the time saved per call shrinks too (DGEMM 8192:
138–149 ms against 190 ms; ZGEMM 6144: 88 ms against 173–225 ms), so a fixed
conversion cost does not explain the difference. A Strassen call on this path also
runs one more command buffer and takes its workspace from a second pool slot. These
runs record no stage times, so the difference is not apportioned. DGEMM ran twice.

| call | GPU | GPU with `AB_STRASSEN=1` | speedup | Accelerate, above |
|---|---:|---:|---:|---:|
| DGEMM 4096³ | 200.4, 201.7 ms | 188.7, 188.5 ms | 1.06×, 1.07× | 224.6 ms |
| DGEMM 6144³ | 670.0, 697.2 ms | 638.9, 656.1 ms | 1.05×, 1.06× | 726.7 ms |
| DGEMM 8192³ | 1660, 1756 ms | 1522, 1606 ms | 1.09×, 1.09× | 1772 ms |
| ZGEMM 4096³ | 601.3 ms | 579.0 ms | 1.04× | 881.1 ms |
| ZGEMM 6144³ | 2108.6 ms | 2020.3 ms | 1.04× | 2919.7 ms |

These 'N','N' calls are faster on the GPU than on Accelerate even without
Strassen. `AB_MODE=auto` still sends them to Accelerate, because its table was
measured on 1.4.0's host path and the staged path has not been calibrated.

### Where the time goes

Measured at 274e778 and not repeated. The classical product took 199.7 ms of
GPU time at 4096 and Strassen 185.0 ms; 7/8 of the classical time would be
174.7 ms. The 10 ms difference came in two halves of about 5 ms each:
- **Half-size products** ran a little less efficiently. Seven 2048³ products
  took 7 × 25.7 = 179.6 ms, at 670 GFLOP/s against 690 for 4096³.
- **Block passes.** On their own (measured with the products left out) they
  took 6.0 ms. They move about 1.9 GB at ~320 GB/s, close to the memory
  bandwidth, so a faster addition kernel would not help.

Things that were tried or ruled out:
- **Concurrent encoder.** Running the additions beside the GEMMs in a
  concurrent compute encoder (5 product blocks instead of 3) changed nothing:
  186.3 ms against 186.5 ms of GPU time at 4096, in either encoding order. A
  GEMM dispatch fills the GPU, and the additions ran after it rather than
  beside it.
- **Removing the 6 copies** would need a GEMM kernel that reads blocks with a
  leading dimension.
- **Removing most of the additions** would need kernels that fuse them into
  the GEMM's loads and stores (Huang, Smith and van de Geijn, "Strassen's
  algorithm reloaded", SC16).

Both mean new kernels, and this path deliberately uses the existing ones.

Odd dimensions cost a little. 4097³ splits into 2049³ products, each with a
33rd mostly empty tile row and column, so it gains only 1.02×.

A small M and N with a long K loses: 2048 × 32768 × 2048 has 1024 × 1024 outputs
per product, which take the 32 × 32 tiles. A product then has 1024 threadgroups,
as the classical GEMM has. At the same K a 32 × 32 threadgroup costs about 0.29× a
64 × 64 one (kernel times at 2825d2a), so by those times seven products at half K
cost about as much as the classical product, and the block passes come on top.

Strassen's first call for a shape grows the workspace (now the pool's
buffers): 25–28 ms more than a later call for DGEMM 4096 and 42–46 ms for ZGEMM
4096. The first call in a process
also compiles its kernel (about 20 ms at 274e778). From about 2.5 GB of buffers
per command buffer (DGEMM 8192, ZGEMM ≥ 6144), each Strassen call shows 11–59 ms
between commit and the GPU starting, about 1% of the call. Shared and private
storage for the workspace measured the same at 274e778.

### Threshold

`AB_STRASSEN_MIN` applies to M, N and K together. The default, 4096, is where
every measured shape gained:
- square DGEMM 1.07–1.13×, square ZGEMM 1.06–1.13×;
- rectangular 1.07–1.14×;
- the odd 4097³ 1.02×.

Square DGEMMs from 2560 to 3584 gain 1.03–1.06× and ZGEMM 3072 gains 1.03×;
2048 loses (0.93×, ZGEMM 0.89×). A small M and N loses even with a long K
(0.97×). To use Strassen below 4096, set
`AB_STRASSEN_MIN` after timing your own shapes. For example, 2560 suits
square-ish products of that size.

## Energy

Measured at 274e778, on 1.4.0's kernels, and not repeated. Energy per product,
from a no-sudo probe (IOReport "GPU Energy" and SMC power, the library called
in a loop for 30 s, in 10 s windows;
[output](../benchmarks/results/2026-10-20-274e778-strassen/)).
The charger was recharging the battery during the run, so the DC input
(PDTR) is not usable. The system figures are the SMC system total (PSTR)
above idle (34 W). J per GFLOP counts 2N³ per call for all three methods:

| | 4096: system | 4096: GPU | 8192: system | 8192: GPU |
|---|---:|---:|---:|---:|
| classical DD | 0.124 J/GF | 0.091 J/GF | 0.124 J/GF | 0.085 J/GF |
| Strassen | 0.111 J/GF | 0.081 J/GF | 0.110 J/GF | 0.080 J/GF |
| Accelerate (AMX) | 0.053 J/GF | – | 0.051 J/GF | – |

Strassen uses about 10% less energy per product than the classical DD GEMM;
Accelerate still uses less than half.

## Accuracy

### The bounds

Write u = 8u₃₂² = 2⁻⁴⁵ (u₃₂ = 2⁻²⁴), a round figure for the DD arithmetic. The
[precision envelope](vv/PRECISION_ENVELOPE.md) proves `dd_mul` ≤ 6.000001u₃₂²,
`dd_fma` ≤ 13u₃₂²|ab| + 3u₃₂²|c| to first order and GEMM ≤ (3K + 13)u₃₂²|A||B|,
which K u covers for K ≥ 3. It states no bound for `dd_add`; the
Strassen-Winograd row below takes its error to be at most u. For C = AB with
inner dimension K, ‖X‖ the largest |x_ij| and |A||B| the product of the
elementwise absolute values:

| | bound | kind |
|---|---|---|
| classical DD GEMM | \|Ĉ − C\| ≤ γ_K \|A\|\|B\|, γ_K ≈ K u | componentwise: each element against its own terms |
| | ‖Ĉ − C‖ ≤ K² u ‖A‖ ‖B‖ | the same, in the max norm |
| one level of Strassen-Winograd | ‖Ĉ − C‖ ≤ (4.5 K² + 48 K) u ‖A‖ ‖B‖ | normwise only |

The Strassen-Winograd bound is Theorem 23.3 of Higham, *Accuracy and
Stability of Numerical Algorithms* (2nd ed., 2002), for one level of
recursion (n₀ = K/2); in the max norm only the inner dimension enters.
Strassen's original variant has 3K² + 25K (Theorem 23.2): Winograd's saves
three of its 18 block additions and pays with the larger constant. In the
worst case one level costs 4.5× the classical bound, about 2.2 bits.

The kind of bound matters more than its constant. An algorithm that saves
multiplications this way cannot satisfy a componentwise bound (Miller, 1975;
Higham §23.2.2). The sums S and T mix rows and columns of different blocks,
and the products of those sums are later cancelled against each other. So an
element of C whose own terms are small receives the rounding errors of large
ones. When rows of A or columns of B differ widely in scale, the small ones
lose relative accuracy. Data of uniform scale does not notice.

Both bounds are worst cases, and actual errors are far smaller. The
Strassen-Winograd one is loose. For entries uniform in [−1, 1), rounding the
exact result to FP32 leaves a largest error of 3.8 × 10⁻⁶ ‖A‖ ‖B‖ at 4096³,
where the bound allows 2.2 × 10⁻⁶, and 7.6 × 10⁻⁶ at 8192³, where it allows
8.6 × 10⁻⁶. So from about K = 8192 the bound does not rule out a result with no
DD low words. That is why `tests/test_strassen.c` also holds Strassen to a
Frobenius error of 2⁻⁴⁰ relative to ‖ |A||B| ‖_F: losing the DD low words would
exceed it.

### Measured

Errors against a Dot2 reference of all of C (compensated FP64, as accurate
as twice FP64 precision), `bench_strassen accuracy` at 8b48826, for DGEMM
([data](../benchmarks/results/2026-11-06-8b48826-pretag/strassen/accuracy_175832.csv)).
They equal the runs at 274e778 to every digit shown here, as they should: the port leaves the algorithm as it was, and 1.5.0's
kernel changes keep every product's bits. ZGEMM's are checked by the tests
below. Under `AB_PRECISION=fast` the products use its accumulation, a
combination that has not been measured. The three kinds of data are:

- **uniform**: entries uniform in [−1, 1);
- **positive**: entries in [0, 1), so there is no cancellation and |C| = |A||B|;
- **scaled**: uniform data with the rows of A and the columns of B multiplied
  by 2^e, e an integer uniform in [−16, 16].

Normwise error, ‖Ĉ − C‖_F / ‖C‖_F:

| M × K × N | data | classical DD | Strassen | Strassen ÷ classical | Accelerate FP64 |
|---|---|---:|---:|---:|---:|
| 4096³ | uniform | 5.09e−14 | 9.52e−14 | 1.87 | 2.14e−15 |
| 6144³ | uniform | 6.22e−14 | 1.17e−13 | 1.87 | 2.61e−15 |
| 8192³ | uniform | 7.19e−14 | 1.35e−13 | 1.87 | 3.02e−15 |
| 4096³ | positive | 3.75e−14 | 1.90e−14 | 0.51 | 1.57e−15 |
| 6144³ | positive | 5.23e−14 | 2.62e−14 | 0.50 | 2.19e−15 |
| 8192³ | positive | 5.29e−14 | 2.67e−14 | 0.50 | 2.22e−15 |
| 4096³ | scaled | 5.11e−14 | 9.58e−14 | 1.88 | 2.14e−15 |
| 8192³ | scaled | 7.19e−14 | 1.35e−13 | 1.87 | 3.01e−15 |
| 4097³ | uniform | 5.09e−14 | 9.51e−14 | 1.87 | 1.07e−15 |
| 8192 × 4096 × 8192 | uniform | 5.09e−14 | 9.52e−14 | 1.87 | 2.14e−15 |
| 4096 × 8192 × 4096 | uniform | 7.19e−14 | 1.35e−13 | 1.87 | 3.02e−15 |

Componentwise error, max_ij |Ĉ − C|_ij / (|A||B|)_ij:

| M × K × N | data | classical DD | Strassen | Accelerate FP64 |
|---|---|---:|---:|---:|
| 4096³ | uniform | 1.46e−14 | 2.76e−14 | 6.8e−16 |
| 8192³ | uniform | 1.64e−14 | 2.72e−14 | 6.6e−16 |
| 4096³ | positive | 1.96e−13 | 1.08e−13 | 8.8e−15 |
| 8192³ | positive | 3.30e−13 | 1.55e−13 | 1.4e−14 |
| 4096³ | scaled | 1.44e−14 | **3.7e+04** | 7.2e−16 |
| 6144³ | scaled | 1.55e−14 | **2.2e+04** | 6.6e−16 |
| 8192³ | scaled | 1.47e−14 | **5.0e+04** | 6.7e−16 |
| 8192 × 4096 × 8192 | scaled | 1.58e−14 | **6.9e+04** | 7.8e−16 |

What this shows:

- **Uniform scale, mixed signs.** Strassen's normwise error is 1.87× the
  classical one at every size and shape, a little under one bit; its
  componentwise error is 1.5–2.0×. Both normwise errors grow like √K. Against
  Accelerate's normwise FP64 error the classical one is 24× larger and
  Strassen's 45×, except at 4097³, where Accelerate's error is half its 4096³
  value and the factors are 47× and 89×.
- **Positive data.** Strassen's error is half the classical one. With no
  cancellation, the error is dominated by the long accumulation in each
  product, and Strassen's products are half as long. (The March 2026 probes
  that reported Strassen "gaining bits" used data in [0, 1).) This is a
  property of the data, not of the algorithm.
- **Scaled rows and columns.** The normwise error is unchanged (1.87–1.88×), but
  componentwise Strassen fails. The error in the small elements reaches 2–7 × 10⁴
  times the size of their own terms, depending on the shape, so they have no
  correct digits; their error relative to |C_ij| reaches 6 × 10⁹. On the same
  data the classical GEMM's error stays within 1.6 × 10⁻¹⁴ of its own terms, as
  its componentwise bound promises, and reaches 7.5 × 10⁻⁶ relative to |C_ij|.
- The normwise errors are 2 × 10⁴ to 4 × 10⁵ below the worst-case bound across
  the six shapes, with the smallest margin on positive data. At 4096³, max|Ĉ − C|
  is 2.7 × 10⁻¹¹ ‖A‖ ‖B‖ on uniform data and 1.1 × 10⁻¹⁰ on positive data,
  against 2.2 × 10⁻⁶ allowed.

Use it for well-scaled problems where a normwise error of ~10⁻¹³ is enough.
Don't use it where small elements of C must be accurate relative to their own
terms. Such cases include products of badly scaled matrices, matrices with
rows or columns of very different norms, and results that are later divided by
small elements.

### What the tests check

`make test` runs `tests/test_strassen.c` against a compensated FP64 (Dot2)
reference:

- Shapes: even, odd, tiny (1 × 1 × 1 to 5 × 7 × 3), short and long K, and a
  tall product whose half-size GEMMs take the tall-skinny kernel.
- Error bounds: Strassen within the normwise bound and the Frobenius limit;
  the classical GEMM within its componentwise bound (K + 2) u.
- Epilogue: alpha and beta, and beta = 0 with NaN in C.
- ZGEMM, against the same bounds.
- ZGEMM with imaginary parts 2⁻²⁴ of the real ones: it runs as 4M, and each
  part stays within twice the bound for its own size (Gauss-3M would put Ci's
  error at about 10⁻⁵ of that size, 700 times the bound).
- Scaled data: A near 2⁸⁴, B near 2⁻²⁰ and C near 2⁶⁴ (DGEMM with alpha and
  beta), and ZGEMMs on the same scales, by Gauss-3M (Ai 2⁻²⁰ of Ar, whose sum
  needs them aligned) and by 4M (both imaginary parts 2⁻²⁰ of the real ones,
  whose products are rescaled before they are combined), give the unscaled
  results times the scales, bit for bit, and the unscaled ZGEMMs are within
  the bound (the comparison alone would pass if both runs went wrong alike).
- Argument checks.
- Bitwise determinism after the workspace served other shapes and after
  `ab_shutdown()`.
- Routing, in child processes with their own environment: `AB_STRASSEN`
  unset or 0 leaves every GEMM bitwise classical even with
  `AB_STRASSEN_MIN=1`. `AB_STRASSEN=1` sends a call with M, N, K ≥ the
  threshold to Strassen, bitwise, through the native API, through `dgemm_`
  and `zgemm_` (balanced and 4M) on the staged path and on the host path
  (`AB_HOST_PACK=1`), and through `ab_dev_dgemm` and `ab_dev_zgemm`, and
  leaves smaller ones classical.
- Lines with scales of their own: `dgemm_` and `zgemm_`, on the staged path
  and on the host path, whose rows of op(A) from M/2 on are 2⁻²¹ of the
  others take classical products, every element within the classical
  componentwise bound of its own |A||B|.
- The library's own uses of its GEMMs (ZHERK, a one-element
  `ab_zgemm_batched`, DTRSM, `ab_dgesv_mpir`) give the same bits with
  `AB_STRASSEN=1 AB_STRASSEN_MIN=1` as without.

Each of these planted defects fails a test: the scales ignored and 3M
forced, as before the port (the scaled-data, 4M and staged-4M checks); the
line-scale rule off on the staged path or on the host path (the lines checks
of that path); the library's own uses routed (`test_routing_internal`);
Gauss-3M's alignment skipped (the unscaled ZGEMM of `test_scaled_data`); 4M's
rescaling skipped (`test_zgemm_imbalanced`, `test_scaled_data`'s 4M and the
staged-4M check).

`test_dgemm_scaled_rows` prints the case the componentwise bound is about. It
uses a 512³ product whose rows of A are scaled by 2^e, e ∈ [−20, 20]. The
classical GEMM's componentwise error is 1.1 × 10⁻¹⁴, Strassen's 3.8 × 10⁻³.
Both stay within their normwise bounds.

## Reproducing

```bash
make build/bench_strassen
# speed, native API, matrices resident on the GPU (one CSV row per method)
./build/bench_strassen speed 4096x4096x4096 8192x8192x8192
./build/bench_strassen speed --complex 4096x4096x4096
# Dot2 references (CPU only, minutes at 8192), then errors against them
./build/bench_strassen reference --ref-dir /tmp/refs 4096x4096x4096
./build/bench_strassen accuracy --ref-dir /tmp/refs 4096x4096x4096
# end to end through the BLAS ABI, as the environment routes it
AB_MODE=gpu AB_STRASSEN=1 ./build/bench_strassen abi 8192x8192x8192
```

Shapes are M×K×N. `speed` warms each method up for 0.5 s; the GPU raises its
clock only after about 0.3 s of load. It then times the classical and
Strassen GPU calls back to back in alternating order, then Accelerate
(`cblas_dgemm`/`cblas_zgemm`, FP64, all cores) on its own, so no GPU call
follows an idle GPU.
