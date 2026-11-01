# Strassen-Winograd GEMM (opt-in)

apple-bottom's double-float GEMM kernel already runs within about 10% of the
M2 Max GPU's instruction-issue ceiling, so a large product only gets faster by
doing less arithmetic. One level of Winograd's variant of Strassen's algorithm
multiplies two matrices with 7 products of half-size blocks instead of 8,
plus 15 additions of blocks. That removes an eighth of the multiplications,
at the price of a weaker error bound: it is normwise, not componentwise.

Measured on an M2 Max at 274e778 (October 20, 2026, on 1.4.0's kernels), it
made resident DGEMMs 1.08× faster at N = 4096 and 1.12× at 8192, and ZGEMMs
1.10–1.16×, with about 10% less energy per product. Those runs predate its
port to 1.6.0 and have not been repeated on it ([Speed](#speed)). On data of
uniform scale and mixed sign the error is 1.9× the classical DD GEMM's.
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

These runs are from October 20, 2026, at 274e778: before the port, on 1.4.0's
kernels and its BLAS ABI. Since then 1.5.0 made the kernels faster with the
same bits (float4 tile loads, 32 × 32 tiles for small grids, register
prefetch, concurrent Gauss-3M products), kept the classical ZGEMM's
temporaries between calls, and moved the BLAS ABI's conversions to the GPU
(the staged path). All three change the comparison, so the tables below are
not 1.6.0's; they are to be repeated on 1.6.0 before its release.

**Setup.**
- Machine and power: M2 Max (38-core GPU), macOS 27.0.1, AC power.
- Build: `bench_strassen speed` at 274e778.
- Method: matrices resident on the GPU (native API, no conversion or
  transfer timed); median of 5 calls (3 for ZGEMM 8192) after a 0.5 s
  warm-up of each method. Classical and Strassen calls alternate back to
  back; Accelerate (`cblas_*gemm`, FP64, all CPU cores) runs on its own.
- Passes: two, ten minutes apart. The tables give the second; the
  first's speedup is in brackets where it was also on AC
  ([data](../benchmarks/results/2026-10-20-274e778-strassen/)).

DGEMM, square, wall time per call:

| N | classical DD | Strassen | Strassen speedup | Accelerate | Strassen vs Accelerate |
|---:|---:|---:|---:|---:|---:|
| 2048 | 25.9 ms | 27.5 ms | 0.94× (0.94×) | 28.9 ms | 1.05× |
| 2560 | 49.9 ms | 47.0 ms | 1.06× (1.05×) | 55.7 ms | 1.19× |
| 3072 | 85.7 ms | 80.5 ms | 1.06× (1.06×) | 90.8 ms | 1.13× |
| 3584 | 134.2 ms | 127.9 ms | 1.05× (1.04×) | 147.0 ms | 1.15× |
| 4096 | 200.0 ms | 185.3 ms | **1.08×** (1.08×) | 222.8 ms | 1.20× |
| 6144 | 674.6 ms | 612.1 ms | **1.10×** (1.10×) | 729.6 ms | 1.19× |
| 8192 | 1645 ms | 1468 ms | **1.12×** | 1762 ms | 1.20× |

At 8192 the classical DD GEMM runs at 668 GFLOP/s. Strassen's effective rate
there (2N³ per call) is 749 GFLOP/s. The ceiling is 8/7 = 1.143×.

DGEMM, rectangular and odd, M × K × N:

| shape | classical DD | Strassen | Strassen speedup | Accelerate |
|---|---:|---:|---:|---:|
| 8192 × 4096 × 8192 | 830.8 ms | 742.4 ms | 1.12× | 876.4 ms |
| 4096 × 8192 × 4096 | 420.4 ms | 380.2 ms | 1.11× | 456.0 ms |
| 8192 × 2048 × 8192 | 431.8 ms | 383.6 ms | 1.13× | 429.6 ms |
| 16384 × 4096 × 4096 (tall-skinny kernel) | 965.3 ms | 852.1 ms | 1.13× | 859.3 ms |
| 4096 × 4096 × 16384 | 840.2 ms | 747.3 ms | 1.12× | 907.7 ms |
| 4097³ | 207.9 ms | 201.2 ms | 1.03× | 255.3 ms |
| 2048 × 32768 × 2048 | 419.3 ms | 446.1 ms | **0.94×** | 477.6 ms |

ZGEMM, square, wall time per call and the speedup in command-buffer GPU time
alone:

| N | classical DD | Strassen | speedup, wall | speedup, GPU time | Accelerate | Strassen vs Accelerate |
|---:|---:|---:|---:|---:|---:|---:|
| 2048 | 82.6 ms | 83.4 ms | 0.99× (0.99×) | 0.95× | 111.4 ms | 1.34× |
| 3072 | 269.6 ms | 244.6 ms | 1.10× (1.11×) | 1.06× | 366.0 ms | 1.50× |
| 4096 | 622.0 ms | 563.4 ms | 1.10× (1.11×) | 1.08× | 877.3 ms | 1.56× |
| 6144 | 2212.6 ms | 1932.0 ms | 1.15× (1.14×) | 1.11× | 2935.8 ms | 1.52× |
| 8192 | 5266 ms | 4529 ms | 1.16× (1.14×) | 1.13× | 6987 ms | 1.54× |

At 274e778 the classical ZGEMM allocated its four Gauss-3M temporaries afresh
on every call. Their first use cost 3–150 ms between commit and the GPU
starting, while Strassen's were kept in its workspace. That is why the
wall-time speedup exceeds the GPU-time one. Since 1.5.0 the classical ZGEMM
keeps its temporaries too (pooled ZGEMM temporaries), so on 1.6.0 the
wall-time speedup should fall toward the GPU-time column.

**Through the BLAS ABI** (`ab_*gemm_blas`, column-major FP64 in and out,
`AB_MODE=gpu`; at 274e778 the host path, which converted on the CPU):
Strassen speeds up the whole call less than the native one. Packing,
FP64 ⇄ DD conversion and the epilogue take the same time as before.

| call | GPU | GPU with `AB_STRASSEN=1` | speedup |
|---|---:|---:|---:|
| DGEMM 4096³ | 490.7 ms | 476.2 ms | 1.03× |
| DGEMM 6144³ | 1344.1 ms | 1259.4 ms | 1.07× |
| DGEMM 8192³ | 2749.8 ms | 2600.3 ms | 1.06× |
| ZGEMM 4096³ | 1193.3 ms | 1139.7 ms | 1.05× |
| ZGEMM 6144³ | 3363.7 ms | 3169.8 ms | 1.06× |

These 'N','N' shapes stay slower than Accelerate (DGEMM 4096: 230.0 ms),
which is why `AB_MODE=auto` keeps them there.

### Where the time goes

At 4096 the classical product takes 199.7 ms of GPU time and Strassen
185.0 ms; 7/8 of the classical time would be 174.7 ms. The 10 ms difference
comes in two halves of about 5 ms each:
- **Half-size products** run a little less efficiently. Seven 2048³ products
  take 7 × 25.7 = 179.6 ms, at 670 GFLOP/s against 690 for 4096³.
- **Block passes.** On their own (measured with the products left out) they
  take 6.0 ms. They move about 2.1 GB at ~350 GB/s, close to the memory
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
65th mostly empty tile row and column, so it gains only 1.03×. Shapes whose
half-size products fill the GPU poorly lose: 2048 × 32768 × 2048 has
1024 × 1024 outputs per product, 256 threadgroups.

Strassen's first call for a shape grows the workspace (now the pool's
buffers): 31 ms more than a later call at 4096 and 48 ms at 8192. The first call in a process also
compiles its kernel, about 20 ms. From about 2.5 GB of buffers per command
buffer (DGEMM 8192, ZGEMM ≥ 6144), each Strassen call shows 12–50 ms
between commit and the GPU starting, about 1% of the call. Shared and private
storage for the workspace measured the same.

### Threshold

`AB_STRASSEN_MIN` applies to M, N and K together. The default, 4096, is where
every measured shape gained:
- square from 1.08×;
- rectangular 1.11–1.13×;
- the odd 4097³ 1.03×.

Square sizes from 2560 to 3584 gain 1.04–1.06×, and 2048 loses (0.94×). A
small M and N loses even with a long K. To use Strassen below 4096, set
`AB_STRASSEN_MIN` after timing your own shapes. For example, 2560 suits
square-ish products of that size.

## Energy

Energy per product, from a no-sudo probe (IOReport "GPU Energy" and SMC
power, the library called in a loop for 30 s, in 10 s windows;
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

Write u for the unit roundoff of the DD arithmetic. The library's precision
analysis ([precision envelope](vv/PRECISION_ENVELOPE.md)) uses a conservative
u = 8u₃₂² = 2⁻⁴⁵ (u₃₂ = 2⁻²⁴) for the GEMM inner loop's fused multiply-add;
DD addition and multiplication are tighter (3u₃₂² and 5u₃₂²). For C = AB with
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
Strassen-Winograd one is loose: at K = 4096 it allows 2.1 × 10⁻⁶ ‖A‖ ‖B‖,
which a result rounded to FP32 would meet on random data. That is why
`tests/test_strassen.c` also holds Strassen to a Frobenius error of 2⁻⁴⁰
relative to ‖ |A||B| ‖_F: losing the DD low words would exceed it.

### Measured

Errors against a Dot2 reference of all of C (compensated FP64, as accurate
as twice FP64 precision), `bench_strassen accuracy` at 274e778
([data](../benchmarks/results/2026-10-20-274e778-strassen/)). The algorithm
is unchanged by the port, and 1.5.0's kernel changes keep every product's
bits, so these errors are expected to hold on 1.6.0 for the default
precision; they are to be measured again with the speed runs. Under
`AB_PRECISION=fast` the products use its accumulation, a combination that
has not been measured. The three kinds of data are:

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
| 4096³ | scaled | 5.11e−14 | 9.58e−14 | 1.87 | 2.14e−15 |
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

- **Uniform scale, mixed signs.** Strassen's error is 1.87× the classical
  one at every size and shape, a little under one bit. Both grow like √K;
  against Accelerate's FP64 error the classical one is 24× larger,
  Strassen's 45×.
- **Positive data.** Strassen's error is half the classical one. With no
  cancellation, the error is dominated by the long accumulation in each
  product, and Strassen's products are half as long. (The March 2026 probes
  that reported Strassen "gaining bits" used data in [0, 1).) This is a
  property of the data, not of the algorithm.
- **Scaled rows and columns.** The normwise error is unchanged (1.87×), but
  componentwise Strassen fails. The error in the small elements is 10⁴ times
  the size of their own terms, so they have no correct digits; their error
  relative to |C_ij| reaches 6 × 10⁹. The classical GEMM keeps 1.5 × 10⁻¹⁴ on
  the same data, as its componentwise bound promises.
- The normwise errors are about 10⁵ below the worst-case bound: max|Ĉ − C|
  is 2.7 × 10⁻¹¹ ‖A‖ ‖B‖ at 4096³, against 2.1 × 10⁻⁶ allowed.

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
