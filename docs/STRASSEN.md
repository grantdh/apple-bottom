# Strassen-Winograd GEMM (opt-in)

apple-bottom's double-float GEMM kernel already runs within about 10% of the
M2 Max GPU's instruction-issue ceiling, so a large product only gets faster by
doing less arithmetic. One level of Winograd's variant of Strassen's algorithm
multiplies two matrices with 7 products of half-size blocks instead of 8,
plus 15 additions of blocks. That removes an eighth of the multiplications,
at the price of a weaker error bound: it is normwise, not componentwise.

It is off by default. With it off, every result is bitwise what it was.

## Using it

| | |
|---|---|
| `ab_dgemm_strassen(alpha, A, B, beta, C)` | what `ab_dgemm_scaled` computes, C = alpha A B + beta C |
| `ab_zgemm_strassen(Ar, Ai, Br, Bi, Cr, Ci)` | what `ab_zgemm` computes: Gauss's three-multiplication form, each of its three real products by Strassen-Winograd |
| `AB_STRASSEN=1` | `ab_dgemm`, `ab_dgemm_scaled`, `ab_zgemm` and `ab_zgemm_ex` call the two above when M, N and K are all at least `AB_STRASSEN_MIN`. That includes the BLAS-ABI (`dgemm_`, `zgemm_`) and device-API GEMMs that the routing sends to the GPU, which run through them. |
| `AB_STRASSEN_MIN=<n>` | that threshold, default 4096 |

Both variables are read once per process. The batched, asynchronous and
`ab_batch_*` GEMMs, DSYRK and ZHERK are never routed. The two entry points
work for any shape: an odd dimension is padded with zeros as the blocks are
read and cropped as C is written.

**Memory.** A Strassen call needs a workspace of two ⌈M/2⌉ × ⌈K/2⌉ blocks,
two ⌈K/2⌉ × ⌈N/2⌉ and three ⌈M/2⌉ × ⌈N/2⌉: 1.75 N² double-float values,
14 bytes per element of a square C, so 940 MB at N = 8192. ZGEMM adds
M × K, K × N and M × N for the sums Ar + Ai and Br + Bi and one of the real
products, 4.75 N² in all. The workspace is kept between calls (a new Metal
buffer costs 12–16 ms per GB the first time the GPU uses it), grows when a
call needs more, and is freed by `ab_shutdown()`.

**Threads.** Calls that use the workspace run one at a time; a second
thread's Strassen call waits for the first.

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
encodes one call in one serial compute encoder of one command buffer, so each
step sees the writes of the ones before it:

- **The 7 products** run on the library's own GEMM kernels, dispatched exactly
  as the classical ZGEMM dispatches its products, so any change to the kernels
  or their dispatch applies to them too. They need contiguous operands: the
  sums S and T are written that way, and P1–P4's quadrant operands are
  copied (6 copies).
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

### What the tests check

`make test` runs `tests/test_strassen.c` against a compensated FP64 (Dot2)
reference:

- Shapes: even, odd, tiny (1 × 1 × 1 to 5 × 7 × 3), short and long K, and a
  tall product whose half-size GEMMs take the tall-skinny kernel.
- Error bounds: Strassen within the normwise bound and the Frobenius limit;
  the classical GEMM within its componentwise bound (K + 2) u.
- Epilogue: alpha and beta, and beta = 0 with NaN in C.
- ZGEMM, against the same bounds.
- Argument checks.
- Bitwise determinism after the workspace served other shapes and after
  `ab_shutdown()`.
- Routing, in child processes with their own environment: `AB_STRASSEN`
  unset or 0 leaves every GEMM bitwise classical even with
  `AB_STRASSEN_MIN=1`. `AB_STRASSEN=1` sends a call with M, N, K ≥ the
  threshold to Strassen, bitwise, through the native API and through
  `dgemm_` and `zgemm_`, and leaves smaller ones classical.

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
