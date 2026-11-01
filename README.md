<h1 align="center">apple-bottom</h1>

<p align="center"><strong>FP64-class BLAS for Apple Silicon GPUs — because two floats are better than no doubles.</strong></p>

<p align="center">
  <a href="https://github.com/grantdh/apple-bottom/actions/workflows/ci.yml"><img src="https://github.com/grantdh/apple-bottom/actions/workflows/ci.yml/badge.svg" alt="CI"></a>
  <a href="LICENSE"><img src="https://img.shields.io/badge/license-MIT-blue" alt="License: MIT"></a>
  <a href="#requirements"><img src="https://img.shields.io/badge/platform-macOS%20%7C%20Apple%20Silicon-lightgrey" alt="Platform: macOS on Apple Silicon"></a>
</p>

Apple Silicon GPUs have no FP64 units. apple-bottom stores each double as
the unevaluated sum of two floats (double-float, "DD") and does the
arithmetic in Metal compute shaders, on the GPU's FP32 ALUs. That gives
about 48 significant bits, ~10⁻¹⁵ relative error per element: close to
IEEE FP64 (53 bits), not the same.

You get DGEMM, ZGEMM and DSYRK, plus batched, asynchronous and
triangular-solve helpers, and the matrix-vector products DGEMV, ZGEMV and
ZHEMV ([BLAS-2](#blas-2-matrix-vector-products)), three ways:

- a **C API** on matrices that stay on the GPU between calls,
- a **device-buffer API** for codes that manage their own GPU memory,
- a drop-in **BLAS ABI** (`dgemm_`, `zgemm_`, `dgemv_`, `zgemv_`, `zhemv_`)
  for Fortran codes such as Quantum ESPRESSO, which sends each call to the
  GPU or to Accelerate according to measured timings.

## Is it faster than Accelerate?

It depends on where the matrices live.

- **Matrices kept on the GPU** (C API): for large GEMMs, yes, modestly.
  On an M2 Max the GPU overtakes Accelerate from about N = 1,500 for DGEMM
  and N = 1,000 for ZGEMM, and from N = 2,048 runs at 1.10–1.16×
  Accelerate's speed for DGEMM and 1.37–1.39× for ZGEMM.
- **One BLAS call at a time from host memory** (`dgemm_`/`zgemm_`): not by
  default. Since 1.5.0 a call routed to the GPU copies its operands into
  GPU buffers the library reuses, and the GPU converts them from FP64 to DD
  and back (the staged path). That path has not been calibrated yet, so the
  default routing (`AB_MODE=auto`) still follows 1.4.0's measurement, in
  which Accelerate was faster for every DGEMM and every Quantum ESPRESSO
  shape, and keeps those calls on Accelerate. Forced onto the GPU, Quantum
  ESPRESSO takes 1.01–1.06× its time on Accelerate (1.4.0: 1.06–1.43×).
  Linking apple-bottom is safe but, on this hardware, not a speed-up yet.
- **Accelerate and the GPU together** (in user code for now): splitting one
  GEMM between them finished 1.5–1.9× sooner than the faster engine alone
  in all 20 well-scaled cases of the
  [split benchmark](benchmarks/results/2026-10-28-415cb5d-hybrid-split/),
  with bit-for-bit repeatable results, and 1.47–1.63× sooner at 8192² over
  80 s of back-to-back work, where the power limit slows the GPU. The split
  keeps no more CPU cores busy than Accelerate alone
  ([CPU use](benchmarks/results/2026-10-28-415cb5d-hybrid-split/cpu-use/)).
  `dgemm_` and `zgemm_` do not split calls in 1.5.0; a split inside the
  library is planned for 1.6.0.
- **Matrix-vector products** (DGEMV, ZGEMV, ZHEMV) **on a matrix kept on the
  GPU**: yes, by a lot. They are bound by memory bandwidth, not arithmetic,
  and at 8192²–16384² the GPU runs them at 260–360 GB/s, 2.8–36× Accelerate's
  speed and 2.7–4.4× a threaded C loop's. One-shot BLAS-ABI calls pay to
  map the matrix each time; on the M2 Max, where it was measured,
  `AB_MODE=auto` sends only the large DGEMV `'N'`, ZGEMV `'N'` and ZHEMV
  calls that still win to the GPU ([BLAS-2](#blas-2-matrix-vector-products)).

The measurements are below, each with the data it came from.
[docs/COMPARISON.md](docs/COMPARISON.md) compares apple-bottom with the
other ways to multiply FP64 matrices on a Mac.

## Quick start

```bash
git clone https://github.com/grantdh/apple-bottom.git
cd apple-bottom
make && make test
```

Needs macOS 14 or later on Apple Silicon and the Xcode command-line tools;
see [Requirements](#requirements).

## Performance

M2 Max (38-core GPU, 8P+4E CPU cores, 64 GB), macOS 27.0.1, on AC power;
Accelerate is Apple's `cblas_*gemm` on the CPU (AMX).

**Matrices resident on the GPU** (native API: upload once, time
`ab_dgemm`/`ab_zgemm` only), GFLOP/s, median of three runs of
`bench_paper`, each one warm-up then the mean of 3–10 calls, measured at
3253a22 before the 1.4.0 release
([data](benchmarks/results/2026-10-14-3253a22-bench-paper/)). 1.5.0's
kernel changes (float4 tile loads, 32 × 32 tiles for small outputs,
register prefetch, pooled ZGEMM temporaries) keep every result's bits and
are not in these numbers:

| N | DGEMM, GPU | DGEMM, Accelerate | GPU vs Accelerate | ZGEMM, GPU | ZGEMM, Accelerate | GPU vs Accelerate |
|---:|---:|---:|---:|---:|---:|---:|
| 512 | 284 | 634 | 0.45× | 502 | 705 | 0.71× |
| 1024 | 535 | 606 | 0.88× | 683 | 611 | 1.12× |
| 1536 | 640 | 615 | 1.04× | 806 | 622 | 1.30× |
| 2048 | 664 | 580 | 1.15× | 844 | 619 | 1.37× |
| 3072 | 683 | 625 | 1.10× | 868 | 625 | 1.39× |
| 4096 | 699 | 604 | 1.16× | – | – | – |

ZGEMM rates count 8N³ operations for both; the GPU uses Gauss's
three-multiplication form, which performs 25% fewer. Normwise relative
error of the GPU result: 1.8×10⁻¹⁴ at N = 512 to 5.1×10⁻¹⁴ at N = 4096.

**Through the BLAS ABI, on 1.4.0's host path** (`ab_*gemm_blas`, i.e.
`dgemm_`/`zgemm_`, with packing and conversion on the CPU and transfer
included; `AB_HOST_PACK=1` selects that path in 1.5.0), Accelerate time ÷
GPU time, median of 7 after 3 warm-ups, measured at 52f1880
([data](benchmarks/results/2026-10-14-52f1880-calibrate/)):

| Shapes | GPU vs Accelerate | `AB_MODE=auto` |
|--------|------------------:|----------------|
| DGEMM, square and transposed, 256–6144 | 0.06–0.84× | Accelerate |
| ZGEMM, square, 256–6144 | 0.17–0.89× | Accelerate |
| ZGEMM, one operand transposed, ≤ 6144 | 0.45–1.02× | Accelerate |
| ZGEMM, both operands transposed, ≥ 4096 | 1.11–1.20× | **GPU** |
| Quantum ESPRESSO si64 Davidson ZGEMMs | 0.14–0.76× | Accelerate |
| Quantum ESPRESSO si64 PAW DGEMMs | 0.09–0.15× | Accelerate |

The built-in tables are keyed by chip and GPU core count, and so far only the
M2 Max (38-core GPU) has one: on any other chip `auto` keeps every call on
Accelerate. `make calibrate` repeats this measurement on your machine, through
the staged path, and files the result as your chip's table
(`benchmarks/calibration/<chip>/`; rebuild to compile it in, and a pull
request with it adds your chip for everyone), or point `AB_DISPATCH_TABLE` at
its `dispatch_table.txt`.

The staged path, the default in 1.5.0, has not been calibrated on the M2
Max yet. With registered, well-scaled operands it ran large squares faster
than Accelerate in the
[split benchmark](benchmarks/results/2026-10-28-415cb5d-hybrid-split/)
(median ms, GPU alone against Accelerate alone: DGEMM 4096² 195.6 against
222.6, ZGEMM 4096² 581.7 against 878.9). In Quantum ESPRESSO it is not
faster than Accelerate (below).

**Quantum ESPRESSO**, 64-atom Si, PAW, inputs A and B at 1 rank and at
8 ranks on one QE develop build compiled `-O3`
([1.5.0 release check](benchmarks/results/2026-10-28-415cb5d-qe-si64-release-check/)
and the [staged-path fix](benchmarks/results/2026-10-29-351d56c-staged-refusals/)
measured after it; [VAL-1](tests/validation/VAL001_QE_Si64.md) has the
earlier runs with QE compiled `-O0`):

- In the release check the total energy, the energy of every SCF iteration
  and every printed band energy are the same in every mode and run. With
  the fix, every run of an input gives the same total energy and number of
  SCF iterations.
- `AB_MODE=auto` keeps all of QE's GEMMs on Accelerate; its wall time is
  within −1.1% to +0.6% of `cpu` mode's.
- Forcing the GPU (`AB_MODE=gpu`) is not faster than Accelerate. With the
  fix, 1.5.0 takes 1.01–1.06× the `cpu` time (mean of two `gpu` runs
  against one `cpu` run): input B at 1 rank 191.3 s against 182.5 s, A at
  8 ranks 102.4 s against 96.6 s, B at 8 ranks 54.6 s against 54.3 s. The
  release check's build before the fix took 1.17–1.20× in all four
  configurations, and 1.4.0 on the same QE build 1.06–1.43×.

## BLAS-2: matrix-vector products

DGEMV, ZGEMV (`'N'`, `'T'`, `'C'`) and ZHEMV run on the GPU straight from the
caller's FP64 matrix (column-major, any `lda`): the kernels convert each
element to double-float in registers, accumulate in double-float and add
partial sums in an order fixed by the shape, so results are bitwise
reproducible. A matrix-vector product reads each element of A once, so it is
bound by memory bandwidth, where the GPU is far ahead of the CPU and the
double-float arithmetic costs nothing extra. What costs is getting A in
front of the GPU, so keep it there, as a Lanczos, Haydock or other Krylov
solver can: register the array once, without a copy, and call the
device-buffer routines for every product.

```c
#include "apple_bottom_device.h"

ab_dev_buffer_t hA = ab_dev_register(A, (size_t)lda * n * sizeof(double));  // no copy
for (int it = 0; it < iterations; it++) {
    ab_dev_dgemv(AB_NO_TRANS, m, n, 1.0, hA, 0, lda, v, 1, 0.0, w, 1);       // w = A v
    /* ... the solver's own vector work on v and w ... */
}
ab_dev_free(hA);   // before A is freed
```

M2 Max, y = A x for square A of side 4096 / 8192 / 16384: GB/s of A read
(ZHEMV: of one triangle) per call, median wall time with the GPU warmed up;
GEMV: the median of five runs at 4096 and 8192 and two at 16384, ZHEMV: of
five runs ([data](benchmarks/results/2026-10-19-ba345df-gemv/), ZHEMV
[data](benchmarks/results/2026-10-29-c37db7c-zhemv/), `make bench-gemv`):

| y = A x | GPU, A resident | GPU, A mapped per call (`dgemv_`) | Accelerate | plain C, best of 4/8/12 threads |
|---|---:|---:|---:|---:|
| DGEMV `'N'` | 223 / 302 / 317 | 59–65 | 19–21 | 53 / 68 / 85 |
| DGEMV `'T'` | 229 / 320 / 354 | 60–72 | 51–60 | 115 / 116 / 114 |
| ZGEMV `'N'` | 274 / 316 / 320 | 42–64 | 37–39 | 77 / 77 / 86 |
| ZGEMV `'T'`, `'C'` | 276–277 / 334–336 / 358–359 | 43–66 | 108–120 | 123–124 / 112–123 / 117–119 |
| ZHEMV `'U'`, `'L'` | 159–161 / 261–273 / 305–312 | 35–39 | 8.3–9.1 | 59–71 / 86–92 / 100–102 |

- **Resident:** 3.8–16× Accelerate's speed for DGEMV, 2.4–8.6× for ZGEMV
  and 17–36× for ZHEMV, and 2.0–4.4× the best threaded C loop. At 16384²
  GEMV reads A at 317–359 GB/s of the M2 Max's 400. ZHEMV reads its stored
  triangle once, adding each element into its row's sum and, conjugated,
  into its column's: twice GEMV's arithmetic per byte, which holds it to
  305–312 GB/s.
- **Per call** there are about 0.2 ms of submit and wait on top, so 4096²
  (0.59–0.98 ms per call) runs at 223–277 GB/s, ZHEMV's triangle
  (0.84–0.85 ms) at 159–161 GB/s, and smaller matrices lose more. A cold
  GPU runs about the first 0.3 s of calls at half speed while its clocks
  rise.
- **Accuracy:** 2.5×10⁻¹⁶ to 1×10⁻¹⁵ relative to |A||x| on these matrices,
  against 2×10⁻¹⁷ to 2×10⁻¹⁶ for Accelerate: double-float level, about ten
  times FP64's error. The kernels also track the FP64 exponents of every
  product, at no extra memory traffic, and a call whose data fall outside
  double-float's full-precision range (FP32's exponents), or that involves
  NaN or Inf, runs on Accelerate instead, before y is written.
- **One-shot calls** (`dgemv_`, `zgemv_`, `zhemv_`, `ab_*gemv_blas`) map A
  for the GPU on every call, at 10–22 ms/GB. On the M2 Max (38-core GPU),
  where this was measured, `AB_MODE=auto` sends one to the GPU only where it
  was at least 1.10× faster than Accelerate in two runs; on any other chip
  every one-shot call stays on Accelerate until it is measured there. On
  the M2 Max: DGEMV `'N'` from 2048² to 16384² elements (3.0–3.2× in the
  table), ZGEMV `'N'` from 3072² to 12288², and ZHEMV from 2048 to 16384
  (3.8–4.7×). DGEMV `'T'` (0.94–1.40×, too close to call) and ZGEMV `'T'`
  and `'C'` (0.4–0.6×) stay on Accelerate, as do smaller and larger calls;
  most one-shot calls in a real program are small. These are one-process
  numbers: MPI ranks sharing the GPU should expect less
  ([routing data](benchmarks/results/2026-10-19-ba345df-gemv/README.md),
  [ZHEMV's](benchmarks/results/2026-10-19-1bc503a-zhemv/README.md),
  [Integration Guide](docs/INTEGRATION.md#routing-ab_modeauto)).

## Using it

### From C

```c
#include "apple_bottom.h"

ab_init();

ABMatrix A = ab_matrix_create(2048, 2048);
ABMatrix B = ab_matrix_create(2048, 2048);
ABMatrix C = ab_matrix_create(2048, 2048);
ab_matrix_upload(A, data_A, true);   // FP64 -> DD, once
ab_matrix_upload(B, data_B, true);

ab_dgemm(A, B, C);                   // C = A * B on the GPU, FP64-class

ab_matrix_download(C, result, true); // DD -> FP64
ab_matrix_destroy(A); ab_matrix_destroy(B); ab_matrix_destroy(C);
ab_shutdown();
```

`ABMatrix` data is row-major (C order); the BLAS ABI below is
column-major, as in BLAS. Compile against the in-tree build with

```bash
clang -I/path/to/apple-bottom/include my_prog.c \
      -L/path/to/apple-bottom/build -Wl,-rpath,/path/to/apple-bottom/build -lapplebottom
```

(static linking and pkg-config: [Integration Guide](docs/INTEGRATION.md#building-your-program)).
A runnable version is in [`examples/01_basic_dgemm`](examples/01_basic_dgemm/main.c)
(`make examples`).

| Category | Functions |
|----------|-----------|
| GEMM and friends | `ab_dgemm`, `ab_dgemm_scaled`, `ab_zgemm`, `ab_zgemm_ex`, `ab_dsyrk`, `ab_zherk`, `ab_dtrsm`, `ab_dgesv_mpir` |
| Matrices | `ab_matrix_create`, `ab_matrix_upload`, `ab_matrix_download`, `ab_matrix_zero`, `ab_matrix_copy`, `ab_matrix_add`, `ab_matrix_sub`, `ab_matrix_scale` |
| Batches and async | `ab_batch_*`, `ab_dgemm_batched`, `ab_zgemm_batched`, `ab_dgemm_async`, `ab_zgemm_async`, `ab_future_*` |
| Pools and sessions | `ab_pool_*`, `ab_session_*` (fewer allocations in iterative codes) |
| Device buffers | `ab_dev_malloc`, `ab_dev_register`, `ab_dev_memcpy_*`, `ab_dev_dgemm`, `ab_dev_zgemm`, `ab_dev_dgemv`, `ab_dev_zgemv`, `ab_dev_zhemv`, … ([`apple_bottom_device.h`](include/apple_bottom_device.h)) |
| BLAS ABI | `ab_dgemm_blas`, `ab_zgemm_blas`, `ab_dgemv_blas`, `ab_zgemv_blas`, `ab_zhemv_blas` (column-major, BLAS arguments), `ab_register_host_memory` / `ab_unregister_host_memory` (GEMM operands the GPU reads in place), `ab_get_last_dispatch_path` |

### From Fortran (and other BLAS hosts)

The library exports `dgemm_`, `zgemm_`, `dgemv_`, `zgemv_` and `zhemv_`.
Link it ahead of your BLAS and the existing `CALL DGEMM` / `CALL ZGEMM` /
`CALL DGEMV` / `CALL ZGEMV` / `CALL ZHEMV` reach it, with reference-BLAS
argument checking; everything else still comes from your BLAS library:

```makefile
LIBS = -L/path/to/apple-bottom/build -Wl,-rpath,/path/to/apple-bottom/build -lapplebottom $(BLAS_LIBS)
```

| Variable | Effect |
|----------|--------|
| `AB_MODE=auto` (default) | GPU only for shapes measured ≥ 1.10× faster there on this chip (`src/dispatch_table.h`, one table per calibrated chip, the M2 Max's measured on 1.4.0's host path; an uncalibrated chip stays on Accelerate) |
| `AB_MODE=cpu` / `AB_MODE=gpu` | everything on Accelerate / everything above `AB_MIN_GPU_DIM` on the GPU |
| `AB_REPORT=1` | at exit: calls and time by route, GPU stage costs, fallbacks, and why calls left the staged path |
| `AB_TRACE=<path>` | a line per call in `<path>.<pid>`: arguments, timestamps, route, GPU stages |
| `AB_DISPATCH_TABLE=<file>` | route from another calibration, e.g. one from your own machine |
| `AB_HOST_PACK=1` | GPU calls pack and convert on the CPU, as in 1.4.0, instead of on the GPU; the same bits |
| `AB_STAGING_MB=<n>` | GPU buffers kept between calls, at most n MB per process (default 2048) |

The [Integration Guide](docs/INTEGRATION.md) covers C, Fortran, Python
(experimental bindings in [`python/`](python/)), a Quantum ESPRESSO
`make.inc` and every environment variable.

## How it works

Each value is stored as an unevaluated sum `hi + lo` of two FP32 numbers,
normalized so that `|lo|` is at most half an ulp of `hi`; the pair holds
about 48 significant bits of the FP64 value. Arithmetic keeps that form
with Dekker/Knuth error-free transformations (TwoSum, FMA-based TwoProd).
Products use a two-FMA variant of the DWTimesDW2 algorithm of Joldes,
Muller and Popescu (2017). Its relative error is proven below 6.000001u²
for u = 2⁻²⁴. The fused multiply-add of the GEMM inner loop has absolute
error at most 13u²|ab| + 3u²|c|, so a GEMM with inner dimension K is within
(3K+13)u²|A||B| componentwise
([proof and scope](docs/vv/PRECISION_ENVELOPE.md); checked by
`make test-dd-bound`).

- The shaders are compiled without fast math (`MTLMathModeSafe`, or
  `fastMathEnabled = NO` before macOS 15). Fast math lets the compiler
  reassociate the error-free transformations and costs about seven
  decimal digits.
- GEMM tiles are 64 × 64 with 4 × 4 outputs per thread, with a 128 × 16
  variant for tall-skinny shapes (M ≥ 4N) and a 32 × 32 one (2 × 2 per
  thread) for grids of up to about a thousand threadgroups (fewer for
  concurrent products), whose finer tiles spread more evenly over the GPU's
  cores. On grids that leave
  the GPU's cores with few threadgroups, and on the 32 × 32 tiles always,
  a kernel loads the next K-tile into registers while it computes the
  current one. Every variant computes each output the same way. Long dot
  products are renormalized every 128 terms.
- ZGEMM uses Gauss's three-multiplication form, three real DD GEMMs
  instead of four, unless its real and imaginary parts are imbalanced
  enough to cost the imaginary part its accuracy; then it takes four. The
  BLAS ABI and the device-buffer API decide per row of op(A) and column of
  op(B), the native API per matrix
  ([details](docs/vv/LIMITATIONS.md#complex-products)).
- Through the BLAS ABI, a call routed to the GPU copies its operands into
  GPU buffers the library keeps between calls (`AB_STAGING_MB`), or reads
  memory registered with `ab_register_host_memory` in place. The GPU then
  converts them to double-float, scales rows and columns far from 1 by
  powers of two, multiplies, converts back and applies alpha and beta (the
  host applies a ZGEMM's alpha and beta unless alpha = 1 and beta is 0 or
  1). Its conversions work on the FP64 bit patterns and round as the host
  does, so the results are bit for bit those of the host path
  (`AB_HOST_PACK=1`), which packs, converts and applies alpha and beta on
  the CPU as 1.4.0 did.

## Accuracy and testing

DGEMM's normwise relative error grows like √N·10⁻¹⁵ on random matrices,
about 5 × 10⁻¹⁴ at N = 4096 (V-2 convergence study). `make test` checks
precision against a compensated FP64 reference, correctness of every API
against Accelerate, the device-buffer API, the routing rules, the BLAS-ABI
entry points in a conformance harness (with long-K precision, data far
outside FP32's exponent range, imbalanced complex parts, determinism,
concurrency, memory and fork checks), the staged path against the host
path bit for bit, and the per-call trace and forced GPU failures; it also
holds GPU-routed BLAS results to the bits 1.4.0 produced. `make ci-local`
adds adversarial inputs and the convergence study;
[tests/README.md](tests/README.md) gives the test counts.

[V&V report](docs/vv/VV_REPORT.md) · [precision envelope](docs/vv/PRECISION_ENVELOPE.md) ·
[limitations and scope](docs/vv/LIMITATIONS.md) ·
[Quantum ESPRESSO validation](tests/validation/VAL001_QE_Si64.md)

## Requirements

- Apple Silicon (M1 or later) with macOS 14 or later
- Xcode or the Command Line Tools (clang); gfortran only for the Fortran tests
- Measured on an M2 Max; other chips are expected to work but have no
  calibration or validation yet

## Limitations

- **Not IEEE FP64.** About 48 significant bits instead of 53, and no
  bit-for-bit agreement with CPU BLAS.
- **FP32's exponent range, worked around.** DD values are FP32 pairs.
  Products below about 2⁻⁷⁸ lose their low part, values below 2⁻¹²⁶ flush to
  zero on the GPU, and sums overflow at 2¹²⁸.
  - The BLAS ABI scales each row of op(A) and column of op(B) whose largest
    entry lies outside 2^±20 by an exact power of two, and undoes it in C.
    It runs on Accelerate whatever that cannot keep in range, and any call
    with NaN or Inf in A or B (or in C, for DGEMM with beta ≠ 0).
  - The native `ABMatrix` API keeps one power-of-two scale per matrix, so a
    matrix whose nonzero entries span more than about 2⁷⁸ (10²³) can still
    lose precision in its products there.
  - Data inside those windows computes exactly as before.
  - [Details](docs/vv/LIMITATIONS.md#exponent-range).
- **Per-call overhead.** Through 1.4.0's BLAS ABI, which packed and
  converted on the CPU, every measured shape was slower than Accelerate
  except ZGEMM with both operands transposed at N ≥ 4096; see the table
  above. 1.5.0's staged path does that work on the GPU but has not been
  calibrated yet, so `AB_MODE=auto` still routes by that table. BLAS-2 calls win
  on the GPU only with the matrix resident, or for the large one-shot shapes
  listed under [BLAS-2](#blas-2-matrix-vector-products).
- **BLAS-2 range.** A DGEMV/ZGEMV/ZHEMV row whose largest |a_ij x_j| is below
  about 10⁻¹⁸ max |x| (or above 10³³), or a product that loses bits to
  FP32's range and matters at double-float precision, sends the call to
  Accelerate: a matrix scaled far outside 10⁻¹⁸–10³³ always runs there.
- **Threads.** Calls on different matrices may come from different
  threads; one matrix, pool or session must not be used by two threads at
  once. All work shares one Metal command queue.
- **Size.** rows × cols ≤ 2³¹ − 1 per matrix and within the GPU's
  `maxBufferLength`; larger BLAS-ABI operands run on Accelerate.
- **ZHERK on the GPU is slower than Accelerate**; call `cblas_zherk`.

## Project layout

```
include/apple_bottom.h          public API (and apple_bottom_device.h)
src/apple_bottom.m              Metal context and DD kernels
src/device_api.m                device-buffer API (apple_bottom_device.h)
src/blas_wrapper.c              BLAS ABI: the staged path, the host path, fallbacks
src/gpu_stage.h                 the staged path's GPU buffers and conversion (internal)
src/blas2.m, blas2_abi.c        BLAS-2 kernels; dgemv_ / zgemv_ / zhemv_, ab_dev_*gemv
src/dispatch.c, dispatch_table.h  AB_MODE routing and the calibrated tables, one per chip
src/fortran_bridge.c            dgemm_ / zgemm_ / ab_*gemm_ entry points
src/profiling/                  AB_PROFILE / AB_REPORT accounting, AB_TRACE
tests/                          test suites (tests/README.md)
benchmarks/                     benchmarks, calibration and their results
docs/                           integration guide, V&V package, design notes
python/                         experimental ctypes bindings
```

## Contributing

Bug reports, calibration results from other chips (`make calibrate`) and
pull requests are welcome; see [CONTRIBUTING.md](CONTRIBUTING.md).

## Citation

```bibtex
@software{heileman2026applebottom,
  author    = {Heileman, Grant David},
  title     = {apple-bottom: FP64-class BLAS for Apple Silicon GPUs},
  year      = {2026},
  url       = {https://github.com/grantdh/apple-bottom},
  version   = {1.6.0}
}
```

## License

MIT — see [LICENSE](LICENSE).
