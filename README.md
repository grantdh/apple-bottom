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
triangular-solve helpers, three ways:

- a **C API** on matrices that stay on the GPU between calls,
- a **device-buffer API** for codes that manage their own GPU memory,
- a drop-in **BLAS ABI** (`dgemm_`, `zgemm_`) for Fortran codes such as
  Quantum ESPRESSO, which sends each call to the GPU or to Accelerate
  according to measured timings.

There is also a [double-float FFT](#double-float-fft): batched 1D, 2D and
3D complex transforms, FP64 in and out.

## Is it faster than Accelerate?

It depends on where the matrices live.

- **Matrices kept on the GPU** (C API): for large GEMMs, yes, modestly.
  On an M2 Max the GPU overtakes Accelerate from about N = 1,500 for DGEMM
  and N = 1,000 for ZGEMM, and from N = 2,048 runs at 1.10–1.16×
  Accelerate's speed for DGEMM and 1.37–1.39× for ZGEMM.
- **One BLAS call at a time from host memory** (`dgemm_`/`zgemm_`): almost
  never on an M2 Max. Each call converts FP64 to DD and back, and
  Accelerate is faster for every DGEMM and every Quantum ESPRESSO shape
  measured. The default routing (`AB_MODE=auto`) therefore keeps those
  calls on Accelerate, so linking apple-bottom is safe but, on this
  hardware, not a speed-up.

The measurements are below, each with the data it came from.

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
`bench_paper`, each one warm-up then the mean of 3–10 calls
([data](benchmarks/results/2026-10-14-3253a22-bench-paper/)):

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

**Through the BLAS ABI** (`ab_*gemm_blas`, i.e. `dgemm_`/`zgemm_`:
conversion and transfer included), Accelerate time ÷ GPU time, median of
7 after 3 warm-ups ([data](benchmarks/results/2026-10-14-52f1880-calibrate/)):

| Shapes | GPU vs Accelerate | `AB_MODE=auto` |
|--------|------------------:|----------------|
| DGEMM, square and transposed, 256–6144 | 0.06–0.84× | Accelerate |
| ZGEMM, square, 256–6144 | 0.17–0.89× | Accelerate |
| ZGEMM, one operand transposed, ≤ 6144 | 0.45–1.02× | Accelerate |
| ZGEMM, both operands transposed, ≥ 4096 | 1.11–1.20× | **GPU** |
| Quantum ESPRESSO si64 Davidson ZGEMMs | 0.14–0.76× | Accelerate |
| Quantum ESPRESSO si64 PAW DGEMMs | 0.09–0.15× | Accelerate |

`make calibrate` repeats this measurement on your machine and writes a new
routing table: rebuild to compile it in, or point `AB_DISPATCH_TABLE` at its
`dispatch_table.txt`.

**Quantum ESPRESSO** (develop, 7.5), 64-atom Si, PAW
([VAL-1](tests/validation/VAL001_QE_Si64.md), QE compiled `-O0`, and the
[release check](benchmarks/results/2026-10-15-c5a1fa6-qe-si64-opt/), QE
compiled `-O3`): the total energy and every printed band energy are the same
in every mode and run, also in `gpu` mode, where the calls that make up
88–99% of the GEMM work run on the GPU. `AB_MODE=auto` keeps all of QE's
GEMMs on Accelerate and matches `cpu` mode (within 0.2% at `-O0`, 1.7% at
`-O3`). Forcing the GPU makes the runs slower: 1.3–17.3% at `-O0`, and at
`-O3` 14–40% with one rank and 5–21% with 8 ranks. With one rank the GPU
kernels alone take 1.00–1.72× as long as Accelerate needs for the same calls
(2.7–12.6× for the PAW DGEMMs). With 8 ranks, where each rank's Accelerate
has one thread, they take 0.65–0.78× (PAW DGEMMs 1.5–4.3×), but packing,
transfers and the epilogue cost more than that margin.

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
| Device buffers | `ab_dev_malloc`, `ab_dev_memcpy_*`, `ab_dev_dgemm`, `ab_dev_zgemm`, … ([`apple_bottom_device.h`](include/apple_bottom_device.h)) |
| BLAS ABI | `ab_dgemm_blas`, `ab_zgemm_blas` (column-major, BLAS arguments), `ab_get_last_dispatch_path` |

### From Fortran (and other BLAS hosts)

The library exports `dgemm_` and `zgemm_`. Link it ahead of your BLAS and
the existing `CALL DGEMM` / `CALL ZGEMM` reach it, with reference-BLAS
argument checking; everything else still comes from your BLAS library:

```makefile
LIBS = -L/path/to/apple-bottom/build -Wl,-rpath,/path/to/apple-bottom/build -lapplebottom $(BLAS_LIBS)
```

| Variable | Effect |
|----------|--------|
| `AB_MODE=auto` (default) | GPU only for shapes measured ≥ 1.10× faster there (`src/dispatch_table.h`) |
| `AB_MODE=cpu` / `AB_MODE=gpu` | everything on Accelerate / everything above `AB_MIN_GPU_DIM` on the GPU |
| `AB_REPORT=1` | at exit: calls and time by route, GPU stage costs, fallbacks |
| `AB_TRACE=<path>` | a line per call in `<path>.<pid>`: arguments, timestamps, route, GPU stages |
| `AB_DISPATCH_TABLE=<file>` | route from another calibration, e.g. one from your own machine |

The [Integration Guide](docs/INTEGRATION.md) covers C, Fortran, Python
(experimental bindings in [`python/`](python/)) and a Quantum ESPRESSO
`make.inc`.

## Double-float FFT

[`apple_bottom_fft.h`](include/apple_bottom_fft.h) adds batched 1D, 2D and
3D complex FFTs on interleaved FP64 data (`fftw_complex`, Fortran
`COMPLEX(8)`), computed on the GPU in double-float. It needs no `ab_init()`.

```c
#include "apple_bottom_fft.h"

ABStatus st;
ab_fft_plan_t p = ab_fft_plan_3d(72, 72, 72, nbands, AB_FFT_BACKWARD, AB_FFT_NORM_NONE, &st);
ab_fft_execute(p, psic, psic);       // in place, on any array; FP64 in and out
ab_fft_destroy(p);
```

- **Layout:** column-major, `n[0]` contiguous, as in Fortran and Quantum
  ESPRESSO's `psic(nr1x, nr2x, nr3x)`. `ab_fft_plan_many` takes FFTW's
  advanced-interface strides, distances and embeddings, with the
  dimensions in FFTW's Fortran order. Forward is `exp(−i…)` as
  `FFTW_FORWARD`; each plan states its scale: 1, 1/N or 1/√N.
- **Lengths:** factors 2, 3, 5, 7, 11 and 13; up to 2048 per dimension,
  and up to 2048² in 1D (two passes).
- **Memory:** any array works, through staging buffers the plan reuses,
  copied in 32 MB chunks that overlap the GPU work. Memory from
  `ab_fft_malloc`, or registered with `ab_fft_register`, is used in place.
  Nothing is wrapped implicitly.

**Accuracy.** Not FP64: against a double-double reference the relative L2
error on random data is 5.9–7.5 × 10⁻¹⁵ for 48³ to 128³ grids, 20–24× FFTW's
FP64 error (2.5–3.7 × 10⁻¹⁶). Additions use a faster double-float sum
whose error bound is absolute, which suits an FFT's normwise bound; the
exponent range is FP32's.

**Speed.** Forward 3D transforms, ms per transform, M2 Max on AC power.
FFTW is 3.3.10 from Homebrew with `FFTW_PATIENT` plans, faster in every case
than a NEON build of FFTW. "8 cores" is the faster of one 8-thread plan and
8 single-thread workers, as 8 MPI ranks would run them
([data, all eight grids and vDSP](benchmarks/results/2026-10-23-c138543-fft/)):

| n³ | batch | GPU staged | GPU zero copy | FFTW 1 thread | FFTW 8 cores |
|---:|---:|---:|---:|---:|---:|
| 48 | 1 | 0.310 | 0.223 | 0.527 | 0.164 |
| 72 | 1 | 0.668 | 0.387 | 1.923 | 0.327 |
| 96 | 1 | 0.959 | 0.551 | 4.899 | 0.825 |
| 120 | 1 | 1.637 | 1.024 | 10.513 | 2.442 |
| 48 | 128 | 0.061 | 0.044 | 0.600 | 0.106 |
| 72 | 128 | 0.207 | 0.173 | 1.996 | 0.310 |
| 96 | 128 | 0.482 | 0.370 | 4.936 | 0.781 |
| 120 | 128 | 0.932 | 0.810 | 10.551 | 1.554 |

- **Against FFTW on one thread** the GPU is faster on every grid from 48³ to
  128³: 1.7–6.4× staged and 2.4–15× with zero copy one transform at a time,
  9–14× and 12–20× in batches of 128.
- **Against eight cores**, a batch of 128 (a band set) is 1.3–2.3× faster
  staged and 1.8–3.2× faster with zero copy on every grid. One transform at
  a time it takes 96³ or more with zero copy (1.5–2.4×) and 120³ or more
  staged (1.0–1.5×); below that FFTW's eight threads win.
- **Energy** per transform (wall power above idle, 72³ and 120³): zero copy
  needs 16–20% less than FFTW on eight single-thread workers, staged 13–16%
  more, spent on the copies.

## How it works

Each value is stored as an unevaluated sum `hi + lo` of two FP32 numbers,
normalized so that `|lo|` is at most half an ulp of `hi`; the pair holds
about 48 significant bits of the FP64 value. Arithmetic keeps that form
with Dekker/Knuth error-free transformations (TwoSum, FMA-based TwoProd). Products use the DWTimesDW2 algorithm of Joldes,
Muller and Popescu (2017), with relative error below 5u² for u = 2⁻²⁴; the
fused multiply-add of the GEMM inner loop has a conservative 8u² envelope
([derivation and scope](docs/vv/PRECISION_ENVELOPE.md)).

- The shaders are compiled without fast math (`MTLMathModeSafe`, or
  `fastMathEnabled = NO` before macOS 15). Fast math lets the compiler
  reassociate the error-free transformations and costs about seven
  decimal digits.
- GEMM tiles are 64 × 64 with 4 × 4 outputs per thread and a 128 × 16
  variant for tall-skinny shapes (M ≥ 4N); long dot products are
  renormalized every 128 terms.
- ZGEMM uses Gauss's three-multiplication form: three real DD GEMMs
  instead of four.

## Accuracy and testing

DGEMM's normwise relative error grows like √N·10⁻¹⁵ on random matrices,
about 5 × 10⁻¹⁴ at N = 4096 (V-2 convergence study). `make test` checks
precision against a compensated FP64 reference, correctness of every API
against Accelerate, the device-buffer API, the routing rules, the BLAS-ABI
entry points in a conformance harness (with long-K precision,
determinism, concurrency, memory and fork checks), and the per-call trace
and forced GPU failures; it also holds GPU-routed BLAS results to the bits
1.4.0 produced. The FFT is checked against a double-double reference and,
bit for bit, against a CPU replay of its double-float arithmetic. `make ci-local` adds adversarial inputs and the
convergence study; [tests/README.md](tests/README.md) gives the test counts.

[V&V report](docs/vv/VV_REPORT.md) · [precision envelope](docs/vv/PRECISION_ENVELOPE.md) ·
[limitations and scope](docs/vv/LIMITATIONS.md) ·
[Quantum ESPRESSO validation](tests/validation/VAL001_QE_Si64.md)

## Requirements

- Apple Silicon (M1 or later) with macOS 14 or later
- Xcode or the Command Line Tools (clang); gfortran only for the Fortran tests
- Measured on an M2 Max; other chips are expected to work but have no
  calibration or validation yet

## Limitations

- **Not IEEE FP64.** About 48 significant bits instead of 53, FP32's
  exponent range (about 10⁻³⁸ to 10³⁸) instead of FP64's, and no
  bit-for-bit agreement with CPU BLAS.
- **Per-call overhead.** Through the BLAS ABI every measured shape is
  slower than Accelerate except ZGEMM with both operands transposed at
  N ≥ 4096; see the table above.
- **Threads.** Calls on different matrices may come from different
  threads; one matrix, pool or session must not be used by two threads at
  once. All work shares one Metal command queue.
- **Size.** rows × cols ≤ 2³¹ − 1 per matrix and within the GPU's
  `maxBufferLength`; larger BLAS-ABI operands run on Accelerate.
- **ZHERK on the GPU is slower than Accelerate**; call `cblas_zherk`.
- **FFT:** about 20× FFTW's FP64 error, and one transform at a time is
  faster on eight CPU threads below 96³ (below 120³ through staging).

## Project layout

```
include/apple_bottom.h          public API (and apple_bottom_device.h, apple_bottom_fft.h)
src/apple_bottom.m              Metal context and DD kernels
src/fft.m, src/fft/             double-float FFT: plans, staging, kernel source
src/blas_wrapper.c              BLAS ABI: packing, conversion, fallbacks
src/dispatch.c, dispatch_table.h  AB_MODE routing and the calibrated table
src/fortran_bridge.c            dgemm_ / zgemm_ / ab_*gemm_ entry points
src/profiling/                  AB_PROFILE / AB_REPORT accounting, AB_TRACE
tests/                          test suites (tests/README.md)
benchmarks/                     benchmarks, calibration and their results
docs/                           integration guide, V&V package, design notes
python/                         experimental ctypes bindings
```

## Related projects

By the same author:
[Quantum-Espressivo](https://github.com/grantdh/Quantum-Espressivo) (Quantum ESPRESSO on Apple Silicon),
[YAMBOrghini](https://github.com/grantdh/YAMBOrghini) (Yambo GW/BSE on Apple Silicon),
[MEEPhistopheles](https://github.com/grantdh/MEEPhistopheles) (Metal FDTD kernels for MEEP, FP32) and
[rainbow-connection](https://github.com/grantdh/rainbow-connection) (QE → Yambo → MEEP pipelines).

Prior art: [metal-float64](https://github.com/philipturner/metal-float64)
set out to emulate IEEE FP64 in software on Metal (unfinished, archived);
apple-bottom gives up exact IEEE semantics to run on the FP32 units
directly.
[MLX](https://github.com/ml-explore/mlx) runs FP64 on the CPU only.
[docs/COMPARISON.md](docs/COMPARISON.md) has more.

## Contributing

Bug reports, calibration results from other chips (`make calibrate`) and
pull requests are welcome; see [CONTRIBUTING.md](CONTRIBUTING.md).

## Citation

```bibtex
@software{heileman2026applebottom,
  author    = {Heileman, Grant David},
  title     = {apple-bottom: FP64-class BLAS for Apple Silicon GPU},
  year      = {2026},
  url       = {https://github.com/grantdh/apple-bottom},
  version   = {1.4.0}
}
```

## License

MIT — see [LICENSE](LICENSE).
