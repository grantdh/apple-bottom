# Integration Guide

Complete guide for integrating apple-bottom into scientific computing codes.

---

## Table of Contents

- [Quick Decision](#quick-decision-should-you-use-apple-bottom)
- [C Integration](#c-integration)
- [Python Integration](#python-integration)
- [Fortran Integration](#fortran-integration)
- [Quantum ESPRESSO Integration](#quantum-espresso-integration)
- [Environment Variables and Run Reports](#environment-variables-and-run-reports)

---

## Quick Decision: Should You Use apple-bottom?

**Yes**, if:
- You multiply large matrices that can stay on the GPU between calls (the
  native `ABMatrix` API). On an M2 Max the GPU overtakes Accelerate from
  about N = 1,500 (DGEMM) and N = 1,000 (ZGEMM) with the matrices resident
  ([README](../README.md#performance)).
- Results with about 48 significant bits are enough: each value carries
  ~10⁻¹⁵ relative error, and a GEMM's normwise error grows like √N·10⁻¹⁵
  (about 5 × 10⁻¹⁴ at N = 4,096).

**Not yet**, if:
- Your code calls `dgemm_`/`zgemm_` (or the device-buffer API, which also
  converts on every call) on matrices in host memory, one call at a time.
  Each call then converts FP64 to DD and back, and on an M2 Max Accelerate
  is faster for nearly every shape measured. `AB_MODE=auto` keeps
  such calls on Accelerate, so linking apple-bottom costs little but gains
  nothing there yet.
- You need IEEE FP64 results bit for bit, or code that runs off Apple
  Silicon.

---

## C Integration

### Building your program

With the library built in place (`make`), the shared library carries its
Metal, Foundation and C++ dependencies:

```bash
clang -I/path/to/apple-bottom/include my_prog.c -o my_prog \
      -L/path/to/apple-bottom/build -Wl,-rpath,/path/to/apple-bottom/build -lapplebottom
```

Linking the static archive instead needs those dependencies spelled out:

```bash
clang -I/path/to/apple-bottom/include my_prog.c -o my_prog \
      /path/to/apple-bottom/build/libapplebottom.a \
      -lc++ -framework Metal -framework Foundation -framework Accelerate
```

After `make install PREFIX=$HOME/.local` (default `/usr/local`), pkg-config
knows the flags:
`clang my_prog.c $(pkg-config --cflags --libs applebottom)`, with
`PKG_CONFIG_PATH=$HOME/.local/lib/pkgconfig` for a non-default prefix.

### Storage order

`ab_matrix_upload` / `ab_matrix_download` take **row-major** (C order) data:
element (i, j) of an M × N matrix is `data[i * N + j]`. The BLAS ABI
(`ab_dgemm_blas`, `dgemm_`, …) and the device-buffer API are column-major,
as in BLAS. Column-major arrays uploaded into `ABMatrix` handles are read
transposed, and `ab_dgemm` then returns B·A instead of A·B, with no error;
use the BLAS ABI for column-major data.

### Values far from 1

Double-float values are pairs of FP32 numbers, so on their own they have
FP32's exponent range: products below about 2⁻⁷⁸ lose their low part,
values below 2⁻¹²⁶ flush to zero on the GPU, and sums overflow at 2¹²⁸.
The library bridges that with exact powers of two, which leave the bits of
in-range data unchanged:

- **BLAS ABI** (`dgemm_`, `zgemm_`, `ab_*gemm_blas`).
  - A row of op(A) or column of op(B) whose largest entry lies outside
    2^±20 is scaled on its way to the GPU, by the power of two that brings
    that entry to about 2³⁰, and C back.
  - A DGEMM alpha or beta outside 2^±20 joins C's scale.
  - A call the scaled data still cannot keep in range runs on Accelerate.
    So does a call with NaN or Inf in A or B (or in C, for DGEMM with
    beta ≠ 0), which then gets IEEE results. `AB_REPORT` counts such calls
    as `range` fallbacks.
- **Native `ABMatrix` API.**
  - `ab_matrix_upload` gives each matrix one power-of-two scale, and
    `ab_matrix_download` removes it; the GPU operations carry it through.
  - Data such as 1e-25 × uniform or 1e25 × uniform therefore multiplies
    correctly.
  - A single matrix whose nonzero entries span more than about 2⁷⁸ (10²³)
    keeps FP32's range relative to its scale, and has no fallback.
- **Device-buffer API** (`ab_dev_dgemm`, `ab_dev_zgemm`): converts through
  `ABMatrix` handles, so it behaves as the native API does.

The thresholds and their measurements are in
[LIMITATIONS.md](vv/LIMITATIONS.md#exponent-range).

### Complex data with imbalanced parts

ZGEMM uses Gauss's three-product form, whose imaginary part loses accuracy
when the real and imaginary parts differ greatly in size (nearly real or
nearly imaginary data). Such calls run as four real products instead, at
1.00–1.10× the time through `zgemm_`. The BLAS ABI and the device API decide
per row of op(A) and column of op(B); the native API decides per matrix, from
the magnitudes `ab_matrix_upload` measured, and keeps the three-product form
for matrices computed on the GPU. Details: [LIMITATIONS.md](vv/LIMITATIONS.md#complex-products).

### Where the GPU pays off

```
BLAS ABI, one call at a time (dgemm_, ab_dgemm_blas):
  [copy into reused GPU buffers] → [GPU: FP64→DD, GEMM, DD→FP64, alpha/beta] → [copy into C]
  every call (1.4.0: host pack and FP64→DD, upload, GEMM, DD→FP64, host epilogue)

Native API, matrices resident (ABMatrix):
  [FP64→DD, upload] → [GPU GEMM] → [GPU GEMM] → … → [download, DD→FP64]
        once                many products                   once
```

With the 1.4.0 BLAS ABI only the second pattern beat Accelerate, and only
for large products: on an M2 Max from about N = 1,500 for DGEMM and
N = 1,000 for ZGEMM ([README](../README.md#performance)). Uploading an
N × N operand costs O(N²) against the product's O(N³), so uploading one new
operand per product is fine at those sizes; converting and transferring
everything for every call on the host is not. The staged path of the BLAS
ABI leaves the host a plain copy of each operand into GPU buffers it
reuses (about 50-100 GB/s; none for memory registered with
`ab_register_host_memory`, see [Zero-copy operands](#zero-copy-operands))
and converts on the GPU. It has not been calibrated for `AB_MODE=auto`
yet: the routing table is still 1.4.0's.

### Pattern 1: a chain of large products

```c
// P = M[0] · M[1] · … · M[k-1], all N × N, row-major; P stays on the GPU.
ab_init();                                   // once, at program start
ABMatrix P = ab_matrix_create(N, N);
ABMatrix M = ab_matrix_create(N, N);
ABMatrix T = ab_matrix_create(N, N);
ab_matrix_upload(P, mats[0], true);
for (int i = 1; i < k; i++) {
    ab_matrix_upload(M, mats[i], true);      // O(N^2) upload ...
    ab_dgemm(P, M, T);                       // ... for an O(N^3) product
    ABMatrix t = P; P = T; T = t;            // swap handles, no copy
}
ab_matrix_download(P, result, true);
ab_matrix_destroy(P); ab_matrix_destroy(M); ab_matrix_destroy(T);
ab_shutdown();                               // once, at program end
```

### Pattern 2: iterative codes with named matrices

Sessions keep named matrices for the life of a computation, and pools hand
out matrices of a given shape without a new allocation each iteration:

```c
ABSession s = ab_session_create();
ab_session_add(s, "A", N, N);
ab_session_add(s, "B", N, N);
ab_session_add(s, "C", N, N);
ab_session_upload(s, "A", A_host);            // constant across iterations
for (int it = 0; it < n_iter; it++) {
    ab_session_upload(s, "B", B_host[it]);
    ab_session_dgemm(s, "A", "B", "C");       // C = A · B
    ab_session_download(s, "C", C_host[it]);
}
ab_session_destroy(s);
```

`ab_pool_create` / `ab_pool_get_matrix` / `ab_pool_reset` do the same for
anonymous temporaries.

### Pattern 3: several products in one submission

`ab_batch_create`, `ab_batch_dgemm` (or `ab_batch_zgemm`),
`ab_batch_barrier` between dependent products, `ab_batch_commit` and
`ab_batch_wait` encode many products into one command buffer, which saves
the per-submission latency. Each product still has to be large enough to
beat Accelerate on its own.

### Performance Tips

#### 1. Minimize Transfers

```c
// BAD: Upload/download every iteration
for (int i = 0; i < 100; i++) {
    ab_matrix_upload(mA, A, true);   // 100 uploads!
    ab_dgemm(mA, mB, mC);
    ab_matrix_download(mC, C, true); // 100 downloads!
}

// GOOD: Upload once, download once
ab_matrix_upload(mA, A, true);       // 1 upload
for (int i = 0; i < 100; i++) {
    ab_dgemm(mA, mB, mC);            // Pure GPU
}
ab_matrix_download(mC, C, true);     // 1 download
```

#### 2. Reuse Matrix Handles

```c
// BAD: Create/destroy per operation
for (int i = 0; i < 100; i++) {
    ABMatrix m = ab_matrix_create(N, N);  // 100 allocations!
    ab_dgemm(mA, mB, m);
    ab_matrix_destroy(m);                  // 100 frees!
}

// GOOD: Reuse handles
ABMatrix m = ab_matrix_create(N, N);      // 1 allocation
for (int i = 0; i < 100; i++) {
    ab_dgemm(mA, mB, m);
}
ab_matrix_destroy(m);                      // 1 free
```

#### 3. Pass `parallel = true` to uploads and downloads

The library converts serially below about 10⁶ elements and in eight
parallel chunks above, so `true` is never slower:

```c
ab_matrix_upload(m, data, true);
```

#### 4. Let measurement decide the route

For calls with host arrays, `ab_dgemm_blas` / `ab_zgemm_blas` route each
call from the calibrated table (`AB_MODE=auto`), so a fixed size cut-off in
your code is not needed. `make calibrate` measures the table on your own
machine, and `AB_DISPATCH_TABLE=<file>` uses it without a rebuild
([Routing](#routing-ab_modeauto)).

### Troubleshooting

**"No Metal GPU available"**
- Check you're on Apple Silicon (M1/M2/M3/M4)
- Run `system_profiler SPDisplaysDataType` to verify GPU

**Slow performance**
1. Check matrix sizes: on an M2 Max the GPU is slower than Accelerate
   below about N = 1,500 (DGEMM) or N = 1,000 (ZGEMM) even with the
   matrices resident ([README](../README.md#performance))
2. Ensure you're not uploading/downloading every iteration
3. Use `ab_print_stats()` to see time breakdown; for BLAS-ABI calls,
   `AB_REPORT=1`

**Precision issues**
- Double-float gives ~10⁻¹⁵ precision, not 10⁻¹⁶
- For ~10⁻¹⁶, use IEEE FP64 on the CPU (Accelerate)

---

## Python Integration

The experimental bindings in [`python/`](../python/) wrap the C API for
numpy:

```bash
make                        # builds build/libapplebottom.dylib
pip install -e python/      # finds ../build/libapplebottom.dylib; or set AB_LIBRARY
```

```python
import numpy as np
import apple_bottom as ab

ab.init()
C = ab.dgemm(A, B)          # one call: allocate, upload, multiply, download

s = ab.Session()            # resident matrices for repeated products
s.add("A", n, n); s.add("B", n, n); s.add("C", n, n)
s.upload("A", A); s.upload("B", B)
s.dgemm("A", "B", "C")
C = s.download("C")
```

`ab.dgemm` converts and transfers every operand on every call, the regime
in which Accelerate is faster; use a `Session` (or `Matrix` handles) when
the same matrices take part in several large products. Details:
[`python/README.md`](../python/README.md).

---

## Fortran Integration

### Overview

libapplebottom exports the Fortran BLAS symbols `dgemm_` and `zgemm_`, and
`ab_dgemm_` / `ab_zgemm_` with the same arguments. Link it ahead of your
other BLAS library and the program's existing `CALL DGEMM` and `CALL ZGEMM`
reach apple-bottom without a source change; every other BLAS and LAPACK
routine still comes from that other library. Each call is then routed to
the GPU or to Accelerate ([Automatic Routing](#automatic-routing)).

| Fortran call | Provided by apple-bottom |
|--------------|--------------------------|
| `CALL DGEMM(...)` | yes (`dgemm_`; also `CALL ab_dgemm(...)`) |
| `CALL ZGEMM(...)` | yes (`zgemm_`; also `CALL ab_zgemm(...)`) |
| any other BLAS or LAPACK routine | no: your BLAS/LAPACK library |

The arguments are those of reference BLAS, and so is the behaviour: TRANS
is case-insensitive (`'N'`, `'T'`, `'C'`; `'C'` means `'T'` for DGEMM), an
illegal argument is reported through `xerbla` and stops the program, and
`alpha = 0` returns `C = beta*C` without reading A or B.

To send only some call sites to apple-bottom, call `ab_zgemm` / `ab_dgemm`
there, declared `EXTERNAL` (not through a module, so array sections pass by
address as with any BLAS call), and leave `-lapplebottom` after your BLAS
library:

```fortran
EXTERNAL :: ab_zgemm
CALL ab_zgemm('C', 'N', nvec, nvec, ndim, alpha, psi, ndim, psi, ndim, beta, s, nvec)
```

### Build Integration

Shared library (recommended; it carries its own Metal and C++ dependencies
and re-exports Accelerate, so with no other BLAS library `-lapplebottom`
also provides the rest of BLAS and LAPACK):

```makefile
AB_LIBS = -L/path/to/apple-bottom/build -Wl,-rpath,/path/to/apple-bottom/build -lapplebottom
LIBS    = $(AB_LIBS) $(OTHER_BLAS_LIBS)
```

Static archive:

```makefile
LIBS = /path/to/apple-bottom/build/libapplebottom.a \
       -lc++ -framework Metal -framework Foundation -framework Accelerate
```

Keep apple-bottom first. The linker binds `dgemm_`/`zgemm_` to the first
library on the command line that exports them, and in a flat namespace
(`-Wl,-flat_namespace`, which Homebrew's Open MPI wrappers add) the first
loaded library wins at run time. Without `-lc++`, a static link fails with
an undefined `___gxx_personality_v0`.

### Check that the calls arrive

```bash
AB_REPORT=1 ./my_app 2> report.txt
grep AB_REPORT report.txt
```

There is one `AB_REPORT[pid] zgemm calls=…` line per routine that reached
apple-bottom, with the calls and time by route. No `AB_REPORT` lines: the
program does not load libapplebottom (`otool -L my_app`). No line for a
routine the program calls: that symbol is bound to another library; put
apple-bottom first.

### Automatic Routing

`dgemm_`, `zgemm_`, `ab_dgemm_` and `ab_zgemm_` forward each call to
`ab_dgemm_blas` / `ab_zgemm_blas`, which route it. By default
(`AB_MODE=auto`) a call goes to the GPU only for shapes where the GPU was
measured at least 1.10× faster than Accelerate end to end; everything else
runs on Accelerate. `AB_MODE=gpu` sends every call with all dimensions above
`AB_MIN_GPU_DIM` to the GPU, `AB_MODE=cpu` none. See
[Routing](#routing-ab_modeauto).

---

## Quantum ESPRESSO Integration

### Overview

Quantum ESPRESSO calls ZGEMM and DGEMM through the Fortran BLAS ABI: the
Davidson solvers (`cegterg` for k-points, `regterg` at the gamma point) and
the PAW routines (`newq`, `addusdens`), among others. Linking apple-bottom
ahead of the configured BLAS is all it takes; no QE source change is
needed.

**What to expect on an M2 Max** ([VAL-1](../tests/validation/VAL001_QE_Si64.md),
QE develop (7.5), 64-atom Si, PAW): the total energy and every printed band energy
are the same in all three modes. `AB_MODE=auto` keeps every GEMM of these
runs on Accelerate, because the calibrated table has no GPU row for QE's
shapes, so its run time is that of `AB_MODE=cpu` (within 0.2%).
`AB_MODE=gpu` forces the GEMMs onto the GPU and is slower:
1.3–17.3% more wall time on the three inputs with QE itself compiled at
`-O0`, because its GEMMs take 2.5–4.9× as long. With QE compiled `-O3` it
is 14–40% slower with one rank and 5–21% slower with 8 ranks
([release check](../benchmarks/results/2026-10-15-c5a1fa6-qe-si64-opt/)). On this hardware, linking apple-bottom into QE gives you
the `AB_REPORT` accounting and a routing that follows measurement, not a
speed-up.

### make.inc

QE's `QELIBS` lists `$(LAPACK_LIBS)` before `$(BLAS_LIBS)`, so put
apple-bottom at the front of both:

```makefile
AB_LIBS     = -L/path/to/apple-bottom/build -Wl,-rpath,/path/to/apple-bottom/build -lapplebottom
LAPACK_LIBS := $(AB_LIBS) $(LAPACK_LIBS)
BLAS_LIBS   := $(AB_LIBS) $(BLAS_LIBS)
```

Then `make pw` as usual.

### Run and read the report

```bash
AB_REPORT=1 mpirun -np 1 -x AB_REPORT -x AB_MODE pw.x -in si64.in > si64.out 2> si64.err
grep AB_REPORT si64.err
```

Each rank prints its own report at exit (calls, time on the GPU and on
Accelerate, the GPU calls' stage costs, fallbacks); see
[Reading a run report](#reading-a-run-report). With `AB_MODE=auto` on an M2
Max, expect `gpu_calls=0` for every routine.

### Troubleshooting

**No `AB_REPORT` lines.** pw.x does not load libapplebottom: check
`otool -L bin/pw.x` and the rpath.

**A `zgemm` / `dgemm` line is missing.** The symbol is bound to another
library. Put `$(AB_LIBS)` first in both `LAPACK_LIBS` and `BLAS_LIBS`. With
a flat-namespace link, `DYLD_PRINT_BINDINGS=1 bin/pw.x -in /dev/null 2>&1 |
grep -E '/_[dz]gemm_>'` shows where each one went.

**Energies differ between `AB_MODE=cpu` and `AB_MODE=gpu` by more than
`conv_thr`.** That is a bug; please open an issue with the input and both
`AB_REPORT` outputs.

---

## Environment Variables and Run Reports

The BLAS-ABI entry points (`dgemm_`, `zgemm_`, `ab_dgemm_`, `ab_zgemm_`,
`ab_dgemm_blas`, `ab_zgemm_blas`) read these once per process, at their first
call. The native `ABMatrix` and device APIs do not use them, except
`AB_PREFETCH`, which the library reads when it initializes and which applies
to every API.

| Variable | Default | Effect |
|----------|---------|--------|
| `AB_MODE` | `auto` | `cpu`: every call on Accelerate. `gpu`: on the GPU unless a dimension is ≤ `AB_MIN_GPU_DIM`. `auto` (or any other value): by the calibrated dispatch table, or by a FLOP threshold if one is set (see [Routing](#routing-ab_modeauto)). |
| `AB_MIN_GPU_DIM` | `32` | Calls with M, N or K ≤ this run on Accelerate in every mode. |
| `AB_DISPATCH_TABLE` | unset | `auto`: read the dispatch table from this file (the `dispatch_table.txt` that `make calibrate` writes) instead of using the built-in one. A file that cannot be read or parsed is reported on stderr and the built-in table is used. |
| `AB_CROSSOVER_FLOPS` | unset | `auto`: replaces the table by a FLOP threshold. Calls with at least this many FLOPs (2·M·N·K real, 8·M·N·K complex) go to the GPU: ZGEMM, and DGEMM unless `AB_CROSSOVER_FLOPS_REAL` is set. |
| `AB_CROSSOVER_FLOPS_REAL` | unset | `auto`: the same for DGEMM only. |
| `AB_PREFETCH` | unset | Which GPU GEMM kernels load the next K-tile into registers while they compute the current one (register prefetch). Unset or any other value: where the GPU holds few threadgroups, by `use_prefetch` in `src/apple_bottom.m` (on an M2 Max up to 1.43× faster there). `0`: never. `1`: on every grid. The results are bitwise the same in all three; `0` and `1` are for timing comparisons. |
| `AB_REPORT` | off | `1`: a short summary on stderr at exit (below). |
| `AB_PROFILE` | off | `1`: the same counts as a detailed table on stderr at exit, with a dimension histogram, plus one line per GPU call and per fallback. |
| `AB_PROFILE_JSON` | unset | With `AB_PROFILE=1`: also write the profile as JSON to this path at exit. |
| `AB_PROFILE_THRESHOLD_NS` | `0` | Count only calls that took at least this many ns, in both reports. |
| `AB_TRACE` | unset | A path: each process appends one line per call to `<path>.<pid>` (see [Per-call trace](#per-call-trace-ab_trace)). Unset or empty: no trace. |
| `AB_TRACE_HASH` | off | With `AB_TRACE`, `1`: also hash the operands each call reads. |
| `AB_HOST_PACK` | off | `1`: every GPU call takes the host path (host pack, FP64 → DD and epilogue) instead of the staged path. The results are the same bits either way. |
| `AB_STAGING_MB` | `2048` | The GPU buffers kept between calls by the staged path and by the native ZGEMMs (their Gauss-3M or 4M temporaries), at most this many MB per process; `0`: none kept, each call makes its own (and pays for fresh buffers). |
| `AB_TEST_FAIL_STAGE` | unset | Test-only. `alloc`: every GPU buffer allocation (`ab_matrix_create()`, and the buffer pool's) fails; `kernel`: the command buffer of every `ab_dgemm_scaled()`, `ab_zgemm()` and staged call, the GEMMs of the GPU path, reports an error. A GPU-routed call then fails as it would on a real failure there and runs on Accelerate instead (a `gpu_failed` fallback). Never set it in production. |

### Routing (AB_MODE=auto)

One function (`src/dispatch.c`) decides where each BLAS-ABI call runs, in
this order:

1. Invalid arguments and `alpha == 0`: Accelerate (reference BLAS checks and
   quick return), in every mode.
2. `AB_MODE=cpu`: Accelerate.
3. M, N or K ≤ `AB_MIN_GPU_DIM`: Accelerate, in every mode.
4. `AB_MODE=gpu`: the GPU.
5. `auto` with `AB_CROSSOVER_FLOPS_REAL` (DGEMM) or `AB_CROSSOVER_FLOPS`
   (ZGEMM, and DGEMM without `AB_CROSSOVER_FLOPS_REAL`) set: the GPU if the
   call has at least that many FLOPs.
6. `auto`: the dispatch table, from the file `AB_DISPATCH_TABLE` names or
   else the built-in `src/dispatch_table.h`. The GPU if a row of the call's
   routine and TRANS class has min_M ≤ M, min_N ≤ N and min_K ≤ K, else
   Accelerate.

The table comes from measurements, not a model. `make calibrate` (about 15
minutes) times each shape of a grid end to end, through `ab_*gemm_blas`
with the GPU path forced and through Accelerate `cblas_*gemm` in the same
process: the Quantum ESPRESSO si64 PAW DGEMMs and Davidson ZGEMMs, and
square and transposed-square DGEMM and ZGEMM from 256 to 6144, with 3
warmups, the median of 7 runs, and GPU results checked against cblas. A
shape becomes a row if the GPU was at least 1.10× faster and no slower
shape of its routine and TRANS class is at least as large in M, N and K.
A table file has one row per line; `'T'` also covers `'C'`, and a file
without rows keeps every `auto` call on Accelerate:

```
# routine transA transB min_M min_N min_K
zgemm T T 4096 4096 4096
```

The built-in table was measured on an M2 Max (38-core GPU, macOS 27.0.1;
data in `benchmarks/results/2026-10-14-52f1880-calibrate/`). The GPU path
was at least 1.10× faster only for ZGEMM with both operands transposed or
conjugated, from 4096 up (1.11× at 4096, 1.20× at 6144): that is the
table's one row. Every QE si64 shape (0.09–0.76× Accelerate's speed) and
every DGEMM (at most 0.84×) was slower on the GPU, and the other ZGEMMs
reached at most 1.02× (6144, one operand transposed). So `auto` keeps QE's
calls on Accelerate. To
calibrate another machine, run `make calibrate` and set
`AB_DISPATCH_TABLE` to the `dispatch_table.txt` it writes, or rebuild:
it also rewrites `src/dispatch_table.h`.

### Zero-copy operands

`ab_register_host_memory(ptr, nbytes)` makes the GPU read an operand of a
BLAS-ABI call in place when it lies wholly inside the registered range and
starts on a multiple of its element size (8 bytes for DGEMM, 16 for
ZGEMM), and write C there when it does not read it (DGEMM with beta = 0,
ZGEMM with alpha = 1 and beta = 0); other operands are copied as usual. Registering
maps the pages for the GPU, about 10 ms per GB, so it pays for arrays that
many calls use, such as a solver's work arrays:

```c
double complex *psi = malloc(bytes);
ab_register_host_memory(psi, bytes);
for (...) zgemm_(..., psi, &lda, ...);     // psi is read in place
ab_unregister_host_memory(psi);            // before free(psi)
free(psi);
```

The library does not copy, retain or watch the memory: keep it allocated
at the same address and size until `ab_unregister_host_memory()` has
returned, and do not unregister it while a call that uses it runs. Memory
freed or reallocated while registered leaves the GPU reading its old pages,
and the results are wrong without an error. Up to
`AB_MAX_HOST_REGISTRATIONS` (64) ranges, which must not overlap;
`ab_shutdown()` drops them all. The results are the same bits as without
registration.

### Reading a run report

`AB_REPORT=1` prints, when the process exits:

```
AB_REPORT[<pid>] apple-bottom <version> (<commit>[-dirty]) AB_MODE=<mode|unset>
AB_REPORT[<pid>] <routine> calls=… gpu_calls=… cpu_calls=… gpu_calls%=… total_ms=… gpu_ms=… cpu_ms=… gpu_time%=… fallbacks=…
AB_REPORT[<pid>] <routine> gpu_stages_ms init=… alloc=… pack=… upload=… kernel=… download=… epilogue=… other=…
AB_REPORT[<pid>] <routine> fallbacks no_device=… too_large=… gpu_failed=… other=… range=…
AB_REPORT[<pid>] <routine> staged kept=… kept_4m=… host_pack=… fpcr=… scalar=… alloc=… gpu_failed=… nonfinite=… scaled=… range=…
```

- **The first line names the library build**, as `ab_build_info()` returns it:
  the version and the git commit it was built from, `-dirty` if tracked
  files differed from that commit, `unknown` if it was built outside a git
  checkout. Keep it with any timing you quote. The `AB_PROFILE` table starts
  with the same string.
- **`gpu_calls%` is a share of calls, `gpu_time%` a share of time**: the
  routine's BLAS wall time spent in calls that ran on the GPU. They can
  differ widely, e.g. many small calls on Accelerate and a few large ones on
  the GPU. A call counts as GPU only if the GPU computed C.
- **The stage line** splits the GPU calls' `gpu_ms`: `init` (routing, size
  checks, and creating the Metal context on the first GPU call), `alloc`
  (GPU buffers and host scratch, allocation and release), `pack` (copying
  column-major op(A), op(B) and C to row-major, and the exponent-range
  check), `upload` (FP64 → DD, with the power-of-two scaling),
  `kernel` (command-buffer GPU execution time; wall time if Metal reports
  none), `download` (DD → FP64), `epilogue` (the row-major result into
  column-major C, with alpha and beta for ZGEMM) and `other`, the rest:
  mostly command-buffer encode, submit and wait latency, and ZGEMM's GPU
  temporaries. `AB_TRACE` splits it per call (below). That is the host
  path (`AB_HOST_PACK=1`). On the staged path, the default, `alloc` is the
  pool's buffers, `pack` the exponent-range check (on statistics the GPU
  gathers), `upload` the copies of the operands into the GPU buffers,
  `kernel` includes the GPU's FP64 ↔ DD conversion, scaling and epilogue
  (and, for a ZGEMM whose parts call for 4M, a second command buffer that
  forms 4M's imaginary part), `download` is the copy of the result into C,
  and `epilogue` only ZGEMM's host alpha and beta when they are not 1 and 0
  or 1. A call the staged path gives back to the host path (see the staged
  line below) adds both paths' stages.
- **Fallbacks** are calls the routing sent to the GPU that ran on Accelerate
  instead: `no_device` (no Metal context: no GPU, or a `fork()`ed child),
  `too_large` (an operand beyond the size limits), `gpu_failed` (an
  allocation, upload or kernel failed before C was written), `range` (NaN
  or Inf, or values that power-of-two scaling cannot keep in double-float's
  exponent range; see [Values far from 1](#values-far-from-1)). They count
  as CPU calls, and their `cpu_ms` includes the failed GPU attempt.
- **The staged line** counts the GPU-routed calls that reached the staged
  path: `kept` its result, `kept_4m` (a ZGEMM whose real and imaginary
  parts call for 4M, see [Complex data with imbalanced
  parts](#complex-data-with-imbalanced-parts), kept after the second
  command buffer), or else the first reason it gave the call to the host
  path, in the order it checks them: `host_pack` (`AB_HOST_PACK=1`),
  `fpcr` (the caller changed the floating-point control: rounding mode,
  flush to zero or default NaN), `scalar` (a DGEMM alpha or beta outside
  2^±20, or not finite), `alloc` (no GPU buffers), `gpu_failed` (its
  command buffer failed), `nonfinite` (Inf or NaN in the GPU epilogue),
  `scaled` (the GPU's power-of-two scales of the lines differ from the
  host path's: a defect, never expected) and `range` (the guard does not
  certify the call, which goes straight to Accelerate as a `range`
  fallback). Each call counts once, whether it then ran on the host path
  or on Accelerate.
- **Each process reports once**, tagged with its pid, so MPI ranks sharing
  stderr stay apart. A `fork()`ed child starts with empty counters and
  reports only its own calls (nothing if it made none). There is no report
  after `_exit()` or a crash. Calls on several threads add up their wall
  times.

`AB_PROFILE=1` prints the same counts as a table (`calls`, `gpu_calls`,
`gpu_calls%`, `total_ms`, `gpu_ms`, `cpu_ms`, `gpu_time%`, `fallbacks`),
tables of the GPU stages, the fallbacks and the staged path's outcomes, and
the dimension histogram of the routine with the most time. During the run it
prints an `AB_PROFILE <routine> M=… N=… K=…` line with the stages of each GPU
call and its staged outcome (`staged=kept`, or the first reason it took the
host path), and one for each fallback, with its reason.

### Per-call trace (AB_TRACE)

`AB_TRACE=<path>` records every BLAS-ABI call, on Accelerate or the GPU and
in every `AB_MODE`, so that a run can be replayed or its timeline lined up
with other processes'. Each process appends to its own file,
`<path>.<pid>`: a `fork()`ed child, or an MPI rank, gets a file of its own.
A file starts with `#` lines naming the build (`ab_build_info()`), the pid,
`AB_MODE`, `AB_MIN_GPU_DIM` as the routing resolved it, the MPI rank and size
the launcher exported (`OMPI_COMM_WORLD_RANK`/`_SIZE`, else
`PMI_RANK`/`PMI_SIZE`; `mpi_rank=-1 mpi_size=-1` outside MPI) and the field
legend. Each call then adds one line of `key=value` fields when it returns,
written with a single `writev(2)`:

```
seq=2 t0_ns=411972254072166 t1_ns=411972258538707 routine=zgemm ta=C tb=N M=96 N=80 K=72 lda=74 ldb=74 ldc=98 alpha=0x1.4p+0,-0x1p-1 beta=-0x1.8p-1,0x1p-2 path=gpu init_ns=1291 alloc_ns=53998 pack_ns=7291 upload_ns=24291 kernel_ns=577500 download_ns=10958 epilogue_ns=10958 other_ns=3780254 temps_ns=33083 encode_ns=42208 submit_ns=3560959 wake_ns=135916 rest_ns=8088 gpu_start_ns=411972257776500 gpu_end_ns=411972258354000
```

- **`seq`** numbers the process's calls from 1. **`t0_ns`, `t1_ns`** are the
  call's start and end in ns of `mach_absolute_time` (`CLOCK_UPTIME_RAW`),
  one clock for every process on the machine, so traces of different
  processes line up. `t1_ns - t0_ns` is the call's time as `AB_REPORT`
  counts it.
- **The arguments** as the call passed them: `ta`, `tb` (the TRANS
  characters, lower case kept), `M N K lda ldb ldc`, and `alpha`, `beta`
  exactly as C99 `%a` hex floats (`real,imaginary` for `zgemm`; Python reads
  them with `float.fromhex`).
- **`path`**: `cpu`, `gpu`, or `fallback:<reason>` for a call the routing
  sent to the GPU that ran on Accelerate (`no_device`, `too_large`,
  `gpu_failed`, `other`, `range`; see the run report).
- **GPU calls** add the stages of the run report in ns (`init_ns` …
  `epilogue_ns`, then `other_ns`), which sum exactly to `t1_ns - t0_ns`, and
  split `other_ns` exactly into `temps_ns` (taking ZGEMM's GPU temporaries
  from the buffer pool and handing them back on the host path, making any
  the pool lacks; 0 for DGEMM, which has none, and on the staged path, whose
  buffers are in `alloc_ns`), `encode_ns` (building the
  command buffer), `submit_ns` (from commit until the GPU started it:
  scheduling, and waiting behind other GPU work, including other
  processes'), `wake_ns` (from the GPU's end until `waitUntilCompleted`
  returned: the completion notice and the waiting thread's wake-up) and
  `rest_ns` (everything else; a few ns below zero is clock rounding).
  `gpu_start_ns` and `gpu_end_ns` are the command buffer's `GPUStartTime`
  and `GPUEndTime` on the same clock (0 if Metal reported none).
- **`staged`**, on a call that reached the staged path: `kept` or
  `kept_4m`, or the reasons of the run report's staged line that sent it
  to the host path, joined by `+` if more than one applied.
- **`AB_TRACE_HASH=1`** adds `hash_a` and `hash_b`, 64-bit hashes (16 hex
  digits) of op(A) and op(B) as stored, when the call reads them
  (M, N, K > 0 and alpha ≠ 0), and `hash_c` of C on entry when beta ≠ 0.
  Each is FNV-1a over the 64-bit words of the referenced elements, column
  by column: from the offset basis `0xcbf29ce484222325`,
  `h = (h ^ word) * 0x100000001b3` for each word, a complex element being
  two words (real, then imaginary). The padding of a leading dimension is
  not included, so equal operands hash equal whatever their `ld`. Hashing
  reads every operand once more; the call's times leave it out.

`benchmarks/replay.c` replays a run from its traces (see "Replay" in
`benchmarks/qe/README.md`).

Every call is traced, whatever `AB_PROFILE_THRESHOLD_NS` says. A call that
Accelerate rejects for an illegal argument ends the process before its line
is written. Unset, the trace costs a call one predictable branch.
A file that cannot be created is reported once on stderr, and that process
is not traced. A file that already exists (a pid reused since an earlier
run) is appended to, after a new header.

---

## Support

- Main documentation: [`README.md`](../README.md)
- V&V documentation: [`docs/vv/VV_REPORT.md`](vv/VV_REPORT.md)
- Issues: https://github.com/grantdh/apple-bottom/issues
