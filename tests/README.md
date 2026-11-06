# apple-bottom test suite

## Adding a new test to a registry-based file

The registry-based test runners (`test_correctness.c`, `test_chaos.c`,
`test_device_api.c`, `test_dispatch.c`, `test_trace.c`, `test_blas2.c`, `test_fft.c`,
`test_strassen.c`) dispatch via a
`TESTS[]` registry array of `{name, fn}` pairs declared just above
`main()`. To add a test:

1. Define the test as `static void test_my_thing(void)`. Use the
   file's existing accounting macros — `PASS()` / `FAIL("msg")` for
   the standard files, `CHECK(cond, msg)` for `test_device_api.c`.
   Every dispatched test must invoke at least one of these macros, or
   the per-test ghost-guard in `main()` will `abort()` the run.
2. Add an entry to the `TESTS[]` array: `{"test_my_thing", test_my_thing}`.
3. Recompile: `make clean && make test`.

For parameterized helpers, register a `void(void)` wrapper per argument
value — convention is `test_helper_<argvalue>` — or, as the GEMM tests in
`test_device_api.c` do, call a case-checking helper from each registered
test (`check_gemm()` runs one case and makes three `CHECK`s).

## What the per-test delta guard catches and what it does not

The guard wraps each `TESTS[i].fn()` call with a pre/post counter
snapshot. If the counter does not advance, the dispatcher names the
offending test and `abort()`s. This catches:

- A test that falls off the end without invoking PASS / FAIL / CHECK.
- A test that returns early before invoking the macro.

It does **not** catch:

- A test that segfaults — those are caught at OS level via the
  binary's non-zero exit code (CI flags via the same path).
- A test that calls `exit()` mid-run — bypasses the dispatcher
  entirely. Test bodies must not call `exit()`; `grep -n 'exit\s*('
  tests/test_correctness.c tests/test_chaos.c tests/test_device_api.c`
  must be empty.

## Coverage status

The test counts live here only (the README, `docs/RELEASING.md` and
`docs/REPRODUCIBILITY.md` point here): update them with any change of
count. Counting rule (as `make ci-local` counts): a *test* is one
`test_precision` case, one `TESTS[]` entry, or one BLAS-ABI suite; the
count of each file is listed below. `make test` runs 210, every test but
the chaos ones, and `make ci-local` all 228 (`TEST_TOTAL` in
`build/CI_REPORT.txt`). Device-API `CHECK`s and BLAS-ABI cases are
reported on their own `CI_REPORT.txt` lines (`DEVICE_API_CHECKS`,
`BLAS_ABI_CASES`) and are not part of the total, nor is `test_qe_golden`
(its `TEST_QE_GOLDEN` line says `PASS`, or `SKIP` on another device).

- **Registry-based (ghost-guarded):**
  - `tests/test_correctness.c` — 79 tests; `test_prefetch_same_bits`
    runs every register-prefetch kernel (`AB_PREFETCH=1`) against the
    plain ones (`AB_PREFETCH=0`) and requires bitwise-equal results
  - `tests/test_chaos.c` — 18 tests
  - `tests/test_device_api.c` — 19 tests (199 `CHECK`s); the GEMM
    tests compare `ab_dev_dgemm` / `ab_dev_zgemm` with column-major
    cblas over all trans pairs, non-square and tall-skinny shapes,
    `ld > rows`, buffer offsets and a C of 2²⁰ elements, and check that
    the device path is bitwise the host API on the transposed operands;
    `test_gemm_registered` runs them on `ab_dev_register` memory that is
    not page-aligned, bitwise against allocated buffers, between sentinels;
    `test_zgemm_imbalance` holds `ab_dev_zgemm_offset` to the Dot2 check
    of the BLAS ABI's `test_zgemm_imbalance` (below, `tests/test_dot2.h`)
    on both its alpha/beta paths, every way it gathers a line's parts, and
    operands of 2²⁰ elements, which it stages in chunks across cores;
    `test_gemm_pool` checks the GEMMs on the staging pool's reused
    buffers: a call repeated leaves the pool as it was, and each call
    gives its first bits again right after NaN data (which a beta = 0
    call must not read), after larger and smaller shapes, and from 4
    threads at once
  - `tests/test_dispatch.c` — 13 tests of the BLAS-ABI routing, no GPU,
    see below
  - `tests/test_blas_abi.c` — 33 suites (1073 cases), see below
  - `tests/test_trace.c` — 5 tests of `AB_TRACE` and
    `AB_TEST_FAIL_STAGE`, see below
  - `tests/test_blas2.c` — 19 tests (957 cases) of DGEMV, ZGEMV and
    ZHEMV, see below
  - `tests/test_fft.c` — 14 tests of the double-float FFT, see below
  - `tests/test_strassen.c` — 11 tests of the opt-in Strassen-Winograd
    products (`ab_dgemm_strassen`, `ab_zgemm_strassen`, `AB_STRASSEN`):
    error against a compensated Dot2 reference within the normwise bound of
    `docs/STRASSEN.md` over odd, tiny and rectangular shapes, alpha and
    beta, ZGEMM and badly scaled rows (whose componentwise errors it
    prints); a ZGEMM with imaginary parts 2⁻²⁴ of the real ones within the
    bound of its imaginary part's own size (4M); data far outside FP32's
    range bit for bit the unscaled result times the scales (DGEMM with alpha
    and beta rescaled, ZGEMM by Gauss-3M with aligned parts and by 4M with
    rescaled products, the unscaled ZGEMMs within the bound); argument checks; bitwise determinism across
    workspace reuse and `ab_shutdown()`; and, in child processes with their
    own environment, that `AB_STRASSEN` unset or 0 leaves every GEMM
    bitwise classical, that `AB_STRASSEN=1` routes calls with M, N, K ≥
    `AB_STRASSEN_MIN` bitwise to Strassen (native; through `dgemm_` /
    `zgemm_`, balanced and 4M, on the staged path and on the host path, each
    checked to have taken it; through `ab_dev_dgemm` / `ab_dev_zgemm`),
    that `dgemm_` / `zgemm_` with rows scaled line by line stay within the
    classical componentwise bound on both paths, and that ZHERK, a
    one-element ZGEMM batch, DTRSM and `ab_dgesv_mpir` give the same bits
    with `AB_STRASSEN=1` and every size routed as without it

- **Not yet registry-based (one `main`, no ghost guard):**
  - `tests/test_precision.c` — 17 tests (the last compares
    `AB_PRECISION=fast` with the default mode); needs decomposition strategy
  - `tests/test_qe_golden.c` — 32 golden-bits cases, see below
  - `tests/test_staging.c` — 127 cases of the staged path against the
    host path, bit for bit, see below (its `TEST_STAGING` line is not
    part of the total either)
  - `tests/verification/test_convergence.c` — monolithic numerical
    V&V file; decomposition needs care because behavior changes are
    load-bearing

## Targets outside `make test`

| Target | What it runs |
|---|---|
| `make ci-local` | `make version-check`, `make dispatch-table-check` (the built-in routing tables against `benchmarks/calibration/`), every suite of `make test`, the chaos tests and the V-2 convergence study (the DGEMM error-versus-N study of `docs/vv/VV_REPORT.md`, `tests/verification/test_convergence.c`); writes `build/CI_REPORT.txt` |
| `make test-chaos` | pathological inputs, IEEE edge cases, concurrent threads |
| `make test-verification` | the V-2 convergence study alone |
| `make test-rectangular` | `test_rectangular.c`: tall, wide, thin-K and conjugate-transposed shapes up to 18277 × 150 × 18277, componentwise against cblas, then timings (peaks at ~21 GB) |
| `make test-rectangular-diag` | eight rectangular patterns against cblas |
| `make test-dd-bound` | `test_dd_fma_bound.cpp`: cuts the shader's `DD` primitives out of `src/apple_bottom.m`, compiles them on the CPU without contraction, and checks `dd_fma` on 5×10⁶ adversarial and cancelling triples against MPFR and the bound of `docs/vv/PRECISION_ENVELOPE.md` §1.2 (`DD_BOUND_N=` to change the count; needs `brew install mpfr`, else SKIP; ~5 s) |
| `make test-python` | the Python bindings (`tests/python/`) |
| `make test-replay` | `tests/test_replay.py`: the QE call-mix replay (`benchmarks/replay.c`) on a synthetic 2-rank trace, in every mode, with the trace header's `AB_MIN_GPU_DIM`, a killed and a hung worker, and `benchmarks/replay_summary.py` |
| `make test-asan`, `make test-ubsan` | the library and `test_correctness` under AddressSanitizer / UndefinedBehaviorSanitizer, in `build/asan`, `build/ubsan`. On macOS 27.0.1 with Apple clang 17.0.0 the AddressSanitizer runtime hangs in its own initialization, before `main()`, even for an empty program; `test-ubsan` runs |

## BLAS-ABI conformance harness

`tests/test_blas_abi.c` checks `ab_dgemm_blas` / `ab_zgemm_blas`,
`ab_dgemm_` / `ab_zgemm_`, the plain `dgemm_` / `zgemm_` symbols and the
native GEMMs against Accelerate `cblas_*gemm` (column-major), using a
componentwise bound of 1e-11, and checks that `ab_get_last_dispatch_path()`
names what computed C. Each `TESTS[]` entry is a suite that runs in
its own child process (`AB_MODE` is cached per process; the `*_noinit`
suites never call `ab_init()`), and a crashing case fails only itself.
`test_illegal_args` passes each illegal argument (TRANS flag, negative
M/N/K, short lda/ldb/ldc) through every entry point; the case passes only
if cblas rejects it as reference xerbla would, naming the parameter and
ending the child with status 255. `test_fork_after_gpu` forks after a GPU
call; the child must run the same call on cblas, still pass, and be
profiled as a CPU call. `test_qe_shapes` runs the Quantum ESPRESSO PAW
DGEMM shapes that the old 46340-per-dimension limit kept on cblas (see
`gen_qe`), plus a synthetic ZGEMM with K = 100000, through `dgemm_` /
`zgemm_` without `ab_init()`; each must run on the GPU. It checks
robustness only: those DGEMMs are faster on Accelerate (the GPU ran them
at 0.09–0.15× in `make calibrate`).
`test_auto_mode` runs `AB_MODE=auto` with this chip's built-in table (none
on an uncalibrated chip, where the shapes stay on Accelerate either way):
the QE cegterg-like call and QE si64 PAW and Davidson shapes, which the GPU
path was measured slower on, must stay on Accelerate through every entry
point. `test_profile_path` runs `auto` with a test `AB_DISPATCH_TABLE`
(every `'N','N'` call on the GPU), so the table file is exercised end to end.
`test_exit_reports` runs `test_blas_abi --report-child` with `AB_REPORT=1`
and `AB_PROFILE=1`: three GPU and three CPU calls through the six BLAS-ABI
entry points, then a `fork()`ed child repeats a GPU call, which falls back
to cblas. Both processes' exit reports must start with `ab_build_info()`
(naming the Makefile's `AB_BUILD_SHA`) and give the calls made by route,
`gpu_time%` within [0, 100], GPU stages that sum to at most `gpu_ms`, one
`AB_PROFILE` line per GPU call, and the child only its own fallback.

Five suites guard what a faster GPU path must keep, all through `dgemm_` /
`zgemm_` in `AB_MODE=gpu` without `ab_init()`; the long-K, concurrency and
memory suites print their measured values on every run:
- `test_long_k_precision`: dgemm ('T','N' and 'N','N', both on the
  32 × 32 kernel) and zgemm ('C','N') with small M, N and
  K up to 143666 (QE's 2*ngm_l per rank), against a compensated Dot2
  reference (`test_dot2.h`, shared with `test_precision.c`). The error is
  max |C − R| / (|A||B|), componentwise, in units of the DD unit roundoff
  u² = 2⁻⁴⁸; for these random-sign inputs it does not grow with K, and must
  stay within `LONG_K_BOUND`. `gen_long_k` derives the bound and records
  what 1.4.0 measures.
- `test_determinism`: a call repeated from the same C0, and again after GPU
  calls of other shapes and routines, gives memcmp-equal C.
- `test_concurrency`: 8 threads, released together, each make 3 dgemm_ and
  3 zgemm_ calls on their own data: first each with its own shapes, before
  the Metal context exists, then all with the same shapes, so that state
  kept per shape is shared between threads. Every result must be bitwise
  the same call made single-threaded afterwards, and that within 1e-11 of
  cblas.
- `test_memory_growth`: GPU calls after a warm-up may grow the bytes
  malloc has handed out by less than `MEM_MALLOC_LIMIT_KB` over the first
  or the second half of the calls (a leak grows both), which catches small
  leaks, and `phys_footprint` (`task_info`) by less than
  `MEM_FOOTPRINT_LIMIT_KB`, which catches a leaked operand or GPU buffer.
  The child runs with `MallocNanoZone=0`, so that malloc's statistics count
  only live blocks. `gen_memory` records what 1.4.0 measures, what planted
  leaks cost, and how the limits follow.
- `test_fork_exit`: as `test_fork_after_gpu` without `AB_PROFILE`, but the
  forked child ends with `exit()` (atexit handlers and static destructors
  run, as at a Fortran `STOP`), not `_exit()`; it must exit with status 0
  within 30 s (an alarm ends it then, and the case fails).

`test_exponent_range` gives the GEMMs data outside FP32's exponent range,
which double-float shares. Each case starts from magnitudes in [0.5, 1)
with random signs, so the pattern alone sets the span.
- **Through `dgemm_` / `zgemm_`** in `AB_MODE=gpu` without `ab_init()`:
  - every entry × 1e-25 or × 1e25;
  - rows of op(A) and columns of op(B) scaled from 2⁻¹²⁰ to 2¹²⁰, with
    C0 alike;
  - even-k / odd-k factors 2^±70, aligned (the GPU must certify it) and
    misaligned (Accelerate must take it);
  - an Inf in A;
  - DGEMM alpha 1e-45 and 1e45;
  - DGEMM with a zero row of op(A), where beta·C0 is all of C: with that
    row of C0 × 2⁻¹⁰⁰ (Accelerate must take it), and with the other rows
    × 2⁻¹⁰⁰ instead (the GPU must certify it);
  - DGEMM C0 × 2¹³⁵ with beta = 2⁻²⁰: beta·C0 fits FP32's range but C0 does
    not, so Accelerate must take it.

  These run with every TRANS kind of op(). Each must give C within 1e-11
  of cblas (Inf and NaN where cblas has them), on the route the case
  names.
- **Exactness**: C of data scaled by 2^{-100, 0, 100} per row and column
  must be bit for bit that power of two times C of the unscaled data.
- **Native `ab_dgemm`, `ab_dgemm_scaled` and `ab_zgemm`**: the same 1e±25
  data, rows spread over 2⁻¹⁰⁰…2⁻⁶⁰, columns over 2⁶⁰…2¹⁰⁰, and an
  exactness case (A and B × 2⁹⁰).

`test_zgemm_imbalance` gives `zgemm_` (`AB_MODE=gpu`, alpha = 1, beta = 0,
every TRANS kind) data whose real and imaginary parts are 10⁻⁶ apart:
nearly real A and B, nearly imaginary ones, and rows of op(A) and columns
of op(B) that alternate between the two. Re(C) and Im(C) must each be
within 8 u² of a Dot2 reference, relative to their own componentwise
scales. Gauss's three-product form misses Im(C) by about 10⁶ u² there, so
the call must run as four products.

Known failures live in the file's `XFAIL` table, each with a short tag
naming the defect. An XFAIL that now passes (XPASS), or an entry that matches
no case (STALE), fails the run: when a fix lands, delete its entries.
`./build/test_blas_abi -v` lists every case, and `./build/test_blas_abi
<test_name>…` runs only those suites (the STALE check then needs a full run). `make test-symbols` is the
strict export check: the dylib and the static archive must each export
exactly `_dgemm_` / `_zgemm_`, and not the `$NEWLAPACK` variants (before 1.4.0 they exported only the latter, so Fortran calls bypassed the library).

## Golden bits

`tests/test_qe_golden.c` makes one call per case through `dgemm_` /
`zgemm_` in `AB_MODE=gpu` and hashes each whole C array, leading-dimension
padding included, with 64-bit FNV-1a. `tests/golden/qe_golden.txt` holds
the hashes apple-bottom 1.4.0 produced, so every later change must keep the
results bit for bit. The cases: zgemm for every (transA, transB) in
{N,T,C}², dgemm for {N,T}² and 'C' (read as 'T'); alpha 1 or not, beta 0,
1 or neither; ld > rows with NaN in the padding of A, B and C; both GEMM
kernels; mixed magnitudes; beta = 0 with NaN and Inf in C; cancellation
(Ai close to −Ar, rows of op(A) that sum to zero); signed zeros; and
Quantum ESPRESSO shapes (zgemm 'C','N' 160×160×9140 and 'N','N'
9140×160×160, dgemm 'T','N' 36×64×73062 with alpha = 2, dgemm 'N','T'
4000×171×64). Inputs come from splitmix64 seeded with the case name. Every
case also checks that the GPU computed C, that A, B and C's padding are
unchanged, and that C is within 1e-11 of cblas, so a golden file cannot
freeze a wrong result.

The bits depend on the GPU and the Metal compiler: the file's header
records `ab_device_name()`, the macOS build (`sysctl kern.osversion`) and,
on its `library:` line, the `ab_build_info()` of the library that wrote it
(version and commit, `-dirty` for a modified tree). On another device the
hashes are not compared and the test prints `SKIP` with the reason (the
cblas checks still run); `AB_GOLDEN_STRICT=1` makes that a failure. On
another macOS build they are compared all the same, and a mismatch fails
naming both builds.

Re-baseline (a new device, or a macOS update that changed the bits) only
from the library the file names, never from the tree under test, which
must then match the new file:
`DYLD_LIBRARY_PATH=<directory of its libapplebottom.dylib> ./build/test_qe_golden --generate`
(`--file PATH` reads or writes another file). Start the binary directly,
not through `make` or `sh -c`: macOS drops `DYLD_*` variables when it
starts a system binary. Check that the `library:` it prints is the one the
file names. The tree's own library replaces the file only for an intended
change of results, said in the commit; the file's `library:` line then
names the tree's commit. The committed file was generated by the 1.4.0
build of the QE si64 release check (`apple-bottom 1.4.0 (c5a1fa6a8166)`);
this tree's 1.4.0 library gives the same bits.

## Staged path

`tests/test_staging.c` runs every case of its table in four child
processes, all with `AB_MODE=gpu AB_MIN_GPU_DIM=0`: with `AB_HOST_PACK=1`
(the host path: host pack, FP64 ↔ DD and epilogue), by default (the staged
path, `src/gpu_stage.h`), with `AB_STAGING_MB=0` (no buffers kept between
calls), and with A, B and C registered by `ab_register_host_memory` (read
and written in place; in every third case the complex arrays start 8 bytes
off a 16-byte boundary, so they are copied instead). Each child makes each
case twice in a row, the second time on the buffers the first left. Every
case must leave A and B alone, repeat its bits and take the expected path,
as `ab_internal_last_staged()` reports it: the staged result stands
exactly where the exponent-range guard certifies the call, its lines
scaled or not and a ZGEMM by Gauss-3M or 4M (the test asks the guard's own
functions, `src/ab_internal.h`), a DGEMM's alpha and beta lie in their
window, and no Inf or NaN reaches the GPU epilogue; every other call is
made on the host path. It must run on the GPU and hash the
whole C array (padding included) the same in all four children, unless the
guard sends it to Accelerate (NaN or Inf in A, B or a DGEMM's C, a range
the scaling cannot keep), which it must then do in every child. The
registered child also checks how many operands were used in place and
`ab_register_host_memory()`'s refusals. The cases: zgemm and dgemm for
every TRANS pair in {N,T,C}², alpha 1 or not, beta 0, 1 or neither; both
GEMM kernels, several tiles, M = 1, N = 1, no padding and padding wide
enough to copy column by column; mixed magnitudes, operands the guard
scales, rows it certifies only on the operands (its second stage), complex
parts imbalanced enough for 4M, a DGEMM alpha outside its window, K = 1
products of random bit patterns (every conversion and epilogue rounding
shows in C), signed zeros, tiny and subnormal C, cancelling C, NaN, Inf
and overflow; and Quantum ESPRESSO shapes large enough that the copies run
in pieces.

The native ZGEMMs take their Gauss-3M or 4M temporaries from the same
pool, so after the run the pool must hold buffers in the host child and
the staged one, and none with `AB_STAGING_MB=0`. A fifth child
(`--child-native`) checks them directly. `ab_zgemm`, `ab_zgemm_async`,
`ab_batch_zgemm` and `ab_zgemm_batched` (with a change of shape mid-batch)
must give the bits of the first `ab_zgemm`. So must repeats after the pool
served larger and smaller shapes, and 4 threads calling at once. On nearly
real data (imaginary parts 1e-6 of the real ones), which runs as 4M, they
must give the bits of an explicit 4M call, which differ from 3M's, also in
a batch that switches between 3M and 4M. A repeated call must not grow
the pool, `ab_shutdown()` must empty it, and after small, mid and big
shapes in turn it must hold what big alone leaves (a slot remakes buffers
that are too small rather than adding new ones).

## Per-call trace and forced GPU failures

`tests/test_trace.c` re-executes itself (`--child <scenario> <dir>`, via
`posix_spawn` so the child can use Metal; the helpers are shared with
`test_blas_abi.c` in `tests/test_spawn.h`) with the variables under test,
after dropping every `AB_*` variable, `MTL_DEBUG_LAYER` and `MallocNanoZone`
of the caller's environment, and checks the child's output and the files it leaves in a
scratch directory under `$TMPDIR` (kept, and named in the failure, when a
test fails). A child still running after 120 s, or its `fork()`ed child
after 30 s, is ended by an alarm, and the test fails. `test_trace_lines`
runs GPU and CPU calls through the six BLAS-ABI entry points with
`AB_MODE=gpu` and `AB_TRACE`, then a `fork()`ed child opens a file of its
own on the descriptor number of the trace file it inherited and repeats a
GPU call, which falls back to cblas. There must
be exactly one trace file per pid, each starting with the `#` header
(`ab_build_info()`, pid, `AB_MODE` with `AB_MIN_GPU_DIM`, the MPI rank
line, legend), with one line per call in
`seq` order, the documented keys in order, and the arguments (TRANS as
passed, leading dimensions, `%a` alpha and beta) read back exactly. On GPU
lines the stages sum exactly to `t1_ns - t0_ns`, the split of `other_ns`
sums exactly to it, and `gpu_start_ns`, `gpu_end_ns` lie within the call;
the child's line starts after its parent's last, and the child's own file
holds exactly what it wrote. `test_trace_hash` (`AB_TRACE_HASH=1`) checks that
each call hashes exactly the operands it reads, against the hash the test
computes from the values on its own (FNV-1a over the 64-bit words of the
referenced elements, column by column), so equal operands stored with
other leading dimensions hash equal and a changed element does not.
`test_trace_off` checks that no file appears with `AB_TRACE` unset or
empty. `test_fail_stage_alloc` and `test_fail_stage_kernel` set
`AB_TEST_FAIL_STAGE` (every `ab_matrix_create()` returns NULL, or the
command buffer of every `ab_dgemm_scaled()` and `ab_zgemm()` reports an
error) with `AB_REPORT=1`: a GPU-routed DGEMM and ZGEMM must give C bitwise
equal to cblas's, `ab_get_last_dispatch_path()` must say `"cpu"`,
`ab_get_stats()` must count exactly the elements converted and the GEMM
kernels of the GPU work before the failure (none for `alloc`; for
`kernel` the uploads, and the ZGEMM, which counts its kernel before it
reports the failure), `AB_REPORT` must count each as a `gpu_failed`
fallback and the trace must say `fallback:gpu_failed`.

## BLAS-2 (GEMV, HEMV)

`tests/test_blas2.c` checks `ab_dev_dgemv` / `ab_dev_zgemv` /
`ab_dev_zhemv` on `ab_dev_malloc()` and `ab_dev_register()` buffers,
`ab_dgemv_blas` / `ab_zgemv_blas` / `ab_zhemv_blas` and the plain
`dgemv_` / `zgemv_` / `zhemv_` symbols, in `AB_MODE=gpu`. A case passes
when y is within tol(K) = 16 K w u² + 8·2⁻⁵³ (u² = 2⁻⁴⁸, K the
dot-product length, w = 2 for complex) of a compensated FP64 reference
(`tests/test_blas2_ref.h`, shared with `benchmarks/bench_gemv.c`)
componentwise, relative to |alpha| (|op(A)| |x|)_i + |beta| |y0_i|, and so
is Accelerate's result; the expected path computed y ("gpu", or "cpu" with
y bitwise equal to Accelerate's run on the same memory, since Accelerate's
bits depend on A's address modulo 256 bytes); and A, x and every element of
y's array outside the call's strided elements are bitwise unchanged.
Leading-dimension padding, the unreferenced triangle of a Hermitian matrix,
the imaginary parts of its diagonal and the gaps between strided elements
hold NaN.

- the three semantics tests run every TRANS or UPLO flag on shapes from
  1 × 1 to 30001 × 5 (skinny shapes take the grid-split and reduction
  paths), padded and tight lda, increments ±1, 2, −2, −3 and complex
  alpha and beta, through all four entry points (488 cases);
- ZHEMV's tiling: each of panels of 32 or 96 rows with chunks of one to
  three panels, forced through `ab_internal_b2_hemv_override`, at sizes
  from 1 to a panel ± 1, a chunk + 1 and a ragged last panel, chunk and
  group of four columns, both triangles (the semantics matrix sees only
  32-row panels and one-panel chunks);
- the kernels' conversion of A's elements to double-float, run on the GPU
  alone (`ab_internal_b2_convert`), bit for bit against the host's
  `fp64_to_dd` with FP32 subnormals flushed as the GPU flushes them: every
  sign and FP64 exponent with FP32 mantissas 0, 1, even, odd and all ones
  (rounding up carries into the exponent) above rounding ties and their
  neighbours, 989 random mantissas each, and residuals x − hi just around
  2⁻¹²⁶, where lo starts to flush (4.2 million inputs); where hi would not
  be finite it must be Inf or NaN;
- beta = 0 with NaN and ±Inf in y (y not read); NaN and Inf in A, x,
  alpha or beta give Accelerate's result bitwise (the call runs there),
  and Inf in y with beta ≠ 0 stays on the GPU;
- the range guard: an element of 1e200, all elements 1e-200 or 1e38 run on
  Accelerate; A scaled by 1e30, x by 1e200 or 1e-200, x with components 1
  and 1e-30, zero rows, columns and x elements, and FP64 subnormals in A
  stay on the GPU and within tol; with its x element 1.5 × 2⁻⁷¹, so that
  the exponents pass, an element just under what FP32 rounds to 2¹²⁸ stays
  on the GPU, and the least that rounds to it (hi = Inf) runs on
  Accelerate;
- m or n = 0 and alpha = 0, with A and x all NaN, give Accelerate's result
  bitwise;
- determinism (two device-API calls and a BLAS-ABI call give the same
  bits), 8 threads against single-threaded bits, long dot products
  (K = 65536; error under 2e-14 of |A||x| as well as tol), registered
  memory (updates seen by the next call, the caller's bytes at their
  offset, unmapped, holed and PROT_NONE spans refused), invalid
  device-API arguments;
- the `AB_MODE=auto` rule against its measurements: every shape of
  `benchmarks/results/2026-10-19-ba345df-gemv/sweep_*.csv` (GEMV) and
  `2026-10-19-1bc503a-zhemv/sweep_*.csv` (ZHEMV) that the rule sends to the
  GPU was measured at least 1.10× faster there, and each routed class has
  such shapes (no GPU needed);
- in child processes: `AB_MODE=cpu`, `gpu`, `auto` and `AB_MIN_GPU_DIM`
  routing, illegal arguments through the Fortran symbols (cblas names the
  parameter and exits with status 255), `AB_TEST_FAIL_STAGE=kernel` and
  `alloc` (bitwise Accelerate results, `gpu_failed` in `AB_REPORT`) and the
  `AB_REPORT` counts.

## Routing unit tests

`tests/test_dispatch.c` tests the routing decision of the BLAS-ABI entry
points (`src/dispatch.c`, see `dispatch.h`), built from that file alone,
without the library or a GPU. It checks every built-in table
(`src/dispatch_table.h`, one per calibrated chip, generated from
`benchmarks/calibration/<chip>/` by `scripts/ingest_calibration.py`) against
the `calibrate_summary.csv` it was derived from: every measured shape routes
as recorded there, and no shape that was not at least 1.10x faster on the
GPU goes there; each Quantum ESPRESSO si64 shape must have been measured.
It checks the choice of table by chip and GPU core count (`AB_CHIP`,
`AB_GPU_CORES`; a sibling with another core count or an uncalibrated chip
gets no rows), the chip slug and the `# chip:` / `# gpu_cores:` header
keys of a table file. The host it runs on may have no table (a CI runner).
It also checks the precedence of `AB_MODE`, `AB_MIN_GPU_DIM` and
`AB_CROSSOVER_FLOPS[_REAL]` over the table, the row bounds and TRANS
classes (`'C'` is `'T'`), and the `AB_DISPATCH_TABLE` format: parsing,
loading, malformed and missing files.

## Fortran binding probe

`tests/test_fortran_blas.f90` is a plain gfortran `CALL DGEMM` / `CALL
ZGEMM` (the QE cegterg `'T'`/`'C'`, `'N'` pattern with M != K and padded
leading dimensions). `make test-fortran` links it once against the dylib and
once against the static archive and runs both with `AB_MODE=gpu`. Like QE,
it never calls `ab_init()`. Each call must reach the GPU path
(`ab_get_last_dispatch_path()` is `"gpu"` and the `ab_get_stats()` GEMM
counter advanced) and match a reference loop within 1e-11. Then `CALL DGEMV`,
`CALL ZGEMV` (`'C'`) and `CALL ZHEMV` (`'U'`, NaN in the unreferenced
triangle) the same way: a 1 × 1 DGEMM, which runs on Accelerate, sets the
path to `"cpu"` first, so `"gpu"` afterwards shows the call reached
apple-bottom. The target is skipped, with a message, when neither
`gfortran-15` nor `gfortran` is on `PATH`; set `GFORTRAN=` to pick a compiler.

## Double-float FFT

`tests/test_fft.c` checks `apple_bottom_fft.h` two ways for every case.
Against a reference in double-double arithmetic (`tests/fft_ref.h`, about
106 bits: a mixed-radix FFT validated against the O(n²) sum to 5e-32), the
relative L2 and max errors must stay within 4e-15 × (1 + log₂N / 4) and
twice that, about three times the worst measured. Against
`tests/fft_emu.cpp`, which runs the GPU's double-float arithmetic
(`src/fft/fft_device.h`) on the CPU in the same order, every output bit
must match: this pins the kernels' indexing, pass order, conversions and
scaling exactly. The cases: 1D lengths from 1 to 2048 with every radix
(2, 3, 4, 5, 7, 8, 11, 13), four-step lengths up to 2048² (and 2049, which
has no split, rejected), 2D and 3D batches up to 72³, FFTW-style strides,
embeddings and gaps, interleaved batches in and out, with the padding
between output elements left untouched, in place (staged and zero copy),
`ab_fft_malloc` and `ab_fft_register`d memory (bitwise the staged result,
nothing kept after `ab_fft_unregister`, new data at a reused address read
afresh), batches split into chunks (bitwise the unsplit result), the three
normalizations, the FP64 ↔ double-float conversions on 4 million values and
pairs (ties, powers of two), bad arguments, the FP32 exponent range, six
threads on shared and own plans, and a `fork()`ed child, which gets
`AB_ERROR_NO_DEVICE` instead of a crash.

## Meta-test

`tests/test_ghost_detection.c` is a self-contained meta-test that
deliberately includes one ghost test and asserts the per-test delta
guard fires. Runs via `make test-ghost-detection`. The wrapper inverts
the inner binary's exit code (the inner returns 42 on success, the
target returns 0). Wired into both `test:` and `ci-local:` aggregates.
