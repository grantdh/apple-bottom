# apple-bottom test suite

## Adding a new test to a registry-based file

The registry-based test runners (`test_correctness.c`, `test_chaos.c`,
`test_device_api.c`, `test_dispatch.c`, `test_trace.c`) dispatch via a
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
count of each file is listed below. `make test` runs 150, every test but
the chaos ones, and `make ci-local` all 168 (`TEST_TOTAL` in
`build/CI_REPORT.txt`). Device-API `CHECK`s and BLAS-ABI cases are
reported on their own `CI_REPORT.txt` lines (`DEVICE_API_CHECKS`,
`BLAS_ABI_CASES`) and are not part of the total, nor is `test_qe_golden`
(its `TEST_QE_GOLDEN` line says `PASS`, or `SKIP` on another device).

- **Registry-based (ghost-guarded):**
  - `tests/test_correctness.c` — 73 tests
  - `tests/test_chaos.c` — 18 tests
  - `tests/test_device_api.c` — 16 tests (152 `CHECK`s); the GEMM
    tests compare `ab_dev_dgemm` / `ab_dev_zgemm` with column-major
    cblas over all trans pairs, non-square and tall-skinny shapes,
    `ld > rows` and buffer offsets, and check that
    the device path is bitwise the host API on the transposed operands
  - `tests/test_dispatch.c` — 9 tests of the BLAS-ABI routing, no GPU,
    see below
  - `tests/test_blas_abi.c` — 31 suites (1028 cases), see below
  - `tests/test_trace.c` — 5 tests of `AB_TRACE` and
    `AB_TEST_FAIL_STAGE`, see below

- **Not yet registry-based (one `main`, no ghost guard):**
  - `tests/test_precision.c` — 16 tests; needs decomposition strategy
  - `tests/test_qe_golden.c` — 32 golden-bits cases, see below
  - `tests/verification/test_convergence.c` — monolithic numerical
    V&V file; decomposition needs care because behavior changes are
    load-bearing

## Targets outside `make test`

| Target | What it runs |
|---|---|
| `make ci-local` | `make version-check`, every suite of `make test`, the chaos tests and the V-2 convergence study (the DGEMM error-versus-N study of `docs/vv/VV_REPORT.md`, `tests/verification/test_convergence.c`); writes `build/CI_REPORT.txt` |
| `make test-chaos` | pathological inputs, IEEE edge cases, concurrent threads |
| `make test-verification` | the V-2 convergence study alone |
| `make test-rectangular` | `test_rectangular.c`: tall, wide, thin-K and conjugate-transposed shapes up to 18277 × 150 × 18277, componentwise against cblas, then timings (peaks at ~21 GB) |
| `make test-rectangular-diag` | eight rectangular patterns against cblas |
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
`test_auto_mode` runs `AB_MODE=auto` with the built-in calibrated table:
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
  malloc has handed out by less than `MEM_MALLOC_LIMIT_KB`, which catches
  small leaks, and `phys_footprint` (`task_info`) by less than
  `MEM_FOOTPRINT_LIMIT_KB`, which catches a leaked operand or GPU buffer.
  `gen_memory` records what 1.4.0 measures, what planted leaks cost, and
  how the limits follow.
- `test_fork_exit`: as `test_fork_after_gpu` without `AB_PROFILE`, but the
  forked child ends with `exit()` (atexit handlers and static destructors
  run, as at a Fortran `STOP`), not `_exit()`; it must exit with status 0
  within 30 s (an alarm ends it then, and the case fails).

Known failures live in the file's `XFAIL` table, each with a short tag
naming the defect. An XFAIL that now passes (XPASS), or an entry that matches
no case (STALE), fails the run: when a fix lands, delete its entries.
`./build/test_blas_abi -v` lists every case. `make test-symbols` is the
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

## Per-call trace and forced GPU failures

`tests/test_trace.c` re-executes itself (`--child <scenario> <dir>`, via
`posix_spawn` so the child can use Metal; the helpers are shared with
`test_blas_abi.c` in `tests/test_spawn.h`) with the variables under test,
after dropping every `AB_*` variable and `MTL_DEBUG_LAYER` of the caller's
environment, and checks the child's output and the files it leaves in a
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

## Routing unit tests

`tests/test_dispatch.c` tests the routing decision of the BLAS-ABI entry
points (`src/dispatch.c`, see `dispatch.h`), built from that file alone,
without the library or a GPU. It checks the built-in table
(`src/dispatch_table.h`, written by `make calibrate`) against the
`calibrate_summary.csv` it was derived from: every measured shape routes as
recorded there, and no shape that was not at least 1.10x faster on the GPU
goes there; each Quantum ESPRESSO si64 shape must have been measured. It
also checks the precedence of `AB_MODE`, `AB_MIN_GPU_DIM` and
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
counter advanced) and match a reference loop within 1e-11. The target is
skipped, with a message, when neither `gfortran-15` nor `gfortran` is on
`PATH`; set `GFORTRAN=` to pick a compiler.

## Meta-test

`tests/test_ghost_detection.c` is a self-contained meta-test that
deliberately includes one ghost test and asserts the per-test delta
guard fires. Runs via `make test-ghost-detection`. The wrapper inverts
the inner binary's exit code (the inner returns 42 on success, the
target returns 0). Wired into both `test:` and `ci-local:` aggregates.
