# apple-bottom test suite

## Adding a new test to a registry-based file

The registry-based test runners (`test_correctness.c`, `test_chaos.c`,
`test_device_api.c`) dispatch via a `TESTS[]` registry array of
`{name, fn}` pairs declared just above `main()`. To add a test:

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

Counting rule (README, `make ci-local`): a *test* is one `test_precision`
case, one `TESTS[]` entry, or one BLAS-ABI suite. `make test` runs 138
(14 precision + 73 correctness + 16 device-API + 9 routing + 26 BLAS-ABI)
and `make ci-local` adds the 18 chaos tests, for a `TEST_TOTAL` of 156.
Device-API `CHECK`s and BLAS-ABI cases are reported on their own
`CI_REPORT.txt` lines and are not part of the total.

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
  - `tests/test_blas_abi.c` — 26 suites (1007 cases), see below

- **Not yet registry-based (one `main`, no ghost guard):**
  - `tests/test_precision.c` — needs decomposition strategy
  - `tests/verification/test_convergence.c` — monolithic numerical
    V&V file; decomposition needs care because behavior changes are
    load-bearing

## Targets outside `make test`

| Target | What it runs |
|---|---|
| `make ci-local` | `make version-check`, every suite of `make test`, the 18 chaos tests and the V-2 convergence study (the DGEMM error-versus-N study of `docs/vv/VV_REPORT.md`, `tests/verification/test_convergence.c`); writes `build/CI_REPORT.txt` |
| `make test-chaos` | pathological inputs, IEEE edge cases, concurrent threads |
| `make test-verification` | the V-2 convergence study alone |
| `make test-rectangular` | `test_rectangular.c`: tall, wide, thin-K and conjugate-transposed shapes up to 18277 × 150 × 18277, componentwise against cblas, then timings (peaks at ~21 GB) |
| `make test-rectangular-diag` | eight rectangular patterns against cblas |
| `make test-python` | the Python bindings (`tests/python/`) |
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
Known failures live in the file's `XFAIL` table, each with a short tag
naming the defect. An XFAIL that now passes (XPASS), or an entry that matches
no case (STALE), fails the run: when a fix lands, delete its entries.
`./build/test_blas_abi -v` lists every case. `make test-symbols` is the
strict export check: the dylib and the static archive must each export
exactly `_dgemm_` / `_zgemm_`, and not the `$NEWLAPACK` variants (before 1.4.0 they exported only the latter, so Fortran calls bypassed the library).

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
