# apple-bottom test suite

## Adding a new test in a registry-migrated file

The migrated test runners (`test_correctness.c`, `test_chaos.c`,
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
  entirely. PR-1 acceptance criterion #7 explicitly forbids `exit()`
  in test bodies; `grep -n 'exit\s*(' tests/test_correctness.c
  tests/test_chaos.c tests/test_device_api.c` must be empty.

## Coverage status

Counting rule (README, `make ci-local`): a *test* is one `test_precision`
case, one `TESTS[]` entry, or one BLAS-ABI suite. `make test` runs 127
(14 precision + 72 correctness + 16 device-API + 25 BLAS-ABI) and
`make ci-local` adds the 18 chaos tests, for a `TEST_TOTAL` of 145.
Device-API `CHECK`s and BLAS-ABI cases are reported on their own
`CI_REPORT.txt` lines and are not part of the total.

- **Registry-migrated (PR-1, ghost-guarded):**
  - `tests/test_correctness.c` — 72 tests
  - `tests/test_chaos.c` — 18 tests
  - `tests/test_device_api.c` — 16 tests (152 `CHECK`s); the GEMM
    tests compare `ab_dev_dgemm` / `ab_dev_zgemm` with column-major
    cblas over all trans pairs, non-square and tall-skinny shapes,
    `ld > rows` and buffer offsets (K-P0-2 / H-P0-4), and check that
    the device path is bitwise the host API on the transposed operands
  - `tests/test_blas_abi.c` — 25 suites (999 cases), see below

- **Pending PR-1.5 (still single-main, no ghost-guard):**
  - `tests/test_precision.c` — needs decomposition strategy
  - `tests/verification/test_convergence.c` — monolithic numerical
    V&V file; decomposition needs care because behavior changes are
    load-bearing

- **Orphan files (not in any Makefile build target — separate
  delete-or-wire-up decision):**
  - `tests/test_rectangular.c`
  - `tests/test_original_rectangular.c`
  - `tests/test_rectangle_debug.c`

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
robustness only: those DGEMMs are memory-bound and faster on Accelerate.
Known failures live in the file's `XFAIL` table, tagged
with audit IDs. An XFAIL that now passes (XPASS), or an entry that matches
no case (STALE), fails the run: when a fix lands, delete its entries.
`./build/test_blas_abi -v` lists every case. `make test-symbols` is the
strict export check: the dylib and the static archive must each export
exactly `_dgemm_` / `_zgemm_`, and not the `$NEWLAPACK` variants (H-P0-1).

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
