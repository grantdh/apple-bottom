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

For parameterized helpers (e.g., `test_dgemm_bit_identical(int N)`),
register a `void(void)` wrapper per argument value — convention is
`test_helper_<argvalue>` (see `test_dgemm_bit_identical_1024`,
`test_dgemm_bit_identical_768` in `test_device_api.c`).

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

- **Registry-migrated (PR-1, ghost-guarded):**
  - `tests/test_correctness.c` — 66 tests
  - `tests/test_chaos.c` — 18 tests
  - `tests/test_device_api.c` — 11 tests (7 unparameterized + 4
    wrappers around 2 parameterized helpers)

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

## Meta-test

`tests/test_ghost_detection.c` is a self-contained meta-test that
deliberately includes one ghost test and asserts the per-test delta
guard fires. Runs via `make test-ghost-detection`. The wrapper inverts
the inner binary's exit code (the inner returns 42 on success, the
target returns 0). Wired into both `test:` and `ci-local:` aggregates.
