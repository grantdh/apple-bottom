# Changelog

All notable changes to this project will be documented in this file.

## [Unreleased]

### Added
- **The staged path of the BLAS ABI**: a GPU-routed `dgemm_`/`zgemm_` copies
  each operand (`memcpy`, column by column unless it is dense) into GPU
  buffers the library keeps and reuses between calls (per process, at most
  `AB_STAGING_MB`, default 2048 MB), and the GPU does the rest in one
  command buffer: transposing and conjugating op(A) and op(B) and converting
  them to double-float (and Gauss-3M's Ar + Ai, Br + Bi), the GEMM kernels,
  and the conversion back and, for ZGEMM with alpha = 1 and beta = 0 or 1
  (and every DGEMM, whose alpha and beta the kernel applies), the epilogue.
  The results are bit for bit those of the host path (host pack, FP64 ↔ DD
  conversion and epilogue): the GPU's conversions and epilogue work on the
  bit patterns with integer arithmetic, rounding as the host does, and
  follow the operations of the compiled host epilogue, signed zeros
  included. The conversion also gathers the exponent-range guard's
  statistics (see Fixed), from which the GPU scales lines, in the same
  command buffer, as the host path does: the elements of a scaled row of
  op(A) or column of op(B) (and a DGEMM's C) are converted again at
  x · 2^-scale, rounded as the host rounds, and C is multiplied back in the
  epilogue. A ZGEMM whose parts call for 4M runs as Gauss-3M first, then a
  second command buffer forms 4M's imaginary part from the same operands
  (its real part is Gauss-3M's) and runs the epilogue again. The staged
  result stands where the guard certifies the call and the GPU's scales
  are the host's. A call the guard does not certify goes straight to
  Accelerate (the host path's guard, on the same statistics, would send it
  there). One that meets Inf or NaN in the GPU epilogue, one whose staged
  run fails, and a DGEMM whose alpha or beta lies outside 2^±20 are made on
  the host path, as is every
  call when the CPU's floating-point control is not the default (rounding
  mode, flush to zero, default NaN). `AB_HOST_PACK=1` keeps every call on
  the host path. A staged call's stages in the run report and trace: `pack`
  is the guard's check, `upload` the copies into the GPU buffers, `kernel`
  includes the GPU's conversion, scaling and epilogue, and `download` is
  the copy of the result into C.
- **`ab_register_host_memory(ptr, nbytes)` /
  `ab_unregister_host_memory(ptr)`**: BLAS-ABI operands that lie in
  registered memory are read in place by the GPU, without the copies, and C
  is written in place when the GPU does not read it. The caller keeps the
  memory alive and in place until it is unregistered; nothing is ever
  wrapped or cached by address otherwise.
- **`tests/test_staging.c`** (in `make test`): 127 cases run on the host
  path, the staged path, the staged path with `AB_STAGING_MB=0` and with
  registered operands must give the same bits, with every TRANS pair, alpha
  and beta, padding, shapes from M = 1 to QE's, random bit patterns, signed
  zeros, subnormal and cancelling C, and data the exponent-range guard
  certifies only in its second stage, scales, runs as 4M (imbalanced complex
  parts) or sends to Accelerate (Inf, NaN, overflow): the staged result must
  stand exactly where the host path computes the call unscaled on the GPU.
- **Why a call left the staged path**: `AB_REPORT` adds a line
  `<routine> staged kept=… kept_4m=… host_pack=… fpcr=… scalar=… alloc=…
  gpu_failed=… nonfinite=… range=…` that counts each GPU-routed call once,
  by whether it kept the staged result (`kept_4m`: by 4M) or else the
  first reason it went to the host path; `AB_PROFILE` adds the same as a
  table and `staged=<outcome>` to each GPU call's line, and `AB_TRACE`
  adds `staged=<outcome>` to each call that reached the staged path. See "Reading a run report" in
  `docs/INTEGRATION.md`. On Quantum ESPRESSO si64 (input B, one rank) it
  showed that 512 of the 1,110 calls that reached the staged path left it:
  492 ZGEMMs whose parts call for 4M and 20 DGEMMs with scaled lines, which
  the staged path now keeps (above). Measurements:
  `benchmarks/results/2026-10-29-351d56c-staged-refusals/`.
- **`AB_TRACE=<path>`**: every BLAS-ABI call (`dgemm_`, `zgemm_`,
  `ab_*gemm_`, `ab_*gemm_blas`; Accelerate and GPU, every `AB_MODE`) appends
  one `key=value` line to `<path>.<pid>` when it returns (a call with an
  illegal argument, for which Accelerate ends the process, gets none), so
  each process, including a `fork()`ed child or an MPI rank, writes a file
  of its own, which starts with `#` lines naming the build, `AB_MODE`, the
  effective `AB_MIN_GPU_DIM`, the MPI rank and size and the fields. A line
  gives the call's arguments as passed (TRANS, M, N, K, the leading
  dimensions, alpha and beta exactly as `%a` hex floats), its start and end
  on the machine-wide `mach_absolute_time` clock in ns, so traces of
  several processes line up, and its path (`cpu`, `gpu` or
  `fallback:<reason>`).
  GPU calls add the stages of `AB_REPORT` in ns, the command buffer's
  `GPUStartTime` and `GPUEndTime` on the same clock, and split the stage
  `other` into `temps` (ZGEMM's GPU temporaries), `encode`, `submit`
  (commit until the GPU starts), `wake` (GPU end until the waiting thread
  runs) and `rest`; both sums are exact. `AB_REPORT` and the `AB_PROFILE`
  tables are unchanged. See "Per-call trace" in `docs/INTEGRATION.md`.
- **`benchmarks/replay.c`** (`make bench-replay`) replays a traced pw.x
  run's BLAS calls, one process per MPI rank, with the traced CPU time
  between calls and k-point pool barriers, in `AB_MODE=cpu` or `gpu`, or
  with the GPU-routed calls as native GEMMs on resident operands
  (`nohost`); `benchmarks/replay_summary.py` compares replays with the run,
  and `make test-replay` checks it. See "Replay" in `benchmarks/qe/README.md`.
- **`AB_TRACE_HASH=1`** adds hashes of op(A) and op(B) as stored and of C
  on entry, when the call reads them: FNV-1a over the 64-bit words of the
  referenced elements, column by column, so a replay can tell repeated
  operands whatever their leading dimension.
- **`AB_TEST_FAIL_STAGE=<alloc|kernel>`** (test-only) fails GPU work where
  it really can fail: `alloc` makes every `ab_matrix_create()` return NULL,
  `kernel` the command buffer of every `ab_dgemm_scaled()` and `ab_zgemm()`
  report an error. A GPU-routed BLAS-ABI call then takes the whole-call
  fallback to Accelerate, counted as `gpu_failed`.
- **`tests/test_trace.c`** (5 tests, in `make test` and `make ci-local`):
  trace files per pid, header, fields and exact arguments, the stage and
  `other` sums, GPU times within each call, a `fork()`ed child's file of
  its own on the inherited trace descriptor left intact, hashes, no file
  without `AB_TRACE`, and for each `AB_TEST_FAIL_STAGE` value a result
  bitwise equal to cblas, the path record `"cpu"`, the `ab_get_stats()`
  GPU work done before the failure, and the `gpu_failed` fallback in
  `AB_REPORT` and the trace.
- **Guards for results that faster GPU code must keep** (in `make test`
  and `make ci-local`). `tests/test_qe_golden.c` hashes C after 32
  GPU-routed `dgemm_`/`zgemm_` cases (every TRANS pair, alpha and beta
  variants, padded leading dimensions with NaN, beta = 0 with NaN and Inf
  in C, cancellation, signed zeros, Quantum ESPRESSO shapes) against
  `tests/golden/qe_golden.txt`, the bits of the 1.4.0 release-check build
  `c5a1fa6a8166`, and checks each case against cblas, so the file cannot
  freeze a wrong result; another device gives `SKIP` (a failure under
  `AB_GOLDEN_STRICT=1`), reported on the `TEST_QE_GOLDEN` line of
  `make ci-local`. `tests/test_blas_abi.c` adds long-K precision through
  the BLAS ABI (K up to 143666, against a compensated Dot2 reference),
  determinism of repeated calls, 8 threads on distinct and on identical
  shapes, memory growth over 2000 GPU calls, and a `fork()`ed child that
  ends with `exit()`. The test counts are kept in `tests/README.md` only.

### Changed
- `make version-check` skips an `[Unreleased]` section at the top of
  `CHANGELOG.md` and checks the first versioned entry.
- **ZGEMM runs Gauss-3M's three real products concurrently** (one encoder
  with `MTLDispatchTypeConcurrent`) in every ZGEMM path, so the products of
  a small C share the GPU instead of running one after another. A call whose
  `Ci` shares a buffer with an input keeps the serial order. Results are
  bitwise unchanged.
- **The 64×64 GEMM kernels read their threadgroup tiles as float4**:
  `dd_dgemm`, `dd_dgemm_ab` and `dd_dsyrk` store the A tile k-major and
  read each thread's values with two float4 loads per operand, 2-7% faster
  on M2 Max. The tall-skinny kernel keeps its layout, which measured
  faster there. Results are bitwise unchanged.
- **Small outputs run on 32×32 tiles** (`dd_dgemm_narrow`,
  `dd_dgemm_ab_narrow`, 2×2 outputs per thread): a C of at most 480×480
  elements gets four times the threadgroups of the 64×64 kernels, 1.06-2.9×
  faster on M2 Max (2.6× on 160×160×18432). A tall C (M ≥ 4N) moves off the
  tall-skinny kernel only where the 32-column tiles add no padding or the
  tall-skinny grid would have at most 16 threadgroups (2.1× on
  200×40×18277). The rule is `use_narrow_tiles` in `src/apple_bottom.m`.
  Results are bitwise unchanged. `tests/test_precision.c` gains a 64×64 scaled case and a
  tall-skinny ZGEMM case (16 tests), so every kernel stays covered.
- `tests/test_blas_abi` takes test names (`./build/test_blas_abi
  test_exponent_range`) to run only those suites.
- `AB_TRACE`'s `temps_ns` is 0 on the staged path: ZGEMM's GPU temporaries
  come from the reused buffers, counted in `alloc_ns`.
- **Small grids prefetch the next K-tile into registers.** Each tiled GEMM
  kernel has a register-prefetch instance (`dd_dgemm_pf`, `dd_dgemm_ab_pf`,
  `dd_dgemm_ab_ts_pf`, `dd_dgemm_narrow_pf`, `dd_dgemm_ab_narrow_pf`,
  `dd_dsyrk_pf`) that loads K-tile k+1 from device memory while it computes
  tile k, which hides the load latency where a GPU core holds one or two
  threadgroups. `use_prefetch` in `src/apple_bottom.m` picks it by the
  grid: up to 64 threadgroups for the 64×64 kernels and DSYRK, and for the
  64×64 kernels only in a serial encoder, since concurrent products share
  the GPU; up to 512 for tall-skinny; every grid on the 32×32 tiles.
  Kernel GPU time of one product on M2 Max: 1.43× on 160×160×18432 and
  1.42× on 150×150×18277 (QE si64's overlap), 1.08× on 512×512×18432,
  1.04× on 384×384×18432 and 1.18× on 4096×64×4608; up to 1.26× on 64×64
  grids and 1.47× on tall-skinny ones. ZGEMM's three concurrent products:
  1.01-1.11× on the 32×32 tiles (1.11× at 384×384×18432), unchanged on
  64×64 ones. Through the BLAS ABI's host path, where packing and
  conversion take most of these calls, the gain is the GPU's share:
  `dgemm_` 150×150×18277 and 160×160×18432 run 1.09-1.10× end to end.
  When a host-bound caller leaves the GPU's clock low, mid-size grids gain
  little (within ±6% of the kernel time). For timing comparisons,
  `AB_PREFETCH=0` keeps the plain kernels and `AB_PREFETCH=1` prefetches
  on every grid. Measurements:
  `benchmarks/results/2026-10-24-15ef7b9-prefetch/`. Results are bitwise
  unchanged:
  `tests/test_correctness.c` gains `test_prefetch_same_bits` (79 tests),
  which runs every prefetch kernel (`AB_PREFETCH=1`) against the plain ones.
- **The native ZGEMMs keep their temporaries between calls.** `ab_zgemm`,
  `ab_zgemm_ex`, `ab_zgemm_batched`, `ab_zgemm_async` and `ab_batch_zgemm`
  used to make their four temporaries (Gauss-3M's two products and two
  sums, or 4M's four products) as fresh GPU buffers every call. They now
  take them from the staged path's pool (at most `AB_STAGING_MB`), so a
  call no longer pays for those buffers between commit and the GPU's
  start: on an M2 Max, `ab_zgemm` at M = N = K waited 3.1 ms there at
  2048, 16.6 ms at 4096 and 36 ms at 6144, now 0.2 ms. At 8192 the four
  (2.25 GB with the pool's margin) exceed the default cap and one is
  remade each call: 65 ms → 18 ms, or 0.2 ms with `AB_STAGING_MB=4096`.
  Results are bitwise unchanged. Only these four are pooled: `ab_zgemm_ex`'s
  transposed copies of A and B, the copies that bring real and imaginary
  parts to one scale, and the operands that the host path of the BLAS ABI
  (`AB_HOST_PACK=1`) and `ab_dev_zgemm` copy in are still made every call.

### Fixed
- **Data outside FP32's exponent range no longer gives silently wrong GPU
  results.** DD shares FP32's exponent range: products below about 2⁻⁷⁸
  lost their low part, values below 2⁻¹²⁶ flushed to zero, and sums
  overflowed at 2¹²⁸, all without a warning. Values of 1e±25 gave relative
  errors near 1 or NaN, and an Inf in A gave NaN where cblas gives Inf.
  - **BLAS ABI** (`dgemm_`, `zgemm_`, `ab_*gemm_blas`):
    - Rows of op(A) and columns of op(B) whose largest entry lies outside
      2^±20 are scaled by exact powers of two as they are converted, their
      largest entry to about 2³⁰, which leaves room for both the products
      of smaller entries and the sums; C is scaled back. A DGEMM alpha or
      beta outside 2^±20 joins C's scale.
    - A guard certifies from the scaled data that no sum can overflow and
      every product keeps DD precision. Calls it cannot certify run on
      Accelerate, as do calls with NaN or Inf in A or B (or in C for DGEMM
      with beta ≠ 0). They are counted as a new fallback reason, `range`;
      the `AB_REPORT` fallback line ends `other=… range=…`.
  - **Native `ABMatrix` API**: a matrix carries one power-of-two scale,
    which `ab_matrix_upload` chooses. It is 0 inside 2^±39, and otherwise
    centres entries that span up to 2⁷⁸. The GPU operations carry it and
    `ab_matrix_download` removes it; `ab_dsyrk` keeps the lower triangle of
    C it does not write. The device-buffer API (`ab_dev_dgemm`,
    `ab_dev_zgemm`) converts through `ABMatrix` handles and behaves the same.
  - Data inside those windows computes exactly as before: the golden bits
    of 1.4.0 hold. The new exponent-range suite of `tests/test_blas_abi.c`
    covers dgemm_, zgemm_ and the native GEMMs with 1e±25 data, row and
    column scales from 2⁻¹²⁰ to 2¹²⁰, wide spans, Inf, alpha outside FP32's
    range, and bit-for-bit power-of-two scaling. Four new tests in
    `tests/test_correctness.c` run every native operation that carries a
    scale (the ZGEMM, DGEMM, batch and async entry points, `ab_zgemm_ex`,
    `ab_dsyrk`, `ab_zherk`, add, sub, scale, copy) on data near 2^±73
    whose real and imaginary parts get different scales. Limits:
    `docs/vv/LIMITATIONS.md`, "Exponent range".
- **ZGEMM's imaginary part keeps DD accuracy when real and imaginary parts
  are imbalanced** (audit K-P1-2). Gauss's three-multiplication form
  computes Im(C) as (Ar+Ai)(Br+Bi) − ArBr − AiBi, whose rounding errors
  follow the larger real products. With imaginary parts 10⁻⁶ of the real
  ones, Im(C) missed its componentwise scale by about 10⁶ u², against about
  1 u² with four products.
  - A ZGEMM whose parts give ρ = (1+p)(1+q)/(p+q) above 4, where p and q are
    the parts' magnitude ratios, now runs as four real products (4M). The
    BLAS ABI and the device-buffer API (`ab_dev_zgemm`) take p and q per row
    of op(A) and column of op(B); the native API per uploaded matrix.
  - 4M costs 1.07–1.34× the native call time, 1.05–1.19× through
    `ab_dev_zgemm` and 1.00–1.10× end to end through `zgemm_` (M2 Max).
    Balanced data keeps the three-product form and its bits.
  - `ab_dev_zgemm` gathers the parts as it copies op(A) and op(B) out of
    their buffers, a copy that now runs across cores, as the FP64→DD
    conversion after it already did: on balanced data its calls take 2–6%
    less time than before at QE shapes, and 3% less at 1024³.
  - New tests: `test_blas_abi test_zgemm_imbalance` (6 cases),
    `test_device_api test_zgemm_imbalance` (13 cases) and
    `test_correctness test_imbalance_zgemm_paths`. Limits:
    `docs/vv/LIMITATIONS.md`, "Complex products".

## [1.4.0] - 2026-10-19

### Added
- **`ab_build_info()`** returns the library version and the git commit it
  was built from, with `-dirty` if tracked files differed from it (the
  Makefile passes `AB_BUILD_SHA`; `unknown` outside a git checkout and in a
  copy vendored inside another repository). Every
  `AB_PROFILE` and `AB_REPORT` report starts with it.
- **`AB_REPORT=1`**: a short summary on stderr at exit, once per process
  and tagged with the pid. Per routine: calls and wall time by where C was
  computed, the GPU calls' stage costs (init, alloc, pack, upload, kernel as
  command-buffer GPU time, download, epilogue, other) and fallbacks to cblas
  by reason (`no_device`, `too_large`, `gpu_failed`). See the environment
  variable table in `docs/INTEGRATION.md`.
- **`make calibrate`** measures the BLAS-ABI GPU path against Accelerate and
  derives the `AB_MODE=auto` dispatch table from the result.
  `benchmarks/calibrate.c` times each shape of its grid end to end, through
  `ab_*gemm_blas` with the GPU path forced and through Accelerate
  `cblas_*gemm` in the same process, under `caffeinate -is` (3 warmups,
  median of 7, GPU results checked against cblas). The grid holds the
  Quantum ESPRESSO si64 PAW DGEMMs and Davidson ZGEMMs and square and
  transposed-square DGEMM/ZGEMM from 256 to 6144. It writes the raw and
  summary CSVs and `dispatch_table.txt` under
  `benchmarks/results/<date>-<sha>-calibrate/` and regenerates the built-in
  table, `src/dispatch_table.h`, whose header records the machine, OS, SDK,
  library commit and date.
- **`AB_DISPATCH_TABLE=<path>`** makes `AB_MODE=auto` use a table file, such
  as the `dispatch_table.txt` of a calibration on another machine, instead
  of the built-in table.
- **Quantum ESPRESSO validation kit** (`benchmarks/qe/`): the si64 inputs,
  `run_campaign.sh` (inputs × `AB_MODE`s × repetitions, fresh directories,
  rotated mode order) and `analyze.py` / `compare.py`, which summarize the
  BLAS time by route and the GPU stage costs and compare every energy and
  band energy with the `cpu` run. Plan lines can add pw.x arguments such as
  `-nk 4`; with several ranks the runner caps each rank's OpenBLAS and
  Accelerate threads, and `analyze.py` sums over the ranks. VAL-1 was repeated with it;
  results in `benchmarks/results/2026-10-14-1dbdb52-qe-si64/`. The 1.4.0
  release check on a QE build compiled `-O3`, with 1 and 8 ranks, is in
  `benchmarks/results/2026-10-15-c5a1fa6-qe-si64-opt/`.
- **`make version-check`** (`scripts/check_version.sh`, first step of
  `make ci-local` and of CI) checks that the Python package, CITATION.cff,
  the README citation and the top CHANGELOG entry state the version in
  `include/apple_bottom.h`, which the Makefile now reads for the dylib and
  pkg-config versions.
- **`make test-rectangular`** builds and runs `tests/test_rectangular.c`
  (shapes up to 18277 × 150 × 18277), which no target built before.
- **CI** builds both libraries and runs `make version-check`,
  `make build-check`, `make test-symbols` and the routing unit tests on
  macos-14 and macos-15 (it only compiled the sources before). GitHub's
  runners have no usable Metal GPU, so the GPU suites stay in
  `make ci-local`.

### Fixed
- **Morton remap dropped output tiles on non-square power-of-two grids**
  (K-P0-1). The remap is now applied only when it is a bijection
  (gridW == gridH or gridW == 2·gridH). Before, shapes such as 512×256,
  1024×256 and tall-skinny 1024×64 / 2048×128 returned AB_OK with tiles of C
  unwritten, which also corrupted ZGEMM, the BLAS wrappers, MPIR and DTRSM.
- **`ab_dgemm_async` left kernel buffers 6/7 unbound** (K-P1-3), which
  aborts under Metal API validation. It now shares the dispatch helper with
  `ab_dgemm`, so the two give bitwise-identical results.
- **Fortran `CALL DGEMM` / `CALL ZGEMM` never reached apple-bottom**
  (H-P0-1). Under `ACCELERATE_NEW_LAPACK` the `dgemm_`/`zgemm_` definitions
  were emitted as `_dgemm$NEWLAPACK`/`_zgemm$NEWLAPACK`, so Fortran hosts
  linked with `-lapplebottom` (dylib or static) bound to Accelerate. Both
  libraries now export exactly `_dgemm_` and `_zgemm_` (see Changed for the
  `$NEWLAPACK` names). `make verify-symbols` checks the exact names (it
  accepted `_ab_dgemm_` before), and `make test-fortran` runs a gfortran
  probe.
- **BLAS-ABI calls ran on cblas unless the host called `ab_init()`**
  (H-P1-2). `ab_dgemm_blas`/`ab_zgemm_blas`, and through them
  `dgemm_`/`zgemm_`, now create the Metal context on first GPU use, so
  hosts such as QE reach the GPU without an init call.
- **Dispatch path and AB_PROFILE reported the routing decision, not the path
  taken** (H-P2-8). `ab_get_last_dispatch_path()` says `"gpu"` only
  once the GPU result is in C and `"cpu"` for every cblas run, including
  calls the Fortran bridge keeps on the CPU; the routing no longer sets
  it. The AB_PROFILE table counts a `dgemm_`/`zgemm_` call as GPU only if it
  ran there, and a GPU failure that falls back to cblas prints its reason
  under AB_PROFILE.
- **Lowercase TRANS flags and `'C'` for real data were misread** (K-P0-3,
  H-P1-1). `dgemm_`, `zgemm_`, `ab_dgemm_blas`, `ab_zgemm_blas` and the
  `ab_*gemm_` wrappers compared TRANS with upper-case letters only, and the
  CPU and GPU paths disagreed: `'n'` could run as a transpose or conjugate
  transpose, `'t'` as no transpose or a conjugate transpose, and DGEMM `'C'`
  as no transpose. TRANS is now case-insensitive as in reference BLAS,
  `'C'` means `'T'` for real data, and any other character is rejected by
  the cblas parameter check.
- **Rectangular `'T'`/`'C'` ZGEMM calls returned `beta*C` or read out of
  bounds** (H-P0-2). `ab_zgemm_blas` and `zgemm_` uploaded the transposed
  operands in their stored shape, so the GPU product failed or overran A
  and B; QE's `'C','N'` call in `cegterg` could crash. op(A) and op(B) are
  now packed as M×K and K×N.
- **A failed GPU step in a BLAS-ABI call could leave C wrong** (H-P1-3).
  Allocations and status results on the GPU path were not checked. Every
  step is now checked, and a failure before C is written reruns the whole
  call on cblas. `ab_dgemm`, `ab_dgemm_scaled` and the `ab_zgemm_ex`
  transposes now return `AB_ERROR_KERNEL_FAILED` when the command buffer
  fails.
- **BLAS-ABI calls with `alpha == 0` read A and B** (H-P1-7), so NaN or Inf
  in A or B reached C. They now take the reference BLAS quick return,
  C = beta·C on cblas, without reading A or B; with `beta == 0`, C is never
  read. Results for `alpha == 0` can differ in the last bit from the
  former GPU result.
- **Illegal BLAS arguments could read and write out of bounds on the GPU
  path.** Only the cblas path checked leading dimensions, so an lda, ldb or
  ldc below the operand's row count made `dgemm_`, `zgemm_`,
  `ab_*gemm_blas` and `ab_*gemm_` index past the caller's arrays; with lazy
  init this became the default for hosts such as QE. Calls with an invalid
  TRANS, a negative M, N or K, or a short leading dimension now always go
  to cblas, which names the bad parameter and stops the program as
  reference BLAS (`xerbla`) does, whatever `AB_MODE` says.
- **A `fork()`ed child crashed on its first GPU-routed BLAS call.** Metal
  does not survive `fork()`, and since lazy init every BLAS-ABI host that
  forks after a GPU call was exposed. The first `ab_init()` now registers a
  fork handler. In the child, `ab_init()` returns `AB_ERROR_NO_DEVICE`,
  `ab_is_initialized()` is false, native GEMMs return `AB_ERROR_NO_DEVICE`,
  and GPU-routed BLAS calls run on cblas (dispatch path `"cpu"`).
- **Parallel `ab_matrix_upload`/`ab_matrix_download` ran at ~125 MB/s**
  (H-P0-3). The FP64↔DD conversion used one GCD block per element, which
  slowed every BLAS-ABI call, DTRSM, MPIR and device-API transfer. It now
  converts in a few contiguous chunks, with bit-identical results. On an
  M2 Max at N=2048: upload 122 MB/s → 72 GB/s, download 123 MB/s → 59 GB/s,
  and an `ab_dgemm_blas` `'N','N'` call 862 ms → 94 ms.
- **A memory-pool matrix reused for a smaller shape kept its old element
  count** (K-P1-4), so `ab_matrix_upload`/`ab_matrix_download` read or
  wrote past the caller's buffer. Reuse now resets the count; the buffer
  capacity is tracked separately.
- **GPU calls leaked about 170 bytes each in programs without an
  autorelease pool**, such as Fortran and C hosts. Every entry point that
  creates Metal command buffers or encoders now drains its own pool, so
  memory stays flat.
- **Shaders lost DD precision before macOS 15** (K-P1-6). Without
  `MTLMathModeSafe` the library compiled them with fast math, which gives
  FP32-level errors. It now sets `fastMathEnabled = NO` there, which keeps
  full DD precision, and keeps `MTLMathModeSafe` on macOS 15 and later. The
  choice is made with a run-time selector check, so a static archive built
  for an older deployment target still links with gfortran (no compiler-rt
  `___isPlatformVersionAtLeast`).
- **`ab_dev_dgemm` / `ab_dev_zgemm` ignored their column-major contract**
  (K-P0-2, H-P0-4). Column-major buffers were read as row-major, so square
  inputs returned B·A, transpose flags were ignored or returned -3, and other
  shapes were wrong. Both now compute Cᵀ = op(B)ᵀ·op(A)ᵀ on the transposed
  operands and honour transA/transB, ld and offsets. Invalid flags, short
  leading dimensions and operands that run past their buffer return
  `AB_ERROR_INVALID_ARG`. `test_device_api` compared against the row-major
  API, which enforced the bug; it now checks against column-major cblas.
- **The Python bindings did not work** (H-P1-5). `ab.init()` and every
  status check raised `TypeError`, and matrices from
  `MemoryPool.get_matrix()` or `Session.get()` were freed twice, which
  crashed. Every handle is now freed exactly once. Borrowed matrices keep
  their pool or session alive and raise `InvalidArgumentError` once it is
  closed, or once their pool is `reset()`: the pool may hand the handle out
  again with another shape, and a stale wrapper's `download()` would
  overflow its buffer. numpy is optional, `AB_LIBRARY` selects the dylib,
  and `make test-python` runs the binding tests. The bindings remain
  experimental.
- **Matrices with more than 46,340 rows or columns never reached the GPU.**
  The limit was 46,340 per dimension, a square-matrix stand-in for 2³¹
  elements. In Quantum ESPRESSO Si64 (PAW, 2 MPI ranks) it rejected only
  the PAW DGEMMs in `newq` (called from `newd`) and `addusdens`,
  171 × 64 × 2·ngm_l and 2·ngm_l × 171 × 64: 171 = nh(nh+1)/2 projector
  pairs (nh = 18 per Si atom), 64 = the atoms of that species, and
  2·ngm_l = 71,834 (gamma point) or 143,666 (k-points) = the rank's
  dense-grid G-vectors, complex values passed as real pairs. Those
  BLAS-ABI calls always ran on cblas; the Davidson GEMMs (`regterg`,
  `cegterg`; long dimension ≈ 9,000 per rank) were never affected. This
  is a robustness fix, not a speed-up: the PAW DGEMMs run at 0.09–0.15×
  Accelerate's speed on the GPU, and calibrated `AB_MODE=auto` routing
  keeps them on Accelerate. The limit is now what actually overflows, for any shape:
  rows × cols ≤ `AB_MAX_MATRIX_ELEMENTS` (2³¹ − 1; the kernels index in
  32-bit unsigned arithmetic) and the DD buffer within the device's
  `maxBufferLength`. Every kernel was checked bit-exact on matrices past
  2²⁹ elements (byte offsets above 4 GiB).
  Beyond the limit, `ab_matrix_create` returns NULL; the new
  `ab_matrix_check_size()`, `ab_session_add` and `ab_dev_*gemm` return
  the new `AB_ERROR_TOO_LARGE` (`ab_dev_*gemm` returned
  `AB_ERROR_ALLOC_FAILED`); BLAS-ABI calls run on cblas, and `AB_PROFILE=1`
  prints why. In Python, `MAX_ELEMENTS` (2³¹ − 1) bounds rows × cols and
  `MAX_DIMENSION` is now 2³¹ − 1.
- **The first GPU call left `FE_OVERFLOW` raised in the caller.**
  `MTLCreateSystemDefaultDevice()` raises it on its first call in a
  process (macOS 27), so every Quantum ESPRESSO run that reached the GPU
  ended with gfortran's `IEEE_OVERFLOW_FLAG is signalling` note.
  `ab_init()` and the device API now restore the caller's floating-point
  environment around device creation (`test_init_keeps_host_fp_flags`).
- **`AB_PROFILE` stage lines could interleave** when threads called BLAS
  concurrently; each line is now written with one call.
- **Build tooling.** `make build-check` wrote its objects over the
  library's (a later `make lib` then reported the build as `unknown`) and
  skipped `src/device_api.m`; `make test-asan`/`test-ubsan` could not
  build; a copy vendored inside another git repository stamped that
  repository's commit as its build SHA; `make calibrate` with `--only`
  replaced the built-in table with one derived from part of the grid.
- **`tests/test_rectangular.c` failed correct results.** It compared
  elementwise relative errors with 1e-14, which cancellation defeats; it now
  measures componentwise error against |op(A)| |B| and runs the
  conjugate-transpose case it used to skip.

### Changed
- **`AB_MODE=auto`, the default, is calibrated from measured end-to-end
  times.** A BLAS-ABI call (`dgemm_`, `zgemm_`, `ab_*gemm_`,
  `ab_*gemm_blas`) now goes to the GPU only if a row of the dispatch table
  covers it: a shape the GPU was measured at least 1.10× faster on, with no
  slower shape of the same routine and TRANS class at least as large in M,
  N and K. Before, fixed FLOP thresholds (10⁸ complex, 5·10⁷ real, applied
  differently by the Fortran bridge and `ab_*gemm_blas`) sent most large
  calls to the GPU, including Quantum ESPRESSO's Davidson ZGEMMs (VAL-1
  input A: ZGEMM time 57.9 s on Accelerate, 144.6 s with the old thresholds,
  which sent the same 951 large ZGEMMs to the GPU). On the M2 Max calibration
  (`benchmarks/results/2026-10-14-52f1880-calibrate/`) the GPU was at least
  1.10× faster only for ZGEMM with both operands transposed or conjugated
  and M, N, K ≥ 4096 (1.11× at 4096, 1.20× at 6144), the table's one row.
  Every QE si64 shape (0.09–0.76× Accelerate's speed) and every DGEMM up to
  6144 (at most 0.84×) is slower on the GPU and now stays on Accelerate.
  One module (`src/dispatch.c`) routes every entry point; precedence is in
  `docs/INTEGRATION.md`. `AB_MODE=cpu`/`gpu`, `AB_MIN_GPU_DIM` and the
  native `ABMatrix` and device APIs are unchanged.
- **`AB_CROSSOVER_FLOPS` and `AB_CROSSOVER_FLOPS_REAL` now replace the
  table** and are unset by default. When set, `auto` sends a call with at
  least that many FLOPs to the GPU: `AB_CROSSOVER_FLOPS` for ZGEMM, and for
  DGEMM unless `AB_CROSSOVER_FLOPS_REAL` is set. Before, they adjusted
  default thresholds, `AB_CROSSOVER_FLOPS_REAL` only for `ab_dgemm_blas`
  calls with every dimension ≥ 64.
- **The AB_PROFILE table separates call share from time share.** Its
  `gpu_pct` column was the share of calls that ran on the GPU, not of time,
  and was read as a time share. It is replaced by `gpu_calls`, `gpu_calls%`,
  `gpu_ms`, `cpu_ms`, `gpu_time%` and `fallbacks`, followed by tables of the
  GPU stage costs and of the fallbacks by reason. Every GPU call now prints
  a stage line, DGEMM included (it was ZGEMM only), with the stages named as
  in `AB_REPORT`. Direct `ab_dgemm_blas` / `ab_zgemm_blas` calls are now
  counted too (only `dgemm_`, `zgemm_` and `ab_*gemm_` were). A `fork()`ed
  child starts with empty counters, so it no longer reports the parent's
  calls as its own. In the `AB_PROFILE_JSON` file, `gpu_dispatch_count` and
  `amx_fallback_count` are now `gpu_calls` and `cpu_calls`, next to
  `gpu_ns`, `cpu_ns`, `gpu_stage_ns`, `fallbacks` and the top-level `build`.
- **`_dgemm$NEWLAPACK` / `_zgemm$NEWLAPACK` are no longer exported.** C code
  that calls `dgemm_`/`zgemm_` through the `ACCELERATE_NEW_LAPACK` headers
  now gets Accelerate, not apple-bottom; call `ab_dgemm_blas` /
  `ab_zgemm_blas` to use the GPU. Binaries already linked against the old
  dylib still resolve those names through the Accelerate re-export.
- **`ab_dev_dgemm` / `ab_dev_zgemm` now read column-major buffers as
  documented** (see Fixed). Code that swapped or transposed operands to work
  around the old row-major reading gets different results and must drop
  that workaround.
- **The precision tests fail when DD precision is lost** (K-P1-5).
  `test_precision` measures normwise and componentwise error against a
  compensated FP64 reference and also covers the scaled, tall-skinny and
  ZGEMM kernels (8 → 14 cases). The V-2 convergence study fails when the
  Frobenius error exceeds √N·u². A build without safe math now fails both.
- **`make ci-local` runs `test_device_api` and counts tests one way**
  (RH-P1-1). `TEST_TOTAL` counts tests (precision cases, `TESTS[]` entries
  and BLAS-ABI suites: 156), with device-API `CHECK`s and BLAS-ABI cases on
  their own lines. `test_convergence` takes its CSV path as `argv[1]`, so
  `make ci-local BUILD=...` reports the convergence data instead of
  `missing`.

### Removed
- `CMakeLists.txt`. It built only `src/apple_bottom.m`, a library without
  the BLAS ABI, routing, device API or profiler; the Makefile is the
  supported build.
- Internal review notes (`docs/review/`), `python/INDEX.md` (a copy of the
  Python README), two committed test binaries and two scratch test sources
  that no target built.

### Corrected claims
Earlier releases and documents made claims that measurement does not
support. They are withdrawn, and the documents now say what was measured:
- **"22% faster" / "1.22× vs 6-thread OpenBLAS" / "+12% (Si128)" on Quantum
  ESPRESSO** (README, INTEGRATION, VV_REPORT, VAL-1, blog). Versions
  1.0–1.3 needed an explicit `ab_init()`, which no QE build called, so those
  runs never used the GPU, and the 22% compared a serial QE build with an
  MPI build. Repeated with the GPU path working (VAL-1, revision 2), forcing
  the GPU makes Si64 runs 1–17% slower on an M2 Max (QE itself compiled at
  `-O0`), and `AB_MODE=auto`
  keeps QE's GEMMs on Accelerate at no cost.
- **"11 decimal places of agreement"** with QE. The March runs compared CPU
  with CPU. With the calls that make up 89–99% of the GEMM work on the GPU,
  the energies and band energies of VAL-1 revision 2 agree with the CPU run
  in every printed digit (10⁻⁸ Ry, 10⁻⁴ eV).
- **"2.7× speedup"** (CITATION.cff, rectangular-tiling note). No measurement
  supports it.
- **"618 GFLOP/s" / "643 GFLOP/s"** headline DGEMM figures. They were
  steady-state figures with matrices resident on the GPU, not BLAS-call
  throughput, and are replaced by dated measurements of both regimes in the
  README.
- **"~24 GFLOP/s" for metal-float64 and "26× faster" than it** (README,
  COMPARISON, blog). metal-float64 is an unfinished scalar library with no
  GEMM; there is nothing to benchmark against.
- **DSYRK through the Fortran ABI**, **SGEMM/TGEMM coverage** and
  **multi-GPU** claims (INTEGRATION, COMPARISON): the library exports only
  `dgemm_`/`zgemm_` (and `ab_dgemm_`/`ab_zgemm_`) to Fortran, and has no
  SGEMM, TGEMM or multi-GPU support.

## [1.3.0] - 2026-04-18

### Added (Phase 6)
- **Fused ZGEMM** — single command buffer with 4 encoders (was 8 separate round-trips)
  - Encoder boundaries enforce GPU barriers for the Gauss-trick dependency DAG
  - Eliminates ~350μs of per-call command buffer commit+wait overhead
  - Same precision (DD ~10⁻¹⁵) with lower dispatch latency
- **True async ZGEMM** (`ab_zgemm_async`) — non-blocking GPU execution
  - Metal completion handler cleans up temporaries when GPU finishes
  - Returns `ABFuture` that can be polled or waited on
  - Replaces previous synchronous wrapper implementation
- **Batched GEMM API** (`ABBatch`) — amortizes Metal overhead across many GEMMs
  - `ab_batch_create/dgemm/dgemm_scaled/zgemm/barrier/commit/wait/destroy`
  - Single command buffer for hundreds of GEMMs (QE fires 100s per SCF iteration)
  - `ab_batch_barrier()` inserts encoder boundary for dependent operations
  - Batched ZGEMM uses same fused 4-encoder pattern within the batch
  - 8 new correctness tests (batch DGEMM, multi-DGEMM, barrier, null safety,
    batch ZGEMM, committed reuse, fused ZGEMM precision, true async ZGEMM)
- **Performance regression CI** (`scripts/perf_regression.sh`)
  - Automated benchmark suite with baseline comparison
  - Configurable regression threshold (default 15%)
  - JSON baseline storage, markdown regression report
  - `--save` to capture baseline, `--ci` for non-zero exit on regression
  - Benchmarks: DGEMM (square + tall-skinny), ZGEMM, batched DGEMM

### Added (Tier 2)
- **Mixed-Precision Iterative Refinement** (`ab_dgesv_mpir`) — solves A*X=B via MPIR
  - FP32 LU factorization (Accelerate LAPACK sgetrf/sgetrs) for O(N³) bulk work
  - DD-DGEMM residual computation (10⁻¹⁵ fidelity) catches cancellation errors
  - Converges to DD precision in 1-3 iterations for well-conditioned matrices
  - Avoids explicit inversion (κ² error) and bespoke DD-DTRSM shaders (Paper §6)
  - 6 new correctness tests (basic, single-RHS, moderate-κ, null, dim-mismatch, non-square)
- **Broadened tall-skinny DGEMM dispatch** for domain-science matrices
  - Relaxed heuristic from `(M > 4*N) && (N <= 64)` to pure aspect-ratio `(M >= 4*N)`
  - Catches QE hot-path shapes like M=18277, N=150 that previously fell through to
    64×64 square kernel (~34% boundary waste → ~4% with 128×16 tiles) (Paper §5)

### Added
- **DTRSM**: Triangular solve (op(A) * X = alpha * B) — blocked forward/back-substitution with DGEMM panel updates
  - New enums: `ABSide`, `ABUplo`, `ABDiag` for BLAS-standard calling convention
  - GPU-efficient for N >= 1024
- **Python Bindings** (`python/` directory)
  - `ctypes` wrapper with Pythonic API: `ab.dgemm(A, B)` on numpy arrays
  - High-level classes: `Matrix`, `Session`, `MemoryPool`
  - Functional one-liner API: `dgemm()`, `zgemm()`, `dsyrk()`
  - Context managers and automatic cleanup
  - Full type hints (PEP 484 + PEP 561)
- **Benchmark Report Script** (`scripts/bench_report.sh`)
  - Automated benchmark suite with formatted markdown output
  - `make bench-report` target
  - System info collection, `--quick` mode
- **Competitor Comparison** (`docs/COMPARISON.md`)
  - Detailed technical comparison vs metal-float64, MLX, Accelerate, AppleNumericalComputing
  - Decision guide for when to use each approach
- **Blog Post Draft** (`docs/blog/fp64-on-apple-silicon.md`)
  - HN/Reddit-ready technical writeup
- **Dynamic Wilkinson-scaled error thresholds** in precision tests
  - Threshold: `C_safety × sqrt(K) × u_DD` instead of static `1e-14`
  - Mathematically correct per Higham (2002) §3.1 probabilistic analysis
  - Precision tests now include rectangular (K=4096) and tall-skinny shapes
  - Test count: 6 → 8 precision tests
- **GPU-native transpose DGEMM** in BLAS wrapper
  - `transA='T'` and `transB='T'` now route to GPU instead of CPU fallback
  - Upload-time reordering handles column-major transpose conversion
  - Improves GPU call ratio in QE workloads (was ~85%, now higher)
- **Block-wise compensated accumulation** in GEMM kernels
  - Periodic `twoSum` renormalization every 128 K-elements (8 tiles)
  - Final renormalization before epilogue alpha/beta scaling
  - Prevents accumulator drift for large-K dot products
  - Applied to all GEMM kernels: dd_dgemm, dd_dgemm_ab, dd_dgemm_ab_ts, dd_dsyrk
- **Morton Z-order threadgroup dispatch** for SLC cache locality
  - Space-filling curve remaps 2D threadgroup IDs to improve L2/SLC hit rate
  - Applied to all GPU kernels: dd_dgemm, dd_dgemm_ab, dd_dgemm_ab_ts, dd_dsyrk
  - Inline `morton_remap` function with 16-bit deinterleave + bounds checking
  - Fallback to identity mapping when coordinates exceed grid dimensions
- **DTRSM implementation** (blocked forward/back-substitution)
  - Left-side: blocked panel algorithm with NB=64
  - CPU (AMX) solves small triangular blocks, GPU DGEMM updates remaining panels
  - Supports upper/lower, no-trans/trans, unit/non-unit diagonal
  - Right-side falls back to cblas_dtrsm (AMX) for now
  - Alpha scaling support
  - 6 new correctness tests (lower, upper, trans, alpha, large N=256, null safety)
- **AMX heterogeneous dispatch** for small GEMMs
  - Dimension-aware heuristic: matrices with any dim ≤ 32 always route to AMX
  - Separate thresholds for real (50M FLOPs) vs complex (100M FLOPs) DGEMM
  - Skinny matrix penalty: higher threshold when any dimension < 64
  - Reduces GPU dispatch overhead for QE's many small ZGEMM calls

### Fixed
- **BUG-8: ARC leak in Metal-ref structs** — `ABMatrix_s`, `ABFuture_s`, `ABBatch_s`, and `ab_dev_buffer_s` now allocate via `new`/`delete` so ARC destructors release their `id<MTL…>` strong refs (was `calloc`/`free`, which does not run ARC destructors and leaked the buffers, command buffers, and encoders). Landed in 8c18ef3.

### Changed
- **README.md**: Complete rewrite as high-conversion landing page
  - One-line pitch above the fold with benchmark numbers
  - Performance comparison table (DD-GPU vs AMX-CPU)
  - Streamlined quickstart and API reference
  - Prior art comparison table
  - CI badge from GitHub Actions
- **Citation URLs**: Fixed TechnologyResidue → grantdh/apple-bottom in research/td-dgemm/README.md
- **Version**: Bumped to 1.3.0-dev

### Notes
- Test count: 50 → 70 (8 precision + 62 correctness)
- DTRSM right-side GPU acceleration planned for v1.4
- Rectangular matrix fix in progress (fix/rectangular-gemm branch)

## [1.2.0] - 2026-04-02

### Added
- **V&V Documentation Package** (NASA-STD-7009A compliant)
  - VV_REPORT.md: Master validation document with traceability matrix
  - PRECISION_ENVELOPE.md: Precision guarantees and validated envelope
  - VAL-1: Quantum ESPRESSO Si64 production validation
- **CI Pipeline** (GitHub Actions)
  - Automated compile-only checks on push/PR
  - Local validation script with convergence study
- **API Documentation**
  - Version alignment: 1.0.0 → 1.2.0 in all headers
  - API limits: AB_MAX_DIMENSION=46340, pool capacity 128, session capacity 64
  - Async ZGEMM implementation note (synchronous wrapper, true async planned for v2.0)
  - Precision scaling clarification: per-element ~10⁻¹⁵, DGEMM Frobenius ~N×10⁻¹⁵

### Fixed (Critical)
- **BUG-1/BUG-2**: Async DGEMM dimension packing and pipeline selection
- **BUG-3**: ab_dgemm_scaled alpha/beta truncation to float (precision loss)
- **BUG-4**: ab_matrix_scale alpha truncation to float (precision loss)
- **BUG-5**: dispatch_once_t reset undefined behavior (V&V audit flag)
- **BUG-6**: Memory pool overflow leak (returned unmanaged matrices)
- **BUG-7**: Missing pipeline creation error checks (crash on init failure)
- **ab_zherk**: Added error checking for ab_dgemm/ab_matrix_add/ab_matrix_sub with goto cleanup

### Changed
- **Integration Documentation**: Fixed INTEGRATION.md examples and cleanup
- **MTLMathModeSafe**: SDK compatibility (KVC pattern, no SDK dependency)
- **Build System**: Removed phantom CMake targets (cg_solver, zgemm_example, eigenvalue_solver)
- **Build System**: Commented out pkg-config section pending cmake/applebottom.pc.in creation
- **README**: Added precision scaling clarification (Frobenius error ~N×10⁻¹⁵)
- Test badge: 37 → 48 passing tests (6 precision + 42 correctness)

### Notes
- Validated baseline: Git tag v1.0.2-bugfix
- Production-validated for DFT, MD, FEM iterative solvers
- Version alignment across all components (headers, CMake, API)

## [1.0.2] - 2026-03-31

### Fixed (Critical)
- **BUG-1/BUG-2**: Async DGEMM dimension packing and pipeline selection (ship-blocker)
  - Fixed dimension packing: separate `setBytes` calls instead of array
  - Always use `dgemmPipeline` (async path doesn't support alpha/beta)
- **BUG-3**: `ab_dgemm_scaled` alpha/beta truncation to float (precision loss)
  - Convert alpha/beta to DD format on host before kernel dispatch
  - Metal kernel now accepts `constant DD&` instead of `constant float&`
- **BUG-4**: `ab_matrix_scale` alpha truncation to float (precision loss)
  - Convert alpha to DD format on host
- **BUG-5**: `dispatch_once_t` reset undefined behavior (V&V audit flag)
  - Replaced with `os_unfair_lock` and bool flag for reinit safety
- **BUG-6**: Memory pool overflow leak (returned unmanaged matrices)
  - Now returns NULL when pool is full instead of leaking
- **BUG-7**: Missing pipeline creation error checks (crash on init failure)
  - Added error checks for all 10 pipeline creations

### Added
- **V&V Documentation** (NASA-STD-7009A compliant)
  - `docs/vv/VV_REPORT.md`: Master validation report with traceability matrix
  - `docs/vv/PRECISION_ENVELOPE.md`: Precision guarantees and validated envelope
  - `tests/validation/VAL001_QE_Si64.md`: QE Si64 production validation
- **Verification Tests**
  - `tests/verification/test_convergence.c`: V-2 convergence study (N ∈ {64..4096})
  - 5 regression tests for BUG-1 through BUG-6
- **Test Coverage**: 42 tests (37 original + 5 regression)

### Changed
- Test badge: 37 → 42 passing tests
- Tagline: "FP64-class BLAS for scientific computing" (general-purpose)
- README: Added Verification & Validation section

### Validation Results
- **Frobenius error**: 6.5×10⁻¹⁵ to 5.1×10⁻¹⁴ (N ≤ 4096)
- **Max element error**: 3.4×10⁻¹¹ to 6.5×10⁻⁶ (documented for element-sensitive apps)
- **QE Si64 validation**: 11 decimal place agreement (-2990.44276157 Ry)
- **Performance**: 1.22× speedup vs 6-thread OpenBLAS on QE benchmark
- **Status**: VALIDATED for production DFT, MD, FEM iterative solvers

### Notes
- Validated baseline: Git tag `v1.0.2-bugfix` (SHA: f3e2fc1)
- All precision claims backed by empirical convergence data
- Known limitation: Rectangular matrices (aspect ratio >10:1) fail correctness

## [1.1.0] - 2026-03-23

### Added
- `ABTranspose` enum for matrix transpose operations (NO_TRANS, TRANS, CONJ_TRANS)
- `ab_zgemm_ex()`: Extended ZGEMM with transpose support for Quantum ESPRESSO compatibility
- GPU kernels for conjugate transpose and regular transpose operations
- Test coverage for conjugate transpose pattern (37 total tests)

### Changed
- `ab_zgemm()` now wraps `ab_zgemm_ex()` with NO_TRANS for both matrices (backward compatible)

### Deprecated
- `ab_zherk()`: Marked deprecated - CPU decomposition is 20x slower than AMX
  - Use `cblas_zherk()` from Accelerate framework instead
  - Will be removed in v2.0.0

### Technical Details
- New GPU kernels: `dd_transpose`, `dd_conj_transpose`
- Transpose pattern enables QE calbec optimization: C = A^H × B
- Maintains ~10⁻¹⁵ precision for all transpose variants

## [1.0.0] - 2026-03-22

### Added
- Initial release
- DGEMM: Double-precision general matrix multiply (618 GFLOP/s on M2 Max)
- ZGEMM: Complex double-precision matrix multiply (Gauss algorithm)
- DSYRK: Symmetric rank-k update
- ZHERK: Hermitian rank-k update
- Session API for named matrix management
- Thread-safe statistics tracking
- 22-test correctness suite
- Precision validation (~10⁻¹⁵ Frobenius error)

### Technical Details
- Double-float (FP32×2) emulation using Dekker/Knuth algorithms
- Tiled GEMM kernel: BM=BN=64, TM=TN=4, TK=16
- MTLMathModeSafe for precision guarantees
- Parallel FP64↔DD conversion for large matrices
