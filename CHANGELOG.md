# Changelog

All notable changes to this project will be documented in this file.

## [1.6.0] - Unreleased

### Added
- **Kernel dispatch counts in the tests.** The library counts every
  GEMM-family dispatch by kernel and by plain or register-prefetch instance
  where it binds the pipeline (`gemm_pipeline`), and `tests/test_kernels.h`
  reads the counts. The coverage cases of `test_correctness`,
  `test_precision` and `test_blas_abi` name the kernel they are for and check
  that it ran, so a routing change that moves one fails the test instead of
  leaving a comment wrong.
- **Routing tables per chip.** `AB_MODE=auto` picks the built-in table
  calibrated on this machine's chip (`machdep.cpu.brand_string`, which is
  also the Metal device name) and GPU core count (the IORegistry's
  `gpu-core-count`) from `src/dispatch_table.h`, which now holds one table
  per calibrated chip. A chip without a table, or a sibling with another
  core count (an M2 Max with 30 cores instead of 38), gets no rows: every
  `auto` call runs on Accelerate until that chip is calibrated, instead of
  being routed by the M2 Max's measurements. `AB_CHIP` and `AB_GPU_CORES`
  override the detected identity. The tables live in
  `benchmarks/calibration/<chip>/` (`M2-Max-38c`, the former single
  built-in table, from `benchmarks/results/2026-10-14-52f1880-calibrate/`);
  `make calibrate` files this machine's result there through the new
  `scripts/ingest_calibration.py`, which also files a calibration made on
  another machine, checks a table against its measurements and refuses one
  from part of the grid, and `make dispatch-table-check` (in `ci-local` and
  CI) verifies the header matches the filed tables. `dispatch_table.txt`
  carries `# chip:`, `# gpu_cores:` and `# cpu_matrix:` (`AMX` or `SME`)
  header keys, of which the loader reads back the chip and core count: an
  `AB_DISPATCH_TABLE` from another chip is used with a note on stderr.
  `AB_REPORT`'s first line and the `AB_TRACE` header name the chip and the
  table a run used (`table=builtin:M2-Max-38c`, `file:<path>` or
  `none:uncalibrated-chip`).
  The library and `test_dispatch` now link IOKit (a static link adds
  `-framework IOKit`; pkg-config carries it); `test_dispatch` checks
  every built-in table against its own measurements (23 tests, were 9;
  `test_split` has 12).

- **BLAS-2 on the GPU: DGEMV, ZGEMV (`'N'`, `'T'`, `'C'`) and ZHEMV**
  (`src/blas2.m`, `src/blas2_abi.c`). The kernels read the caller's
  column-major FP64 matrix in place, convert each element to double-float
  in registers and accumulate in double-float, adding partial sums in an
  order fixed by the shape (bitwise reproducible); ZHEMV reads its stored
  triangle once, each element feeding its row's and, conjugated, its
  column's sum. A matrix-vector product is bound by memory bandwidth,
  where the GPU leads the CPU; see "BLAS-2" in README.md for the measured
  rates.
  - `ab_dev_register(ptr, nbytes)`: a device buffer over memory the caller
    already owns, without a copy (a no-copy `MTLBuffer` over the
    page-rounded span; NULL if part of it is unmapped or unreadable).
    Every `ab_dev_*` call works on it. Free the handle before the memory.
  - `ab_dev_dgemv`, `ab_dev_zgemv`, `ab_dev_zhemv`: products with a matrix
    kept in a device buffer, for solvers that use one matrix many times
    (Lanczos, Haydock, other Krylov methods).
  - `ab_dgemv_blas`, `ab_zgemv_blas`, `ab_zhemv_blas` and the Fortran
    symbols `dgemv_`, `zgemv_`, `zhemv_`, routed like GEMM (`AB_MODE`,
    then `AB_MIN_GPU_DIM`, through `ab_dispatch_mode_route()`), except that a
    call with M or N below `AB_BLAS2_MIN_DIM` (default 256) stays on Accelerate
    in every mode:
    `AB_MODE=auto` sends a call to the GPU only on the M2 Max (38-core
    GPU) it was measured on, matched as the GEMM tables are, and only for
    shapes measured at least 1.10× faster end to end in two runs, mapping
    the matrix for the call included (DGEMV `'N'` 2048²–16384² elements, ZGEMV `'N'`
    3072²–12288², ZHEMV 2048–16384; data in
    `benchmarks/results/2026-10-19-ba345df-gemv/` and, for ZHEMV,
    `2026-10-19-1bc503a-zhemv/`). Smaller calls, DGEMV `'T'`/`'C'`
    (0.94–1.40×), ZGEMV `'T'`/`'C'` and calls beyond the measured sizes
    stay on Accelerate. Counted in `AB_REPORT` and `AB_PROFILE`, with no
    `staged=` field; not traced.
  - A range guard computed by the kernels from the FP64 exponents, at no
    extra memory traffic. A call it refuses runs on Accelerate, before y is
    written, counted as a `range` fallback (an A that cannot be mapped for
    the GPU counts as `other`). Each part of a complex output is held to its
    own products, so a part far below the other sends the call to Accelerate
    instead of losing its bits. Rule and limits:
    [Exponent range](docs/vv/LIMITATIONS.md#exponent-range).
  - `tests/test_blas2.c` (19 tests, 995 cases; in `make test` and
    `make ci-local`, including a check of the `AB_MODE=auto` rule against
    its measurements and of the kernels' FP64-to-double-float conversion,
    bit for bit, on 4.2 million inputs) and `benchmarks/bench_gemv.c`
    (`make bench-gemv`; `--sweep` and `--rect` for the routing data). The
    Fortran binding probe (`make test-fortran`) also calls `DGEMV`, `ZGEMV`
    and `ZHEMV`, and `make test-symbols` checks `_dgemv_`, `_zgemv_` and
    `_zhemv_`. `ab_get_last_dispatch_path()` also reports BLAS-2 calls.

- **Double-float FFT** (`include/apple_bottom_fft.h`, `src/fft.m`,
  `src/fft/`): batched 1D, 2D and 3D complex transforms on interleaved FP64
  data, computed on the GPU in double-float. Column-major layout (`n[0]`
  contiguous, as Fortran and Quantum ESPRESSO), FFTW advanced-interface
  strides, distances and embeddings, forward or backward, and the scale (1,
  1/N or 1/√N) chosen per plan. Lengths with factors 2, 3, 5, 7, 11 and 13,
  up to 2048 per dimension and up to 2048² in 1D (four-step). Each axis is
  one GPU pass of Stockham line FFTs in threadgroup memory, adapted from the
  Quantum ESPRESSO band-FFT library (libqemetal); the first pass converts the
  FP64 input on the GPU, the last rounds to FP64 exactly with integer
  arithmetic, without a software FP64 add. Caller arrays go through staging
  buffers the plan reuses, in 32 MB chunks that overlap the GPU work;
  `ab_fft_malloc` / `ab_fft_register` memory is used in place. Relative L2
  error 5.9–7.5 × 10⁻¹⁵ on 48³–128³ grids, 20–24× FFTW's FP64 error. Against
  Homebrew FFTW with `FFTW_PATIENT` plans on an M2 Max: in batches of 128,
  1.5–2.4× faster than eight cores staged and 1.9–3.2× with zero copy (the
  medians of five runs); one transform at a time, faster than one thread
  everywhere and than eight threads from 96³ with zero copy. See
  "Double-float FFT" in `README.md`,
  `benchmarks/results/2026-10-23-c138543-fft/` and
  `benchmarks/results/2026-10-29-9fc4547-fft/`.
- **`tests/test_fft.c`** (14 tests, in `make test` and `make ci-local`):
  every result against a double-double reference DFT (`tests/fft_ref.h`)
  and, bit for bit, against a CPU emulator of the GPU arithmetic
  (`tests/fft_emu.cpp`); see "Double-float FFT" in `tests/README.md`.
- **`benchmarks/bench_fft.c`** (`make bench-fft`): time per transform
  against FFTW (1 thread, an 8-thread plan, 8 single-thread workers) and
  vDSP, and accuracy against the double-double reference.

- **`AB_PRECISION=fast`** (opt-in, read when the library initializes, every
  API): the GEMM kernels accumulate lazily (Dot2-style). Each product's high
  part goes into a running sum through TwoSum and every rounding error into a
  separate running error, renormalized once per 16 terms of K. About 1 bit
  less accurate: on `tests/test_precision.c` the normwise error is 1.8–1.9×
  and the componentwise error 1.4–2.8× the default's; its new last case
  re-initializes in fast mode, runs every case again and fails one above 4×.
  The proven error bounds (`docs/vv/PRECISION_ENVELOPE.md` §1.2) cover the
  default kernels only; none is proven for this mode. On an M2 Max, 1.05–1.08×
  faster for DGEMM and ZGEMM of N = 2048 and 4096
  ([data](benchmarks/results/2026-11-06-8b48826-pretag/README.md#ab_precisionfast)). `ab_precision_mode()` returns the mode,
  the `AB_REPORT`, `AB_PROFILE` and `AB_TRACE` headers add
  `AB_PRECISION=fast` in fast mode, and `make calibrate` refuses it. The
  default kernels are compiled as before and keep the 1.4.0 bits
  (`test_qe_golden`).
- **Opt-in Strassen-Winograd products** (`docs/STRASSEN.md`):
  `ab_dgemm_strassen` and `ab_zgemm_strassen`, and `AB_STRASSEN=1`, under
  which a GPU GEMM with M, N and K all at least `AB_STRASSEN_MIN` (default
  4096) takes each real product by one level of Winograd's variant: 7
  half-size products on the existing DD GEMM kernels and 15 block additions,
  for any shape (odd dimensions are zero-padded).
  - It takes the place of the product inside the GEMM, so the GEMM keeps
    each matrix's power-of-two scale, its choice between Gauss-3M and 4M
    and, in the BLAS ABI, the staged path. The native GEMMs, the BLAS ABI
    (staged and host path) and the device API are routed. Not routed: the
    batched, asynchronous and `ab_batch_*` GEMMs, the library's own uses of
    its GEMMs (ZHERK, DTRSM, `ab_dgesv_mpir`), and BLAS-ABI calls with a row
    of op(A) or column of op(B) outside 2^±20: the guard scales those line
    by line, and Strassen-Winograd's block sums would add lines on scales up
    to 2^50 apart.
  - Its workspace (1.75 N² DD values for a square N) comes from the staging
    pool, so calls on different threads run side by side; the staged path
    reserves it with the call's other buffers.
  - The error bound is normwise only: rows or columns much smaller than the
    rest lose relative accuracy. With `AB_STRASSEN` unset every result is
    bitwise unchanged.
  - `make test` adds `test_strassen` (11 tests), among them scaled data bit
    for bit the unscaled result times the scales (with alpha and beta
    rescaled, Gauss-3M's alignment and 4M's rescaling), an imbalanced ZGEMM
    that must run as 4M, BLAS-ABI calls with scaled lines within the
    classical componentwise bound, the library's own GEMM uses bitwise
    unchanged, and routing through every path above;
    `benchmarks/bench_strassen.c` measures speed and accuracy.
  - Measured on an M2 Max at 8b48826
    (`benchmarks/results/2026-11-06-8b48826-pretag/strassen/`): resident DGEMMs run
    1.07× faster at N = 4096, 1.13× at 8192 and 0.93× at 2048, ZGEMMs
    1.06–1.13× from 4096 up and 0.89× at 2048. Through the BLAS ABI with
    `AB_MODE=gpu`, DGEMM gains 1.05–1.09× and ZGEMM 1.04×. The normwise error
    is 1.9× the classical GEMM's on mixed-sign data and half of it on positive
    data; with rows of A and columns of B scaled by 2^±16 the componentwise
    error (relative to |A||B|) rises from 1.5×10⁻¹⁴ to 2–7×10⁴. At 274e778,
    before this port and on 1.4.0's kernels, it used about 10% less energy per
    product (`benchmarks/results/2026-10-20-274e778-strassen/`); that was not
    repeated.
- **Tests that pin the range guard's floors.** `test_range_guard`'s oracle
  reads the library's floor macros and calls its stage 1, so moving the entry
  floor to 2⁻⁷⁹, moving a macro, or dropping stage 1's product term passed
  every test (hostile audit 2026-11-10, L8). `test_guard_literal_floors`
  calls `ab_range_guard_stage1` and `ab_range_guard_c_stage1` with literal
  exponents. It checks that entries and products of 2⁻⁷⁸ are certified and
  those of 2⁻⁷⁹ go to stage 2, alpha's effect on the product floor, and the
  overflow headroom; a `_Static_assert` pins `AB_ENTRY_MIN_EXP`,
  `AB_PRODUCT_MIN_EXP` and `AB_SUM_MAX_EXP`. `test_blas_abi` adds four DGEMMs
  with NaN or Inf alpha or beta, which must give Accelerate's results with no
  GPU GEMM. On mutants built from fresh trees:
  - either floor at 2⁻⁷⁹ or 2⁻⁷⁷, or a dropped product term, fails the new
    test (the earlier tests missed the entry floor and the dropped term);
  - `AB_ENTRY_MIN_EXP` or `AB_PRODUCT_MIN_EXP` at −79 stops the build;
  - removing the host path's check of the scalars fails two DGEMM cases;
  - removing only the staged path's check changes no result: the staged run
    flags the non-finite result and sends the call to Accelerate anyway.

### Changed
- **Stage 2 of the exponent-range guard takes a fraction of the host time,
  with the same decisions.** With the complex-parts fix above, nearly every
  large Quantum ESPRESSO ZGEMM enters stage 2 (the projectors hold parts far
  below their rows' largest), where per-line scans strided across op(A)'s
  storage and a loop over every pair of lines cost more host time than
  Accelerate takes for the whole call. Stage 2 now finds each line's largest
  element in one pass over a contiguous line, or for lines strided across
  the storage in one streaming sweep over it, works out the need's terms
  once per line, gathers op(B)'s probe row once per group of rows that share
  it, and passes a row by a bound (its probe element times that row's
  smallest parts against its largest need) before it looks at the row's
  pairs; a column's own probe element is found only when a pair needs it.
  The DGEMM C check finds its probe elements the same way. Every decision is
  the old loop's: `tests/test_range_guard.c` (new, host only) keeps the old
  loops as the oracle and compares them on random and crafted data, real and
  complex, every transpose combination, refusals included, and with the CPU
  flushing subnormals to zero.
- **32 × 32 tiles are chosen by grid size, not output size**
  (`use_narrow_tiles` in `src/apple_bottom.m`). In 1.5.0 they took every C of
  at most 480 × 480 elements, a tall C on a stricter test. Alone on the GPU, a
  grid's time rises in steps
  of about 33 threadgroups for every kernel, and a step of 32 × 32
  threadgroups costs 0.29× a 64 × 64 step, so the finer tiles win while a
  grid takes few steps. They now take grids of up to 1024 of their
  threadgroups alone on the GPU, 441 for ZGEMM's concurrent products. Alone on
  the GPU, the 64 × 64 kernels keep grids of at most 64 threadgroups, where
  they prefetch, unless the 32 × 32 grid has at most 239. In ZGEMM's concurrent
  products they do not prefetch and keep no such grid. A tall C (M ≥ 4N) moves
  off the tall-skinny kernel while the 32 × 32 grid takes at most twice as
  many steps. Kernel GPU time on M2 Max against the 1.5.0 rule: 1.04–1.46×
  on most squares from 544 × 544 to 1024 × 1024 (0.92–0.98× at 832 × 832 and
  896 × 896), 1.41× on 600 × 600 × 9140 and 1.26× on 1024 × 40 × 4608 as one
  product alone (1.07× and 1.08× as ZGEMM's three concurrent products), and down
  to 0.88× on some concurrent tall-skinny products with tiny grids. Over all
  482 measured lines the geometric mean is 1.043×; the 228 that change
  kernel run 1.093× faster (194 faster, 34 slower). Results are bitwise
  unchanged. Measurements:
  `benchmarks/results/2026-10-25-2825d2a-narrow-tiles/`. Those are kernel
  times, and a whole call also converts its operands and C: through `dgemm_`
  and `zgemm_` with the GPU forced, against 1.5.1, the five calls measured take
  0.85–1.05× the time, with DGEMM 832² slower and ZGEMM 500 × 500 × 9140 'C','N'
  level ([whole calls](benchmarks/results/2026-11-06-8b48826-pretag/README.md#narrow-tiles-in-whole-calls)). The coverage cases
  move to shapes that still reach each kernel: 1100 × 1000 for the 64 × 64
  prefetch instances, 4000 × 40 for tall-skinny.
- **The device-buffer API converts its operands where they lie, across
  cores.** `ab_dev_dgemm` and `ab_dev_zgemm` copied each operand out of its
  buffer into FP64 arrays (complex ones split into real and imaginary
  parts) before converting those to double-float, and copied C back the
  same way, the copies of C and of real operands on one core. One pass now
  converts each operand in its buffer, in chunks across cores, measuring
  each part's range for its scale and gathering ZGEMM's per-line
  statistics as it goes; C converts back into its buffer the same way, and
  a transposed DGEMM operand through a small tile in cache. The host
  epilogue of a ZGEMM with general alpha and beta runs across cores too.
  Results are bitwise unchanged. On an M2 Max, at QE's shapes, a call
  takes 0.79× the time at 160×160×9140 'C','N' (`ab_dev_zgemm`, 10.5 →
  8.3 ms) and 0.75× at 9140×160×160 'N','N' (0.70× with general alpha and
  beta); `ab_dev_dgemm` takes 0.64× at 'T','N' 160×160×18280 and 0.77× at
  'N','N' 18280×160×160.
- **The device-buffer API keeps its double-float buffers between calls.**
  `ab_dev_dgemm` and `ab_dev_zgemm` made the `ABMatrix` handles of their
  operands (3 and 6; 11.7 MB each for a part of a 9140×160 complex operand)
  as fresh GPU buffers every call, and paid for them in page faults as the
  conversion first wrote them, before the GPU started on them and when the
  host first read the result. They now take them from the staged path's
  pool (at most `AB_STAGING_MB`), as the native ZGEMMs take their
  temporaries. Results are bitwise unchanged. On an M2 Max, at QE's
  shapes, a call takes 0.79× the time at 160×160×9140 'C','N'
  (`ab_dev_zgemm`, 7.9 → 6.3 ms) and 0.76× at 9140×160×160 'N','N';
  `ab_dev_dgemm` takes 0.65× at 'T','N' 160×160×18280 and 0.64× at 'N','N'
  18280×160×160. The host's part of these calls, above the GEMM alone,
  fell from 2.8–4.4 ms to 1.3–1.6 ms.
- **The device ZGEMM's general alpha/beta epilogue runs where C lies.**
  With alpha ≠ 1 or beta ≠ 0, `ab_dev_zgemm` downloaded T = op(A) op(B)
  and C into fresh FP64 arrays, formed alpha T + beta C, uploaded the
  result and downloaded it into C's buffer: about nine passes over C, four
  fresh m × n arrays and two GPU handles for T. The GEMM now writes T into
  C's handles, and three passes over C's buffer give the same values: C's
  ranges, the epilogue from T's double-float and C, and the result's round
  trip through double-float in place. Results are bitwise unchanged: C,
  padding included, matches the previous build in 41 cases (every
  alpha/beta path; scaled, imbalanced and line-wise data; padded leading
  dimensions; shapes from 40 × 300 × 56 to 1024³). On an M2 Max, a
  9140 × 160 × 160 'N','N' call takes 0.78× the time with general alpha
  and beta (7.33 → 5.73 ms) and 0.71× with alpha alone; 160 × 160 × 9140
  'C','N', whose C is small, is unchanged (0.99×).
- **A program that links apple-bottom ahead of its BLAS now gets apple-bottom's
  `dgemv_`, `zgemv_` and `zhemv_`**, which 1.5 did not export. On the M2 Max
  the large calls listed above go to the GPU, the rest to Accelerate, and on
  any other chip all of them go to Accelerate. `AB_MODE=cpu` sends them all to
  Accelerate; only link order restores another BLAS. Relinked with this
  library, Quantum ESPRESSO's `dgemv_` and `zgemv_` move from OpenBLAS to
  apple-bottom; its si64 energies, bands and forces do not change.
- **An illegal argument to `dgemv_`, `zgemv_` or `zhemv_` ends the process**
  with status 255: Accelerate's cblas names the argument and exits, as it has
  for `dgemm_` and `zgemm_`, and the program's XERBLA is not called. With
  apple-bottom linked first, the error-exit parts of Netlib's `dblat2` and
  `zblat2` stop there, as those of `dblat3` and `zblat3` already did.
- **The Makefile states `-ffp-contract=on` for the host code.** It is clang's
  default; the Makefile now names it in `CFLAGS`, `OBJCFLAGS` and
  `FFT_OBJCFLAGS`, so a change of toolchain default cannot change the host
  FP64 code's contraction. With Apple clang 17.0.0 every object file,
  `libapplebottom.dylib` and `libapplebottom.a` are byte-identical with and
  without the flag (checked 2026-11-10). What contraction does to the host
  epilogues' bits is in
  [LIMITATIONS.md](docs/vv/LIMITATIONS.md#framing) (hostile audit
  2026-11-10, L7).

### Fixed
- **`ab_dtrsm` and `ab_dgesv_mpir` no longer ignore a failed GPU step.**
  Their GPU panel updates (DTRSM) and residuals (MPIR) dropped the statuses
  of `ab_matrix_upload`, `ab_dgemm_scaled` and `ab_matrix_download`. When a
  command buffer failed (macOS stops long GPU work while the display is
  busy, `kIOGPUCommandBufferCallbackErrorImpactingInteractivity`), the solve
  went on with whatever the failed command buffer left in those rows and
  returned `AB_OK`. A step that fails is now done on Accelerate in FP64
  (`cblas_dgemm`), as the BLAS interface runs a call whose GPU step fails,
  so `AB_OK` again means B holds the solution, and `AB_PROFILE=1` prints the
  BLAS interface's fallback line for each such step (`AB_PROFILE ab_dtrsm
  M=… N=… K=…  GPU fallback to cblas: GPU step failed`), so a timing that
  fell back is not taken for the GPU's. This also covers a GPU buffer the
  solve cannot get, which used to fail the whole call with
  `AB_ERROR_ALLOC_FAILED`. The downloads of A and B and the upload of the
  result now return their status. New test `test_fail_stage_solvers` in
  `tests/test_trace.c`.
- **`ab_dgesv_mpir` no longer returns `AB_OK` for a refinement that did not
  converge.** After ten iterations it returned whatever X it had. In one
  probe on M2 Max (N = 64), the relative error of X for κ(A) = 10⁸, 10¹⁰ and
  10¹² was 2×10⁻², 54 and 9×10³, where LAPACK `dgesv` gets 9×10⁻¹⁰, 10⁻⁷ and
  6×10⁻⁶; for the Hilbert matrices of order 8 and 10 it was 1.0 and 3.7. It
  now does what LAPACK's `dsgesv` does. X is solved in FP64 instead, by
  `dgetrf` and `dgetrs` on the host copies, when refinement does not converge
  (next entry), when the FP32 factorization is singular, or when A or B lies
  outside FP32's range. `AB_PROFILE=1` prints `AB_PROFILE ab_dgesv_mpir N=…
  NRHS=…  FP64 LU fallback: <why>` for such a solve. Those solves now get
  3×10⁻⁹, 3×10⁻⁷ and 2×10⁻⁵, and 10⁻⁷ and 2×10⁻⁴, with backward errors near
  10⁻¹⁶ (A and B reach the solver rounded to double-float, as all native API
  data does). A singular FP32 factorization used to return
  `AB_ERROR_KERNEL_FAILED` even where FP64 solves A, as for
  [[1, 1], [1, 1 + 2⁻³⁰]]; that status now means A is singular in FP64. The
  `sgetrs` status inside the loop is now checked too (it can only report an
  illegal argument).
- **`ab_dgesv_mpir` also ends refinement where its DD residual levels off.**
  For dense A, ‖R‖_F / ‖B‖_F stops falling above the 10⁻¹⁴ the solver stops
  on, because A and X are rounded to double-float for the residual: on
  uniform random A it levels off at 1.3×10⁻¹⁴ for N = 150, 3.5×10⁻¹⁴ for 1024
  and 5×10⁻¹⁴ for 2048. Such solves had converged after 3–4 residuals but ran
  all ten. Refinement now also stops when a correction does not at least
  halve the worst column's ‖r_j‖∞ / ‖x_j‖∞, or after ten residuals, and then
  keeps X only if every column passes dsgesv's test,
  ‖r_j‖∞ ≤ √N · 2⁻⁴⁸ · ‖A‖∞ · ‖x_j‖∞ (double-float's unit roundoff 2⁻⁴⁸ in place
  of FP64's 2⁻⁵³; converged dense solves read 0.04–0.4 of the bound), and
  otherwise solves in FP64. Dense solves at N = 1024–4096 now take 5
  residuals with the backward error the ten gave (8×10⁻¹⁶). Solves that met
  10⁻¹⁴ before still stop there, with the same X.
  New tests `test_mpir_ill_conditioned` and `test_mpir_fp32_singular` in
  `tests/test_correctness.c`, and `test_mpir_fallback_lines` in
  `tests/test_trace.c`.
- **The exponent-range guard now has an entry floor.** In 1.5.1 it certified
  a call when every product of scaled entries reached 2⁻⁷⁸, with no floor on
  single entries. An entry below about 2⁻¹⁰² in a row (or column) that needs
  no scaling has a subnormal FP32 low part, which the GPU reads as zero, so
  such products kept only FP32 accuracy: `dgemm_` with a row [1, 2⁻¹⁰⁵(1+¾·2⁻²⁴)]
  of A and a column [0, 2³⁰] of B returned a relative error of 4.5×10⁻⁸ on
  the GPU and reported success. Stage 1 now also requires every nonzero
  scaled entry to be at least 2⁻⁷⁸; stage 2 accepts smaller entries where the
  sum through the largest entries covers K·2⁻⁷⁸ times the other line's
  largest entry (the most such entries can lose), and Accelerate takes the
  rest. Data whose scaled entries all reach 2⁻⁷⁸ is routed and computed as
  before. New cases `range-entry-low-*` in `tests/test_blas_abi.c`.
- **The guard's floors now hold for DGEMM's C and for each complex part.**
  Three more gaps of the same kind, each confirmed on the GPU in 1.5.1, every
  call reporting success:
  - For DGEMM with beta ≠ 0 the guard checked C′ (C as scaled for the upload)
    only through |β′C′| ≥ 2⁻⁷⁸, so with β = 2²⁰ a C′ down to 2⁻⁹⁸ passed,
    whose FP32 low word is subnormal: a zero row of op(A) with
    c = 2⁻⁹⁸(1+2⁻²⁹) came back with relative error 2⁻²⁹. Stage 1 now also
    requires every nonzero C′ to be at least 2⁻⁷⁸.
  - Below that floor it checked C only in zero rows of op(A) and zero columns
    of op(B), not where a row and a column have disjoint supports, which also
    leaves C_ij's product part zero: with β = 1, row [1, 0] and column [0, 1],
    c = 2⁻ᵏ(1+2⁻²⁹) lost its low word for k ≥ 98 and came back as zero for
    k ≥ 127. Now each C′_ij below the floors needs a product in its own
    entry, through the largest element of its row or of its column, that α′
    brings to 2⁻⁷⁸·max(1, |β′|), so that what C′_ij can lose stays below u²
    of it; without one (a zero line, or supports that miss each other) the
    call runs on Accelerate.
  - Complex statistics measured an element by its larger part, so a small
    part went unchecked: with every element of A and B 1 + t·i, Im C (by 4M)
    lost 2⁻²⁹ of itself for t below 2⁻⁹⁷ and came back as zero below 2⁻¹²⁶.
    The floors now apply to every nonzero real and imaginary part, and stage 2
    checks each part of C_ij in its own scale (|Ar||Br| + |Ai||Bi| for the
    real part, |Ar||Bi| + |Ai||Br| for the imaginary part).

  Calls whose scaled entries (for ZGEMM, every nonzero part and every product
  of two parts) and whose C′ and β′C′ reach 2⁻⁷⁸ are routed and computed as
  before (golden bits unchanged). ZGEMMs with smaller parts or products of
  parts now go through stage 2, which reads the operands on the host, and
  some of them to Accelerate. New cases `range-c-floor`,
  `range-c-disjoint` and `imag-part-low` (Accelerate; they fail on 1.5.1),
  and `range-c-near` and `imag-part-ok` (they must stay on the GPU), in
  `tests/test_blas_abi.c`.
- **The guard's statistics now see magnitudes below 2⁻¹⁰⁴².** They are FP64
  high words, and a nonzero value below 2⁻¹⁰⁴² has high word 0, as zero has.
  Such a value in op(A), op(B) or DGEMM's C, or a C′ that the scaling took
  below 2⁻¹⁰⁴² or to 0, passed stage 1 unseen, and a line made only of such
  values read as a zero line and was not scaled. The GPU reads them as zero.
  Confirmed on the GPU on the staged and the host path, every call reporting
  success (M = N = 8, K = 2, α = 1):
  - β = 1, row 0 of op(A) zero, C_00 = 2⁻ᵏ: C_00 came back as 0 for
    k ≥ 1043 (up to k = 1042 the call went to Accelerate);
  - the same with column 0 of op(B) at 2¹⁰⁰⁰ (scale 2⁻⁹⁷⁰), so that
    C′_00 = 2⁻⁽ᵏ⁺⁹⁷⁰⁾: C_00 came back as 0 for every k tried from 73
    (C′_00 = 2⁻¹⁰⁴³) to 110, k ≥ 105 included, where C′_00 underflows to 0;
  - row 0 of op(A) = [2⁻ᵏ, 0] against a column of 2¹⁰⁰⁰: C_00 = 2¹⁰⁰⁰⁻ᵏ came
    back as 0 for k ≥ 1043, through op(B) and for ZGEMM alike, and with every
    element 1 + 2⁻ᵏ·i, Im C came back as 0.

  The host pack, the C upload and the Metal pre-pass now count a nonzero
  value whose high word (after its scale) is 0 as the smallest nonzero high
  word. As a line's largest magnitude it reads as 2⁻¹⁰⁴², so a line of such
  values is scaled (by 2¹⁰⁷²) and runs on the GPU, exactly in the probes; as
  a smallest magnitude it reads as 2⁻¹⁰⁷⁴, below every floor, so stage 2
  decides, and a C′ that is all of C_ij goes to Accelerate. Data without
  values below 2⁻¹⁰⁴¹ is routed and computed as before (golden bits
  unchanged; the word also stands for [2⁻¹⁰⁴², 2⁻¹⁰⁴¹), whose minima now read
  low, conservatively). New cases `range-c-subnormal`,
  `range-c-scaled-out`, `range-entry-subnormal` and `imag-part-subnormal`
  (Accelerate) and `range-a-subnormal`, `range-b-subnormal` (GPU) in
  `tests/test_blas_abi.c`, which all fail on the previous guard (errors from
  2.8×10⁻⁹ to 1, Im 2.7×10⁷ u²); `subnormal` and `subnormalC` data in
  `tests/test_staging.c` hold the host and GPU statistics to the same counts.
- **The calls that `benchmarks/probe_guard_gaps.c` found passing the guard
  with a result degraded by FP32's exponent range now run on Accelerate, or on
  the GPU at DD accuracy.** At 8b48826, before the fixes above and BLAS-2's
  per-part guard, they returned up to 2.7×10⁶ u² of a part's own scale, or a
  part as zero. At fb4dfe0 each of those calls runs on Accelerate, or on the GPU
  within 2.2 u², and imaginary parts 2⁻⁸⁰ of the real parts run on Accelerate
  too. The probe's other rows are unchanged, the `spread` rows' 3M imaginary
  parts (up to 58 u²) included
  ([what the guard misses](docs/vv/LIMITATIONS.md#what-the-guard-misses),
  [results](benchmarks/results/2026-11-07-fb4dfe0-guard-gaps/)).
- **The native API now scales a matrix made only of values below 2⁻¹⁰⁴².**
  `ab_matrix_upload` measured such a matrix as zero and stored it as zeros:
  `ab_dgemm` and `ab_zgemm` with every entry of A at 2⁻ᵏ against B at 2¹⁰⁰⁰
  returned 0 for k ≥ 1043 (confirmed on the GPU, through A and through B).
  It now counts such values (at their own exponents: next entry), so such a
  matrix gets a scale, and the probes are exact for every k up to 1074. New
  cases `range-all-subnormal` (`ab_dgemm`, `ab_dgemm_scaled`, `ab_zgemm`) in
  `tests/test_blas_abi.c`, which fail on 1.5.1.
- **The native API centres a matrix whose values below 2⁻¹⁰⁴² lie within 2⁷⁸
  of its largest.** `ab_matrix_upload` left a nonzero value below 2⁻¹⁰⁴²
  (FP64 high word 0) out of a matrix's smallest magnitude when the matrix
  also held a larger value, so its scale could store that
  value below 2⁻³⁹ and its products below the 2⁻⁷⁸ floor, where they keep
  FP32's 24 bits. ULP Fiction's guard audit found A = [[0, t], [2⁻⁹⁹⁷, 0]],
  t = 177571·2⁻¹⁰⁷⁴ (span 2^59.6), and B = [[2¹⁰²³, 0],
  [2⁹⁴⁵·1.7382651127463331, 0]] (span 2^77.2): `ab_dgemm` returned C₀₀ =
  0x1.2d6e5ep-111, 2.035×10⁵ times the bound (3K+19)u²|A||B| + 2⁻⁵³|C| +
  2⁻¹⁰⁷⁴ (u = 2⁻²⁴), with `AB_OK`, on 56216ee and on 1.5.1 (the BLAS ABI was
  right on these operands). Such values now count at their own exponents,
  read from their low words, when the whole span, theirs included, is within
  2⁷⁸ (binary exponents at most 78 apart), so the matrix is centred as
  [LIMITATIONS](docs/vv/LIMITATIONS.md) says: C₀₀ is now
  0x1.2d6e5e5b83328p-111, 0.00116 times the bound. A wider span keeps the
  scale it had, so a matrix of ordinary values with one stray value below
  2⁻¹⁰⁴² keeps its scale and its bits. Bits change only for matrices that hold
  such a value and whose whole span is within 2⁷⁸ (all their values below
  about 2⁻⁹⁶⁴), a matrix made only of such values included. The upload's vectorized
  measure now counts such a value as the guard's statistics do, in the
  smallest magnitude too, and a scalar pass reads the exponents only of a
  matrix whose smallest nonzero magnitude is below 2⁻¹⁰⁴¹ (high word 0 or 1).
  On ordinary data (values at 2⁰ and 2³⁰⁰, n = 512, 2048 and 4096, four
  interleaved rounds on the M2 Max) `ab_matrix_upload` takes 0.92-0.96×
  56216ee's median time on one thread and 0.97-1.01× with `parallel` at
  n = 2048 and 4096 (the measure now computes one high word per value, not
  two). `ab_dev_dgemm` and `ab_dev_zgemm` convert through the same measure and
  get the same scales. The device ZGEMM's epilogue scales C's parts and the
  result's by the same rule, though no output there depends on it and no
  test can tell the two scales apart (the comment in
  `ab_internal_zgemm_epilogue` says why). The BLAS ABI's measure and bits are
  unchanged, and native results on data without such values are bitwise
  those of 56216ee (432 outputs at 2⁰ and 2^±300). New: `test_native_tiny_span` in
  `tests/test_device_api.c` (10 checks: the counterexample through handles
  and device buffers, real and complex; exponents 78 and 79 apart; a matrix
  made only of such values; a stray one among ordinary values; the epilogue's
  alpha and beta on the counterexample's operands) and `range-tiny-span`
  (`ab_dgemm`, `ab_dgemm_scaled`, `ab_zgemm`) in `tests/test_blas_abi.c`. All
  but the epilogue check fail on a build without their part of the fix.
- **A ZGEMM through the BLAS ABI that reads a NaN or Inf in C runs on
  Accelerate**, whether its epilogue would run on the GPU or on the host, as a
  DGEMM's has since 1.5.0. The epilogue's complex arithmetic,
  which the GPU's follows bit for bit, made (NaN, NaN) of such an element: an
  (Inf, y) lost its Inf, where Accelerate keeps it. The host epilogue has
  computed it this way since at least 1.3.0, and 1.5.0's GPU epilogue copied it.
  `ab_dev_zgemm` has no Accelerate fallback and still applies its epilogue to
  such a C. The calls count
  as `range` fallbacks. Results on finite C keep their bits; a call with a
  general alpha or beta scans C once on the host, the memory its epilogue reads
  anyway, and one with beta = 0 does not read C. `test_blas_abi` adds four
  cases, on the GPU epilogue's route and on the host epilogue's, and
  `test_staging`'s NaN and Inf cases now expect Accelerate.
- **A ZGEMM through the BLAS ABI whose FP64 epilogue could overflow runs on
  Accelerate.** The host epilogue (alpha ≠ 1, or beta not 0 or 1, on the staged
  path; every ZGEMM on the host path, `AB_HOST_PACK=1`) scaled each part of
  the GPU's product back by its row's and column's factors before applying
  alpha, and kept the result unchecked, so a product beyond FP64's range
  became Inf there, and the complex arithmetic NaN, even where alpha brought C
  back into range. A 32 × 32 × 32 `ab_zgemm_blas` with every entry of A at
  2⁶⁰⁰ and of B at 2⁵⁰⁰, alpha 2⁻²⁰⁰ and beta 0 (C = 2⁹⁰⁵) returned NaN + NaN i
  in all 1,024 entries with path `gpu`, staged and on the host path, at 56216ee
  and in 1.5.1; Accelerate returns Inf + NaN i (its own product overflows).
  With alpha 1 and beta 0 the same data came back as Inf + 0i, and an alpha
  or beta of Inf on ordinary data as NaN + NaN i where Accelerate gives
  Inf + NaN i, all with path `gpu`.
  A ZGEMM now keeps its GPU product only where bounds the guard already has
  (the largest parts of op(A), op(B) and C, K, alpha and beta) keep every value
  the epilogue forms finite
  ([Exponent range](docs/vv/LIMITATIONS.md#exponent-range)). Any other call
  runs on Accelerate, returns its bits and counts as a `range` fallback, as
  does one whose alpha or beta is NaN or Inf, or finite with |Re| + |Im|
  beyond FP64's range (DBL_MAX (1 + i), say), whatever its data: each call
  above now returns Accelerate's bits with path `cpu`, and the GEMM counters
  do not move. The host path decides before the GPU runs. The staged path
  decides C before its run and the product from the run's own statistics,
  before it copies a result into C; a call with alpha 1 and beta 0 whose C is
  registered memory has had its product written in place by then, and
  Accelerate writes C again (beta 0 does not read it). Its GPU epilogue
  (alpha = 1, beta 0 or 1), which flags an overflow, now also stops on a C of
  2¹⁰²² or more beside beta = 1; the host path then refuses such a call, as it
  would on its own. AB_SPLIT's GPU parts follow the same rule (a `range`
  outcome, made again on Accelerate). The bound refuses only calls whose data,
  with those largest parts, could make a part of P reach 2¹⁰¹⁹, of alpha·P
  2¹⁰¹⁷ (2¹⁰¹⁸ for a real or purely imaginary alpha) or of beta·C 2¹⁰²¹, so it
  also refuses some finite results (alpha = beta = 1 with C at 2¹⁰²³ and a
  small product now runs on Accelerate); lines in the guard's window
  [2⁻²⁰, 2²⁰) give E ≤ 73 in its 2^E bound on P, far from it. Calls it passes
  keep their bits: 100 ordinary ZGEMMs over every scalar class
  (`benchmarks/probe_zgemm_epilogue.c bits`) hash the same as on 56216ee,
  staged and on the host path, and `test_qe_golden`'s 32 cases are unchanged
  ([results](benchmarks/results/2026-11-09-f0c325b-zgemm-epilogue/)). The
  bounds cost nothing measurable: on M2 Max (140 W adapter, load1 under 5),
  `ab_zgemm_blas`'s median time on the fix over 56216ee's, balanced for which
  library ran first, over 8 rounds, is 1.0022 at 9139 × 500 × 512 with
  general alpha (39.2 ms), 0.9982 there with general alpha and beta
  (44.8 ms), 0.9986 at 2048³ with general alpha (112.2 ms) and 0.9935 at
  'C','N' 512 × 512 × 9139 with alpha 1 (29.8 ms; 0.9940 over 12 more
  rounds), with C bitwise the same in every call. Two earlier runs from a
  scratch copy, on other data, gave 0.9998 to 1.0013 and, at 'C','N', 1.0078
  (with a run-order effect: 1.026 when 56216ee ran first, 0.990 when the fix
  did) and then 1.0012 over 12 rounds. DGEMM is
  unaffected: it applies alpha and beta in DD before its one scale-back, so
  the reproduction as a DGEMM returns the finite 2⁹⁰⁵ on the GPU (Accelerate:
  Inf), and a DGEMM whose result itself overflows returns ±Inf.
  `ab_dev_zgemm`, which has no Accelerate fallback, is unchanged: on the
  reproduction's data it returns NaN + NaN i with `AB_OK`. `test_blas_abi`
  adds 16 cases on both sides of each bound (the GPU epilogue's flag at C =
  2¹⁰²² exactly and just below it among them), `test_staging` two kinds of
  data (7 cases) that every child sends to Accelerate, `test_split` four
  refused tails and their four kept mirrors, `test_range_guard` a test of the
  bounds at their edges, and `tests/split_defects.py` two planted defects in
  AB_SPLIT's range labels; each fails on a build without its part of the fix.
- **`make test-asan` no longer hangs where AddressSanitizer's runtime does not
  start.** On macOS 27.0.1, with Apple clang 17.0.0 or Homebrew clang 19.1.7, it
  hangs in its own initialization, before `main()`. `test-asan` now runs an empty
  program built with it first and stops after 30 s, and each sanitizer test run
  stops after 30 minutes ([`tests/README.md`](tests/README.md)).
- **Native operations report a failed command buffer.** `ab_matrix_zero`,
  `ab_matrix_copy`, `ab_matrix_add`, `ab_matrix_sub`, `ab_matrix_scale`,
  `ab_dsyrk`, `ab_dgemm_batched`, `ab_zherk`'s transposes and the command buffers
  of `ab_zgemm_batched` before its last waited for the GPU and returned `AB_OK`
  without reading the command buffer's status, so a buffer that macOS stopped
  left an unfinished output and no error. They now return
  `AB_ERROR_KERNEL_FAILED`, and `ab_zgemm_batched` stops at the failed buffer.
  Calls that succeed keep their bits. `AB_TEST_FAIL_STAGE=kernel` now reaches
  every synchronous native call but `ab_fft_execute`, and the new test-only
  `AB_TEST_FAIL_ONLY=n` fails only the n-th command buffer; `test_trace` adds
  `test_fail_native`.
- **The device API's element-wise helpers check their buffers.**
  `ab_dev_conjg_c`, `ab_dev_scale_z` and `ab_dev_axpy_z` checked only for
  NULL and n == 0, so an offset, n or increment that ran past a buffer made
  the call read and write outside it, returning `AB_OK` (hostile audit
  2026-11-10, L1). They now return `AB_ERROR_INVALID_ARG`, with every buffer
  unchanged, when a vector reaches past its buffer (the check counts complex
  elements, so its own arithmetic cannot overflow). A negative increment of
  `ab_dev_axpy_z` now starts at the far end, as in BLAS and as the header's
  BLAS-2 text says; before, it walked backwards from the offset (no caller in
  this repository or in Espressivo passes one). Positive and zero increments index the same elements with the
  same arithmetic. The header now states the units, the increment rule and
  the return codes. `test_device_api` adds `test_elementwise_bounds`, which
  fails on c8be67d.
- **Every native ZGEMM entry checks all six handles.** `zgemm_ex` compared
  only op(A)'s and op(B)'s K and Cr's shape, and `ab_zgemm_async`,
  `ab_batch_zgemm` and `ab_zgemm_batched` with more than one entry compared
  no dimension, so an imaginary part whose shape differed from its real
  part's was read or written past its end with `AB_OK` (hostile audit
  2026-11-10, L5). Each now checks, before any GPU work, that every imaginary
  part has its real part's shape and that op(A) is M × K, op(B) K × N and C
  M × N. Otherwise it returns `AB_ERROR_DIMENSION_MISMATCH`: `ab_zgemm_async`
  as a completed future, `ab_zgemm_batched` before any entry runs. Valid
  shapes take the same path as before. `test_correctness` adds one test per
  entry point, each failing on c8be67d.

### Corrected claims
- **"No silent errors outside FP32's exponent range" and "every product keeps
  DD precision" (1.5.0).** Too strong. In 1.5.x some data passed the range
  guard and lost accuracy with no fallback; the guard fixes above send the
  measured cases to Accelerate or keep them at DD accuracy
  ([what the guard misses](docs/vv/LIMITATIONS.md#what-the-guard-misses)). And,
  in 1.6.0 too, a product may lose bits where a larger product carries the
  element, which the guard accepts because the loss stays at that element's
  rounding level.
- **"ZGEMM keeps its imaginary part's accuracy" (1.5.0), and "3M's imaginary
  part stays within 8 u²" (`docs/vv/LIMITATIONS.md`).** Too strong. The rule
  between 3M and 4M reads each line's largest parts, not each element. Data
  whose magnitudes spread widely within a line stays on 3M, and its imaginary
  part reaches 26–887 u² depending on the call and route, against about 2 u² on
  uniform data at those calls
  ([Complex products](docs/vv/LIMITATIONS.md#complex-products)). Parts that leave
  FP32's range are lost under either form.
- **"A split inside the library is planned for 1.6.0" (README, roadmap, split
  benchmark docs).** 1.6.0 has no such split: `dgemm_` and `zgemm_` make each
  call on one engine, and the split stays in user code
  (`benchmarks/bench_split.c`).

### Known issues
- **No AddressSanitizer run.** Its runtime does not start on the release machine
  (Fixed, above), so 1.6.0 is checked under UndefinedBehaviorSanitizer only.
- **`AB_MODE=gpu AB_MIN_GPU_DIM=0` still fails Netlib's DGEMM and ZGEMM tests**,
  at double-float accuracy, as in 1.5.1. BLAS-2 calls below `AB_BLAS2_MIN_DIM` stay
  on Accelerate in that mode. With `AB_MODE` unset every Netlib routine passes and LAPACK fails
  1 test of about 2.04 million, against 2 in 1.5.1 and 0 with Netlib's own BLAS
  ([runs and attribution](benchmarks/results/2026-11-06-8b48826-pretag/README.md#netlib-and-lapack)).
- **The staged path is not calibrated.** `AB_MODE=auto` routes by 1.4.0's
  table, measured on the host path. Forced calls now beat Accelerate on some
  shapes the table keeps there: DGEMM 'N','N' 4096³ 200 ms against 225,
  ZGEMM 'N','N' 4096³ 601 ms against 881. `make calibrate` measures the staged
  path on your machine and writes a table for it.
- **Forcing the GPU in Quantum ESPRESSO is not faster than Accelerate** on an M2
  Max. On si64 at 8b48826, `AB_MODE=gpu` takes 1.04× the time of `AB_MODE=cpu` on
  input B at 8 ranks, 1.06× at 1 rank and 1.09× on input A at 8 ranks (two runs
  each), with the same energies. It takes 0.99–1.01× 1.5.1's time in the same runs.
- **Special values follow Accelerate.** A ZGEMM through the BLAS ABI that reads a
  NaN or Inf in C runs there, and Accelerate and Netlib differ on the other part
  of some such elements ([Exponent range](docs/vv/LIMITATIONS.md#exponent-range)).
- **macOS can stop a long GPU command buffer** while the display needs the GPU,
  as under 1.5.0's Known issues. BLAS-ABI calls, the device BLAS-2 calls,
  `ab_dtrsm` and `ab_dgesv_mpir` then finish on the host path or on Accelerate,
  with a correct result (the solvers redo the failed GEMM step with
  `cblas_dgemm`). Other native and device calls return `AB_ERROR_KERNEL_FAILED`:
  [the list](docs/vv/LIMITATIONS.md#command-buffers-stopped-by-macos).
- **Limits documented, not fixed** (hostile audit 2026-11-10). Each is in
  [Exponent range](docs/vv/LIMITATIONS.md#exponent-range) unless noted; the
  failing inputs are predicted from the code, not run:
  - the device GEMMs have no range guard and no Accelerate fallback (L2);
  - native results are not re-centred, so chains of products drift out of
    the window and lose accuracy with `AB_OK` (L3), and an alpha with |alpha|
    in [2⁻³⁹, 2⁴⁰) multiplies the stored values (L4);
  - no API limits K, and the closed-form bounds are stated for K ≤ 2²⁰ (L6,
    [Framing](docs/vv/LIMITATIONS.md#framing));
  - the BLAS-ABI ZGEMM epilogue can underflow before alpha, BLAS-2 rounds its
    sum to FP64 before alpha, and the FFT has FP32's exponent range (L9).

## [1.5.1] - 2026-11-01

Documentation corrections and a new check (`make test-dd-bound`); results are
those of 1.5.0.

### Added
- **`make test-dd-bound`**: checks the shader's `dd_fma` against MPFR (SKIP
  if MPFR is missing). The `DD` primitives are cut out of `src/apple_bottom.m`
  and compiled on the CPU without contraction; it runs 5×10⁶ adversarial and
  cancelling triples. It is not part of `make test`, so the test counts are
  unchanged.

### Corrected claims
- **Documented error bounds of `dd_mul` and `dd_fma`.** The default product
  is a two-FMA variant of DWTimesDW2 (JMP 2017 Algorithm 11), not the
  algorithm itself. The `5u²` of Muller–Rideau 2022 Theorem 2.7, quoted for it
  since 1.3.0, therefore did not apply. Neither did the unproven `8u²` envelope
  of `dd_fma`. `docs/vv/PRECISION_ENVELOPE.md` §1.2 now proves:
  - `dd_mul` ≤ 6.000001u²;
  - `dd_fma` ≤ (13+43u)u²|ab| + (3+5u)u²|c|, with its final `fastTwoSum`
    always exact;
  - GEMM ≤ (3K+13)u²|A||B| componentwise, so the "hard ceiling" of about
    1.5×10⁻¹¹ at K = 4096 in §1.3 is struck (the proven bound is about
    4.4×10⁻¹¹ of |A||B|).

### Fixed
- `docs/vv/VV_REPORT.md` §4.2.2–4.2.3: the VAL-1 figures withdrawn in 1.4.0
  (see [1.4.0] Corrected claims) are now struck through (Revision 1.4).
- The 1.5.0 release check (`benchmarks/results/2026-10-28-415cb5d-qe-si64-release-check/`)
  and `2026-10-29-351d56c-staged-refusals/` now say that the build they label
  "1.5.0" is release candidate 415cb5d, from before the staged-path fix; the
  tagged 1.5.0 includes that fix.

## [1.5.0] - 2026-10-31

### Highlights
- **The BLAS ABI's work moves to the GPU.** A `dgemm_`/`zgemm_` call routed
  to the GPU is now converted to double-float, scaled, multiplied and
  converted back on the GPU (the staged path), from GPU buffers the
  library reuses or from memory registered with `ab_register_host_memory`.
  Its results are bit for bit those of the CPU-side path, which
  `AB_HOST_PACK=1` keeps.
- **No silent errors outside FP32's exponent range.** Rows and columns far
  from 1 are scaled by powers of two, and what the range guard cannot
  certify runs on Accelerate. (Too strong; see Corrected claims under 1.6.0.)
- **ZGEMM keeps its imaginary part's accuracy** when real and imaginary
  parts are imbalanced, by taking four real products instead of Gauss's
  three, decided per row and column in the BLAS ABI and the device-buffer
  API. (Too strong; see Corrected claims under 1.6.0.)
- **Faster kernels with unchanged bits**: float4 tile loads, 32 × 32 tiles
  for small outputs, register prefetch on small grids, concurrent Gauss-3M
  products and pooled ZGEMM temporaries.
- **Per-call tracing** (`AB_TRACE`), the reasons calls leave the staged
  path (`AB_REPORT`), and a replay of traced Quantum ESPRESSO runs.
- **Measured on an M2 Max:**
  - Quantum ESPRESSO (64-atom Si, PAW) forced onto the GPU takes 1.01–1.06×
    its time on Accelerate, against 1.06–1.43× for 1.4.0 on the same QE
    build, with the same energies in every run. The routing table is still
    1.4.0's, so `AB_MODE=auto` keeps these calls on Accelerate.
  - One GEMM split between Accelerate and the GPU, in user code
    (`benchmarks/bench_split.c`), finished 1.5–1.9× sooner than the faster
    engine alone in all 20 well-scaled cases.
  - Data, in `benchmarks/results/`: `2026-10-28-415cb5d-qe-si64-release-check/`,
    `2026-10-29-351d56c-staged-refusals/`, `2026-10-28-415cb5d-hybrid-split/`
    (with `cpu-use/`) and `2026-10-24-15ef7b9-prefetch/`.

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
- **`benchmarks/bench_split.c`** (`make build/bench_split`) splits one DGEMM
  or ZGEMM between Accelerate, on the calling thread, and the BLAS ABI's
  staged path, on a second thread: by columns or rows of C, or by K with
  the GPU's partial sum added afterwards. It sweeps the GPU's share, times
  Accelerate alone, the GPU alone and the split in interleaved rounds at
  the best share, checks that the split's bits repeat, and measures errors
  against a compensated reference; with `np` processes, every repeat runs
  in lockstep. `benchmarks/split/` holds the runner, power sampling and
  table scripts. Results:
  `benchmarks/results/2026-10-28-415cb5d-hybrid-split/`.

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
  `dgemm_` 150×150×18277 runs 1.11× faster end to end and 160×160×18432
  1.03× (ratios of the median wall times per call, on battery).
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
- **The documentation describes 1.5.0.** The README's performance section
  names the build behind each table and adds the staged path, the Quantum
  ESPRESSO results and the split benchmark; `docs/INTEGRATION.md`, the API
  header and `docs/ROADMAP.md` follow. The v1.0.2 V&V report is kept as a
  record of that version, with its production approvals withdrawn.
  `docs/vv/PRECISION_ENVELOPE.md` gives `dd_mul` below 5u² and the fused
  `dd_fma` within 8u². `SECURITY.md` supports 1.5.x.
- **CI runs on the macos-15 and macos-26 images.** GitHub retires its
  macos-14 runner image on November 2, 2026. That image's job was the only
  one that built against the macOS 14 SDK, which has no `MTLMathModeSafe`
  (the shaders get `fastMathEnabled = NO` there); no CI job builds against
  that SDK now.

### Removed
- `docs/blog/fp64-on-apple-silicon.md`, an April 2026 post whose claims
  1.4.0 corrected (see "Corrected claims" under 1.4.0).
- The README's "Related projects" section.

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
      every product keeps DD precision (too strong; see Corrected claims
      under 1.6.0). Calls it cannot certify run on
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
  are imbalanced** (audit K-P1-2; too strong, see Corrected claims under 1.6.0).
  Gauss's three-multiplication form
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

### Known issues
- **The staged path is not calibrated.** `AB_MODE=auto` routes by 1.4.0's
  table, measured on the host path; `make calibrate` measures the staged
  path on your machine and writes a table for it.
- **Forcing the GPU in Quantum ESPRESSO is not yet faster than
  Accelerate** on an M2 Max (1.01–1.06× Accelerate's time, above).
- **macOS can stop a long GPU command buffer** while the display needs the
  GPU (`kIOGPUCommandBufferCallbackErrorImpactingInteractivity`). The
  staged path counts the call as `gpu_failed` and makes it on the host
  path, and a stopped host-path call runs on Accelerate; the result is
  correct either way. Quantum ESPRESSO's input A at 8 ranks met 1–6 such
  stops per run while the Mac was in interactive use, and none in the
  timing runs.

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
