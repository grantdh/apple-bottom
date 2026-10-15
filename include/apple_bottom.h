// =============================================================================
// apple_bottom.h — FP64-class BLAS for Apple Silicon GPU
// Copyright (c) 2026 Grant Heileman, Technology Residue. MIT License.
// =============================================================================

// =============================================================================
// Requirements:
//   - macOS 14+ on Apple Silicon
//   - With the macOS 15 SDK (Xcode 16+) on macOS 15+, shaders use MTLMathModeSafe;
//     otherwise they are compiled with fastMathEnabled = NO. Both keep full DD precision.
//
// Storage: ABMatrix data passed to ab_matrix_upload / ab_matrix_download is
// row-major (C order). The BLAS ABI (ab_*gemm_blas, dgemm_/zgemm_) and the
// device-buffer API (apple_bottom_device.h) are column-major, as in BLAS.
//
// Thread Safety:
//   - ab_init/ab_shutdown: safe to call from any thread
//   - Calls on different matrices may come from different threads; one matrix,
//     pool or session must not be used by two threads at once. All work shares
//     one Metal command queue.
//
// Performance on an M2 Max, against Accelerate (README.md#performance):
//
// Native ABMatrix API, matrices resident (upload and FP64<->DD conversion
// amortized; benchmarks/results/2026-10-14-3253a22-bench-paper/):
//   DGEMM: faster from about N = 1536 (1.04x), 1.10-1.16x for N = 2048-4096
//   ZGEMM: faster from about N = 1024 (1.12x), 1.30-1.39x for N = 1536-3072
//   ZHERK: slower at every measured size (0.06-0.67x, N <= 2048); use cblas_zherk
//   DSYRK: not re-measured for 1.4.0
//
// End-to-end regime — BLAS-ABI (ab_*_blas, raw pointers, no handle persistence; this is
// the path Fortran host codes hit through dgemm_/zgemm_):
//   The GPU is >= 1.10x faster than Accelerate only for ZGEMM with both operands
//   transposed or conjugated, M = N = K >= 4096 (1.11x at 4096, 1.20x at 6144). Every
//   DGEMM through N=6144 and every Quantum ESPRESSO si64 shape is slower on the GPU
//   (0.06-0.84x); other ZGEMMs reach at most 1.02x (6144, one operand transposed).
//   Anchor: benchmarks/results/2026-10-14-52f1880-calibrate/
//   (`make calibrate`). AB_MODE=auto, the default, follows that measurement
//   (src/dispatch_table.h): a call goes to the GPU only where the GPU was >= 1.10x
//   faster. DSYRK/ZHERK end-to-end behavior expected to be similar to GEMM (same
//   conversion path); not directly measured.
//
// API Limits:
//   - Matrix size: rows * cols <= AB_MAX_MATRIX_ELEMENTS (2^31 - 1) and the DD
//     buffer, rows * cols * 8 bytes, <= the Metal device's maxBufferLength. The
//     limit is on the total, not on each dimension: tall-skinny shapes such as
//     143666 x 171 are fine. ab_matrix_check_size() says whether a shape fits;
//     BLAS-ABI calls with a larger operand run on Accelerate (AB_PROFILE=1
//     prints why).
//   - Memory pool capacity: 128 entries (ab_pool_get_matrix returns NULL when full)
//   - Session capacity: 64 matrices per session
// =============================================================================
#ifndef APPLE_BOTTOM_H
#define APPLE_BOTTOM_H

#include <stdint.h>
#include <stdbool.h>
#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

#define APPLE_BOTTOM_VERSION_MAJOR 1
#define APPLE_BOTTOM_VERSION_MINOR 4
#define APPLE_BOTTOM_VERSION_PATCH 0
#define APPLE_BOTTOM_VERSION_STRING "1.4.0"

typedef struct ABMatrix_s* ABMatrix;
typedef struct ABSession_s* ABSession;
typedef struct ABMemoryPool_s* ABMemoryPool;
typedef struct ABFuture_s* ABFuture;

typedef enum {
    AB_OK = 0,
    AB_ERROR_NO_DEVICE = -1,
    AB_ERROR_ALLOC_FAILED = -2,
    AB_ERROR_DIMENSION_MISMATCH = -3,
    AB_ERROR_NOT_UPLOADED = -4,
    AB_ERROR_KERNEL_FAILED = -5,
    AB_ERROR_INVALID_ARG = -6,
    AB_ERROR_SHADER_COMPILE = -7,
    AB_ERROR_TOO_LARGE = -8,      // matrix beyond the size limits (see API Limits)
} ABStatus;

typedef enum {
    AB_NO_TRANS = 0,      // Use matrix as-is
    AB_TRANS = 1,         // Transpose (swap rows/cols)
    AB_CONJ_TRANS = 2     // Conjugate transpose A^H (for complex)
} ABTranspose;

typedef enum {
    AB_LEFT  = 0,         // op(A) * X = alpha * B  (solve for X from left)
    AB_RIGHT = 1          // X * op(A) = alpha * B  (solve for X from right)
} ABSide;

typedef enum {
    AB_UPPER = 0,         // Upper triangular
    AB_LOWER = 1          // Lower triangular
} ABUplo;

typedef enum {
    AB_NON_UNIT = 0,      // Diagonal is general
    AB_UNIT_DIAG = 1      // Diagonal is implicitly 1
} ABDiag;

typedef struct {
    double upload_time_ms;
    double download_time_ms;
    double kernel_time_ms;
    uint64_t dgemm_count;
    uint64_t zgemm_count;
    uint64_t elements_converted;
} ABStats;

// Initialization
ABStatus ab_init(void);
void ab_shutdown(void);
const char* ab_device_name(void);
bool ab_is_initialized(void);

// Matrix lifecycle
// Most elements (rows * cols) one matrix may hold: the GPU kernels index
// elements with 32-bit unsigned arithmetic.
#define AB_MAX_MATRIX_ELEMENTS 2147483647

// AB_OK if a rows x cols matrix fits the size limits; AB_ERROR_INVALID_ARG if
// rows or cols <= 0, AB_ERROR_TOO_LARGE beyond AB_MAX_MATRIX_ELEMENTS or the
// device's maxBufferLength, AB_ERROR_NO_DEVICE before ab_init(). Allocates nothing.
ABStatus ab_matrix_check_size(int rows, int cols);
// NULL if ab_matrix_check_size(rows, cols) fails or the allocation does.
ABMatrix ab_matrix_create(int rows, int cols);
void ab_matrix_destroy(ABMatrix m);
void ab_matrix_dims(ABMatrix m, int* rows, int* cols);
size_t ab_matrix_count(ABMatrix m);

// Data transfer
ABStatus ab_matrix_upload(ABMatrix m, const double* data, bool parallel);
ABStatus ab_matrix_download(ABMatrix m, double* data, bool parallel);
ABStatus ab_matrix_zero(ABMatrix m);
ABStatus ab_matrix_copy(ABMatrix src, ABMatrix dst);

// Memory Pool API (reduces allocation overhead in iterative codes)
ABMemoryPool ab_pool_create(size_t size_hint);
void ab_pool_destroy(ABMemoryPool pool);
ABMatrix ab_pool_get_matrix(ABMemoryPool pool, int rows, int cols);
void ab_pool_reset(ABMemoryPool pool);  // Mark all matrices as available

// Async API (overlap GPU compute with CPU work)
ABFuture ab_dgemm_async(ABMatrix A, ABMatrix B, ABMatrix C);

// True async ZGEMM: fused single command buffer with completion handler cleanup.
// Returns immediately; use ab_future_wait() or ab_future_is_ready() to synchronize.
ABFuture ab_zgemm_async(ABMatrix Ar, ABMatrix Ai, ABMatrix Br, ABMatrix Bi,
                        ABMatrix Cr, ABMatrix Ci);
ABStatus ab_future_wait(ABFuture f);
bool ab_future_is_ready(ABFuture f);
ABStatus ab_future_status(ABFuture f);
void ab_future_destroy(ABFuture f);

// BLAS operations
ABStatus ab_dgemm(ABMatrix A, ABMatrix B, ABMatrix C);
ABStatus ab_dgemm_scaled(double alpha, ABMatrix A, ABMatrix B, double beta, ABMatrix C);
ABStatus ab_zgemm(ABMatrix Ar, ABMatrix Ai, ABMatrix Br, ABMatrix Bi, ABMatrix Cr, ABMatrix Ci);

// Extended ZGEMM with transpose support (for QE compatibility)
ABStatus ab_zgemm_ex(
    ABTranspose transA, ABTranspose transB,
    ABMatrix Ar, ABMatrix Ai,
    ABMatrix Br, ABMatrix Bi,
    ABMatrix Cr, ABMatrix Ci
);

ABStatus ab_dsyrk(ABMatrix A, ABMatrix C);

// Triangular solve: solves op(A) * X = alpha * B  or  X * op(A) = alpha * B
// where A is triangular. Result overwrites B.
// Uses blocked forward/back-substitution with DGEMM for the panel updates.
// GPU-efficient for N >= 1024 (panel solves are CPU-bound for small blocks).
//
// Parameters:
//   side:   AB_LEFT  → op(A) * X = alpha * B
//           AB_RIGHT → X * op(A) = alpha * B
//   uplo:   AB_UPPER or AB_LOWER
//   transA: AB_NO_TRANS, AB_TRANS, or AB_CONJ_TRANS
//   diag:   AB_UNIT_DIAG (diagonal implicitly 1) or AB_NON_UNIT
//   alpha:  scalar multiplier
//   A:      triangular matrix (N × N)
//   B:      right-hand side / solution matrix (M × N), overwritten with X
ABStatus ab_dtrsm(ABSide side, ABUplo uplo, ABTranspose transA, ABDiag diag,
                  double alpha, ABMatrix A, ABMatrix B);

// Mixed-Precision Iterative Refinement: solves A * X = B
// Uses FP32 LU factorization (Accelerate LAPACK) + DD-DGEMM residual correction.
// Avoids explicit inversion (κ² error) and bespoke DD-DTRSM shaders.
// Converges to DD precision (~10⁻¹⁵) in 1-3 iterations for well-conditioned A.
//
// Parameters:
//   A: N×N coefficient matrix (not modified)
//   B: N×NRHS right-hand side, overwritten with solution X on output
ABStatus ab_dgesv_mpir(ABMatrix A, ABMatrix B);

// ZHERK: C = A × Aᴴ (Hermitian rank-K update)
// Computes Cr = Ar×Arᵀ + Ai×Aiᵀ, Ci = Ai×Arᵀ - Ar×Aiᵀ
// Uses GPU DSYRK for real part + GPU transpose + DGEMM for imaginary part.
// All operations stay on GPU — no CPU-side data transfer.
ABStatus ab_zherk(ABMatrix Ar, ABMatrix Ai, ABMatrix Cr, ABMatrix Ci);

// Element-wise operations
ABStatus ab_matrix_add(ABMatrix A, ABMatrix B, ABMatrix C);
ABStatus ab_matrix_sub(ABMatrix A, ABMatrix B, ABMatrix C);
ABStatus ab_matrix_scale(double alpha, ABMatrix A);

// Batched GEMM API — amortizes Metal command buffer overhead across many GEMMs.
// QE fires hundreds of small GEMMs per SCF iteration; batching them into a single
// command buffer eliminates the ~50μs per-call commit+wait overhead.
//
// Usage:
//   ABBatch batch = ab_batch_create();
//   ab_batch_dgemm(batch, A1, B1, C1);
//   ab_batch_dgemm(batch, A2, B2, C2);
//   ab_batch_dgemm_scaled(batch, alpha, A3, B3, beta, C3);
//   ab_batch_commit(batch);           // submits all GEMMs to GPU in one shot
//   ab_batch_wait(batch);             // blocks until GPU finishes
//   ab_batch_destroy(batch);
//
// For operations with dependencies, insert a barrier between groups:
//   ab_batch_dgemm(batch, A, B, T);   // T = A * B
//   ab_batch_barrier(batch);           // ensure T is written
//   ab_batch_dgemm(batch, T, C, D);   // D = T * C
typedef struct ABBatch_s* ABBatch;

ABBatch ab_batch_create(void);
void ab_batch_destroy(ABBatch batch);
ABStatus ab_batch_dgemm(ABBatch batch, ABMatrix A, ABMatrix B, ABMatrix C);
ABStatus ab_batch_dgemm_scaled(ABBatch batch, double alpha, ABMatrix A, ABMatrix B, double beta, ABMatrix C);
ABStatus ab_batch_zgemm(ABBatch batch, ABMatrix Ar, ABMatrix Ai, ABMatrix Br, ABMatrix Bi, ABMatrix Cr, ABMatrix Ci);
ABStatus ab_batch_barrier(ABBatch batch);
ABStatus ab_batch_commit(ABBatch batch);
ABStatus ab_batch_wait(ABBatch batch);

// Simple batched DGEMM/ZGEMM (array-of-pointers interface)
// Alternative to ABBatch for straightforward all-same-size workloads.
ABStatus ab_dgemm_batched(int batch_count, ABMatrix* As, ABMatrix* Bs, ABMatrix* Cs);
ABStatus ab_zgemm_batched(int batch_count, ABMatrix* Ars, ABMatrix* Ais,
                          ABMatrix* Brs, ABMatrix* Bis, ABMatrix* Crs, ABMatrix* Cis);

// Session API
ABSession ab_session_create(void);
void ab_session_destroy(ABSession s);
ABStatus ab_session_add(ABSession s, const char* name, int rows, int cols);
ABMatrix ab_session_get(ABSession s, const char* name);
ABStatus ab_session_upload(ABSession s, const char* name, const double* data);
ABStatus ab_session_download(ABSession s, const char* name, double* data);
ABStatus ab_session_dgemm(ABSession s, const char* A, const char* B, const char* C);
ABStatus ab_session_zgemm(ABSession s, const char* Ar, const char* Ai, const char* Br, const char* Bi, const char* Cr, const char* Ci);

// Statistics
ABStats ab_get_stats(void);
void ab_reset_stats(void);
void ab_print_stats(void);

// Utility
const char* ab_status_string(ABStatus status);

// BLAS-compatible dispatch-aware entry points: each call runs on the GPU or on
// Accelerate as AB_MODE and, in the default AB_MODE=auto, the calibrated dispatch
// table decide (docs/INTEGRATION.md, "Routing").
// These are the entry points host codes reach via the dgemm_/zgemm_ Fortran symbols;
// benchmarks use them directly to measure end-to-end dispatch behavior.
// They create the Metal context on first GPU use, so callers need not call ab_init().
void ab_dgemm_blas(char transA, char transB, int M, int N, int K,
                   double alpha, const double* A, int ldA,
                   const double* B, int ldB,
                   double beta, double* C, int ldC);
void ab_zgemm_blas(char transA, char transB, int M, int N, int K,
                   double _Complex alpha,
                   const double _Complex* A, int ldA,
                   const double _Complex* B, int ldB,
                   double _Complex beta,
                   double _Complex* C, int ldC);

// Returns the dispatch path taken by the most recent ab_*_blas (or dgemm_/zgemm_)
// call on this thread: "gpu" only if the GPU computed C, "cpu" if cblas did
// (including a fallback after a GPU failure), or "none" before any call. Thread-local.
const char* ab_get_last_dispatch_path(void);

// Library version and the git commit it was built from, "-dirty" if tracked
// files differed from that commit, e.g. "apple-bottom 1.4.0 (428f1cc1a1d3)";
// "unknown" in place of the commit when built outside a git checkout or from a copy
// vendored inside another repository. Static string.
const char* ab_build_info(void);

// BLAS-ABI diagnostics (dgemm_, zgemm_, ab_*gemm_, ab_*gemm_blas), read once
// per process from the environment; both reports start with ab_build_info():
//   AB_REPORT=1   at exit, a short summary on stderr, one "AB_REPORT[pid]" line
//                 per item: per routine the calls and wall time by where C was
//                 computed (gpu_calls% is a share of calls, gpu_time% a share
//                 of time), the GPU calls' stage costs (init, alloc, pack,
//                 upload, kernel = command-buffer GPU time, download,
//                 epilogue, other) and fallbacks to cblas by reason
//                 (no_device, too_large, gpu_failed, other).
//   AB_PROFILE=1  the same counts as a detailed table with dimension
//                 histograms at exit, plus a stderr line per GPU call and per
//                 fallback. AB_PROFILE_JSON=path also writes JSON at exit.
//   AB_PROFILE_THRESHOLD_NS=n  count only calls of at least n ns (both reports).
// A call that falls back counts as a CPU call, its wall time including the
// failed GPU attempt. A fork()ed child starts with empty counters and reports
// only calls of its own. The native ABMatrix API is not counted. Formats and
// the routing variables (AB_MODE, ...): docs/INTEGRATION.md.

#ifdef __cplusplus
}
#endif

#endif // APPLE_BOTTOM_H
