// =============================================================================
// apple_bottom.h — FP64-class BLAS for Apple Silicon GPU
// Copyright (c) 2026 Grant Heileman, Technology Residue. MIT License.
// =============================================================================

// =============================================================================
// Requirements:
//   - macOS 14+ with Xcode 16+ SDK (required for MTLMathModeSafe)
//   - Without Xcode 16+ SDK: compiles but achieves only ~10⁻⁸ precision
//
// Thread Safety:
//   - ab_init/ab_shutdown: Safe to call from any thread (uses dispatch_once)
//   - Matrix operations: NOT thread-safe — Metal command queue serializes
//   - Use separate contexts for concurrent workloads (future feature)
//
// Performance Notes:
//   - DGEMM: GPU wins for N >= 2048
//   - ZGEMM: GPU wins for N >= 1024
//   - DSYRK: GPU wins for N >= 3072
//   - ZHERK: Use cblas_zherk (AMX) instead — GPU decomposition is 20x slower
//
// API Limits:
//   - AB_MAX_DIMENSION = 46340 (max matrix dimension, overflow protection)
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
#define APPLE_BOTTOM_VERSION_MINOR 3
#define APPLE_BOTTOM_VERSION_PATCH 0
#define APPLE_BOTTOM_VERSION_STRING "1.3.0-dev"

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

// DEPRECATED: ab_zherk is 20x slower than cblas_zherk due to CPU-side transpose overhead.
// Use cblas_zherk from Accelerate instead. This function will be removed in v2.0.
#ifdef __GNUC__
__attribute__((deprecated("Use cblas_zherk instead - GPU decomposition is 20x slower than AMX")))
#endif
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

#ifdef __cplusplus
}
#endif

#endif // APPLE_BOTTOM_H
