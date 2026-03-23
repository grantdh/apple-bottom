// =============================================================================
// apple_bottom.h — FP64-class BLAS for Apple Silicon GPU
// Copyright (c) 2026 Grant Heileman, Technology Residue. MIT License.
// =============================================================================

// =============================================================================
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
#define APPLE_BOTTOM_VERSION_MINOR 0
#define APPLE_BOTTOM_VERSION_PATCH 0
#define APPLE_BOTTOM_VERSION_STRING "1.0.0"

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
ABStatus ab_dsyrk(ABMatrix A, ABMatrix C);
ABStatus ab_zherk(ABMatrix Ar, ABMatrix Ai, ABMatrix Cr, ABMatrix Ci);

// Element-wise operations
ABStatus ab_matrix_add(ABMatrix A, ABMatrix B, ABMatrix C);
ABStatus ab_matrix_sub(ABMatrix A, ABMatrix B, ABMatrix C);
ABStatus ab_matrix_scale(double alpha, ABMatrix A);

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
