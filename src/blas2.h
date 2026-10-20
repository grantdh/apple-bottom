// =============================================================================
// blas2.h — double-float BLAS-2 on the GPU (internal)
// =============================================================================
//
// blas2.m runs GEMV and HEMV on the GPU; blas2_abi.c holds the public entry
// points (BLAS ABI, Fortran symbols, device-buffer API), their argument
// checks, routing and the Accelerate path.
// =============================================================================
#ifndef AB_BLAS2_H
#define AB_BLAS2_H

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

#include "apple_bottom_device.h"
#include "profiling/blas_profiler.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef enum { AB_B2_DGEMV, AB_B2_ZGEMV, AB_B2_ZHEMV } ab_b2_routine_t;

// Bytes per element of A, x and y.
static inline size_t ab_b2_elem_size(ab_b2_routine_t r) {
    return r == AB_B2_DGEMV ? sizeof(double) : 2 * sizeof(double);
}

// One BLAS-2 call. op is the TRANS ('N', 'T', 'C') or UPLO ('U', 'L') flag in
// upper case; for HEMV m == n. Complex scalars and vector elements are (re,
// im) pairs. x and y are the vector arguments as passed to BLAS, whose
// element 0 is at the far end for a negative increment. The GPU functions
// below take only calls with valid arguments, m, n >= 1, alpha and beta
// finite and alpha != 0.
typedef struct {
    ab_b2_routine_t routine;
    char op;
    int m, n;
    double alpha[2], beta[2];
    const double *x;
    int incx;
    double *y;
    int incy;
} ab_b2_call_t;

// How a GPU attempt ended. Only AB_B2_GPU has written y; after any other
// the caller runs the call on Accelerate.
typedef enum {
    AB_B2_GPU,          // y holds the GPU result
    AB_B2_RANGE,        // NaN or Inf, or values beyond double-float's full-precision range
    AB_B2_NO_DEVICE,    // no Metal context: no device, or a fork()ed child
    AB_B2_TOO_LARGE,    // A spans more than the device's maxBufferLength
    AB_B2_UNSUPPORTED,  // A could not be mapped for the GPU (unmapped pages, unaligned)
    AB_B2_FAILED,       // a buffer allocation or the command buffer failed
} ab_b2_result_t;

// The GPU run of c on A in host memory, column-major with leading dimension
// lda (elements), mapped for this call only by a no-copy wrapper. stage_ns,
// unless NULL, gets the time of each step (ab_prof_stage_t).
ab_b2_result_t ab_internal_b2_host(const ab_b2_call_t *c, const void *A, int lda,
                                   uint64_t stage_ns[AB_PROF_STAGE_COUNT]);

// The GPU run of c on A in a device buffer, a_offset elements in; the caller
// has checked that the matrix lies inside the buffer.
ab_b2_result_t ab_internal_b2_dev(const ab_b2_call_t *c, ab_dev_buffer_t A, size_t a_offset,
                                  int lda);

// AB_MODE=auto's measured rule (blas2_abi.c): true if a call of routine with
// flag op (upper case) on an m x n matrix goes to the GPU.
bool ab_internal_b2_auto_gpu(ab_b2_routine_t routine, char op, int m, int n);

// For tests: the calling thread's ZHEMV calls with rows_per_lane rows per
// lane (1 or 3: panels of 32 rows_per_lane rows) and chunks of chunk_panels
// panels, in place of the decomposition blas2.m chooses for n; 0 leaves
// either choice automatic.
void ab_internal_b2_hemv_override(int rows_per_lane, int chunk_panels);

// The panel height and chunk width, in rows and columns, of an n x n ZHEMV.
void ab_internal_b2_hemv_layout(int n, int *panel, int *chunk);

// For tests: x[0 .. n) converted on the GPU as the kernels convert A's
// elements (blas2.m, f64_to_dd), (hi, lo) of x[i] in dd[2 i] and
// dd[2 i + 1]; false if the GPU could not run it.
bool ab_internal_b2_convert(const double *x, size_t n, float *dd);

#ifdef __cplusplus
}
#endif

#endif // AB_BLAS2_H
