// =============================================================================
// gpu_stage.h — the BLAS-ABI staged path: reused GPU buffers, GPU conversion
// (internal)
// =============================================================================
//
// A GPU-routed dgemm_/zgemm_ (blas_wrapper.c) first tries this path:
//   1. ab_internal_stage_acquire(): the call's GPU buffers, from a per-process
//      pool (no fresh MTLBuffers once it has grown). An operand lying in
//      memory registered with ab_register_host_memory() is used in place.
//   2. The caller copies each other operand the GPU reads into its staging
//      buffer (ab_internal_stage_host()), packed: column by column, one
//      block when the operand is dense.
//   3. ab_internal_stage_run(): one command buffer converts op(A), op(B) (and
//      a DGEMM's C) to row-major DD on the GPU, runs the GEMM kernels of the
//      ABMatrix API, and converts the result and applies the epilogue on the
//      GPU, into C's staging buffer or C in place.
//   4. The caller copies the result into C. A ZGEMM with alpha != 1 or beta
//      not 0 or 1 ("split") instead takes the product as the host epilogue
//      reads it (ab_internal_stage_split()) and runs that epilogue.
// Every value is bit-identical to the host path (host pack, FP64 <-> DD and
// epilogue) under the CPU's default floating-point control: the GPU works on
// IEEE bit patterns with integer arithmetic and follows the operations of the
// compiled host epilogue. The pre-pass also gathers the exponent-range
// guard's line statistics (ab_internal.h), and the result stands only where
// the guard needs no scaling, keeps a ZGEMM on Gauss-3M and certifies the
// call from them (stage 2 on the caller's operands, which the run leaves as
// they were): there the host path computes exactly the same. Any other call,
// and one whose GPU epilogue meets Inf or NaN, is made on the host path
// instead, which scales, runs 4M or runs on Accelerate, with its own bits; so
// the GPU writes C in place only when it does not read it.
//
// The pool keeps at most AB_STAGING_MB MB between calls
// (ab_staging_cap_bytes()). Slots are per call, so threads calling at once
// each get their own. ab_shutdown() empties it.
// The native ZGEMMs (ab_zgemm, ab_zgemm_ex, the batched and asynchronous
// ones) take their Gauss-3M or 4M temporaries from it too.
// =============================================================================
#ifndef AB_GPU_STAGE_H
#define AB_GPU_STAGE_H

#include <stdbool.h>
#include <stddef.h>

#include "ab_internal.h"
#include "apple_bottom.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct ab_stage_s ab_stage_t;

typedef enum { AB_STAGE_A, AB_STAGE_B, AB_STAGE_C } ab_stage_operand_t;

// An operand as the caller passed it: stored rows x cols, column-major with
// leading dimension ld (in elements, complex ones for ZGEMM).
typedef struct {
    const void *ptr;
    int ld, rows, cols;
} ab_stage_matrix_t;

// One call: op(A) is M x K and op(B) K x N, with ta, tb 'N', 'T' or 'C'
// ('C' is 'T' for DGEMM).
typedef struct {
    char routine;                 // 'd' or 'z'
    char ta, tb;
    int M, N, K;
    ab_stage_matrix_t x[3];       // A, B, C
    bool c_in;                    // the GPU reads C: DGEMM beta != 0, ZGEMM alpha = 1, beta = 1
    bool split;                   // ZGEMM with other alpha or beta: the host epilogue
    double alpha[2], beta[2];     // real, imaginary; a DGEMM's applied in DD by the GEMM kernel
} ab_stage_gemm_t;

// The buffers for call g: NULL if Metal is not set up or a buffer cannot be
// made (as under AB_TEST_FAIL_STAGE=alloc).
ab_stage_t *ab_internal_stage_acquire(const ab_stage_gemm_t *g);

// The staging buffer to copy operand `which` into, and its leading dimension
// in *ld; NULL for one the GPU uses in place or does not read. For C (not
// split) the same buffer holds the result to copy out, NULL when C is in place.
void *ab_internal_stage_host(ab_stage_t *s, ab_stage_operand_t which, int *ld);

// The operands of s used in place.
int ab_internal_stage_in_place(const ab_stage_t *s);

// Runs g on the GPU: true if it ran (the result as ab_internal_stage_host()
// or ab_internal_stage_split() gives it, to use if the guard lets it stand);
// false if a value was Inf or NaN or the GPU failed, and then C is as it was
// unless it is in place.
bool ab_internal_stage_run(ab_stage_t *s, const ab_stage_gemm_t *g);

// The guard's statistics the run gathered: op(A)'s rows into *sa, op(B)'s
// columns into *sb, as ab_line_stats_scan leaves them (their arrays are the
// slot's), and C's range into *c (DGEMM, beta != 0). Valid until the slot is
// released.
void ab_internal_stage_stats(ab_stage_t *s, const ab_stage_gemm_t *g, ab_line_stats_t *sa,
                             ab_line_stats_t *sb, ab_range_t *c);

// A split call's product, as the host epilogue reads it: row-major real and
// imaginary FP64 parts (M x N each), converted as the host path converts them.
void ab_internal_stage_split(ab_stage_t *s, const ab_stage_gemm_t *g, const double **Cr,
                             const double **Ci);

// Counts a staged call whose result is in C in ab_get_stats(), as the host
// path counts its steps; copy_in_ms and copy_out_ms are the host's parts.
void ab_internal_stage_count(const ab_stage_gemm_t *g, double copy_in_ms, double copy_out_ms);

// Returns the slot to the pool, or frees what is beyond AB_STAGING_MB.
void ab_internal_stage_release(ab_stage_t *s);

// Bytes the pool holds, in use or idle (tests).
size_t ab_internal_stage_pool_bytes(void);

// Whether this thread's last BLAS-ABI call took its result from the staged
// path, and how many of its operands it used in place (blas_wrapper.c; tests).
bool ab_internal_last_staged(int *in_place);

#ifdef __cplusplus
}
#endif

#endif // AB_GPU_STAGE_H
