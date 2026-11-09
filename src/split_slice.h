// =============================================================================
// split_slice.h — the cut and the two parts of an AB_SPLIT call, and the
// executor's seam (internal)
// =============================================================================
//
// A call the split table covers (dispatch.h) is cut along one axis of
// length L: Accelerate computes the head [0, L - g) on the calling thread,
// the GPU the tail [L - g, L) on a thread of its own (blas_wrapper.c). Both
// parts keep the call's TRANS flags, leading dimensions and alpha:
//   N  C's columns: the GPU's part reads B from column (tb 'N') or row
//      (otherwise) L - g, and writes C from column L - g; A whole.
//   M  C's rows: the GPU's part reads A from row (ta 'N') or column L - g,
//      and writes C from row L - g; B whole. C is staged, never in place.
//   K  the sum: the head adds beta C into C; the tail reads A from column
//      (ta 'N') or row L - g and B from row (tb 'N') or column L - g, and
//      writes P = alpha op(A_t) op(B_t) (beta 0, M x N, ld M), which is
//      added to C in FP64 after the join.
// The cut is a multiple of 32 on N and M, 64 on K; ab_split_cut has the
// rule. The cut and the parts are pure: no state, no GPU.
// =============================================================================
#ifndef AB_SPLIT_SLICE_H
#define AB_SPLIT_SLICE_H

#include <math.h>
#include <stdbool.h>
#include <stddef.h>

#include "split_times.h"   // ab_split_times_t

#define AB_SPLIT_ALIGN_NM 32
#define AB_SPLIT_ALIGN_K  64

// The cut's unit on an axis.
static inline int ab_split_align(char axis) {
    return axis == 'K' ? AB_SPLIT_ALIGN_K : AB_SPLIT_ALIGN_NM;
}

// g, the GPU's part of an axis of length L at share f, with min_dim the
// AB_MIN_GPU_DIM floor; 0: no split. g = round(f L / a) a, clamped to
// [a, L - a]; a g at or below the floor rises to the next multiple of a above
// it, and if that leaves the head shorter than a, nothing is split. L < 2a
// (or f <= 0) is not split.
static inline int ab_split_cut(char axis, int L, double f, int min_dim) {
    const long a = ab_split_align(axis);
    if (!(f > 0.0) || (long)L < 2 * a) return 0;
    long g = lround(f * (double)L / (double)a) * a;
    if (g < a) g = a;
    if (g > (long)L - a) g = (long)L - a;
    if (g <= (long)min_dim) {
        g = ((long)min_dim / a + 1) * a;
        if ((long)L - g < a) return 0;
    }
    return (int)g;
}

// One part: its op shape and where its operands start in the caller's A, B
// and C, in elements (complex ones for ZGEMM). into_p: the K axis's GPU part,
// which writes P (beta 0) instead of C.
typedef struct {
    int M, N, K;
    size_t a, b, c;
    bool into_p;
} ab_split_part_t;

static inline bool ab_split_notrans(char t) { return t == 'N' || t == 'n'; }

// The head (cpu) and tail (gpu) of a call cut on `axis` with GPU part g
// (0 < g < L).
static inline void ab_split_parts(char axis, int g, char ta, char tb, int M, int N, int K,
                                  int ldA, int ldB, int ldC, ab_split_part_t *cpu,
                                  ab_split_part_t *gpu) {
    const bool na = ab_split_notrans(ta), nb = ab_split_notrans(tb);
    *cpu = (ab_split_part_t){M, N, K, 0, 0, 0, false};
    *gpu = *cpu;
    if (axis == 'N') {
        const size_t cut = (size_t)(N - g);
        cpu->N = (int)cut;
        gpu->N = g;
        gpu->b = nb ? cut * (size_t)ldB : cut;
        gpu->c = cut * (size_t)ldC;
    } else if (axis == 'M') {
        const size_t cut = (size_t)(M - g);
        cpu->M = (int)cut;
        gpu->M = g;
        gpu->a = na ? cut : cut * (size_t)ldA;
        gpu->c = cut;
    } else {
        const size_t cut = (size_t)(K - g);
        cpu->K = (int)cut;
        gpu->K = g;
        gpu->a = na ? cut * (size_t)ldA : cut;
        gpu->b = nb ? cut : cut * (size_t)ldB;
        gpu->into_p = true;
    }
}

// =============================================================================
// The executor's seam (blas_wrapper.c)
// =============================================================================
// ab_internal_dgemm_split and ab_internal_zgemm_split run one BLAS-ABI call
// cut as f says, whatever the split table says, so that tests and
// benchmarks can compare cuts and engines in one process. Not profiled.

typedef enum {
    AB_SPLIT_KEPT,          // the GPU's part stood
    AB_SPLIT_KEPT_4M,       // ... a ZGEMM part by 4M
    AB_SPLIT_CPU,           // cblas made both parts (engine 1, or g = 0)
    AB_SPLIT_RANGE,         // the GPU's part was made again on Accelerate: the range guard
                            // did not certify it, or its C held NaN or Inf
    AB_SPLIT_GPU_FAILED,    // ... its run failed: no buffers, a command buffer failed twice,
                            // line scales unlike the host's, Inf or NaN in its epilogue, or
                            // the worker's floating-point control
    AB_SPLIT_DECLINED,      // not split (record.declined says why): the seam made the whole
                            // call on cblas, dgemm_/zgemm_ by their own routing
    AB_SPLIT_NOT_CUT,       // dgemm_/zgemm_: a row with f = 0, or an axis too short to cut;
                            // attributed to the row, made by the call's own routing
    AB_SPLIT_INVALID,       // arguments the seam does not cut (invalid BLAS arguments, alpha
                            // 0, g outside [0, L]); C untouched
} ab_split_outcome_t;

typedef struct {
    char axis;      // 'N', 'M' or 'K'
    int g;          // the GPU's part of the axis: L for the GPU only, 0 for the CPU only
    int order;      // 0 free; 1 the CPU's part finishes first; 2 the GPU's part finishes first
    int engine;     // 0 the staged path; 1 cblas (no GPU; tests)
} ab_split_forced_t;

// This thread's last split attempt, by the seam or by dgemm_/zgemm_ under
// AB_SPLIT.
typedef struct {
    int outcome;        // ab_split_outcome_t
    int declined;       // why it was not split (ab_prof_split_decline_t), -1 if it was
    int staged;         // the GPU part's staged-path outcome (ab_prof_staged_t), -1 if none ran
    int retries;        // staged runs repeated after a failed command buffer
    bool retry_kept;    // ... and the repeat stood
    bool scales;        // the GPU part's line scales were not the host's
    int in_place;       // operands the GPU part used in place
    int row;            // the split table's row (dgemm_/zgemm_), -1 for the seam
    char axis;
    int L, g;
    ab_split_times_t times;
} ab_split_record_t;

int ab_internal_dgemm_split(const ab_split_forced_t *f, char ta, char tb, int M, int N, int K,
                            double alpha, const double *A, int ldA, const double *B, int ldB,
                            double beta, double *C, int ldC);
int ab_internal_zgemm_split(const ab_split_forced_t *f, char ta, char tb, int M, int N, int K,
                            double _Complex alpha, const double _Complex *A, int ldA,
                            const double _Complex *B, int ldB, double _Complex beta,
                            double _Complex *C, int ldC);

// The record of this thread's last BLAS-ABI GEMM if it reached the split (the
// seam, or a call a row covered); false otherwise.
bool ab_internal_last_split(ab_split_record_t *out);

#endif // AB_SPLIT_SLICE_H
