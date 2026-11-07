// =============================================================================
// split_slice.h — the cut and the two parts of an AB_SPLIT call (internal)
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
// rule. Pure: no state, no GPU.
// =============================================================================
#ifndef AB_SPLIT_SLICE_H
#define AB_SPLIT_SLICE_H

#include <math.h>
#include <stdbool.h>
#include <stddef.h>

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

#endif // AB_SPLIT_SLICE_H
