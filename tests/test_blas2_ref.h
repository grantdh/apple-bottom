#ifndef AB_TEST_BLAS2_REF_H
#define AB_TEST_BLAS2_REF_H

// The compensated FP64 reference of a BLAS-2 product, for test_blas2.c and
// benchmarks/bench_gemv.c: (op(A) x)_i by Dot2 (test_dot2.h), as accurate as
// accumulating in twice FP64 precision and rounding once, and
// (|op(A)| |x|)_i.

#include "test_dot2.h"
#include <math.h>
#include <stddef.h>
#include <stdlib.h>

typedef enum { B2_DGEMV, B2_ZGEMV, B2_ZHEMV } B2Routine;

// Element i of a BLAS vector of len elements with increment inc.
static inline size_t b2_vidx(int len, int inc, int i) {
    return inc > 0 ? (size_t)i * inc : (size_t)(len - 1 - i) * (size_t)(-inc);
}

// Element (i, k), as (re, im), of the matrix the call multiplies x by: op(A)
// for GEMV (op 'N', 'T' or 'C'), or the Hermitian matrix whose op ('U' or
// 'L') triangle A stores, the imaginary parts of its diagonal taken as zero.
// A is column-major with lda elements per column, (re, im) pairs for Z.
static inline void b2_elem(B2Routine r, char op, const double *A, int lda, int i, int k,
                           double *re, double *im) {
    size_t at;
    double sg = 1;
    if (r == B2_ZHEMV) {
        int stored = op == 'U' ? i <= k : i >= k;
        at = stored ? (size_t)k * lda + i : (size_t)i * lda + k;
        if (!stored) sg = -1;
        if (i == k) {
            *re = A[2 * at];
            *im = 0;
            return;
        }
    } else {
        at = op == 'N' ? (size_t)k * lda + i : (size_t)i * lda + k;
        if (op == 'C') sg = -1;
    }
    *re = r == B2_DGEMV ? A[at] : A[2 * at];
    *im = r == B2_DGEMV ? 0 : sg * A[2 * at + 1];
}

// Term k of output i into its sums s, e (real, imaginary) and magnitude mag.
static inline void b2_term(B2Routine r, char op, const double *A, int lda, int i, int k,
                           const double *xk, double abs_xk, double s[2], double e[2], double *mag) {
    double ar, ai;
    b2_elem(r, op, A, lda, i, k, &ar, &ai);
    dot2_step(&s[0], &e[0], ar, xk[0]);
    if (r != B2_DGEMV) {
        dot2_step(&s[0], &e[0], -ai, xk[1]);
        dot2_step(&s[1], &e[1], ar, xk[1]);
        dot2_step(&s[1], &e[1], ai, xk[0]);
    }
    *mag += (r == B2_DGEMV ? fabs(ar) : hypot(ar, ai)) * abs_xk;
}

// t[2q], t[2q + 1] = (op(A) x)_i, real and imaginary, and mag[q] =
// (|op(A)| |x|)_i for the outputs i = rows[q], q < nrows (i = q if rows is
// NULL); x has xlen elements with increment incx. Every output adds its terms
// in ascending k; for GEMV 'N' the loop over k is the outer one, so A is read
// down its columns.
static inline void b2_reference(B2Routine r, char op, const double *A, int lda, const double *x,
                                int xlen, int incx, const int *rows, int nrows, double *t,
                                double *mag) {
    const int w = r == B2_DGEMV ? 1 : 2;
    double *s = (double *)calloc((size_t)nrows * 4 + 1, sizeof *s), *e = s + (size_t)nrows * 2;
    double *ax = (double *)malloc(((size_t)xlen + 1) * sizeof *ax);
    for (int k = 0; k < xlen; k++) {
        const double *xk = x + b2_vidx(xlen, incx, k) * w;
        ax[k] = w == 2 ? hypot(xk[0], xk[1]) : fabs(xk[0]);
    }
    for (int q = 0; q < nrows; q++) mag[q] = 0;
#define B2_TERM(q, k) b2_term(r, op, A, lda, rows ? rows[q] : (q), (k), \
                              x + b2_vidx(xlen, incx, (k)) * w, ax[k], &s[2 * (q)], &e[2 * (q)], &mag[q])
    if (r != B2_ZHEMV && op == 'N') {
        for (int k = 0; k < xlen; k++)
            for (int q = 0; q < nrows; q++) B2_TERM(q, k);
    } else {
        for (int q = 0; q < nrows; q++)
            for (int k = 0; k < xlen; k++) B2_TERM(q, k);
    }
#undef B2_TERM
    for (int q = 0; q < nrows; q++) {
        t[2 * q] = s[2 * q] + e[2 * q];
        t[2 * q + 1] = s[2 * q + 1] + e[2 * q + 1];
    }
    free(s);
    free(ax);
}

#endif /* AB_TEST_BLAS2_REF_H */
