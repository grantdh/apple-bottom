#ifndef AB_TEST_DOT2_H
#define AB_TEST_DOT2_H

// Compensated FP64 GEMM reference for test_precision.c, test_blas_abi.c and
// test_device_api.c: Dot2 (Ogita, Rump & Oishi, SIAM J. Sci. Comput. 26(6),
// 2005), as accurate as accumulating in twice FP64 precision and rounding
// once; the componentwise scales errors are measured against; and the ZGEMM
// tests' real/imaginary imbalance check.

#include <Accelerate/Accelerate.h>
#include <dispatch/dispatch.h>
#include <float.h>
#include <math.h>
#include <stdbool.h>
#include <stddef.h>
#include <stdlib.h>

#define U2 0x1p-48      // u² = 2⁻⁴⁸, DD unit roundoff (u = 2⁻²⁴, FP32)

typedef struct {
    int N, K;
    double sign;
    const double *A, *B;
    double *S, *E;
} Dot2Job;

// Row i of S + E += sign·A·B with Dot2: TwoProduct via fma, TwoSum on S, and
// the rounding errors summed into E.
static inline void dot2_row(void *ctx, size_t i) {
#pragma STDC FP_CONTRACT OFF    // the error-free transformations need every rounding
    const Dot2Job *job = ctx;
    const int N = job->N, K = job->K;
    double *restrict s = job->S + i * N;
    double *restrict e = job->E + i * N;
    for (int k = 0; k < K; k++) {
        const double a = job->sign * job->A[i * K + k];
        const double *restrict b = job->B + (size_t)k * N;
        for (int j = 0; j < N; j++) {
            double p = a * b[j];
            double pe = fma(a, b[j], -p);
            double t = s[j] + p;
            double z = t - s[j];
            e[j] += ((s[j] - (t - z)) + (p - z)) + pe;
            s[j] = t;
        }
    }
}

// S + E += sign·A·B (row-major M×K times K×N), rows in parallel.
static inline void dot2_gemm_acc(int M, int N, int K, double sign, const double *A,
                                 const double *B, double *S, double *E) {
    Dot2Job job = {N, K, sign, A, B, S, E};
    dispatch_apply_f((size_t)M, DISPATCH_APPLY_AUTO, &job, dot2_row);
}

// The same for split-complex A (Ar + i Ai) and B: S[0] + E[0] += Ar Br - Ai Bi
// and S[1] + E[1] += Ar Bi + Ai Br.
static inline void dot2_zgemm_acc(int M, int N, int K, const double *Ar, const double *Ai,
                                  const double *Br, const double *Bi, double *const S[2],
                                  double *const E[2]) {
    dot2_gemm_acc(M, N, K, 1.0, Ar, Br, S[0], E[0]);
    dot2_gemm_acc(M, N, K, -1.0, Ai, Bi, S[0], E[0]);
    dot2_gemm_acc(M, N, K, 1.0, Ar, Bi, S[1], E[1]);
    dot2_gemm_acc(M, N, K, 1.0, Ai, Br, S[1], E[1]);
}

// |X1| + |X2| elementwise (X2 may be NULL).
static inline double *abs_sum(const double *X1, const double *X2, size_t n) {
    double *X = malloc(n * sizeof(double));
    for (size_t i = 0; i < n; i++)
        X[i] = fabs(X1[i]) + (X2 ? fabs(X2[i]) : 0.0);
    return X;
}

// L += (|X1| + |X2|)(|Y1| + |Y2|) in FP64 (X2, Y2 may be NULL). All terms are
// non-negative, so the relative error is at most γ_K: harmless in a bound.
static inline void abs_gemm_acc(int M, int N, int K, const double *X1, const double *X2,
                                const double *Y1, const double *Y2, double *L) {
    double *X = abs_sum(X1, X2, (size_t)M * K);
    double *Y = abs_sum(Y1, Y2, (size_t)K * N);
    cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, M, N, K,
                1.0, X, K, Y, N, 1.0, L, N);
    free(X); free(Y);
}

// Gauss-3M's imaginary part under Re/Im imbalance (audit K-P1-2): entries of
// magnitude [0.5, 1) with random signs (half_to_one), then one part of each
// row of op(A) and each column of op(B) multiplied by IMAG_RATIO. Re(C) and
// Im(C) must each be within IMAG_BOUND u² of Dot2 (dot2_zgemm_parts_err);
// 3M misses Im(C) by about 1/(2 IMAG_RATIO) of its scale there.
#define IMAG_BOUND 8.0          // units of u²
#define IMAG_RATIO 1e-6         // the smaller part's magnitude, relative

typedef enum {
    IMAG_SMALL,         // Im x IMAG_RATIO: nearly real A and B
    IMAG_LARGE,         // Re x IMAG_RATIO: nearly imaginary
    IMAG_LINES,         // even rows of op(A) and columns of op(B) nearly real, odd ones nearly
                        // imaginary: every row and column imbalanced, A and B as a whole not
} ImagData;

static inline const char *imag_tag(ImagData d) {
    return d == IMAG_SMALL ? "small-im" : d == IMAG_LARGE ? "small-re" : "lines";
}

// Whether line l of op(A) (a row) or op(B) (a column) gets its real part
// (else its imaginary part) multiplied by IMAG_RATIO.
static inline bool imag_small_re(ImagData d, int l) {
    return d == IMAG_LARGE || (d == IMAG_LINES && l % 2);
}

// x's sign, with a magnitude in [0.5, 1).
static inline double half_to_one(double x) { return copysign(0.5 + 0.5 * fabs(x), x); }

// The largest errors of Re(C) and Im(C) (err[0], err[1]) against alpha op(A)
// op(B) in Dot2, each relative to its own componentwise scale, alpha times
// |Ar||Br| + |Ai||Bi| for Re and |Ar||Bi| + |Ai||Br| for Im; NaN if C has a
// NaN. op(A) (M x K) and op(B) (K x N) are row-major split-complex, C
// interleaved column-major with ldc complex per column.
static inline void dot2_zgemm_parts_err(int M, int N, int K, const double *Ar, const double *Ai,
                                        const double *Br, const double *Bi, double alpha,
                                        const double *C, size_t ldc, double err[2]) {
    const size_t nr = (size_t)M * N;
    double *S[2], *E[2], *D[2];
    for (int p = 0; p < 2; p++) {
        S[p] = calloc(nr, sizeof(double)); E[p] = calloc(nr, sizeof(double));
        D[p] = calloc(nr, sizeof(double));
    }
    dot2_zgemm_acc(M, N, K, Ar, Ai, Br, Bi, S, E);
    abs_gemm_acc(M, N, K, Ar, NULL, Br, NULL, D[0]);
    abs_gemm_acc(M, N, K, Ai, NULL, Bi, NULL, D[0]);
    abs_gemm_acc(M, N, K, Ar, NULL, Bi, NULL, D[1]);
    abs_gemm_acc(M, N, K, Ai, NULL, Br, NULL, D[1]);
    err[0] = err[1] = 0.0;
    for (int j = 0; j < N; j++)
        for (int i = 0; i < M; i++)
            for (int p = 0; p < 2; p++) {
                size_t y = (size_t)i * N + j;
                double got = C[2 * ((size_t)j * ldc + i) + p];
                double e = fabs(got - alpha * (S[p][y] + E[p][y])) / (alpha * D[p][y] + DBL_MIN);
                if (isnan(e) || e > err[p]) err[p] = e;
            }
    for (int p = 0; p < 2; p++) { free(S[p]); free(E[p]); free(D[p]); }
}

#endif /* AB_TEST_DOT2_H */
