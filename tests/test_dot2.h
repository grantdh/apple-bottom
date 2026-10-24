#ifndef AB_TEST_DOT2_H
#define AB_TEST_DOT2_H

// Compensated FP64 GEMM reference for test_precision.c and test_blas_abi.c:
// Dot2 (Ogita, Rump & Oishi, SIAM J. Sci. Comput. 26(6), 2005), as accurate
// as accumulating in twice FP64 precision and rounding once; and the
// componentwise scales errors are measured against.

#include <Accelerate/Accelerate.h>
#include <dispatch/dispatch.h>
#include <math.h>
#include <stddef.h>
#include <stdlib.h>

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

#endif /* AB_TEST_DOT2_H */
