#ifndef AB_TEST_DOT2_H
#define AB_TEST_DOT2_H

// Compensated FP64 GEMM reference for test_precision.c and test_blas_abi.c, and
// its step for the BLAS-2 reference (test_blas2_ref.h):
// Dot2 (Ogita, Rump & Oishi, SIAM J. Sci. Comput. 26(6), 2005), as accurate
// as accumulating in twice FP64 precision and rounding once.

#include <dispatch/dispatch.h>
#include <math.h>
#include <stddef.h>

typedef struct {
    int N, K;
    double sign;
    const double *A, *B;
    double *S, *E;
} Dot2Job;

// *s + *e += a·b: TwoProduct via fma, TwoSum on *s, and the rounding errors
// summed into *e (also the BLAS-2 reference, test_blas2_ref.h).
static inline void dot2_step(double *s, double *e, double a, double b) {
#pragma STDC FP_CONTRACT OFF    // the error-free transformations need every rounding
    double p = a * b;
    double pe = fma(a, b, -p);
    double t = *s + p;
    double z = t - *s;
    *e += ((*s - (t - z)) + (p - z)) + pe;
    *s = t;
}

// Row i of S + E += sign·A·B with Dot2.
static inline void dot2_row(void *ctx, size_t i) {
    const Dot2Job *job = ctx;
    const int N = job->N, K = job->K;
    double *restrict s = job->S + i * N;
    double *restrict e = job->E + i * N;
    for (int k = 0; k < K; k++) {
        const double a = job->sign * job->A[i * K + k];
        const double *restrict b = job->B + (size_t)k * N;
        for (int j = 0; j < N; j++) dot2_step(&s[j], &e[j], a, b[j]);
    }
}

// S + E += sign·A·B (row-major M×K times K×N), rows in parallel.
static inline void dot2_gemm_acc(int M, int N, int K, double sign, const double *A,
                                 const double *B, double *S, double *E) {
    Dot2Job job = {N, K, sign, A, B, S, E};
    dispatch_apply_f((size_t)M, DISPATCH_APPLY_AUTO, &job, dot2_row);
}

#endif /* AB_TEST_DOT2_H */
