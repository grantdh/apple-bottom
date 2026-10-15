#ifndef AB_TEST_BLAS_CHECK_H
#define AB_TEST_BLAS_CHECK_H

// A column-major BLAS-ABI GEMM result against Accelerate cblas, for
// test_blas_abi.c and test_qe_golden.c: the Fortran-ABI symbols, operand
// layouts, the cblas reference and the componentwise error
//   max_ij |C_ij - Cref_ij| / (|alpha| (|op(A)| |op(B)|)_ij + |beta| |C0_ij| + tiny),
// plus test data and environment helpers.

#include "apple_bottom.h"
#include <Accelerate/Accelerate.h>
#include <complex.h>
#include <float.h>
#include <math.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

#define TOL 1e-11   // componentwise bound of a BLAS-ABI result against cblas

// The plain Fortran BLAS symbols, bound by their exact Mach-O names. Under
// ACCELERATE_NEW_LAPACK a C reference to dgemm_ becomes _dgemm$NEWLAPACK,
// which is not the symbol a gfortran call site links against.
void fortran_dgemm(const char *transA, const char *transB,
                   const int *M, const int *N, const int *K,
                   const double *alpha, const double *A, const int *ldA,
                   const double *B, const int *ldB,
                   const double *beta, double *C, const int *ldC) __asm__("_dgemm_");
void fortran_zgemm(const char *transA, const char *transB,
                   const int *M, const int *N, const int *K,
                   const double complex *alpha, const double complex *A, const int *ldA,
                   const double complex *B, const int *ldB,
                   const double complex *beta, double complex *C, const int *ldC) __asm__("_zgemm_");

// Exported by fortran_bridge.c but not declared in apple_bottom.h.
void ab_dgemm_(const char *transA, const char *transB,
               const int *M, const int *N, const int *K,
               const double *alpha, const double *A, const int *ldA,
               const double *B, const int *ldB,
               const double *beta, double *C, const int *ldC);
void ab_zgemm_(const char *transA, const char *transB,
               const int *M, const int *N, const int *K,
               const double complex *alpha, const double complex *A, const int *ldA,
               const double complex *B, const int *ldB,
               const double complex *beta, double complex *C, const int *ldC);

static inline int imax(int a, int b) { return a > b ? a : b; }

typedef struct { int rows, cols, ld; } Layout;   // column-major storage

static inline bool is_notrans(char t) { return t == 'N' || t == 'n'; }

// Storage of an operand whose op() is op_rows x op_cols.
static inline Layout stored(char t, int op_rows, int op_cols, int pad) {
    Layout l = {is_notrans(t) ? op_rows : op_cols, is_notrans(t) ? op_cols : op_rows, 0};
    l.ld = imax(1, l.rows + pad);
    return l;
}

static inline size_t nelem(Layout l) {
    size_t n = (size_t)l.ld * l.cols;
    return n ? n : 1;
}

static inline enum CBLAS_TRANSPOSE ref_trans(char t, bool cplx) {
    switch (t) {
    case 'N': case 'n': return CblasNoTrans;
    case 'T': case 't': return CblasTrans;
    default:            return cplx ? CblasConjTrans : CblasTrans;   // 'C', 'c'
    }
}

// True if X equals X0 bitwise everywhere outside the rows x cols block.
static inline bool outside_block_unchanged(const void *X, const void *X0, size_t total,
                                           Layout l, size_t es) {
    const char *x = X, *x0 = X0;
    size_t pos = 0;
    for (int j = 0; j < l.cols; j++) {
        size_t start = (size_t)j * l.ld;
        if (memcmp(x + pos * es, x0 + pos * es, (start - pos) * es)) return false;
        pos = start + l.rows;
    }
    return !memcmp(x + pos * es, x0 + pos * es, (total - pos) * es);
}

// out (rows x cols, ld rows) = |op(X)| elementwise, moduli for a complex X.
static inline void abs_op(const void *X, Layout l, char t, bool cplx, int rows, int cols,
                          double *out) {
    for (int j = 0; j < cols; j++)
        for (int i = 0; i < rows; i++) {
            size_t x = is_notrans(t) ? (size_t)j * l.ld + i : (size_t)i * l.ld + j;
            out[(size_t)j * rows + i] = cplx ? cabs(((const double complex *)X)[x])
                                             : fabs(((const double *)X)[x]);
        }
}

// out (M x N, ld M) = |op(A)| |op(B)|; M, N, K > 0.
static inline void abs_product(bool cplx, const void *A, Layout la, char ta,
                               const void *B, Layout lb, char tb,
                               int M, int N, int K, double *out) {
    double *pa = malloc((size_t)M * K * sizeof *pa), *pb = malloc((size_t)K * N * sizeof *pb);
    abs_op(A, la, ta, cplx, M, K, pa);
    abs_op(B, lb, tb, cplx, K, N, pb);
    cblas_dgemm(CblasColMajor, CblasNoTrans, CblasNoTrans, M, N, K,
                1.0, pa, M, pb, K, 0.0, out, M);
    free(pa); free(pb);
}

// The reference for C = alpha op(A) op(B) + beta C0 (real calls use the real
// parts of alpha and beta). Cref holds a copy of C0 on entry; its M x N block
// becomes beta C0 (0 for beta = 0, whatever C0 holds) plus alpha op(A) op(B)
// by cblas, and absAB (M x N, ld M, zero on entry) |op(A)||op(B)|. Nothing is
// done for M or N = 0, and the product is skipped for alpha = 0 or K = 0.
static inline void blas_reference(bool cplx, char ta, char tb, int M, int N, int K,
                                  double complex alpha, double complex beta,
                                  const void *A, Layout la, const void *B, Layout lb,
                                  const void *C0, int ldc, void *Cref, double *absAB) {
    if (M <= 0 || N <= 0) return;
    if (!cplx) {
        alpha = creal(alpha);
        beta = creal(beta);
    }
    for (int j = 0; j < N; j++)
        for (int i = 0; i < M; i++) {
            size_t k = (size_t)j * ldc + i;
            if (cplx)
                ((double complex *)Cref)[k] =
                    beta == 0.0 ? 0.0 : beta * ((const double complex *)C0)[k];
            else
                ((double *)Cref)[k] = beta == 0.0 ? 0.0 : creal(beta) * ((const double *)C0)[k];
        }
    if (alpha == 0.0 || K <= 0) return;
    if (cplx) {
        const double complex one = 1.0;
        cblas_zgemm(CblasColMajor, ref_trans(ta, true), ref_trans(tb, true),
                    M, N, K, &alpha, A, la.ld, B, lb.ld, &one, Cref, ldc);
    } else {
        cblas_dgemm(CblasColMajor, ref_trans(ta, false), ref_trans(tb, false),
                    M, N, K, creal(alpha), A, la.ld, B, lb.ld, 1.0, Cref, ldc);
    }
    abs_product(cplx, A, la, ta, B, lb, tb, M, N, K, absAB);
}

// Componentwise error over the M x N block; NaN if any entry is NaN.
static inline double err_real(int M, int N, const double *C, const double *Cref, int ldc,
                              const double *absAB, int ldab, const double *C0,
                              double alpha, double beta) {
    double worst = 0.0;
    for (int j = 0; j < N; j++)
        for (int i = 0; i < M; i++) {
            size_t k = (size_t)j * ldc + i;
            double d = DBL_MIN;
            if (alpha != 0.0) d += fabs(alpha) * absAB[(size_t)j * ldab + i];
            if (beta != 0.0) d += fabs(beta) * fabs(C0[k]);
            double e = fabs(C[k] - Cref[k]) / d;
            if (isnan(e)) return NAN;
            if (e > worst) worst = e;
        }
    return worst;
}

static inline double err_complex(int M, int N, const double complex *C,
                                 const double complex *Cref, int ldc, const double *absAB,
                                 int ldab, const double complex *C0,
                                 double complex alpha, double complex beta) {
    double worst = 0.0;
    for (int j = 0; j < N; j++)
        for (int i = 0; i < M; i++) {
            size_t k = (size_t)j * ldc + i;
            double d = DBL_MIN;
            if (alpha != 0.0) d += cabs(alpha) * absAB[(size_t)j * ldab + i];
            if (beta != 0.0) d += cabs(beta) * cabs(C0[k]);
            double e = cabs(C[k] - Cref[k]) / d;
            if (isnan(e)) return NAN;
            if (e > worst) worst = e;
        }
    return worst;
}

// The componentwise error (top of file) of C, the result of
// C = alpha op(A) op(B) + beta C0 with C and C0 stored as lc, against cblas;
// NaN if any entry is NaN.
static inline double blas_error(bool cplx, char ta, char tb, int M, int N, int K,
                                double complex alpha, double complex beta,
                                const void *A, Layout la, const void *B, Layout lb,
                                const void *C0, Layout lc, const void *C) {
    const size_t bytes = nelem(lc) * (cplx ? sizeof(double complex) : sizeof(double));
    const int ldab = imax(1, M);
    void *Cref = malloc(bytes);
    double *absAB = calloc((size_t)ldab * imax(1, N), sizeof *absAB);
    memcpy(Cref, C0, bytes);
    blas_reference(cplx, ta, tb, M, N, K, alpha, beta, A, la, B, lb, C0, lc.ld, Cref, absAB);
    double err = cplx ? err_complex(M, N, C, Cref, lc.ld, absAB, ldab, C0, alpha, beta)
                      : err_real(M, N, C, Cref, lc.ld, absAB, ldab, C0, creal(alpha),
                                 creal(beta));
    free(Cref); free(absAB);
    return err;
}

// GEMMs the GPU has run in this process.
static inline uint64_t gpu_gemm_count(void) {
    ABStats s = ab_get_stats();
    return s.dgemm_count + s.zgemm_count;
}

// 64-bit FNV-1a of len bytes, continuing from h (FNV_OFFSET for a new hash).
#define FNV_OFFSET 0xcbf29ce484222325ULL
#define FNV_PRIME  0x100000001b3ULL
static inline uint64_t fnv1a(const void *data, size_t len, uint64_t h) {
    const unsigned char *p = data;
    for (size_t i = 0; i < len; i++) {
        h ^= p[i];
        h *= FNV_PRIME;
    }
    return h;
}

// The output of splitmix64 (Steele, Lea & Flood, OOPSLA 2014) from state x,
// which the generator then advances by 0x9E3779B97F4A7C15.
static inline uint64_t splitmix(uint64_t x) {
    uint64_t z = x + 0x9E3779B97F4A7C15ULL;
    z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ULL;
    z = (z ^ (z >> 27)) * 0x94D049BB133111EBULL;
    return z ^ (z >> 31);
}

extern char **environ;

// True for an environment entry ("NAME=value") of the library's settings
// (AB_*) or of Metal API validation (MTL_DEBUG_LAYER).
static inline bool is_ab_env(const char *entry) {
    return !strncmp(entry, "AB_", 3) || !strncmp(entry, "MTL_DEBUG_LAYER=", 16);
}

// Unsets every variable is_ab_env() matches, so that a test sees only the
// settings it makes itself. The names are collected first: unsetenv changes
// environ.
static inline void clear_ab_env(void) {
    size_t n = 0, k = 0;
    while (environ[n]) n++;
    char **names = calloc(n + 1, sizeof *names);
    for (size_t i = 0; i < n; i++)
        if (is_ab_env(environ[i])) names[k++] = strndup(environ[i], strcspn(environ[i], "="));
    for (size_t i = 0; i < k; i++) {
        unsetenv(names[i]);
        free(names[i]);
    }
    free(names);
}

#endif /* AB_TEST_BLAS_CHECK_H */
