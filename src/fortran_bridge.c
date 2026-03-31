#include <complex.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

/* Forward declarations — these exist in blas_wrapper.c */
extern void ab_dgemm_blas(char transA, char transB, int M, int N, int K,
                          double alpha, const double* A, int ldA,
                          const double* B, int ldB,
                          double beta, double* C, int ldC);
extern void ab_zgemm_blas(char transA, char transB, int M, int N, int K,
                          double complex alpha, const double complex* A, int ldA,
                          const double complex* B, int ldB,
                          double complex beta, double complex* C, int ldC);

/* Fortran BLAS symbols for sub-threshold passthrough */
extern void dgemm_(const char*, const char*, const int*, const int*, const int*,
                   const double*, const double*, const int*,
                   const double*, const int*,
                   const double*, double*, const int*);
extern void zgemm_(const char*, const char*, const int*, const int*, const int*,
                   const double complex*, const double complex*, const int*,
                   const double complex*, const int*,
                   const double complex*, double complex*, const int*);

#define CROSSOVER_FLOPS 100000000ULL

/* Optional profiling — set AB_PROFILE_FILE env var to enable */
static FILE* _ab_prof = NULL;
static int _ab_prof_init = 0;

static inline void ab_log(const char* fn, int M, int N, int K, int gpu) {
    if (!_ab_prof_init) {
        const char* path = getenv("AB_PROFILE_FILE");
        if (path) {
            _ab_prof = fopen(path, "w");
            if (_ab_prof) fprintf(_ab_prof, "func M N K MNK gpu\n");
        }
        _ab_prof_init = 1;
    }
    if (_ab_prof) {
        fprintf(_ab_prof, "%s %d %d %d %llu %d\n",
                fn, M, N, K, (unsigned long long)M * N * K, gpu);
    }
}

void ab_dgemm_(
    const char *transA, const char *transB,
    const int *M, const int *N, const int *K,
    const double *alpha, const double *A, const int *ldA,
    const double *B, const int *ldB,
    const double *beta, double *C, const int *ldC)
{
    uint64_t flops = (uint64_t)(*M) * (*N) * (*K) * 2;
    int gpu = flops >= CROSSOVER_FLOPS;
    ab_log("dgemm", *M, *N, *K, gpu);
    if (gpu) {
        ab_dgemm_blas(*transA, *transB, *M, *N, *K,
                      *alpha, A, *ldA, B, *ldB, *beta, C, *ldC);
        return;
    }
    dgemm_(transA, transB, M, N, K, alpha, A, ldA, B, ldB, beta, C, ldC);
}

void ab_zgemm_(
    const char *transA, const char *transB,
    const int *M, const int *N, const int *K,
    const double complex *alpha, const double complex *A, const int *ldA,
    const double complex *B, const int *ldB,
    const double complex *beta, double complex *C, const int *ldC)
{
    uint64_t flops = (uint64_t)(*M) * (*N) * (*K) * 8;
    int gpu = flops >= CROSSOVER_FLOPS;
    ab_log("zgemm", *M, *N, *K, gpu);
    if (gpu) {
        ab_zgemm_blas(*transA, *transB, *M, *N, *K,
                      *alpha, A, *ldA, B, *ldB, *beta, C, *ldC);
        return;
    }
    zgemm_(transA, transB, M, N, K, alpha, A, ldA, B, ldB, beta, C, ldC);
}
