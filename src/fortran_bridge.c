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

#define DEFAULT_CROSSOVER_FLOPS 100000000ULL

/* AB_MODE runtime knob (cpu/gpu/auto) — cached on first call.
 * cpu  : always passthrough to system zgemm_/dgemm_ (baseline)
 * gpu  : always dispatch to ab_*_blas (which forces GPU above AB_MIN_GPU_DIM)
 * auto : FLOP-threshold routing — threshold is AB_CROSSOVER_FLOPS env (default 1e8)
 */
enum { FB_AUTO = 0, FB_CPU = 1, FB_GPU = 2 };
static int _fb_mode = -1;
static int fb_get_mode(void) {
    if (_fb_mode >= 0) return _fb_mode;
    const char* s = getenv("AB_MODE");
    if (s && (s[0] == 'c' || s[0] == 'C')) _fb_mode = FB_CPU;
    else if (s && (s[0] == 'g' || s[0] == 'G')) _fb_mode = FB_GPU;
    else _fb_mode = FB_AUTO;
    return _fb_mode;
}

/* Runtime-tunable FLOP threshold for auto-mode dispatch.
 * This is the knob that actually controls QE's dispatch split (72/795 on si64).
 * Lower value → more calls go to GPU; higher value → more calls stay on CPU.
 * Same default (1e8) as before, so existing behavior is preserved. */
static uint64_t _fb_cross = 0;
static uint64_t fb_get_crossover(void) {
    if (_fb_cross) return _fb_cross;
    const char* s = getenv("AB_CROSSOVER_FLOPS");
    unsigned long long v = (s && *s) ? strtoull(s, NULL, 10) : DEFAULT_CROSSOVER_FLOPS;
    if (v == 0) v = DEFAULT_CROSSOVER_FLOPS;
    _fb_cross = (uint64_t)v;
    return _fb_cross;
}

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
    int mode = fb_get_mode();
    uint64_t flops = (uint64_t)(*M) * (*N) * (*K) * 2;
    int dispatch = (mode == FB_CPU) ? 0
                 : (mode == FB_GPU) ? 1
                 : (flops >= fb_get_crossover());
    ab_log("dgemm", *M, *N, *K, dispatch);
    if (dispatch) {
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
    int mode = fb_get_mode();
    uint64_t flops = (uint64_t)(*M) * (*N) * (*K) * 8;
    int dispatch = (mode == FB_CPU) ? 0
                 : (mode == FB_GPU) ? 1
                 : (flops >= fb_get_crossover());
    ab_log("zgemm", *M, *N, *K, dispatch);
    if (dispatch) {
        ab_zgemm_blas(*transA, *transB, *M, *N, *K,
                      *alpha, A, *ldA, B, *ldB, *beta, C, *ldC);
        return;
    }
    zgemm_(transA, transB, M, N, K, alpha, A, ldA, B, ldB, beta, C, ldC);
}
