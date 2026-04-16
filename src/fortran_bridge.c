/* =============================================================================
 * fortran_bridge.c — Fortran BLAS entry points for apple-bottom
 * =============================================================================
 *
 * Exports the *standard* Fortran BLAS symbols (dgemm_, zgemm_) so any
 * Fortran code linked against -lapplebottom picks us up without call-site
 * edits. On macOS two-level namespace, these symbols shadow Accelerate's
 * when -lapplebottom appears before -framework Accelerate.
 *
 * CPU fallback goes through the CBLAS interface (cblas_dgemm / cblas_zgemm)
 * so we never self-recurse into our own dgemm_/zgemm_.
 *
 * The legacy ab_dgemm_ / ab_zgemm_ names are kept as weak aliases so any
 * call-site that was already patched during the transition continues to work.
 * =============================================================================
 */
#include <Accelerate/Accelerate.h>
#include <complex.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

#include "profiling/blas_profiler.h"

/* Forward declarations — GPU-path DD kernels live in blas_wrapper.c */
extern void ab_dgemm_blas(char transA, char transB, int M, int N, int K,
                          double alpha, const double* A, int ldA,
                          const double* B, int ldB,
                          double beta, double* C, int ldC);
extern void ab_zgemm_blas(char transA, char transB, int M, int N, int K,
                          double complex alpha, const double complex* A, int ldA,
                          const double complex* B, int ldB,
                          double complex beta, double complex* C, int ldC);

#define DEFAULT_CROSSOVER_FLOPS 100000000ULL

/* AB_MODE runtime knob (cpu/gpu/auto) — cached on first call. */
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

static uint64_t _fb_cross = 0;
static uint64_t fb_get_crossover(void) {
    if (_fb_cross) return _fb_cross;
    const char* s = getenv("AB_CROSSOVER_FLOPS");
    unsigned long long v = (s && *s) ? strtoull(s, NULL, 10) : DEFAULT_CROSSOVER_FLOPS;
    if (v == 0) v = DEFAULT_CROSSOVER_FLOPS;
    _fb_cross = (uint64_t)v;
    return _fb_cross;
}

/* Convert Fortran trans char to CBLAS enum. */
static inline enum CBLAS_TRANSPOSE trans_to_cblas(char t) {
    if (t == 'N' || t == 'n') return CblasNoTrans;
    if (t == 'T' || t == 't') return CblasTrans;
    return CblasConjTrans;  /* 'C'/'c' */
}

/* =============================================================================
 * dgemm_ — shadows Accelerate's dgemm_
 * =============================================================================
 */
void dgemm_(
    const char *transA, const char *transB,
    const int *M, const int *N, const int *K,
    const double *alpha, const double *A, const int *ldA,
    const double *B, const int *ldB,
    const double *beta, double *C, const int *ldC)
{
    int mode = fb_get_mode();
    uint64_t flops = (uint64_t)(*M) * (*N) * (*K) * 2ULL;
    int gpu = (mode == FB_CPU) ? 0
            : (mode == FB_GPU) ? 1
            : (flops >= fb_get_crossover());

    uint64_t t0 = ab_prof_enabled() ? ab_prof_now_ticks() : 0;

    if (gpu) {
        ab_dgemm_blas(*transA, *transB, *M, *N, *K,
                      *alpha, A, *ldA, B, *ldB, *beta, C, *ldC);
    } else {
        cblas_dgemm(CblasColMajor,
                    trans_to_cblas(*transA), trans_to_cblas(*transB),
                    *M, *N, *K,
                    *alpha, A, *ldA, B, *ldB, *beta, C, *ldC);
    }

    if (ab_prof_enabled()) {
        uint64_t ns = ab_prof_ticks_to_ns(ab_prof_now_ticks() - t0);
        ab_prof_record("dgemm", ns, (size_t)*M, (size_t)*N, (size_t)*K, gpu);
    }
}

/* =============================================================================
 * zgemm_ — shadows Accelerate's zgemm_
 * =============================================================================
 */
void zgemm_(
    const char *transA, const char *transB,
    const int *M, const int *N, const int *K,
    const double complex *alpha, const double complex *A, const int *ldA,
    const double complex *B, const int *ldB,
    const double complex *beta, double complex *C, const int *ldC)
{
    int mode = fb_get_mode();
    uint64_t flops = (uint64_t)(*M) * (*N) * (*K) * 8ULL;
    int gpu = (mode == FB_CPU) ? 0
            : (mode == FB_GPU) ? 1
            : (flops >= fb_get_crossover());

    uint64_t t0 = ab_prof_enabled() ? ab_prof_now_ticks() : 0;

    if (gpu) {
        ab_zgemm_blas(*transA, *transB, *M, *N, *K,
                      *alpha, A, *ldA, B, *ldB, *beta, C, *ldC);
    } else {
        cblas_zgemm(CblasColMajor,
                    trans_to_cblas(*transA), trans_to_cblas(*transB),
                    *M, *N, *K,
                    alpha, A, *ldA, B, *ldB, beta, C, *ldC);
    }

    if (ab_prof_enabled()) {
        uint64_t ns = ab_prof_ticks_to_ns(ab_prof_now_ticks() - t0);
        ab_prof_record("zgemm", ns, (size_t)*M, (size_t)*N, (size_t)*K, gpu);
    }
}

/* =============================================================================
 * Legacy ab_*_ prefixed names (thin wrappers)
 * =============================================================================
 * Darwin doesn't support __attribute__((alias)), so these are explicit
 * forwarding functions. Any code patched during the transition
 * (CALL ZGEMM → CALL ab_zgemm) continues to resolve.
 */
void ab_dgemm_(
    const char *transA, const char *transB,
    const int *M, const int *N, const int *K,
    const double *alpha, const double *A, const int *ldA,
    const double *B, const int *ldB,
    const double *beta, double *C, const int *ldC)
{
    dgemm_(transA, transB, M, N, K, alpha, A, ldA, B, ldB, beta, C, ldC);
}

void ab_zgemm_(
    const char *transA, const char *transB,
    const int *M, const int *N, const int *K,
    const double complex *alpha, const double complex *A, const int *ldA,
    const double complex *B, const int *ldB,
    const double complex *beta, double complex *C, const int *ldC)
{
    zgemm_(transA, transB, M, N, K, alpha, A, ldA, B, ldB, beta, C, ldC);
}
