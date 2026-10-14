/* =============================================================================
 * fortran_bridge.c — Fortran BLAS entry points for apple-bottom
 * =============================================================================
 *
 * Exports the *standard* Fortran BLAS symbols (dgemm_, zgemm_) so any
 * Fortran code linked against -lapplebottom picks us up without call-site
 * edits. On macOS two-level namespace, these symbols shadow Accelerate's
 * when -lapplebottom appears before -framework Accelerate.
 *
 * In C the entry points are named fortran_dgemm/fortran_zgemm and bound to
 * the exact symbols with asm labels. Under ACCELERATE_NEW_LAPACK,
 * <Accelerate/Accelerate.h> declares dgemm_/zgemm_ with
 * __asm("_dgemm$NEWLAPACK"), so a function spelled dgemm_ would be emitted
 * under that name and a Fortran CALL DGEMM (a reference to _dgemm_) would
 * bind to Accelerate (H-P0-1). The $NEWLAPACK names are deliberately not
 * exported: C code calling dgemm_ through the new Accelerate headers keeps
 * getting Accelerate.
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
#include <stdlib.h>

/* One BLAS-ABI call, timed and recorded for AB_PROFILE / AB_REPORT
 * (blas_wrapper.c): gpu = 0 runs cblas, gpu = 1 the ab_*gemm_blas path,
 * which applies its own routing and falls back to cblas if the GPU fails. */
extern void ab_internal_dgemm(int gpu, char transA, char transB, int M, int N, int K,
                              double alpha, const double* A, int ldA,
                              const double* B, int ldB,
                              double beta, double* C, int ldC);
extern void ab_internal_zgemm(int gpu, char transA, char transB, int M, int N, int K,
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

/* =============================================================================
 * dgemm_ — shadows Accelerate's dgemm_
 * =============================================================================
 */
void fortran_dgemm(
    const char *transA, const char *transB,
    const int *M, const int *N, const int *K,
    const double *alpha, const double *A, const int *ldA,
    const double *B, const int *ldB,
    const double *beta, double *C, const int *ldC) __asm__("_dgemm_");

void fortran_dgemm(
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

    ab_internal_dgemm(gpu, *transA, *transB, *M, *N, *K,
                      *alpha, A, *ldA, B, *ldB, *beta, C, *ldC);
}

/* =============================================================================
 * zgemm_ — shadows Accelerate's zgemm_
 * =============================================================================
 */
void fortran_zgemm(
    const char *transA, const char *transB,
    const int *M, const int *N, const int *K,
    const double complex *alpha, const double complex *A, const int *ldA,
    const double complex *B, const int *ldB,
    const double complex *beta, double complex *C, const int *ldC) __asm__("_zgemm_");

void fortran_zgemm(
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

    ab_internal_zgemm(gpu, *transA, *transB, *M, *N, *K,
                      *alpha, A, *ldA, B, *ldB, *beta, C, *ldC);
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
    fortran_dgemm(transA, transB, M, N, K, alpha, A, ldA, B, ldB, beta, C, ldC);
}

void ab_zgemm_(
    const char *transA, const char *transB,
    const int *M, const int *N, const int *K,
    const double complex *alpha, const double complex *A, const int *ldA,
    const double complex *B, const int *ldB,
    const double complex *beta, double complex *C, const int *ldC)
{
    fortran_zgemm(transA, transB, M, N, K, alpha, A, ldA, B, ldB, beta, C, ldC);
}
