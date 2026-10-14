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
 * Each call forwards to ab_dgemm_blas / ab_zgemm_blas (blas_wrapper.c),
 * which route it (AB_MODE and the calibrated table, dispatch.h), time it for
 * AB_PROFILE / AB_REPORT, and run it on the GPU or on Accelerate through the
 * CBLAS interface, so we never self-recurse into our own dgemm_/zgemm_.
 *
 * The legacy ab_dgemm_ / ab_zgemm_ names are kept as weak aliases so any
 * call-site that was already patched during the transition continues to work.
 * =============================================================================
 */
#include "apple_bottom.h"
#include <complex.h>

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
    ab_dgemm_blas(*transA, *transB, *M, *N, *K, *alpha, A, *ldA, B, *ldB, *beta, C, *ldC);
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
    ab_zgemm_blas(*transA, *transB, *M, *N, *K, *alpha, A, *ldA, B, *ldB, *beta, C, *ldC);
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
