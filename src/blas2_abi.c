// =============================================================================
// blas2_abi.c — BLAS-2 entry points: BLAS ABI, Fortran symbols, device buffers
// =============================================================================
//
// dgemv_, zgemv_ and zhemv_ shadow Accelerate's as dgemm_ and zgemm_ do
// (fortran_bridge.c explains the asm labels) and forward to ab_dgemv_blas,
// ab_zgemv_blas and ab_zhemv_blas. Those run a call on Accelerate when an
// argument is invalid (cblas then rejects it as reference xerbla would), m or
// n is 0, alpha is 0, or alpha or beta is NaN or Inf; otherwise routing
// decides (dispatch.h steps 1-3: AB_MODE=cpu, AB_MIN_GPU_DIM, AB_MODE=gpu,
// then AB_MODE=auto's measured rule, kRoutes below). A GPU call that cannot
// finish there (the range guard, no device, A not mappable, a failed
// command buffer) reruns on Accelerate before y is touched, counted as a
// fallback.
//
// The device-buffer entry points (ab_dev_dgemv, ...) always try the GPU and
// fall back the same way, to Accelerate on the buffer's memory.
//
// AB_REPORT and AB_PROFILE count the BLAS-ABI calls as dgemv, zgemv and
// zhemv (K = 1); AB_TRACE does not trace them.
// =============================================================================

#include "apple_bottom.h"

#include <Accelerate/Accelerate.h>
#include <complex.h>
#include <ctype.h>
#include <math.h>

#include "blas2.h"
#include "blas_trans.h"
#include "device_internal.h"
#include "dispatch.h"

// =============================================================================
// AB_MODE=auto routing
// =============================================================================
// A row sends a call of `routine` whose op falls in class `op` ('N', or 'T'
// for both 'T' and 'C'; HEMV rows use 'N' for both triangles) to the GPU
// when min_elems <= m * n <= max_elems and min(m, n) >= min_dim. No row:
// Accelerate.
//
// Measured on an M2 Max (38-core GPU, macOS 27.0.1), staged GPU calls
// (ab_*_blas, AB_MODE=gpu, each call mapping A) against Accelerate cblas in
// one process, alpha 1, beta 0, tight lda, 3 warm-ups and the median of 7,
// twice: benchmarks/results/2026-10-19-317bb91-gemv/ for GEMV (sweep_square:
// 256-16384 square; sweep_rect: 4096 x 1024 to 262144 x 1024 and their
// transposes) and 2026-10-19-1bc503a-zhemv/ for ZHEMV (sweep_square, the
// single-read kernel); *_run1: the first run; tests/test_blas2.c checks the
// rows against both.
// A row covers only shapes measured >= 1.10x in both runs with no slower
// shape inside it, and stops at the largest such size: the cost of mapping A
// grows with its size (about 10 ms/GB at 1 GB, 22 ms/GB at 4 GB), so larger
// calls stay on Accelerate.
//   dgemv N   1.63-1.78x at 2048^2, 2.8-3.7x from 3072^2 to 16384^2;
//             rectangles of 2048^2 elements and up 1.12-3.46x
//   dgemv T   1.04-1.32x from 4096^2, 12288^2 at 1.12x and 1.096x: too close
//             to the line to route; no row
//   zgemv N   1.22-1.83x from 2048^2 to 12288^2; 16384^2 1.10x and 1.32x
//             (1.02x in a third run); 256 x 65536 1.07-1.08x
//   zgemv T,C at most 0.79x: no row
//   zhemv     2.35-4.51x from 2048 to 16384 (1.07-1.14x at 1536)
typedef struct {
    ab_b2_routine_t routine;
    char op;
    double min_elems, max_elems;
    int min_dim;
} Route;

static const Route kRoutes[] = {
    {AB_B2_DGEMV, 'N', 2048.0 * 2048, 16384.0 * 16384, 256},
    {AB_B2_ZGEMV, 'N', 2048.0 * 2048, 12288.0 * 12288, 512},
    {AB_B2_ZHEMV, 'N', 2048.0 * 2048, 16384.0 * 16384, 2048},
};

bool ab_internal_b2_auto_gpu(ab_b2_routine_t routine, char op, int m, int n) {
    char cls = routine == AB_B2_ZHEMV || op == 'N' ? 'N' : 'T';
    double elems = (double)m * n;
    for (size_t i = 0; i < sizeof kRoutes / sizeof *kRoutes; i++) {
        const Route *r = &kRoutes[i];
        if (r->routine == routine && r->op == cls && elems >= r->min_elems &&
            elems <= r->max_elems && (m < n ? m : n) >= r->min_dim)
            return true;
    }
    return false;
}

static bool routes_gpu(const ab_b2_call_t *c) {
    int by_mode = ab_dispatch_mode_route(ab_dispatch_process_config(), c->m < c->n ? c->m : c->n);
    return by_mode >= 0 ? by_mode : ab_internal_b2_auto_gpu(c->routine, c->op, c->m, c->n);
}

// =============================================================================
// Arguments and the Accelerate path
// =============================================================================

static enum CBLAS_UPLO blas_uplo(char u) {
    return u == 'U' ? CblasUpper : u == 'L' ? CblasLower : (enum CBLAS_UPLO)0;
}

static int max1(int v) { return v > 1 ? v : 1; }

// Reference BLAS argument checks (xerbla) of xGEMV and ZHEMV.
static bool args_valid(const ab_b2_call_t *c, int lda) {
    if (c->incx == 0 || c->incy == 0 || c->n < 0) return false;
    if (c->routine == AB_B2_ZHEMV) return blas_uplo(c->op) != (enum CBLAS_UPLO)0 && lda >= max1(c->n);
    return ab_blas_trans(c->op, true) != AB_TRANS_INVALID && c->m >= 0 && lda >= max1(c->m);
}

static bool finite2(const double v[2]) { return isfinite(v[0]) && isfinite(v[1]); }

// Whether a call with valid arguments is one the GPU path takes on at all.
static bool gpu_eligible(const ab_b2_call_t *c) {
    return c->m > 0 && c->n > 0 && (c->alpha[0] != 0.0 || c->alpha[1] != 0.0) &&
           finite2(c->alpha) && finite2(c->beta);
}

// The whole call on Accelerate; also for invalid arguments, which cblas
// rejects.
static void cpu_run(const ab_b2_call_t *c, const void *A, int lda) {
    ab_internal_set_last_path(false);
    const double complex *alpha = (const double complex *)c->alpha;
    const double complex *beta = (const double complex *)c->beta;
    switch (c->routine) {
    case AB_B2_DGEMV:
        cblas_dgemv(CblasColMajor, ab_blas_trans(c->op, false), c->m, c->n, c->alpha[0], A, lda,
                    c->x, c->incx, c->beta[0], c->y, c->incy);
        break;
    case AB_B2_ZGEMV:
        cblas_zgemv(CblasColMajor, ab_blas_trans(c->op, true), c->m, c->n, alpha, A, lda,
                    (const double complex *)c->x, c->incx, beta, (double complex *)c->y, c->incy);
        break;
    case AB_B2_ZHEMV:
        cblas_zhemv(CblasColMajor, blas_uplo(c->op), c->n, alpha, A, lda,
                    (const double complex *)c->x, c->incx, beta, (double complex *)c->y, c->incy);
        break;
    }
}

static ab_prof_fallback_t fallback_of(ab_b2_result_t r) {
    switch (r) {
    case AB_B2_RANGE:     return AB_PROF_FB_RANGE;
    case AB_B2_NO_DEVICE: return AB_PROF_FB_NO_DEVICE;
    case AB_B2_TOO_LARGE: return AB_PROF_FB_TOO_LARGE;
    case AB_B2_FAILED:    return AB_PROF_FB_GPU_FAILED;
    default:              return AB_PROF_FB_OTHER;
    }
}

static const char *result_name(ab_b2_result_t r) {
    switch (r) {
    case AB_B2_RANGE:       return "NaN, Inf or values beyond double-float's range";
    case AB_B2_NO_DEVICE:   return "no Metal device";
    case AB_B2_TOO_LARGE:   return "A larger than the device's maxBufferLength";
    case AB_B2_UNSUPPORTED: return "A could not be mapped for the GPU";
    case AB_B2_FAILED:      return "GPU step failed";
    default:                return "gpu";
    }
}

// =============================================================================
// BLAS ABI
// =============================================================================

static const char *const kRoutineName[] = {"dgemv", "zgemv", "zhemv"};

// One BLAS-ABI call, timed for AB_PROFILE / AB_REPORT when either is set.
static void blas_call(const ab_b2_call_t *c, const void *A, int lda) {
    bool timed = ab_prof_enabled();
    uint64_t t0 = timed ? ab_prof_now_ticks() : 0, stage_ns[AB_PROF_STAGE_COUNT] = {0};
    bool tried = args_valid(c, lda) && gpu_eligible(c) && routes_gpu(c);
    ab_b2_result_t result = tried ? ab_internal_b2_host(c, A, lda, timed ? stage_ns : NULL)
                                  : AB_B2_NO_DEVICE;
    bool gpu = result == AB_B2_GPU;
    if (gpu) ab_internal_set_last_path(true);
    else cpu_run(c, A, lda);
    if (!timed) return;

    const char *name = kRoutineName[c->routine];
    int m = c->m > 0 ? c->m : 0, n = c->n > 0 ? c->n : 0;
    uint64_t ns = ab_prof_ticks_to_ns(ab_prof_now_ticks() - t0);
    ab_prof_record(name, ns, (size_t)m, (size_t)n, 1, gpu, gpu ? stage_ns : NULL,
                   tried && !gpu ? fallback_of(result) : AB_PROF_FB_NONE);
    if (gpu) ab_prof_print_gpu_call(name, m, n, 1, ns, stage_ns);
    else if (tried) ab_prof_print_fallback(name, m, n, 1, result_name(result));
}

static char flag(char c) { return (char)toupper((unsigned char)c); }

void ab_dgemv_blas(char trans, int M, int N, double alpha, const double *A, int ldA,
                   const double *x, int incx, double beta, double *y, int incy) {
    ab_b2_call_t c = {AB_B2_DGEMV, flag(trans), M, N, {alpha, 0.0}, {beta, 0.0}, x, incx, y, incy};
    blas_call(&c, A, ldA);
}

void ab_zgemv_blas(char trans, int M, int N, double _Complex alpha, const double _Complex *A,
                   int ldA, const double _Complex *x, int incx, double _Complex beta,
                   double _Complex *y, int incy) {
    ab_b2_call_t c = {AB_B2_ZGEMV, flag(trans), M, N, {creal(alpha), cimag(alpha)},
                      {creal(beta), cimag(beta)}, (const double *)x, incx, (double *)y, incy};
    blas_call(&c, A, ldA);
}

void ab_zhemv_blas(char uplo, int N, double _Complex alpha, const double _Complex *A, int ldA,
                   const double _Complex *x, int incx, double _Complex beta, double _Complex *y,
                   int incy) {
    ab_b2_call_t c = {AB_B2_ZHEMV, flag(uplo), N, N, {creal(alpha), cimag(alpha)},
                      {creal(beta), cimag(beta)}, (const double *)x, incx, (double *)y, incy};
    blas_call(&c, A, ldA);
}

// Fortran symbols, bound to the plain names as in fortran_bridge.c.
void fortran_dgemv(const char *trans, const int *m, const int *n, const double *alpha,
                   const double *A, const int *lda, const double *x, const int *incx,
                   const double *beta, double *y, const int *incy) __asm__("_dgemv_");
void fortran_dgemv(const char *trans, const int *m, const int *n, const double *alpha,
                   const double *A, const int *lda, const double *x, const int *incx,
                   const double *beta, double *y, const int *incy) {
    ab_dgemv_blas(*trans, *m, *n, *alpha, A, *lda, x, *incx, *beta, y, *incy);
}

void fortran_zgemv(const char *trans, const int *m, const int *n, const double _Complex *alpha,
                   const double _Complex *A, const int *lda, const double _Complex *x,
                   const int *incx, const double _Complex *beta, double _Complex *y,
                   const int *incy) __asm__("_zgemv_");
void fortran_zgemv(const char *trans, const int *m, const int *n, const double _Complex *alpha,
                   const double _Complex *A, const int *lda, const double _Complex *x,
                   const int *incx, const double _Complex *beta, double _Complex *y,
                   const int *incy) {
    ab_zgemv_blas(*trans, *m, *n, *alpha, A, *lda, x, *incx, *beta, y, *incy);
}

void fortran_zhemv(const char *uplo, const int *n, const double _Complex *alpha,
                   const double _Complex *A, const int *lda, const double _Complex *x,
                   const int *incx, const double _Complex *beta, double _Complex *y,
                   const int *incy) __asm__("_zhemv_");
void fortran_zhemv(const char *uplo, const int *n, const double _Complex *alpha,
                   const double _Complex *A, const int *lda, const double _Complex *x,
                   const int *incx, const double _Complex *beta, double _Complex *y,
                   const int *incy) {
    ab_zhemv_blas(*uplo, *n, *alpha, A, *lda, x, *incx, *beta, y, *incy);
}

// =============================================================================
// Device buffers
// =============================================================================

static ABStatus dev_call(const ab_b2_call_t *c, ab_dev_buffer_t A, size_t a_offset, int lda) {
    size_t elem = ab_b2_elem_size(c->routine);
    if (!args_valid(c, lda) || !c->x || !c->y ||
        !ab_internal_dev_matrix_fits(A, a_offset, lda, c->m, c->n, elem))
        return AB_ERROR_INVALID_ARG;
    if (gpu_eligible(c) && ab_internal_b2_dev(c, A, a_offset, lda) == AB_B2_GPU)
        ab_internal_set_last_path(true);
    else
        cpu_run(c, (const char *)ab_internal_dev_data(A) + a_offset * elem, lda);
    return AB_OK;
}

static char trans_flag(ABTranspose t) {
    return t == AB_NO_TRANS ? 'N' : t == AB_TRANS ? 'T' : t == AB_CONJ_TRANS ? 'C' : '?';
}

ABStatus ab_dev_dgemv(ABTranspose trans, int m, int n, double alpha, ab_dev_buffer_t A,
                      size_t a_offset, int lda, const double *x, int incx, double beta, double *y,
                      int incy) {
    ab_b2_call_t c = {AB_B2_DGEMV, trans_flag(trans), m, n, {alpha, 0.0}, {beta, 0.0}, x, incx, y,
                      incy};
    return dev_call(&c, A, a_offset, lda);
}

ABStatus ab_dev_zgemv(ABTranspose trans, int m, int n, const double alpha[2], ab_dev_buffer_t A,
                      size_t a_offset, int lda, const double *x, int incx, const double beta[2],
                      double *y, int incy) {
    if (!alpha || !beta) return AB_ERROR_INVALID_ARG;
    ab_b2_call_t c = {AB_B2_ZGEMV, trans_flag(trans), m, n, {alpha[0], alpha[1]},
                      {beta[0], beta[1]}, x, incx, y, incy};
    return dev_call(&c, A, a_offset, lda);
}

ABStatus ab_dev_zhemv(ABUplo uplo, int n, const double alpha[2], ab_dev_buffer_t A,
                      size_t a_offset, int lda, const double *x, int incx, const double beta[2],
                      double *y, int incy) {
    if (!alpha || !beta) return AB_ERROR_INVALID_ARG;
    char u = uplo == AB_UPPER ? 'U' : uplo == AB_LOWER ? 'L' : '?';
    ab_b2_call_t c = {AB_B2_ZHEMV, u, n, n, {alpha[0], alpha[1]}, {beta[0], beta[1]}, x, incx, y,
                      incy};
    return dev_call(&c, A, a_offset, lda);
}
