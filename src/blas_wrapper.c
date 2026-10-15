// =============================================================================
// blas_wrapper.c
// BLAS-compatible interface for Fortran interoperability
// =============================================================================

#include "apple_bottom.h"
#include <complex.h>
#include <stdlib.h>
#include <string.h>
#include <stdbool.h>
#include <Accelerate/Accelerate.h>
#include <stdio.h>

#include "blas_trans.h"
#include "dispatch.h"
#include "profiling/blas_profiler.h"

// Thread-local record of the path the last BLAS-ABI call on this thread
// actually took, for bench harnesses and the profiler (prof_end). Only the
// two places that produce C write it: "cpu" in ab_internal_*gemm_cpu, "gpu"
// once the GPU result is in C (H-P1-2). Routing (dispatch.c) never touches
// it. Zero runtime cost.
static __thread const char *_ab_last_path = "none";
const char* ab_get_last_dispatch_path(void) { return _ab_last_path; }

// =============================================================================
// Host-side helpers shared by the BLAS-ABI entry points
// =============================================================================
// Inputs are column-major with leading dimension ld; ABMatrix is row-major.
// All index arithmetic is size_t: ld*cols can exceed 2^31 even when every
// operand is within the ABMatrix size limits.

// Pack op(X), rows x cols, into row-major dst. op(X)[i][j] is X[j*ld+i] for
// CblasNoTrans and X[i*ld+j] otherwise; each loop order walks X contiguously.
static void pack_op_real(enum CBLAS_TRANSPOSE t, const double* X, int ld,
                         int rows, int cols, double* dst) {
    size_t r = (size_t)rows, c = (size_t)cols, l = (size_t)ld;
    if (t == CblasNoTrans) {
        for (size_t j = 0; j < c; j++)
            for (size_t i = 0; i < r; i++)
                dst[i * c + j] = X[j * l + i];
    } else {
        for (size_t i = 0; i < r; i++)
            for (size_t j = 0; j < c; j++)
                dst[i * c + j] = X[i * l + j];
    }
}

// Complex variant: split op(X) into row-major real and imaginary parts,
// conjugating for CblasConjTrans.
static void pack_op_complex(enum CBLAS_TRANSPOSE t, const double complex* X, int ld,
                            int rows, int cols, double* re, double* im) {
    size_t r = (size_t)rows, c = (size_t)cols, l = (size_t)ld;
    if (t == CblasNoTrans) {
        for (size_t j = 0; j < c; j++)
            for (size_t i = 0; i < r; i++) {
                double complex val = X[j * l + i];
                re[i * c + j] = creal(val);
                im[i * c + j] = cimag(val);
            }
    } else {
        bool conjugate = (t == CblasConjTrans);
        for (size_t i = 0; i < r; i++)
            for (size_t j = 0; j < c; j++) {
                double complex val = X[i * l + j];
                if (conjugate) val = conj(val);
                re[i * c + j] = creal(val);
                im[i * c + j] = cimag(val);
            }
    }
}

// The gate of dgemm_blas and zgemm_blas (dispatch.h): Accelerate where the
// routing says so, and for the BLAS quick return for alpha == 0 (A and B must
// not be read) and invalid arguments.
bool ab_internal_blas_routes_gpu(char routine, char transA, char transB, int M, int N, int K,
                                 int ldA, int ldB, int ldC, bool alpha_nonzero) {
    bool z = routine == 'z';
    return ab_blas_args_valid(ab_blas_trans(transA, z), ab_blas_trans(transB, z),
                              M, N, K, ldA, ldB, ldC) &&
           alpha_nonzero && ab_dispatch_route_gpu(routine, transA, transB, M, N, K);
}

// Whole-call CPU paths (Accelerate cblas, column-major). Used for calls the
// routing keeps on Accelerate, for BLAS quick returns and invalid arguments,
// and as the fallback when any GPU step fails before C is written, so C
// always gets a correct result.
static void ab_internal_dgemm_cpu(enum CBLAS_TRANSPOSE ta, enum CBLAS_TRANSPOSE tb,
                                  int M, int N, int K, double alpha,
                                  const double* A, int ldA, const double* B, int ldB,
                                  double beta, double* C, int ldC) {
    _ab_last_path = "cpu";
    cblas_dgemm(CblasColMajor, ta, tb, M, N, K, alpha, A, ldA, B, ldB, beta, C, ldC);
}

static void ab_internal_zgemm_cpu(enum CBLAS_TRANSPOSE ta, enum CBLAS_TRANSPOSE tb,
                                  int M, int N, int K, double complex alpha,
                                  const double complex* A, int ldA,
                                  const double complex* B, int ldB,
                                  double complex beta, double complex* C, int ldC) {
    _ab_last_path = "cpu";
    cblas_zgemm(CblasColMajor, ta, tb, M, N, K, &alpha, A, ldA, B, ldB, &beta, C, ldC);
}

// =============================================================================
// Call profile (AB_PROFILE, AB_REPORT, AB_TRACE)
// =============================================================================
// Every BLAS-ABI call is timed once, around routing and compute, and recorded
// by where C was computed (the path record). The GPU path charges each step
// to a stage as it finishes. p is NULL when profiling and tracing are off,
// so such a call pays one branch per stage.

typedef struct {
    uint64_t t0, mark;                         // ticks: call start, end of the last stage
    uint64_t stage_ns[AB_PROF_STAGE_COUNT];
    ab_prof_fallback_t fallback;               // why a GPU-routed call ran on cblas
    bool trace;                                // AB_TRACE: write the call's trace line
    ab_trace_call_t call;                      // its arguments, hashes and kernel split
} CallProf;

static CallProf *prof_begin(CallProf *p) {
    if (!ab_prof_active()) return NULL;
    memset(p, 0, sizeof *p);
    p->fallback = AB_PROF_FB_NONE;
    p->trace = ab_trace_enabled();
    p->t0 = p->mark = ab_prof_now_ticks();
    (void)ab_internal_take_gpu_seconds(NULL); // drop GPU time of earlier native-API calls
    return p;
}

static void prof_stage(CallProf *p, ab_prof_stage_t s) {
    if (!p) return;
    uint64_t now = ab_prof_now_ticks();
    p->stage_ns[s] += ab_prof_ticks_to_ns(now - p->mark);
    p->mark = now;
}

// The kernel stage (ab_prof_take_kernel_ns); the rest of the kernel call is
// left to "other", which AB_TRACE splits.
static void prof_kernel(CallProf *p) {
    if (!p) return;
    uint64_t now = ab_prof_now_ticks();
    p->stage_ns[AB_PROF_STAGE_KERNEL] +=
        ab_prof_take_kernel_ns(ab_prof_ticks_to_ns(now - p->mark), &p->call.split);
    p->mark = now;
}

// Records the finished call; under AB_PROFILE also prints the stages of a GPU
// call, under AB_TRACE writes its trace line.
static void prof_end(CallProf *p, const char *routine, int M, int N, int K) {
    if (!p) return;
    uint64_t ns = ab_prof_ticks_to_ns(ab_prof_now_ticks() - p->t0);
    bool gpu = !strcmp(_ab_last_path, "gpu");
    ab_prof_record(routine, ns, M > 0 ? (size_t)M : 0, N > 0 ? (size_t)N : 0,
                   K > 0 ? (size_t)K : 0, gpu, gpu ? p->stage_ns : NULL, p->fallback);
    if (p->trace)
        ab_trace_record(&p->call, ab_prof_ticks_to_ns(p->t0), ns, gpu ? p->stage_ns : NULL,
                        p->fallback);
    if (!gpu || !ab_prof_verbose()) return;
    // One write per line, so lines from concurrent callers do not interleave.
    char line[512];
    size_t n = (size_t)snprintf(line, sizeof line, "AB_PROFILE %s M=%d N=%d K=%d ", routine, M, N, K);
    uint64_t named = 0;
    for (int s = 0; s < AB_PROF_STAGE_COUNT && n < sizeof line; s++) {
        named += p->stage_ns[s];
        n += (size_t)snprintf(line + n, sizeof line - n, " %s=%.3f", ab_prof_stage_names[s],
                              p->stage_ns[s] / 1e6);
    }
    if (n < sizeof line)
        snprintf(line + n, sizeof line - n, " other=%.3f  total=%.3f ms\n",
                 (ns > named ? ns - named : 0) / 1e6, ns / 1e6);
    fputs(line, stderr);
}

// A GPU-routed call that has to run on cblas instead. The path record says
// "cpu" either way; the profile counts the reason and AB_PROFILE prints it,
// so a fallback is not mistaken for a routing decision (H-P2-8).
static void note_gpu_fallback(CallProf *p, const char *routine, int M, int N, int K,
                              ab_prof_fallback_t reason, const char *why) {
    if (!p) return;
    p->fallback = reason;
    if (ab_prof_verbose())
        fprintf(stderr, "AB_PROFILE %s M=%d N=%d K=%d  GPU fallback to cblas: %s\n",
                routine, M, N, K, why);
}

// FNV-1a over the 64-bit words of the referenced elements of a stored
// rows x cols operand (column-major, leading dimension ld, elem = 8 or 16
// bytes), column by column: from the 64-bit offset basis, h = (h ^ word) *
// FNV prime per word. The padding below row `rows` is not read.
static uint64_t hash_stored(const void *X, size_t elem, int ld, int rows, int cols) {
    const unsigned char *x = X;
    size_t col_words = (size_t)rows * (elem / sizeof(uint64_t));
    uint64_t h = 0xcbf29ce484222325ULL;
    for (size_t j = 0; j < (size_t)cols; j++) {
        const unsigned char *col = x + j * (size_t)ld * elem;
        for (size_t w = 0; w < col_words; w++) {
            uint64_t word;
            memcpy(&word, col + w * sizeof word, sizeof word);
            h = (h ^ word) * 0x100000001b3ULL;
        }
    }
    return h;
}

// AB_TRACE: the call's arguments as passed and, under AB_TRACE_HASH, the
// hashes of what it reads: op(A) and op(B) as stored unless the BLAS quick
// return skips them (M, N or K == 0, alpha == 0), and C on entry when
// beta != 0. Invalid arguments are not hashed (cblas rejects the call). The
// clock restarts after hashing, so the hashes cost the call nothing.
static void trace_begin(CallProf *p, const char *routine, char transA, char transB,
                        int M, int N, int K, const double alpha[2],
                        const void *A, int ldA, const void *B, int ldB,
                        const double beta[2], const void *C, int ldC) {
    bool z = routine[0] == 'z';
    ab_trace_call_t *c = &p->call;
    c->routine = routine;
    c->ta = transA;
    c->tb = transB;
    c->m = M; c->n = N; c->k = K;
    c->lda = ldA; c->ldb = ldB; c->ldc = ldC;
    memcpy(c->alpha, alpha, sizeof c->alpha);
    memcpy(c->beta, beta, sizeof c->beta);
    if (!ab_trace_hash_enabled()) return;

    enum CBLAS_TRANSPOSE ta = ab_blas_trans(transA, z), tb = ab_blas_trans(transB, z);
    size_t elem = z ? sizeof(double complex) : sizeof(double);
    if (ab_blas_args_valid(ta, tb, M, N, K, ldA, ldB, ldC) && M > 0 && N > 0) {
        if (K > 0 && (alpha[0] != 0.0 || alpha[1] != 0.0)) {
            bool na = ta == CblasNoTrans, nb = tb == CblasNoTrans;
            c->hash[0] = hash_stored(A, elem, ldA, na ? M : K, na ? K : M);
            c->hash[1] = hash_stored(B, elem, ldB, nb ? K : N, nb ? N : K);
            c->hashed[0] = c->hashed[1] = true;
        }
        if (beta[0] != 0.0 || beta[1] != 0.0) {
            c->hash[2] = hash_stored(C, elem, ldC, M, N);
            c->hashed[2] = true;
        }
    }
    p->t0 = p->mark = ab_prof_now_ticks();
}

// Hosts that reach us through dgemm_/zgemm_ (QE, Yambo) never call ab_init(),
// so create the Metal context on first GPU use (H-P1-2); ab_init() is
// idempotent and thread-safe. It fails for no device and in a fork() child.
// Then op(A) (M x K), op(B) (K x N) and C (M x N) must each fit the ABMatrix
// size limits.
static ABStatus gpu_ready(int M, int N, int K) {
    ABStatus st = ab_init();
    if (st == AB_OK) st = ab_matrix_check_size(M, K);
    if (st == AB_OK) st = ab_matrix_check_size(K, N);
    if (st == AB_OK) st = ab_matrix_check_size(M, N);
    return st;
}

static ab_prof_fallback_t fallback_reason(ABStatus st) {
    return st == AB_ERROR_NO_DEVICE ? AB_PROF_FB_NO_DEVICE
         : st == AB_ERROR_TOO_LARGE ? AB_PROF_FB_TOO_LARGE : AB_PROF_FB_OTHER;
}

// =============================================================================
// BLAS-compatible ZGEMM - this is what Fortran calls
// Handles: transA, transB, alpha, beta, leading dimensions
// =============================================================================
static void zgemm_blas(
    CallProf *p,
    char transA, char transB,
    int M, int N, int K,
    double complex alpha,
    const double complex* A, int ldA,
    const double complex* B, int ldB,
    double complex beta,
    double complex* C, int ldC
) {
    enum CBLAS_TRANSPOSE ta = ab_blas_trans(transA, true);
    enum CBLAS_TRANSPOSE tb = ab_blas_trans(transB, true);

    if (!ab_internal_blas_routes_gpu('z', transA, transB, M, N, K, ldA, ldB, ldC, alpha != 0.0)) {
        ab_internal_zgemm_cpu(ta, tb, M, N, K, alpha, A, ldA, B, ldB, beta, C, ldC);
        return;
    }

    ABStatus ready = gpu_ready(M, N, K);
    if (ready != AB_OK) {
        note_gpu_fallback(p, "zgemm", M, N, K, fallback_reason(ready), ab_status_string(ready));
        ab_internal_zgemm_cpu(ta, tb, M, N, K, alpha, A, ldA, B, ldB, beta, C, ldC);
        return;
    }
    prof_stage(p, AB_PROF_STAGE_INIT);

    // GPU path: compute op(A)*op(B) on GPU, apply alpha/beta in epilogue
    bool ok = false;
    double *Ar_data = NULL, *Ai_data = NULL, *Br_data = NULL, *Bi_data = NULL;
    double *Cr_data = NULL, *Ci_data = NULL;

    // Allocate split-complex GPU matrices: op(A) is M×K, op(B) is K×N
    ABMatrix mAr = ab_matrix_create(M, K);
    ABMatrix mAi = ab_matrix_create(M, K);
    ABMatrix mBr = ab_matrix_create(K, N);
    ABMatrix mBi = ab_matrix_create(K, N);
    ABMatrix mCr = ab_matrix_create(M, N);
    ABMatrix mCi = ab_matrix_create(M, N);
    if (!mAr || !mAi || !mBr || !mBi || !mCr || !mCi) goto cleanup;

    // Upload op(A) (column-major to row-major, transposed/conjugated as asked)
    Ar_data = malloc((size_t)M * K * sizeof(double));
    Ai_data = malloc((size_t)M * K * sizeof(double));
    if (!Ar_data || !Ai_data) goto cleanup;
    prof_stage(p, AB_PROF_STAGE_ALLOC);
    pack_op_complex(ta, A, ldA, M, K, Ar_data, Ai_data);
    prof_stage(p, AB_PROF_STAGE_PACK);
    if (ab_matrix_upload(mAr, Ar_data, true) != AB_OK ||
        ab_matrix_upload(mAi, Ai_data, true) != AB_OK) goto cleanup;
    prof_stage(p, AB_PROF_STAGE_UPLOAD);
    free(Ar_data); free(Ai_data);
    Ar_data = Ai_data = NULL;

    // Upload op(B)
    Br_data = malloc((size_t)K * N * sizeof(double));
    Bi_data = malloc((size_t)K * N * sizeof(double));
    if (!Br_data || !Bi_data) goto cleanup;
    prof_stage(p, AB_PROF_STAGE_ALLOC);
    pack_op_complex(tb, B, ldB, K, N, Br_data, Bi_data);
    prof_stage(p, AB_PROF_STAGE_PACK);
    if (ab_matrix_upload(mBr, Br_data, true) != AB_OK ||
        ab_matrix_upload(mBi, Bi_data, true) != AB_OK) goto cleanup;
    prof_stage(p, AB_PROF_STAGE_UPLOAD);
    free(Br_data); free(Bi_data);
    Br_data = Bi_data = NULL;
    prof_stage(p, AB_PROF_STAGE_ALLOC);

    // Compute C = op(A) * op(B)
    if (ab_zgemm(mAr, mAi, mBr, mBi, mCr, mCi) != AB_OK) goto cleanup;
    prof_kernel(p);

    // Download result and apply alpha/beta epilogue
    Cr_data = malloc((size_t)M * N * sizeof(double));
    Ci_data = malloc((size_t)M * N * sizeof(double));
    if (!Cr_data || !Ci_data) goto cleanup;
    prof_stage(p, AB_PROF_STAGE_ALLOC);
    if (ab_matrix_download(mCr, Cr_data, true) != AB_OK ||
        ab_matrix_download(mCi, Ci_data, true) != AB_OK) goto cleanup;
    prof_stage(p, AB_PROF_STAGE_DOWNLOAD);

    // Convert row-major back to column-major with alpha/beta:
    // C = alpha * (A*B) + beta * C_old
    double ar = creal(alpha), ai = cimag(alpha);
    double br = creal(beta), bi = cimag(beta);
    bool needs_beta = (br != 0.0 || bi != 0.0);
    bool non_unit_alpha = (ar != 1.0 || ai != 0.0);

    for (size_t col = 0; col < (size_t)N; col++) {
        for (size_t row = 0; row < (size_t)M; row++) {
            size_t src = row * N + col;
            size_t dst = col * ldC + row;
            double pr = Cr_data[src], pi = Ci_data[src];

            // Apply alpha: alpha * (A*B)
            double result_r, result_i;
            if (non_unit_alpha) {
                result_r = ar * pr - ai * pi;
                result_i = ar * pi + ai * pr;
            } else {
                result_r = pr;
                result_i = pi;
            }

            // Apply beta: + beta * C_old
            if (needs_beta) {
                double complex c_old = C[dst];
                result_r += br * creal(c_old) - bi * cimag(c_old);
                result_i += br * cimag(c_old) + bi * creal(c_old);
            }

            C[dst] = result_r + I * result_i;
        }
    }
    ok = true;
    prof_stage(p, AB_PROF_STAGE_EPILOGUE);

cleanup:
    free(Ar_data); free(Ai_data); free(Br_data); free(Bi_data);
    free(Cr_data); free(Ci_data);
    ab_matrix_destroy(mAr); ab_matrix_destroy(mAi);
    ab_matrix_destroy(mBr); ab_matrix_destroy(mBi);
    ab_matrix_destroy(mCr); ab_matrix_destroy(mCi);
    prof_stage(p, AB_PROF_STAGE_ALLOC);

    // Any failure happened before C was written: redo the whole call on CPU.
    if (!ok) {
        note_gpu_fallback(p, "zgemm", M, N, K, AB_PROF_FB_GPU_FAILED, "GPU step failed");
        ab_internal_zgemm_cpu(ta, tb, M, N, K, alpha, A, ldA, B, ldB, beta, C, ldC);
        return;
    }
    _ab_last_path = "gpu";
}

// =============================================================================
// BLAS-compatible DGEMM: C = alpha * op(A) * op(B) + beta * C
// Handles transA, transB, alpha, beta, leading dimensions.
// GPU path supports all transpose combinations via upload-time reordering.
// =============================================================================
static void dgemm_blas(
    CallProf *p,
    char transA, char transB,
    int M, int N, int K,
    double alpha,
    const double* A, int ldA,
    const double* B, int ldB,
    double beta,
    double* C, int ldC
) {
    enum CBLAS_TRANSPOSE ta = ab_blas_trans(transA, false);
    enum CBLAS_TRANSPOSE tb = ab_blas_trans(transB, false);

    if (!ab_internal_blas_routes_gpu('d', transA, transB, M, N, K, ldA, ldB, ldC, alpha != 0.0)) {
        ab_internal_dgemm_cpu(ta, tb, M, N, K, alpha, A, ldA, B, ldB, beta, C, ldC);
        return;
    }

    ABStatus ready = gpu_ready(M, N, K);
    if (ready != AB_OK) {
        note_gpu_fallback(p, "dgemm", M, N, K, fallback_reason(ready), ab_status_string(ready));
        ab_internal_dgemm_cpu(ta, tb, M, N, K, alpha, A, ldA, B, ldB, beta, C, ldC);
        return;
    }
    prof_stage(p, AB_PROF_STAGE_INIT);

    // GPU path: supports arbitrary alpha/beta and transpose via scaled kernel
    bool ok = false;
    double *A_row = NULL, *B_row = NULL, *C_row = NULL;
    ABMatrix mA = ab_matrix_create(M, K);
    ABMatrix mB = ab_matrix_create(K, N);
    ABMatrix mC = ab_matrix_create(M, N);
    if (!mA || !mB || !mC) goto cleanup;

    // Upload op(A) as M×K and op(B) as K×N, row-major
    A_row = malloc((size_t)M * K * sizeof(double));
    if (!A_row) goto cleanup;
    prof_stage(p, AB_PROF_STAGE_ALLOC);
    pack_op_real(ta, A, ldA, M, K, A_row);
    prof_stage(p, AB_PROF_STAGE_PACK);
    if (ab_matrix_upload(mA, A_row, true) != AB_OK) goto cleanup;
    prof_stage(p, AB_PROF_STAGE_UPLOAD);
    free(A_row);
    A_row = NULL;

    B_row = malloc((size_t)K * N * sizeof(double));
    if (!B_row) goto cleanup;
    prof_stage(p, AB_PROF_STAGE_ALLOC);
    pack_op_real(tb, B, ldB, K, N, B_row);
    prof_stage(p, AB_PROF_STAGE_PACK);
    if (ab_matrix_upload(mB, B_row, true) != AB_OK) goto cleanup;
    prof_stage(p, AB_PROF_STAGE_UPLOAD);
    free(B_row);
    B_row = NULL;

    // Upload existing C only when beta != 0 (C = alpha*A*B + beta*C); with
    // beta == 0 C is never read, per BLAS (the kernel skips beta*C).
    C_row = malloc((size_t)M * N * sizeof(double));
    if (!C_row) goto cleanup;
    prof_stage(p, AB_PROF_STAGE_ALLOC);
    if (beta != 0.0) {
        pack_op_real(CblasNoTrans, C, ldC, M, N, C_row);
        prof_stage(p, AB_PROF_STAGE_PACK);
        if (ab_matrix_upload(mC, C_row, true) != AB_OK) goto cleanup;
        prof_stage(p, AB_PROF_STAGE_UPLOAD);
    }

    // Compute C = alpha*A*B + beta*C on GPU
    if (ab_dgemm_scaled(alpha, mA, mB, beta, mC) != AB_OK) goto cleanup;
    prof_kernel(p);

    // Download result (row-major to column-major)
    if (ab_matrix_download(mC, C_row, true) != AB_OK) goto cleanup;
    prof_stage(p, AB_PROF_STAGE_DOWNLOAD);
    for (size_t j = 0; j < (size_t)N; j++)
        for (size_t i = 0; i < (size_t)M; i++)
            C[j * ldC + i] = C_row[i * N + j];
    ok = true;
    prof_stage(p, AB_PROF_STAGE_EPILOGUE);

cleanup:
    free(A_row); free(B_row); free(C_row);
    ab_matrix_destroy(mA);
    ab_matrix_destroy(mB);
    ab_matrix_destroy(mC);
    prof_stage(p, AB_PROF_STAGE_ALLOC);

    // Any failure happened before C was written: redo the whole call on CPU.
    if (!ok) {
        note_gpu_fallback(p, "dgemm", M, N, K, AB_PROF_FB_GPU_FAILED, "GPU step failed");
        ab_internal_dgemm_cpu(ta, tb, M, N, K, alpha, A, ldA, B, ldB, beta, C, ldC);
        return;
    }
    _ab_last_path = "gpu";
}

// =============================================================================
// BLAS-ABI entry points
// =============================================================================
// Every BLAS-ABI call ends here: dgemm_, zgemm_ and ab_*gemm_ forward to
// these (fortran_bridge.c). Each call is timed and recorded for AB_PROFILE
// and AB_REPORT, and traced for AB_TRACE.

void ab_dgemm_blas(char transA, char transB, int M, int N, int K,
                   double alpha, const double* A, int ldA,
                   const double* B, int ldB,
                   double beta, double* C, int ldC) {
    CallProf prof, *p = prof_begin(&prof);
    if (p && p->trace)
        trace_begin(p, "dgemm", transA, transB, M, N, K, (const double[2]){alpha, 0.0},
                    A, ldA, B, ldB, (const double[2]){beta, 0.0}, C, ldC);
    dgemm_blas(p, transA, transB, M, N, K, alpha, A, ldA, B, ldB, beta, C, ldC);
    prof_end(p, "dgemm", M, N, K);
}

void ab_zgemm_blas(char transA, char transB, int M, int N, int K,
                   double complex alpha, const double complex* A, int ldA,
                   const double complex* B, int ldB,
                   double complex beta, double complex* C, int ldC) {
    CallProf prof, *p = prof_begin(&prof);
    if (p && p->trace)
        trace_begin(p, "zgemm", transA, transB, M, N, K,
                    (const double[2]){creal(alpha), cimag(alpha)}, A, ldA, B, ldB,
                    (const double[2]){creal(beta), cimag(beta)}, C, ldC);
    zgemm_blas(p, transA, transB, M, N, K, alpha, A, ldA, B, ldB, beta, C, ldC);
    prof_end(p, "zgemm", M, N, K);
}

// Fortran-callable wrappers (ab_zgemm_, ab_dgemm_) are in fortran_bridge.c
