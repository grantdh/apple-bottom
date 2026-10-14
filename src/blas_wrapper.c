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
#include "profiling/blas_profiler.h"

// GPU execution time of the command buffers this thread completed since the
// last call, in seconds; 0 if Metal reported none (apple_bottom.m).
extern double ab_internal_take_gpu_seconds(void);

// =============================================================================
// AMX/GPU Heterogeneous Dispatch Heuristic
// =============================================================================
// GPU dispatch has ~50μs overhead (command buffer creation, encode, commit).
// AMX (via Accelerate cblas) has near-zero dispatch overhead.
// Crossover point: GPU wins when compute time dominates dispatch overhead.
//
// Rules:
// 1. Any dimension ≤ 32: always CPU (AMX) — GPU threadgroups underutilized
// 2. FLOP threshold: 100M FLOPs for complex (ZGEMM), 50M for real (DGEMM)
//    Complex has 4x more arithmetic per element, so GPU wins at smaller N.
// 3. Skinny matrices (any dim < 64): CPU unless total FLOPs are very large
// =============================================================================

#define DEFAULT_CROSSOVER_FLOPS      100000000ULL  // For complex (ZGEMM)
#define DEFAULT_CROSSOVER_FLOPS_REAL  50000000ULL  // For real (DGEMM)
#define DEFAULT_MIN_GPU_DIM           32           // Minimum dimension for GPU dispatch

// Runtime-tunable thresholds (read once from env on first call, then cached).
//   AB_MIN_GPU_DIM       — dim floor below which GPU is never used (default 32)
//   AB_CROSSOVER_FLOPS   — FLOP threshold for complex routing (default 1e8)
//   AB_CROSSOVER_FLOPS_REAL — FLOP threshold for real routing (default 5e7)
// All negative/zero values are rejected and fall back to defaults.
static int      _ab_min_dim   = -1;
static uint64_t _ab_cross_z   = 0;  // 0 = uninitialized
static uint64_t _ab_cross_d   = 0;
static int ab_get_min_gpu_dim(void) {
    if (_ab_min_dim >= 0) return _ab_min_dim;
    const char* s = getenv("AB_MIN_GPU_DIM");
    int v = (s && *s) ? atoi(s) : DEFAULT_MIN_GPU_DIM;
    if (v < 0) v = DEFAULT_MIN_GPU_DIM;
    _ab_min_dim = v;
    return _ab_min_dim;
}
static uint64_t ab_get_crossover_flops(void) {
    if (_ab_cross_z) return _ab_cross_z;
    const char* s = getenv("AB_CROSSOVER_FLOPS");
    unsigned long long v = (s && *s) ? strtoull(s, NULL, 10) : DEFAULT_CROSSOVER_FLOPS;
    if (v == 0) v = DEFAULT_CROSSOVER_FLOPS;
    _ab_cross_z = (uint64_t)v;
    return _ab_cross_z;
}
static uint64_t ab_get_crossover_flops_real(void) {
    if (_ab_cross_d) return _ab_cross_d;
    const char* s = getenv("AB_CROSSOVER_FLOPS_REAL");
    unsigned long long v = (s && *s) ? strtoull(s, NULL, 10) : DEFAULT_CROSSOVER_FLOPS_REAL;
    if (v == 0) v = DEFAULT_CROSSOVER_FLOPS_REAL;
    _ab_cross_d = (uint64_t)v;
    return _ab_cross_d;
}

// AB_MODE runtime knob: cpu = force AMX/cblas, gpu = force GPU (above AB_MIN_GPU_DIM),
// auto (default) = current heterogeneous heuristic.
// Cached on first call; set AB_MODE env var before launching.
enum { AB_MODE_AUTO = 0, AB_MODE_CPU = 1, AB_MODE_GPU = 2 };
static int _ab_mode = -1;
static int ab_get_mode(void) {
    if (_ab_mode >= 0) return _ab_mode;
    const char* s = getenv("AB_MODE");
    if (s && (s[0] == 'c' || s[0] == 'C')) _ab_mode = AB_MODE_CPU;
    else if (s && (s[0] == 'g' || s[0] == 'G')) _ab_mode = AB_MODE_GPU;
    else _ab_mode = AB_MODE_AUTO;
    return _ab_mode;
}

// Thread-local record of the path the last BLAS-ABI call on this thread
// actually took, for bench harnesses and the profiler (prof_end). Only the
// two places that produce C write it: "cpu" in ab_internal_*gemm_cpu, "gpu"
// once the GPU result is in C (H-P1-2). The routing queries below never
// touch it. Zero runtime cost.
static __thread const char *_ab_last_path = "none";
const char* ab_get_last_dispatch_path(void) { return _ab_last_path; }

bool ab_use_gpu(int m, int n, int k) {
    int mode = ab_get_mode();
    if (mode == AB_MODE_CPU) return false;
    int min_dim = ab_get_min_gpu_dim();
    // Dims below GPU minimum are never profitable (threadgroups underutilized)
    if (m <= min_dim || n <= min_dim || k <= min_dim) return false;
    if (mode == AB_MODE_GPU) return true;
    // AUTO: heterogeneous heuristic
    uint64_t flops = 2ULL * m * n * k;
    return (m < 64 || n < 64 || k < 64)
           ? (flops >= ab_get_crossover_flops())
           : (flops >= ab_get_crossover_flops_real());
}

// Complex version accounts for 4x arithmetic density
static bool ab_use_gpu_complex(int m, int n, int k) {
    int mode = ab_get_mode();
    if (mode == AB_MODE_CPU) return false;
    int min_dim = ab_get_min_gpu_dim();
    if (m <= min_dim || n <= min_dim || k <= min_dim) return false;
    if (mode == AB_MODE_GPU) return true;
    uint64_t flops = 8ULL * m * n * k;
    return flops >= ab_get_crossover_flops();
}

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

// Reference BLAS argument checks (xerbla): TRANS flags, M, N, K >= 0 and
// each leading dimension >= max(1, rows of the stored operand). The GPU path
// indexes the caller's arrays with these, so a call that fails a check goes
// to cblas instead, which rejects it before touching A, B or C.
static bool blas_args_valid(enum CBLAS_TRANSPOSE ta, enum CBLAS_TRANSPOSE tb,
                            int M, int N, int K, int ldA, int ldB, int ldC) {
    int rowsA = ta == CblasNoTrans ? M : K, rowsB = tb == CblasNoTrans ? K : N;
    return ta != AB_TRANS_INVALID && tb != AB_TRANS_INVALID &&
           M >= 0 && N >= 0 && K >= 0 &&
           ldA >= 1 && ldA >= rowsA && ldB >= 1 && ldB >= rowsB &&
           ldC >= 1 && ldC >= M;
}

// Whole-call CPU paths (Accelerate cblas, column-major). Used below the GPU
// threshold, for BLAS quick returns and invalid arguments, for the calls the
// Fortran bridge routes to the CPU, and as the fallback when any GPU step
// fails before C is written, so C always gets a correct result.
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
// Call profile (AB_PROFILE, AB_REPORT)
// =============================================================================
// Every BLAS-ABI call is timed once, around routing and compute, and recorded
// by where C was computed (the path record). The GPU path charges each step
// to a stage as it finishes. p is NULL when profiling is off, so an
// unprofiled call pays one branch per stage.

typedef struct {
    uint64_t t0, mark;                         // ticks: call start, end of the last stage
    uint64_t stage_ns[AB_PROF_STAGE_COUNT];
    ab_prof_fallback_t fallback;               // why a GPU-routed call ran on cblas
} CallProf;

static CallProf *prof_begin(CallProf *p) {
    if (!ab_prof_enabled()) return NULL;
    memset(p, 0, sizeof *p);
    p->fallback = AB_PROF_FB_NONE;
    p->t0 = p->mark = ab_prof_now_ticks();
    (void)ab_internal_take_gpu_seconds();     // drop GPU time of earlier native-API calls
    return p;
}

static void prof_stage(CallProf *p, ab_prof_stage_t s) {
    if (!p) return;
    uint64_t now = ab_prof_now_ticks();
    p->stage_ns[s] += ab_prof_ticks_to_ns(now - p->mark);
    p->mark = now;
}

// The kernel stage is the command buffers' GPU execution time, or the wall
// time if Metal reports none. The rest of the kernel call (encoding, submit
// and wait latency, ZGEMM's GPU-side temporaries) is left to "other".
static void prof_kernel(CallProf *p) {
    if (!p) return;
    uint64_t now = ab_prof_now_ticks();
    uint64_t wall = ab_prof_ticks_to_ns(now - p->mark);
    uint64_t gpu = (uint64_t)(ab_internal_take_gpu_seconds() * 1e9);
    p->stage_ns[AB_PROF_STAGE_KERNEL] += gpu && gpu < wall ? gpu : wall;
    p->mark = now;
}

// Records the finished call; under AB_PROFILE also prints the stages of a GPU call.
static void prof_end(CallProf *p, const char *routine, int M, int N, int K) {
    if (!p) return;
    uint64_t ns = ab_prof_ticks_to_ns(ab_prof_now_ticks() - p->t0);
    bool gpu = !strcmp(_ab_last_path, "gpu");
    ab_prof_record(routine, ns, M > 0 ? (size_t)M : 0, N > 0 ? (size_t)N : 0,
                   K > 0 ? (size_t)K : 0, gpu, gpu ? p->stage_ns : NULL, p->fallback);
    if (!gpu || !ab_prof_verbose()) return;
    uint64_t named = 0;
    fprintf(stderr, "AB_PROFILE %s M=%d N=%d K=%d ", routine, M, N, K);
    for (int s = 0; s < AB_PROF_STAGE_COUNT; s++) {
        named += p->stage_ns[s];
        fprintf(stderr, " %s=%.3f", ab_prof_stage_names[s], p->stage_ns[s] / 1e6);
    }
    fprintf(stderr, " other=%.3f  total=%.3f ms\n", (ns > named ? ns - named : 0) / 1e6, ns / 1e6);
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

    // AMX path for small matrices (heterogeneous dispatch); also the BLAS
    // quick return for alpha == 0 (A and B must not be read) and invalid arguments.
    if (!blas_args_valid(ta, tb, M, N, K, ldA, ldB, ldC) || alpha == 0.0 ||
        !ab_use_gpu_complex(M, N, K)) {
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

    // AMX path for small matrices (heterogeneous dispatch); also the BLAS
    // quick return for alpha == 0 (A and B must not be read) and invalid arguments.
    if (!blas_args_valid(ta, tb, M, N, K, ldA, ldB, ldC) || alpha == 0.0 ||
        !ab_use_gpu(M, N, K)) {
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
// Profiled BLAS-ABI entry points
// =============================================================================
// ab_internal_*gemm run one call, timed and recorded for AB_PROFILE and
// AB_REPORT. gpu != 0: the caller's routing allows the GPU, and the
// ab_*gemm_blas path applies its own; gpu == 0: cblas. Internal, not in
// apple_bottom.h: the Fortran bridge (fortran_bridge.c) passes its routing
// decision, ab_*gemm_blas pass 1.

void ab_internal_dgemm(int gpu, char transA, char transB, int M, int N, int K,
                       double alpha, const double* A, int ldA,
                       const double* B, int ldB,
                       double beta, double* C, int ldC) {
    CallProf prof, *p = prof_begin(&prof);
    if (gpu)
        dgemm_blas(p, transA, transB, M, N, K, alpha, A, ldA, B, ldB, beta, C, ldC);
    else
        ab_internal_dgemm_cpu(ab_blas_trans(transA, false), ab_blas_trans(transB, false),
                              M, N, K, alpha, A, ldA, B, ldB, beta, C, ldC);
    prof_end(p, "dgemm", M, N, K);
}

void ab_internal_zgemm(int gpu, char transA, char transB, int M, int N, int K,
                       double complex alpha, const double complex* A, int ldA,
                       const double complex* B, int ldB,
                       double complex beta, double complex* C, int ldC) {
    CallProf prof, *p = prof_begin(&prof);
    if (gpu)
        zgemm_blas(p, transA, transB, M, N, K, alpha, A, ldA, B, ldB, beta, C, ldC);
    else
        ab_internal_zgemm_cpu(ab_blas_trans(transA, true), ab_blas_trans(transB, true),
                              M, N, K, alpha, A, ldA, B, ldB, beta, C, ldC);
    prof_end(p, "zgemm", M, N, K);
}

void ab_dgemm_blas(char transA, char transB, int M, int N, int K,
                   double alpha, const double* A, int ldA,
                   const double* B, int ldB,
                   double beta, double* C, int ldC) {
    ab_internal_dgemm(1, transA, transB, M, N, K, alpha, A, ldA, B, ldB, beta, C, ldC);
}

void ab_zgemm_blas(char transA, char transB, int M, int N, int K,
                   double complex alpha, const double complex* A, int ldA,
                   const double complex* B, int ldB,
                   double complex beta, double complex* C, int ldC) {
    ab_internal_zgemm(1, transA, transB, M, N, K, alpha, A, ldA, B, ldB, beta, C, ldC);
}

// Fortran-callable wrappers (ab_zgemm_, ab_dgemm_) are in fortran_bridge.c
