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
#include <time.h>

// Per-stage profiling for ab_{d,z}gemm_blas, gated on AB_PROFILE=1.
// Zero cost when disabled (single cached env read, single branch).
// Prints stage breakdown to stderr at the end of each call.
static int _ab_profile = -1;
static int ab_profile_enabled(void) {
    if (_ab_profile >= 0) return _ab_profile;
    const char *s = getenv("AB_PROFILE");
    _ab_profile = (s && *s && s[0] != '0') ? 1 : 0;
    return _ab_profile;
}
static double ab_now_sec(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec + ts.tv_nsec * 1e-9;
}

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

// Thread-local record of the last dispatch decision. Used by bench harnesses
// to confirm AB_MODE routing without touching internals. Zero runtime cost.
static __thread const char *_ab_last_path = "none";
const char* ab_get_last_dispatch_path(void) { return _ab_last_path; }

bool ab_use_gpu(int m, int n, int k) {
    int mode = ab_get_mode();
    if (mode == AB_MODE_CPU) { _ab_last_path = "cpu"; return false; }
    int min_dim = ab_get_min_gpu_dim();
    // Dims below GPU minimum are never profitable (threadgroups underutilized)
    if (m <= min_dim || n <= min_dim || k <= min_dim) {
        _ab_last_path = "cpu"; return false;
    }
    if (mode == AB_MODE_GPU) { _ab_last_path = "gpu"; return true; }
    // AUTO: heterogeneous heuristic
    uint64_t flops = 2ULL * m * n * k;
    bool use = (m < 64 || n < 64 || k < 64)
               ? (flops >= ab_get_crossover_flops())
               : (flops >= ab_get_crossover_flops_real());
    _ab_last_path = use ? "gpu" : "cpu";
    return use;
}

// Complex version accounts for 4x arithmetic density
static bool ab_use_gpu_complex(int m, int n, int k) {
    int mode = ab_get_mode();
    if (mode == AB_MODE_CPU) { _ab_last_path = "cpu"; return false; }
    int min_dim = ab_get_min_gpu_dim();
    if (m <= min_dim || n <= min_dim || k <= min_dim) {
        _ab_last_path = "cpu"; return false;
    }
    if (mode == AB_MODE_GPU) { _ab_last_path = "gpu"; return true; }
    uint64_t flops = 8ULL * m * n * k;
    bool use = flops >= ab_get_crossover_flops();
    _ab_last_path = use ? "gpu" : "cpu";
    return use;
}

// =============================================================================
// BLAS-compatible ZGEMM - this is what Fortran calls
// Handles: transA, transB, alpha, beta, leading dimensions
// =============================================================================
void ab_zgemm_blas(
    char transA, char transB,
    int M, int N, int K,
    double complex alpha,
    const double complex* A, int ldA,
    const double complex* B, int ldB,
    double complex beta,
    double complex* C, int ldC
) {
    // AMX path for small matrices (heterogeneous dispatch)
    if (!ab_use_gpu_complex(M, N, K)) {
        cblas_zgemm(CblasColMajor,
                    transA == 'N' ? CblasNoTrans :
                    transA == 'T' ? CblasTrans : CblasConjTrans,
                    transB == 'N' ? CblasNoTrans :
                    transB == 'T' ? CblasTrans : CblasConjTrans,
                    M, N, K, &alpha, A, ldA, B, ldB, &beta, C, ldC);
        return;
    }

    // GPU path: compute A*B on GPU, apply alpha/beta in epilogue
    int _prof = ab_profile_enabled();
    double _t0 = _prof ? ab_now_sec() : 0;
    double _t_alloc=0, _t_A_repack=0, _t_A_upload=0,
           _t_B_repack=0, _t_B_upload=0, _t_C_upload=0,
           _t_kernel=0, _t_download=0, _t_epilogue=0, _t_destroy=0;

    // Dimensions of A and B after transpose
    int A_rows = (transA == 'N') ? M : K;
    int A_cols = (transA == 'N') ? K : M;
    int B_rows = (transB == 'N') ? K : N;
    int B_cols = (transB == 'N') ? N : K;

    // Allocate split-complex GPU matrices
    ABMatrix mAr = ab_matrix_create(A_rows, A_cols);
    ABMatrix mAi = ab_matrix_create(A_rows, A_cols);
    ABMatrix mBr = ab_matrix_create(B_rows, B_cols);
    ABMatrix mBi = ab_matrix_create(B_rows, B_cols);
    ABMatrix mCr = ab_matrix_create(M, N);
    ABMatrix mCi = ab_matrix_create(M, N);
    if (_prof) { double t = ab_now_sec(); _t_alloc = t - _t0; _t0 = t; }
    
    if (!mAr || !mAi || !mBr || !mBi || !mCr || !mCi) {
        if (mAr) ab_matrix_destroy(mAr);
        if (mAi) ab_matrix_destroy(mAi);
        if (mBr) ab_matrix_destroy(mBr);
        if (mBi) ab_matrix_destroy(mBi);
        if (mCr) ab_matrix_destroy(mCr);
        if (mCi) ab_matrix_destroy(mCi);
        cblas_zgemm(CblasColMajor,
                    transA == 'N' ? CblasNoTrans : 
                    transA == 'T' ? CblasTrans : CblasConjTrans,
                    transB == 'N' ? CblasNoTrans :
                    transB == 'T' ? CblasTrans : CblasConjTrans,
                    M, N, K, &alpha, A, ldA, B, ldB, &beta, C, ldC);
        return;
    }
    
    // Upload A with transpose handling (column-major to row-major)
    size_t A_count = (size_t)A_rows * A_cols;
    double* Ar_data = malloc(A_count * sizeof(double));
    double* Ai_data = malloc(A_count * sizeof(double));

    for (int col = 0; col < A_cols; col++) {
        for (int row = 0; row < A_rows; row++) {
            int src_row, src_col;
            if (transA == 'N') {
                src_row = row; src_col = col;
            } else {
                src_row = col; src_col = row;
            }
            double complex val = A[src_col * ldA + src_row];
            if (transA == 'C') val = conj(val);

            size_t dst = (size_t)row * A_cols + col;
            Ar_data[dst] = creal(val);
            Ai_data[dst] = cimag(val);
        }
    }
    if (_prof) { double t = ab_now_sec(); _t_A_repack = t - _t0; _t0 = t; }
    ab_matrix_upload(mAr, Ar_data, true);
    ab_matrix_upload(mAi, Ai_data, true);
    free(Ar_data); free(Ai_data);
    if (_prof) { double t = ab_now_sec(); _t_A_upload = t - _t0; _t0 = t; }

    // Upload B with transpose handling
    size_t B_count = (size_t)B_rows * B_cols;
    double* Br_data = malloc(B_count * sizeof(double));
    double* Bi_data = malloc(B_count * sizeof(double));

    for (int col = 0; col < B_cols; col++) {
        for (int row = 0; row < B_rows; row++) {
            int src_row, src_col;
            if (transB == 'N') {
                src_row = row; src_col = col;
            } else {
                src_row = col; src_col = row;
            }
            double complex val = B[src_col * ldB + src_row];
            if (transB == 'C') val = conj(val);

            size_t dst = (size_t)row * B_cols + col;
            Br_data[dst] = creal(val);
            Bi_data[dst] = cimag(val);
        }
    }
    if (_prof) { double t = ab_now_sec(); _t_B_repack = t - _t0; _t0 = t; }
    ab_matrix_upload(mBr, Br_data, true);
    ab_matrix_upload(mBi, Bi_data, true);
    free(Br_data); free(Bi_data);
    if (_prof) { double t = ab_now_sec(); _t_B_upload = t - _t0; _t0 = t; }
    (void)_t_C_upload;  // beta != 0 path not instrumented in this pass

    // Compute C = A * B
    ab_zgemm(mAr, mAi, mBr, mBi, mCr, mCi);
    if (_prof) { double t = ab_now_sec(); _t_kernel = t - _t0; _t0 = t; }

    // Download result and apply alpha/beta epilogue
    double* Cr_data = malloc(M * N * sizeof(double));
    double* Ci_data = malloc(M * N * sizeof(double));
    ab_matrix_download(mCr, Cr_data, true);
    ab_matrix_download(mCi, Ci_data, true);
    if (_prof) { double t = ab_now_sec(); _t_download = t - _t0; _t0 = t; }

    // Convert row-major back to column-major with alpha/beta:
    // C = alpha * (A*B) + beta * C_old
    double ar = creal(alpha), ai = cimag(alpha);
    double br = creal(beta), bi = cimag(beta);
    bool needs_beta = (br != 0.0 || bi != 0.0);
    bool non_unit_alpha = (ar != 1.0 || ai != 0.0);

    for (int col = 0; col < N; col++) {
        for (int row = 0; row < M; row++) {
            size_t src = (size_t)row * N + col;
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
                double complex c_old = C[col * ldC + row];
                result_r += br * creal(c_old) - bi * cimag(c_old);
                result_i += br * cimag(c_old) + bi * creal(c_old);
            }

            C[col * ldC + row] = result_r + I * result_i;
        }
    }
    free(Cr_data); free(Ci_data);
    if (_prof) { double t = ab_now_sec(); _t_epilogue = t - _t0; _t0 = t; }

    ab_matrix_destroy(mAr); ab_matrix_destroy(mAi);
    ab_matrix_destroy(mBr); ab_matrix_destroy(mBi);
    ab_matrix_destroy(mCr); ab_matrix_destroy(mCi);
    if (_prof) {
        double t = ab_now_sec(); _t_destroy = t - _t0;
        double sum = _t_alloc + _t_A_repack + _t_A_upload +
                     _t_B_repack + _t_B_upload + _t_kernel +
                     _t_download + _t_epilogue + _t_destroy;
        fprintf(stderr,
            "AB_PROFILE zgemm M=%d N=%d K=%d  "
            "alloc=%.3f A_repack=%.3f A_upload=%.3f B_repack=%.3f "
            "B_upload=%.3f kernel=%.3f download=%.3f epilogue=%.3f "
            "destroy=%.3f  sum=%.3f ms\n",
            M, N, K,
            _t_alloc*1e3, _t_A_repack*1e3, _t_A_upload*1e3,
            _t_B_repack*1e3, _t_B_upload*1e3, _t_kernel*1e3,
            _t_download*1e3, _t_epilogue*1e3, _t_destroy*1e3,
            sum*1e3);
    }
}

// =============================================================================
// BLAS-compatible DGEMM: C = alpha * op(A) * op(B) + beta * C
// Handles transA, transB, alpha, beta, leading dimensions.
// GPU path supports all transpose combinations via upload-time reordering.
// =============================================================================
void ab_dgemm_blas(
    char transA, char transB,
    int M, int N, int K,
    double alpha,
    const double* A, int ldA,
    const double* B, int ldB,
    double beta,
    double* C, int ldC
) {
    // AMX path for small matrices (heterogeneous dispatch)
    if (!ab_use_gpu(M, N, K)) {
        cblas_dgemm(CblasColMajor,
                    transA == 'T' ? CblasTrans : CblasNoTrans,
                    transB == 'T' ? CblasTrans : CblasNoTrans,
                    M, N, K, alpha, A, ldA, B, ldB, beta, C, ldC);
        return;
    }

    // GPU path: supports arbitrary alpha/beta and transpose via scaled kernel
    ABMatrix mA = ab_matrix_create(M, K);
    ABMatrix mB = ab_matrix_create(K, N);
    ABMatrix mC = ab_matrix_create(M, N);

    if (!mA || !mB || !mC) {
        if (mA) ab_matrix_destroy(mA);
        if (mB) ab_matrix_destroy(mB);
        if (mC) ab_matrix_destroy(mC);
        cblas_dgemm(CblasColMajor,
                    transA == 'T' ? CblasTrans : CblasNoTrans,
                    transB == 'T' ? CblasTrans : CblasNoTrans,
                    M, N, K, alpha, A, ldA, B, ldB, beta, C, ldC);
        return;
    }

    // Upload op(A) as M×K row-major (handles column-major + transpose)
    // BLAS convention: op(A) is M×K
    //   transA='N': A is col-major M×K, A[col*ldA+row] → row-major[row*K+col]
    //   transA='T': A is col-major K×M, op(A)[i,j] = A[j,i] = A[i*ldA+j]
    double* A_row = malloc((size_t)M * K * sizeof(double));
    if (transA == 'N') {
        for (int j = 0; j < K; j++)
            for (int i = 0; i < M; i++)
                A_row[i * K + j] = A[j * ldA + i];
    } else {
        // transA='T': A stored as col-major K×M
        // op(A)[i,j] = A^T[i,j] = A_colmaj[j,i] = A[i*ldA+j]
        for (int i = 0; i < M; i++)
            for (int j = 0; j < K; j++)
                A_row[i * K + j] = A[i * ldA + j];
    }
    ab_matrix_upload(mA, A_row, true);
    free(A_row);

    // Upload op(B) as K×N row-major
    //   transB='N': B is col-major K×N → row-major[row*N+col]
    //   transB='T': B is col-major N×K, op(B)[i,j] = B^T[i,j] = B[i*ldB+j]
    double* B_row = malloc((size_t)K * N * sizeof(double));
    if (transB == 'N') {
        for (int j = 0; j < N; j++)
            for (int i = 0; i < K; i++)
                B_row[i * N + j] = B[j * ldB + i];
    } else {
        // transB='T': B stored as col-major N×K
        for (int i = 0; i < K; i++)
            for (int j = 0; j < N; j++)
                B_row[i * N + j] = B[i * ldB + j];
    }
    ab_matrix_upload(mB, B_row, true);
    free(B_row);

    // Upload existing C when beta != 0 (needed for accumulation C = alpha*A*B + beta*C)
    double* C_row = malloc((size_t)M * N * sizeof(double));
    if (beta != 0.0) {
        for (int j = 0; j < N; j++)
            for (int i = 0; i < M; i++)
                C_row[i * N + j] = C[j * ldC + i];
        ab_matrix_upload(mC, C_row, true);
    }

    // Compute C = alpha*A*B + beta*C on GPU
    ab_dgemm_scaled(alpha, mA, mB, beta, mC);

    // Download result (row-major to column-major)
    ab_matrix_download(mC, C_row, true);
    for (int j = 0; j < N; j++)
        for (int i = 0; i < M; i++)
            C[j * ldC + i] = C_row[i * N + j];
    free(C_row);

    ab_matrix_destroy(mA);
    ab_matrix_destroy(mB);
    ab_matrix_destroy(mC);
}

// Fortran-callable wrappers (ab_zgemm_, ab_dgemm_) are in fortran_bridge.c

