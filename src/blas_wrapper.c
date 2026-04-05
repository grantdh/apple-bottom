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

#define CROSSOVER_FLOPS 100000000ULL

bool ab_use_gpu(int m, int n, int k) {
    uint64_t flops = 8ULL * m * n * k;
    return flops >= CROSSOVER_FLOPS;
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
    uint64_t flops = 8ULL * M * N * K;
    
    // CPU fallback for small matrices only
    if (flops < CROSSOVER_FLOPS) {
        cblas_zgemm(CblasColMajor,
                    transA == 'N' ? CblasNoTrans :
                    transA == 'T' ? CblasTrans : CblasConjTrans,
                    transB == 'N' ? CblasNoTrans :
                    transB == 'T' ? CblasTrans : CblasConjTrans,
                    M, N, K, &alpha, A, ldA, B, ldB, &beta, C, ldC);
        return;
    }

    // GPU path: compute A*B on GPU, apply alpha/beta in epilogue
    
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
    ab_matrix_upload(mAr, Ar_data, true);
    ab_matrix_upload(mAi, Ai_data, true);
    free(Ar_data); free(Ai_data);
    
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
    ab_matrix_upload(mBr, Br_data, true);
    ab_matrix_upload(mBi, Bi_data, true);
    free(Br_data); free(Bi_data);
    
    // Compute C = A * B
    ab_zgemm(mAr, mAi, mBr, mBi, mCr, mCi);
    
    // Download result and apply alpha/beta epilogue
    double* Cr_data = malloc(M * N * sizeof(double));
    double* Ci_data = malloc(M * N * sizeof(double));
    ab_matrix_download(mCr, Cr_data, true);
    ab_matrix_download(mCi, Ci_data, true);

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
    ab_matrix_destroy(mAr); ab_matrix_destroy(mAi);
    ab_matrix_destroy(mBr); ab_matrix_destroy(mBi);
    ab_matrix_destroy(mCr); ab_matrix_destroy(mCi);
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
    uint64_t flops = 2ULL * M * N * K;

    // CPU fallback for small matrices only
    if (flops < CROSSOVER_FLOPS) {
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

