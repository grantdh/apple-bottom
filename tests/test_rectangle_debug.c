// Quick debug test for rectangular matrices
#include "apple_bottom.h"
#include <stdio.h>
#include <stdlib.h>
#include <Accelerate/Accelerate.h>

int main(void) {
    ab_init();

    // Small rectangular test
    int M = 100, N = 50, K = 100;

    double* A = (double*)malloc(M * K * sizeof(double));
    double* B = (double*)malloc(K * N * sizeof(double));
    double* C_gpu = (double*)calloc(M * N, sizeof(double));
    double* C_ref = (double*)calloc(M * N, sizeof(double));

    // Simple pattern
    for (int i = 0; i < M; i++) {
        for (int j = 0; j < K; j++) {
            A[i*K + j] = (double)(i + j);
        }
    }
    for (int i = 0; i < K; i++) {
        for (int j = 0; j < N; j++) {
            B[i*N + j] = (double)(i * j + 1);
        }
    }

    // Reference
    cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                M, N, K, 1.0, A, K, B, N, 0.0, C_ref, N);

    // GPU
    ABMatrix mA = ab_matrix_create(M, K);
    ABMatrix mB = ab_matrix_create(K, N);
    ABMatrix mC = ab_matrix_create(M, N);
    ab_matrix_upload(mA, A, false);
    ab_matrix_upload(mB, B, false);
    ab_dgemm(mA, mB, mC);
    ab_matrix_download(mC, C_gpu, false);

    // Compare first few elements
    printf("M=%d, N=%d, K=%d\n", M, N, K);
    printf("\nFirst 5 results:\n");
    printf("Index  Reference         GPU               Error\n");
    for (int i = 0; i < 5; i++) {
        double ref = C_ref[i];
        double gpu = C_gpu[i];
        double err = (ref != 0) ? fabs((gpu - ref) / ref) : fabs(gpu - ref);
        printf("%3d    %16.8f  %16.8f  %.2e\n", i, ref, gpu, err);
    }

    // Find max error
    double max_err = 0.0;
    int max_idx = 0;
    for (int i = 0; i < M * N; i++) {
        double ref = C_ref[i];
        double gpu = C_gpu[i];
        if (fabs(ref) < 1e-300) continue;
        double err = fabs((gpu - ref) / ref);
        if (err > max_err) {
            max_err = err;
            max_idx = i;
        }
    }

    printf("\nMax error: %.2e at index %d\n", max_err, max_idx);
    printf("  Reference: %.8f\n", C_ref[max_idx]);
    printf("  GPU:       %.8f\n", C_gpu[max_idx]);

    free(A); free(B); free(C_gpu); free(C_ref);
    ab_matrix_destroy(mA);
    ab_matrix_destroy(mB);
    ab_matrix_destroy(mC);
    ab_shutdown();

    return 0;
}
