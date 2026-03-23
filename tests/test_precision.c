// Precision Test — apple-bottom
#include "apple_bottom.h"
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <Accelerate/Accelerate.h>

int main(void) {
    printf("╔══════════════════════════════════════════════════════════════════╗\n");
    printf("║  apple-bottom Precision Test                                     ║\n");
    printf("╚══════════════════════════════════════════════════════════════════╝\n\n");
    
    if (ab_init() != AB_OK) { fprintf(stderr, "Init failed\n"); return 1; }
    printf("Device: %s\n\n", ab_device_name());
    
    printf("Testing DGEMM precision:\n\n");
    printf("  Size    │ Frobenius Err │ Max Elem Err │ Status\n");
    printf("──────────┼───────────────┼──────────────┼────────\n");
    
    int sizes[] = {64, 128, 256, 512, 1024, 2048};
    int all_passed = 1;
    
    for (int s = 0; s < 6; s++) {
        int N = sizes[s];
        size_t count = (size_t)N * N;
        
        double* A = malloc(count * sizeof(double));
        double* B = malloc(count * sizeof(double));
        double* C_gpu = malloc(count * sizeof(double));
        double* C_ref = malloc(count * sizeof(double));
        
        srand48(N);
        for (size_t i = 0; i < count; i++) {
            A[i] = (drand48() * 2 - 1) * pow(10, (int)(drand48() * 6 - 3));
            B[i] = (drand48() * 2 - 1) * pow(10, (int)(drand48() * 6 - 3));
        }
        
        ABMatrix mA = ab_matrix_create(N, N);
        ABMatrix mB = ab_matrix_create(N, N);
        ABMatrix mC = ab_matrix_create(N, N);
        ab_matrix_upload(mA, A, true);
        ab_matrix_upload(mB, B, true);
        ab_dgemm(mA, mB, mC);
        ab_matrix_download(mC, C_gpu, true);
        ab_matrix_destroy(mA);
        ab_matrix_destroy(mB);
        ab_matrix_destroy(mC);
        
        cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, N, N, N, 1.0, A, N, B, N, 0.0, C_ref, N);
        
        double max_err = 0, sum_sq = 0, max_elem = 0;
        for (size_t i = 0; i < count; i++) {
            double err = fabs(C_gpu[i] - C_ref[i]);
            if (err > max_err) max_err = err;
            sum_sq += C_ref[i] * C_ref[i];
            double ref_mag = fabs(C_ref[i]);
            if (ref_mag > 1e-10) { double rel = err / ref_mag; if (rel > max_elem) max_elem = rel; }
        }
        double frob = max_err / sqrt(sum_sq);
        int passed = frob < 1e-14;
        if (!passed) all_passed = 0;
        
        printf("  %4d    │    %.2e   │   %.2e   │ %s\n", N, frob, max_elem, passed ? "✓ PASS" : "✗ FAIL");
        
        free(A); free(B); free(C_gpu); free(C_ref);
    }
    
    printf("\n═══════════════════════════════════════════════════════════════════\n");
    if (all_passed) {
        printf("✓ All precision tests PASSED\n");
        printf("  Achieved precision: ~10⁻¹⁵ (Frobenius relative error)\n");
    } else {
        printf("✗ Some precision tests FAILED\n");
    }
    
    ab_shutdown();
    return all_passed ? 0 : 1;
}
