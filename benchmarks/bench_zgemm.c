// ZGEMM Benchmark — apple-bottom
#include "apple_bottom.h"
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <complex.h>
#include <time.h>
#include <Accelerate/Accelerate.h>

static double get_time_sec(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec + ts.tv_nsec * 1e-9;
}

int main(void) {
    printf("╔══════════════════════════════════════════════════════════════════╗\n");
    printf("║  apple-bottom ZGEMM Benchmark                                    ║\n");
    printf("╚══════════════════════════════════════════════════════════════════╝\n\n");
    
    if (ab_init() != AB_OK) { fprintf(stderr, "Init failed\n"); return 1; }
    printf("Device: %s\n\n", ab_device_name());
    
    int sizes[] = {256, 512, 1024, 2048, 3072};
    
    printf("  Size    │  AMX GFLOP/s │  GPU GFLOP/s │  Speedup  │ Status\n");
    printf("──────────┼──────────────┼──────────────┼───────────┼────────\n");
    
    for (int s = 0; s < 5; s++) {
        int N = sizes[s];
        size_t count = (size_t)N * N;
        double flops = 8.0 * N * N * N;
        int iters = (N <= 1024) ? 5 : 3;
        
        double* Ar = malloc(count * sizeof(double));
        double* Ai = malloc(count * sizeof(double));
        double* Br = malloc(count * sizeof(double));
        double* Bi = malloc(count * sizeof(double));
        double* Cr_gpu = malloc(count * sizeof(double));
        double* Ci_gpu = malloc(count * sizeof(double));
        
        double complex* A_accel = malloc(count * sizeof(double complex));
        double complex* B_accel = malloc(count * sizeof(double complex));
        double complex* C_accel = malloc(count * sizeof(double complex));
        
        srand48(42);
        for (size_t i = 0; i < count; i++) {
            Ar[i] = drand48() * 2 - 1;
            Ai[i] = drand48() * 2 - 1;
            Br[i] = drand48() * 2 - 1;
            Bi[i] = drand48() * 2 - 1;
            A_accel[i] = Ar[i] + I * Ai[i];
            B_accel[i] = Br[i] + I * Bi[i];
        }
        
        // AMX benchmark
        double complex alpha = 1.0, beta = 0.0;
        cblas_zgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, N, N, N,
                    &alpha, A_accel, N, B_accel, N, &beta, C_accel, N);
        double t0 = get_time_sec();
        for (int i = 0; i < iters; i++)
            cblas_zgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, N, N, N,
                        &alpha, A_accel, N, B_accel, N, &beta, C_accel, N);
        double amx_gflops = flops * iters / ((get_time_sec() - t0) * 1e9);
        
        // GPU benchmark
        ABMatrix mAr = ab_matrix_create(N, N);
        ABMatrix mAi = ab_matrix_create(N, N);
        ABMatrix mBr = ab_matrix_create(N, N);
        ABMatrix mBi = ab_matrix_create(N, N);
        ABMatrix mCr = ab_matrix_create(N, N);
        ABMatrix mCi = ab_matrix_create(N, N);
        
        ab_matrix_upload(mAr, Ar, true);
        ab_matrix_upload(mAi, Ai, true);
        ab_matrix_upload(mBr, Br, true);
        ab_matrix_upload(mBi, Bi, true);
        
        ab_zgemm(mAr, mAi, mBr, mBi, mCr, mCi); // warmup
        
        t0 = get_time_sec();
        for (int i = 0; i < iters; i++)
            ab_zgemm(mAr, mAi, mBr, mBi, mCr, mCi);
        double gpu_gflops = flops * iters / ((get_time_sec() - t0) * 1e9);
        
        ab_matrix_download(mCr, Cr_gpu, true);
        ab_matrix_download(mCi, Ci_gpu, true);
        
        // Precision check
        double max_err = 0;
        for (size_t i = 0; i < count; i++) {
            double err_r = fabs(Cr_gpu[i] - creal(C_accel[i]));
            double err_i = fabs(Ci_gpu[i] - cimag(C_accel[i]));
            if (err_r > max_err) max_err = err_r;
            if (err_i > max_err) max_err = err_i;
        }
        
        double speedup = gpu_gflops / amx_gflops;
        printf("  %4d    │    %6.0f    │    %6.0f    │  %5.2fx %s │  %s\n",
               N, amx_gflops, gpu_gflops, speedup, speedup >= 1.0 ? "✓" : " ",
               max_err < 1e-10 ? "✓ PASS" : "✗ FAIL");
        
        ab_matrix_destroy(mAr); ab_matrix_destroy(mAi);
        ab_matrix_destroy(mBr); ab_matrix_destroy(mBi);
        ab_matrix_destroy(mCr); ab_matrix_destroy(mCi);
        free(Ar); free(Ai); free(Br); free(Bi);
        free(Cr_gpu); free(Ci_gpu);
        free(A_accel); free(B_accel); free(C_accel);
    }
    
    printf("\n");
    printf("Note: GPU wins for N >= 1024 (crossover point)\n");
    
    ab_shutdown();
    return 0;
}
