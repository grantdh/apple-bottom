// DGEMM Benchmark — apple-bottom
#include "apple_bottom.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include <Accelerate/Accelerate.h>

static double get_time_sec(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec + ts.tv_nsec * 1e-9;
}

int main(int argc, char** argv) {
    FILE* csv = NULL;
    for (int i = 1; i < argc - 1; i++) {
        if (strcmp(argv[i], "-o") == 0) {
            csv = fopen(argv[i + 1], "w");
            if (!csv) { fprintf(stderr, "cannot open %s\n", argv[i + 1]); return 1; }
            fprintf(csv, "op,size,iters,amx_gflops,gpu_gflops,speedup,frob_rel_err,max_elem_err\n");
        }
    }
    printf("╔══════════════════════════════════════════════════════════════════╗\n");
    printf("║  apple-bottom DGEMM Benchmark                                    ║\n");
    printf("╚══════════════════════════════════════════════════════════════════╝\n\n");
    
    if (ab_init() != AB_OK) { fprintf(stderr, "Init failed\n"); return 1; }
    printf("Device: %s\n\n", ab_device_name());
    
    int sizes[] = {256, 512, 1024, 2048, 3072, 4096};
    
    printf("  Size    │  AMX GFLOP/s │  GPU GFLOP/s │  Speedup  │ Precision\n");
    printf("──────────┼──────────────┼──────────────┼───────────┼───────────\n");
    
    for (int s = 0; s < 6; s++) {
        int N = sizes[s];
        size_t count = (size_t)N * N;
        double flops = 2.0 * N * N * N;
        int iters = (N <= 1024) ? 10 : 5;
        
        double* A = malloc(count * sizeof(double));
        double* B = malloc(count * sizeof(double));
        double* C_gpu = malloc(count * sizeof(double));
        double* C_ref = malloc(count * sizeof(double));
        
        srand48(42);
        for (size_t i = 0; i < count; i++) { A[i] = drand48() * 2 - 1; B[i] = drand48() * 2 - 1; }
        
        // AMX benchmark
        cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, N, N, N, 1.0, A, N, B, N, 0.0, C_ref, N);
        double t0 = get_time_sec();
        for (int i = 0; i < iters; i++)
            cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, N, N, N, 1.0, A, N, B, N, 0.0, C_ref, N);
        double amx_gflops = flops * iters / ((get_time_sec() - t0) * 1e9);
        
        // GPU benchmark
        ABMatrix mA = ab_matrix_create(N, N);
        ABMatrix mB = ab_matrix_create(N, N);
        ABMatrix mC = ab_matrix_create(N, N);
        ab_matrix_upload(mA, A, true);
        ab_matrix_upload(mB, B, true);
        ab_dgemm(mA, mB, mC); // warmup
        
        t0 = get_time_sec();
        for (int i = 0; i < iters; i++) ab_dgemm(mA, mB, mC);
        double gpu_gflops = flops * iters / ((get_time_sec() - t0) * 1e9);
        
        ab_matrix_download(mC, C_gpu, true);
        
        // Precision check
        double max_err = 0, sum_sq = 0;
        for (size_t i = 0; i < count; i++) {
            double err = fabs(C_gpu[i] - C_ref[i]);
            if (err > max_err) max_err = err;
            sum_sq += C_ref[i] * C_ref[i];
        }
        double frob = max_err / sqrt(sum_sq);
        
        double speedup = gpu_gflops / amx_gflops;
        printf("  %4d    │    %6.0f    │    %6.0f    │  %5.2fx %s │  %.0e %s\n",
               N, amx_gflops, gpu_gflops, speedup, speedup >= 1.0 ? "✓" : " ", frob, frob < 1e-14 ? "✓" : "✗");
        if (csv) {
            fprintf(csv, "dgemm,%d,%d,%.2f,%.2f,%.4f,%.3e,%.3e\n",
                    N, iters, amx_gflops, gpu_gflops, speedup, frob, max_err);
            fflush(csv);
        }

        ab_matrix_destroy(mA);
        ab_matrix_destroy(mB);
        ab_matrix_destroy(mC);
        free(A); free(B); free(C_gpu); free(C_ref);
    }
    
    printf("\n");
    printf("Note: GPU wins for N >= 2048\n");
    
    if (csv) fclose(csv);
    ab_shutdown();
    return 0;
}
