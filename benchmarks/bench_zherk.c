// ZHERK Benchmark — apple-bottom
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
    printf("║  apple-bottom ZHERK Benchmark                                    ║\n");
    printf("╚══════════════════════════════════════════════════════════════════╝\n\n");
    
    if (ab_init() != AB_OK) { fprintf(stderr, "Init failed\n"); return 1; }
    printf("Device: %s\n\n", ab_device_name());
    
    int sizes[] = {256, 512, 1024, 2048};
    int K = 256;
    
    printf("  N       │  AMX GFLOP/s │  GPU GFLOP/s │  Speedup  │ Status\n");
    printf("──────────┼──────────────┼──────────────┼───────────┼────────\n");
    
    for (int s = 0; s < 4; s++) {
        int N = sizes[s];
        size_t countA = (size_t)N * K;
        size_t countC = (size_t)N * N;
        double flops = 4.0 * N * N * K;  // Complex symmetric
        int iters = (N <= 1024) ? 5 : 3;
        
        double* Ar = malloc(countA * sizeof(double));
        double* Ai = malloc(countA * sizeof(double));
        double* Cr_gpu = malloc(countC * sizeof(double));
        double* Ci_gpu = malloc(countC * sizeof(double));
        
        double complex* A_ref = malloc(countA * sizeof(double complex));
        double complex* C_ref = malloc(countC * sizeof(double complex));
        
        srand48(42);
        for (size_t i = 0; i < countA; i++) {
            Ar[i] = drand48() * 2 - 1;
            Ai[i] = drand48() * 2 - 1;
            A_ref[i] = Ar[i] + I * Ai[i];
        }
        memset(C_ref, 0, countC * sizeof(double complex));
        
        // AMX benchmark
        cblas_zherk(CblasRowMajor, CblasUpper, CblasNoTrans, N, K, 1.0, A_ref, K, 0.0, C_ref, N);
        double t0 = get_time_sec();
        for (int i = 0; i < iters; i++)
            cblas_zherk(CblasRowMajor, CblasUpper, CblasNoTrans, N, K, 1.0, A_ref, K, 0.0, C_ref, N);
        double amx_gflops = flops * iters / ((get_time_sec() - t0) * 1e9);
        
        // GPU benchmark
        ABMatrix mAr = ab_matrix_create(N, K);
        ABMatrix mAi = ab_matrix_create(N, K);
        ABMatrix mCr = ab_matrix_create(N, N);
        ABMatrix mCi = ab_matrix_create(N, N);
        ab_matrix_upload(mAr, Ar, true);
        ab_matrix_upload(mAi, Ai, true);
        ab_zherk(mAr, mAi, mCr, mCi); // warmup
        
        t0 = get_time_sec();
        for (int i = 0; i < iters; i++)
            ab_zherk(mAr, mAi, mCr, mCi);
        double gpu_gflops = flops * iters / ((get_time_sec() - t0) * 1e9);
        
        ab_matrix_download(mCr, Cr_gpu, true);
        ab_matrix_download(mCi, Ci_gpu, true);
        
        // Precision check (upper triangle)
        double max_err = 0;
        for (int i = 0; i < N; i++) {
            for (int j = i; j < N; j++) {
                double err_r = fabs(Cr_gpu[i * N + j] - creal(C_ref[i * N + j]));
                double err_i = fabs(Ci_gpu[i * N + j] - cimag(C_ref[i * N + j]));
                if (err_r > max_err) max_err = err_r;
                if (err_i > max_err) max_err = err_i;
            }
        }
        
        double speedup = gpu_gflops / amx_gflops;
        printf("  %4d    │    %6.0f    │    %6.0f    │  %5.2fx %s │  %s\n",
               N, amx_gflops, gpu_gflops, speedup, speedup >= 1.0 ? "✓" : " ",
               max_err < 1e-10 ? "✓ PASS" : "✗ FAIL");
        
        ab_matrix_destroy(mAr); ab_matrix_destroy(mAi);
        ab_matrix_destroy(mCr); ab_matrix_destroy(mCi);
        free(Ar); free(Ai); free(Cr_gpu); free(Ci_gpu);
        free(A_ref); free(C_ref);
    }
    
    printf("\nNote: K fixed at %d\n", K);

    // ========================================================================
    // Part 2: Tall-skinny ZGEMM — QE-realistic dimensions
    // QE profile shows M≈18K, N=150, K=150-300 for 931 ZGEMM calls/SCF run.
    // This exercises the encode_dgemm_dispatch tall-skinny routing path.
    // ========================================================================
    printf("\n╔══════════════════════════════════════════════════════════════════╗\n");
    printf("║  Tall-Skinny ZGEMM Benchmark (QE workload)                       ║\n");
    printf("╚══════════════════════════════════════════════════════════════════╝\n\n");

    typedef struct { int M; int N; int K; } ZGEMMShape;
    ZGEMMShape shapes[] = {
        {4096,  150, 150},   // small tall-skinny
        {8192,  150, 150},   // medium
        {18277, 150, 150},   // QE exact: typical ZGEMM from SCF
        {18277, 150, 300},   // QE with larger K
    };
    int nshapes = sizeof(shapes) / sizeof(shapes[0]);

    printf("  M       │   N  │   K  │  AMX GFLOP/s │  GPU GFLOP/s │  Speedup  │ Status\n");
    printf("──────────┼──────┼──────┼──────────────┼──────────────┼───────────┼────────\n");

    for (int s = 0; s < nshapes; s++) {
        int M = shapes[s].M, N2 = shapes[s].N, K2 = shapes[s].K;
        size_t countA = (size_t)M * K2;
        size_t countB = (size_t)K2 * N2;
        size_t countC = (size_t)M * N2;
        // Complex GEMM: 8*M*N*K flops (4 real muls + 4 real adds per complex MAC)
        double flops = 8.0 * M * N2 * K2;
        int iters = 3;

        double* Ar = malloc(countA * sizeof(double));
        double* Ai = malloc(countA * sizeof(double));
        double* Br = malloc(countB * sizeof(double));
        double* Bi = malloc(countB * sizeof(double));
        double* Cr_gpu = malloc(countC * sizeof(double));
        double* Ci_gpu = malloc(countC * sizeof(double));

        double complex* A_ref = malloc(countA * sizeof(double complex));
        double complex* B_ref = malloc(countB * sizeof(double complex));
        double complex* C_ref = malloc(countC * sizeof(double complex));

        srand48(42);
        for (size_t i = 0; i < countA; i++) {
            Ar[i] = drand48() * 2 - 1;
            Ai[i] = drand48() * 2 - 1;
            A_ref[i] = Ar[i] + I * Ai[i];
        }
        for (size_t i = 0; i < countB; i++) {
            Br[i] = drand48() * 2 - 1;
            Bi[i] = drand48() * 2 - 1;
            B_ref[i] = Br[i] + I * Bi[i];
        }
        memset(C_ref, 0, countC * sizeof(double complex));

        // AMX reference
        double complex alpha_c = 1.0 + 0.0*I, beta_c = 0.0 + 0.0*I;
        cblas_zgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                    M, N2, K2, &alpha_c, A_ref, K2, B_ref, N2, &beta_c, C_ref, N2);
        double t0 = get_time_sec();
        for (int i = 0; i < iters; i++)
            cblas_zgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                        M, N2, K2, &alpha_c, A_ref, K2, B_ref, N2, &beta_c, C_ref, N2);
        double amx_gflops = flops * iters / ((get_time_sec() - t0) * 1e9);

        // GPU benchmark
        ABMatrix mAr = ab_matrix_create(M, K2);
        ABMatrix mAi = ab_matrix_create(M, K2);
        ABMatrix mBr = ab_matrix_create(K2, N2);
        ABMatrix mBi = ab_matrix_create(K2, N2);
        ABMatrix mCr = ab_matrix_create(M, N2);
        ABMatrix mCi = ab_matrix_create(M, N2);
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

        // Precision check (sample first 1000 elements)
        double max_err = 0;
        size_t check_count = countC < 1000 ? countC : 1000;
        for (size_t i = 0; i < check_count; i++) {
            double err_r = fabs(Cr_gpu[i] - creal(C_ref[i]));
            double err_i = fabs(Ci_gpu[i] - cimag(C_ref[i]));
            if (err_r > max_err) max_err = err_r;
            if (err_i > max_err) max_err = err_i;
        }

        double speedup = gpu_gflops / amx_gflops;
        printf("  %5d   │ %4d │ %4d │    %6.0f    │    %6.0f    │  %5.2fx %s │  %s\n",
               M, N2, K2, amx_gflops, gpu_gflops, speedup, speedup >= 1.0 ? "✓" : " ",
               max_err < 1e-10 ? "✓ PASS" : "✗ FAIL");

        ab_matrix_destroy(mAr); ab_matrix_destroy(mAi);
        ab_matrix_destroy(mBr); ab_matrix_destroy(mBi);
        ab_matrix_destroy(mCr); ab_matrix_destroy(mCi);
        free(Ar); free(Ai); free(Br); free(Bi);
        free(Cr_gpu); free(Ci_gpu);
        free(A_ref); free(B_ref); free(C_ref);
    }

    // ========================================================================
    // Part 3: Batched vs Sequential ZGEMM — measures amortization gains
    // QE profile: 427 batchable sequences, avg ~5 ZGEMMs per batch.
    // ========================================================================
    printf("\n╔══════════════════════════════════════════════════════════════════╗\n");
    printf("║  Batched ZGEMM (single cmd buffer vs sequential)                 ║\n");
    printf("╚══════════════════════════════════════════════════════════════════╝\n\n");

    {
        int M_b = 18277, N_b = 150, K_b = 150;
        int batch_sizes[] = {1, 5, 10, 20};
        int nbatch = sizeof(batch_sizes) / sizeof(batch_sizes[0]);
        size_t cA = (size_t)M_b * K_b, cB = (size_t)K_b * N_b, cC = (size_t)M_b * N_b;

        printf("  Batch │  Sequential ms │  Batched ms │  Speedup │  AMX ms  │ vs AMX\n");
        printf("────────┼────────────────┼─────────────┼──────────┼──────────┼────────\n");

        for (int bi = 0; bi < nbatch; bi++) {
            int bs = batch_sizes[bi];

            // Allocate host data
            double* h_Ar = malloc(cA * sizeof(double));
            double* h_Ai = malloc(cA * sizeof(double));
            double* h_Br = malloc(cB * sizeof(double));
            double* h_Bi = malloc(cB * sizeof(double));

            srand48(42);
            for (size_t i = 0; i < cA; i++) { h_Ar[i] = drand48()*2-1; h_Ai[i] = drand48()*2-1; }
            for (size_t i = 0; i < cB; i++) { h_Br[i] = drand48()*2-1; h_Bi[i] = drand48()*2-1; }

            // Create GPU matrices for the batch
            ABMatrix* gAr = malloc(bs * sizeof(ABMatrix));
            ABMatrix* gAi = malloc(bs * sizeof(ABMatrix));
            ABMatrix* gBr = malloc(bs * sizeof(ABMatrix));
            ABMatrix* gBi = malloc(bs * sizeof(ABMatrix));
            ABMatrix* gCr = malloc(bs * sizeof(ABMatrix));
            ABMatrix* gCi = malloc(bs * sizeof(ABMatrix));

            for (int i = 0; i < bs; i++) {
                gAr[i] = ab_matrix_create(M_b, K_b); ab_matrix_upload(gAr[i], h_Ar, true);
                gAi[i] = ab_matrix_create(M_b, K_b); ab_matrix_upload(gAi[i], h_Ai, true);
                gBr[i] = ab_matrix_create(K_b, N_b); ab_matrix_upload(gBr[i], h_Br, true);
                gBi[i] = ab_matrix_create(K_b, N_b); ab_matrix_upload(gBi[i], h_Bi, true);
                gCr[i] = ab_matrix_create(M_b, N_b);
                gCi[i] = ab_matrix_create(M_b, N_b);
            }

            // Warmup
            ab_zgemm(gAr[0], gAi[0], gBr[0], gBi[0], gCr[0], gCi[0]);

            // Sequential: individual ab_zgemm calls
            double t_seq_start = get_time_sec();
            for (int i = 0; i < bs; i++)
                ab_zgemm(gAr[i], gAi[i], gBr[i], gBi[i], gCr[i], gCi[i]);
            double t_seq = (get_time_sec() - t_seq_start) * 1000.0;

            // Batched: single command buffer
            double t_bat_start = get_time_sec();
            ab_zgemm_batched(bs, gAr, gAi, gBr, gBi, gCr, gCi);
            double t_bat = (get_time_sec() - t_bat_start) * 1000.0;

            // AMX reference timing
            double complex* A_r = malloc(cA * sizeof(double complex));
            double complex* B_r = malloc(cB * sizeof(double complex));
            double complex* C_r = malloc(cC * sizeof(double complex));
            for (size_t i = 0; i < cA; i++) A_r[i] = h_Ar[i] + I * h_Ai[i];
            for (size_t i = 0; i < cB; i++) B_r[i] = h_Br[i] + I * h_Bi[i];
            double complex alpha_c = 1.0, beta_c = 0.0;

            // warmup
            cblas_zgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                        M_b, N_b, K_b, &alpha_c, A_r, K_b, B_r, N_b, &beta_c, C_r, N_b);
            double t_amx_start = get_time_sec();
            for (int i = 0; i < bs; i++)
                cblas_zgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                            M_b, N_b, K_b, &alpha_c, A_r, K_b, B_r, N_b, &beta_c, C_r, N_b);
            double t_amx = (get_time_sec() - t_amx_start) * 1000.0;

            double batch_speedup = t_seq / t_bat;
            double vs_amx = t_amx / t_bat;

            printf("  %3d   │    %8.1f    │  %8.1f   │  %5.2fx  │ %7.1f  │ %5.2fx%s\n",
                   bs, t_seq, t_bat, batch_speedup, t_amx, vs_amx, vs_amx >= 1.0 ? " ✓" : "");

            for (int i = 0; i < bs; i++) {
                ab_matrix_destroy(gAr[i]); ab_matrix_destroy(gAi[i]);
                ab_matrix_destroy(gBr[i]); ab_matrix_destroy(gBi[i]);
                ab_matrix_destroy(gCr[i]); ab_matrix_destroy(gCi[i]);
            }
            free(gAr); free(gAi); free(gBr); free(gBi); free(gCr); free(gCi);
            free(h_Ar); free(h_Ai); free(h_Br); free(h_Bi);
            free(A_r); free(B_r); free(C_r);
        }
    }

    printf("\n");
    ab_shutdown();
    return 0;
}
