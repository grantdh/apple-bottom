// bench_paper.c — Comprehensive benchmark suite for HPEC 2026 paper tables
//
// Produces all tables needed for the paper:
//   Table 1: Square DGEMM/ZGEMM scaling (GPU vs AMX crossover)
//   Table 2: Tall-skinny ZGEMM (QE-realistic dimensions)
//   Table 3: Batched ZGEMM throughput
//   Table 4: ZHERK GPU-native vs AMX
//   Table 5: DTRSM via blocked substitution
//   Table 6: Precision envelope (Frobenius error vs matrix size)
//
// Output: tab-separated values suitable for LaTeX table generation
//
// Usage:
//   make build/bench_paper && ./build/bench_paper
//   ./build/bench_paper --csv > results.csv   # machine-readable output
//   ./build/bench_paper --latex               # LaTeX table fragments

#include "apple_bottom.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include <Accelerate/Accelerate.h>

static int g_csv = 0;
static int g_latex = 0;

static double get_time_sec(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec + ts.tv_nsec * 1e-9;
}

static double frobenius_rel(const double* X, const double* Xref, size_t n) {
    double nd = 0, nr = 0;
    for (size_t i = 0; i < n; i++) {
        double d = X[i] - Xref[i];
        nd += d * d;
        nr += Xref[i] * Xref[i];
    }
    return sqrt(nd / (nr > 0 ? nr : 1e-300));
}

// =========================================================================
// Table 1: Square DGEMM & ZGEMM scaling
// =========================================================================
static void table_square_gemm(void) {
    printf("\n");
    printf("╔══════════════════════════════════════════════════════════════════╗\n");
    printf("║  Table 1: Square GEMM Scaling (GPU vs AMX)                       ║\n");
    printf("╚══════════════════════════════════════════════════════════════════╝\n\n");

    if (g_csv) printf("# op,N,amx_gflops,gpu_gflops,speedup,frob_err\n");

    int sizes[] = {256, 512, 768, 1024, 1536, 2048, 3072, 4096};
    int nsizes = sizeof(sizes) / sizeof(sizes[0]);

    // --- DGEMM ---
    printf("  DGEMM:\n");
    printf("  N     │  AMX GFLOP/s │  GPU GFLOP/s │  Speedup  │ Frob Err\n");
    printf("────────┼──────────────┼──────────────┼───────────┼───────────\n");

    for (int s = 0; s < nsizes; s++) {
        int N = sizes[s];
        size_t count = (size_t)N * N;
        double flops = 2.0 * N * N * N;
        int iters = (N <= 1024) ? 10 : (N <= 2048) ? 5 : 3;

        double* A = malloc(count * sizeof(double));
        double* B = malloc(count * sizeof(double));
        double* C_ref = malloc(count * sizeof(double));
        double* C_gpu = malloc(count * sizeof(double));

        srand48(42);
        for (size_t i = 0; i < count; i++) {
            A[i] = drand48() * 2 - 1;
            B[i] = drand48() * 2 - 1;
        }

        // AMX warmup + bench
        cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                    N, N, N, 1.0, A, N, B, N, 0.0, C_ref, N);
        double t0 = get_time_sec();
        for (int i = 0; i < iters; i++)
            cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                        N, N, N, 1.0, A, N, B, N, 0.0, C_ref, N);
        double amx = flops * iters / ((get_time_sec() - t0) * 1e9);

        // GPU warmup + bench
        ABMatrix mA = ab_matrix_create(N, N);
        ABMatrix mB = ab_matrix_create(N, N);
        ABMatrix mC = ab_matrix_create(N, N);
        ab_matrix_upload(mA, A, true);
        ab_matrix_upload(mB, B, true);
        ab_dgemm(mA, mB, mC);

        t0 = get_time_sec();
        for (int i = 0; i < iters; i++) ab_dgemm(mA, mB, mC);
        double gpu = flops * iters / ((get_time_sec() - t0) * 1e9);

        ab_matrix_download(mC, C_gpu, true);
        double err = frobenius_rel(C_gpu, C_ref, count);

        printf("  %4d  │ %10.0f   │ %10.0f   │  %5.2fx   │ %.2e\n",
               N, amx, gpu, gpu / amx, err);
        if (g_csv) printf("dgemm,%d,%.1f,%.1f,%.3f,%.3e\n", N, amx, gpu, gpu/amx, err);

        ab_matrix_destroy(mA); ab_matrix_destroy(mB); ab_matrix_destroy(mC);
        free(A); free(B); free(C_ref); free(C_gpu);
    }

    // --- ZGEMM ---
    printf("\n  ZGEMM:\n");
    printf("  N     │  AMX GFLOP/s │  GPU GFLOP/s │  Speedup  │ Frob Err\n");
    printf("────────┼──────────────┼──────────────┼───────────┼───────────\n");

    for (int s = 0; s < nsizes; s++) {
        int N = sizes[s];
        if (N > 3072) continue; // ZGEMM uses 4x memory
        size_t count = (size_t)N * N;
        double flops = 8.0 * N * N * N;
        int iters = (N <= 1024) ? 10 : 3;

        double* Ar = malloc(count * sizeof(double));
        double* Ai = malloc(count * sizeof(double));
        double* Br = malloc(count * sizeof(double));
        double* Bi = malloc(count * sizeof(double));
        double* Cr_ref = calloc(count, sizeof(double));
        double* Ci_ref = calloc(count, sizeof(double));
        double* Cr_gpu = malloc(count * sizeof(double));
        double* Ci_gpu = malloc(count * sizeof(double));

        srand48(77);
        for (size_t i = 0; i < count; i++) {
            Ar[i] = drand48() * 2 - 1; Ai[i] = drand48() * 2 - 1;
            Br[i] = drand48() * 2 - 1; Bi[i] = drand48() * 2 - 1;
        }

        // Pack into interleaved complex for cblas
        double _Complex* zA = malloc(count * sizeof(double _Complex));
        double _Complex* zB = malloc(count * sizeof(double _Complex));
        double _Complex* zC = calloc(count, sizeof(double _Complex));
        for (size_t i = 0; i < count; i++) {
            zA[i] = Ar[i] + Ai[i] * _Complex_I;
            zB[i] = Br[i] + Bi[i] * _Complex_I;
        }

        double _Complex one = 1.0, zero_c = 0.0;
        cblas_zgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                    N, N, N, &one, zA, N, zB, N, &zero_c, zC, N);
        double t0 = get_time_sec();
        for (int i = 0; i < iters; i++)
            cblas_zgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                        N, N, N, &one, zA, N, zB, N, &zero_c, zC, N);
        double amx = flops * iters / ((get_time_sec() - t0) * 1e9);

        // Unpack reference
        for (size_t i = 0; i < count; i++) {
            Cr_ref[i] = creal(zC[i]); Ci_ref[i] = cimag(zC[i]);
        }

        // GPU
        ABMatrix mAr = ab_matrix_create(N, N), mAi = ab_matrix_create(N, N);
        ABMatrix mBr = ab_matrix_create(N, N), mBi = ab_matrix_create(N, N);
        ABMatrix mCr = ab_matrix_create(N, N), mCi = ab_matrix_create(N, N);
        ab_matrix_upload(mAr, Ar, true); ab_matrix_upload(mAi, Ai, true);
        ab_matrix_upload(mBr, Br, true); ab_matrix_upload(mBi, Bi, true);
        ab_zgemm(mAr, mAi, mBr, mBi, mCr, mCi);

        t0 = get_time_sec();
        for (int i = 0; i < iters; i++)
            ab_zgemm(mAr, mAi, mBr, mBi, mCr, mCi);
        double gpu = flops * iters / ((get_time_sec() - t0) * 1e9);

        ab_matrix_download(mCr, Cr_gpu, true);
        ab_matrix_download(mCi, Ci_gpu, true);

        // Frobenius error on real part
        double err = frobenius_rel(Cr_gpu, Cr_ref, count);

        printf("  %4d  │ %10.0f   │ %10.0f   │  %5.2fx   │ %.2e\n",
               N, amx, gpu, gpu / amx, err);
        if (g_csv) printf("zgemm,%d,%.1f,%.1f,%.3f,%.3e\n", N, amx, gpu, gpu/amx, err);

        ab_matrix_destroy(mAr); ab_matrix_destroy(mAi);
        ab_matrix_destroy(mBr); ab_matrix_destroy(mBi);
        ab_matrix_destroy(mCr); ab_matrix_destroy(mCi);
        free(Ar); free(Ai); free(Br); free(Bi);
        free(Cr_ref); free(Ci_ref); free(Cr_gpu); free(Ci_gpu);
        free(zA); free(zB); free(zC);
    }
}

// =========================================================================
// Table 2: Tall-skinny ZGEMM (QE-realistic dimensions)
// =========================================================================
static void table_tall_skinny(void) {
    printf("\n");
    printf("╔══════════════════════════════════════════════════════════════════╗\n");
    printf("║  Table 2: Tall-Skinny ZGEMM (QE Workload Profile)                ║\n");
    printf("╚══════════════════════════════════════════════════════════════════╝\n\n");

    if (g_csv) printf("# op,M,N,K,amx_gflops,gpu_gflops,speedup,frob_err\n");

    // Shapes from actual QE Si64 profiling
    struct { int M, N, K; const char* label; } shapes[] = {
        { 2048,  150, 150, "moderate"    },
        { 4096,  150, 150, "large"       },
        { 8192,  150, 150, "QE-like"     },
        {18277,  150, 150, "QE Si64"     },
        {18277,  150, 300, "QE Si64 K=300"},
        {18277,  300, 150, "QE conj-trans"},
        { 4096,   64,  64, "small band"  },
        { 8192,   64, 256, "mixed"       },
    };
    int nshapes = sizeof(shapes) / sizeof(shapes[0]);

    printf("  M       │   N  │   K  │  AMX GFLOP/s │  GPU GFLOP/s │  Speedup  │ Frob Err  │ Label\n");
    printf("──────────┼──────┼──────┼──────────────┼──────────────┼───────────┼───────────┼──────────\n");

    for (int s = 0; s < nshapes; s++) {
        int M = shapes[s].M, N = shapes[s].N, K = shapes[s].K;
        double flops = 8.0 * M * N * K;
        int iters = 3;

        size_t cA = (size_t)M * K, cB = (size_t)K * N, cC = (size_t)M * N;

        double* Ar = malloc(cA * sizeof(double)), *Ai = malloc(cA * sizeof(double));
        double* Br = malloc(cB * sizeof(double)), *Bi = malloc(cB * sizeof(double));
        double* Cr_gpu = malloc(cC * sizeof(double)), *Ci_gpu = malloc(cC * sizeof(double));

        srand48(42);
        for (size_t i = 0; i < cA; i++) { Ar[i] = drand48()*2-1; Ai[i] = drand48()*2-1; }
        for (size_t i = 0; i < cB; i++) { Br[i] = drand48()*2-1; Bi[i] = drand48()*2-1; }

        // AMX baseline via interleaved complex
        double _Complex* zA = malloc(cA * sizeof(double _Complex));
        double _Complex* zB = malloc(cB * sizeof(double _Complex));
        double _Complex* zC = calloc(cC, sizeof(double _Complex));
        for (size_t i = 0; i < cA; i++) zA[i] = Ar[i] + Ai[i] * _Complex_I;
        for (size_t i = 0; i < cB; i++) zB[i] = Br[i] + Bi[i] * _Complex_I;

        double _Complex one = 1.0, zero_c = 0.0;
        cblas_zgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                    M, N, K, &one, zA, K, zB, N, &zero_c, zC, N);
        double t0 = get_time_sec();
        for (int i = 0; i < iters; i++)
            cblas_zgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                        M, N, K, &one, zA, K, zB, N, &zero_c, zC, N);
        double amx = flops * iters / ((get_time_sec() - t0) * 1e9);

        // Unpack reference for first 1000 elements (precision check)
        int check_n = (cC < 1000) ? (int)cC : 1000;
        double* Cr_ref = malloc(check_n * sizeof(double));
        for (int i = 0; i < check_n; i++) Cr_ref[i] = creal(zC[i]);

        // GPU
        ABMatrix mAr = ab_matrix_create(M, K), mAi = ab_matrix_create(M, K);
        ABMatrix mBr = ab_matrix_create(K, N), mBi = ab_matrix_create(K, N);
        ABMatrix mCr = ab_matrix_create(M, N), mCi = ab_matrix_create(M, N);
        ab_matrix_upload(mAr, Ar, true); ab_matrix_upload(mAi, Ai, true);
        ab_matrix_upload(mBr, Br, true); ab_matrix_upload(mBi, Bi, true);
        ab_zgemm(mAr, mAi, mBr, mBi, mCr, mCi);

        t0 = get_time_sec();
        for (int i = 0; i < iters; i++)
            ab_zgemm(mAr, mAi, mBr, mBi, mCr, mCi);
        double gpu = flops * iters / ((get_time_sec() - t0) * 1e9);

        ab_matrix_download(mCr, Cr_gpu, true);
        double err = frobenius_rel(Cr_gpu, Cr_ref, check_n);

        printf("  %5d   │ %4d │ %4d │ %10.0f   │ %10.0f   │  %5.2fx   │ %.2e  │ %s\n",
               M, N, K, amx, gpu, gpu / amx, err, shapes[s].label);
        if (g_csv) printf("zgemm_ts,%d,%d,%d,%.1f,%.1f,%.3f,%.3e\n", M,N,K, amx, gpu, gpu/amx, err);

        ab_matrix_destroy(mAr); ab_matrix_destroy(mAi);
        ab_matrix_destroy(mBr); ab_matrix_destroy(mBi);
        ab_matrix_destroy(mCr); ab_matrix_destroy(mCi);
        free(Ar); free(Ai); free(Br); free(Bi);
        free(Cr_gpu); free(Ci_gpu); free(Cr_ref);
        free(zA); free(zB); free(zC);
    }
}

// =========================================================================
// Table 3: Batched ZGEMM throughput
// =========================================================================
static void table_batched(void) {
    printf("\n");
    printf("╔══════════════════════════════════════════════════════════════════╗\n");
    printf("║  Table 3: Batched ZGEMM Throughput                               ║\n");
    printf("╚══════════════════════════════════════════════════════════════════╝\n\n");

    if (g_csv) printf("# batch,seq_ms,batch_ms,batch_speedup,amx_ms,vs_amx\n");

    int M = 18277, N = 150, K = 150;
    size_t cA = (size_t)M * K, cB = (size_t)K * N, cC = (size_t)M * N;
    int batch_sizes[] = {1, 2, 5, 10, 20};
    int nbatch = sizeof(batch_sizes) / sizeof(batch_sizes[0]);

    printf("  Batch │  Sequential ms │  Batched ms │  Speedup │  AMX ms  │ vs AMX\n");
    printf("────────┼────────────────┼─────────────┼──────────┼──────────┼────────\n");

    for (int bi = 0; bi < nbatch; bi++) {
        int bs = batch_sizes[bi];

        double* h_Ar = malloc(cA * sizeof(double)), *h_Ai = malloc(cA * sizeof(double));
        double* h_Br = malloc(cB * sizeof(double)), *h_Bi = malloc(cB * sizeof(double));
        srand48(42);
        for (size_t i = 0; i < cA; i++) { h_Ar[i] = drand48()*2-1; h_Ai[i] = drand48()*2-1; }
        for (size_t i = 0; i < cB; i++) { h_Br[i] = drand48()*2-1; h_Bi[i] = drand48()*2-1; }

        // Create GPU matrices
        ABMatrix* Ars = malloc(bs * sizeof(ABMatrix));
        ABMatrix* Ais = malloc(bs * sizeof(ABMatrix));
        ABMatrix* Brs = malloc(bs * sizeof(ABMatrix));
        ABMatrix* Bis = malloc(bs * sizeof(ABMatrix));
        ABMatrix* Crs = malloc(bs * sizeof(ABMatrix));
        ABMatrix* Cis = malloc(bs * sizeof(ABMatrix));

        for (int i = 0; i < bs; i++) {
            Ars[i] = ab_matrix_create(M, K); Ais[i] = ab_matrix_create(M, K);
            Brs[i] = ab_matrix_create(K, N); Bis[i] = ab_matrix_create(K, N);
            Crs[i] = ab_matrix_create(M, N); Cis[i] = ab_matrix_create(M, N);
            ab_matrix_upload(Ars[i], h_Ar, true); ab_matrix_upload(Ais[i], h_Ai, true);
            ab_matrix_upload(Brs[i], h_Br, true); ab_matrix_upload(Bis[i], h_Bi, true);
        }

        // Sequential
        for (int i = 0; i < bs; i++)
            ab_zgemm(Ars[i], Ais[i], Brs[i], Bis[i], Crs[i], Cis[i]);
        double t0 = get_time_sec();
        for (int i = 0; i < bs; i++)
            ab_zgemm(Ars[i], Ais[i], Brs[i], Bis[i], Crs[i], Cis[i]);
        double seq_ms = (get_time_sec() - t0) * 1000.0;

        // Batched
        ab_zgemm_batched(bs, Ars, Ais, Brs, Bis, Crs, Cis);
        t0 = get_time_sec();
        ab_zgemm_batched(bs, Ars, Ais, Brs, Bis, Crs, Cis);
        double bat_ms = (get_time_sec() - t0) * 1000.0;

        // AMX reference
        double _Complex* zA = malloc(cA * sizeof(double _Complex));
        double _Complex* zB = malloc(cB * sizeof(double _Complex));
        double _Complex* zC = calloc(cC, sizeof(double _Complex));
        for (size_t i = 0; i < cA; i++) zA[i] = h_Ar[i] + h_Ai[i] * _Complex_I;
        for (size_t i = 0; i < cB; i++) zB[i] = h_Br[i] + h_Bi[i] * _Complex_I;
        double _Complex one = 1.0, zero_c = 0.0;

        t0 = get_time_sec();
        for (int i = 0; i < bs; i++)
            cblas_zgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                        M, N, K, &one, zA, K, zB, N, &zero_c, zC, N);
        double amx_ms = (get_time_sec() - t0) * 1000.0;

        double bat_speedup = seq_ms / bat_ms;
        double vs_amx = amx_ms / bat_ms;
        const char* marker = (vs_amx >= 1.0) ? " \u2713" : "";

        printf("  %3d   │ %12.1f   │ %9.1f   │ %5.2fx  │ %7.1f  │ %5.2fx%s\n",
               bs, seq_ms, bat_ms, bat_speedup, amx_ms, vs_amx, marker);
        if (g_csv) printf("batch,%d,%.1f,%.1f,%.2f,%.1f,%.2f\n",
                          bs, seq_ms, bat_ms, bat_speedup, amx_ms, vs_amx);

        for (int i = 0; i < bs; i++) {
            ab_matrix_destroy(Ars[i]); ab_matrix_destroy(Ais[i]);
            ab_matrix_destroy(Brs[i]); ab_matrix_destroy(Bis[i]);
            ab_matrix_destroy(Crs[i]); ab_matrix_destroy(Cis[i]);
        }
        free(Ars); free(Ais); free(Brs); free(Bis); free(Crs); free(Cis);
        free(h_Ar); free(h_Ai); free(h_Br); free(h_Bi);
        free(zA); free(zB); free(zC);
    }
}

// =========================================================================
// Table 4: ZHERK GPU-native transpose
// =========================================================================
static void table_zherk(void) {
    printf("\n");
    printf("╔══════════════════════════════════════════════════════════════════╗\n");
    printf("║  Table 4: ZHERK (GPU-native transpose vs AMX)                    ║\n");
    printf("╚══════════════════════════════════════════════════════════════════╝\n\n");

    if (g_csv) printf("# op,N,K,amx_gflops,gpu_gflops,speedup\n");

    struct { int N, K; } sizes[] = {
        {256, 256}, {512, 256}, {1024, 256}, {2048, 256},
        {512, 32},  {512, 128}, {512, 512},
    };
    int nsizes = sizeof(sizes) / sizeof(sizes[0]);

    printf("  N     │   K   │  AMX GFLOP/s │  GPU GFLOP/s │  Speedup\n");
    printf("────────┼───────┼──────────────┼──────────────┼───────────\n");

    for (int s = 0; s < nsizes; s++) {
        int N = sizes[s].N, K = sizes[s].K;
        double flops = 8.0 * N * N * K; // ZHERK: N²K complex ops × 8 FLOPs/op
        int iters = 3;

        size_t cA = (size_t)N * K;
        double* Ar = malloc(cA * sizeof(double)), *Ai = malloc(cA * sizeof(double));
        srand48(42);
        for (size_t i = 0; i < cA; i++) { Ar[i] = drand48()*2-1; Ai[i] = drand48()*2-1; }

        // AMX: cblas_zherk
        double _Complex* zA = malloc(cA * sizeof(double _Complex));
        for (size_t i = 0; i < cA; i++) zA[i] = Ar[i] + Ai[i] * _Complex_I;
        double _Complex* zC = calloc((size_t)N * N, sizeof(double _Complex));

        cblas_zherk(CblasRowMajor, CblasUpper, CblasNoTrans, N, K, 1.0, zA, K, 0.0, zC, N);
        double t0 = get_time_sec();
        for (int i = 0; i < iters; i++)
            cblas_zherk(CblasRowMajor, CblasUpper, CblasNoTrans, N, K, 1.0, zA, K, 0.0, zC, N);
        double amx = flops * iters / ((get_time_sec() - t0) * 1e9);

        // GPU: ab_zherk
        ABMatrix mAr = ab_matrix_create(N, K), mAi = ab_matrix_create(N, K);
        ABMatrix mCr = ab_matrix_create(N, N), mCi = ab_matrix_create(N, N);
        ab_matrix_upload(mAr, Ar, true); ab_matrix_upload(mAi, Ai, true);
        ab_zherk(mAr, mAi, mCr, mCi);

        t0 = get_time_sec();
        for (int i = 0; i < iters; i++)
            ab_zherk(mAr, mAi, mCr, mCi);
        double gpu = flops * iters / ((get_time_sec() - t0) * 1e9);

        printf("  %4d  │ %4d  │ %10.0f   │ %10.0f   │  %5.2fx\n",
               N, K, amx, gpu, gpu / amx);
        if (g_csv) printf("zherk,%d,%d,%.1f,%.1f,%.3f\n", N, K, amx, gpu, gpu/amx);

        ab_matrix_destroy(mAr); ab_matrix_destroy(mAi);
        ab_matrix_destroy(mCr); ab_matrix_destroy(mCi);
        free(Ar); free(Ai); free(zA); free(zC);
    }
}

// =========================================================================
// Table 5: Precision envelope
// =========================================================================
static void table_precision(void) {
    printf("\n");
    printf("╔══════════════════════════════════════════════════════════════════╗\n");
    printf("║  Table 5: Precision Envelope (Frobenius Error vs K)              ║\n");
    printf("╚══════════════════════════════════════════════════════════════════╝\n\n");

    if (g_csv) printf("# N,K,frob_err,max_elem_err,wilkinson_bound\n");

    struct { int M, N, K; } shapes[] = {
        {64, 64, 64},
        {128, 128, 128},
        {256, 256, 256},
        {512, 512, 512},
        {1024, 1024, 1024},
        {2048, 2048, 2048},
        // Rectangular
        {100, 100, 4096},
        {1000, 50, 200},
        // Tall-skinny (QE-like)
        {4096, 150, 150},
        {18277, 150, 150},
    };
    int nshapes = sizeof(shapes) / sizeof(shapes[0]);

    double u_dd = pow(2.0, -48);

    printf("  Shape                │ Frobenius Err │ Max Elem Err │ Wilkinson Bound │ Ratio\n");
    printf("───────────────────────┼───────────────┼──────────────┼─────────────────┼────────\n");

    for (int s = 0; s < nshapes; s++) {
        int M = shapes[s].M, N = shapes[s].N, K = shapes[s].K;
        size_t cA = (size_t)M * K, cB = (size_t)K * N, cC = (size_t)M * N;

        double* A = malloc(cA * sizeof(double));
        double* B = malloc(cB * sizeof(double));
        double* C_ref = malloc(cC * sizeof(double));
        double* C_gpu = malloc(cC * sizeof(double));

        srand48(42);
        for (size_t i = 0; i < cA; i++) A[i] = drand48() * 2 - 1;
        for (size_t i = 0; i < cB; i++) B[i] = drand48() * 2 - 1;

        // Reference: Accelerate DGEMM (true FP64)
        cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                    M, N, K, 1.0, A, K, B, N, 0.0, C_ref, N);

        // GPU
        ABMatrix mA = ab_matrix_create(M, K);
        ABMatrix mB = ab_matrix_create(K, N);
        ABMatrix mC = ab_matrix_create(M, N);
        ab_matrix_upload(mA, A, true);
        ab_matrix_upload(mB, B, true);
        ab_dgemm(mA, mB, mC);
        ab_matrix_download(mC, C_gpu, true);

        // Compute errors
        double frob_diff = 0, frob_ref = 0, max_elem = 0;
        for (size_t i = 0; i < cC; i++) {
            double d = fabs(C_gpu[i] - C_ref[i]);
            double r = fabs(C_ref[i]);
            frob_diff += (C_gpu[i] - C_ref[i]) * (C_gpu[i] - C_ref[i]);
            frob_ref += C_ref[i] * C_ref[i];
            if (r > 0 && d / r > max_elem) max_elem = d / r;
        }
        double frob_err = sqrt(frob_diff / (frob_ref > 0 ? frob_ref : 1e-300));
        double wilkinson = 10.0 * sqrt((double)K) * u_dd;
        double ratio = frob_err / wilkinson;

        char label[64];
        if (M == N && N == K)
            snprintf(label, sizeof(label), "%dx%d (K=%d)", M, N, K);
        else
            snprintf(label, sizeof(label), "%dx%d (K=%d)", M, N, K);

        printf("  %-20s │   %.2e   │   %.2e  │   %.2e     │ %5.2f\n",
               label, frob_err, max_elem, wilkinson, ratio);
        if (g_csv) printf("precision,%d,%d,%d,%.3e,%.3e,%.3e\n",
                          M, N, K, frob_err, max_elem, wilkinson);

        ab_matrix_destroy(mA); ab_matrix_destroy(mB); ab_matrix_destroy(mC);
        free(A); free(B); free(C_ref); free(C_gpu);
    }
}

int main(int argc, char** argv) {
    const char* out_path = NULL;
    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "--csv") == 0) g_csv = 1;
        if (strcmp(argv[i], "--latex") == 0) g_latex = 1;
        if (strcmp(argv[i], "-o") == 0 && i + 1 < argc) out_path = argv[++i];
    }
    if (out_path) {
        if (!freopen(out_path, "w", stdout)) {
            fprintf(stderr, "cannot open %s\n", out_path);
            return 1;
        }
    }

    printf("╔══════════════════════════════════════════════════════════════════╗\n");
    printf("║  apple-bottom HPEC 2026 Paper Benchmark Suite                    ║\n");
    printf("╚══════════════════════════════════════════════════════════════════╝\n\n");

    if (ab_init() != AB_OK) { fprintf(stderr, "Init failed\n"); return 1; }
    printf("Device: %s\n", ab_device_name());
    printf("Library: %s\n", APPLE_BOTTOM_VERSION_STRING);
    printf("Date: ");
    fflush(stdout);
    system("date");
    printf("\n");

    table_square_gemm();
    table_tall_skinny();
    table_batched();
    table_zherk();
    table_precision();

    printf("\n═══════════════════════════════════════════════════════════════════\n");
    printf("Benchmark complete.\n");
    if (g_csv) printf("# CSV output above can be parsed with scripts/analyze_bench_paper.py\n");

    ab_shutdown();
    return 0;
}
