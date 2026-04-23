// SGEMM Benchmark — apple-bottom
//
// Measures cblas_sgemm (Accelerate/AMX) FP32 throughput at matched sizes with
// bench_dgemm so FP32 utilization analyses can reference an attainable-peak
// baseline rather than the theoretical 13.6 TFLOP/s ALU peak.
//
// CPU-only. A GPU-native SGEMM kernel is out of scope for this bench;
// apple-bottom's Metal path operates on DD pairs (FP32x2), not native FP32.

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
            fprintf(csv, "op,size,iters,amx_gflops\n");
        }
    }

    printf("╔══════════════════════════════════════════════════════════════════╗\n");
    printf("║  apple-bottom SGEMM Reference (AMX FP32 peak, matched to DGEMM)  ║\n");
    printf("╚══════════════════════════════════════════════════════════════════╝\n\n");

    int sizes[] = {256, 512, 1024, 2048, 3072, 4096};

    printf("  Size    │  AMX GFLOP/s (FP32)\n");
    printf("──────────┼────────────────────\n");

    for (int s = 0; s < 6; s++) {
        int N = sizes[s];
        size_t count = (size_t)N * N;
        double flops = 2.0 * N * N * N;
        int iters = (N <= 1024) ? 10 : 5;

        float* A = malloc(count * sizeof(float));
        float* B = malloc(count * sizeof(float));
        float* C = malloc(count * sizeof(float));

        srand48(42);
        for (size_t i = 0; i < count; i++) {
            A[i] = (float)(drand48() * 2 - 1);
            B[i] = (float)(drand48() * 2 - 1);
        }

        cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, N, N, N,
                    1.0f, A, N, B, N, 0.0f, C, N);
        double t0 = get_time_sec();
        for (int i = 0; i < iters; i++)
            cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, N, N, N,
                        1.0f, A, N, B, N, 0.0f, C, N);
        double amx_gflops = flops * iters / ((get_time_sec() - t0) * 1e9);

        printf("  %4d    │        %6.0f\n", N, amx_gflops);
        if (csv) {
            fprintf(csv, "sgemm,%d,%d,%.2f\n", N, iters, amx_gflops);
            fflush(csv);
        }

        free(A); free(B); free(C);
    }

    printf("\n");
    printf("Note: AMX FP32 peak is the attainable single-precision reference for\n");
    printf("      FP32 utilization calculations. Theoretical GPU FP32 peak\n");
    printf("      (~13.6 TFLOP/s @ 1.398 GHz) is a separate upper bound.\n");

    if (csv) fclose(csv);
    return 0;
}
