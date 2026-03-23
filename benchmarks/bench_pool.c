// Memory Pool Benchmark — apple-bottom
#include "apple_bottom.h"
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

static double get_time_sec(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec + ts.tv_nsec * 1e-9;
}

int main(void) {
    printf("╔══════════════════════════════════════════════════════════════════╗\n");
    printf("║  apple-bottom Memory Pool Benchmark                              ║\n");
    printf("╚══════════════════════════════════════════════════════════════════╝\n\n");
    
    if (ab_init() != AB_OK) { fprintf(stderr, "Init failed\n"); return 1; }
    printf("Device: %s\n\n", ab_device_name());
    
    int sizes[] = {64, 128, 256, 512, 1024};
    int iterations = 100;
    
    printf("Simulating %d SCF iterations per size:\n\n", iterations);
    printf("  Size    │  No Pool (ms) │  With Pool (ms) │  Speedup\n");
    printf("──────────┼───────────────┼─────────────────┼──────────\n");
    
    for (int s = 0; s < 5; s++) {
        int N = sizes[s];
        size_t count = (size_t)N * N;
        double* data = (double*)malloc(count * sizeof(double));
        for (size_t i = 0; i < count; i++) data[i] = 1.0;
        
        // Without pool
        double t0 = get_time_sec();
        for (int iter = 0; iter < iterations; iter++) {
            ABMatrix A = ab_matrix_create(N, N);
            ABMatrix B = ab_matrix_create(N, N);
            ABMatrix C = ab_matrix_create(N, N);
            ab_matrix_upload(A, data, false);
            ab_matrix_upload(B, data, false);
            ab_dgemm(A, B, C);
            ab_matrix_destroy(A);
            ab_matrix_destroy(B);
            ab_matrix_destroy(C);
        }
        double no_pool_ms = (get_time_sec() - t0) * 1000;
        
        // With pool
        ABMemoryPool pool = ab_pool_create(0);
        t0 = get_time_sec();
        for (int iter = 0; iter < iterations; iter++) {
            ABMatrix A = ab_pool_get_matrix(pool, N, N);
            ABMatrix B = ab_pool_get_matrix(pool, N, N);
            ABMatrix C = ab_pool_get_matrix(pool, N, N);
            ab_matrix_upload(A, data, false);
            ab_matrix_upload(B, data, false);
            ab_dgemm(A, B, C);
            ab_pool_reset(pool);
        }
        double pool_ms = (get_time_sec() - t0) * 1000;
        ab_pool_destroy(pool);
        
        double speedup = no_pool_ms / pool_ms;
        printf("  %4d    │    %7.1f    │     %7.1f     │  %5.2fx %s\n",
               N, no_pool_ms, pool_ms, speedup, speedup > 1.0 ? "✓" : "");
        
        free(data);
    }
    
    printf("\n");
    ab_shutdown();
    return 0;
}
