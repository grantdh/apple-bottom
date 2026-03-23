// Async API Benchmark — apple-bottom
#include "apple_bottom.h"
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>

static double get_time_sec(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec + ts.tv_nsec * 1e-9;
}

// Simulate CPU work (e.g., preparing next iteration)
static double do_cpu_work(int iterations) {
    volatile double sum = 0;
    for (int i = 0; i < iterations; i++) {
        sum += sin((double)i * 0.001);
    }
    return sum;
}

int main(void) {
    printf("╔══════════════════════════════════════════════════════════════════╗\n");
    printf("║  apple-bottom Async API Benchmark                                ║\n");
    printf("╚══════════════════════════════════════════════════════════════════╝\n\n");
    
    if (ab_init() != AB_OK) { fprintf(stderr, "Init failed\n"); return 1; }
    printf("Device: %s\n\n", ab_device_name());
    
    int N = 1024;
    int iterations = 20;
    int cpu_work_size = 50000;  // Tuned to roughly match GPU time
    
    size_t count = (size_t)N * N;
    double* data = (double*)malloc(count * sizeof(double));
    for (size_t i = 0; i < count; i++) data[i] = 1.0;
    
    ABMatrix A = ab_matrix_create(N, N);
    ABMatrix B = ab_matrix_create(N, N);
    ABMatrix C = ab_matrix_create(N, N);
    ab_matrix_upload(A, data, false);
    ab_matrix_upload(B, data, false);
    
    printf("Matrix size: %dx%d, Iterations: %d\n\n", N, N, iterations);
    
    // Sync: GPU then CPU (serial)
    double t0 = get_time_sec();
    for (int i = 0; i < iterations; i++) {
        ab_dgemm(A, B, C);
        do_cpu_work(cpu_work_size);
    }
    double sync_ms = (get_time_sec() - t0) * 1000;
    
    // Async: GPU and CPU overlap
    t0 = get_time_sec();
    for (int i = 0; i < iterations; i++) {
        ABFuture f = ab_dgemm_async(A, B, C);
        do_cpu_work(cpu_work_size);
        ab_future_wait(f);
        ab_future_destroy(f);
    }
    double async_ms = (get_time_sec() - t0) * 1000;
    
    printf("  Mode      │  Total Time (ms)  │  Per Iteration\n");
    printf("────────────┼───────────────────┼────────────────\n");
    printf("  Sync      │     %7.1f       │    %5.2f ms\n", sync_ms, sync_ms / iterations);
    printf("  Async     │     %7.1f       │    %5.2f ms\n", async_ms, async_ms / iterations);
    printf("────────────┼───────────────────┼────────────────\n");
    printf("  Speedup   │     %5.2fx        │\n", sync_ms / async_ms);
    
    printf("\nNote: Speedup depends on CPU/GPU work balance.\n");
    printf("      Best when CPU prep time ≈ GPU compute time.\n");
    
    ab_matrix_destroy(A);
    ab_matrix_destroy(B);
    ab_matrix_destroy(C);
    free(data);
    ab_shutdown();
    return 0;
}
