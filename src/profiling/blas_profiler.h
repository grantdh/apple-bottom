// =============================================================================
// blas_profiler.h — BLAS call profiler for apple-bottom
// =============================================================================
//
// Per-routine call profiler with dimension histograms and GPU/AMX dispatch
// accounting. Zero-overhead when disabled (single atomic load + branch).
//
// Usage:
//   AB_PROFILE=1                            enable profiling
//   AB_PROFILE_JSON=/path/to/profile.json   dump JSON on exit
//   AB_PROFILE_THRESHOLD_NS=1000            only record calls above threshold
//
// Output (stderr, on exit):
//   === apple-bottom BLAS profile ===
//   routine        calls   total_ms   gpu_pct   max_dim   hotbin
//   --------------------------------------------------------------
//   dgemm          931     1250.4       42.1%     18277   8K+
//   zgemm          427      843.2       68.3%     18277   8K+
//   ...
// =============================================================================
#ifndef AB_BLAS_PROFILER_H
#define AB_BLAS_PROFILER_H

#include <stdint.h>
#include <stddef.h>
#include <stdio.h>

#define AB_PROF_MAX_ROUTINES     256
#define AB_PROF_NAME_LEN          32
#define AB_PROF_HISTOGRAM_BINS     8

// Dimension bins (by max(M,N)):
//   [0] 1-32     [1] 33-128   [2] 129-512   [3] 513-1024
//   [4] 1025-2048 [5] 2049-4096 [6] 4097-8192 [7] 8193+
typedef struct {
    char     name[AB_PROF_NAME_LEN];
    uint64_t call_count;
    uint64_t total_ns;
    uint64_t gpu_dispatch_count;
    uint64_t amx_fallback_count;

    // Dimension histogram — by max(M,N)
    uint64_t dim_histogram[AB_PROF_HISTOGRAM_BINS];
    uint64_t ns_by_bin[AB_PROF_HISTOGRAM_BINS];

    // Aggregate dimension stats for threshold tuning
    size_t   max_m, max_n, max_k;
    uint64_t sum_m, sum_n, sum_k;  // for computing averages
} ab_prof_entry_t;

// Record a BLAS call. Called from dispatch layer after timing.
// No-op if AB_PROFILE env var not set.
void ab_prof_record(const char* routine, uint64_t elapsed_ns,
                    size_t m, size_t n, size_t k, int used_gpu);

// Dump human-readable profile to an open stream.
void ab_prof_dump(FILE* out);

// Dump JSON profile to a file path. Used by atexit when AB_PROFILE_JSON set.
void ab_prof_dump_json(const char* path);

// Check if profiling is enabled (reads env var once, caches result).
int  ab_prof_enabled(void);

// Convert mach_absolute_time() ticks to nanoseconds using timebase.
uint64_t ab_prof_ticks_to_ns(uint64_t ticks);

// Get current time in mach_absolute_time ticks.
uint64_t ab_prof_now_ticks(void);

#endif // AB_BLAS_PROFILER_H
