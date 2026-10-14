// =============================================================================
// blas_profiler.h — BLAS call profiler for apple-bottom
// =============================================================================
//
// Per-routine accounting of BLAS-ABI calls (dgemm_, zgemm_, ab_*gemm_,
// ab_*gemm_blas): calls and wall time by where C was computed (GPU or
// cblas), the stage costs of the GPU calls, fallbacks to cblas by reason,
// and dimension histograms. Zero-overhead when disabled (a cached env read
// and a branch).
//
// Usage:
//   AB_REPORT=1                             concise summary on stderr at exit
//   AB_PROFILE=1                            detailed table on stderr at exit,
//                                           plus a line per GPU call / fallback
//   AB_PROFILE_JSON=/path/to/profile.json   with AB_PROFILE: also dump JSON
//   AB_PROFILE_THRESHOLD_NS=1000            count only calls above threshold
//
// Both reports start with ab_build_info(). "gpu_calls%" is a share of calls,
// "gpu_time%" a share of wall time; a call counts as GPU only if the GPU
// computed C. A fork()ed child starts with empty counters.
// =============================================================================
#ifndef AB_BLAS_PROFILER_H
#define AB_BLAS_PROFILER_H

#include <stdint.h>
#include <stddef.h>
#include <stdio.h>

#define AB_PROF_MAX_ROUTINES     256
#define AB_PROF_NAME_LEN          32
#define AB_PROF_HISTOGRAM_BINS     8

// Where the wall time of a GPU call went, in call order.
typedef enum {
    AB_PROF_STAGE_INIT,      // routing, size checks, Metal context creation on first use
    AB_PROF_STAGE_ALLOC,     // ABMatrix and host scratch allocation and release
    AB_PROF_STAGE_PACK,      // column-major op(A), op(B), C -> row-major host copies
    AB_PROF_STAGE_UPLOAD,    // FP64 -> DD conversion into the GPU buffers
    AB_PROF_STAGE_KERNEL,    // command-buffer GPU time (wall time if Metal reports none)
    AB_PROF_STAGE_DOWNLOAD,  // DD -> FP64 conversion out of the GPU buffers
    AB_PROF_STAGE_EPILOGUE,  // row-major result -> column-major C (ZGEMM: with alpha, beta)
    AB_PROF_STAGE_COUNT
} ab_prof_stage_t;

// Why a call the routing sent to the GPU ran on cblas.
typedef enum {
    AB_PROF_FB_NONE = -1,
    AB_PROF_FB_NO_DEVICE,    // no Metal context: no device, or a fork()ed child
    AB_PROF_FB_TOO_LARGE,    // an operand beyond the ABMatrix size limits
    AB_PROF_FB_GPU_FAILED,   // an allocation, upload or kernel failed before C was written
    AB_PROF_FB_OTHER,
    AB_PROF_FB_COUNT
} ab_prof_fallback_t;

// Dimension bins (by max(M,N)):
//   [0] 1-32     [1] 33-128   [2] 129-512   [3] 513-1024
//   [4] 1025-2048 [5] 2049-4096 [6] 4097-8192 [7] 8193+
typedef struct {
    char     name[AB_PROF_NAME_LEN];
    uint64_t call_count;
    uint64_t total_ns;
    uint64_t gpu_calls, cpu_calls;          // where C was computed
    uint64_t gpu_ns, cpu_ns;                // wall time of those calls
    uint64_t stage_ns[AB_PROF_STAGE_COUNT]; // GPU calls only; the rest of gpu_ns is "other"
    uint64_t fallbacks[AB_PROF_FB_COUNT];   // counted in cpu_calls

    // Dimension histogram — by max(M,N)
    uint64_t dim_histogram[AB_PROF_HISTOGRAM_BINS];
    uint64_t ns_by_bin[AB_PROF_HISTOGRAM_BINS];

    // Aggregate dimension stats for threshold tuning
    size_t   max_m, max_n, max_k;
    uint64_t sum_m, sum_n, sum_k;  // for computing averages
} ab_prof_entry_t;

// Record a BLAS call. Called from the BLAS-ABI layer after timing.
// used_gpu: the GPU computed C, not just the routing chose it; a GPU-routed
// call that ran on cblas is a CPU call with fallback set. stage_ns: the
// call's AB_PROF_STAGE_COUNT stage times (GPU calls; NULL otherwise).
// No-op unless AB_PROFILE or AB_REPORT is set.
void ab_prof_record(const char* routine, uint64_t elapsed_ns,
                    size_t m, size_t n, size_t k, int used_gpu,
                    const uint64_t* stage_ns, ab_prof_fallback_t fallback);

// Dump the AB_PROFILE table to an open stream.
void ab_prof_dump(FILE* out);

// Dump the AB_REPORT summary to an open stream.
void ab_prof_report(FILE* out);

// Dump JSON profile to a file path. Used at exit when AB_PROFILE_JSON set.
void ab_prof_dump_json(const char* path);

// Whether calls are recorded: AB_PROFILE or AB_REPORT (env read once, cached).
int  ab_prof_enabled(void);

// Whether AB_PROFILE asks for per-call lines and the detailed table.
int  ab_prof_verbose(void);

// Stage names, for per-call lines.
extern const char* const ab_prof_stage_names[AB_PROF_STAGE_COUNT];

// Convert mach_absolute_time() ticks to nanoseconds using timebase.
uint64_t ab_prof_ticks_to_ns(uint64_t ticks);

// Get current time in mach_absolute_time ticks.
uint64_t ab_prof_now_ticks(void);

#endif // AB_BLAS_PROFILER_H
