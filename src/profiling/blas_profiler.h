// =============================================================================
// blas_profiler.h — BLAS call profiler for apple-bottom
// =============================================================================
//
// Per-routine accounting of BLAS-ABI calls (dgemm_, zgemm_, ab_*gemm_,
// ab_*gemm_blas, and the BLAS-2 dgemv_, zgemv_, zhemv_, ab_*gemv_blas,
// ab_zhemv_blas): calls and wall time by where the result was computed (GPU
// or cblas), the stage costs of the GPU calls, fallbacks to cblas by reason,
// and dimension histograms. Zero-overhead when disabled (a cached env read
// and a branch). AB_TRACE covers GEMM only.
//
// Usage:
//   AB_REPORT=1                             concise summary on stderr at exit
//   AB_PROFILE=1                            detailed table on stderr at exit,
//                                           plus a line per GPU call / fallback
//                                           (also per GEMM step of ab_dtrsm and
//                                           ab_dgesv_mpir that falls back)
//   AB_PROFILE_JSON=/path/to/profile.json   with AB_PROFILE: also dump JSON
//   AB_PROFILE_THRESHOLD_NS=1000            count only calls above threshold
//   AB_TRACE=/path/to/trace                 a line per call in /path/to/trace.<pid>
//   AB_TRACE_HASH=1                         with AB_TRACE: hash the operands read
//   AB_TEST_FAIL_STAGE=alloc|kernel         test-only: fail GPU allocations or command buffers
//                                           (and the other failures below)
//   AB_TEST_FAIL_ONLY=n                     test-only: with kernel, only the n-th fails
//   AB_HOST_PACK=1                          GPU calls take the host path (gpu_stage.h)
//   AB_STAGING_MB=n                         the GPU buffer pool keeps at most n MB
//
// Both reports start with ab_build_info(). "gpu_calls%" is a share of calls,
// "gpu_time%" a share of wall time; a call counts as GPU only if the GPU
// computed C. A fork()ed child starts with empty counters and its own trace
// file. The trace format is in ab_trace_record() and docs/INTEGRATION.md.
// =============================================================================
#ifndef AB_BLAS_PROFILER_H
#define AB_BLAS_PROFILER_H

#include <stdbool.h>
#include <stdint.h>
#include <stddef.h>
#include <stdio.h>

#include "../split_times.h"   // ab_split_times_t

#ifdef __cplusplus
extern "C" {
#endif

#define AB_PROF_MAX_ROUTINES     256
#define AB_PROF_NAME_LEN          32
#define AB_PROF_HISTOGRAM_BINS     8

// Where the wall time of a GPU call went, in call order. The host path
// (host pack, conversion and epilogue), the staged path (gpu_stage.h) and
// BLAS-2 (blas2.m) fill them differently.
typedef enum {
    AB_PROF_STAGE_INIT,      // routing, size checks, Metal context creation on first use
    AB_PROF_STAGE_ALLOC,     // ABMatrix and host scratch allocation and release; staged:
                             // the pool's buffers for the call; BLAS-2: its GPU scratch
    AB_PROF_STAGE_PACK,      // column-major op(A), op(B), C -> row-major host copies; range check;
                             // staged: the range check; BLAS-2: staging x
    AB_PROF_STAGE_UPLOAD,    // FP64 -> DD conversion into the GPU buffers; staged: the
                             // copies of the operands into them; BLAS-2: mapping A for the GPU
    AB_PROF_STAGE_KERNEL,    // command-buffer GPU time (wall time if Metal reports none);
                             // staged: with the GPU's conversion and epilogue
    AB_PROF_STAGE_DOWNLOAD,  // DD -> FP64 conversion out of the GPU buffers; staged: the
                             // copy of the result into C; BLAS-2: the range guard's check
    AB_PROF_STAGE_EPILOGUE,  // row-major result -> column-major C (ZGEMM: with alpha, beta);
                             // staged: ZGEMM's host alpha and beta, when not 1 and 0 or 1;
                             // BLAS-2: y with alpha and beta
    AB_PROF_STAGE_COUNT
} ab_prof_stage_t;

// Why a call the routing sent to the GPU ran on cblas.
typedef enum {
    AB_PROF_FB_NONE = -1,
    AB_PROF_FB_NO_DEVICE,    // no Metal context: no device, or a fork()ed child
    AB_PROF_FB_TOO_LARGE,    // an operand beyond the ABMatrix size limits
    AB_PROF_FB_GPU_FAILED,   // an allocation, upload or kernel failed before C was written
    AB_PROF_FB_OTHER,        // anything else, e.g. BLAS-2's A not mappable for the GPU
    AB_PROF_FB_RANGE,        // values beyond double-float's exponent range, or NaN/Inf
                             // (BLAS-2: its range guard)
    AB_PROF_FB_COUNT
} ab_prof_fallback_t;

// Whether a GPU-routed call kept the staged path's result (gpu_stage.h), and
// if not, why not, in the order the staged path checks: the host path then
// computes the call, except for RANGE, which Accelerate computes.
typedef enum {
    AB_PROF_STAGED_NONE = -1,   // the call never reached the staged path
    AB_PROF_STAGED_KEPT,        // the staged path computed C
    AB_PROF_STAGED_KEPT_4M,     // ... a ZGEMM by 4M, after its Gauss-3M run
    AB_PROF_STAGED_HOST_PACK,   // AB_HOST_PACK=1
    AB_PROF_STAGED_FPCR,        // the caller's floating-point control is not the default
    AB_PROF_STAGED_SCALAR,      // a DGEMM alpha or beta outside 2^±AB_SCALAR_WINDOW, or not finite
    AB_PROF_STAGED_ALLOC,       // no GPU buffers for it
    AB_PROF_STAGED_GPU_FAILED,  // its command buffer failed, or its line scales were not the host's
    AB_PROF_STAGED_NONFINITE,   // Inf or NaN in the GPU epilogue, or a ZGEMM C there of 2^1022 or
                                // more beside beta = 1 (the host path then refuses it: range)
    AB_PROF_STAGED_RANGE,       // the exponent-range guard does not certify it
    AB_PROF_STAGED_COUNT
} ab_prof_staged_t;

// Why an AB_SPLIT call a row covered was not split (it then takes its own
// route, as with AB_SPLIT unset), in the order the executor checks.
typedef enum {
    AB_PROF_SPLIT_DECLINE_NONE = -1,
    AB_PROF_SPLIT_DECLINE_NO_DEVICE,   // no Metal context: no device, or a fork()ed child
    AB_PROF_SPLIT_DECLINE_TOO_LARGE,   // an operand beyond the ABMatrix size limits
    AB_PROF_SPLIT_DECLINE_HOST_PACK,   // AB_HOST_PACK=1
    AB_PROF_SPLIT_DECLINE_FPCR,        // the caller's floating-point control is not the default
    AB_PROF_SPLIT_DECLINE_SCALAR,      // a DGEMM alpha or beta the staged path does not take
    AB_PROF_SPLIT_DECLINE_SCRATCH,     // no memory for the K axis's partial product
    AB_PROF_SPLIT_DECLINE_THREAD,      // no thread for the GPU's part
    AB_PROF_SPLIT_DECLINE_COUNT
} ab_prof_split_decline_t;

// AB_SPLIT's counts and times for one routine (AB_REPORT's "<routine> split"
// line). calls = kept + kept_4m + range + gpu_failed + the declined.
typedef struct {
    uint64_t calls;                     // calls a row cut: split, or declined
    uint64_t kept, kept_4m;             // the GPU's part stood (by 4M)
    uint64_t range, gpu_failed;         // it was made again on Accelerate: the guard; a failure
    uint64_t scales, alloc, nonfinite, fpcr;   // among gpu_failed: line scales unlike the host's,
                                               // no buffers, Inf/NaN, the thread's FP control
    uint64_t retries, retry_kept;       // staged runs repeated after a failed command buffer
    uint64_t declined[AB_PROF_SPLIT_DECLINE_COUNT];
    uint64_t axis[3];                   // calls split on N, M, K
    uint64_t init_ns, cpu_ns, gpu_ns;   // the caller until the tail's thread ran; the head; the
                                        // tail's thread
    uint64_t wait_cpu_ns, wait_gpu_ns;  // the head done, waiting for the tail; and the reverse
    uint64_t redo_ns, sum_ns;           // tails made again on Accelerate; the K axis's sums
    uint64_t wasted_gpu_ns;             // the threads of tails that did not stand
    double gf_cpu, gf_gpu;              // GFLOP of the heads and of the tails
} ab_prof_split_stats_t;

// One BLAS-ABI call a row of the split table covered, as the executor
// (blas_wrapper.c) records it for the reports.
typedef struct {
    int row;                            // the split table's row; -1 if none covered the call
    char axis;
    int L, g;                           // g = 0: the row did not cut the call
    bool split;                         // the executor ran both parts
    bool kept;                          // ... and the GPU's stood: the call counts as split
    ab_prof_split_decline_t declined;   // why a cut call was not split
    ab_prof_staged_t staged;            // the GPU part's staged-path outcome
    bool scales;                        // ... line scales unlike the host's
    int retries;
    bool retry_kept;
    // The caller's stages; with the call's other_ns they sum to its wall time.
    // join_ns is the wait for the GPU part and the reading of how it ended
    // (the scan of a part stopped on NaN or Inf); redo_ns only Accelerate's
    // remaking of a part that did not stand.
    uint64_t init_ns, cpu_part_ns, join_ns, redo_ns, sum_ns;
    // The GPU part's thread: its stages, and with gpu_other its wall time.
    uint64_t gpu_stage_ns[AB_PROF_STAGE_COUNT], gpu_wall_ns;
    uint64_t wait_cpu_ns, wait_gpu_ns;
    double gf_cpu, gf_gpu;
    ab_split_times_t times;             // when each step happened (AB_TRACE's split lines)
} ab_prof_split_t;

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
    uint64_t staged[AB_PROF_STAGED_COUNT];  // calls that reached the staged path, by outcome
    uint64_t split_calls, split_ns;         // AB_SPLIT: calls whose GPU part stood, and their
                                            // wall time (calls = gpu + cpu + split)
    ab_prof_split_stats_t split;            // AB_SPLIT: the executor's counts and times

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
// staged: the call's staged-path outcome. No-op unless AB_PROFILE or
// AB_REPORT is set.
void ab_prof_record(const char* routine, uint64_t elapsed_ns,
                    size_t m, size_t n, size_t k, int used_gpu,
                    const uint64_t* stage_ns, ab_prof_fallback_t fallback,
                    ab_prof_staged_t staged);

// Record a call a row of the AB_SPLIT table covered (s->row >= 0): its row's
// and its routine's split counts and times, and, when its GPU part stood
// (s->kept), the call itself as a split call (ab_prof_record is not called
// for it). No-op unless AB_PROFILE or AB_REPORT is set.
void ab_prof_record_split(const char* routine, uint64_t elapsed_ns, size_t m, size_t n, size_t k,
                          const ab_prof_split_t* s);

// Under AB_PROFILE, on stderr: the line of a call the executor split, with
// its caller's stages, its GPU part's thread and outcome.
void ab_prof_print_split_call(const char* routine, int m, int n, int k, uint64_t total_ns,
                              const ab_prof_split_t* s);

extern const char* const ab_prof_split_decline_names[AB_PROF_SPLIT_DECLINE_COUNT];

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

// Stage names and staged-path outcome names, for per-call lines.
extern const char* const ab_prof_stage_names[AB_PROF_STAGE_COUNT];
extern const char* const ab_prof_staged_names[AB_PROF_STAGED_COUNT];

// Adds the time since *mark (ticks) to stage_ns[s] and restarts *mark from
// now; nothing if stage_ns is NULL (the call is not timed).
void ab_prof_stage_mark(uint64_t* stage_ns, uint64_t* mark, ab_prof_stage_t s);

// Under AB_PROFILE, on stderr in one write each: the line of a GPU call that
// took total_ns with its stage times (and "other", the rest) and its staged
// outcome (none printed for AB_PROF_STAGED_NONE), and the line of a GPU-routed
// call that ran on cblas instead, with why.
void ab_prof_print_gpu_call(const char* routine, int m, int n, int k, uint64_t total_ns,
                            const uint64_t* stage_ns, ab_prof_staged_t staged);
void ab_prof_print_fallback(const char* routine, int m, int n, int k, const char* why);

// Convert mach_absolute_time() ticks to nanoseconds using timebase.
uint64_t ab_prof_ticks_to_ns(uint64_t ticks);

// The time from tick t0 to tick t1 in ns: the difference of the two
// converted, so that t0's conversion plus it is t1's, as on a trace line.
// Converting t1 - t0 instead can come out 1 ns less (Apple silicon's
// timebase is 125/3).
uint64_t ab_prof_span_ns(uint64_t t0, uint64_t t1);

// Get current time in mach_absolute_time ticks.
uint64_t ab_prof_now_ticks(void);

// Whether calls are timed at all: AB_PROFILE, AB_REPORT or AB_TRACE (env
// read once, cached).
int  ab_prof_active(void);

// =============================================================================
// Per-call trace (AB_TRACE)
// =============================================================================

// ab_prof_kernel_split_t, the host time of a GPU call's kernel step and its
// command buffers, is in split_times.h.

// GPU execution time of the command buffers this thread completed since the
// last take, in seconds; 0 if Metal reported none. split, unless NULL, gets
// the rest of their host time and their GPU start and end. Both restart from
// zero (apple_bottom.m).
double ab_internal_take_gpu_seconds(ab_prof_kernel_split_t* split);

// The kernel stage of a GPU call whose kernel step (ab_dgemm_scaled(),
// ab_zgemm() or a staged run) took wall_ns: the GPU time of the command
// buffers taken now, with the rest of their host time added to split
// (which keeps its first GPU start and gets the last end, so a call with two
// kernel steps adds up); or wall_ns if Metal reported none or not less than
// wall_ns, and then nothing of the step is in "other" and split's host parts
// get nothing. The BLAS-ABI layer and benchmarks/replay.c both stage a
// kernel call with it.
uint64_t ab_prof_take_kernel_ns(uint64_t wall_ns, ab_prof_kernel_split_t* split);

// The same rule for a step whose command buffers ran gpu_seconds on the GPU
// (0 if Metal reported none): that time, or wall_ns.
uint64_t ab_prof_kernel_ns(uint64_t wall_ns, double gpu_seconds);

// One BLAS-ABI call: its arguments as the entry point received them, and
// what the trace adds to the profile's record of it.
typedef struct {
    const char* routine;                 // "dgemm" or "zgemm"
    char ta, tb;                         // TRANS flags as passed
    int m, n, k, lda, ldb, ldc;
    double alpha[2], beta[2];            // real and imaginary parts (zgemm)
    ab_prof_kernel_split_t split;        // GPU calls: the kernel step's part of "other"
    bool hashed[3];                      // AB_TRACE_HASH: hash[i] is set
    uint64_t hash[3];                    // hashes of op(A), op(B) and C on entry, as stored
} ab_trace_call_t;

// Whether AB_TRACE names a trace file, and AB_TRACE_HASH=1 asks for hashes.
int  ab_trace_enabled(void);
int  ab_trace_hash_enabled(void);

// Append the call's line to this process's trace file, <AB_TRACE>.<pid>,
// which the first line of each process creates with a '#' header. The call
// started at tick t0 and ended at tick t1 (mach_absolute_time); the line's
// t0_ns and t1_ns are each converted from its tick, as every host clock read
// on it is (the at_* steps), so that no step of the call reads before t0_ns
// or after t1_ns (ab_prof_span_ns).
// The other arguments are those of ab_prof_record (stage_ns non-NULL: the GPU
// computed C).
void ab_trace_record(const ab_trace_call_t* call, uint64_t t0, uint64_t t1,
                     const uint64_t* stage_ns, ab_prof_fallback_t fallback,
                     ab_prof_staged_t staged);

// The same for a call a row of the AB_SPLIT table covered: a split call's
// line (path=split, or fallback:<reason> for a GPU part made again on
// Accelerate) with s's stages, or, for one the row did not split, the line
// ab_trace_record writes with the row added.
void ab_trace_record_split(const ab_trace_call_t* call, uint64_t t0, uint64_t t1,
                           const uint64_t* stage_ns, ab_prof_fallback_t fallback,
                           ab_prof_staged_t staged, const ab_prof_split_t* s);

// AB_TEST_FAIL_STAGE (test-only): AB_PROF_STAGE_ALLOC makes every
// ab_matrix_create() fail, AB_PROF_STAGE_KERNEL command buffers
// (ab_test_fail_cmdbuf), as real failures there would; -1 when unset. The
// values below go past the stages.
int  ab_test_fail_stage(void);

// Whether AB_TEST_FAIL_STAGE=kernel fails the command buffer that has just
// completed: every one or, with AB_TEST_FAIL_ONLY=n, the process's n-th. Call
// it once per command buffer.
int  ab_test_fail_cmdbuf(void);

enum {
    AB_PROF_FAIL_KERNEL_ONCE = AB_PROF_STAGE_COUNT,  // kernel_once: the process's first GEMM command
                                                     // buffer (commit_and_wait) fails, no other
    AB_PROF_FAIL_REDO_4M,        // redo_4m: every staged ZGEMM's 4M run fails
    AB_PROF_FAIL_SCALES,         // scales: the staged path's line scales read as unlike the host's
    AB_PROF_FAIL_SPLIT_THREAD,   // split_thread: AB_SPLIT cannot make its thread
    AB_PROF_FAIL_SPLIT_SCRATCH,  // split_scratch: AB_SPLIT cannot allocate the K axis's product
    AB_PROF_FAIL_SPLIT_FPCR,     // split_fpcr: AB_SPLIT's thread runs with flush to zero
    AB_PROF_FAIL_SPLIT_POISON,   // split_poison: AB_SPLIT fills the K axis's product with 1.0 once made
    AB_PROF_FAIL_NONFINITE,      // nonfinite: every staged run ends as if its epilogue met a NaN
};

// AB_HOST_PACK=1 (as the other flags: set, not empty, not starting with 0):
// every GPU-routed BLAS-ABI call takes the host path. AB_STAGING_MB=n: the
// bytes the pool of GPU buffers (gpu_stage.h) keeps between calls, n MB
// (default 2048; 0 or less: none). Read with the other settings, once.
int  ab_host_pack(void);
size_t ab_staging_cap_bytes(void);

#ifdef __cplusplus
}
#endif

#endif // AB_BLAS_PROFILER_H
