// AB_SPLIT's per-call timeline (ab_split_times_t), shared by the executor
// (split_slice.h, blas_wrapper.c) and the profile (blas_profiler.h), and the
// record of a GPU call's kernel step it holds (ab_prof_kernel_split_t, which
// apple_bottom.m keeps per thread). Plain structs, valid in C and C++.
#ifndef AB_SPLIT_TIMES_H
#define AB_SPLIT_TIMES_H

#include <stdint.h>

// The host time of a GPU call's kernel step besides GPU execution, which the
// stages leave in "other", and when the GPU ran; ns on the mach_absolute_time
// clock (CLOCK_UPTIME_RAW). apple_bottom.m adds up the command buffers each
// thread completes; the BLAS-ABI layer takes the record once per call, with
// their GPU time.
typedef struct {
    uint64_t temps_ns;      // taking the call's GPU temporaries from the pool and back
                            // (ZGEMM on the host path; the staged path's are in alloc)
    uint64_t encode_ns;     // building the command buffer and its encoders
    uint64_t submit_ns;     // from commit until the GPU started (GPUStartTime)
    uint64_t wake_ns;       // from GPUEndTime until waitUntilCompleted returned
    uint64_t gpu_start_ns;  // GPUStartTime of the first command buffer; 0 if none reported
    uint64_t gpu_end_ns;    // GPUEndTime of the last
    // For AB_SPLIT's timeline below; the trace prints them on split lines
    // only. Those of the command buffers count the GEMMs' (commit_and_wait),
    // those of the dispatches every GEMM-family encode on the thread.
    uint64_t commit_ns;     // the first commit; 0 if none
    uint64_t done_ns;       // when the last waitUntilCompleted returned
    uint64_t groups;        // threadgroups of the GEMM-family dispatches, summed
    uint32_t groups_max;    // ... the largest grid among them
    uint32_t dispatches;    // GEMM-family dispatches
    uint32_t cmdbufs;       // command buffers completed
    uint32_t kinds;         // bit 2 * kind + prefetch of each GEMM-family kernel dispatched
    int32_t mtl_error;      // the first failed one's MTLCommandBufferError code, -1 if Metal
                            // gave none, -2 if a test hook failed it; 0 none failed
} ab_prof_kernel_split_t;

// When a split attempt's steps happened, in ns on the mach_absolute_time clock
// (CLOCK_UPTIME_RAW, the time base of GPUStartTime and GPUEndTime and of
// AB_TRACE's t0_ns); 0 for a step that did not happen.
typedef struct {
    uint64_t cut_ns;            // the attempt began: the cut decided (dgemm_/zgemm_) or the seam called
    uint64_t create_ns;         // just before the GPU part's thread was created
    uint64_t cpu0_ns, cpu1_ns;  // the CPU part's start and end, on the caller
    uint64_t gpu0_ns, gpu1_ns;  // the GPU part's start and end, on its thread
    uint64_t join_ns;           // the join returned
    uint64_t end_ns;            // the call's result complete (after a redo or the K axis's sum)
    ab_prof_kernel_split_t gpu; // the command buffers the GPU part's thread completed (engine 0)
} ab_split_times_t;

#endif  // AB_SPLIT_TIMES_H
