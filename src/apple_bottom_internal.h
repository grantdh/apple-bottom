// =============================================================================
// apple_bottom_internal.h — apple_bottom.m's interface to the library's other
// Objective-C++ files
// Copyright (c) 2026 Grant Heileman, Technology Residue. MIT License.
// =============================================================================
//
// C++ linkage, as apple_bottom.m defines them; it includes this header, so the
// declarations are checked against the definitions.
// =============================================================================
#ifndef AB_APPLE_BOTTOM_INTERNAL_H
#define AB_APPLE_BOTTOM_INTERNAL_H

#import <Metal/Metal.h>
#include <stdint.h>
#include "apple_bottom.h"

// The library's command queue; nil without a context.
id<MTLCommandQueue> ab_internal_command_queue(void);

// The DD buffer of m.
id<MTLBuffer> ab_internal_matrix_buffer(ABMatrix m);

// C = A * B for the row-major M x K, K x N and M x N matrices at the start
// of the buffers, encoded as the classical ZGEMM encodes its products.
void ab_internal_encode_dgemm(id<MTLComputeCommandEncoder> encoder, id<MTLBuffer> A,
                              id<MTLBuffer> B, id<MTLBuffer> C, int M, int N, int K);

// Runs the work encode puts in a serial compute encoder in a command buffer
// of its own and waits for it, as ab_dgemm_scaled runs its kernel: the call
// counts as a DGEMM (ZGEMM if complex_gemm) in ab_get_stats(), its GPU time
// and host-side split, with temps_ns of host time the caller spent providing
// temporaries, go to the BLAS-ABI profile, and AB_TEST_FAIL_STAGE=kernel
// fails it. AB_ERROR_KERNEL_FAILED if the command buffer failed.
ABStatus ab_internal_run_gemm(bool complex_gemm, uint64_t temps_ns,
                              void (^encode)(id<MTLComputeCommandEncoder> encoder));

#endif // AB_APPLE_BOTTOM_INTERNAL_H
