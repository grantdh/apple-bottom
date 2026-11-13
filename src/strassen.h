// =============================================================================
// strassen.h — the opt-in Strassen-Winograd product (strassen.m), internal
// Copyright (c) 2026 Grant Heileman, Technology Residue. MIT License.
// =============================================================================
//
// One level of Winograd's variant of Strassen's algorithm takes the place of
// one real DD product, P = X*Y or alpha*X*Y + beta*P, where apple_bottom.m
// forms one: ab_dgemm and ab_dgemm_scaled, each product of a ZGEMM's
// Gauss-3M or 4M, and the BLAS ABI's staged path. The callers keep the
// operands' power-of-two scales, the choice between 3M and 4M and their
// epilogues, so only the product changes. Its workspace is seven DD
// matrices from the staging pool (gpu_stage.h).
// =============================================================================
#ifndef AB_STRASSEN_H
#define AB_STRASSEN_H

#include <stdbool.h>

#include "apple_bottom.h"

#ifdef __cplusplus
extern "C" {
#endif

// The DD matrices of one product's workspace.
#define AB_STRASSEN_WORKSPACE 7

// True if AB_STRASSEN is set (and not 0) and M, N and K are all at least
// AB_STRASSEN_MIN (default 4096); both are read once per process.
bool ab_internal_strassen_wanted(int M, int N, int K);

#ifdef __cplusplus
}
#endif

#ifdef __OBJC__
#import <Metal/Metal.h>

// The workspace's shapes for an M x K by K x N product. False if Metal
// refuses dev the pipeline of the block additions, which this builds on
// first use.
bool ab_internal_strassen_prepare(id<MTLDevice> dev, int M, int N, int K,
                                  int rows[AB_STRASSEN_WORKSPACE],
                                  int cols[AB_STRASSEN_WORKSPACE]);

// Encodes P = alpha*X*Y + beta*P (P not read where beta is 0; alpha = 1 and
// beta = 0 skip the epilogue, as ab_dgemm does) into cmdBuf on the workspace
// W (ab_internal_strassen_prepare, which must have succeeded), in one serial
// compute encoder of its own: the steps before it and after it see P whole.
// X and Y must not share P's buffer. ctx is the caller's library context,
// passed back to ab_internal_encode_product.
void ab_internal_encode_strassen(id ctx, id<MTLCommandBuffer> cmdBuf, double alpha, ABMatrix X,
                                 ABMatrix Y, double beta, ABMatrix P,
                                 const ABMatrix W[AB_STRASSEN_WORKSPACE]);

// apple_bottom.m: P = X*Y encoded into enc with the context ctx, as the GEMMs
// encode their products, and m's DD buffer (row-major from byte 0).
void ab_internal_encode_product(id ctx, id<MTLComputeCommandEncoder> enc, ABMatrix X, ABMatrix Y,
                                ABMatrix P);
id<MTLBuffer> ab_internal_matrix_buffer(ABMatrix m);
#endif

#endif  // AB_STRASSEN_H
