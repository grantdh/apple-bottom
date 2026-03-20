// apple-bottom — BLASphemy
// Copyright 2026 Technology Residue
// Author: Grant David Heileman, Ph.D.
//
// DGEMM.metal — Metal compute shader for double-precision general matrix multiply
// C = alpha * A * B + beta * C

#include <metal_stdlib>
using namespace metal;

kernel void dgemm(
    device const double* A       [[ buffer(0) ]],
    device const double* B       [[ buffer(1) ]],
    device double*       C       [[ buffer(2) ]],
    constant uint&       M       [[ buffer(3) ]],
    constant uint&       N       [[ buffer(4) ]],
    constant uint&       K       [[ buffer(5) ]],
    constant double&     alpha   [[ buffer(6) ]],
    constant double&     beta    [[ buffer(7) ]],
    uint2 gid [[ thread_position_in_grid ]]
) {
    if (gid.x >= N || gid.y >= M) return;

    double acc = 0.0;
    for (uint k = 0; k < K; k++) {
        acc += A[gid.y * K + k] * B[k * N + gid.x];
    }

    C[gid.y * N + gid.x] = alpha * acc + beta * C[gid.y * N + gid.x];
}
