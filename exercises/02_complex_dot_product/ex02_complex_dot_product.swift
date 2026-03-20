#!/usr/bin/env swift
// Exercise 2: Complex Dot Product — Parallel Reduction
// swiftc -O -framework Metal -framework Foundation -framework Accelerate ex02_complex_dot_product.swift -o ex02
// Grant Heileman — UNM ECE — 2026

import Foundation
import Metal
import Accelerate

let shaderSource = """
#include <metal_stdlib>
using namespace metal;

inline float2 cmul(float2 a, float2 b) {
    return float2(a.x * b.x - a.y * b.y, a.x * b.y + a.y * b.x);
}
inline float2 cconj(float2 z) { return float2(z.x, -z.y); }

#define REDUCE_TG_SIZE 256

kernel void zdotc_reduce(
    device const float2 *A [[buffer(0)]], device const float2 *B [[buffer(1)]],
    device float2 *partials [[buffer(2)]], constant uint &count [[buffer(3)]],
    uint gid [[thread_position_in_grid]], uint lid [[thread_position_in_threadgroup]],
    uint tgid [[threadgroup_position_in_grid]]
) {
    threadgroup float2 shared_data[REDUCE_TG_SIZE];
    shared_data[lid] = (gid < count) ? cmul(cconj(A[gid]), B[gid]) : float2(0.0);
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint s = REDUCE_TG_SIZE / 2; s > 0; s >>= 1) {
        if (lid < s) shared_data[lid] += shared_data[lid + s];
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    if (lid == 0) partials[tgid] = shared_data[0];
}

kernel void zaxpy(
    device const float2 *x [[buffer(0)]], device float2 *y [[buffer(1)]],
    constant uint &count [[buffer(2)]], constant float2 &alpha [[buffer(3)]],
    uint gid [[thread_position_in_grid]]
) {
    if (gid >= count) return;
    y[gid] += cmul(alpha, x[gid]);
}
"""

// MARK: - Setup

guard let device = MTLCreateSystemDefaultDevice() else { fatalError("No Metal device") }
assert(device.hasUnifiedMemory, "Requires Apple Silicon")

let opts = MTLCompileOptions(); opts.mathMode = .fast
let library = try device.makeLibrary(source: shaderSource, options: opts)
guard let queue = device.makeCommandQueue() else { fatalError("No command queue") }

let TG_SIZE = 256

func pipeline(_ name: String) -> MTLComputePipelineState {
    try! device.makeComputePipelineState(function: library.makeFunction(name: name)!)
}

print("Exercise 2: Complex Dot Product — GPU: \(device.name)\n")

// MARK: - Helper: run ZDOTC and return result

func runZDOTC(_ pip: MTLComputePipelineState, A: MTLBuffer, B: MTLBuffer, N: Int) -> SIMD2<Float> {
    let numTG = (N + TG_SIZE - 1) / TG_SIZE
    let partials = device.makeBuffer(length: numTG * MemoryLayout<SIMD2<Float>>.stride, options: .storageModeShared)!

    guard let cb = queue.makeCommandBuffer(), let enc = cb.makeComputeCommandEncoder() else { return .zero }
    cb.label = "zdotc_N\(N)"
    enc.setComputePipelineState(pip)
    enc.setBuffer(A, offset: 0, index: 0)
    enc.setBuffer(B, offset: 0, index: 1)
    enc.setBuffer(partials, offset: 0, index: 2)
    var count = UInt32(N)
    enc.setBytes(&count, length: 4, index: 3)
    enc.dispatchThreadgroups(MTLSize(width: numTG, height: 1, depth: 1),
                              threadsPerThreadgroup: MTLSize(width: TG_SIZE, height: 1, depth: 1))
    enc.endEncoding(); cb.commit(); cb.waitUntilCompleted()
    if cb.status == .error { fatalError("GPU: \(cb.error?.localizedDescription ?? "?")") }

    let ptr = partials.contents().bindMemory(to: SIMD2<Float>.self, capacity: numTG)
    var result = SIMD2<Float>.zero
    for i in 0..<numTG { result += ptr[i] }
    return result
}

// MARK: - Test 1: Small ZDOTC (N=8)

do {
    let pip = pipeline("zdotc_reduce")
    let N = 8
    let A = device.makeBuffer(length: N * 8, options: .storageModeShared)!
    let B = device.makeBuffer(length: N * 8, options: .storageModeShared)!

    let pA = A.contents().bindMemory(to: SIMD2<Float>.self, capacity: N)
    let pB = B.contents().bindMemory(to: SIMD2<Float>.self, capacity: N)
    let vals: [SIMD2<Float>] = [SIMD2(1,0), SIMD2(0,1), SIMD2(1,1), SIMD2(2,0),
                                 SIMD2(0,2), SIMD2(1,-1), SIMD2(3,0), SIMD2(0,0)]
    for i in 0..<N { pA[i] = vals[i]; pB[i] = SIMD2(1, 0) }

    // Σ conj(A[i]) * B[i] = Σ conj(A[i]) = (8, -3)
    let result = runZDOTC(pip, A: A, B: B, N: N)
    let err = max(abs(result.x - 8), abs(result.y + 3))
    print("  zdotc N=8:       err = \(err)  \(err < 1e-5 ? "✓" : "✗")")
}

// MARK: - Test 2: Large ZDOTC (N=65536)

do {
    let pip = pipeline("zdotc_reduce")
    let N = 65536
    let sz = N * MemoryLayout<SIMD2<Float>>.stride
    let A = device.makeBuffer(length: sz, options: .storageModeShared)!
    let B = device.makeBuffer(length: sz, options: .storageModeShared)!

    let pA = A.contents().bindMemory(to: SIMD2<Float>.self, capacity: N)
    let pB = B.contents().bindMemory(to: SIMD2<Float>.self, capacity: N)
    for i in 0..<N {
        pA[i] = SIMD2<Float>(Float(i), Float(i + 1))
        pB[i] = SIMD2<Float>(1.0, 0.5)
    }

    // CPU reference
    var ref = SIMD2<Float>.zero
    for i in 0..<N { let fi = Float(i); ref.x += 1.5 * fi + 0.5; ref.y += -0.5 * fi - 1.0 }

    let gpu = runZDOTC(pip, A: A, B: B, N: N)
    let relErr = max(abs(gpu.x - ref.x) / abs(ref.x), abs(gpu.y - ref.y) / abs(ref.y))
    print("  zdotc N=65536:   rel err = \(relErr)  \(relErr < 1e-4 ? "✓" : "✗")")
}

// MARK: - Test 3: ZAXPY

do {
    let pip = pipeline("zaxpy")
    let N = 1024, sz = N * MemoryLayout<SIMD2<Float>>.stride
    let X = device.makeBuffer(length: sz, options: .storageModeShared)!
    let Y = device.makeBuffer(length: sz, options: .storageModeShared)!

    let pX = X.contents().bindMemory(to: SIMD2<Float>.self, capacity: N)
    let pY = Y.contents().bindMemory(to: SIMD2<Float>.self, capacity: N)
    for i in 0..<N { pX[i] = SIMD2<Float>(Float(i), Float(i)); pY[i] = SIMD2<Float>(100, 0) }

    var alpha = SIMD2<Float>(0, 1), count = UInt32(N)
    guard let cb = queue.makeCommandBuffer(), let enc = cb.makeComputeCommandEncoder() else { fatalError() }
    cb.label = "zaxpy"
    enc.setComputePipelineState(pip)
    enc.setBuffer(X, offset: 0, index: 0); enc.setBuffer(Y, offset: 0, index: 1)
    enc.setBytes(&count, length: 4, index: 2); enc.setBytes(&alpha, length: 8, index: 3)
    enc.dispatchThreads(MTLSize(width: N, height: 1, depth: 1),
                        threadsPerThreadgroup: MTLSize(width: min(N, pip.maxTotalThreadsPerThreadgroup), height: 1, depth: 1))
    enc.endEncoding(); cb.commit(); cb.waitUntilCompleted()

    // y[i] = (100,0) + (0,1)*(i,i) = (100-i, i)
    var maxErr: Float = 0
    for i in 0..<N {
        let fi = Float(i)
        maxErr = max(maxErr, max(abs(pY[i].x - (100 - fi)), abs(pY[i].y - fi)))
    }
    print("  zaxpy:           max err = \(maxErr)  \(maxErr < 1e-4 ? "✓" : "✗")")
}

// MARK: - Test 4: Throughput (GPU timestamps)

do {
    let pip = pipeline("zdotc_reduce")
    let sizes = [1024, 16384, 65536, 262144, 1048576]

    print("\n  Throughput (GPU timestamps):")
    print("           N    GPU (ms)    GFLOP/s")

    for N in sizes {
        let sz = N * MemoryLayout<SIMD2<Float>>.stride
        let numTG = (N + TG_SIZE - 1) / TG_SIZE
        let A = device.makeBuffer(length: sz, options: .storageModeShared)!
        let B = device.makeBuffer(length: sz, options: .storageModeShared)!
        let P = device.makeBuffer(length: numTG * 8, options: .storageModeShared)!

        let pA = A.contents().bindMemory(to: SIMD2<Float>.self, capacity: N)
        let pB = B.contents().bindMemory(to: SIMD2<Float>.self, capacity: N)
        for i in 0..<N { pA[i] = SIMD2(1, 0); pB[i] = SIMD2(1, 0) }

        // Warmup
        for _ in 0..<3 { let _ = runZDOTC(pip, A: A, B: B, N: N) }

        let reps = 100
        var gpuTime: Double = 0
        for _ in 0..<reps {
            autoreleasepool {
                guard let cb = queue.makeCommandBuffer(), let enc = cb.makeComputeCommandEncoder() else { return }
                enc.setComputePipelineState(pip)
                enc.setBuffer(A, offset: 0, index: 0); enc.setBuffer(B, offset: 0, index: 1)
                enc.setBuffer(P, offset: 0, index: 2)
                var count = UInt32(N)
                enc.setBytes(&count, length: 4, index: 3)
                enc.dispatchThreadgroups(MTLSize(width: numTG, height: 1, depth: 1),
                                          threadsPerThreadgroup: MTLSize(width: TG_SIZE, height: 1, depth: 1))
                enc.endEncoding(); cb.commit(); cb.waitUntilCompleted()
                gpuTime += cb.gpuEndTime - cb.gpuStartTime
            }
        }
        let ms = gpuTime * 1000 / Double(reps)
        let gflops = Double(8 * N) / (ms * 1e6)
        print(String(format: "  %10d   %8.3f   %8.2f", N, ms, gflops))
    }
}
