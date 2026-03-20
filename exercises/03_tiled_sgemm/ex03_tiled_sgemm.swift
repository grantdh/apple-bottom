#!/usr/bin/env swift
// Exercise 3: Tiled SGEMM on Metal GPU
// swiftc -O -Xcc -DACCELERATE_NEW_LAPACK -framework Metal -framework Foundation -framework Accelerate ex03_tiled_sgemm.swift -o ex03
// Grant Heileman — UNM ECE — 2026

import Foundation
import Metal
import Accelerate

let shaderSource = """
#include <metal_stdlib>
using namespace metal;

#define TILE 16

kernel void sgemm_naive(
    device const float *A [[buffer(0)]], device const float *B [[buffer(1)]],
    device float *C [[buffer(2)]], constant uint &M [[buffer(3)]],
    constant uint &N [[buffer(4)]], constant uint &K_dim [[buffer(5)]],
    uint2 gid [[thread_position_in_grid]]
) {
    if (gid.y >= M || gid.x >= N) return;
    float sum = 0.0;
    for (uint k = 0; k < K_dim; k++) sum += A[gid.y * K_dim + k] * B[k * N + gid.x];
    C[gid.y * N + gid.x] = sum;
}

kernel void sgemm_tiled(
    device const float *A [[buffer(0)]], device const float *B [[buffer(1)]],
    device float *C [[buffer(2)]], constant uint &M [[buffer(3)]],
    constant uint &N [[buffer(4)]], constant uint &K_dim [[buffer(5)]],
    uint2 lid [[thread_position_in_threadgroup]], uint2 tgid [[threadgroup_position_in_grid]]
) {
    uint row = tgid.y * TILE + lid.y, col = tgid.x * TILE + lid.x;
    threadgroup float tA[TILE][TILE], tB[TILE][TILE];
    float sum = 0.0;

    for (uint t = 0; t < (K_dim + TILE - 1) / TILE; t++) {
        uint ac = t * TILE + lid.x, br = t * TILE + lid.y;
        tA[lid.y][lid.x] = (row < M && ac < K_dim) ? A[row * K_dim + ac] : 0.0;
        tB[lid.y][lid.x] = (br < K_dim && col < N) ? B[br * N + col]     : 0.0;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (uint k = 0; k < TILE; k++) sum += tA[lid.y][k] * tB[k][lid.x];
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    if (row < M && col < N) C[row * N + col] = sum;
}
"""

// MARK: - Setup

guard let device = MTLCreateSystemDefaultDevice() else { fatalError("No Metal device") }
assert(device.hasUnifiedMemory, "Requires Apple Silicon")

let opts = MTLCompileOptions(); opts.mathMode = .fast
let library = try device.makeLibrary(source: shaderSource, options: opts)
guard let queue = device.makeCommandQueue() else { fatalError("No command queue") }

let TILE = 16

func pipeline(_ name: String) -> MTLComputePipelineState {
    try! device.makeComputePipelineState(function: library.makeFunction(name: name)!)
}

// GPU dispatch helper for GEMM — returns GPU time in seconds
func dispatchGEMM(_ pip: MTLComputePipelineState, A: MTLBuffer, B: MTLBuffer, C: MTLBuffer,
                   M: Int, N: Int, K: Int, tiled: Bool) -> Double {
    guard let cb = queue.makeCommandBuffer(), let enc = cb.makeComputeCommandEncoder() else { return 0 }
    enc.setComputePipelineState(pip)
    enc.setBuffer(A, offset: 0, index: 0); enc.setBuffer(B, offset: 0, index: 1); enc.setBuffer(C, offset: 0, index: 2)
    var m = UInt32(M), n = UInt32(N), k = UInt32(K)
    enc.setBytes(&m, length: 4, index: 3); enc.setBytes(&n, length: 4, index: 4); enc.setBytes(&k, length: 4, index: 5)
    if tiled {
        enc.dispatchThreadgroups(MTLSize(width: (N+TILE-1)/TILE, height: (M+TILE-1)/TILE, depth: 1),
                                  threadsPerThreadgroup: MTLSize(width: TILE, height: TILE, depth: 1))
    } else {
        enc.dispatchThreads(MTLSize(width: N, height: M, depth: 1),
                            threadsPerThreadgroup: MTLSize(width: TILE, height: TILE, depth: 1))
    }
    enc.endEncoding(); cb.commit(); cb.waitUntilCompleted()
    if cb.status == .error { fatalError("GPU: \(cb.error?.localizedDescription ?? "?")") }
    return cb.gpuEndTime - cb.gpuStartTime
}

func maxErr(_ a: UnsafePointer<Float>, _ b: UnsafePointer<Float>, _ n: Int) -> Float {
    var mx: Float = 0; for i in 0..<n { mx = max(mx, abs(a[i] - b[i])) }; return mx
}
func frobNorm(_ a: UnsafePointer<Float>, _ n: Int) -> Float {
    var s: Float = 0; for i in 0..<n { s += a[i] * a[i] }; return sqrt(s)
}

let naivePip = pipeline("sgemm_naive")
let tiledPip = pipeline("sgemm_tiled")

print("Exercise 3: Tiled SGEMM — GPU: \(device.name)\n")

// MARK: - Test 1: Correctness (64×64)

do {
    let M = 64, N = 64, K = 64, count = M * N
    let szA = M*K*4, szB = K*N*4, szC = count*4
    let A = device.makeBuffer(length: szA, options: .storageModeShared)!
    let B = device.makeBuffer(length: szB, options: .storageModeShared)!
    let Cn = device.makeBuffer(length: szC, options: .storageModeShared)!
    let Ct = device.makeBuffer(length: szC, options: .storageModeShared)!

    let pA = A.contents().bindMemory(to: Float.self, capacity: M*K)
    let pB = B.contents().bindMemory(to: Float.self, capacity: K*N)
    srand48(42)
    for i in 0..<(M*K) { pA[i] = Float(drand48() - 0.5) }
    for i in 0..<(K*N) { pB[i] = Float(drand48() - 0.5) }

    var ref = [Float](repeating: 0, count: count)
    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                Int32(M), Int32(N), Int32(K), 1.0, pA, Int32(K), pB, Int32(N), 0.0, &ref, Int32(N))

    let _ = dispatchGEMM(naivePip, A: A, B: B, C: Cn, M: M, N: N, K: K, tiled: false)
    let _ = dispatchGEMM(tiledPip, A: A, B: B, C: Ct, M: M, N: N, K: K, tiled: true)

    let norm = frobNorm(&ref, count)
    let pN = Cn.contents().bindMemory(to: Float.self, capacity: count)
    let pT = Ct.contents().bindMemory(to: Float.self, capacity: count)
    print("  64×64 naive:     rel = \(maxErr(pN, &ref, count) / norm)  ✓")
    print("  64×64 tiled:     rel = \(maxErr(pT, &ref, count) / norm)  ✓")
}

// MARK: - Test 2: Non-aligned (288×100, K=137)

do {
    let M = 288, N = 100, K = 137, count = M * N
    let A = device.makeBuffer(length: M*K*4, options: .storageModeShared)!
    let B = device.makeBuffer(length: K*N*4, options: .storageModeShared)!
    let C = device.makeBuffer(length: count*4, options: .storageModeShared)!

    let pA = A.contents().bindMemory(to: Float.self, capacity: M*K)
    let pB = B.contents().bindMemory(to: Float.self, capacity: K*N)
    srand48(99)
    for i in 0..<(M*K) { pA[i] = Float(drand48() - 0.5) }
    for i in 0..<(K*N) { pB[i] = Float(drand48() - 0.5) }

    var ref = [Float](repeating: 0, count: count)
    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                Int32(M), Int32(N), Int32(K), 1.0, pA, Int32(K), pB, Int32(N), 0.0, &ref, Int32(N))
    let _ = dispatchGEMM(tiledPip, A: A, B: B, C: C, M: M, N: N, K: K, tiled: true)

    let pC = C.contents().bindMemory(to: Float.self, capacity: count)
    let rel = maxErr(pC, &ref, count) / frobNorm(&ref, count)
    print("  288×100 tiled:   rel = \(rel)  \(rel < 1e-5 ? "✓" : "✗")")
}

// MARK: - Test 3: Performance

do {
    print("\n  Performance (GPU timestamps):")
    print("     M=N=K    Naive ms    Tiled ms   Accel ms   Tiled GFLOP/s")

    for (sz, reps) in [(128,20), (256,20), (512,10), (1024,5), (2048,3)] as [(Int,Int)] {
        let M = sz, N = sz, K = sz, flops = 2.0 * Double(M*N) * Double(K)
        let A = device.makeBuffer(length: M*K*4, options: .storageModeShared)!
        let B = device.makeBuffer(length: K*N*4, options: .storageModeShared)!
        let C = device.makeBuffer(length: M*N*4, options: .storageModeShared)!

        let pA = A.contents().bindMemory(to: Float.self, capacity: M*K)
        let pB = B.contents().bindMemory(to: Float.self, capacity: K*N)
        for i in 0..<(M*K) { pA[i] = 1 }; for i in 0..<(K*N) { pB[i] = 1 }

        func bench(_ pip: MTLComputePipelineState, tiled: Bool) -> Double {
            for _ in 0..<2 { let _ = dispatchGEMM(pip, A: A, B: B, C: C, M: M, N: N, K: K, tiled: tiled) }
            var t: Double = 0
            for _ in 0..<reps { autoreleasepool { t += dispatchGEMM(pip, A: A, B: B, C: C, M: M, N: N, K: K, tiled: tiled) } }
            return t * 1000 / Double(reps)
        }

        let naiveMs = bench(naivePip, tiled: false)
        let tiledMs = bench(tiledPip, tiled: true)

        var cpuC = [Float](repeating: 0, count: M*N)
        cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, Int32(M), Int32(N), Int32(K),
                    1.0, pA, Int32(K), pB, Int32(N), 0.0, &cpuC, Int32(N))
        let t0 = CFAbsoluteTimeGetCurrent()
        for _ in 0..<reps { cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, Int32(M), Int32(N), Int32(K),
                                         1.0, pA, Int32(K), pB, Int32(N), 0.0, &cpuC, Int32(N)) }
        let accelMs = (CFAbsoluteTimeGetCurrent() - t0) * 1000 / Double(reps)

        print(String(format: "  %8d   %8.2f   %8.2f   %8.2f    %10.1f",
                     sz, naiveMs, tiledMs, accelMs, flops / (tiledMs * 1e6)))
    }
}

// MARK: - Test 4: Tiny traceable multiply (4×2 × 2×4)

do {
    let M = 4, N = 4, K = 2
    let A = device.makeBuffer(length: M*K*4, options: .storageModeShared)!
    let B = device.makeBuffer(length: K*N*4, options: .storageModeShared)!
    let C = device.makeBuffer(length: M*N*4, options: .storageModeShared)!

    let pA = A.contents().bindMemory(to: Float.self, capacity: M*K)
    let pB = B.contents().bindMemory(to: Float.self, capacity: K*N)
    for i in 0..<(M*K) { pA[i] = Float(i + 1) }  // [1,2,3,4,5,6,7,8]
    pB[0]=1; pB[1]=0; pB[2]=1; pB[3]=0; pB[4]=0; pB[5]=1; pB[6]=0; pB[7]=1
    memset(C.contents(), 0, M*N*4)

    let _ = dispatchGEMM(tiledPip, A: A, B: B, C: C, M: M, N: N, K: K, tiled: true)

    let pC = C.contents().bindMemory(to: Float.self, capacity: M*N)
    let expected: [Float] = [1,2,1,2, 3,4,3,4, 5,6,5,6, 7,8,7,8]
    var pass = true
    for i in 0..<(M*N) { if abs(pC[i] - expected[i]) > 1e-5 { pass = false } }
    print("\n  4×2 × 2×4:       \(pass ? "✓" : "✗")")
}
