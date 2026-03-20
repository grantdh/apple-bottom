#!/usr/bin/env swift
//
// Exercise 4: Complex Matrix Multiply (CGEMM) with Register Blocking
//
// WHAT YOU'LL LEARN:
//   - Register blocking: each thread computes a 4×4 sub-tile, not 1 element
//   - Why register blocking is THE technique that gets you to 2-3 TFLOPS
//   - Complex tiled GEMM using float2 in threadgroup memory
//   - How cmul (Exercise 1) + tiling (Exercise 3) compose into real BLAS
//   - Comparing against Accelerate's cblas_cgemm
//
// COMPILE & RUN:
//   swiftc -O -framework Metal -framework Foundation -framework Accelerate \
//       ex04_register_blocked_cgemm.swift -o ex04
//   ./ex04
//
// Grant Heileman — UNM ECE — 2026
//

import Foundation
import Metal
import Accelerate

let shaderSource = """
#include <metal_stdlib>
using namespace metal;

// ─── Complex helpers (same as Exercise 1) ─────────────────────────

inline float2 cmul(float2 a, float2 b) {
    return float2(a.x * b.x - a.y * b.y,
                  a.x * b.y + a.y * b.x);
}

// ─── Constants ────────────────────────────────────────────────────

#define BM 64       // output rows per threadgroup
#define BN 64       // output cols per threadgroup
#define TM 4        // output rows per thread (register block height)
#define TN 4        // output cols per thread (register block width)
#define TILE_K 16   // K elements per shared-memory pass

// Threads per threadgroup: (BM/TM) × (BN/TN) = 16 × 16 = 256
#define THREADS_Y (BM / TM)   // 16
#define THREADS_X (BN / TN)   // 16
#define NUM_THREADS (THREADS_Y * THREADS_X)  // 256 — used for cooperative loads

// ─── Kernel 1: Simple tiled CGEMM (1 element per thread) ─────────

#define TILE_SIMPLE 16

kernel void cgemm_simple(
    device const float2 *A     [[buffer(0)]],
    device const float2 *B     [[buffer(1)]],
    device float2       *C     [[buffer(2)]],
    constant uint       &M     [[buffer(3)]],
    constant uint       &N     [[buffer(4)]],
    constant uint       &K_dim [[buffer(5)]],
    uint2 lid  [[thread_position_in_threadgroup]],
    uint2 tgid [[threadgroup_position_in_grid]]
) {
    uint row = tgid.y * TILE_SIMPLE + lid.y;
    uint col = tgid.x * TILE_SIMPLE + lid.x;

    threadgroup float2 tA[TILE_SIMPLE][TILE_SIMPLE];
    threadgroup float2 tB[TILE_SIMPLE][TILE_SIMPLE];

    float2 sum = float2(0.0);
    uint numTiles = (K_dim + TILE_SIMPLE - 1) / TILE_SIMPLE;

    for (uint t = 0; t < numTiles; t++) {
        uint a_col = t * TILE_SIMPLE + lid.x;
        uint b_row = t * TILE_SIMPLE + lid.y;

        tA[lid.y][lid.x] = (row < M && a_col < K_dim)
            ? A[row * K_dim + a_col] : float2(0.0);
        tB[lid.y][lid.x] = (b_row < K_dim && col < N)
            ? B[b_row * N + col] : float2(0.0);

        threadgroup_barrier(mem_flags::mem_threadgroup);

        for (uint k = 0; k < TILE_SIMPLE; k++) {
            sum += cmul(tA[lid.y][k], tB[k][lid.x]);
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    if (row < M && col < N) {
        C[row * N + col] = sum;
    }
}

// ─── Kernel 2: Register-blocked CGEMM ────────────────────────────
//
// Each thread computes a TM×TN = 4×4 sub-tile of output.
// 16 float2 accumulators living in registers.

kernel void cgemm_register_blocked(
    device const float2 *A     [[buffer(0)]],
    device const float2 *B     [[buffer(1)]],
    device float2       *C     [[buffer(2)]],
    constant uint       &M     [[buffer(3)]],
    constant uint       &N     [[buffer(4)]],
    constant uint       &K_dim [[buffer(5)]],
    uint2 lid   [[thread_position_in_threadgroup]],
    uint2 tgid  [[threadgroup_position_in_grid]],
    uint  flatId [[thread_index_in_threadgroup]]
) {
    uint blockRowStart = tgid.y * BM;
    uint blockColStart = tgid.x * BN;
    uint ty = lid.y;
    uint tx = lid.x;

    float2 acc[TM][TN];
    for (uint i = 0; i < TM; i++)
        for (uint j = 0; j < TN; j++)
            acc[i][j] = float2(0.0);

    threadgroup float2 tileA[BM * TILE_K];
    threadgroup float2 tileB[TILE_K * BN];

    uint numKTiles = (K_dim + TILE_K - 1) / TILE_K;

    for (uint kt = 0; kt < numKTiles; kt++) {

        // Cooperative load of tileA: BM*TILE_K elements / NUM_THREADS loads each
        for (uint i = flatId; i < BM * TILE_K; i += NUM_THREADS) {
            uint tileRow = i / TILE_K;
            uint tileCol = i % TILE_K;
            uint globalRow = blockRowStart + tileRow;
            uint globalCol = kt * TILE_K + tileCol;
            tileA[i] = (globalRow < M && globalCol < K_dim)
                ? A[globalRow * K_dim + globalCol] : float2(0.0);
        }

        // Cooperative load of tileB: TILE_K*BN elements / NUM_THREADS loads each
        for (uint i = flatId; i < TILE_K * BN; i += NUM_THREADS) {
            uint tileRow = i / BN;
            uint tileCol = i % BN;
            uint globalRow = kt * TILE_K + tileRow;
            uint globalCol = blockColStart + tileCol;
            tileB[i] = (globalRow < K_dim && globalCol < N)
                ? B[globalRow * N + globalCol] : float2(0.0);
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);

        // Register-blocked multiply-accumulate (outer product pattern)
        for (uint k = 0; k < TILE_K; k++) {
            float2 a_vals[TM];
            for (uint i = 0; i < TM; i++) {
                a_vals[i] = tileA[(ty * TM + i) * TILE_K + k];
            }
            float2 b_vals[TN];
            for (uint j = 0; j < TN; j++) {
                b_vals[j] = tileB[k * BN + tx * TN + j];
            }
            for (uint i = 0; i < TM; i++) {
                for (uint j = 0; j < TN; j++) {
                    acc[i][j] += cmul(a_vals[i], b_vals[j]);
                }
            }
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    // Write TM×TN results to global memory
    for (uint i = 0; i < TM; i++) {
        for (uint j = 0; j < TN; j++) {
            uint globalRow = blockRowStart + ty * TM + i;
            uint globalCol = blockColStart + tx * TN + j;
            if (globalRow < M && globalCol < N) {
                C[globalRow * N + globalCol] = acc[i][j];
            }
        }
    }
}
"""

// ═══════════════════════════════════════════════════════════════════════════
// SWIFT HOST CODE
// ═══════════════════════════════════════════════════════════════════════════

print("""

╔═══════════════════════════════════════════════════════════════╗
║  Exercise 4: Complex GEMM with Register Blocking             ║
╚═══════════════════════════════════════════════════════════════╝

""")

guard let device = MTLCreateSystemDefaultDevice() else { print("❌ No Metal device"); exit(1) }
assert(device.hasUnifiedMemory, "These exercises require Apple Silicon (unified memory)")
print("GPU: \(device.name)")

func gpuCheck(_ cb: MTLCommandBuffer, label: String) {
    if cb.status == .error {
        print("❌ GPU error [\(label)]: \(cb.error?.localizedDescription ?? "unknown")")
        exit(1)
    }
}

let compileOptions = MTLCompileOptions()
compileOptions.mathMode = .fast

let library: MTLLibrary
do {
    library = try device.makeLibrary(source: shaderSource, options: compileOptions)
    print("✓ Shaders compiled\n")
} catch {
    print("❌ Shader compilation failed: \(error)"); exit(1)
}

guard let commandQueue = device.makeCommandQueue() else { print("❌ Command queue failed"); exit(1) }

// ── Constants matching shader #defines ──────────────────────────────────
let TILE_SIMPLE = 16
let BM = 64, BN = 64, TM = 4, TN = 4

func maxComplexError(_ a: UnsafePointer<SIMD2<Float>>, _ b: UnsafePointer<SIMD2<Float>>, count: Int) -> Float {
    var mx: Float = 0
    for i in 0..<count { mx = max(mx, max(abs(a[i].x - b[i].x), abs(a[i].y - b[i].y))) }
    return mx
}

func complexFrobNorm(_ a: UnsafePointer<SIMD2<Float>>, count: Int) -> Float {
    var s: Float = 0
    for i in 0..<count { s += a[i].x * a[i].x + a[i].y * a[i].y }
    return sqrt(s)
}

// Accelerate reference — using cblas_cgemm (single-precision complex GEMM)
func accelerateCGEMM(A: UnsafePointer<SIMD2<Float>>, B: UnsafePointer<SIMD2<Float>>,
                     C: UnsafeMutablePointer<SIMD2<Float>>, M: Int, N: Int, K: Int) {
    // Use withUnsafePointer for type-safe alpha/beta passing.
    // cblas_cgemm expects pointers to interleaved [real, imag] float pairs.
    var alpha = SIMD2<Float>(1.0, 0.0)
    var beta = SIMD2<Float>(0.0, 0.0)
    withUnsafePointer(to: &alpha) { alphaPtr in
        withUnsafePointer(to: &beta) { betaPtr in
            cblas_cgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                        Int32(M), Int32(N), Int32(K),
                        OpaquePointer(alphaPtr),
                        OpaquePointer(A), Int32(K),
                        OpaquePointer(B), Int32(N),
                        OpaquePointer(betaPtr),
                        OpaquePointer(C), Int32(N))
        }
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// TEST 1: Correctness — both kernels vs Accelerate
// ═══════════════════════════════════════════════════════════════════════════

print("── Test 1: Correctness (128×128, both kernels) ──────────────")

do {
    let M = 128, N = 128, K = 128, count = M * N
    let sizeA = M * K * MemoryLayout<SIMD2<Float>>.stride
    let sizeB = K * N * MemoryLayout<SIMD2<Float>>.stride
    let sizeC = count * MemoryLayout<SIMD2<Float>>.stride

    guard let bufA  = device.makeBuffer(length: sizeA, options: .storageModeShared),
          let bufB  = device.makeBuffer(length: sizeB, options: .storageModeShared),
          let bufC1 = device.makeBuffer(length: sizeC, options: .storageModeShared),
          let bufC2 = device.makeBuffer(length: sizeC, options: .storageModeShared) else { exit(1) }

    let ptrA = bufA.contents().bindMemory(to: SIMD2<Float>.self, capacity: M * K)
    let ptrB = bufB.contents().bindMemory(to: SIMD2<Float>.self, capacity: K * N)
    srand48(42)
    for i in 0..<(M * K) { ptrA[i] = SIMD2<Float>(Float(drand48() - 0.5), Float(drand48() - 0.5)) }
    for i in 0..<(K * N) { ptrB[i] = SIMD2<Float>(Float(drand48() - 0.5), Float(drand48() - 0.5)) }
    memset(bufC1.contents(), 0, sizeC)
    memset(bufC2.contents(), 0, sizeC)

    var refC = [SIMD2<Float>](repeating: .zero, count: count)
    accelerateCGEMM(A: ptrA, B: ptrB, C: &refC, M: M, N: N, K: K)
    let norm = complexFrobNorm(&refC, count: count)

    guard let simpleFunc = library.makeFunction(name: "cgemm_simple"),
          let blockedFunc = library.makeFunction(name: "cgemm_register_blocked") else { exit(1) }
    let simplePipeline = try device.makeComputePipelineState(function: simpleFunc)
    let blockedPipeline = try device.makeComputePipelineState(function: blockedFunc)

    autoreleasepool {
        guard let cb = commandQueue.makeCommandBuffer(), let enc = cb.makeComputeCommandEncoder() else { exit(1) }
        cb.label = "cgemm_simple_correctness"
        enc.setComputePipelineState(simplePipeline)
        enc.setBuffer(bufA, offset: 0, index: 0); enc.setBuffer(bufB, offset: 0, index: 1); enc.setBuffer(bufC1, offset: 0, index: 2)
        var m = UInt32(M), n = UInt32(N), k = UInt32(K)
        enc.setBytes(&m, length: 4, index: 3); enc.setBytes(&n, length: 4, index: 4); enc.setBytes(&k, length: 4, index: 5)
        enc.dispatchThreadgroups(
            MTLSize(width: (N + TILE_SIMPLE - 1) / TILE_SIMPLE, height: (M + TILE_SIMPLE - 1) / TILE_SIMPLE, depth: 1),
            threadsPerThreadgroup: MTLSize(width: TILE_SIMPLE, height: TILE_SIMPLE, depth: 1))
        enc.endEncoding(); cb.commit(); cb.waitUntilCompleted()
        gpuCheck(cb, label: "cgemm_simple_correctness")
    }

    autoreleasepool {
        guard let cb = commandQueue.makeCommandBuffer(), let enc = cb.makeComputeCommandEncoder() else { exit(1) }
        cb.label = "cgemm_blocked_correctness"
        enc.setComputePipelineState(blockedPipeline)
        enc.setBuffer(bufA, offset: 0, index: 0); enc.setBuffer(bufB, offset: 0, index: 1); enc.setBuffer(bufC2, offset: 0, index: 2)
        var m = UInt32(M), n = UInt32(N), k = UInt32(K)
        enc.setBytes(&m, length: 4, index: 3); enc.setBytes(&n, length: 4, index: 4); enc.setBytes(&k, length: 4, index: 5)
        enc.dispatchThreadgroups(
            MTLSize(width: (N + BN - 1) / BN, height: (M + BM - 1) / BM, depth: 1),
            threadsPerThreadgroup: MTLSize(width: BN / TN, height: BM / TM, depth: 1))
        enc.endEncoding(); cb.commit(); cb.waitUntilCompleted()
        gpuCheck(cb, label: "cgemm_blocked_correctness")
    }

    let ptrC1 = bufC1.contents().bindMemory(to: SIMD2<Float>.self, capacity: count)
    let ptrC2 = bufC2.contents().bindMemory(to: SIMD2<Float>.self, capacity: count)
    let errSimple = maxComplexError(ptrC1, &refC, count: count) / norm
    let errBlocked = maxComplexError(ptrC2, &refC, count: count) / norm

    print("  Simple vs Accelerate:  rel err = \(errSimple)  \(errSimple < 1e-5 ? "✓" : "✗")")
    print("  Blocked vs Accelerate: rel err = \(errBlocked)  \(errBlocked < 1e-5 ? "✓" : "✗")")
}

// ═══════════════════════════════════════════════════════════════════════════
// TEST 2: Non-aligned QE-like dimensions
// ═══════════════════════════════════════════════════════════════════════════

print("\n── Test 2: Non-aligned QE dimensions (288×100, K=137) ───────")

do {
    let M = 288, N = 100, K = 137, count = M * N
    let sizeA = M * K * MemoryLayout<SIMD2<Float>>.stride
    let sizeB = K * N * MemoryLayout<SIMD2<Float>>.stride
    let sizeC = count * MemoryLayout<SIMD2<Float>>.stride

    guard let bufA = device.makeBuffer(length: sizeA, options: .storageModeShared),
          let bufB = device.makeBuffer(length: sizeB, options: .storageModeShared),
          let bufC = device.makeBuffer(length: sizeC, options: .storageModeShared) else { exit(1) }

    let ptrA = bufA.contents().bindMemory(to: SIMD2<Float>.self, capacity: M * K)
    let ptrB = bufB.contents().bindMemory(to: SIMD2<Float>.self, capacity: K * N)
    srand48(99)
    for i in 0..<(M * K) { ptrA[i] = SIMD2<Float>(Float(drand48() - 0.5), Float(drand48() - 0.5)) }
    for i in 0..<(K * N) { ptrB[i] = SIMD2<Float>(Float(drand48() - 0.5), Float(drand48() - 0.5)) }
    memset(bufC.contents(), 0, sizeC)

    var refC = [SIMD2<Float>](repeating: .zero, count: count)
    accelerateCGEMM(A: ptrA, B: ptrB, C: &refC, M: M, N: N, K: K)
    let norm = complexFrobNorm(&refC, count: count)

    guard let func2 = library.makeFunction(name: "cgemm_register_blocked") else { exit(1) }
    let pipeline = try device.makeComputePipelineState(function: func2)

    autoreleasepool {
        guard let cb = commandQueue.makeCommandBuffer(), let enc = cb.makeComputeCommandEncoder() else { exit(1) }
        cb.label = "cgemm_blocked_nonaligned"
        enc.setComputePipelineState(pipeline)
        enc.setBuffer(bufA, offset: 0, index: 0); enc.setBuffer(bufB, offset: 0, index: 1); enc.setBuffer(bufC, offset: 0, index: 2)
        var m = UInt32(M), n = UInt32(N), k = UInt32(K)
        enc.setBytes(&m, length: 4, index: 3); enc.setBytes(&n, length: 4, index: 4); enc.setBytes(&k, length: 4, index: 5)
        enc.dispatchThreadgroups(
            MTLSize(width: (N + BN - 1) / BN, height: (M + BM - 1) / BM, depth: 1),
            threadsPerThreadgroup: MTLSize(width: BN / TN, height: BM / TM, depth: 1))
        enc.endEncoding(); cb.commit(); cb.waitUntilCompleted()
        gpuCheck(cb, label: "cgemm_blocked_nonaligned")
    }

    let ptrC = bufC.contents().bindMemory(to: SIMD2<Float>.self, capacity: count)
    let rel = maxComplexError(ptrC, &refC, count: count) / norm
    print("  Dimensions: \(M)×\(K) × \(K)×\(N) → \(M)×\(N)")
    print("  Register-blocked vs Accelerate: rel err = \(rel)  \(rel < 1e-5 ? "✓ PASS" : "✗ FAIL")")
}

// ═══════════════════════════════════════════════════════════════════════════
// TEST 3: Performance — GPU timestamps
// ═══════════════════════════════════════════════════════════════════════════

print("\n── Test 3: Performance (GPU timestamps) ─────────────────────")

do {
    guard let simpleFunc = library.makeFunction(name: "cgemm_simple"),
          let blockedFunc = library.makeFunction(name: "cgemm_register_blocked") else { exit(1) }
    let simplePipeline = try device.makeComputePipelineState(function: simpleFunc)
    let blockedPipeline = try device.makeComputePipelineState(function: blockedFunc)

    let sizes: [(Int, Int)] = [(128, 20), (256, 15), (512, 10), (1024, 5), (2048, 3)]

    print("     M=N=K   Simple (ms)  Blocked (ms)  Accel (ms)  Blocked GFLOP/s  Accel GFLOP/s  Speedup")
    print("  " + String(repeating: "─", count: 92))

    for (size, reps) in sizes {
        let M = size, N = size, K = size
        let flops = 8.0 * Double(M) * Double(N) * Double(K)
        let sizeA = M * K * MemoryLayout<SIMD2<Float>>.stride
        let sizeB = K * N * MemoryLayout<SIMD2<Float>>.stride
        let sizeC = M * N * MemoryLayout<SIMD2<Float>>.stride

        guard let bufA = device.makeBuffer(length: sizeA, options: .storageModeShared),
              let bufB = device.makeBuffer(length: sizeB, options: .storageModeShared),
              let bufC = device.makeBuffer(length: sizeC, options: .storageModeShared) else { continue }

        let ptrA = bufA.contents().bindMemory(to: SIMD2<Float>.self, capacity: M * K)
        let ptrB = bufB.contents().bindMemory(to: SIMD2<Float>.self, capacity: K * N)
        for i in 0..<(M * K) { ptrA[i] = SIMD2<Float>(1.0, 0.0) }
        for i in 0..<(K * N) { ptrB[i] = SIMD2<Float>(1.0, 0.0) }

        func timeKernel(_ pipeline: MTLComputePipelineState, tgWidth: Int, tgHeight: Int,
                        gridW: Int, gridH: Int) -> Double {
            for _ in 0..<2 {
                autoreleasepool {
                    guard let cb = commandQueue.makeCommandBuffer(),
                          let enc = cb.makeComputeCommandEncoder() else { return }
                    enc.setComputePipelineState(pipeline)
                    enc.setBuffer(bufA, offset: 0, index: 0); enc.setBuffer(bufB, offset: 0, index: 1); enc.setBuffer(bufC, offset: 0, index: 2)
                    var m = UInt32(M), n = UInt32(N), k = UInt32(K)
                    enc.setBytes(&m, length: 4, index: 3); enc.setBytes(&n, length: 4, index: 4); enc.setBytes(&k, length: 4, index: 5)
                    enc.dispatchThreadgroups(MTLSize(width: gridW, height: gridH, depth: 1),
                                              threadsPerThreadgroup: MTLSize(width: tgWidth, height: tgHeight, depth: 1))
                    enc.endEncoding(); cb.commit(); cb.waitUntilCompleted()
                }
            }
            var totalGpu: Double = 0
            for _ in 0..<reps {
                autoreleasepool {
                    guard let cb = commandQueue.makeCommandBuffer(),
                          let enc = cb.makeComputeCommandEncoder() else { return }
                    enc.setComputePipelineState(pipeline)
                    enc.setBuffer(bufA, offset: 0, index: 0); enc.setBuffer(bufB, offset: 0, index: 1); enc.setBuffer(bufC, offset: 0, index: 2)
                    var m = UInt32(M), n = UInt32(N), k = UInt32(K)
                    enc.setBytes(&m, length: 4, index: 3); enc.setBytes(&n, length: 4, index: 4); enc.setBytes(&k, length: 4, index: 5)
                    enc.dispatchThreadgroups(MTLSize(width: gridW, height: gridH, depth: 1),
                                              threadsPerThreadgroup: MTLSize(width: tgWidth, height: tgHeight, depth: 1))
                    enc.endEncoding(); cb.commit(); cb.waitUntilCompleted()
                    gpuCheck(cb, label: "cgemm_bench")
                    totalGpu += cb.gpuEndTime - cb.gpuStartTime
                }
            }
            return totalGpu * 1000.0 / Double(reps)
        }

        let simpleGridW = (N + TILE_SIMPLE - 1) / TILE_SIMPLE
        let simpleGridH = (M + TILE_SIMPLE - 1) / TILE_SIMPLE
        let simpleMs = timeKernel(simplePipeline, tgWidth: TILE_SIMPLE, tgHeight: TILE_SIMPLE,
                                   gridW: simpleGridW, gridH: simpleGridH)

        let blockedGridW = (N + BN - 1) / BN
        let blockedGridH = (M + BM - 1) / BM
        let blockedMs = timeKernel(blockedPipeline, tgWidth: BN / TN, tgHeight: BM / TM,
                                    gridW: blockedGridW, gridH: blockedGridH)

        // Accelerate
        var cpuC = [SIMD2<Float>](repeating: .zero, count: M * N)
        accelerateCGEMM(A: ptrA, B: ptrB, C: &cpuC, M: M, N: N, K: K)
        let t0 = CFAbsoluteTimeGetCurrent()
        for _ in 0..<reps {
            accelerateCGEMM(A: ptrA, B: ptrB, C: &cpuC, M: M, N: N, K: K)
        }
        let accelMs = (CFAbsoluteTimeGetCurrent() - t0) * 1000.0 / Double(reps)

        let blockedGF = flops / (blockedMs * 1e6)
        let accelGF = flops / (accelMs * 1e6)
        let speedup = simpleMs / blockedMs

        print(String(format: "  %8d  %10.2f   %11.2f  %10.2f    %12.1f  %12.1f    %.1fx",
                     size, simpleMs, blockedMs, accelMs, blockedGF, accelGF, speedup))
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// TEST 4: Verify the complex arithmetic with known values
// ═══════════════════════════════════════════════════════════════════════════

print("\n── Test 4: Known complex multiply ───────────────────────────")

do {
    let M = 1, N = 1, K = 2
    let sizeA = M * K * MemoryLayout<SIMD2<Float>>.stride
    let sizeB = K * N * MemoryLayout<SIMD2<Float>>.stride
    let sizeC = M * N * MemoryLayout<SIMD2<Float>>.stride

    guard let bufA = device.makeBuffer(length: sizeA, options: .storageModeShared),
          let bufB = device.makeBuffer(length: sizeB, options: .storageModeShared),
          let bufC = device.makeBuffer(length: sizeC, options: .storageModeShared) else { exit(1) }

    let ptrA = bufA.contents().bindMemory(to: SIMD2<Float>.self, capacity: M * K)
    let ptrB = bufB.contents().bindMemory(to: SIMD2<Float>.self, capacity: K * N)
    ptrA[0] = SIMD2<Float>(1, 1); ptrA[1] = SIMD2<Float>(2, 0)
    ptrB[0] = SIMD2<Float>(1, 0); ptrB[1] = SIMD2<Float>(0, 1)
    memset(bufC.contents(), 0, sizeC)

    guard let func2 = library.makeFunction(name: "cgemm_register_blocked") else { exit(1) }
    let pipeline = try device.makeComputePipelineState(function: func2)

    autoreleasepool {
        guard let cb = commandQueue.makeCommandBuffer(), let enc = cb.makeComputeCommandEncoder() else { exit(1) }
        cb.label = "cgemm_known_values"
        enc.setComputePipelineState(pipeline)
        enc.setBuffer(bufA, offset: 0, index: 0); enc.setBuffer(bufB, offset: 0, index: 1); enc.setBuffer(bufC, offset: 0, index: 2)
        var m = UInt32(M), n = UInt32(N), k = UInt32(K)
        enc.setBytes(&m, length: 4, index: 3); enc.setBytes(&n, length: 4, index: 4); enc.setBytes(&k, length: 4, index: 5)
        enc.dispatchThreadgroups(MTLSize(width: 1, height: 1, depth: 1),
                                  threadsPerThreadgroup: MTLSize(width: BN / TN, height: BM / TM, depth: 1))
        enc.endEncoding(); cb.commit(); cb.waitUntilCompleted()
        gpuCheck(cb, label: "cgemm_known_values")
    }

    let result = bufC.contents().bindMemory(to: SIMD2<Float>.self, capacity: 1)[0]
    let expected = SIMD2<Float>(1, 3)
    let err = max(abs(result.x - expected.x), abs(result.y - expected.y))
    print("  A = [(1+i), (2+0i)]   B = [(1+0i), (0+i)]^T")
    print("  C = (1+i)(1) + (2)(i) = 1 + 3i")
    print("  GPU result: \(result.x) + \(result.y)i  \(err < 1e-5 ? "✓ PASS" : "✗ FAIL")")
}

print("""

═══════════════════════════════════════════════════════════════
  Exercise 4 complete.

  What you learned:
    • Register blocking: each thread computes TM×TN output elements
    • NUM_THREADS = (BM/TM)*(BN/TN) — cooperative load stride is derived,
      never hardcoded. Change BM or TM and it auto-adjusts.
    • 4× arithmetic intensity over the simple tiled kernel
    • The outer-product pattern: load a[TM] and b[TN] from shared
      memory, produce TM×TN multiply-adds (the inner loop)
    • Complex GEMM = tiled SGEMM + cmul

  NEXT: Exercise 5 — 1D FFT (Stockham radix-2).
═══════════════════════════════════════════════════════════════
""")
