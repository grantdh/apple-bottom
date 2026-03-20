#!/usr/bin/env swift
//
// Exercise 3: Tiled Matrix Multiply (SGEMM) on Metal GPU
//
// WHAT YOU'LL LEARN:
//   - 2D thread dispatch (threads arranged in a grid, not a line)
//   - Tiled matrix multiply: load tiles into threadgroup memory, multiply, accumulate
//   - Why tiling is necessary (memory bandwidth is the bottleneck, not compute)
//   - How threadgroup memory turns a memory-bound kernel into a compute-bound one
//   - Comparing your GPU kernel against Apple's Accelerate (AMX) SGEMM
//
// COMPILE & RUN:
//   swiftc -O -framework Metal -framework Foundation -framework Accelerate \
//       ex03_tiled_sgemm.swift -o ex03
//   ./ex03
//
// Grant Heileman — UNM ECE — 2026
//

import Foundation
import Metal
import Accelerate

// ═══════════════════════════════════════════════════════════════════════════
// THE METAL SHADERS
//
// Two kernels: naive (for comparison) and tiled (the real thing).
//
// Matrix layout: ROW-MAJOR, C[row][col] = C[row * N + col]
//   C = A * B
//   A is M×K, B is K×N, C is M×N
//
// THE TILING IDEA:
//   Naive: each thread reads a full row/column from global memory = chaos.
//   Tiled: threads cooperate to load tiles into shared memory, then multiply.
//          TILE× reduction in memory traffic.
// ═══════════════════════════════════════════════════════════════════════════

let shaderSource = """
#include <metal_stdlib>
using namespace metal;

// Tile size — each threadgroup processes a TILE×TILE output block.
// 16×16 = 256 threads per threadgroup.
// With 32 KB threadgroup memory: two 16×16 float tiles = 2 KB. Plenty of room.
#define TILE 16

// ─── Kernel 1: Naive SGEMM (no tiling) ───────────────────────────

kernel void sgemm_naive(
    device const float *A      [[buffer(0)]],
    device const float *B      [[buffer(1)]],
    device float       *C      [[buffer(2)]],
    constant uint      &M      [[buffer(3)]],
    constant uint      &N      [[buffer(4)]],
    constant uint      &K_dim  [[buffer(5)]],
    uint2 gid [[thread_position_in_grid]]
) {
    uint row = gid.y;
    uint col = gid.x;

    if (row >= M || col >= N) return;

    float sum = 0.0;
    for (uint k = 0; k < K_dim; k++) {
        sum += A[row * K_dim + k] * B[k * N + col];
    }

    C[row * N + col] = sum;
}

// ─── Kernel 2: Tiled SGEMM ──────────────────────────────────────

kernel void sgemm_tiled(
    device const float *A      [[buffer(0)]],
    device const float *B      [[buffer(1)]],
    device float       *C      [[buffer(2)]],
    constant uint      &M      [[buffer(3)]],
    constant uint      &N      [[buffer(4)]],
    constant uint      &K_dim  [[buffer(5)]],

    uint2 gid  [[thread_position_in_grid]],
    uint2 lid  [[thread_position_in_threadgroup]],
    uint2 tgid [[threadgroup_position_in_grid]]
) {
    uint row = tgid.y * TILE + lid.y;
    uint col = tgid.x * TILE + lid.x;

    threadgroup float tileA[TILE][TILE];
    threadgroup float tileB[TILE][TILE];

    float sum = 0.0;
    uint numTiles = (K_dim + TILE - 1) / TILE;

    for (uint t = 0; t < numTiles; t++) {
        uint a_col = t * TILE + lid.x;
        uint b_row = t * TILE + lid.y;

        if (row < M && a_col < K_dim) {
            tileA[lid.y][lid.x] = A[row * K_dim + a_col];
        } else {
            tileA[lid.y][lid.x] = 0.0;
        }

        if (b_row < K_dim && col < N) {
            tileB[lid.y][lid.x] = B[b_row * N + col];
        } else {
            tileB[lid.y][lid.x] = 0.0;
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);

        for (uint k = 0; k < TILE; k++) {
            sum += tileA[lid.y][k] * tileB[k][lid.x];
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    if (row < M && col < N) {
        C[row * N + col] = sum;
    }
}
"""

// ═══════════════════════════════════════════════════════════════════════════
// SWIFT HOST CODE
// ═══════════════════════════════════════════════════════════════════════════

print("""

╔═══════════════════════════════════════════════════════════════╗
║  Exercise 3: Tiled Matrix Multiply (SGEMM) on Metal GPU     ║
╚═══════════════════════════════════════════════════════════════╝

""")

guard let device = MTLCreateSystemDefaultDevice() else {
    print("❌ No Metal device"); exit(1)
}
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

guard let commandQueue = device.makeCommandQueue() else {
    print("❌ Command queue failed"); exit(1)
}

// ── TILE constant: must match #define TILE in shader ───────────────────
// If these diverge, dispatch geometry and threadgroup memory sizing break
// silently. Keep them locked together.
let TILE = 16
// Compile-time assertion: if you change the shader's #define TILE, update this too.

func maxError(_ a: UnsafePointer<Float>, _ b: UnsafePointer<Float>, count: Int) -> Float {
    var mx: Float = 0
    for i in 0..<count { mx = max(mx, abs(a[i] - b[i])) }
    return mx
}

func frobNorm(_ a: UnsafePointer<Float>, count: Int) -> Float {
    var s: Float = 0
    for i in 0..<count { s += a[i] * a[i] }
    return sqrt(s)
}

// ═══════════════════════════════════════════════════════════════════════════
// TEST 1: Small matrix — verify correctness against Accelerate
// ═══════════════════════════════════════════════════════════════════════════

print("── Test 1: Correctness (64×64 × 64×64) ─────────────────────")

do {
    let M = 64, N = 64, K = 64
    let sizeA = M * K * MemoryLayout<Float>.stride
    let sizeB = K * N * MemoryLayout<Float>.stride
    let sizeC = M * N * MemoryLayout<Float>.stride

    guard let bufA = device.makeBuffer(length: sizeA, options: .storageModeShared),
          let bufB = device.makeBuffer(length: sizeB, options: .storageModeShared),
          let bufC_naive = device.makeBuffer(length: sizeC, options: .storageModeShared),
          let bufC_tiled = device.makeBuffer(length: sizeC, options: .storageModeShared) else {
        print("❌ Buffer allocation failed"); exit(1)
    }

    let ptrA = bufA.contents().bindMemory(to: Float.self, capacity: M * K)
    let ptrB = bufB.contents().bindMemory(to: Float.self, capacity: K * N)

    srand48(42)
    for i in 0..<(M * K) { ptrA[i] = Float(drand48() - 0.5) }
    for i in 0..<(K * N) { ptrB[i] = Float(drand48() - 0.5) }
    memset(bufC_naive.contents(), 0, sizeC)
    memset(bufC_tiled.contents(), 0, sizeC)

    var refC = [Float](repeating: 0, count: M * N)
    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                Int32(M), Int32(N), Int32(K), 1.0, ptrA, Int32(K), ptrB, Int32(N), 0.0, &refC, Int32(N))

    // GPU: Naive
    guard let naiveFunc = library.makeFunction(name: "sgemm_naive") else { print("❌ Kernel not found"); exit(1) }
    let naivePipeline = try device.makeComputePipelineState(function: naiveFunc)

    autoreleasepool {
        guard let cmdBuf = commandQueue.makeCommandBuffer(),
              let enc = cmdBuf.makeComputeCommandEncoder() else { exit(1) }
        cmdBuf.label = "sgemm_naive_correctness"
        enc.setComputePipelineState(naivePipeline)
        enc.setBuffer(bufA, offset: 0, index: 0)
        enc.setBuffer(bufB, offset: 0, index: 1)
        enc.setBuffer(bufC_naive, offset: 0, index: 2)
        var m = UInt32(M), n = UInt32(N), k = UInt32(K)
        enc.setBytes(&m, length: 4, index: 3)
        enc.setBytes(&n, length: 4, index: 4)
        enc.setBytes(&k, length: 4, index: 5)
        enc.dispatchThreads(
            MTLSize(width: N, height: M, depth: 1),
            threadsPerThreadgroup: MTLSize(width: TILE, height: TILE, depth: 1))
        enc.endEncoding(); cmdBuf.commit(); cmdBuf.waitUntilCompleted()
        gpuCheck(cmdBuf, label: "sgemm_naive_correctness")
    }

    // GPU: Tiled
    guard let tiledFunc = library.makeFunction(name: "sgemm_tiled") else { print("❌ Kernel not found"); exit(1) }
    let tiledPipeline = try device.makeComputePipelineState(function: tiledFunc)

    autoreleasepool {
        guard let cmdBuf = commandQueue.makeCommandBuffer(),
              let enc = cmdBuf.makeComputeCommandEncoder() else { exit(1) }
        cmdBuf.label = "sgemm_tiled_correctness"
        enc.setComputePipelineState(tiledPipeline)
        enc.setBuffer(bufA, offset: 0, index: 0)
        enc.setBuffer(bufB, offset: 0, index: 1)
        enc.setBuffer(bufC_tiled, offset: 0, index: 2)
        var m = UInt32(M), n = UInt32(N), k = UInt32(K)
        enc.setBytes(&m, length: 4, index: 3)
        enc.setBytes(&n, length: 4, index: 4)
        enc.setBytes(&k, length: 4, index: 5)
        let numGroupsX = (N + TILE - 1) / TILE
        let numGroupsY = (M + TILE - 1) / TILE
        enc.dispatchThreadgroups(
            MTLSize(width: numGroupsX, height: numGroupsY, depth: 1),
            threadsPerThreadgroup: MTLSize(width: TILE, height: TILE, depth: 1))
        enc.endEncoding(); cmdBuf.commit(); cmdBuf.waitUntilCompleted()
        gpuCheck(cmdBuf, label: "sgemm_tiled_correctness")
    }

    let ptrNaive = bufC_naive.contents().bindMemory(to: Float.self, capacity: M * N)
    let ptrTiled = bufC_tiled.contents().bindMemory(to: Float.self, capacity: M * N)
    let norm = frobNorm(&refC, count: M * N)
    let relNaive = maxError(ptrNaive, &refC, count: M * N) / norm
    let relTiled = maxError(ptrTiled, &refC, count: M * N) / norm

    print("  Naive GPU vs Accelerate:     rel = \(relNaive)  \(relNaive < 1e-5 ? "✓" : "✗")")
    print("  Tiled GPU vs Accelerate:     rel = \(relTiled)  \(relTiled < 1e-5 ? "✓" : "✗")")
    print("  Naive vs Tiled (bit-exact?): max = \(maxError(ptrNaive, ptrTiled, count: M * N))")
}

// ═══════════════════════════════════════════════════════════════════════════
// TEST 2: Non-square, non-TILE-aligned dimensions
// ═══════════════════════════════════════════════════════════════════════════

print("\n── Test 2: Non-aligned dimensions (288×100, K=137) ─────────")

do {
    let M = 288, N = 100, K = 137
    let sizeA = M * K * MemoryLayout<Float>.stride
    let sizeB = K * N * MemoryLayout<Float>.stride
    let sizeC = M * N * MemoryLayout<Float>.stride

    guard let bufA = device.makeBuffer(length: sizeA, options: .storageModeShared),
          let bufB = device.makeBuffer(length: sizeB, options: .storageModeShared),
          let bufC = device.makeBuffer(length: sizeC, options: .storageModeShared) else {
        print("❌ Buffer allocation failed"); exit(1)
    }

    let ptrA = bufA.contents().bindMemory(to: Float.self, capacity: M * K)
    let ptrB = bufB.contents().bindMemory(to: Float.self, capacity: K * N)
    srand48(99)
    for i in 0..<(M * K) { ptrA[i] = Float(drand48() - 0.5) }
    for i in 0..<(K * N) { ptrB[i] = Float(drand48() - 0.5) }
    memset(bufC.contents(), 0, sizeC)

    var refC = [Float](repeating: 0, count: M * N)
    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                Int32(M), Int32(N), Int32(K), 1.0, ptrA, Int32(K), ptrB, Int32(N), 0.0, &refC, Int32(N))

    guard let tiledFunc = library.makeFunction(name: "sgemm_tiled") else { exit(1) }
    let pipeline = try device.makeComputePipelineState(function: tiledFunc)

    autoreleasepool {
        guard let cmdBuf = commandQueue.makeCommandBuffer(),
              let enc = cmdBuf.makeComputeCommandEncoder() else { exit(1) }
        cmdBuf.label = "sgemm_tiled_nonaligned"
        enc.setComputePipelineState(pipeline)
        enc.setBuffer(bufA, offset: 0, index: 0)
        enc.setBuffer(bufB, offset: 0, index: 1)
        enc.setBuffer(bufC, offset: 0, index: 2)
        var m = UInt32(M), n = UInt32(N), k = UInt32(K)
        enc.setBytes(&m, length: 4, index: 3)
        enc.setBytes(&n, length: 4, index: 4)
        enc.setBytes(&k, length: 4, index: 5)
        let numGroupsX = (N + TILE - 1) / TILE
        let numGroupsY = (M + TILE - 1) / TILE
        enc.dispatchThreadgroups(
            MTLSize(width: numGroupsX, height: numGroupsY, depth: 1),
            threadsPerThreadgroup: MTLSize(width: TILE, height: TILE, depth: 1))
        enc.endEncoding(); cmdBuf.commit(); cmdBuf.waitUntilCompleted()
        gpuCheck(cmdBuf, label: "sgemm_tiled_nonaligned")
    }

    let ptrC = bufC.contents().bindMemory(to: Float.self, capacity: M * N)
    let rel = maxError(ptrC, &refC, count: M * N) / frobNorm(&refC, count: M * N)
    print("  Dimensions: \(M)×\(K) × \(K)×\(N) → \(M)×\(N)")
    print("  Tiled GPU vs Accelerate: rel err = \(rel)  \(rel < 1e-5 ? "✓ PASS" : "✗ FAIL")")
}

// ═══════════════════════════════════════════════════════════════════════════
// TEST 3: Performance — GPU timestamps, autoreleasepool, warmup
// ═══════════════════════════════════════════════════════════════════════════

print("\n── Test 3: Performance (naive vs tiled vs Accelerate) ───────")

do {
    guard let naiveFunc = library.makeFunction(name: "sgemm_naive"),
          let tiledFunc = library.makeFunction(name: "sgemm_tiled") else { exit(1) }
    let naivePipeline = try device.makeComputePipelineState(function: naiveFunc)
    let tiledPipeline = try device.makeComputePipelineState(function: tiledFunc)

    let sizes: [(Int, Int)] = [(128, 20), (256, 20), (512, 10), (1024, 5), (2048, 3)]

    print("         M=N=K   Naive (ms)   Tiled (ms)   Accel (ms)   Tiled GFLOP/s   Accel GFLOP/s")
    print("  " + String(repeating: "─", count: 82))

    for (size, reps) in sizes {
        let M = size, N = size, K = size
        let flops = 2.0 * Double(M) * Double(N) * Double(K)
        let sizeA = M * K * MemoryLayout<Float>.stride
        let sizeB = K * N * MemoryLayout<Float>.stride
        let sizeC = M * N * MemoryLayout<Float>.stride

        guard let bufA = device.makeBuffer(length: sizeA, options: .storageModeShared),
              let bufB = device.makeBuffer(length: sizeB, options: .storageModeShared),
              let bufC = device.makeBuffer(length: sizeC, options: .storageModeShared) else { continue }

        let ptrA = bufA.contents().bindMemory(to: Float.self, capacity: M * K)
        let ptrB = bufB.contents().bindMemory(to: Float.self, capacity: K * N)
        for i in 0..<(M * K) { ptrA[i] = 1.0 }
        for i in 0..<(K * N) { ptrB[i] = 1.0 }

        let numGroupsX = (N + TILE - 1) / TILE
        let numGroupsY = (M + TILE - 1) / TILE

        // GPU timing helper using GPU timestamps
        func timeGPU(_ pipeline: MTLComputePipelineState, useThreadgroups: Bool) -> Double {
            // Warmup
            for _ in 0..<2 {
                autoreleasepool {
                    guard let cb = commandQueue.makeCommandBuffer(),
                          let enc = cb.makeComputeCommandEncoder() else { return }
                    enc.setComputePipelineState(pipeline)
                    enc.setBuffer(bufA, offset: 0, index: 0)
                    enc.setBuffer(bufB, offset: 0, index: 1)
                    enc.setBuffer(bufC, offset: 0, index: 2)
                    var m = UInt32(M), n = UInt32(N), k = UInt32(K)
                    enc.setBytes(&m, length: 4, index: 3)
                    enc.setBytes(&n, length: 4, index: 4)
                    enc.setBytes(&k, length: 4, index: 5)
                    if useThreadgroups {
                        enc.dispatchThreadgroups(
                            MTLSize(width: numGroupsX, height: numGroupsY, depth: 1),
                            threadsPerThreadgroup: MTLSize(width: TILE, height: TILE, depth: 1))
                    } else {
                        enc.dispatchThreads(
                            MTLSize(width: N, height: M, depth: 1),
                            threadsPerThreadgroup: MTLSize(width: TILE, height: TILE, depth: 1))
                    }
                    enc.endEncoding(); cb.commit(); cb.waitUntilCompleted()
                }
            }
            // Timed
            var totalGpu: Double = 0
            for _ in 0..<reps {
                autoreleasepool {
                    guard let cb = commandQueue.makeCommandBuffer(),
                          let enc = cb.makeComputeCommandEncoder() else { return }
                    enc.setComputePipelineState(pipeline)
                    enc.setBuffer(bufA, offset: 0, index: 0)
                    enc.setBuffer(bufB, offset: 0, index: 1)
                    enc.setBuffer(bufC, offset: 0, index: 2)
                    var m = UInt32(M), n = UInt32(N), k = UInt32(K)
                    enc.setBytes(&m, length: 4, index: 3)
                    enc.setBytes(&n, length: 4, index: 4)
                    enc.setBytes(&k, length: 4, index: 5)
                    if useThreadgroups {
                        enc.dispatchThreadgroups(
                            MTLSize(width: numGroupsX, height: numGroupsY, depth: 1),
                            threadsPerThreadgroup: MTLSize(width: TILE, height: TILE, depth: 1))
                    } else {
                        enc.dispatchThreads(
                            MTLSize(width: N, height: M, depth: 1),
                            threadsPerThreadgroup: MTLSize(width: TILE, height: TILE, depth: 1))
                    }
                    enc.endEncoding(); cb.commit(); cb.waitUntilCompleted()
                    gpuCheck(cb, label: "sgemm_bench")
                    totalGpu += cb.gpuEndTime - cb.gpuStartTime
                }
            }
            return totalGpu * 1000.0 / Double(reps)
        }

        let naiveMs = timeGPU(naivePipeline, useThreadgroups: false)
        let tiledMs = timeGPU(tiledPipeline, useThreadgroups: true)

        // Accelerate (CPU/AMX) — wall clock is fair here since it's synchronous
        var cpuC = [Float](repeating: 0, count: M * N)
        cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                    Int32(M), Int32(N), Int32(K), 1.0, ptrA, Int32(K), ptrB, Int32(N), 0.0, &cpuC, Int32(N))
        let t0 = CFAbsoluteTimeGetCurrent()
        for _ in 0..<reps {
            cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                        Int32(M), Int32(N), Int32(K), 1.0, ptrA, Int32(K), ptrB, Int32(N), 0.0, &cpuC, Int32(N))
        }
        let accelMs = (CFAbsoluteTimeGetCurrent() - t0) * 1000.0 / Double(reps)

        let tiledGflops = flops / (tiledMs * 1e6)
        let accelGflops = flops / (accelMs * 1e6)

        print(String(format: "  %8d   %9.2f    %9.2f    %9.2f     %10.1f    %10.1f",
                     size, naiveMs, tiledMs, accelMs, tiledGflops, accelGflops))
    }

    print("")
    print("  Notes:")
    print("  - GPU times use gpuStartTime/gpuEndTime (no dispatch overhead)")
    print("  - Accelerate uses the AMX coprocessor (hardware matrix engine)")
    print("  - Our tiled kernel is a teaching example, not a production GEMM")
}

// ═══════════════════════════════════════════════════════════════════════════
// TEST 4: Verify the tiling math with a traceable example
// ═══════════════════════════════════════════════════════════════════════════

print("\n── Test 4: Trace a tiny multiply (4×4) ─────────────────────")

do {
    let M = 4, N = 4, K = 2
    let sizeA = M * K * MemoryLayout<Float>.stride
    let sizeB = K * N * MemoryLayout<Float>.stride
    let sizeC = M * N * MemoryLayout<Float>.stride

    guard let bufA = device.makeBuffer(length: sizeA, options: .storageModeShared),
          let bufB = device.makeBuffer(length: sizeB, options: .storageModeShared),
          let bufC = device.makeBuffer(length: sizeC, options: .storageModeShared) else { exit(1) }

    let ptrA = bufA.contents().bindMemory(to: Float.self, capacity: M * K)
    let ptrB = bufB.contents().bindMemory(to: Float.self, capacity: K * N)
    ptrA[0] = 1; ptrA[1] = 2; ptrA[2] = 3; ptrA[3] = 4
    ptrA[4] = 5; ptrA[5] = 6; ptrA[6] = 7; ptrA[7] = 8
    ptrB[0] = 1; ptrB[1] = 0; ptrB[2] = 1; ptrB[3] = 0
    ptrB[4] = 0; ptrB[5] = 1; ptrB[6] = 0; ptrB[7] = 1
    memset(bufC.contents(), 0, sizeC)

    guard let tiledFunc = library.makeFunction(name: "sgemm_tiled") else { exit(1) }
    let pipeline = try device.makeComputePipelineState(function: tiledFunc)

    autoreleasepool {
        guard let cmdBuf = commandQueue.makeCommandBuffer(),
              let enc = cmdBuf.makeComputeCommandEncoder() else { exit(1) }
        cmdBuf.label = "sgemm_tiny_trace"
        enc.setComputePipelineState(pipeline)
        enc.setBuffer(bufA, offset: 0, index: 0)
        enc.setBuffer(bufB, offset: 0, index: 1)
        enc.setBuffer(bufC, offset: 0, index: 2)
        var m = UInt32(M), n = UInt32(N), k = UInt32(K)
        enc.setBytes(&m, length: 4, index: 3)
        enc.setBytes(&n, length: 4, index: 4)
        enc.setBytes(&k, length: 4, index: 5)
        enc.dispatchThreadgroups(MTLSize(width: 1, height: 1, depth: 1),
                                  threadsPerThreadgroup: MTLSize(width: TILE, height: TILE, depth: 1))
        enc.endEncoding(); cmdBuf.commit(); cmdBuf.waitUntilCompleted()
        gpuCheck(cmdBuf, label: "sgemm_tiny_trace")
    }

    let ptrC = bufC.contents().bindMemory(to: Float.self, capacity: M * N)
    let expected: [Float] = [1,2,1,2, 3,4,3,4, 5,6,5,6, 7,8,7,8]

    var pass = true
    print("  A (4×2):             B (2×4):")
    print("    [1, 2]               [1, 0, 1, 0]")
    print("    [3, 4]               [0, 1, 0, 1]")
    print("    [5, 6]")
    print("    [7, 8]")
    print("")
    print("  C = A×B (4×4):")
    for row in 0..<M {
        var line = "    ["
        for col in 0..<N {
            let val = ptrC[row * N + col]
            let exp = expected[row * N + col]
            if abs(val - exp) > 1e-5 { pass = false }
            line += String(format: "%3.0f", val)
            if col < N - 1 { line += "," }
        }
        line += "]"
        print(line)
    }
    print("  \(pass ? "✓ PASS" : "✗ FAIL")  (matches hand computation)")
}

// ═══════════════════════════════════════════════════════════════════════════
// SUMMARY
// ═══════════════════════════════════════════════════════════════════════════

print("""

═══════════════════════════════════════════════════════════════
  Exercise 3 complete.

  What you learned:
    • 2D thread dispatch — threads arranged in (col, row) grid
    • Tiled multiply — load tiles into threadgroup memory, reuse
    • Two barriers per tile: after load, after compute
    • Why tiling reduces memory traffic by TILE× (16× here)
    • Non-aligned dimensions handled by bounds-checking in the kernel
    • How your GPU compares to Accelerate/AMX
    • GPU timestamp benchmarking for accurate comparisons

  NEXT: Exercise 4 — Complex matrix multiply (ZGEMM).
    That's four of these tiled SGEMMs with the cmul pattern from Ex 1.
═══════════════════════════════════════════════════════════════
""")
