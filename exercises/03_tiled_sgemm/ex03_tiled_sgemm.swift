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
// WHY THIS MATTERS:
//   - ZGEMM (Exercise 4) is four of these with complex arithmetic
//   - This is the #1 operation in QE by FLOP count
//   - The tiling pattern is what ShoYamanishi benchmarked against MPS
//   - Philip Turner's architecture docs tell us: 32 KB threadgroup memory,
//     8 KB L1 cache, so tiling into shared memory is non-optional
//
// PREREQUISITES: Exercise 1 (GPU basics), Exercise 2 (threadgroup memory)
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
//   C = alpha * A * B + beta * C
//   A is M×K, B is K×N, C is M×N
//
// 2D DISPATCH:
//   In Exercises 1-2, each thread had a 1D index (gid).
//   Here, each thread has a 2D position: (row, col) in the output matrix.
//   Metal gives us this via thread_position_in_grid as a uint2.
//
// THE TILING IDEA:
//   Naive: each thread reads an entire row of A and column of B from
//          global memory. For M=N=K=1024, that's 2048 floats per thread,
//          and 1M threads each reading 2048 floats = chaos. The 8 KB L1
//          cache can't hold any of this.
//
//   Tiled: threads cooperate to load a TILE_SIZE×TILE_SIZE chunk of A and B
//          into threadgroup memory (fast, 32 KB). Then each thread multiplies
//          within that tile. Repeat for all tiles along K dimension.
//          Now each float from global memory is read ONCE per threadgroup,
//          not once per thread. That's a TILE_SIZE× reduction in bandwidth.
// ═══════════════════════════════════════════════════════════════════════════

let shaderSource = """
#include <metal_stdlib>
using namespace metal;

// Tile size — each threadgroup processes a TILE×TILE output block.
// 16×16 = 256 threads per threadgroup (matches Exercise 2).
// With 32 KB threadgroup memory: two 16×16 float tiles = 2 KB. Plenty of room.
// We could go to 32×32 (1024 threads, 8 KB for tiles) but 16×16 is clearer
// for learning and still demonstrates the principle.
#define TILE 16

// ─── Kernel 1: Naive SGEMM (no tiling) ───────────────────────────
//
// Each thread computes one element of C by reading a full row of A
// and a full column of B from global memory.
//
// This is CORRECT but SLOW because:
//   - Thread (0,0) reads A[0,0..K-1] and B[0..K-1,0]
//   - Thread (0,1) reads A[0,0..K-1] and B[0..K-1,1]
//   - They both read the SAME row of A, but independently!
//   - With K=1024, that's 4 KB of redundant reads per thread pair
//   - The 8 KB L1 cache evicts data before neighbors can reuse it

kernel void sgemm_naive(
    device const float *A      [[buffer(0)]],   // M×K
    device const float *B      [[buffer(1)]],   // K×N
    device float       *C      [[buffer(2)]],   // M×N
    constant uint      &M      [[buffer(3)]],
    constant uint      &N      [[buffer(4)]],
    constant uint      &K_dim  [[buffer(5)]],
    uint2 gid [[thread_position_in_grid]]        // (col, row) in output
) {
    uint row = gid.y;
    uint col = gid.x;

    if (row >= M || col >= N) return;

    float sum = 0.0;
    for (uint k = 0; k < K_dim; k++) {
        sum += A[row * K_dim + k] * B[k * N + col];
        // ^ Every thread does K reads from A and K reads from B.
        //   No data sharing between threads at all.
    }

    C[row * N + col] = sum;
}

// ─── Kernel 2: Tiled SGEMM ──────────────────────────────────────
//
// The key insight: threads in a threadgroup SHARE data through
// threadgroup memory. Instead of each thread reading its own
// row/column from global memory, the threadgroup cooperatively
// loads one tile of A and one tile of B into shared memory,
// then ALL 256 threads multiply from that shared copy.
//
// For TILE=16 and K=1024:
//   - 1024/16 = 64 tiles along the K dimension
//   - Each tile load: 16×16 = 256 floats = 1 KB from A, 1 KB from B
//   - Each thread does 16 multiply-adds per tile (not 1024)
//   - Total global reads per threadgroup: 64 × 2 KB = 128 KB
//   - Naive equivalent: 256 threads × 2×4 KB = 2 MB
//   - That's 16× less memory traffic! (= TILE factor)
//
// Memory traffic formula:
//   Naive:  2 × M × N × K × sizeof(float)  total reads
//   Tiled:  2 × M × N × K × sizeof(float) / TILE  total reads
//
// This is why Philip Turner's docs emphasize the tiny L1 (8 KB):
// without tiling, your working set is K×sizeof(float) per thread,
// which exceeds L1 for any useful K. With tiling, your working set
// is TILE×TILE×sizeof(float) = 1 KB, which fits easily.

kernel void sgemm_tiled(
    device const float *A      [[buffer(0)]],   // M×K
    device const float *B      [[buffer(1)]],   // K×N
    device float       *C      [[buffer(2)]],   // M×N
    constant uint      &M      [[buffer(3)]],
    constant uint      &N      [[buffer(4)]],
    constant uint      &K_dim  [[buffer(5)]],

    uint2 gid  [[thread_position_in_grid]],          // global (col, row)
    uint2 lid  [[thread_position_in_threadgroup]],    // local within tile
    uint2 tgid [[threadgroup_position_in_grid]]       // which threadgroup
) {
    // ── Which element of C does this thread compute? ─────────────
    uint row = tgid.y * TILE + lid.y;   // global row
    uint col = tgid.x * TILE + lid.x;   // global col

    // ── Declare shared memory for one tile of A and one tile of B ─
    // These live in the 32 KB threadgroup SRAM.
    // All 256 threads (16×16) in this threadgroup can read/write these.
    threadgroup float tileA[TILE][TILE];
    threadgroup float tileB[TILE][TILE];

    // ── Accumulator for this thread's output element ─────────────
    // Lives in a register (private to this thread, fastest possible).
    float sum = 0.0;

    // ── Loop over tiles along the K dimension ────────────────────
    // If K=1024 and TILE=16, this loops 64 times.
    uint numTiles = (K_dim + TILE - 1) / TILE;

    for (uint t = 0; t < numTiles; t++) {

        // ── Cooperative load: each thread loads ONE element of each tile ──
        //
        // Thread (lid.x, lid.y) loads:
        //   tileA[lid.y][lid.x] = A[row][t*TILE + lid.x]
        //   tileB[lid.y][lid.x] = B[t*TILE + lid.y][col]
        //
        // After all 256 threads execute this, the entire 16×16 tile
        // is in shared memory. One global read per thread per tile,
        // not one per multiply.

        uint a_col = t * TILE + lid.x;
        uint b_row = t * TILE + lid.y;

        // Bounds check: K might not be a multiple of TILE
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

        // ── BARRIER: wait for ALL threads to finish loading ──────
        // Without this, thread (0,0) might start multiplying before
        // thread (15,15) has written its element to tileB.
        // Same pattern as Exercise 2's reduction.
        threadgroup_barrier(mem_flags::mem_threadgroup);

        // ── Multiply-accumulate within the tile ─────────────────
        // This is the inner loop — entirely from shared memory.
        // 16 iterations, 2 reads from shared memory each = 32 reads.
        // Shared memory bandwidth: ~same speed as registers on Apple GPU.
        for (uint k = 0; k < TILE; k++) {
            sum += tileA[lid.y][k] * tileB[k][lid.x];
            //     ↑ row of A tile    ↑ column of B tile
        }

        // ── BARRIER: don't start loading next tile until multiply is done ──
        // Without this, the next iteration's cooperative load could
        // overwrite tileA/tileB while some threads are still reading.
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    // ── Write result to global memory ────────────────────────────
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
print("GPU: \(device.name)")

let library: MTLLibrary
do {
    library = try device.makeLibrary(source: shaderSource, options: nil)
    print("✓ Shaders compiled\n")
} catch {
    print("❌ Shader compilation failed: \(error)"); exit(1)
}

guard let commandQueue = device.makeCommandQueue() else {
    print("❌ Command queue failed"); exit(1)
}

// Helper: compute max absolute error between two float arrays
func maxError(_ a: UnsafePointer<Float>, _ b: UnsafePointer<Float>, count: Int) -> Float {
    var mx: Float = 0
    for i in 0..<count { mx = max(mx, abs(a[i] - b[i])) }
    return mx
}

// Helper: compute Frobenius norm
func frobNorm(_ a: UnsafePointer<Float>, count: Int) -> Float {
    var s: Float = 0
    for i in 0..<count { s += a[i] * a[i] }
    return sqrt(s)
}

let TILE = 16  // must match the #define TILE in the shader

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

    // Fill with small random values
    srand48(42)
    for i in 0..<(M * K) { ptrA[i] = Float(drand48() - 0.5) }
    for i in 0..<(K * N) { ptrB[i] = Float(drand48() - 0.5) }

    // Zero output buffers
    memset(bufC_naive.contents(), 0, sizeC)
    memset(bufC_tiled.contents(), 0, sizeC)

    // ── CPU reference via Accelerate ────────────────────────────
    var refC = [Float](repeating: 0, count: M * N)
    // cblas_sgemm: C = alpha*A*B + beta*C
    cblas_sgemm(
        CblasRowMajor, CblasNoTrans, CblasNoTrans,
        Int32(M), Int32(N), Int32(K),
        1.0,        // alpha
        ptrA, Int32(K),
        ptrB, Int32(N),
        0.0,        // beta
        &refC, Int32(N)
    )

    // ── GPU: Naive kernel ───────────────────────────────────────
    guard let naiveFunc = library.makeFunction(name: "sgemm_naive") else {
        print("❌ Kernel not found"); exit(1)
    }
    let naivePipeline = try device.makeComputePipelineState(function: naiveFunc)

    do {
        guard let cmdBuf = commandQueue.makeCommandBuffer(),
              let enc = cmdBuf.makeComputeCommandEncoder() else {
            print("❌ Encoder failed"); exit(1)
        }

        enc.setComputePipelineState(naivePipeline)
        enc.setBuffer(bufA, offset: 0, index: 0)
        enc.setBuffer(bufB, offset: 0, index: 1)
        enc.setBuffer(bufC_naive, offset: 0, index: 2)
        var m = UInt32(M), n = UInt32(N), k = UInt32(K)
        enc.setBytes(&m, length: 4, index: 3)
        enc.setBytes(&n, length: 4, index: 4)
        enc.setBytes(&k, length: 4, index: 5)

        // 2D dispatch: N threads wide, M threads tall
        // Each thread computes one element of C
        enc.dispatchThreads(
            MTLSize(width: N, height: M, depth: 1),
            threadsPerThreadgroup: MTLSize(width: TILE, height: TILE, depth: 1)
        )
        enc.endEncoding()
        cmdBuf.commit()
        cmdBuf.waitUntilCompleted()
    }

    // ── GPU: Tiled kernel ───────────────────────────────────────
    guard let tiledFunc = library.makeFunction(name: "sgemm_tiled") else {
        print("❌ Kernel not found"); exit(1)
    }
    let tiledPipeline = try device.makeComputePipelineState(function: tiledFunc)

    do {
        guard let cmdBuf = commandQueue.makeCommandBuffer(),
              let enc = cmdBuf.makeComputeCommandEncoder() else {
            print("❌ Encoder failed"); exit(1)
        }

        enc.setComputePipelineState(tiledPipeline)
        enc.setBuffer(bufA, offset: 0, index: 0)
        enc.setBuffer(bufB, offset: 0, index: 1)
        enc.setBuffer(bufC_tiled, offset: 0, index: 2)
        var m = UInt32(M), n = UInt32(N), k = UInt32(K)
        enc.setBytes(&m, length: 4, index: 3)
        enc.setBytes(&n, length: 4, index: 4)
        enc.setBytes(&k, length: 4, index: 5)

        // For the tiled kernel, we dispatch threadgroups, not individual threads.
        // Each threadgroup is TILE×TILE threads and computes a TILE×TILE output block.
        let numGroupsX = (N + TILE - 1) / TILE
        let numGroupsY = (M + TILE - 1) / TILE
        enc.dispatchThreadgroups(
            MTLSize(width: numGroupsX, height: numGroupsY, depth: 1),
            threadsPerThreadgroup: MTLSize(width: TILE, height: TILE, depth: 1)
        )
        enc.endEncoding()
        cmdBuf.commit()
        cmdBuf.waitUntilCompleted()
    }

    // ── Compare results ─────────────────────────────────────────
    let ptrNaive = bufC_naive.contents().bindMemory(to: Float.self, capacity: M * N)
    let ptrTiled = bufC_tiled.contents().bindMemory(to: Float.self, capacity: M * N)

    let errNaive = maxError(ptrNaive, &refC, count: M * N)
    let errTiled = maxError(ptrTiled, &refC, count: M * N)
    let norm = frobNorm(&refC, count: M * N)
    let relNaive = errNaive / norm
    let relTiled = errTiled / norm

    let passNaive = relNaive < 1e-5
    let passTiled = relTiled < 1e-5

    print("  Accelerate (CPU reference):  Frobenius norm = \(norm)")
    print("  Naive GPU vs Accelerate:     max err = \(errNaive), rel = \(relNaive)  \(passNaive ? "✓" : "✗")")
    print("  Tiled GPU vs Accelerate:     max err = \(errTiled), rel = \(relTiled)  \(passTiled ? "✓" : "✗")")

    // Naive vs Tiled should be identical (same math, same data)
    let errNvsT = maxError(ptrNaive, ptrTiled, count: M * N)
    print("  Naive vs Tiled (bit-exact?): max err = \(errNvsT)")
}

// ═══════════════════════════════════════════════════════════════════════════
// TEST 2: Non-square, non-TILE-aligned dimensions
//
// QE matrices are rarely nice multiples of 16. The tiled kernel must
// handle M=288, N=100, K=4477 — none divisible by 16.
// ═══════════════════════════════════════════════════════════════════════════

print("\n── Test 2: Non-aligned dimensions (288×100, K=137) ─────────")

do {
    let M = 288, N = 100, K = 137  // deliberately awkward

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

    // CPU reference
    var refC = [Float](repeating: 0, count: M * N)
    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                Int32(M), Int32(N), Int32(K),
                1.0, ptrA, Int32(K), ptrB, Int32(N), 0.0, &refC, Int32(N))

    // GPU tiled
    guard let tiledFunc = library.makeFunction(name: "sgemm_tiled") else {
        print("❌ Kernel not found"); exit(1)
    }
    let pipeline = try device.makeComputePipelineState(function: tiledFunc)

    guard let cmdBuf = commandQueue.makeCommandBuffer(),
          let enc = cmdBuf.makeComputeCommandEncoder() else {
        print("❌ Encoder failed"); exit(1)
    }

    enc.setComputePipelineState(pipeline)
    enc.setBuffer(bufA, offset: 0, index: 0)
    enc.setBuffer(bufB, offset: 0, index: 1)
    enc.setBuffer(bufC, offset: 0, index: 2)
    var m = UInt32(M), n = UInt32(N), k = UInt32(K)
    enc.setBytes(&m, length: 4, index: 3)
    enc.setBytes(&n, length: 4, index: 4)
    enc.setBytes(&k, length: 4, index: 5)

    let numGroupsX = (N + TILE - 1) / TILE   // ceil(100/16) = 7
    let numGroupsY = (M + TILE - 1) / TILE   // ceil(288/16) = 18
    enc.dispatchThreadgroups(
        MTLSize(width: numGroupsX, height: numGroupsY, depth: 1),
        threadsPerThreadgroup: MTLSize(width: TILE, height: TILE, depth: 1)
    )
    enc.endEncoding()
    cmdBuf.commit()
    cmdBuf.waitUntilCompleted()

    let ptrC = bufC.contents().bindMemory(to: Float.self, capacity: M * N)
    let err = maxError(ptrC, &refC, count: M * N)
    let norm = frobNorm(&refC, count: M * N)
    let rel = err / norm

    print("  Dimensions: \(M)×\(K) × \(K)×\(N) → \(M)×\(N)")
    print("  Threadgroups: \(numGroupsX) × \(numGroupsY) = \(numGroupsX * numGroupsY)")
    print("  Tiled GPU vs Accelerate: rel err = \(rel)  \(rel < 1e-5 ? "✓ PASS" : "✗ FAIL")")
}

// ═══════════════════════════════════════════════════════════════════════════
// TEST 3: Performance — naive vs tiled vs Accelerate
//
// This is where you SEE why tiling matters.
// At M=N=K=1024, the tiled kernel should significantly outperform naive.
// Accelerate uses the AMX coprocessor, so it's a high bar.
// ═══════════════════════════════════════════════════════════════════════════

print("\n── Test 3: Performance (naive vs tiled vs Accelerate) ───────")

do {
    guard let naiveFunc = library.makeFunction(name: "sgemm_naive"),
          let tiledFunc = library.makeFunction(name: "sgemm_tiled") else {
        print("❌ Kernels not found"); exit(1)
    }
    let naivePipeline = try device.makeComputePipelineState(function: naiveFunc)
    let tiledPipeline = try device.makeComputePipelineState(function: tiledFunc)

    let sizes: [(Int, Int)] = [
        (128, 20),    // small: dispatch overhead dominates
        (256, 20),
        (512, 10),
        (1024, 5),    // medium: tiling should win
        (2048, 3),    // large: bandwidth-bound
    ]

    print("         M=N=K   Naive (ms)   Tiled (ms)   Accel (ms)   Tiled GFLOP/s   Accel GFLOP/s")
    print("  " + String(repeating: "─", count: 82))

    for (size, reps) in sizes {
        let M = size, N = size, K = size
        let flops = 2.0 * Double(M) * Double(N) * Double(K)  // 2MNK for GEMM

        let sizeA = M * K * MemoryLayout<Float>.stride
        let sizeB = K * N * MemoryLayout<Float>.stride
        let sizeC = M * N * MemoryLayout<Float>.stride

        guard let bufA = device.makeBuffer(length: sizeA, options: .storageModeShared),
              let bufB = device.makeBuffer(length: sizeB, options: .storageModeShared),
              let bufC = device.makeBuffer(length: sizeC, options: .storageModeShared) else {
            continue
        }

        let ptrA = bufA.contents().bindMemory(to: Float.self, capacity: M * K)
        let ptrB = bufB.contents().bindMemory(to: Float.self, capacity: K * N)

        for i in 0..<(M * K) { ptrA[i] = 1.0 }
        for i in 0..<(K * N) { ptrB[i] = 1.0 }

        let numGroupsX = (N + TILE - 1) / TILE
        let numGroupsY = (M + TILE - 1) / TILE

        // Helper to time a GPU kernel
        func timeGPU(_ pipeline: MTLComputePipelineState, useThreadgroups: Bool) -> Double {
            // Warmup
            for _ in 0..<2 {
                guard let cb = commandQueue.makeCommandBuffer(),
                      let enc = cb.makeComputeCommandEncoder() else { return -1 }
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
                enc.endEncoding()
                cb.commit()
                cb.waitUntilCompleted()
            }

            // Timed
            let t0 = CFAbsoluteTimeGetCurrent()
            for _ in 0..<reps {
                guard let cb = commandQueue.makeCommandBuffer(),
                      let enc = cb.makeComputeCommandEncoder() else { return -1 }
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
                enc.endEncoding()
                cb.commit()
                cb.waitUntilCompleted()
            }
            return (CFAbsoluteTimeGetCurrent() - t0) * 1000.0 / Double(reps)
        }

        let naiveMs = timeGPU(naivePipeline, useThreadgroups: false)
        let tiledMs = timeGPU(tiledPipeline, useThreadgroups: true)

        // Accelerate (CPU/AMX)
        var cpuC = [Float](repeating: 0, count: M * N)
        // Warmup
        cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                    Int32(M), Int32(N), Int32(K),
                    1.0, ptrA, Int32(K), ptrB, Int32(N), 0.0, &cpuC, Int32(N))

        let t0 = CFAbsoluteTimeGetCurrent()
        for _ in 0..<reps {
            cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                        Int32(M), Int32(N), Int32(K),
                        1.0, ptrA, Int32(K), ptrB, Int32(N), 0.0, &cpuC, Int32(N))
        }
        let accelMs = (CFAbsoluteTimeGetCurrent() - t0) * 1000.0 / Double(reps)

        let tiledGflops = flops / (tiledMs * 1e6)
        let accelGflops = flops / (accelMs * 1e6)

        print(String(format: "  %8d   %9.2f    %9.2f    %9.2f     %10.1f    %10.1f",
                     size, naiveMs, tiledMs, accelMs, tiledGflops, accelGflops))
    }

    print("")
    print("  Notes:")
    print("  - Accelerate uses the AMX coprocessor (hardware matrix engine)")
    print("  - Our tiled kernel is a teaching example, not a production GEMM")
    print("  - MPS SGEMM would be ~2-3 TFLOPS on your M2 Max")
    print("  - Production kernels use larger tiles, register blocking, vectorized loads")
}

// ═══════════════════════════════════════════════════════════════════════════
// TEST 4: Verify the tiling math with a traceable example
// ═══════════════════════════════════════════════════════════════════════════

print("\n── Test 4: Trace a tiny multiply (4×4) ─────────────────────")

do {
    // A = [[1,2],[3,4],[5,6],[7,8]]  (4×2)
    // B = [[1,0,1,0],[0,1,0,1]]      (2×4)
    // C = A×B = [[1,2,1,2],[3,4,3,4],[5,6,5,6],[7,8,7,8]]  (4×4)
    let M = 4, N = 4, K = 2

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

    // A: 4×2
    ptrA[0] = 1; ptrA[1] = 2
    ptrA[2] = 3; ptrA[3] = 4
    ptrA[4] = 5; ptrA[5] = 6
    ptrA[6] = 7; ptrA[7] = 8

    // B: 2×4
    ptrB[0] = 1; ptrB[1] = 0; ptrB[2] = 1; ptrB[3] = 0
    ptrB[4] = 0; ptrB[5] = 1; ptrB[6] = 0; ptrB[7] = 1

    memset(bufC.contents(), 0, sizeC)

    guard let tiledFunc = library.makeFunction(name: "sgemm_tiled") else {
        print("❌ Kernel not found"); exit(1)
    }
    let pipeline = try device.makeComputePipelineState(function: tiledFunc)

    guard let cmdBuf = commandQueue.makeCommandBuffer(),
          let enc = cmdBuf.makeComputeCommandEncoder() else {
        print("❌ Encoder failed"); exit(1)
    }

    enc.setComputePipelineState(pipeline)
    enc.setBuffer(bufA, offset: 0, index: 0)
    enc.setBuffer(bufB, offset: 0, index: 1)
    enc.setBuffer(bufC, offset: 0, index: 2)
    var m = UInt32(M), n = UInt32(N), k = UInt32(K)
    enc.setBytes(&m, length: 4, index: 3)
    enc.setBytes(&n, length: 4, index: 4)
    enc.setBytes(&k, length: 4, index: 5)

    enc.dispatchThreadgroups(
        MTLSize(width: 1, height: 1, depth: 1),  // 1 threadgroup covers 16×16, more than enough
        threadsPerThreadgroup: MTLSize(width: TILE, height: TILE, depth: 1)
    )
    enc.endEncoding()
    cmdBuf.commit()
    cmdBuf.waitUntilCompleted()

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

  What the numbers mean:
    • Naive kernel is slow because EVERY thread reads from global memory
    • Tiled kernel is faster because threads SHARE loads via shared memory
    • Accelerate is still faster because AMX is dedicated matrix hardware
      and Apple has optimized it for years
    • Our kernel is a 16×16 teaching example — production kernels use
      32×32 tiles, register blocking (each thread does 4×4 or 8×8),
      and vectorized memory loads

  The precision connection:
    • This exercise used FP32 (float) — that's where GPU throughput is
    • QE needs FP64 for energy convergence — route those to AMX
    • Exercise 4 builds ZGEMM: four SGEMM calls with complex arithmetic

  Things to try:
    • Change TILE from 16 to 8. Does correctness hold? Performance?
    • Change TILE to 32 — what happens to threadgroup memory usage?
      (Two 32×32 float tiles = 8 KB, still fits in 32 KB)
    • Add alpha and beta parameters: C = alpha*A*B + beta*C
      (This is what cblas_sgemm actually computes)
    • Can you beat Accelerate at any matrix size? (Hint: probably
      not with this simple kernel, but you'll learn where the gaps are)

  NEXT: Exercise 4 — Complex matrix multiply (ZGEMM).
    That's four of these tiled SGEMMs with the cmul pattern from Ex 1.
═══════════════════════════════════════════════════════════════
""")
