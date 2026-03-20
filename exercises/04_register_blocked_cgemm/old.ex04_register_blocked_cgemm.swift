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
// THE KEY INSIGHT:
//   Exercise 3: 256 threads, each does 1 output element = 256 outputs/threadgroup
//   This exercise: 256 threads, each does 4×4 = 16 elements = 4096 outputs/threadgroup
//   Same thread count, 16× more work, same shared memory loads.
//   That's free arithmetic intensity — the path to peak FLOPS.
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

// TILE_K: how many K elements we load into shared memory per pass.
// BM, BN:  output block size per threadgroup.
// TM, TN:  sub-tile size per thread (register block).
//
// Threadgroup layout:
//   BM/TM × BN/TN threads = 16×16 = 256 threads
//   Each thread computes a TM×TN (4×4) block of output.
//   Total output per threadgroup = BM × BN = 64×64 = 4096 elements.
//
// Shared memory usage:
//   tileA: BM × TILE_K float2 = 64 × 16 × 8 bytes = 8 KB
//   tileB: TILE_K × BN float2 = 16 × 64 × 8 bytes = 8 KB
//   Total: 16 KB (fits in 32 KB threadgroup memory)

#define BM 64       // output rows per threadgroup
#define BN 64       // output cols per threadgroup
#define TM 4        // output rows per thread (register block height)
#define TN 4        // output cols per thread (register block width)
#define TILE_K 16   // K elements per shared-memory pass

// Threads per threadgroup: (BM/TM) × (BN/TN) = 16 × 16 = 256
#define THREADS_Y (BM / TM)   // 16
#define THREADS_X (BN / TN)   // 16

// ─── Kernel 1: Simple tiled CGEMM (1 element per thread) ─────────
//
// This is Exercise 3's tiled kernel adapted for complex numbers.
// Each thread computes ONE float2 output. Familiar pattern.
// Included for correctness comparison and to show the performance gap.

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
// THE PRODUCTION PATTERN.
//
// Each thread computes a TM×TN = 4×4 sub-tile of output.
// That's 16 float2 accumulators living in registers.
//
// Why this is faster:
//   1. Each shared memory load is reused TM (or TN) times
//      across the sub-tile rows/columns.
//   2. Arithmetic intensity goes up: more FMA per byte loaded.
//   3. The GPU's ALUs stay busy because there's enough independent
//      work per thread to hide memory latency.
//
// Thread mapping:
//   Thread (tx, ty) in a 16×16 threadgroup computes:
//     C[ty*TM + 0..3][tx*TN + 0..3]  (a 4×4 block of the 64×64 output)
//
// Shared memory layout:
//   tileA[BM][TILE_K]  — BM rows, TILE_K columns of A
//   tileB[TILE_K][BN]  — TILE_K rows, BN columns of B
//
// Loading shared memory:
//   256 threads need to fill BM×TILE_K = 64×16 = 1024 elements of A
//   and TILE_K×BN = 16×64 = 1024 elements of B.
//   1024 / 256 = 4 loads per thread for each tile.
//   We use a flat index and stride to distribute the loads evenly.

kernel void cgemm_register_blocked(
    device const float2 *A     [[buffer(0)]],
    device const float2 *B     [[buffer(1)]],
    device float2       *C     [[buffer(2)]],
    constant uint       &M     [[buffer(3)]],
    constant uint       &N     [[buffer(4)]],
    constant uint       &K_dim [[buffer(5)]],
    uint2 lid   [[thread_position_in_threadgroup]],
    uint2 tgid  [[threadgroup_position_in_grid]],
    uint  flatId [[thread_index_in_threadgroup]]     // 0..255
) {
    // ── Where in the output does this threadgroup write? ─────────
    uint blockRowStart = tgid.y * BM;   // first output row for this threadgroup
    uint blockColStart = tgid.x * BN;   // first output col for this threadgroup

    // ── Where in the sub-tile does this thread write? ────────────
    //    tx, ty index within the 16×16 thread grid
    uint ty = lid.y;   // 0..15
    uint tx = lid.x;   // 0..15

    // ── Accumulators: TM × TN = 4 × 4 = 16 float2 values ───────
    //    These live entirely in registers — fastest possible storage.
    //    This is the "register block" that gives the technique its name.
    float2 acc[TM][TN];
    for (uint i = 0; i < TM; i++)
        for (uint j = 0; j < TN; j++)
            acc[i][j] = float2(0.0);

    // ── Shared memory tiles ─────────────────────────────────────
    threadgroup float2 tileA[BM * TILE_K];     // 64 × 16 = 1024 float2
    threadgroup float2 tileB[TILE_K * BN];     // 16 × 64 = 1024 float2

    uint numKTiles = (K_dim + TILE_K - 1) / TILE_K;

    for (uint kt = 0; kt < numKTiles; kt++) {

        // ── Cooperative load of tileA ────────────────────────────
        //    1024 elements, 256 threads → 4 loads per thread.
        //    flatId strides by 256 to cover all 1024 elements.
        for (uint i = flatId; i < BM * TILE_K; i += 256) {
            uint tileRow = i / TILE_K;           // which row in the BM×TILE_K tile
            uint tileCol = i % TILE_K;           // which col
            uint globalRow = blockRowStart + tileRow;
            uint globalCol = kt * TILE_K + tileCol;

            tileA[i] = (globalRow < M && globalCol < K_dim)
                ? A[globalRow * K_dim + globalCol]
                : float2(0.0);
        }

        // ── Cooperative load of tileB ────────────────────────────
        for (uint i = flatId; i < TILE_K * BN; i += 256) {
            uint tileRow = i / BN;
            uint tileCol = i % BN;
            uint globalRow = kt * TILE_K + tileRow;
            uint globalCol = blockColStart + tileCol;

            tileB[i] = (globalRow < K_dim && globalCol < N)
                ? B[globalRow * N + globalCol]
                : float2(0.0);
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);

        // ── Register-blocked multiply-accumulate ────────────────
        //
        //    For each k in the TILE_K dimension:
        //      Load TM values from tileA (this thread's rows)
        //      Load TN values from tileB (this thread's cols)
        //      Outer product: acc[i][j] += cmul(a[i], b[j])
        //
        //    That's TM + TN = 8 shared memory reads
        //    producing TM × TN = 16 multiply-adds.
        //    Arithmetic intensity: 16 cmul / 8 loads = 2 cmul per load.
        //    (The simple kernel does 1 cmul per 2 loads = 0.5)
        //    That's 4× better arithmetic intensity.

        for (uint k = 0; k < TILE_K; k++) {
            // Load TM elements from A tile for this thread's rows
            float2 a_vals[TM];
            for (uint i = 0; i < TM; i++) {
                uint aRow = ty * TM + i;        // local row within BM
                a_vals[i] = tileA[aRow * TILE_K + k];
            }

            // Load TN elements from B tile for this thread's cols
            float2 b_vals[TN];
            for (uint j = 0; j < TN; j++) {
                uint bCol = tx * TN + j;        // local col within BN
                b_vals[j] = tileB[k * BN + bCol];
            }

            // Outer product into accumulators
            for (uint i = 0; i < TM; i++) {
                for (uint j = 0; j < TN; j++) {
                    acc[i][j] += cmul(a_vals[i], b_vals[j]);
                }
            }
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    // ── Write TM×TN results to global memory ────────────────────
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

// Helpers
func maxComplexError(_ a: UnsafePointer<SIMD2<Float>>, _ b: UnsafePointer<SIMD2<Float>>, count: Int) -> Float {
    var mx: Float = 0
    for i in 0..<count {
        let err = max(abs(a[i].x - b[i].x), abs(a[i].y - b[i].y))
        mx = max(mx, err)
    }
    return mx
}

func complexFrobNorm(_ a: UnsafePointer<SIMD2<Float>>, count: Int) -> Float {
    var s: Float = 0
    for i in 0..<count { s += a[i].x * a[i].x + a[i].y * a[i].y }
    return sqrt(s)
}

// Accelerate reference — using cblas_cgemm (single-precision complex GEMM)
// DSPComplex has the same memory layout as SIMD2<Float>: [real, imag] pairs.
func accelerateCGEMM(A: UnsafePointer<SIMD2<Float>>, B: UnsafePointer<SIMD2<Float>>,
                     C: UnsafeMutablePointer<SIMD2<Float>>, M: Int, N: Int, K: Int) {
    var alpha: [Float] = [1.0, 0.0]  // complex 1+0i
    var beta: [Float] = [0.0, 0.0]   // complex 0+0i
    // cblas_cgemm expects pointers to interleaved float pairs
    cblas_cgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                Int32(M), Int32(N), Int32(K),
                &alpha,
                OpaquePointer(A), Int32(K),
                OpaquePointer(B), Int32(N),
                &beta,
                OpaquePointer(C), Int32(N))
}

let TILE_SIMPLE = 16
let BM = 64
let BN = 64
let TM = 4
let TN = 4

// ═══════════════════════════════════════════════════════════════════════════
// TEST 1: Correctness — both kernels vs Accelerate
// ═══════════════════════════════════════════════════════════════════════════

print("── Test 1: Correctness (128×128, both kernels) ──────────────")

do {
    let M = 128, N = 128, K = 128
    let count = M * N
    let sizeA = M * K * MemoryLayout<SIMD2<Float>>.stride
    let sizeB = K * N * MemoryLayout<SIMD2<Float>>.stride
    let sizeC = count * MemoryLayout<SIMD2<Float>>.stride

    guard let bufA  = device.makeBuffer(length: sizeA, options: .storageModeShared),
          let bufB  = device.makeBuffer(length: sizeB, options: .storageModeShared),
          let bufC1 = device.makeBuffer(length: sizeC, options: .storageModeShared),
          let bufC2 = device.makeBuffer(length: sizeC, options: .storageModeShared) else {
        print("❌ Buffer allocation failed"); exit(1)
    }

    let ptrA = bufA.contents().bindMemory(to: SIMD2<Float>.self, capacity: M * K)
    let ptrB = bufB.contents().bindMemory(to: SIMD2<Float>.self, capacity: K * N)

    srand48(42)
    for i in 0..<(M * K) { ptrA[i] = SIMD2<Float>(Float(drand48() - 0.5), Float(drand48() - 0.5)) }
    for i in 0..<(K * N) { ptrB[i] = SIMD2<Float>(Float(drand48() - 0.5), Float(drand48() - 0.5)) }
    memset(bufC1.contents(), 0, sizeC)
    memset(bufC2.contents(), 0, sizeC)

    // CPU reference
    var refC = [SIMD2<Float>](repeating: .zero, count: count)
    accelerateCGEMM(A: ptrA, B: ptrB, C: &refC, M: M, N: N, K: K)
    let norm = complexFrobNorm(&refC, count: count)

    // Simple tiled kernel
    guard let simpleFunc = library.makeFunction(name: "cgemm_simple") else {
        print("❌ cgemm_simple not found"); exit(1)
    }
    let simplePipeline = try device.makeComputePipelineState(function: simpleFunc)

    do {
        guard let cb = commandQueue.makeCommandBuffer(),
              let enc = cb.makeComputeCommandEncoder() else { fatalError() }
        enc.setComputePipelineState(simplePipeline)
        enc.setBuffer(bufA, offset: 0, index: 0)
        enc.setBuffer(bufB, offset: 0, index: 1)
        enc.setBuffer(bufC1, offset: 0, index: 2)
        var m = UInt32(M), n = UInt32(N), k = UInt32(K)
        enc.setBytes(&m, length: 4, index: 3)
        enc.setBytes(&n, length: 4, index: 4)
        enc.setBytes(&k, length: 4, index: 5)
        enc.dispatchThreadgroups(
            MTLSize(width: (N + TILE_SIMPLE - 1) / TILE_SIMPLE,
                    height: (M + TILE_SIMPLE - 1) / TILE_SIMPLE, depth: 1),
            threadsPerThreadgroup: MTLSize(width: TILE_SIMPLE, height: TILE_SIMPLE, depth: 1))
        enc.endEncoding(); cb.commit(); cb.waitUntilCompleted()
    }

    // Register-blocked kernel
    guard let blockedFunc = library.makeFunction(name: "cgemm_register_blocked") else {
        print("❌ cgemm_register_blocked not found"); exit(1)
    }
    let blockedPipeline = try device.makeComputePipelineState(function: blockedFunc)

    do {
        guard let cb = commandQueue.makeCommandBuffer(),
              let enc = cb.makeComputeCommandEncoder() else { fatalError() }
        enc.setComputePipelineState(blockedPipeline)
        enc.setBuffer(bufA, offset: 0, index: 0)
        enc.setBuffer(bufB, offset: 0, index: 1)
        enc.setBuffer(bufC2, offset: 0, index: 2)
        var m = UInt32(M), n = UInt32(N), k = UInt32(K)
        enc.setBytes(&m, length: 4, index: 3)
        enc.setBytes(&n, length: 4, index: 4)
        enc.setBytes(&k, length: 4, index: 5)
        enc.dispatchThreadgroups(
            MTLSize(width: (N + BN - 1) / BN, height: (M + BM - 1) / BM, depth: 1),
            threadsPerThreadgroup: MTLSize(width: BN / TN, height: BM / TM, depth: 1))
        enc.endEncoding(); cb.commit(); cb.waitUntilCompleted()
    }

    let ptrC1 = bufC1.contents().bindMemory(to: SIMD2<Float>.self, capacity: count)
    let ptrC2 = bufC2.contents().bindMemory(to: SIMD2<Float>.self, capacity: count)

    let errSimple = maxComplexError(ptrC1, &refC, count: count) / norm
    let errBlocked = maxComplexError(ptrC2, &refC, count: count) / norm

    print("  Frobenius norm:       \(norm)")
    print("  Simple vs Accelerate: rel err = \(errSimple)  \(errSimple < 1e-5 ? "✓" : "✗")")
    print("  Blocked vs Accelerate: rel err = \(errBlocked)  \(errBlocked < 1e-5 ? "✓" : "✗")")
    print("  Simple vs Blocked:    rel err = \(maxComplexError(ptrC1, ptrC2, count: count) / norm)")
}

// ═══════════════════════════════════════════════════════════════════════════
// TEST 2: Non-aligned QE-like dimensions
// ═══════════════════════════════════════════════════════════════════════════

print("\n── Test 2: Non-aligned QE dimensions (288×100, K=137) ───────")

do {
    let M = 288, N = 100, K = 137
    let count = M * N
    let sizeA = M * K * MemoryLayout<SIMD2<Float>>.stride
    let sizeB = K * N * MemoryLayout<SIMD2<Float>>.stride
    let sizeC = count * MemoryLayout<SIMD2<Float>>.stride

    guard let bufA = device.makeBuffer(length: sizeA, options: .storageModeShared),
          let bufB = device.makeBuffer(length: sizeB, options: .storageModeShared),
          let bufC = device.makeBuffer(length: sizeC, options: .storageModeShared) else {
        print("❌ Buffer allocation failed"); exit(1)
    }

    let ptrA = bufA.contents().bindMemory(to: SIMD2<Float>.self, capacity: M * K)
    let ptrB = bufB.contents().bindMemory(to: SIMD2<Float>.self, capacity: K * N)
    srand48(99)
    for i in 0..<(M * K) { ptrA[i] = SIMD2<Float>(Float(drand48() - 0.5), Float(drand48() - 0.5)) }
    for i in 0..<(K * N) { ptrB[i] = SIMD2<Float>(Float(drand48() - 0.5), Float(drand48() - 0.5)) }
    memset(bufC.contents(), 0, sizeC)

    var refC = [SIMD2<Float>](repeating: .zero, count: count)
    accelerateCGEMM(A: ptrA, B: ptrB, C: &refC, M: M, N: N, K: K)
    let norm = complexFrobNorm(&refC, count: count)

    guard let func2 = library.makeFunction(name: "cgemm_register_blocked") else {
        print("❌ Kernel not found"); exit(1)
    }
    let pipeline = try device.makeComputePipelineState(function: func2)

    guard let cb = commandQueue.makeCommandBuffer(),
          let enc = cb.makeComputeCommandEncoder() else { fatalError() }
    enc.setComputePipelineState(pipeline)
    enc.setBuffer(bufA, offset: 0, index: 0)
    enc.setBuffer(bufB, offset: 0, index: 1)
    enc.setBuffer(bufC, offset: 0, index: 2)
    var m = UInt32(M), n = UInt32(N), k = UInt32(K)
    enc.setBytes(&m, length: 4, index: 3)
    enc.setBytes(&n, length: 4, index: 4)
    enc.setBytes(&k, length: 4, index: 5)
    enc.dispatchThreadgroups(
        MTLSize(width: (N + BN - 1) / BN, height: (M + BM - 1) / BM, depth: 1),
        threadsPerThreadgroup: MTLSize(width: BN / TN, height: BM / TM, depth: 1))
    enc.endEncoding(); cb.commit(); cb.waitUntilCompleted()

    let ptrC = bufC.contents().bindMemory(to: SIMD2<Float>.self, capacity: count)
    let rel = maxComplexError(ptrC, &refC, count: count) / norm
    print("  Dimensions: \(M)×\(K) × \(K)×\(N) → \(M)×\(N)")
    print("  Register-blocked vs Accelerate: rel err = \(rel)  \(rel < 1e-5 ? "✓ PASS" : "✗ FAIL")")
}

// ═══════════════════════════════════════════════════════════════════════════
// TEST 3: Performance — simple vs register-blocked vs Accelerate
//
// Complex GEMM FLOP count: 8*M*N*K (4 real muls + 4 real adds per element)
// ═══════════════════════════════════════════════════════════════════════════

print("\n── Test 3: Performance (simple vs register-blocked vs Accelerate) ──")

do {
    guard let simpleFunc = library.makeFunction(name: "cgemm_simple"),
          let blockedFunc = library.makeFunction(name: "cgemm_register_blocked") else {
        print("❌ Kernels not found"); exit(1)
    }
    let simplePipeline = try device.makeComputePipelineState(function: simpleFunc)
    let blockedPipeline = try device.makeComputePipelineState(function: blockedFunc)

    let sizes: [(Int, Int)] = [
        (128, 20), (256, 15), (512, 10), (1024, 5), (2048, 3),
    ]

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
                enc.dispatchThreadgroups(
                    MTLSize(width: gridW, height: gridH, depth: 1),
                    threadsPerThreadgroup: MTLSize(width: tgWidth, height: tgHeight, depth: 1))
                enc.endEncoding(); cb.commit(); cb.waitUntilCompleted()
            }
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
                enc.dispatchThreadgroups(
                    MTLSize(width: gridW, height: gridH, depth: 1),
                    threadsPerThreadgroup: MTLSize(width: tgWidth, height: tgHeight, depth: 1))
                enc.endEncoding(); cb.commit(); cb.waitUntilCompleted()
            }
            return (CFAbsoluteTimeGetCurrent() - t0) * 1000.0 / Double(reps)
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

    print("")
    print("  The 'Speedup' column shows register-blocked vs simple tiled.")
    print("  This is the gain from each thread doing 4×4 elements instead of 1.")
    print("")
    print("  To go further toward 2-3 TFLOPS complex:")
    print("    • Increase to TM=TN=8 (8×8 sub-tile, 64 accumulators per thread)")
    print("    • Use float4 loads for coalesced memory access")
    print("    • Explore simdgroup_matrix for hardware-accelerated tile multiply")
}

// ═══════════════════════════════════════════════════════════════════════════
// TEST 4: Verify the complex arithmetic is correct with known values
// ═══════════════════════════════════════════════════════════════════════════

print("\n── Test 4: Known complex multiply ───────────────────────────")

do {
    // A = [[1+i, 2+0i]]    (1×2)
    // B = [[1+0i],          (2×1)
    //      [0+i ]]
    // C = A×B = (1+i)(1) + (2)(0+i) = (1+i) + (2i) = 1 + 3i
    let M = 1, N = 1, K = 2
    let sizeA = M * K * MemoryLayout<SIMD2<Float>>.stride
    let sizeB = K * N * MemoryLayout<SIMD2<Float>>.stride
    let sizeC = M * N * MemoryLayout<SIMD2<Float>>.stride

    guard let bufA = device.makeBuffer(length: sizeA, options: .storageModeShared),
          let bufB = device.makeBuffer(length: sizeB, options: .storageModeShared),
          let bufC = device.makeBuffer(length: sizeC, options: .storageModeShared) else {
        print("❌ Buffer allocation failed"); exit(1)
    }

    let ptrA = bufA.contents().bindMemory(to: SIMD2<Float>.self, capacity: M * K)
    let ptrB = bufB.contents().bindMemory(to: SIMD2<Float>.self, capacity: K * N)
    ptrA[0] = SIMD2<Float>(1, 1)   // 1+i
    ptrA[1] = SIMD2<Float>(2, 0)   // 2+0i
    ptrB[0] = SIMD2<Float>(1, 0)   // 1+0i
    ptrB[1] = SIMD2<Float>(0, 1)   // 0+i
    memset(bufC.contents(), 0, sizeC)

    guard let func2 = library.makeFunction(name: "cgemm_register_blocked") else {
        print("❌ Kernel not found"); exit(1)
    }
    let pipeline = try device.makeComputePipelineState(function: func2)
    guard let cb = commandQueue.makeCommandBuffer(),
          let enc = cb.makeComputeCommandEncoder() else { fatalError() }
    enc.setComputePipelineState(pipeline)
    enc.setBuffer(bufA, offset: 0, index: 0)
    enc.setBuffer(bufB, offset: 0, index: 1)
    enc.setBuffer(bufC, offset: 0, index: 2)
    var m = UInt32(M), n = UInt32(N), k = UInt32(K)
    enc.setBytes(&m, length: 4, index: 3)
    enc.setBytes(&n, length: 4, index: 4)
    enc.setBytes(&k, length: 4, index: 5)
    enc.dispatchThreadgroups(
        MTLSize(width: 1, height: 1, depth: 1),
        threadsPerThreadgroup: MTLSize(width: BN / TN, height: BM / TM, depth: 1))
    enc.endEncoding(); cb.commit(); cb.waitUntilCompleted()

    let ptrC = bufC.contents().bindMemory(to: SIMD2<Float>.self, capacity: 1)
    let result = ptrC[0]
    let expected = SIMD2<Float>(1, 3)
    let err = max(abs(result.x - expected.x), abs(result.y - expected.y))

    print("  A = [(1+i), (2+0i)]   B = [(1+0i), (0+i)]^T")
    print("  C = (1+i)(1) + (2)(i) = 1 + i + 2i = 1 + 3i")
    print("  GPU result: \(result.x) + \(result.y)i  \(err < 1e-5 ? "✓ PASS" : "✗ FAIL")")
}

print("""

═══════════════════════════════════════════════════════════════
  Exercise 4 complete.

  What you learned:
    • Register blocking: each thread computes TM×TN output elements
    • 4× arithmetic intensity over the simple tiled kernel
    • The outer-product pattern: load a[TM] and b[TN] from shared
      memory, produce TM×TN multiply-adds (the inner loop)
    • Cooperative loading with flat index striding
    • Complex GEMM = tiled SGEMM + cmul

  The path to peak performance:
    This kernel is the SKELETON of a production GEMM.
    To reach 2-3 TFLOPS on your M2 Max:
    ┌─────────────────────────────────────────────────────────┐
    │ 1. Increase TM×TN from 4×4 to 8×8                      │
    │    → 64 accumulators per thread, 4× more work per load  │
    │ 2. Use float4 (128-bit) loads for coalesced access      │
    │    → halves the number of load instructions              │
    │ 3. Explore simdgroup_matrix (Apple's tensor-core equiv)  │
    │    → hardware-accelerated 8×8 matrix multiply            │
    │ 4. Double-buffer shared memory (ping-pong)               │
    │    → overlap loading next tile with computing current    │
    └─────────────────────────────────────────────────────────┘
    Each step is a MODIFICATION of this kernel, not a rewrite.

  NEXT: Exercise 5 — 1D FFT (Stockham radix-2).
    That's the other half of QE's compute: 50% of wall time.
═══════════════════════════════════════════════════════════════
""")
