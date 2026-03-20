#!/usr/bin/env swift
//
// Exercise 4b: Register Block Size Comparison — 4×4 vs 8×8
//
// COMPILE & RUN:
//   swiftc -O -framework Metal -framework Foundation -framework Accelerate \
//       ex04b_block_size_comparison.swift -o ex04b
//   ./ex04b
//
// Grant Heileman — UNM ECE — 2026
//

import Foundation
import Metal
import Accelerate

let shaderSource = """
#include <metal_stdlib>
using namespace metal;

inline float2 cmul(float2 a, float2 b) {
    return float2(a.x * b.x - a.y * b.y,
                  a.x * b.y + a.y * b.x);
}

// ═══════════════════════════════════════════════════════════════════
// KERNEL A: 4×4 register block
// ═══════════════════════════════════════════════════════════════════

#define BM_4 64
#define BN_4 64
#define TM_4 4
#define TN_4 4
#define TK_4 16
#define NUM_THREADS_4 ((BM_4 / TM_4) * (BN_4 / TN_4))  // = 256

kernel void cgemm_block4x4(
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
    uint blockRowStart = tgid.y * BM_4;
    uint blockColStart = tgid.x * BN_4;
    uint ty = lid.y, tx = lid.x;

    float2 acc[TM_4][TN_4];
    for (uint i = 0; i < TM_4; i++)
        for (uint j = 0; j < TN_4; j++)
            acc[i][j] = float2(0.0);

    threadgroup float2 tileA[BM_4 * TK_4];
    threadgroup float2 tileB[TK_4 * BN_4];

    for (uint kt = 0; kt < (K_dim + TK_4 - 1) / TK_4; kt++) {
        for (uint i = flatId; i < BM_4 * TK_4; i += NUM_THREADS_4) {
            uint r = i / TK_4, c = i % TK_4;
            uint gr = blockRowStart + r, gc = kt * TK_4 + c;
            tileA[i] = (gr < M && gc < K_dim) ? A[gr * K_dim + gc] : float2(0.0);
        }
        for (uint i = flatId; i < TK_4 * BN_4; i += NUM_THREADS_4) {
            uint r = i / BN_4, c = i % BN_4;
            uint gr = kt * TK_4 + r, gc = blockColStart + c;
            tileB[i] = (gr < K_dim && gc < N) ? B[gr * N + gc] : float2(0.0);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        for (uint k = 0; k < TK_4; k++) {
            float2 a_vals[TM_4], b_vals[TN_4];
            for (uint i = 0; i < TM_4; i++)
                a_vals[i] = tileA[(ty * TM_4 + i) * TK_4 + k];
            for (uint j = 0; j < TN_4; j++)
                b_vals[j] = tileB[k * BN_4 + tx * TN_4 + j];
            for (uint i = 0; i < TM_4; i++)
                for (uint j = 0; j < TN_4; j++)
                    acc[i][j] += cmul(a_vals[i], b_vals[j]);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    for (uint i = 0; i < TM_4; i++)
        for (uint j = 0; j < TN_4; j++) {
            uint gr = blockRowStart + ty * TM_4 + i;
            uint gc = blockColStart + tx * TN_4 + j;
            if (gr < M && gc < N) C[gr * N + gc] = acc[i][j];
        }
}

// ═══════════════════════════════════════════════════════════════════
// KERNEL B: 8×8 register block
// ═══════════════════════════════════════════════════════════════════

#define BM_8 128
#define BN_8 128
#define TM_8 8
#define TN_8 8
#define TK_8 8
#define NUM_THREADS_8 ((BM_8 / TM_8) * (BN_8 / TN_8))  // = 256

kernel void cgemm_block8x8(
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
    uint blockRowStart = tgid.y * BM_8;
    uint blockColStart = tgid.x * BN_8;
    uint ty = lid.y, tx = lid.x;

    float2 acc[TM_8][TN_8];
    for (uint i = 0; i < TM_8; i++)
        for (uint j = 0; j < TN_8; j++)
            acc[i][j] = float2(0.0);

    threadgroup float2 tileA[BM_8 * TK_8];
    threadgroup float2 tileB[TK_8 * BN_8];

    uint numKTiles = (K_dim + TK_8 - 1) / TK_8;

    for (uint kt = 0; kt < numKTiles; kt++) {
        for (uint i = flatId; i < BM_8 * TK_8; i += NUM_THREADS_8) {
            uint r = i / TK_8, c = i % TK_8;
            uint gr = blockRowStart + r, gc = kt * TK_8 + c;
            tileA[i] = (gr < M && gc < K_dim) ? A[gr * K_dim + gc] : float2(0.0);
        }
        for (uint i = flatId; i < TK_8 * BN_8; i += NUM_THREADS_8) {
            uint r = i / BN_8, c = i % BN_8;
            uint gr = kt * TK_8 + r, gc = blockColStart + c;
            tileB[i] = (gr < K_dim && gc < N) ? B[gr * N + gc] : float2(0.0);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        for (uint k = 0; k < TK_8; k++) {
            float2 a_vals[TM_8], b_vals[TN_8];
            for (uint i = 0; i < TM_8; i++)
                a_vals[i] = tileA[(ty * TM_8 + i) * TK_8 + k];
            for (uint j = 0; j < TN_8; j++)
                b_vals[j] = tileB[k * BN_8 + tx * TN_8 + j];
            for (uint i = 0; i < TM_8; i++)
                for (uint j = 0; j < TN_8; j++)
                    acc[i][j] += cmul(a_vals[i], b_vals[j]);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    for (uint i = 0; i < TM_8; i++)
        for (uint j = 0; j < TN_8; j++) {
            uint gr = blockRowStart + ty * TM_8 + i;
            uint gc = blockColStart + tx * TN_8 + j;
            if (gr < M && gc < N) C[gr * N + gc] = acc[i][j];
        }
}
"""

// ═══════════════════════════════════════════════════════════════════════════
// HOST CODE
// ═══════════════════════════════════════════════════════════════════════════

print("""

╔═══════════════════════════════════════════════════════════════╗
║  Exercise 4b: Register Block Size — 4×4 vs 8×8 CGEMM        ║
╚═══════════════════════════════════════════════════════════════╝

""")

guard let device = MTLCreateSystemDefaultDevice() else { print("❌ No Metal"); exit(1) }
assert(device.hasUnifiedMemory, "These exercises require Apple Silicon (unified memory)")
print("GPU: \(device.name)")

func gpuCheck(_ cb: MTLCommandBuffer, label: String) {
    if cb.status == .error { print("❌ GPU error [\(label)]: \(cb.error?.localizedDescription ?? "unknown")"); exit(1) }
}

let compileOptions = MTLCompileOptions()
compileOptions.mathMode = .fast

let library: MTLLibrary
do { library = try device.makeLibrary(source: shaderSource, options: compileOptions); print("✓ Shaders compiled\n") }
catch { print("❌ \(error)"); exit(1) }

guard let commandQueue = device.makeCommandQueue() else { exit(1) }

func accelerateCGEMM(A: UnsafePointer<SIMD2<Float>>, B: UnsafePointer<SIMD2<Float>>,
                     C: UnsafeMutablePointer<SIMD2<Float>>, M: Int, N: Int, K: Int) {
    var alpha = SIMD2<Float>(1.0, 0.0)
    var beta = SIMD2<Float>(0.0, 0.0)
    withUnsafePointer(to: &alpha) { a in
        withUnsafePointer(to: &beta) { b in
            cblas_cgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                        Int32(M), Int32(N), Int32(K),
                        OpaquePointer(a), OpaquePointer(A), Int32(K),
                        OpaquePointer(B), Int32(N),
                        OpaquePointer(b), OpaquePointer(C), Int32(N))
        }
    }
}

// ── Correctness check at 256×256 ────────────────────────────────

print("── Correctness (256×256) ────────────────────────────────────")

do {
    let M = 256, N = 256, K = 256, count = M * N
    let sizeA = M * K * MemoryLayout<SIMD2<Float>>.stride
    let sizeB = K * N * MemoryLayout<SIMD2<Float>>.stride
    let sizeC = count * MemoryLayout<SIMD2<Float>>.stride

    guard let bufA = device.makeBuffer(length: sizeA, options: .storageModeShared),
          let bufB = device.makeBuffer(length: sizeB, options: .storageModeShared),
          let bufC4 = device.makeBuffer(length: sizeC, options: .storageModeShared),
          let bufC8 = device.makeBuffer(length: sizeC, options: .storageModeShared) else { exit(1) }

    let ptrA = bufA.contents().bindMemory(to: SIMD2<Float>.self, capacity: M * K)
    let ptrB = bufB.contents().bindMemory(to: SIMD2<Float>.self, capacity: K * N)
    srand48(42)
    for i in 0..<(M*K) { ptrA[i] = SIMD2<Float>(Float(drand48()-0.5), Float(drand48()-0.5)) }
    for i in 0..<(K*N) { ptrB[i] = SIMD2<Float>(Float(drand48()-0.5), Float(drand48()-0.5)) }

    var refC = [SIMD2<Float>](repeating: .zero, count: count)
    accelerateCGEMM(A: ptrA, B: ptrB, C: &refC, M: M, N: N, K: K)
    var norm: Float = 0
    for i in 0..<count { norm += refC[i].x * refC[i].x + refC[i].y * refC[i].y }
    norm = sqrt(norm)

    guard let f4 = library.makeFunction(name: "cgemm_block4x4"),
          let f8 = library.makeFunction(name: "cgemm_block8x8") else { exit(1) }
    let p4 = try device.makeComputePipelineState(function: f4)
    let p8 = try device.makeComputePipelineState(function: f8)

    memset(bufC4.contents(), 0, sizeC)
    autoreleasepool {
        guard let cb = commandQueue.makeCommandBuffer(), let enc = cb.makeComputeCommandEncoder() else { exit(1) }
        cb.label = "cgemm_4x4_correctness"
        enc.setComputePipelineState(p4)
        enc.setBuffer(bufA, offset: 0, index: 0); enc.setBuffer(bufB, offset: 0, index: 1); enc.setBuffer(bufC4, offset: 0, index: 2)
        var m = UInt32(M), n = UInt32(N), k = UInt32(K)
        enc.setBytes(&m, length: 4, index: 3); enc.setBytes(&n, length: 4, index: 4); enc.setBytes(&k, length: 4, index: 5)
        enc.dispatchThreadgroups(MTLSize(width: (N+63)/64, height: (M+63)/64, depth: 1),
                                  threadsPerThreadgroup: MTLSize(width: 16, height: 16, depth: 1))
        enc.endEncoding(); cb.commit(); cb.waitUntilCompleted()
        gpuCheck(cb, label: "cgemm_4x4_correctness")
    }

    memset(bufC8.contents(), 0, sizeC)
    autoreleasepool {
        guard let cb = commandQueue.makeCommandBuffer(), let enc = cb.makeComputeCommandEncoder() else { exit(1) }
        cb.label = "cgemm_8x8_correctness"
        enc.setComputePipelineState(p8)
        enc.setBuffer(bufA, offset: 0, index: 0); enc.setBuffer(bufB, offset: 0, index: 1); enc.setBuffer(bufC8, offset: 0, index: 2)
        var m = UInt32(M), n = UInt32(N), k = UInt32(K)
        enc.setBytes(&m, length: 4, index: 3); enc.setBytes(&n, length: 4, index: 4); enc.setBytes(&k, length: 4, index: 5)
        enc.dispatchThreadgroups(MTLSize(width: (N+127)/128, height: (M+127)/128, depth: 1),
                                  threadsPerThreadgroup: MTLSize(width: 16, height: 16, depth: 1))
        enc.endEncoding(); cb.commit(); cb.waitUntilCompleted()
        gpuCheck(cb, label: "cgemm_8x8_correctness")
    }

    let c4 = bufC4.contents().bindMemory(to: SIMD2<Float>.self, capacity: count)
    let c8 = bufC8.contents().bindMemory(to: SIMD2<Float>.self, capacity: count)
    var err4: Float = 0, err8: Float = 0
    for i in 0..<count {
        err4 = max(err4, max(abs(c4[i].x - refC[i].x), abs(c4[i].y - refC[i].y)))
        err8 = max(err8, max(abs(c8[i].x - refC[i].x), abs(c8[i].y - refC[i].y)))
    }
    print("  4×4 block vs Accelerate: rel err = \(err4/norm)  \(err4/norm < 1e-5 ? "✓" : "✗")")
    print("  8×8 block vs Accelerate: rel err = \(err8/norm)  \(err8/norm < 1e-5 ? "✓" : "✗")")
}

// ── Performance comparison (GPU timestamps) ────────────────────

print("\n── Performance: 4×4 vs 8×8 vs Accelerate (GPU timestamps) ──")

do {
    guard let f4 = library.makeFunction(name: "cgemm_block4x4"),
          let f8 = library.makeFunction(name: "cgemm_block8x8") else { exit(1) }
    let p4 = try device.makeComputePipelineState(function: f4)
    let p8 = try device.makeComputePipelineState(function: f8)

    let sizes: [(Int, Int)] = [(256, 15), (512, 10), (1024, 5), (2048, 3)]

    print("     M=N=K    4×4 GFLOP/s  8×8 GFLOP/s  Accel GFLOP/s  8×8 vs 4×4  8×8 vs Accel")
    print("  " + String(repeating: "─", count: 80))

    for (size, reps) in sizes {
        let M = size, N = size, K = size
        let flops = 8.0 * Double(M) * Double(N) * Double(K)
        let sizeA = M * K * MemoryLayout<SIMD2<Float>>.stride
        let sizeB = K * N * MemoryLayout<SIMD2<Float>>.stride
        let sizeC = M * N * MemoryLayout<SIMD2<Float>>.stride

        guard let bufA = device.makeBuffer(length: sizeA, options: .storageModeShared),
              let bufB = device.makeBuffer(length: sizeB, options: .storageModeShared),
              let bufC = device.makeBuffer(length: sizeC, options: .storageModeShared) else { continue }
        let ptrA = bufA.contents().bindMemory(to: SIMD2<Float>.self, capacity: M*K)
        let ptrB = bufB.contents().bindMemory(to: SIMD2<Float>.self, capacity: K*N)
        for i in 0..<(M*K) { ptrA[i] = SIMD2<Float>(1,0) }
        for i in 0..<(K*N) { ptrB[i] = SIMD2<Float>(1,0) }

        func bench(_ pipeline: MTLComputePipelineState, gW: Int, gH: Int) -> Double {
            for _ in 0..<2 {
                autoreleasepool {
                    guard let cb = commandQueue.makeCommandBuffer(), let e = cb.makeComputeCommandEncoder() else { return }
                    e.setComputePipelineState(pipeline)
                    e.setBuffer(bufA, offset:0, index:0); e.setBuffer(bufB, offset:0, index:1); e.setBuffer(bufC, offset:0, index:2)
                    var m=UInt32(M),n=UInt32(N),k=UInt32(K)
                    e.setBytes(&m,length:4,index:3); e.setBytes(&n,length:4,index:4); e.setBytes(&k,length:4,index:5)
                    e.dispatchThreadgroups(MTLSize(width:gW,height:gH,depth:1), threadsPerThreadgroup:MTLSize(width:16,height:16,depth:1))
                    e.endEncoding(); cb.commit(); cb.waitUntilCompleted()
                }
            }
            var totalGpu: Double = 0
            for _ in 0..<reps {
                autoreleasepool {
                    guard let cb = commandQueue.makeCommandBuffer(), let e = cb.makeComputeCommandEncoder() else { return }
                    e.setComputePipelineState(pipeline)
                    e.setBuffer(bufA, offset:0, index:0); e.setBuffer(bufB, offset:0, index:1); e.setBuffer(bufC, offset:0, index:2)
                    var m=UInt32(M),n=UInt32(N),k=UInt32(K)
                    e.setBytes(&m,length:4,index:3); e.setBytes(&n,length:4,index:4); e.setBytes(&k,length:4,index:5)
                    e.dispatchThreadgroups(MTLSize(width:gW,height:gH,depth:1), threadsPerThreadgroup:MTLSize(width:16,height:16,depth:1))
                    e.endEncoding(); cb.commit(); cb.waitUntilCompleted()
                    gpuCheck(cb, label: "cgemm_bench_\(M)")
                    totalGpu += cb.gpuEndTime - cb.gpuStartTime
                }
            }
            return totalGpu * 1000.0 / Double(reps)
        }

        let ms4 = bench(p4, gW: (N+63)/64, gH: (M+63)/64)
        let ms8 = bench(p8, gW: (N+127)/128, gH: (M+127)/128)

        var cpuC = [SIMD2<Float>](repeating: .zero, count: M*N)
        accelerateCGEMM(A: ptrA, B: ptrB, C: &cpuC, M: M, N: N, K: K)
        let t0 = CFAbsoluteTimeGetCurrent()
        for _ in 0..<reps { accelerateCGEMM(A: ptrA, B: ptrB, C: &cpuC, M: M, N: N, K: K) }
        let msA = (CFAbsoluteTimeGetCurrent() - t0) * 1000.0 / Double(reps)

        let gf4 = flops/(ms4*1e6), gf8 = flops/(ms8*1e6), gfA = flops/(msA*1e6)
        print(String(format: "  %8d   %10.1f  %10.1f   %10.1f      %.2fx        %.2fx",
                     size, gf4, gf8, gfA, gf8/gf4, gf8/gfA))
    }
}

print("""

═══════════════════════════════════════════════════════════════
  Exercise 4b complete.

  Key improvement: cooperative load strides now use NUM_THREADS_4
  and NUM_THREADS_8 derived from block parameters, not hardcoded 256.
  Change BM/TM and the stride auto-adjusts.

  The 8×8 kernel:
    • 64 accumulators per thread (vs 16 for 4×4)
    • 4 cmul per shared memory read (vs 2 for 4×4)
    • 4× more output per threadgroup
═══════════════════════════════════════════════════════════════
""")
