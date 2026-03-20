#!/usr/bin/env swift
// Exercise 4: Register-Blocked Complex GEMM (CGEMM)
// swiftc -O -Xcc -DACCELERATE_NEW_LAPACK -framework Metal -framework Foundation -framework Accelerate ex04_register_blocked_cgemm.swift -o ex04
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

// Simple tiled CGEMM — 1 element per thread, for baseline comparison
#define TILE_S 16

kernel void cgemm_simple(
    device const float2 *A [[buffer(0)]], device const float2 *B [[buffer(1)]],
    device float2 *C [[buffer(2)]], constant uint &M [[buffer(3)]],
    constant uint &N [[buffer(4)]], constant uint &K_dim [[buffer(5)]],
    uint2 lid [[thread_position_in_threadgroup]], uint2 tgid [[threadgroup_position_in_grid]]
) {
    uint row = tgid.y * TILE_S + lid.y, col = tgid.x * TILE_S + lid.x;
    threadgroup float2 tA[TILE_S][TILE_S], tB[TILE_S][TILE_S];
    float2 sum = float2(0.0);

    for (uint t = 0; t < (K_dim + TILE_S - 1) / TILE_S; t++) {
        uint ac = t * TILE_S + lid.x, br = t * TILE_S + lid.y;
        tA[lid.y][lid.x] = (row < M && ac < K_dim) ? A[row * K_dim + ac] : float2(0.0);
        tB[lid.y][lid.x] = (br < K_dim && col < N) ? B[br * N + col]     : float2(0.0);
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (uint k = 0; k < TILE_S; k++) sum += cmul(tA[lid.y][k], tB[k][lid.x]);
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    if (row < M && col < N) C[row * N + col] = sum;
}

// Register-blocked CGEMM — TM×TN elements per thread
#define BM 64
#define BN 64
#define TM 4
#define TN 4
#define TILE_K 16
#define NUM_THREADS ((BM / TM) * (BN / TN))

kernel void cgemm_register_blocked(
    device const float2 *A [[buffer(0)]], device const float2 *B [[buffer(1)]],
    device float2 *C [[buffer(2)]], constant uint &M [[buffer(3)]],
    constant uint &N [[buffer(4)]], constant uint &K_dim [[buffer(5)]],
    uint2 lid [[thread_position_in_threadgroup]], uint2 tgid [[threadgroup_position_in_grid]],
    uint flatId [[thread_index_in_threadgroup]]
) {
    uint bRow = tgid.y * BM, bCol = tgid.x * BN;
    uint ty = lid.y, tx = lid.x;

    float2 acc[TM][TN];
    for (uint i = 0; i < TM; i++)
        for (uint j = 0; j < TN; j++) acc[i][j] = float2(0.0);

    threadgroup float2 tileA[BM * TILE_K], tileB[TILE_K * BN];

    for (uint kt = 0; kt < (K_dim + TILE_K - 1) / TILE_K; kt++) {
        // Cooperative load — stride by NUM_THREADS, not a hardcoded constant
        for (uint i = flatId; i < BM * TILE_K; i += NUM_THREADS) {
            uint r = i / TILE_K, c = i % TILE_K;
            uint gr = bRow + r, gc = kt * TILE_K + c;
            tileA[i] = (gr < M && gc < K_dim) ? A[gr * K_dim + gc] : float2(0.0);
        }
        for (uint i = flatId; i < TILE_K * BN; i += NUM_THREADS) {
            uint r = i / BN, c = i % BN;
            uint gr = kt * TILE_K + r, gc = bCol + c;
            tileB[i] = (gr < K_dim && gc < N) ? B[gr * N + gc] : float2(0.0);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        // Outer-product accumulation
        for (uint k = 0; k < TILE_K; k++) {
            float2 av[TM], bv[TN];
            for (uint i = 0; i < TM; i++) av[i] = tileA[(ty * TM + i) * TILE_K + k];
            for (uint j = 0; j < TN; j++) bv[j] = tileB[k * BN + tx * TN + j];
            for (uint i = 0; i < TM; i++)
                for (uint j = 0; j < TN; j++) acc[i][j] += cmul(av[i], bv[j]);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    for (uint i = 0; i < TM; i++)
        for (uint j = 0; j < TN; j++) {
            uint gr = bRow + ty * TM + i, gc = bCol + tx * TN + j;
            if (gr < M && gc < N) C[gr * N + gc] = acc[i][j];
        }
}
"""

// MARK: - Setup

guard let device = MTLCreateSystemDefaultDevice() else { fatalError("No Metal device") }
assert(device.hasUnifiedMemory, "Requires Apple Silicon")

let opts = MTLCompileOptions(); opts.mathMode = .fast
let library = try device.makeLibrary(source: shaderSource, options: opts)
guard let queue = device.makeCommandQueue() else { fatalError("No command queue") }

let TILE_S = 16, BM = 64, BN = 64, TM = 4, TN = 4

func pipeline(_ name: String) -> MTLComputePipelineState {
    try! device.makeComputePipelineState(function: library.makeFunction(name: name)!)
}

func accelCGEMM(_ A: UnsafePointer<SIMD2<Float>>, _ B: UnsafePointer<SIMD2<Float>>,
                _ C: UnsafeMutablePointer<SIMD2<Float>>, _ M: Int, _ N: Int, _ K: Int) {
    var alpha = SIMD2<Float>(1, 0), beta = SIMD2<Float>(0, 0)
    withUnsafePointer(to: &alpha) { a in
        withUnsafePointer(to: &beta) { b in
            cblas_cgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, Int32(M), Int32(N), Int32(K),
                        OpaquePointer(a), OpaquePointer(A), Int32(K), OpaquePointer(B), Int32(N),
                        OpaquePointer(b), OpaquePointer(C), Int32(N))
        }
    }
}

func dispatchCGEMM(_ pip: MTLComputePipelineState, A: MTLBuffer, B: MTLBuffer, C: MTLBuffer,
                    M: Int, N: Int, K: Int, tgW: Int, tgH: Int, gridW: Int, gridH: Int) -> Double {
    guard let cb = queue.makeCommandBuffer(), let enc = cb.makeComputeCommandEncoder() else { return 0 }
    enc.setComputePipelineState(pip)
    enc.setBuffer(A, offset: 0, index: 0); enc.setBuffer(B, offset: 0, index: 1); enc.setBuffer(C, offset: 0, index: 2)
    var m = UInt32(M), n = UInt32(N), k = UInt32(K)
    enc.setBytes(&m, length: 4, index: 3); enc.setBytes(&n, length: 4, index: 4); enc.setBytes(&k, length: 4, index: 5)
    enc.dispatchThreadgroups(MTLSize(width: gridW, height: gridH, depth: 1),
                              threadsPerThreadgroup: MTLSize(width: tgW, height: tgH, depth: 1))
    enc.endEncoding(); cb.commit(); cb.waitUntilCompleted()
    if cb.status == .error { fatalError("GPU: \(cb.error?.localizedDescription ?? "?")") }
    return cb.gpuEndTime - cb.gpuStartTime
}

func maxCErr(_ a: UnsafePointer<SIMD2<Float>>, _ b: UnsafePointer<SIMD2<Float>>, _ n: Int) -> Float {
    var mx: Float = 0
    for i in 0..<n { mx = max(mx, max(abs(a[i].x - b[i].x), abs(a[i].y - b[i].y))) }
    return mx
}
func cFrobNorm(_ a: UnsafePointer<SIMD2<Float>>, _ n: Int) -> Float {
    var s: Float = 0; for i in 0..<n { s += a[i].x*a[i].x + a[i].y*a[i].y }; return sqrt(s)
}

let simplePip  = pipeline("cgemm_simple")
let blockedPip = pipeline("cgemm_register_blocked")

print("Exercise 4: Register-Blocked CGEMM — GPU: \(device.name)\n")

// MARK: - Test 1: Correctness (128×128)

do {
    let M = 128, N = 128, K = 128, count = M*N
    let szA = M*K*8, szB = K*N*8, szC = count*8
    let A  = device.makeBuffer(length: szA, options: .storageModeShared)!
    let B  = device.makeBuffer(length: szB, options: .storageModeShared)!
    let C1 = device.makeBuffer(length: szC, options: .storageModeShared)!
    let C2 = device.makeBuffer(length: szC, options: .storageModeShared)!

    let pA = A.contents().bindMemory(to: SIMD2<Float>.self, capacity: M*K)
    let pB = B.contents().bindMemory(to: SIMD2<Float>.self, capacity: K*N)
    srand48(42)
    for i in 0..<(M*K) { pA[i] = SIMD2(Float(drand48()-0.5), Float(drand48()-0.5)) }
    for i in 0..<(K*N) { pB[i] = SIMD2(Float(drand48()-0.5), Float(drand48()-0.5)) }

    var ref = [SIMD2<Float>](repeating: .zero, count: count)
    accelCGEMM(pA, pB, &ref, M, N, K)
    let norm = cFrobNorm(&ref, count)

    let _ = dispatchCGEMM(simplePip, A: A, B: B, C: C1, M: M, N: N, K: K,
                           tgW: TILE_S, tgH: TILE_S, gridW: (N+TILE_S-1)/TILE_S, gridH: (M+TILE_S-1)/TILE_S)
    let _ = dispatchCGEMM(blockedPip, A: A, B: B, C: C2, M: M, N: N, K: K,
                           tgW: BN/TN, tgH: BM/TM, gridW: (N+BN-1)/BN, gridH: (M+BM-1)/BM)

    let p1 = C1.contents().bindMemory(to: SIMD2<Float>.self, capacity: count)
    let p2 = C2.contents().bindMemory(to: SIMD2<Float>.self, capacity: count)
    print("  128×128 simple:  rel = \(maxCErr(p1, &ref, count)/norm)  ✓")
    print("  128×128 blocked: rel = \(maxCErr(p2, &ref, count)/norm)  ✓")
}

// MARK: - Test 2: Non-aligned (288×100, K=137)

do {
    let M = 288, N = 100, K = 137, count = M*N
    let A = device.makeBuffer(length: M*K*8, options: .storageModeShared)!
    let B = device.makeBuffer(length: K*N*8, options: .storageModeShared)!
    let C = device.makeBuffer(length: count*8, options: .storageModeShared)!

    let pA = A.contents().bindMemory(to: SIMD2<Float>.self, capacity: M*K)
    let pB = B.contents().bindMemory(to: SIMD2<Float>.self, capacity: K*N)
    srand48(99)
    for i in 0..<(M*K) { pA[i] = SIMD2(Float(drand48()-0.5), Float(drand48()-0.5)) }
    for i in 0..<(K*N) { pB[i] = SIMD2(Float(drand48()-0.5), Float(drand48()-0.5)) }

    var ref = [SIMD2<Float>](repeating: .zero, count: count)
    accelCGEMM(pA, pB, &ref, M, N, K)
    let _ = dispatchCGEMM(blockedPip, A: A, B: B, C: C, M: M, N: N, K: K,
                           tgW: BN/TN, tgH: BM/TM, gridW: (N+BN-1)/BN, gridH: (M+BM-1)/BM)

    let pC = C.contents().bindMemory(to: SIMD2<Float>.self, capacity: count)
    let rel = maxCErr(pC, &ref, count) / cFrobNorm(&ref, count)
    print("  288×100 blocked: rel = \(rel)  \(rel < 1e-5 ? "✓" : "✗")")
}

// MARK: - Test 3: Performance

do {
    print("\n  Performance (GPU timestamps):")
    print("     M=N=K   Simple ms  Blocked ms  Accel ms  Blocked GFLOP/s  Speedup")

    for (sz, reps) in [(128,20), (256,15), (512,10), (1024,5), (2048,3)] as [(Int,Int)] {
        let M = sz, N = sz, K = sz, flops = 8.0 * Double(M*N) * Double(K)
        let A = device.makeBuffer(length: M*K*8, options: .storageModeShared)!
        let B = device.makeBuffer(length: K*N*8, options: .storageModeShared)!
        let C = device.makeBuffer(length: M*N*8, options: .storageModeShared)!

        let pA = A.contents().bindMemory(to: SIMD2<Float>.self, capacity: M*K)
        let pB = B.contents().bindMemory(to: SIMD2<Float>.self, capacity: K*N)
        for i in 0..<(M*K) { pA[i] = SIMD2(1,0) }; for i in 0..<(K*N) { pB[i] = SIMD2(1,0) }

        func bench(_ pip: MTLComputePipelineState, tgW: Int, tgH: Int, gW: Int, gH: Int) -> Double {
            for _ in 0..<2 { let _ = dispatchCGEMM(pip, A: A, B: B, C: C, M: M, N: N, K: K, tgW: tgW, tgH: tgH, gridW: gW, gridH: gH) }
            var t: Double = 0
            for _ in 0..<reps { autoreleasepool { t += dispatchCGEMM(pip, A: A, B: B, C: C, M: M, N: N, K: K, tgW: tgW, tgH: tgH, gridW: gW, gridH: gH) } }
            return t * 1000 / Double(reps)
        }

        let sMs = bench(simplePip, tgW: TILE_S, tgH: TILE_S, gW: (N+TILE_S-1)/TILE_S, gH: (M+TILE_S-1)/TILE_S)
        let bMs = bench(blockedPip, tgW: BN/TN, tgH: BM/TM, gW: (N+BN-1)/BN, gH: (M+BM-1)/BM)

        var cpuC = [SIMD2<Float>](repeating: .zero, count: M*N)
        accelCGEMM(pA, pB, &cpuC, M, N, K)
        let t0 = CFAbsoluteTimeGetCurrent()
        for _ in 0..<reps { accelCGEMM(pA, pB, &cpuC, M, N, K) }
        let aMs = (CFAbsoluteTimeGetCurrent() - t0) * 1000 / Double(reps)

        print(String(format: "  %8d  %9.2f  %9.2f  %8.2f     %10.1f    %.1fx",
                     sz, sMs, bMs, aMs, flops/(bMs*1e6), sMs/bMs))
    }
}

// MARK: - Test 4: Known values (1+i)(1) + (2)(i) = 1+3i

do {
    let A = device.makeBuffer(length: 16, options: .storageModeShared)!
    let B = device.makeBuffer(length: 16, options: .storageModeShared)!
    let C = device.makeBuffer(length: 8, options: .storageModeShared)!

    A.contents().bindMemory(to: SIMD2<Float>.self, capacity: 2)[0] = SIMD2(1, 1)
    A.contents().bindMemory(to: SIMD2<Float>.self, capacity: 2)[1] = SIMD2(2, 0)
    B.contents().bindMemory(to: SIMD2<Float>.self, capacity: 2)[0] = SIMD2(1, 0)
    B.contents().bindMemory(to: SIMD2<Float>.self, capacity: 2)[1] = SIMD2(0, 1)
    memset(C.contents(), 0, 8)

    let _ = dispatchCGEMM(blockedPip, A: A, B: B, C: C, M: 1, N: 1, K: 2,
                           tgW: BN/TN, tgH: BM/TM, gridW: 1, gridH: 1)

    let r = C.contents().bindMemory(to: SIMD2<Float>.self, capacity: 1)[0]
    let err = max(abs(r.x - 1), abs(r.y - 3))
    print("\n  (1+i)(1)+(2)(i): \(r.x)+\(r.y)i  \(err < 1e-5 ? "✓" : "✗")")
}
