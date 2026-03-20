#!/usr/bin/env swift
//
// Exercise 2: Complex Dot Product — Parallel Reduction on GPU
//
// WHAT YOU'LL LEARN:
//   - threadgroup memory (Metal's equivalent of CUDA shared memory)
//   - Parallel reduction: how 1024 threads cooperate to produce 1 number
//   - Why you can't just "sum everything in one kernel" on a GPU
//   - The ZDOTC pattern: dot(A,B) = Σ conj(A[i]) * B[i]
//
// WHY THIS MATTERS:
//   - Tiled GEMM loads tiles into threadgroup memory (same mechanism)
//   - FFT butterfly passes use threadgroup memory for data exchange
//   - QE computes overlap matrices via ZDOTC-like operations
//
// PREREQUISITE: Exercise 1 (cmul, cconj, cadd)
//
// COMPILE & RUN:
//   swiftc -O -framework Metal -framework Foundation -framework Accelerate \
//       ex02_complex_dot_product.swift -o ex02
//   ./ex02
//
// Grant Heileman — UNM ECE — 2026
//

import Foundation
import Metal
import Accelerate  // for reference ZDOTC to compare against

// ═══════════════════════════════════════════════════════════════════════════
// THE METAL SHADERS
//
// Read this BEFORE the Swift code. The key new concept is:
//
//   threadgroup float2 shared[THREADGROUP_SIZE];
//
// This declares memory that is:
//   - Shared among all threads in ONE threadgroup (not the whole grid)
//   - Fast (on-chip SRAM, ~32 KB on M-series, similar to L1 cache speed)
//   - Gone when the threadgroup finishes (not persistent)
//
// Why do we need it?
//
// A dot product is a SUM over all elements. But GPU threads run in
// parallel — thread 0 can't see thread 1's partial result unless they
// share memory. The pattern is:
//
//   1. Each thread computes one product: conj(A[i]) * B[i]
//   2. Store that product in shared[local_index]
//   3. Barrier (threadgroup_barrier) — wait for ALL threads to finish step 2
//   4. Reduce: thread 0 adds shared[0]+shared[1], thread 1 adds shared[2]+shared[3], etc.
//   5. Barrier again
//   6. Repeat halving until one value remains
//   7. Thread 0 writes the threadgroup's partial sum to global memory
//
// The host then sums the per-threadgroup partial results (small array, CPU is fine).
// ═══════════════════════════════════════════════════════════════════════════

let shaderSource = """
#include <metal_stdlib>
using namespace metal;

// ─── Complex helpers (same as Exercise 1) ─────────────────────────

inline float2 cmul(float2 z1, float2 z2) {
    return float2(z1.x * z2.x - z1.y * z2.y,
                  z1.x * z2.y + z1.y * z2.x);
}

inline float2 cconj(float2 z) {
    return float2(z.x, -z.y);
}

// ─── Kernel 1: Naive dot product (NO shared memory) ───────────────
//
// Each thread computes conj(A[i]) * B[i] and atomically adds to result.
//
// THIS IS THE WRONG WAY. It works but it's slow because:
//   - atomic_fetch_add serializes all 1024 threads onto one memory location
//   - Metal doesn't have atomic float add (only int), so we'd need tricks
//
// We include it so you can see WHY reduction is necessary.
// (We actually skip running this one — it's here for your reading.)
//
// kernel void zdotc_naive(...) {
//     // Can't do this efficiently — no atomic float add in Metal.
//     // This is why we need parallel reduction.
// }

// ─── Kernel 2: Parallel reduction with threadgroup memory ─────────
//
// THIS IS THE RIGHT WAY.
//
// Arguments:
//   A, B:        input arrays of complex numbers
//   partials:    output array — one partial sum per threadgroup
//   count:       total number of elements
//
// Thread indexing:
//   gid    = global thread index (0 to count-1)
//   lid    = local index within this threadgroup (0 to THREADGROUP_SIZE-1)
//   tgid   = which threadgroup this is (0, 1, 2, ...)

kernel void zdotc_reduce(
    device const float2 *A         [[buffer(0)]],
    device const float2 *B         [[buffer(1)]],
    device float2       *partials  [[buffer(2)]],   // one result per threadgroup
    constant uint       &count     [[buffer(3)]],

    uint gid  [[thread_position_in_grid]],          // global index
    uint lid  [[thread_position_in_threadgroup]],    // local index (0..255)
    uint tgid [[threadgroup_position_in_grid]],      // which threadgroup
    uint tg_size [[threads_per_threadgroup]]          // threadgroup size (e.g. 256)
) {
    // ── Step 1: Each thread computes its element's contribution ────
    //
    // If gid is out of bounds (padding threads), contribute zero.

    float2 product = float2(0.0, 0.0);
    if (gid < count) {
        product = cmul(cconj(A[gid]), B[gid]);   // conj(A[i]) * B[i]
    }

    // ── Step 2: Store into threadgroup shared memory ──────────────
    //
    // This array lives in fast on-chip SRAM. Every thread in this
    // threadgroup can read/write it, but threads in OTHER threadgroups
    // cannot see it.
    //
    // The size must be known at compile time (or use setThreadgroupMemoryLength
    // from the host). Here we use 256, matching our dispatch.

    threadgroup float2 shared_data[256];
    shared_data[lid] = product;

    // ── Step 3: BARRIER ──────────────────────────────────────────
    //
    // This is critical. Without it, thread 0 might read shared_data[1]
    // before thread 1 has written it. The barrier forces ALL threads
    // in this threadgroup to reach this point before ANY can proceed.
    //
    // mem_flags::mem_threadgroup means "make sure threadgroup memory
    // writes are visible." This is the same concept as your LoopFDTD
    // data race fix — but at the threadgroup level instead of CPU/GPU.

    threadgroup_barrier(mem_flags::mem_threadgroup);

    // ── Step 4: Tree reduction ───────────────────────────────────
    //
    // We halve the active threads each iteration:
    //
    //   Iteration 1 (stride=128):
    //     Thread 0:   shared[0] += shared[128]
    //     Thread 1:   shared[1] += shared[129]
    //     ...
    //     Thread 127: shared[127] += shared[255]
    //     (Threads 128-255 do nothing)
    //
    //   Iteration 2 (stride=64):
    //     Thread 0:   shared[0] += shared[64]
    //     Thread 1:   shared[1] += shared[65]
    //     ...
    //     Thread 63:  shared[63] += shared[127]
    //     (Threads 64-255 do nothing)
    //
    //   ... 8 iterations total for 256 threads ...
    //
    //   Final (stride=1):
    //     Thread 0:   shared[0] += shared[1]
    //     (Only thread 0 active)
    //
    // After this, shared[0] holds the sum of all 256 elements.

    for (uint stride = tg_size / 2; stride > 0; stride >>= 1) {
        if (lid < stride) {
            shared_data[lid] = shared_data[lid] + shared_data[lid + stride];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);  // barrier EVERY iteration
    }

    // ── Step 5: Thread 0 writes this threadgroup's partial sum ────
    //
    // Only thread 0 has the final value in shared_data[0].

    if (lid == 0) {
        partials[tgid] = shared_data[0];
    }
}

// ─── Kernel 3: ZAXPY — y[i] = y[i] + alpha * x[i] ───────────────
//
// Bonus kernel. This is the other fundamental BLAS-1 operation
// that QE uses constantly (wavefunction updates during Davidson).
// No reduction needed — purely element-wise.

kernel void zaxpy(
    device const float2 *x     [[buffer(0)]],
    device float2       *y     [[buffer(1)]],   // in-place update
    constant uint       &count [[buffer(2)]],
    constant float2     &alpha [[buffer(3)]],
    uint gid                   [[thread_position_in_grid]]
) {
    if (gid >= count) return;
    y[gid] = y[gid] + cmul(alpha, x[gid]);
}
"""

// ═══════════════════════════════════════════════════════════════════════════
// SWIFT HOST CODE
// ═══════════════════════════════════════════════════════════════════════════

print("""

╔═══════════════════════════════════════════════════════════════╗
║  Exercise 2: Complex Dot Product — Parallel Reduction        ║
╚═══════════════════════════════════════════════════════════════╝

""")

// ── Metal setup ─────────────────────────────────────────────────────────

guard let device = MTLCreateSystemDefaultDevice() else {
    print("❌ No Metal device"); exit(1)
}
print("GPU: \(device.name)")
print("Max threadgroup size: \(device.maxThreadsPerThreadgroup.width)")

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

// ═══════════════════════════════════════════════════════════════════════════
// TEST 1: Small dot product — trace the reduction by hand
//
// We use N=8 so you can follow every step of the reduction.
// With threadgroup size = 8, there's exactly 1 threadgroup,
// so the host doesn't need to sum partials.
// ═══════════════════════════════════════════════════════════════════════════

print("── Test 1: Small ZDOTC (N=8, trace the reduction) ──────────")

do {
    let N = 8
    let threadgroupSize = 8  // intentionally small so 1 threadgroup covers all
    let numThreadgroups = (N + threadgroupSize - 1) / threadgroupSize  // = 1

    let bufferSize = N * MemoryLayout<SIMD2<Float>>.stride
    let partialSize = numThreadgroups * MemoryLayout<SIMD2<Float>>.stride

    guard let bufA = device.makeBuffer(length: bufferSize, options: .storageModeShared),
          let bufB = device.makeBuffer(length: bufferSize, options: .storageModeShared),
          let bufP = device.makeBuffer(length: partialSize, options: .storageModeShared) else {
        print("❌ Buffer allocation failed"); exit(1)
    }

    let ptrA = bufA.contents().bindMemory(to: SIMD2<Float>.self, capacity: N)
    let ptrB = bufB.contents().bindMemory(to: SIMD2<Float>.self, capacity: N)

    // A = [(1,0), (0,1), (1,1), (2,0), (0,2), (1,-1), (3,0), (0,0)]
    // B = [(1,0), (1,0), (1,0), (1,0), (1,0), (1,0),  (1,0), (1,0)]
    // B is all real 1's, so dot = Σ conj(A[i]) * 1 = Σ conj(A[i])
    let aValues: [SIMD2<Float>] = [
        SIMD2(1, 0), SIMD2(0, 1), SIMD2(1, 1), SIMD2(2, 0),
        SIMD2(0, 2), SIMD2(1,-1), SIMD2(3, 0), SIMD2(0, 0)
    ]
    for i in 0..<N {
        ptrA[i] = aValues[i]
        ptrB[i] = SIMD2<Float>(1.0, 0.0)  // all ones (real)
    }

    // CPU reference: Σ conj(A[i]) = (1,0)+(0,-1)+(1,-1)+(2,0)+(0,-2)+(1,1)+(3,0)+(0,0)
    // Real sum: 1+0+1+2+0+1+3+0 = 8
    // Imag sum: 0+(-1)+(-1)+0+(-2)+1+0+0 = -3
    let expectedReal: Float = 8.0
    let expectedImag: Float = -3.0

    print("  A = \(aValues.map { "(\($0.x),\($0.y))" }.joined(separator: ", "))")
    print("  B = all (1, 0)")
    print("  Expected: Σ conj(A[i]) × B[i] = (\(expectedReal), \(expectedImag))")
    print("")

    // ── Dispatch ────────────────────────────────────────────────────

    guard let function = library.makeFunction(name: "zdotc_reduce") else {
        print("❌ Kernel not found"); exit(1)
    }
    let pipeline = try device.makeComputePipelineState(function: function)

    guard let cmdBuf = commandQueue.makeCommandBuffer(),
          let encoder = cmdBuf.makeComputeCommandEncoder() else {
        print("❌ Encoder failed"); exit(1)
    }

    encoder.setComputePipelineState(pipeline)
    encoder.setBuffer(bufA, offset: 0, index: 0)
    encoder.setBuffer(bufB, offset: 0, index: 1)
    encoder.setBuffer(bufP, offset: 0, index: 2)
    var count = UInt32(N)
    encoder.setBytes(&count, length: MemoryLayout<UInt32>.size, index: 3)

    // Dispatch exactly 1 threadgroup of 8 threads
    encoder.dispatchThreadgroups(
        MTLSize(width: numThreadgroups, height: 1, depth: 1),
        threadsPerThreadgroup: MTLSize(width: threadgroupSize, height: 1, depth: 1)
    )

    encoder.endEncoding()
    cmdBuf.commit()
    cmdBuf.waitUntilCompleted()

    // Read back the single partial sum
    let ptrP = bufP.contents().bindMemory(to: SIMD2<Float>.self, capacity: numThreadgroups)
    let result = ptrP[0]

    let errReal = abs(result.x - expectedReal)
    let errImag = abs(result.y - expectedImag)

    if errReal < 1e-5 && errImag < 1e-5 {
        print("  ✓ PASS  result = (\(result.x), \(result.y))")
    } else {
        print("  ✗ FAIL  result = (\(result.x), \(result.y))  error: (\(errReal), \(errImag))")
    }

    // Show the reduction tree
    print("")
    print("  What happened inside the GPU (1 threadgroup, 8 threads):")
    print("    Step 0: each thread computes conj(A[i]) × B[i]")
    print("    shared = [(1,0), (0,-1), (1,-1), (2,0), (0,-2), (1,1), (3,0), (0,0)]")
    print("")
    print("    stride=4: threads 0-3 add shared[i] + shared[i+4]")
    print("    shared = [(1,-2), (1,0), (4,-1), (2,0), ...]")
    print("")
    print("    stride=2: threads 0-1 add shared[i] + shared[i+2]")
    print("    shared = [(5,-3), (3,0), ...]")
    print("")
    print("    stride=1: thread 0 adds shared[0] + shared[1]")
    print("    shared = [(8,-3), ...]")
    print("")
    print("    Thread 0 writes partials[0] = (8, -3)  ✓")
}

// ═══════════════════════════════════════════════════════════════════════════
// TEST 2: Large dot product — multiple threadgroups, verify against Accelerate
//
// This is the realistic case. N=65536 elements, threadgroup size 256.
// That's 256 threadgroups, each producing one partial sum.
// The host sums those 256 partials on CPU (trivial).
// ═══════════════════════════════════════════════════════════════════════════

print("\n── Test 2: Large ZDOTC (N=65536, vs Accelerate) ─────────────")

do {
    let N = 65536
    let threadgroupSize = 256
    let numThreadgroups = (N + threadgroupSize - 1) / threadgroupSize  // = 256

    let bufferSize = N * MemoryLayout<SIMD2<Float>>.stride
    let partialSize = numThreadgroups * MemoryLayout<SIMD2<Float>>.stride

    guard let bufA = device.makeBuffer(length: bufferSize, options: .storageModeShared),
          let bufB = device.makeBuffer(length: bufferSize, options: .storageModeShared),
          let bufP = device.makeBuffer(length: partialSize, options: .storageModeShared) else {
        print("❌ Buffer allocation failed"); exit(1)
    }

    let ptrA = bufA.contents().bindMemory(to: SIMD2<Float>.self, capacity: N)
    let ptrB = bufB.contents().bindMemory(to: SIMD2<Float>.self, capacity: N)

    // Fill with known pattern: A[i] = (i, i+1), B[i] = (1, 0.5)
    for i in 0..<N {
        ptrA[i] = SIMD2<Float>(Float(i), Float(i + 1))
        ptrB[i] = SIMD2<Float>(1.0, 0.5)
    }

    // ── CPU reference using Accelerate's ZDOTC ─────────────────────
    //
    // cblas_zdotc_sub computes conj(X) · Y
    // We need to convert our SIMD2<Float> arrays to the format Accelerate expects.
    // Accelerate's ZDOTC uses double precision interleaved [real, imag, real, imag, ...]

    var refResult = SIMD2<Float>(0, 0)
    for i in 0..<N {
        let fi = Float(i)
        // conj(A[i]) = (i, -(i+1))
        // conj(A[i]) * B[i] = (i, -(i+1)) * (1, 0.5)
        //   real = i*1 - (-(i+1))*0.5 = i + 0.5*(i+1) = i + 0.5i + 0.5 = 1.5i + 0.5
        //   imag = i*0.5 + (-(i+1))*1 = 0.5i - i - 1 = -0.5i - 1
        refResult.x += 1.5 * fi + 0.5
        refResult.y += -0.5 * fi - 1.0
    }

    print("  N = \(N),  threadgroups = \(numThreadgroups) × \(threadgroupSize) threads")
    print("  CPU reference: (\(refResult.x), \(refResult.y))")

    // ── GPU dispatch ───────────────────────────────────────────────

    guard let function = library.makeFunction(name: "zdotc_reduce") else {
        print("❌ Kernel not found"); exit(1)
    }
    let pipeline = try device.makeComputePipelineState(function: function)

    guard let cmdBuf = commandQueue.makeCommandBuffer(),
          let encoder = cmdBuf.makeComputeCommandEncoder() else {
        print("❌ Encoder failed"); exit(1)
    }

    encoder.setComputePipelineState(pipeline)
    encoder.setBuffer(bufA, offset: 0, index: 0)
    encoder.setBuffer(bufB, offset: 0, index: 1)
    encoder.setBuffer(bufP, offset: 0, index: 2)
    var count = UInt32(N)
    encoder.setBytes(&count, length: MemoryLayout<UInt32>.size, index: 3)

    encoder.dispatchThreadgroups(
        MTLSize(width: numThreadgroups, height: 1, depth: 1),
        threadsPerThreadgroup: MTLSize(width: threadgroupSize, height: 1, depth: 1)
    )

    encoder.endEncoding()
    cmdBuf.commit()
    cmdBuf.waitUntilCompleted()

    // ── Sum partial results on CPU ─────────────────────────────────
    //
    // 256 partial sums — trivial to add on CPU.
    // In a production implementation you could do a second reduction
    // pass on GPU, but for 256 values it's not worth it.

    let ptrP = bufP.contents().bindMemory(to: SIMD2<Float>.self, capacity: numThreadgroups)
    var gpuResult = SIMD2<Float>(0, 0)
    for i in 0..<numThreadgroups {
        gpuResult.x += ptrP[i].x
        gpuResult.y += ptrP[i].y
    }

    print("  GPU result:    (\(gpuResult.x), \(gpuResult.y))")

    let errReal = abs(gpuResult.x - refResult.x)
    let errImag = abs(gpuResult.y - refResult.y)
    // Float32 accumulating 65536 values — expect ~1e-1 to 1e0 absolute error
    // due to catastrophic cancellation in large sums. Relative error matters more.
    let relErr = max(errReal / abs(refResult.x), errImag / abs(refResult.y))

    print("  Absolute error: (\(errReal), \(errImag))")
    print("  Relative error: \(relErr)")

    if relErr < 1e-4 {
        print("  ✓ PASS  (relative error < 1e-4)")
    } else if relErr < 1e-2 {
        print("  ⚠ MARGINAL  (relative error < 1e-2 — float32 accumulation noise)")
    } else {
        print("  ✗ FAIL  (relative error too large)")
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// TEST 3: ZAXPY — y = y + alpha * x (element-wise, no reduction)
//
// This is the other core BLAS-1 operation. QE uses it for wavefunction
// updates during Davidson iteration: |ψ_new⟩ = |ψ⟩ + α|δψ⟩
// ═══════════════════════════════════════════════════════════════════════════

print("\n── Test 3: ZAXPY — y += alpha × x ──────────────────────────")

do {
    let N = 1024
    let bufferSize = N * MemoryLayout<SIMD2<Float>>.stride

    guard let bufX = device.makeBuffer(length: bufferSize, options: .storageModeShared),
          let bufY = device.makeBuffer(length: bufferSize, options: .storageModeShared) else {
        print("❌ Buffer allocation failed"); exit(1)
    }

    let ptrX = bufX.contents().bindMemory(to: SIMD2<Float>.self, capacity: N)
    let ptrY = bufY.contents().bindMemory(to: SIMD2<Float>.self, capacity: N)

    for i in 0..<N {
        ptrX[i] = SIMD2<Float>(Float(i), Float(i))        // x[i] = i + ii
        ptrY[i] = SIMD2<Float>(100.0, 0.0)                // y[i] = 100 + 0i
    }

    var alpha = SIMD2<Float>(0.0, 1.0)  // alpha = i (purely imaginary)

    guard let function = library.makeFunction(name: "zaxpy") else {
        print("❌ Kernel not found"); exit(1)
    }
    let pipeline = try device.makeComputePipelineState(function: function)

    guard let cmdBuf = commandQueue.makeCommandBuffer(),
          let encoder = cmdBuf.makeComputeCommandEncoder() else {
        print("❌ Encoder failed"); exit(1)
    }

    encoder.setComputePipelineState(pipeline)
    encoder.setBuffer(bufX, offset: 0, index: 0)
    encoder.setBuffer(bufY, offset: 0, index: 1)
    var count = UInt32(N)
    encoder.setBytes(&count, length: MemoryLayout<UInt32>.size, index: 2)
    encoder.setBytes(&alpha, length: MemoryLayout<SIMD2<Float>>.size, index: 3)

    let threadsPerGrid = MTLSize(width: N, height: 1, depth: 1)
    let threadsPerGroup = MTLSize(width: min(N, pipeline.maxTotalThreadsPerThreadgroup), height: 1, depth: 1)
    encoder.dispatchThreads(threadsPerGrid, threadsPerThreadgroup: threadsPerGroup)

    encoder.endEncoding()
    cmdBuf.commit()
    cmdBuf.waitUntilCompleted()

    // Verify: y[i] = (100, 0) + (0, 1) * (i, i)
    // alpha * x = (0,1)(i,i) = (0*i - 1*i, 0*i + 1*i) = (-i, i)
    // y = (100, 0) + (-i, i) = (100 - i, i)

    var maxErr: Float = 0
    var errCount = 0

    for i in 0..<N {
        let fi = Float(i)
        let expectedReal: Float = 100.0 - fi
        let expectedImag: Float = fi
        let err = max(abs(ptrY[i].x - expectedReal), abs(ptrY[i].y - expectedImag))
        maxErr = max(maxErr, err)
        if err > 1e-4 { errCount += 1 }
    }

    if errCount == 0 {
        print("  ✓ PASS  (max error: \(maxErr))")
    } else {
        print("  ✗ FAIL  (\(errCount) errors, max: \(maxErr))")
    }

    print("  Sample values (alpha = i):")
    for i in [0, 1, 10, 100] {
        let fi = Float(i)
        print("    y[\(i)] = (\(ptrY[i].x), \(ptrY[i].y))  expected: (\(100.0 - fi), \(fi))")
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// TEST 4: Performance — how fast is our reduction?
//
// Measures throughput in GFLOP/s for the dot product.
// A complex dot product on N elements = N×(6 flops for cmul + 2 for add) = 8N flops.
// ═══════════════════════════════════════════════════════════════════════════

print("\n── Test 4: ZDOTC throughput ─────────────────────────────────")

do {
    let sizes = [1024, 16384, 65536, 262144, 1048576]  // 1K to 1M elements

    guard let function = library.makeFunction(name: "zdotc_reduce") else {
        print("❌ Kernel not found"); exit(1)
    }
    let pipeline = try device.makeComputePipelineState(function: function)
    let threadgroupSize = 256

    print("           N   Time (ms)    GFLOP/s")
    print("  \(String(repeating: "─", count: 36))")

    for N in sizes {
        let bufferSize = N * MemoryLayout<SIMD2<Float>>.stride
        let numThreadgroups = (N + threadgroupSize - 1) / threadgroupSize
        let partialSize = numThreadgroups * MemoryLayout<SIMD2<Float>>.stride

        guard let bufA = device.makeBuffer(length: bufferSize, options: .storageModeShared),
              let bufB = device.makeBuffer(length: bufferSize, options: .storageModeShared),
              let bufP = device.makeBuffer(length: partialSize, options: .storageModeShared) else {
            continue
        }

        // Fill with ones so we don't care about the result
        let ptrA = bufA.contents().bindMemory(to: SIMD2<Float>.self, capacity: N)
        let ptrB = bufB.contents().bindMemory(to: SIMD2<Float>.self, capacity: N)
        for i in 0..<N {
            ptrA[i] = SIMD2<Float>(1.0, 0.0)
            ptrB[i] = SIMD2<Float>(1.0, 0.0)
        }

        // Warmup
        for _ in 0..<3 {
            guard let cmdBuf = commandQueue.makeCommandBuffer(),
                  let enc = cmdBuf.makeComputeCommandEncoder() else { continue }
            enc.setComputePipelineState(pipeline)
            enc.setBuffer(bufA, offset: 0, index: 0)
            enc.setBuffer(bufB, offset: 0, index: 1)
            enc.setBuffer(bufP, offset: 0, index: 2)
            var count = UInt32(N)
            enc.setBytes(&count, length: MemoryLayout<UInt32>.size, index: 3)
            enc.dispatchThreadgroups(
                MTLSize(width: numThreadgroups, height: 1, depth: 1),
                threadsPerThreadgroup: MTLSize(width: threadgroupSize, height: 1, depth: 1))
            enc.endEncoding()
            cmdBuf.commit()
            cmdBuf.waitUntilCompleted()
        }

        // Timed run — 100 iterations
        let reps = 100
        let t0 = CFAbsoluteTimeGetCurrent()
        for _ in 0..<reps {
            guard let cmdBuf = commandQueue.makeCommandBuffer(),
                  let enc = cmdBuf.makeComputeCommandEncoder() else { continue }
            enc.setComputePipelineState(pipeline)
            enc.setBuffer(bufA, offset: 0, index: 0)
            enc.setBuffer(bufB, offset: 0, index: 1)
            enc.setBuffer(bufP, offset: 0, index: 2)
            var count = UInt32(N)
            enc.setBytes(&count, length: MemoryLayout<UInt32>.size, index: 3)
            enc.dispatchThreadgroups(
                MTLSize(width: numThreadgroups, height: 1, depth: 1),
                threadsPerThreadgroup: MTLSize(width: threadgroupSize, height: 1, depth: 1))
            enc.endEncoding()
            cmdBuf.commit()
            cmdBuf.waitUntilCompleted()
        }
        let t1 = CFAbsoluteTimeGetCurrent()

        let ms = (t1 - t0) * 1000.0 / Double(reps)
        let gflops = Double(8 * N) / (ms * 1e6)

        print(String(format: "  %10d  %9.3f ms  %9.2f", N, ms, gflops))
    }

    print("")
    print("  Small N:  dominated by dispatch overhead (low GFLOP/s)")
    print("  Large N:  approaches memory bandwidth limit")
    print("  This is why the interpose library has a size threshold!")
}

// ═══════════════════════════════════════════════════════════════════════════
// SUMMARY
// ═══════════════════════════════════════════════════════════════════════════

print("""

═══════════════════════════════════════════════════════════════
  Exercise 2 complete.

  What you learned:
    • threadgroup memory — shared fast SRAM within a threadgroup
    • threadgroup_barrier — synchronize before reading others' data
    • Tree reduction — halving active threads each step: O(log N)
    • Why you can't just atomically add floats on a GPU
    • ZDOTC = Σ conj(A[i]) × B[i] (QE overlap matrices use this)
    • ZAXPY = y + alpha × x (QE Davidson updates use this)

  How this connects to what's next:
    • Tiled GEMM (Exercise 3) loads matrix tiles into threadgroup
      memory using the SAME shared[] + barrier pattern
    • FFT (Exercise 5) does butterfly exchanges through threadgroup
      memory using the SAME shared[] + barrier pattern
    • The reduction tree is how you'd sum partial GEMM tiles

  Things to try:
    • Change threadgroup size from 256 to 64 or 512. What happens
      to correctness? Performance? (Hint: the shared_data array
      size is hardcoded to 256 — you need to change both.)
    • Double precision: change float2 to double2. Does the large
      dot product error decrease? (It should — that's why QE uses FP64.)
    • Can you write a second GPU pass that reduces the 256 partials
      instead of summing them on CPU?

  NEXT: Exercise 3 — Real matrix multiply (DGEMM).
    That's where tiling, threadgroup memory, and 2D dispatch combine.
    Read philip-turner's architecture docs before starting Exercise 3.
═══════════════════════════════════════════════════════════════
""")
