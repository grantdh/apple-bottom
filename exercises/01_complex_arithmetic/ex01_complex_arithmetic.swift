#!/usr/bin/env swift
//
// Exercise 1: Complex Arithmetic on Metal GPU
//
// WHAT YOU'LL LEARN:
//   - Metal has no native complex type. We use float2 (or double2).
//   - .x = real part, .y = imaginary part
//   - Complex multiply, add, conjugate, magnitude — all inline functions
//   - How to verify GPU results against CPU reference
//
// COMPILE & RUN:
//   swiftc -O -framework Metal -framework Foundation ex01_complex_arithmetic.swift -o ex01
//   ./ex01
//
// THEN: Read the shader source below. Modify cmul(). See what breaks.
//
// Grant Heileman — UNM ECE — 2026
//

import Foundation
import Metal

// ═══════════════════════════════════════════════════════════════════════════
// STEP 1: THE METAL SHADER
//
// This is the GPU code. Read this first.
// It's written in Metal Shading Language (MSL), which is C++14-based.
//
// Key idea: a complex number z = a + bi is stored as float2(a, b).
// Metal's float2 is a SIMD vector type — .x and .y access the components.
// ═══════════════════════════════════════════════════════════════════════════

let shaderSource = """
#include <metal_stdlib>
using namespace metal;

// ─── Complex arithmetic helpers ────────────────────────────────────
// These are "inline" — the compiler pastes them directly into the
// calling kernel, no function call overhead.

// Complex addition: (a+bi) + (c+di) = (a+c) + (b+d)i
inline float2 cadd(float2 z1, float2 z2) {
    return z1 + z2;  // float2 addition is component-wise
}

// Complex multiplication: (a+bi)(c+di) = (ac-bd) + (ad+bc)i
//
// This is THE fundamental operation for ZGEMM and FFT.
// 4 multiplies + 2 adds (or with FMA: 2 FMA + 1 mul + 1 add).
inline float2 cmul(float2 z1, float2 z2) {
    return float2(
        z1.x * z2.x - z1.y * z2.y,   // real: ac - bd
        z1.x * z2.y + z1.y * z2.x    // imag: ad + bc
    );
}

// Complex conjugate: conj(a+bi) = a - bi
inline float2 cconj(float2 z) {
    return float2(z.x, -z.y);
}

// Complex magnitude squared: |z|² = a² + b²
// (Avoids the sqrt — use this when you can, it's cheaper.)
inline float cmag2(float2 z) {
    return z.x * z.x + z.y * z.y;
}

// Complex magnitude: |z| = sqrt(a² + b²)
inline float cmag(float2 z) {
    return sqrt(cmag2(z));
}

// ─── Kernel 1: Element-wise complex multiply ──────────────────────
//
// C[i] = A[i] * B[i]   for each element in parallel
//
// This is the simplest possible Metal compute kernel.
// Each GPU thread handles exactly one element.
//
// The [[buffer(N)]] attributes tell Metal which argument binds to
// which buffer slot — you'll set these from Swift.
//
// [[thread_position_in_grid]] gives this thread's global index,
// same as your FDTD kernel's gid.

kernel void complex_multiply(
    device const float2 *A     [[buffer(0)]],  // input array A
    device const float2 *B     [[buffer(1)]],  // input array B
    device float2       *C     [[buffer(2)]],  // output array C
    constant uint       &count [[buffer(3)]],  // number of elements
    uint gid                   [[thread_position_in_grid]]
) {
    if (gid >= count) return;  // bounds check (some threads are padding)
    C[gid] = cmul(A[gid], B[gid]);
}

// ─── Kernel 2: Complex multiply-accumulate ────────────────────────
//
// C[i] = C[i] + alpha * A[i] * B[i]
//
// This pattern appears everywhere in BLAS:
//   ZAXPY:  y = y + alpha * x       (vector)
//   ZGEMM:  C = alpha * A*B + beta*C (matrix, per element)
//
// Notice "alpha" is complex too — passed as a single float2 via
// constant& (uniform across all threads).

kernel void complex_multiply_accumulate(
    device const float2 *A      [[buffer(0)]],
    device const float2 *B      [[buffer(1)]],
    device float2       *C      [[buffer(2)]],
    constant uint       &count  [[buffer(3)]],
    constant float2     &alpha  [[buffer(4)]],
    uint gid                    [[thread_position_in_grid]]
) {
    if (gid >= count) return;
    // alpha * A[i] * B[i]  — two complex multiplies chained
    float2 product = cmul(alpha, cmul(A[gid], B[gid]));
    C[gid] = cadd(C[gid], product);  // accumulate into C
}

// ─── Kernel 3: Conjugate + magnitude ──────────────────────────────
//
// out_conj[i] = conj(A[i])
// out_mag[i]  = |A[i]|
//
// Two outputs from one kernel. This is fine in Metal — just bind
// more output buffers.

kernel void complex_conj_and_mag(
    device const float2 *A        [[buffer(0)]],
    device float2       *out_conj [[buffer(1)]],
    device float        *out_mag  [[buffer(2)]],
    constant uint       &count    [[buffer(3)]],
    uint gid                      [[thread_position_in_grid]]
) {
    if (gid >= count) return;
    out_conj[gid] = cconj(A[gid]);
    out_mag[gid]  = cmag(A[gid]);
}
"""

// ═══════════════════════════════════════════════════════════════════════════
// STEP 2: SWIFT HOST CODE
//
// This sets up the Metal device, compiles the shader, allocates buffers,
// dispatches the kernel, and reads back results.
//
// You've done all of this in LoopFDTD. The new part is how we represent
// complex numbers on the CPU side and verify against CPU reference.
// ═══════════════════════════════════════════════════════════════════════════

print("""

╔═══════════════════════════════════════════════════════════════╗
║  Exercise 1: Complex Arithmetic on Metal GPU                 ║
╚═══════════════════════════════════════════════════════════════╝

""")

// ── Metal setup (same as your FDTD code) ────────────────────────────────

guard let device = MTLCreateSystemDefaultDevice() else {
    print("❌ No Metal device found"); exit(1)
}
print("GPU: \(device.name)")

// Compile shaders from the string above
let library: MTLLibrary
do {
    library = try device.makeLibrary(source: shaderSource, options: nil)
    print("✓ Shaders compiled\n")
} catch {
    print("❌ Shader compilation failed: \(error)"); exit(1)
}

guard let commandQueue = device.makeCommandQueue() else {
    print("❌ Failed to create command queue"); exit(1)
}

// ── Test parameters ─────────────────────────────────────────────────────

let N = 1024  // Number of complex elements
// On GPU, each complex number is a float2 = 8 bytes.
// 1024 elements = 8 KB per buffer. Tiny — fits in cache.
let bufferSize = N * MemoryLayout<SIMD2<Float>>.stride

print("Array size: \(N) complex numbers (\(bufferSize / 1024) KB per buffer)\n")

// ═══════════════════════════════════════════════════════════════════════════
// TEST 1: Element-wise complex multiply — C[i] = A[i] * B[i]
// ═══════════════════════════════════════════════════════════════════════════

print("── Test 1: C[i] = A[i] × B[i] ─────────────────────────────")

do {
    // Allocate GPU-accessible buffers (storageModeShared = CPU+GPU see same memory)
    guard let bufA = device.makeBuffer(length: bufferSize, options: .storageModeShared),
          let bufB = device.makeBuffer(length: bufferSize, options: .storageModeShared),
          let bufC = device.makeBuffer(length: bufferSize, options: .storageModeShared) else {
        print("❌ Buffer allocation failed"); exit(1)
    }

    // Fill A and B with test data on the CPU side.
    // We use SIMD2<Float> which is Swift's equivalent of Metal's float2.
    let ptrA = bufA.contents().bindMemory(to: SIMD2<Float>.self, capacity: N)
    let ptrB = bufB.contents().bindMemory(to: SIMD2<Float>.self, capacity: N)

    for i in 0..<N {
        // A[i] = (i, i+1)  i.e. the complex number i + (i+1)i
        ptrA[i] = SIMD2<Float>(Float(i), Float(i + 1))
        // B[i] = (1, -1)   i.e. the complex number 1 - i
        ptrB[i] = SIMD2<Float>(1.0, -1.0)
    }

    // Make the pipeline for "complex_multiply" kernel
    guard let function = library.makeFunction(name: "complex_multiply") else {
        print("❌ Kernel 'complex_multiply' not found"); exit(1)
    }
    let pipeline = try device.makeComputePipelineState(function: function)

    // Create command buffer and encoder
    guard let cmdBuf = commandQueue.makeCommandBuffer(),
          let encoder = cmdBuf.makeComputeCommandEncoder() else {
        print("❌ Failed to create encoder"); exit(1)
    }

    // Bind the pipeline and buffers
    encoder.setComputePipelineState(pipeline)
    encoder.setBuffer(bufA, offset: 0, index: 0)  // buffer(0) = A
    encoder.setBuffer(bufB, offset: 0, index: 1)  // buffer(1) = B
    encoder.setBuffer(bufC, offset: 0, index: 2)  // buffer(2) = C

    // Pass the count as a constant (buffer(3))
    var count = UInt32(N)
    encoder.setBytes(&count, length: MemoryLayout<UInt32>.size, index: 3)

    // Dispatch N threads. Metal will figure out how to group them.
    // This is the same pattern as your FDTD: dispatchThreads with 1D grid.
    let threadsPerGrid = MTLSize(width: N, height: 1, depth: 1)
    let threadsPerGroup = MTLSize(width: min(N, pipeline.maxTotalThreadsPerThreadgroup), height: 1, depth: 1)
    encoder.dispatchThreads(threadsPerGrid, threadsPerThreadgroup: threadsPerGroup)

    encoder.endEncoding()
    cmdBuf.commit()
    cmdBuf.waitUntilCompleted()  // Sync — fine for testing, bad for perf

    // ── Read back and verify ────────────────────────────────────────
    let ptrC = bufC.contents().bindMemory(to: SIMD2<Float>.self, capacity: N)

    var maxError: Float = 0
    var errCount = 0

    for i in 0..<N {
        // CPU reference: (a+bi)(c+di) = (ac-bd) + (ad+bc)i
        let a = Float(i), b = Float(i + 1)
        let c: Float = 1.0, d: Float = -1.0
        let expectedReal = a * c - b * d   // a*1 - (a+1)*(-1) = a + a + 1 = 2a + 1
        let expectedImag = a * d + b * c   // a*(-1) + (a+1)*1 = -a + a + 1 = 1
        let expected = SIMD2<Float>(expectedReal, expectedImag)

        let err = max(abs(ptrC[i].x - expected.x), abs(ptrC[i].y - expected.y))
        maxError = max(maxError, err)
        if err > 1e-4 { errCount += 1 }
    }

    if errCount == 0 {
        print("  ✓ PASS  (max error: \(maxError))")
    } else {
        print("  ✗ FAIL  (\(errCount) errors, max: \(maxError))")
    }

    // Show a few values so you can trace the math by hand
    print("  Sample values:")
    for i in [0, 1, 2, N-1] {
        let a = ptrA[i], b = ptrB[i], c = ptrC[i]
        print("    A[\(i)]=(\(a.x),\(a.y)) × B[\(i)]=(\(b.x),\(b.y)) = C[\(i)]=(\(c.x),\(c.y))")
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// TEST 2: Complex multiply-accumulate — C[i] += alpha * A[i] * B[i]
// ═══════════════════════════════════════════════════════════════════════════

print("\n── Test 2: C[i] += alpha × A[i] × B[i] ────────────────────")

do {
    guard let bufA = device.makeBuffer(length: bufferSize, options: .storageModeShared),
          let bufB = device.makeBuffer(length: bufferSize, options: .storageModeShared),
          let bufC = device.makeBuffer(length: bufferSize, options: .storageModeShared) else {
        print("❌ Buffer allocation failed"); exit(1)
    }

    let ptrA = bufA.contents().bindMemory(to: SIMD2<Float>.self, capacity: N)
    let ptrB = bufB.contents().bindMemory(to: SIMD2<Float>.self, capacity: N)
    let ptrC = bufC.contents().bindMemory(to: SIMD2<Float>.self, capacity: N)

    for i in 0..<N {
        ptrA[i] = SIMD2<Float>(Float(i), 0)          // A[i] = i + 0i (purely real)
        ptrB[i] = SIMD2<Float>(0, Float(i))           // B[i] = 0 + ii (purely imaginary)
        ptrC[i] = SIMD2<Float>(100.0, 200.0)          // C[i] = 100 + 200i (pre-existing)
    }

    // alpha = 2 + 0i (real scalar)
    var alpha = SIMD2<Float>(2.0, 0.0)

    guard let function = library.makeFunction(name: "complex_multiply_accumulate") else {
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
    encoder.setBuffer(bufC, offset: 0, index: 2)
    var count = UInt32(N)
    encoder.setBytes(&count, length: MemoryLayout<UInt32>.size, index: 3)
    encoder.setBytes(&alpha, length: MemoryLayout<SIMD2<Float>>.size, index: 4)

    let threadsPerGrid = MTLSize(width: N, height: 1, depth: 1)
    let threadsPerGroup = MTLSize(width: min(N, pipeline.maxTotalThreadsPerThreadgroup), height: 1, depth: 1)
    encoder.dispatchThreads(threadsPerGrid, threadsPerThreadgroup: threadsPerGroup)

    encoder.endEncoding()
    cmdBuf.commit()
    cmdBuf.waitUntilCompleted()

    // Verify: C[i] = (100+200i) + (2+0i) * (i+0i) * (0+ii)
    // Inner: A*B = i * ii = i²i = -i (wait, let me be careful)
    // A[i] = (i, 0), B[i] = (0, i)
    // A*B = (i*0 - 0*i) + (i*i + 0*0)i = 0 + i²·i = (0, i²)
    // alpha * A*B = (2,0) * (0, i²) = (0, 2i²)
    // C += that: C = (100, 200 + 2i²)

    var maxError: Float = 0
    var errCount = 0

    for i in 0..<N {
        let fi = Float(i)
        let expectedReal: Float = 100.0                // unchanged
        let expectedImag: Float = 200.0 + 2.0 * fi * fi  // 200 + 2i²
        let err = max(abs(ptrC[i].x - expectedReal), abs(ptrC[i].y - expectedImag))
        maxError = max(maxError, err)
        if err > 1e-2 { errCount += 1 }  // relaxed for float accumulation
    }

    if errCount == 0 {
        print("  ✓ PASS  (max error: \(maxError))")
    } else {
        print("  ✗ FAIL  (\(errCount) errors, max: \(maxError))")
    }

    print("  Sample values:")
    for i in [0, 1, 10, 31] {
        let fi = Float(i)
        print("    C[\(i)] = (\(ptrC[i].x), \(ptrC[i].y))  expected: (100.0, \(200.0 + 2.0*fi*fi))")
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// TEST 3: Conjugate + magnitude
// ═══════════════════════════════════════════════════════════════════════════

print("\n── Test 3: Conjugate and magnitude ─────────────────────────")

do {
    let magBufferSize = N * MemoryLayout<Float>.stride

    guard let bufA    = device.makeBuffer(length: bufferSize, options: .storageModeShared),
          let bufConj = device.makeBuffer(length: bufferSize, options: .storageModeShared),
          let bufMag  = device.makeBuffer(length: magBufferSize, options: .storageModeShared) else {
        print("❌ Buffer allocation failed"); exit(1)
    }

    let ptrA = bufA.contents().bindMemory(to: SIMD2<Float>.self, capacity: N)
    for i in 0..<N {
        ptrA[i] = SIMD2<Float>(Float(3 * i), Float(4 * i))  // |3i + 4ii| = 5i
    }

    guard let function = library.makeFunction(name: "complex_conj_and_mag") else {
        print("❌ Kernel not found"); exit(1)
    }
    let pipeline = try device.makeComputePipelineState(function: function)

    guard let cmdBuf = commandQueue.makeCommandBuffer(),
          let encoder = cmdBuf.makeComputeCommandEncoder() else {
        print("❌ Encoder failed"); exit(1)
    }

    encoder.setComputePipelineState(pipeline)
    encoder.setBuffer(bufA, offset: 0, index: 0)
    encoder.setBuffer(bufConj, offset: 0, index: 1)
    encoder.setBuffer(bufMag, offset: 0, index: 2)
    var count = UInt32(N)
    encoder.setBytes(&count, length: MemoryLayout<UInt32>.size, index: 3)

    let threadsPerGrid = MTLSize(width: N, height: 1, depth: 1)
    let threadsPerGroup = MTLSize(width: min(N, pipeline.maxTotalThreadsPerThreadgroup), height: 1, depth: 1)
    encoder.dispatchThreads(threadsPerGrid, threadsPerThreadgroup: threadsPerGroup)

    encoder.endEncoding()
    cmdBuf.commit()
    cmdBuf.waitUntilCompleted()

    let ptrConj = bufConj.contents().bindMemory(to: SIMD2<Float>.self, capacity: N)
    let ptrMag = bufMag.contents().bindMemory(to: Float.self, capacity: N)

    var conjErrors = 0, magErrors = 0
    var maxConjErr: Float = 0, maxMagErr: Float = 0

    for i in 0..<N {
        let fi = Float(i)
        // Conjugate: (3i, 4i) → (3i, -4i)
        let conjErr = max(abs(ptrConj[i].x - 3*fi), abs(ptrConj[i].y - (-4*fi)))
        maxConjErr = max(maxConjErr, conjErr)
        if conjErr > 1e-4 { conjErrors += 1 }

        // Magnitude: |(3i, 4i)| = sqrt(9i² + 16i²) = 5i
        let magErr = abs(ptrMag[i] - 5*fi)
        maxMagErr = max(maxMagErr, magErr)
        if magErr > 1e-3 { magErrors += 1 }
    }

    if conjErrors == 0 {
        print("  ✓ Conjugate PASS  (max error: \(maxConjErr))")
    } else {
        print("  ✗ Conjugate FAIL  (\(conjErrors) errors, max: \(maxConjErr))")
    }

    if magErrors == 0 {
        print("  ✓ Magnitude PASS  (max error: \(maxMagErr))")
    } else {
        print("  ✗ Magnitude FAIL  (\(magErrors) errors, max: \(maxMagErr))")
    }

    print("  Sample values:")
    for i in [0, 1, 5, 100] {
        print("    A[\(i)]=(\(ptrA[i].x),\(ptrA[i].y))  conj=(\(ptrConj[i].x),\(ptrConj[i].y))  |A|=\(ptrMag[i])  expected=\(5.0*Float(i))")
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// SUMMARY
// ═══════════════════════════════════════════════════════════════════════════

print("""

═══════════════════════════════════════════════════════════════
  Exercise 1 complete.

  What you just ran on the GPU:
    • cmul:  (a+bi)(c+di) = (ac-bd) + (ad+bc)i
    • cadd:  component-wise float2 addition
    • cconj: flip the sign of .y
    • cmag:  sqrt(x² + y²)

  These four functions are the atoms of everything else:
    ZGEMM = thousands of cmul + cadd
    FFT   = cmul with twiddle factors + cadd in butterfly pattern

  NEXT: Exercise 2 — complex dot product with parallel reduction.
    This teaches threadgroup shared memory, which you'll need for
    both tiled GEMM and FFT.

  Modify the shader to try:
    • What happens if you swap the signs in cmul?
    • What if you use double2 instead of float2?
    • Can you add a kernel that computes A[i] * conj(B[i])?
      (This is the inner product pattern used in QE's overlap matrices)
═══════════════════════════════════════════════════════════════
""")
