#!/usr/bin/env swift
// Exercise 1: Complex Arithmetic on Metal GPU
// swiftc -O -framework Metal -framework Foundation ex01_complex_arithmetic.swift -o ex01
// Grant Heileman — UNM ECE — 2026

import Foundation
import Metal

let shaderSource = """
#include <metal_stdlib>
using namespace metal;

inline float2 cadd(float2 a, float2 b) { return a + b; }
inline float2 cmul(float2 a, float2 b) {
    return float2(a.x * b.x - a.y * b.y, a.x * b.y + a.y * b.x);
}
inline float2 cconj(float2 z) { return float2(z.x, -z.y); }
inline float  cmag2(float2 z) { return dot(z, z); }
inline float  cmag(float2 z)  { return sqrt(cmag2(z)); }

kernel void complex_multiply(
    device const float2 *A [[buffer(0)]], device const float2 *B [[buffer(1)]],
    device float2 *C [[buffer(2)]], constant uint &count [[buffer(3)]],
    uint gid [[thread_position_in_grid]]
) {
    if (gid >= count) return;
    C[gid] = cmul(A[gid], B[gid]);
}

kernel void complex_multiply_accumulate(
    device const float2 *A [[buffer(0)]], device const float2 *B [[buffer(1)]],
    device float2 *C [[buffer(2)]], constant uint &count [[buffer(3)]],
    constant float2 &alpha [[buffer(4)]], uint gid [[thread_position_in_grid]]
) {
    if (gid >= count) return;
    C[gid] = cadd(C[gid], cmul(alpha, cmul(A[gid], B[gid])));
}

kernel void complex_conj_and_mag(
    device const float2 *A [[buffer(0)]], device float2 *out_conj [[buffer(1)]],
    device float *out_mag [[buffer(2)]], constant uint &count [[buffer(3)]],
    uint gid [[thread_position_in_grid]]
) {
    if (gid >= count) return;
    out_conj[gid] = cconj(A[gid]);
    out_mag[gid]  = cmag(A[gid]);
}
"""

// MARK: - Setup

guard let device = MTLCreateSystemDefaultDevice() else { fatalError("No Metal device") }
assert(device.hasUnifiedMemory, "Requires Apple Silicon")

let opts = MTLCompileOptions(); opts.mathMode = .fast
let library = try device.makeLibrary(source: shaderSource, options: opts)
guard let queue = device.makeCommandQueue() else { fatalError("No command queue") }

let N = 1024
let bufSize = N * MemoryLayout<SIMD2<Float>>.stride

func pipeline(_ name: String) -> MTLComputePipelineState {
    try! device.makeComputePipelineState(function: library.makeFunction(name: name)!)
}

func dispatch1D(_ pip: MTLComputePipelineState, label: String, _ encode: (MTLComputeCommandEncoder) -> Void) {
    guard let cb = queue.makeCommandBuffer(), let enc = cb.makeComputeCommandEncoder() else { return }
    cb.label = label
    enc.setComputePipelineState(pip)
    encode(enc)
    enc.dispatchThreads(MTLSize(width: N, height: 1, depth: 1),
                        threadsPerThreadgroup: MTLSize(width: min(N, pip.maxTotalThreadsPerThreadgroup), height: 1, depth: 1))
    enc.endEncoding(); cb.commit(); cb.waitUntilCompleted()
    if cb.status == .error { fatalError("GPU [\(label)]: \(cb.error?.localizedDescription ?? "?")") }
}

print("Exercise 1: Complex Arithmetic — GPU: \(device.name)\n")

// MARK: - Test 1: Element-wise complex multiply

do {
    let pip = pipeline("complex_multiply")
    let A = device.makeBuffer(length: bufSize, options: .storageModeShared)!
    let B = device.makeBuffer(length: bufSize, options: .storageModeShared)!
    let C = device.makeBuffer(length: bufSize, options: .storageModeShared)!

    let pA = A.contents().bindMemory(to: SIMD2<Float>.self, capacity: N)
    let pB = B.contents().bindMemory(to: SIMD2<Float>.self, capacity: N)
    for i in 0..<N {
        pA[i] = SIMD2<Float>(Float(i), Float(i + 1))
        pB[i] = SIMD2<Float>(1.0, -1.0)
    }

    var count = UInt32(N)
    dispatch1D(pip, label: "cmul") { enc in
        enc.setBuffer(A, offset: 0, index: 0); enc.setBuffer(B, offset: 0, index: 1); enc.setBuffer(C, offset: 0, index: 2)
        enc.setBytes(&count, length: 4, index: 3)
    }

    let pC = C.contents().bindMemory(to: SIMD2<Float>.self, capacity: N)
    var maxErr: Float = 0
    for i in 0..<N {
        let a = Float(i), b = Float(i + 1)
        maxErr = max(maxErr, max(abs(pC[i].x - (a + b)), abs(pC[i].y - (-a + b))))
    }
    print("  cmul:            max err = \(maxErr)  \(maxErr < 1e-4 ? "✓" : "✗")")
}

// MARK: - Test 2: Multiply-accumulate

do {
    let pip = pipeline("complex_multiply_accumulate")
    let A = device.makeBuffer(length: bufSize, options: .storageModeShared)!
    let B = device.makeBuffer(length: bufSize, options: .storageModeShared)!
    let C = device.makeBuffer(length: bufSize, options: .storageModeShared)!

    let pA = A.contents().bindMemory(to: SIMD2<Float>.self, capacity: N)
    let pB = B.contents().bindMemory(to: SIMD2<Float>.self, capacity: N)
    let pC = C.contents().bindMemory(to: SIMD2<Float>.self, capacity: N)
    for i in 0..<N {
        pA[i] = SIMD2<Float>(Float(i), 0)
        pB[i] = SIMD2<Float>(0, Float(i))
        pC[i] = SIMD2<Float>(100, 200)
    }

    var alpha = SIMD2<Float>(2, 0), count = UInt32(N)
    dispatch1D(pip, label: "cmac") { enc in
        enc.setBuffer(A, offset: 0, index: 0); enc.setBuffer(B, offset: 0, index: 1); enc.setBuffer(C, offset: 0, index: 2)
        enc.setBytes(&count, length: 4, index: 3); enc.setBytes(&alpha, length: 8, index: 4)
    }

    var maxErr: Float = 0
    for i in 0..<N {
        let fi = Float(i)
        maxErr = max(maxErr, max(abs(pC[i].x - 100), abs(pC[i].y - (200 + 2 * fi * fi))))
    }
    print("  cmac:            max err = \(maxErr)  \(maxErr < 1e-2 ? "✓" : "✗")")
}

// MARK: - Test 3: Conjugate + magnitude

do {
    let pip = pipeline("complex_conj_and_mag")
    let A    = device.makeBuffer(length: bufSize, options: .storageModeShared)!
    let Conj = device.makeBuffer(length: bufSize, options: .storageModeShared)!
    let Mag  = device.makeBuffer(length: N * MemoryLayout<Float>.stride, options: .storageModeShared)!

    let pA = A.contents().bindMemory(to: SIMD2<Float>.self, capacity: N)
    for i in 0..<N { pA[i] = SIMD2<Float>(Float(3 * i), Float(4 * i)) }

    var count = UInt32(N)
    dispatch1D(pip, label: "conj_mag") { enc in
        enc.setBuffer(A, offset: 0, index: 0); enc.setBuffer(Conj, offset: 0, index: 1); enc.setBuffer(Mag, offset: 0, index: 2)
        enc.setBytes(&count, length: 4, index: 3)
    }

    let pConj = Conj.contents().bindMemory(to: SIMD2<Float>.self, capacity: N)
    let pMag = Mag.contents().bindMemory(to: Float.self, capacity: N)
    var conjErr: Float = 0, magErr: Float = 0
    for i in 0..<N {
        let fi = Float(i)
        conjErr = max(conjErr, max(abs(pConj[i].x - 3*fi), abs(pConj[i].y + 4*fi)))
        magErr  = max(magErr, abs(pMag[i] - 5*fi))
    }
    print("  conj:            max err = \(conjErr)  \(conjErr < 1e-4 ? "✓" : "✗")")
    print("  mag:             max err = \(magErr)  \(magErr < 1e-3 ? "✓" : "✗")")
}
