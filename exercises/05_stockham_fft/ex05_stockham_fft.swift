#!/usr/bin/env swift
// Exercise 5: Stockham Radix-2 FFT on Metal GPU
// swiftc -O -framework Metal -framework Foundation -framework Accelerate ex05_stockham_fft.swift -o ex05
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

// Twiddle table: twiddle[j] = exp(-2πij/N), indexed as twiddle[k * (halfN/stride)]
kernel void stockham_radix2(
    device const float2 *input [[buffer(0)]], device float2 *output [[buffer(1)]],
    constant uint &N [[buffer(2)]], constant uint &pass [[buffer(3)]],
    device const float2 *twiddle [[buffer(4)]], uint gid [[thread_position_in_grid]]
) {
    uint halfN = N / 2;
    if (gid >= halfN) return;
    uint stride = 1u << pass, group = gid / stride, k = gid % stride;
    float2 a = input[gid], Wb = cmul(twiddle[k * (halfN / stride)], input[gid + halfN]);
    uint out = group * stride * 2 + k;
    output[out]          = a + Wb;
    output[out + stride] = a - Wb;
}

kernel void stockham_radix2_inverse(
    device const float2 *input [[buffer(0)]], device float2 *output [[buffer(1)]],
    constant uint &N [[buffer(2)]], constant uint &pass [[buffer(3)]],
    device const float2 *twiddle [[buffer(4)]], uint gid [[thread_position_in_grid]]
) {
    uint halfN = N / 2;
    if (gid >= halfN) return;
    uint stride = 1u << pass, group = gid / stride, k = gid % stride;
    float2 tw = twiddle[k * (halfN / stride)];
    float2 a = input[gid], Wb = cmul(float2(tw.x, -tw.y), input[gid + halfN]);
    uint out = group * stride * 2 + k;
    output[out]          = a + Wb;
    output[out + stride] = a - Wb;
}

kernel void scale_by_inv_N(
    device float2 *data [[buffer(0)]], constant float &inv_N [[buffer(1)]],
    constant uint &count [[buffer(2)]], uint gid [[thread_position_in_grid]]
) {
    if (gid >= count) return;
    data[gid] *= inv_N;
}
"""

// MARK: - Setup

guard let device = MTLCreateSystemDefaultDevice() else { fatalError("No Metal device") }
assert(device.hasUnifiedMemory, "Requires Apple Silicon")

let opts = MTLCompileOptions(); opts.mathMode = .fast
let library = try device.makeLibrary(source: shaderSource, options: opts)
guard let queue = device.makeCommandQueue() else { fatalError("No command queue") }

let fwdPip   = try device.makeComputePipelineState(function: library.makeFunction(name: "stockham_radix2")!)
let invPip   = try device.makeComputePipelineState(function: library.makeFunction(name: "stockham_radix2_inverse")!)
let scalePip = try device.makeComputePipelineState(function: library.makeFunction(name: "scale_by_inv_N")!)

// MARK: - Resource caches

var twiddleCache: [Int: MTLBuffer] = [:]
var scratchCache: [Int: MTLBuffer] = [:]

func twiddle(_ N: Int) -> MTLBuffer {
    if let c = twiddleCache[N] { return c }
    let halfN = N / 2
    let buf = device.makeBuffer(length: halfN * 8, options: .storageModeShared)!
    let p = buf.contents().bindMemory(to: SIMD2<Float>.self, capacity: halfN)
    for j in 0..<halfN {
        let a = -2.0 * Float.pi * Float(j) / Float(N)
        p[j] = SIMD2(cos(a), sin(a))
    }
    twiddleCache[N] = buf
    return buf
}

func scratch(_ bytes: Int) -> MTLBuffer {
    if let c = scratchCache[bytes] { return c }
    let buf = device.makeBuffer(length: bytes, options: .storageModeShared)!
    scratchCache[bytes] = buf
    return buf
}

// MARK: - FFT: single command buffer, all passes + scale in one submission

func runFFT(input: MTLBuffer, output: MTLBuffer, N: Int, inverse: Bool) {
    let passes = Int(log2(Double(N)))
    let pip = inverse ? invPip : fwdPip
    let tg = min(N / 2, fwdPip.maxTotalThreadsPerThreadgroup)
    let bytes = N * 8
    let tw = twiddle(N)
    let scr = scratch(bytes)

    guard let cb = queue.makeCommandBuffer() else { return }
    cb.label = inverse ? "ifft_\(N)" : "fft_\(N)"

    // Copy input → scratch (preserves input immutability)
    if let blit = cb.makeBlitCommandEncoder() {
        blit.copy(from: input, sourceOffset: 0, to: scr, destinationOffset: 0, size: bytes)
        blit.endEncoding()
    }

    // All passes — encoder boundaries provide implicit memory barriers
    for pass in 0..<passes {
        let src = (pass % 2 == 0) ? scr : output
        let dst = (pass % 2 == 0) ? output : scr
        guard let enc = cb.makeComputeCommandEncoder() else { return }
        enc.setComputePipelineState(pip)
        enc.setBuffer(src, offset: 0, index: 0); enc.setBuffer(dst, offset: 0, index: 1)
        var n = UInt32(N), p = UInt32(pass)
        enc.setBytes(&n, length: 4, index: 2); enc.setBytes(&p, length: 4, index: 3)
        enc.setBuffer(tw, offset: 0, index: 4)
        enc.dispatchThreads(MTLSize(width: N/2, height: 1, depth: 1),
                            threadsPerThreadgroup: MTLSize(width: tg, height: 1, depth: 1))
        enc.endEncoding()
    }

    // Copy result to output if it landed in scratch
    if passes % 2 == 0, let blit = cb.makeBlitCommandEncoder() {
        blit.copy(from: scr, sourceOffset: 0, to: output, destinationOffset: 0, size: bytes)
        blit.endEncoding()
    }

    // Inverse: scale by 1/N
    if inverse, let enc = cb.makeComputeCommandEncoder() {
        enc.setComputePipelineState(scalePip)
        enc.setBuffer(output, offset: 0, index: 0)
        var invN = 1.0 / Float(N), count = UInt32(N)
        enc.setBytes(&invN, length: 4, index: 1); enc.setBytes(&count, length: 4, index: 2)
        enc.dispatchThreads(MTLSize(width: N, height: 1, depth: 1),
                            threadsPerThreadgroup: MTLSize(width: min(N, scalePip.maxTotalThreadsPerThreadgroup), height: 1, depth: 1))
        enc.endEncoding()
    }

    cb.commit(); cb.waitUntilCompleted()
    if cb.status == .error { fatalError("GPU [\(cb.label ?? "fft")]: \(cb.error?.localizedDescription ?? "?")") }
}

// MARK: - vDSP reference

func vdspFFT(_ input: [SIMD2<Float>], forward: Bool) -> [SIMD2<Float>] {
    let N = input.count, log2N = vDSP_Length(log2(Double(N)))
    guard let setup = vDSP_create_fftsetup(log2N, FFTRadix(kFFTRadix2)) else { return input }
    var real = input.map { $0.x }, imag = input.map { $0.y }
    let dir = forward ? FFTDirection(kFFTDirection_Forward) : FFTDirection(kFFTDirection_Inverse)
    real.withUnsafeMutableBufferPointer { rBuf in
        imag.withUnsafeMutableBufferPointer { iBuf in
            var sc = DSPSplitComplex(realp: rBuf.baseAddress!, imagp: iBuf.baseAddress!)
            vDSP_fft_zip(setup, &sc, 1, log2N, dir)
        }
    }
    vDSP_destroy_fftsetup(setup)
    if !forward {
        var s = 1.0 / Float(N)
        vDSP_vsmul(real, 1, &s, &real, 1, vDSP_Length(N))
        vDSP_vsmul(imag, 1, &s, &imag, 1, vDSP_Length(N))
    }
    return (0..<N).map { SIMD2(real[$0], imag[$0]) }
}

print("Exercise 5: Stockham FFT — GPU: \(device.name)\n")

// MARK: - Test 1: Impulse (N=8)

do {
    let N = 8, sz = N * 8
    let bufIn  = device.makeBuffer(length: sz, options: .storageModeShared)!
    let bufOut = device.makeBuffer(length: sz, options: .storageModeShared)!
    let p = bufIn.contents().bindMemory(to: SIMD2<Float>.self, capacity: N)
    for i in 0..<N { p[i] = .zero }; p[0] = SIMD2(1, 0)

    runFFT(input: bufIn, output: bufOut, N: N, inverse: false)

    // FFT(impulse) = all ones
    let r = bufOut.contents().bindMemory(to: SIMD2<Float>.self, capacity: N)
    var maxErr: Float = 0
    for i in 0..<N { maxErr = max(maxErr, max(abs(r[i].x - 1), abs(r[i].y))) }

    // Verify input immutability
    let pIn = bufIn.contents().bindMemory(to: SIMD2<Float>.self, capacity: N)
    var inputOk = abs(pIn[0].x - 1) < 1e-6
    for i in 1..<N { if abs(pIn[i].x) > 1e-6 || abs(pIn[i].y) > 1e-6 { inputOk = false } }

    print("  impulse N=8:     err = \(maxErr)  \(maxErr < 1e-5 ? "✓" : "✗")  input preserved: \(inputOk ? "✓" : "✗")")
}

// MARK: - Test 2: Sinusoid peak detection (N=64)

do {
    let N = 64, sz = N * 8
    let bufIn  = device.makeBuffer(length: sz, options: .storageModeShared)!
    let bufOut = device.makeBuffer(length: sz, options: .storageModeShared)!
    let p = bufIn.contents().bindMemory(to: SIMD2<Float>.self, capacity: N)
    for i in 0..<N { p[i] = SIMD2(cos(2.0 * .pi * 3.0 * Float(i) / Float(N)), 0) }

    runFFT(input: bufIn, output: bufOut, N: N, inverse: false)

    let r = bufOut.contents().bindMemory(to: SIMD2<Float>.self, capacity: N)
    var maxMag: Float = 0, peak = 0
    for i in 0..<N { let m = sqrt(r[i].x*r[i].x + r[i].y*r[i].y); if m > maxMag { maxMag = m; peak = i } }
    print("  sinusoid bin 3:  peak = \(peak)  \(peak == 3 ? "✓" : "✗")")
}

// MARK: - Test 3: Round-trip (N=256)

do {
    let N = 256, sz = N * 8
    let bufIn  = device.makeBuffer(length: sz, options: .storageModeShared)!
    let bufMid = device.makeBuffer(length: sz, options: .storageModeShared)!
    let bufOut = device.makeBuffer(length: sz, options: .storageModeShared)!

    let p = bufIn.contents().bindMemory(to: SIMD2<Float>.self, capacity: N)
    srand48(42)
    var orig = [SIMD2<Float>]()
    for i in 0..<N { let v = SIMD2<Float>(Float(drand48()-0.5), Float(drand48()-0.5)); p[i] = v; orig.append(v) }

    runFFT(input: bufIn, output: bufMid, N: N, inverse: false)
    runFFT(input: bufMid, output: bufOut, N: N, inverse: true)

    let r = bufOut.contents().bindMemory(to: SIMD2<Float>.self, capacity: N)
    var maxErr: Float = 0
    for i in 0..<N { maxErr = max(maxErr, max(abs(r[i].x - orig[i].x), abs(r[i].y - orig[i].y))) }
    print("  round-trip:      err = \(maxErr)  \(maxErr < 1e-4 ? "✓" : "✗")")
}

// MARK: - Test 4: GPU vs vDSP (N=1024)

do {
    let N = 1024, sz = N * 8
    let bufIn  = device.makeBuffer(length: sz, options: .storageModeShared)!
    let bufOut = device.makeBuffer(length: sz, options: .storageModeShared)!
    let p = bufIn.contents().bindMemory(to: SIMD2<Float>.self, capacity: N)
    srand48(77)
    var cpu = [SIMD2<Float>]()
    for i in 0..<N { let v = SIMD2<Float>(Float(drand48()-0.5), Float(drand48()-0.5)); p[i] = v; cpu.append(v) }

    runFFT(input: bufIn, output: bufOut, N: N, inverse: false)
    let gpu = bufOut.contents().bindMemory(to: SIMD2<Float>.self, capacity: N)
    let ref = vdspFFT(cpu, forward: true)

    var maxErr: Float = 0, norm: Float = 0
    for i in 0..<N {
        maxErr = max(maxErr, max(abs(gpu[i].x - ref[i].x), abs(gpu[i].y - ref[i].y)))
        norm += ref[i].x*ref[i].x + ref[i].y*ref[i].y
    }
    let rel = maxErr / sqrt(norm)
    print("  vs vDSP N=1024:  rel = \(rel)  \(rel < 1e-5 ? "✓" : "✗")")
}

// MARK: - Test 5: Parseval's theorem (N=512)

do {
    let N = 512, sz = N * 8
    let bufIn  = device.makeBuffer(length: sz, options: .storageModeShared)!
    let bufOut = device.makeBuffer(length: sz, options: .storageModeShared)!
    let p = bufIn.contents().bindMemory(to: SIMD2<Float>.self, capacity: N)
    srand48(123)
    var tE: Float = 0
    for i in 0..<N { let v = SIMD2<Float>(Float(drand48()-0.5), Float(drand48()-0.5)); p[i] = v; tE += v.x*v.x + v.y*v.y }

    runFFT(input: bufIn, output: bufOut, N: N, inverse: false)
    let r = bufOut.contents().bindMemory(to: SIMD2<Float>.self, capacity: N)
    var fE: Float = 0
    for i in 0..<N { fE += r[i].x*r[i].x + r[i].y*r[i].y }
    fE /= Float(N)

    let diff = abs(tE - fE) / tE
    print("  Parseval N=512:  diff = \(diff)  \(diff < 1e-4 ? "✓" : "✗")")
}

// MARK: - Test 6: Throughput

do {
    print("\n  Throughput:")
    print("         N   passes   GPU ms    vDSP ms    ratio")

    for N in [256, 1024, 4096, 16384, 65536, 262144] {
        let sz = N * 8, reps = N <= 4096 ? 100 : 20
        let A = device.makeBuffer(length: sz, options: .storageModeShared)!
        let B = device.makeBuffer(length: sz, options: .storageModeShared)!
        let p = A.contents().bindMemory(to: SIMD2<Float>.self, capacity: N)
        for i in 0..<N { p[i] = SIMD2(1, 0) }

        runFFT(input: A, output: B, N: N, inverse: false) // warmup

        let t0 = CFAbsoluteTimeGetCurrent()
        for _ in 0..<reps { runFFT(input: A, output: B, N: N, inverse: false) }
        let gpuMs = (CFAbsoluteTimeGetCurrent() - t0) * 1000 / Double(reps)

        let cpu = [SIMD2<Float>](repeating: SIMD2(1,0), count: N)
        let _ = vdspFFT(cpu, forward: true)
        let t1 = CFAbsoluteTimeGetCurrent()
        for _ in 0..<reps { let _ = vdspFFT(cpu, forward: true) }
        let vMs = (CFAbsoluteTimeGetCurrent() - t1) * 1000 / Double(reps)

        let ratio = vMs / gpuMs
        let winner = ratio > 1 ? "GPU" : "vDSP"
        print(String(format: "  %8d    %5d  %7.3f   %7.3f    %5.1fx", N, Int(log2(Double(N))), gpuMs, vMs, ratio) + " \(winner)")
    }
}
