#!/usr/bin/env swift
//
// Exercise 5: 1D FFT — Stockham Radix-2 on Metal GPU
//
// WHAT YOU'LL LEARN:
//   - The Stockham auto-sort FFT algorithm (no bit-reversal!)
//   - Multi-pass GPU algorithms (log₂(N) kernel dispatches)
//   - Precomputed twiddle tables (eliminates expensive per-thread sin/cos)
//   - Butterfly operations using cmul from Exercise 1
//   - Validating against Accelerate's vDSP FFT
//
// KEY OPTIMIZATION: Twiddle factor precomputation
//   Computing sin/cos per butterfly costs 4-8 GPU ALU cycles each.
//   For N=65536 with 16 passes, that's 524K transcendental evaluations.
//   A precomputed twiddle table replaces all of them with cache-friendly loads.
//   twiddle[j] = exp(-2πij/N) for j=0..N/2-1, indexed per pass.
//
// COMPILE & RUN:
//   swiftc -O -framework Metal -framework Foundation -framework Accelerate \
//       ex05_stockham_fft.swift -o ex05
//   ./ex05
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

// ─── Stockham radix-2 butterfly pass (precomputed twiddle) ────────
//
// Twiddle table: twiddle[j] = exp(-2πij/N) for j=0..N/2-1
// For pass p with stride=2^p, position k uses:
//   W = twiddle[k * (N / (2 * stride))]
// This replaces per-butterfly cos/sin with a single table load.

kernel void stockham_radix2(
    device const float2 *input   [[buffer(0)]],
    device float2       *output  [[buffer(1)]],
    constant uint       &N       [[buffer(2)]],
    constant uint       &pass    [[buffer(3)]],
    device const float2 *twiddle [[buffer(4)]],
    uint gid [[thread_position_in_grid]]
) {
    uint halfN = N / 2;
    if (gid >= halfN) return;

    uint stride = 1u << pass;
    uint group  = gid / stride;
    uint k      = gid % stride;

    float2 a = input[gid];
    float2 b = input[gid + halfN];

    // Table lookup instead of sin/cos
    uint tw_idx = k * (halfN / stride);
    float2 W = twiddle[tw_idx];

    float2 Wb = cmul(W, b);
    uint out_base = group * stride * 2 + k;
    output[out_base]          = a + Wb;
    output[out_base + stride] = a - Wb;
}

// ─── Inverse FFT pass (conjugate twiddle from same table) ────────

kernel void stockham_radix2_inverse(
    device const float2 *input   [[buffer(0)]],
    device float2       *output  [[buffer(1)]],
    constant uint       &N       [[buffer(2)]],
    constant uint       &pass    [[buffer(3)]],
    device const float2 *twiddle [[buffer(4)]],
    uint gid [[thread_position_in_grid]]
) {
    uint halfN = N / 2;
    if (gid >= halfN) return;

    uint stride = 1u << pass;
    uint group  = gid / stride;
    uint k      = gid % stride;

    float2 a = input[gid];
    float2 b = input[gid + halfN];

    uint tw_idx = k * (halfN / stride);
    // INVERSE: conjugate the twiddle factor
    float2 W = float2(twiddle[tw_idx].x, -twiddle[tw_idx].y);

    float2 Wb = cmul(W, b);
    output[group * stride * 2 + k]          = a + Wb;
    output[group * stride * 2 + k + stride] = a - Wb;
}

// ─── Scale kernel (multiply by reciprocal, cheaper than divide) ──

kernel void scale_by_inv_N(
    device float2       *data    [[buffer(0)]],
    constant float      &inv_N   [[buffer(1)]],
    uint gid [[thread_position_in_grid]]
) {
    // Caller passes 1.0/N so we multiply instead of divide per element.
    data[gid] = data[gid] * inv_N;
}
"""

// ═══════════════════════════════════════════════════════════════════════════
// HOST CODE
// ═══════════════════════════════════════════════════════════════════════════

print("""

╔═══════════════════════════════════════════════════════════════╗
║  Exercise 5: Stockham Radix-2 FFT on Metal GPU               ║
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
guard let fwdFunc = library.makeFunction(name: "stockham_radix2"),
      let invFunc = library.makeFunction(name: "stockham_radix2_inverse"),
      let scaleFunc = library.makeFunction(name: "scale_by_inv_N") else {
    print("❌ Kernel not found"); exit(1)
}
let fwdPipeline = try device.makeComputePipelineState(function: fwdFunc)
let invPipeline = try device.makeComputePipelineState(function: invFunc)
let scalePipeline = try device.makeComputePipelineState(function: scaleFunc)

// ── Precompute twiddle table for a given N ─────────────────────────────
// twiddle[j] = exp(-2πij/N) = (cos(2πj/N), -sin(2πj/N)) for j=0..N/2-1

var twiddleCache: [Int: MTLBuffer] = [:]

func getTwiddleBuffer(N: Int) -> MTLBuffer? {
    if let cached = twiddleCache[N] { return cached }
    let halfN = N / 2
    let size = halfN * MemoryLayout<SIMD2<Float>>.stride
    guard let buf = device.makeBuffer(length: size, options: .storageModeShared) else { return nil }
    let ptr = buf.contents().bindMemory(to: SIMD2<Float>.self, capacity: halfN)
    for j in 0..<halfN {
        let angle = -2.0 * Float.pi * Float(j) / Float(N)
        ptr[j] = SIMD2<Float>(cos(angle), sin(angle))
    }
    twiddleCache[N] = buf
    return buf
}

// ── FFT runner: uses a SCRATCH buffer internally, never mutates input ──
//
// API contract: input is read-only, output contains the result.
// Internally allocates a scratch buffer for ping-pong.

func runFFT(input: MTLBuffer, output: MTLBuffer, N: Int, inverse: Bool) {
    let numPasses = Int(log2(Double(N)))
    let pipeline = inverse ? invPipeline : fwdPipeline
    let tgSize = min(N / 2, fwdPipeline.maxTotalThreadsPerThreadgroup)
    let byteSize = N * MemoryLayout<SIMD2<Float>>.stride

    guard let twiddleBuf = getTwiddleBuffer(N: N) else { return }

    // Allocate scratch buffer for ping-pong (input is never written to)
    guard let scratch = device.makeBuffer(length: byteSize, options: .storageModeShared) else { return }

    // Copy input → scratch (so pass 0 reads from scratch, writes to output)
    autoreleasepool {
        guard let cb = commandQueue.makeCommandBuffer(),
              let blit = cb.makeBlitCommandEncoder() else { return }
        cb.label = "fft_copy_input"
        blit.copy(from: input, sourceOffset: 0, to: scratch, destinationOffset: 0, size: byteSize)
        blit.endEncoding()
        cb.commit(); cb.waitUntilCompleted()
    }

    // Ping-pong: even passes read scratch → write output
    //            odd passes read output → write scratch
    for pass in 0..<numPasses {
        let src = (pass % 2 == 0) ? scratch : output
        let dst = (pass % 2 == 0) ? output : scratch

        autoreleasepool {
            guard let cmdBuf = commandQueue.makeCommandBuffer(),
                  let enc = cmdBuf.makeComputeCommandEncoder() else { return }
            cmdBuf.label = "fft_pass_\(pass)"
            enc.setComputePipelineState(pipeline)
            enc.setBuffer(src, offset: 0, index: 0)
            enc.setBuffer(dst, offset: 0, index: 1)
            var n = UInt32(N), p = UInt32(pass)
            enc.setBytes(&n, length: 4, index: 2)
            enc.setBytes(&p, length: 4, index: 3)
            enc.setBuffer(twiddleBuf, offset: 0, index: 4)
            enc.dispatchThreads(
                MTLSize(width: N / 2, height: 1, depth: 1),
                threadsPerThreadgroup: MTLSize(width: tgSize, height: 1, depth: 1))
            enc.endEncoding()
            cmdBuf.commit(); cmdBuf.waitUntilCompleted()
            gpuCheck(cmdBuf, label: "fft_pass_\(pass)")
        }
    }

    // After all passes: if numPasses is odd, result is in output (good).
    // If numPasses is even, result is in scratch — copy to output.
    if numPasses % 2 == 0 {
        autoreleasepool {
            guard let cb = commandQueue.makeCommandBuffer(),
                  let blit = cb.makeBlitCommandEncoder() else { return }
            cb.label = "fft_copy_result"
            blit.copy(from: scratch, sourceOffset: 0, to: output, destinationOffset: 0, size: byteSize)
            blit.endEncoding()
            cb.commit(); cb.waitUntilCompleted()
        }
    }

    // For inverse: scale by 1/N (multiply by reciprocal)
    if inverse {
        autoreleasepool {
            guard let cmdBuf = commandQueue.makeCommandBuffer(),
                  let enc = cmdBuf.makeComputeCommandEncoder() else { return }
            cmdBuf.label = "fft_scale"
            enc.setComputePipelineState(scalePipeline)
            enc.setBuffer(output, offset: 0, index: 0)
            var invN = Float(1.0) / Float(N)
            enc.setBytes(&invN, length: MemoryLayout<Float>.size, index: 1)
            enc.dispatchThreads(
                MTLSize(width: N, height: 1, depth: 1),
                threadsPerThreadgroup: MTLSize(width: min(N, scalePipeline.maxTotalThreadsPerThreadgroup), height: 1, depth: 1))
            enc.endEncoding()
            cmdBuf.commit(); cmdBuf.waitUntilCompleted()
            gpuCheck(cmdBuf, label: "fft_scale")
        }
    }
}

// ── Helper: vDSP FFT reference ──────────────────────────────────

func vdspFFT(_ input: [SIMD2<Float>], forward: Bool) -> [SIMD2<Float>] {
    let N = input.count
    let log2N = vDSP_Length(log2(Double(N)))
    guard let setup = vDSP_create_fftsetup(log2N, FFTRadix(kFFTRadix2)) else { return input }

    var realPart = input.map { $0.x }
    var imagPart = input.map { $0.y }
    let direction = forward ? FFTDirection(kFFTDirection_Forward) : FFTDirection(kFFTDirection_Inverse)

    realPart.withUnsafeMutableBufferPointer { realBuf in
        imagPart.withUnsafeMutableBufferPointer { imagBuf in
            var splitComplex = DSPSplitComplex(realp: realBuf.baseAddress!, imagp: imagBuf.baseAddress!)
            vDSP_fft_zip(setup, &splitComplex, 1, log2N, direction)
        }
    }
    vDSP_destroy_fftsetup(setup)

    if !forward {
        var scale = Float(1.0 / Float(N))
        vDSP_vsmul(realPart, 1, &scale, &realPart, 1, vDSP_Length(N))
        vDSP_vsmul(imagPart, 1, &scale, &imagPart, 1, vDSP_Length(N))
    }

    return (0..<N).map { SIMD2<Float>(realPart[$0], imagPart[$0]) }
}

// ═══════════════════════════════════════════════════════════════════════════
// TEST 1: Tiny FFT (N=8) — trace by hand
// ═══════════════════════════════════════════════════════════════════════════

print("── Test 1: Tiny FFT (N=8) ──────────────────────────────────")

do {
    let N = 8
    let size = N * MemoryLayout<SIMD2<Float>>.stride

    guard let bufIn  = device.makeBuffer(length: size, options: .storageModeShared),
          let bufOut = device.makeBuffer(length: size, options: .storageModeShared) else { exit(1) }

    let ptr = bufIn.contents().bindMemory(to: SIMD2<Float>.self, capacity: N)
    for i in 0..<N { ptr[i] = .zero }
    ptr[0] = SIMD2<Float>(1, 0)

    print("  Input: [1, 0, 0, 0, 0, 0, 0, 0]  (impulse)")
    print("  Expected FFT: [1, 1, 1, 1, 1, 1, 1, 1]  (flat spectrum)")

    runFFT(input: bufIn, output: bufOut, N: N, inverse: false)

    // Verify input was not mutated (API contract)
    let ptrIn = bufIn.contents().bindMemory(to: SIMD2<Float>.self, capacity: N)
    var inputMutated = false
    if abs(ptrIn[0].x - 1.0) > 1e-6 || abs(ptrIn[0].y) > 1e-6 { inputMutated = true }
    for i in 1..<N { if abs(ptrIn[i].x) > 1e-6 || abs(ptrIn[i].y) > 1e-6 { inputMutated = true } }

    let result = bufOut.contents().bindMemory(to: SIMD2<Float>.self, capacity: N)
    var maxErr: Float = 0
    for i in 0..<N {
        let err = max(abs(result[i].x - 1.0), abs(result[i].y - 0.0))
        maxErr = max(maxErr, err)
    }
    print("  Max error: \(maxErr)  \(maxErr < 1e-5 ? "✓ PASS" : "✗ FAIL")")
    print("  Input preserved: \(!inputMutated ? "✓" : "✗ INPUT WAS MUTATED")")
}

// ═══════════════════════════════════════════════════════════════════════════
// TEST 2: Known sinusoid — peak at frequency bin
// ═══════════════════════════════════════════════════════════════════════════

print("\n── Test 2: Sinusoid at bin 3 (N=64) ─────────────────────────")

do {
    let N = 64
    let size = N * MemoryLayout<SIMD2<Float>>.stride

    guard let bufIn  = device.makeBuffer(length: size, options: .storageModeShared),
          let bufOut = device.makeBuffer(length: size, options: .storageModeShared) else { exit(1) }

    let ptr = bufIn.contents().bindMemory(to: SIMD2<Float>.self, capacity: N)
    let freq = 3
    for i in 0..<N {
        let angle = 2.0 * Float.pi * Float(freq) * Float(i) / Float(N)
        ptr[i] = SIMD2<Float>(cos(angle), 0)
    }

    runFFT(input: bufIn, output: bufOut, N: N, inverse: false)
    let result = bufOut.contents().bindMemory(to: SIMD2<Float>.self, capacity: N)

    var maxMag: Float = 0, peakBin = 0
    for i in 0..<N {
        let mag = sqrt(result[i].x * result[i].x + result[i].y * result[i].y)
        if mag > maxMag { maxMag = mag; peakBin = i }
    }
    print("  Peak bin: \(peakBin) (expected: 3)  \(peakBin == 3 ? "✓ PASS" : "✗ FAIL")")
}

// ═══════════════════════════════════════════════════════════════════════════
// TEST 3: Round-trip — FFT then IFFT should recover input
// ═══════════════════════════════════════════════════════════════════════════

print("\n── Test 3: FFT → IFFT round-trip (N=256) ────────────────────")

do {
    let N = 256
    let size = N * MemoryLayout<SIMD2<Float>>.stride

    guard let bufIn   = device.makeBuffer(length: size, options: .storageModeShared),
          let bufMid  = device.makeBuffer(length: size, options: .storageModeShared),
          let bufOut  = device.makeBuffer(length: size, options: .storageModeShared) else { exit(1) }

    let ptrIn = bufIn.contents().bindMemory(to: SIMD2<Float>.self, capacity: N)
    srand48(42)
    var original = [SIMD2<Float>]()
    for i in 0..<N {
        let val = SIMD2<Float>(Float(drand48() - 0.5), Float(drand48() - 0.5))
        ptrIn[i] = val
        original.append(val)
    }

    runFFT(input: bufIn, output: bufMid, N: N, inverse: false)
    runFFT(input: bufMid, output: bufOut, N: N, inverse: true)

    let ptrOut = bufOut.contents().bindMemory(to: SIMD2<Float>.self, capacity: N)
    var maxErr: Float = 0
    for i in 0..<N {
        let err = max(abs(ptrOut[i].x - original[i].x), abs(ptrOut[i].y - original[i].y))
        maxErr = max(maxErr, err)
    }
    print("  Max error: \(maxErr)  \(maxErr < 1e-4 ? "✓ PASS" : "✗ FAIL")")
}

// ═══════════════════════════════════════════════════════════════════════════
// TEST 4: Compare against vDSP
// ═══════════════════════════════════════════════════════════════════════════

print("\n── Test 4: GPU FFT vs vDSP (N=1024) ─────────────────────────")

do {
    let N = 1024
    let size = N * MemoryLayout<SIMD2<Float>>.stride

    guard let bufIn  = device.makeBuffer(length: size, options: .storageModeShared),
          let bufOut = device.makeBuffer(length: size, options: .storageModeShared) else { exit(1) }

    let ptrIn = bufIn.contents().bindMemory(to: SIMD2<Float>.self, capacity: N)
    srand48(77)
    var cpuInput = [SIMD2<Float>]()
    for i in 0..<N {
        let val = SIMD2<Float>(Float(drand48() - 0.5), Float(drand48() - 0.5))
        ptrIn[i] = val; cpuInput.append(val)
    }

    runFFT(input: bufIn, output: bufOut, N: N, inverse: false)
    let gpuResult = bufOut.contents().bindMemory(to: SIMD2<Float>.self, capacity: N)
    let cpuResult = vdspFFT(cpuInput, forward: true)

    var maxErr: Float = 0, norm: Float = 0
    for i in 0..<N {
        maxErr = max(maxErr, max(abs(gpuResult[i].x - cpuResult[i].x), abs(gpuResult[i].y - cpuResult[i].y)))
        norm += cpuResult[i].x * cpuResult[i].x + cpuResult[i].y * cpuResult[i].y
    }
    let relErr = maxErr / sqrt(norm)
    print("  Rel err: \(relErr)  \(relErr < 1e-5 ? "✓ PASS" : relErr < 1e-3 ? "⚠ MARGINAL" : "✗ FAIL")")
}

// ═══════════════════════════════════════════════════════════════════════════
// TEST 5: Parseval's theorem — energy preserved
// ═══════════════════════════════════════════════════════════════════════════

print("\n── Test 5: Parseval's theorem (energy conservation) ─────────")

do {
    let N = 512
    let size = N * MemoryLayout<SIMD2<Float>>.stride

    guard let bufIn  = device.makeBuffer(length: size, options: .storageModeShared),
          let bufOut = device.makeBuffer(length: size, options: .storageModeShared) else { exit(1) }

    let ptrIn = bufIn.contents().bindMemory(to: SIMD2<Float>.self, capacity: N)
    srand48(123)
    var timeEnergy: Float = 0
    for i in 0..<N {
        let v = SIMD2<Float>(Float(drand48()-0.5), Float(drand48()-0.5))
        ptrIn[i] = v
        timeEnergy += v.x*v.x + v.y*v.y
    }

    runFFT(input: bufIn, output: bufOut, N: N, inverse: false)
    let ptrOut = bufOut.contents().bindMemory(to: SIMD2<Float>.self, capacity: N)
    var freqEnergy: Float = 0
    for i in 0..<N { freqEnergy += ptrOut[i].x * ptrOut[i].x + ptrOut[i].y * ptrOut[i].y }
    freqEnergy /= Float(N)

    let relDiff = abs(timeEnergy - freqEnergy) / timeEnergy
    print("  Time energy: \(timeEnergy)  Freq energy: \(freqEnergy)")
    print("  Rel diff: \(relDiff)  \(relDiff < 1e-4 ? "✓ PASS" : "✗ FAIL")")
}

// ═══════════════════════════════════════════════════════════════════════════
// TEST 6: Performance — GPU timestamps
// ═══════════════════════════════════════════════════════════════════════════

print("\n── Test 6: FFT throughput (GPU timestamps) ──────────────────")

do {
    let sizes = [256, 1024, 4096, 16384, 65536, 262144]

    print("         N   Passes   GPU (ms)   vDSP (ms)   GPU vs vDSP")
    print("  " + String(repeating: "─", count: 58))

    for N in sizes {
        let numPasses = Int(log2(Double(N)))
        let size = N * MemoryLayout<SIMD2<Float>>.stride
        let reps = N <= 4096 ? 100 : 20

        guard let bufA = device.makeBuffer(length: size, options: .storageModeShared),
              let bufB = device.makeBuffer(length: size, options: .storageModeShared) else { continue }
        let ptr = bufA.contents().bindMemory(to: SIMD2<Float>.self, capacity: N)
        for i in 0..<N { ptr[i] = SIMD2<Float>(1,0) }

        // Warmup
        runFFT(input: bufA, output: bufB, N: N, inverse: false)

        // GPU timing
        let t0 = CFAbsoluteTimeGetCurrent()
        for _ in 0..<reps {
            runFFT(input: bufA, output: bufB, N: N, inverse: false)
        }
        let gpuMs = (CFAbsoluteTimeGetCurrent() - t0) * 1000.0 / Double(reps)

        // vDSP timing
        let cpuData = [SIMD2<Float>](repeating: SIMD2(1,0), count: N)
        let _ = vdspFFT(cpuData, forward: true)
        let t1 = CFAbsoluteTimeGetCurrent()
        for _ in 0..<reps {
            let _ = vdspFFT(cpuData, forward: true)
        }
        let vdspMs = (CFAbsoluteTimeGetCurrent() - t1) * 1000.0 / Double(reps)

        let ratio = vdspMs / gpuMs
        let winner = ratio > 1.0 ? "(GPU wins)" : "(vDSP wins)"
        print(String(format: "  %8d   %5d    %7.3f    %7.3f      %.2fx",
                     N, numPasses, gpuMs, vdspMs, ratio) + " \(winner)")
    }
}

print("""

═══════════════════════════════════════════════════════════════
  Exercise 5 complete.

  What you learned:
    • Stockham FFT: ping-pong between scratch/output, natural order
    • Precomputed twiddle table eliminates per-butterfly sin/cos
    • Input buffer is never mutated (clean API contract)
    • Scale uses multiply-by-reciprocal (cheaper than per-element divide)
    • Parseval's theorem: ||x||² = (1/N)||X||² (energy conservation)

  NEXT: Exercise 6 — Batched 3D FFT from 1D transforms.
═══════════════════════════════════════════════════════════════
""")
