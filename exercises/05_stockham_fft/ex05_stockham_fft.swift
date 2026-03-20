#!/usr/bin/env swift
//
// Exercise 5: 1D FFT — Stockham Radix-2 on Metal GPU
//
// WHAT YOU'LL LEARN:
//   - The Stockham auto-sort FFT algorithm (no bit-reversal!)
//   - Multi-pass GPU algorithms (log₂(N) kernel dispatches)
//   - Twiddle factors: W_N^k = exp(-2πik/N) = cos(θ) - i·sin(θ)
//   - Butterfly operations using cmul from Exercise 1
//   - Validating against Accelerate's vDSP FFT
//
// WHY THIS MATTERS:
//   - FFT accounts for 50% of QE's wall time (5.38s of 10.85s)
//   - There is NO Metal equivalent of cuFFT or FFTW
//   - This is the kernel you'll intercept via DYLD to accelerate QE
//   - 3D FFT (Exercise 6) composes from batched 1D FFTs
//
// THE STOCKHAM ALGORITHM:
//   Unlike Cooley-Tukey which does in-place computation and needs
//   bit-reversal at the end, Stockham uses TWO buffers (ping-pong)
//   and produces output in natural order after each pass.
//
//   For N elements, log₂(N) passes:
//     Pass 0 (stride=1):   pairs separated by 1
//     Pass 1 (stride=2):   pairs separated by 2
//     ...
//     Pass p (stride=2^p): pairs separated by 2^p
//
//   Each pass: N/2 butterflies, each reading 2 inputs, writing 2 outputs.
//   Butterfly: out_even = in_a + W * in_b
//              out_odd  = in_a - W * in_b
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

// ─── Stockham radix-2 butterfly pass ─────────────────────────────
//
// One dispatch of this kernel performs ONE pass of the Stockham FFT.
// For a length-N FFT, you dispatch this log₂(N) times, alternating
// which buffer is input and which is output (ping-pong).
//
// Parameters:
//   input:  source buffer (read-only this pass)
//   output: destination buffer (write-only this pass)
//   N:      total FFT length (must be power of 2)
//   pass:   which pass (0, 1, 2, ... log₂(N)-1)
//
// Thread mapping: one thread per butterfly (N/2 threads total).
//
// The Stockham indexing:
//   For pass p, stride = 2^p, halfN = N/2
//
//   Thread j computes one butterfly:
//     group_size = stride * 2     (how many elements per group this pass)
//     group  = j / stride         (which group)
//     pos    = j % stride         (position within group)
//
//     idx_a = group * group_size + pos          (first input)
//     idx_b = group * group_size + pos + stride (second input)
//
//     twiddle = exp(-2πi * pos / group_size)
//
//     out[group * group_size + pos]            = a + W*b  (even output)
//     out[group * group_size + pos + halfGroup] = a - W*b (odd output)
//
//   Wait — that's Cooley-Tukey indexing. Stockham is different.
//   Stockham reorganizes so the OUTPUT is in sequential order:
//
//     Thread j (0 ≤ j < N/2):
//       stride = 1 << pass
//       group  = j / stride
//       k      = j % stride       (position within butterfly group)
//
//       Read from input:
//         a = input[j]             (first half of input)
//         b = input[j + N/2]       (second half of input)
//
//       Twiddle factor:
//         angle = -2π * k / (stride * 2)
//         W = (cos(angle), sin(angle))
//
//       Write to output (interleaved for natural order):
//         output[group * stride * 2 + k]          = a + W*b
//         output[group * stride * 2 + k + stride] = a - W*b

kernel void stockham_radix2(
    device const float2 *input  [[buffer(0)]],
    device float2       *output [[buffer(1)]],
    constant uint       &N      [[buffer(2)]],
    constant uint       &pass   [[buffer(3)]],
    uint gid [[thread_position_in_grid]]
) {
    uint halfN = N / 2;
    if (gid >= halfN) return;

    uint stride = 1u << pass;
    uint group  = gid / stride;
    uint k      = gid % stride;

    // Read inputs from first and second halves
    float2 a = input[gid];
    float2 b = input[gid + halfN];

    // Twiddle factor: W = exp(-2πi * k / (2 * stride))
    float angle = -2.0f * M_PI_F * float(k) / float(stride * 2);
    float2 W = float2(cos(angle), sin(angle));

    // Butterfly
    float2 Wb = cmul(W, b);
    float2 even = a + Wb;
    float2 odd  = a - Wb;

    // Write to output in Stockham order (natural order after final pass)
    uint out_base = group * stride * 2 + k;
    output[out_base]          = even;
    output[out_base + stride] = odd;
}

// ─── Inverse FFT pass (same butterfly, conjugate twiddle) ────────

kernel void stockham_radix2_inverse(
    device const float2 *input  [[buffer(0)]],
    device float2       *output [[buffer(1)]],
    constant uint       &N      [[buffer(2)]],
    constant uint       &pass   [[buffer(3)]],
    uint gid [[thread_position_in_grid]]
) {
    uint halfN = N / 2;
    if (gid >= halfN) return;

    uint stride = 1u << pass;
    uint group  = gid / stride;
    uint k      = gid % stride;

    float2 a = input[gid];
    float2 b = input[gid + halfN];

    // INVERSE: positive angle (conjugate twiddle)
    float angle = 2.0f * M_PI_F * float(k) / float(stride * 2);
    float2 W = float2(cos(angle), sin(angle));

    float2 Wb = cmul(W, b);
    output[group * stride * 2 + k]          = a + Wb;
    output[group * stride * 2 + k + stride] = a - Wb;
}

// ─── Scale kernel for inverse FFT (divide by N) ─────────────────

kernel void scale_by_N(
    device float2 *data   [[buffer(0)]],
    constant uint &N      [[buffer(1)]],
    uint gid [[thread_position_in_grid]]
) {
    if (gid >= N) return;
    data[gid] = data[gid] / float(N);
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
print("GPU: \(device.name)")

let library: MTLLibrary
do { library = try device.makeLibrary(source: shaderSource, options: nil); print("✓ Shaders compiled\n") }
catch { print("❌ \(error)"); exit(1) }

guard let commandQueue = device.makeCommandQueue() else { exit(1) }
guard let fwdFunc = library.makeFunction(name: "stockham_radix2"),
      let invFunc = library.makeFunction(name: "stockham_radix2_inverse"),
      let scaleFunc = library.makeFunction(name: "scale_by_N") else {
    print("❌ Kernel not found"); exit(1)
}
let fwdPipeline = try device.makeComputePipelineState(function: fwdFunc)
let invPipeline = try device.makeComputePipelineState(function: invFunc)
let scalePipeline = try device.makeComputePipelineState(function: scaleFunc)

// ── Helper: run full FFT (forward or inverse) with ping-pong ────

func runFFT(input: MTLBuffer, output: MTLBuffer, N: Int, inverse: Bool) {
    let numPasses = Int(log2(Double(N)))
    let pipeline = inverse ? invPipeline : fwdPipeline
    let tgSize = min(N / 2, fwdPipeline.maxTotalThreadsPerThreadgroup)

    // Ping-pong between two buffers.
    // Even passes: read from bufA, write to bufB
    // Odd passes: read from bufB, write to bufA
    // After all passes, result is in whichever buffer was last written.
    var bufA = input
    var bufB = output

    for pass in 0..<numPasses {
        guard let cmdBuf = commandQueue.makeCommandBuffer(),
              let enc = cmdBuf.makeComputeCommandEncoder() else { return }

        enc.setComputePipelineState(pipeline)
        enc.setBuffer(bufA, offset: 0, index: 0)  // read from A
        enc.setBuffer(bufB, offset: 0, index: 1)  // write to B
        var n = UInt32(N), p = UInt32(pass)
        enc.setBytes(&n, length: 4, index: 2)
        enc.setBytes(&p, length: 4, index: 3)
        enc.dispatchThreads(
            MTLSize(width: N / 2, height: 1, depth: 1),
            threadsPerThreadgroup: MTLSize(width: tgSize, height: 1, depth: 1))
        enc.endEncoding()
        cmdBuf.commit()
        cmdBuf.waitUntilCompleted()

        // Swap buffers for next pass
        swap(&bufA, &bufB)
    }

    // After the loop, where is the result?
    // Each pass reads bufA, writes bufB, then swaps.
    // After pass 0: last write to original output → swap → bufA=output
    // After pass 1: last write to original input  → swap → bufA=input
    // After pass 2: last write to original output → swap → bufA=output
    // Pattern: odd passes → result in output (no copy needed)
    //          even passes → result in input (copy to output)
    if numPasses % 2 == 0 {
        // Result ended up in the input buffer — copy to output
        guard let cmdBuf = commandQueue.makeCommandBuffer(),
              let blit = cmdBuf.makeBlitCommandEncoder() else { return }
        blit.copy(from: input, sourceOffset: 0, to: output, destinationOffset: 0,
                  size: N * MemoryLayout<SIMD2<Float>>.stride)
        blit.endEncoding()
        cmdBuf.commit()
        cmdBuf.waitUntilCompleted()
    }

    // For inverse: scale by 1/N
    if inverse {
        guard let cmdBuf = commandQueue.makeCommandBuffer(),
              let enc = cmdBuf.makeComputeCommandEncoder() else { return }
        enc.setComputePipelineState(scalePipeline)
        enc.setBuffer(output, offset: 0, index: 0)
        var n = UInt32(N)
        enc.setBytes(&n, length: 4, index: 1)
        enc.dispatchThreads(
            MTLSize(width: N, height: 1, depth: 1),
            threadsPerThreadgroup: MTLSize(width: min(N, scalePipeline.maxTotalThreadsPerThreadgroup), height: 1, depth: 1))
        enc.endEncoding()
        cmdBuf.commit()
        cmdBuf.waitUntilCompleted()
    }
}

// ── Helper: vDSP FFT reference ──────────────────────────────────

func vdspFFT(_ input: [SIMD2<Float>], forward: Bool) -> [SIMD2<Float>] {
    let N = input.count
    let log2N = vDSP_Length(log2(Double(N)))
    guard let setup = vDSP_create_fftsetup(log2N, FFTRadix(kFFTRadix2)) else { return input }

    // vDSP wants split complex: separate real and imag arrays
    var realPart = input.map { $0.x }
    var imagPart = input.map { $0.y }

    let direction = forward ? FFTDirection(kFFTDirection_Forward) : FFTDirection(kFFTDirection_Inverse)

    // Use withUnsafeMutableBufferPointer to guarantee pointer lifetime
    // spans the entire vDSP call. The &realPart shorthand creates a
    // temporary pointer that is only valid for ONE expression — using
    // it across multiple statements is undefined behavior.
    realPart.withUnsafeMutableBufferPointer { realBuf in
        imagPart.withUnsafeMutableBufferPointer { imagBuf in
            var splitComplex = DSPSplitComplex(
                realp: realBuf.baseAddress!,
                imagp: imagBuf.baseAddress!
            )
            vDSP_fft_zip(setup, &splitComplex, 1, log2N, direction)
        }
    }
    vDSP_destroy_fftsetup(setup)

    // vDSP doesn't scale on inverse — we need to divide by N manually
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

    // Input: [1, 0, 0, 0, 0, 0, 0, 0] — impulse
    // FFT of impulse = all ones: [1, 1, 1, 1, 1, 1, 1, 1]
    for i in 0..<N { ptr[i] = .zero }
    ptr[0] = SIMD2<Float>(1, 0)

    print("  Input: [1, 0, 0, 0, 0, 0, 0, 0]  (impulse)")
    print("  Expected FFT: [1, 1, 1, 1, 1, 1, 1, 1]  (flat spectrum)")
    print("  Passes: \(Int(log2(Double(N)))) (log₂(8) = 3)")
    print("")

    runFFT(input: bufIn, output: bufOut, N: N, inverse: false)

    let result = bufOut.contents().bindMemory(to: SIMD2<Float>.self, capacity: N)
    var maxErr: Float = 0
    print("  GPU result:")
    for i in 0..<N {
        let err = max(abs(result[i].x - 1.0), abs(result[i].y - 0.0))
        maxErr = max(maxErr, err)
        print("    X[\(i)] = (\(String(format: "%.4f", result[i].x)), \(String(format: "%.4f", result[i].y)))")
    }
    print("  Max error: \(maxErr)  \(maxErr < 1e-5 ? "✓ PASS" : "✗ FAIL")")
}

// ═══════════════════════════════════════════════════════════════════════════
// TEST 2: Known sinusoid — FFT should show peak at frequency bin
// ═══════════════════════════════════════════════════════════════════════════

print("\n── Test 2: Sinusoid at bin 3 (N=64) ─────────────────────────")

do {
    let N = 64
    let size = N * MemoryLayout<SIMD2<Float>>.stride

    guard let bufIn  = device.makeBuffer(length: size, options: .storageModeShared),
          let bufOut = device.makeBuffer(length: size, options: .storageModeShared) else { exit(1) }

    let ptr = bufIn.contents().bindMemory(to: SIMD2<Float>.self, capacity: N)

    // Input: cos(2π·3·n/N) = real sinusoid at frequency bin 3
    // FFT should have peaks at bin 3 and bin N-3=61 (complex conjugate)
    let freq = 3
    for i in 0..<N {
        let angle = 2.0 * Float.pi * Float(freq) * Float(i) / Float(N)
        ptr[i] = SIMD2<Float>(cos(angle), 0)  // purely real input
    }

    runFFT(input: bufIn, output: bufOut, N: N, inverse: false)
    let result = bufOut.contents().bindMemory(to: SIMD2<Float>.self, capacity: N)

    // Find magnitude at each bin
    var maxMag: Float = 0, peakBin = 0
    for i in 0..<N {
        let mag = sqrt(result[i].x * result[i].x + result[i].y * result[i].y)
        if mag > maxMag { maxMag = mag; peakBin = i }
    }

    let mag3 = sqrt(result[3].x * result[3].x + result[3].y * result[3].y)
    let mag61 = sqrt(result[61].x * result[61].x + result[61].y * result[61].y)

    print("  Input: cos(2π·3·n/64) — sinusoid at frequency bin 3")
    print("  Peak bin: \(peakBin) (expected: 3)")
    print("  |X[3]| = \(String(format: "%.2f", mag3))   (expected: \(N/2) = 32)")
    print("  |X[61]| = \(String(format: "%.2f", mag61))  (expected: \(N/2) = 32, conjugate)")
    print("  \(peakBin == 3 ? "✓ PASS" : "✗ FAIL")")
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

    // Forward FFT
    runFFT(input: bufIn, output: bufMid, N: N, inverse: false)

    // Inverse FFT
    runFFT(input: bufMid, output: bufOut, N: N, inverse: true)

    let ptrOut = bufOut.contents().bindMemory(to: SIMD2<Float>.self, capacity: N)
    var maxErr: Float = 0
    for i in 0..<N {
        let err = max(abs(ptrOut[i].x - original[i].x), abs(ptrOut[i].y - original[i].y))
        maxErr = max(maxErr, err)
    }

    print("  Random complex input → FFT → IFFT → compare")
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
        ptrIn[i] = val
        cpuInput.append(val)
    }

    // GPU FFT
    runFFT(input: bufIn, output: bufOut, N: N, inverse: false)
    let gpuResult = bufOut.contents().bindMemory(to: SIMD2<Float>.self, capacity: N)

    // vDSP FFT
    let cpuResult = vdspFFT(cpuInput, forward: true)

    var maxErr: Float = 0
    for i in 0..<N {
        let err = max(abs(gpuResult[i].x - cpuResult[i].x), abs(gpuResult[i].y - cpuResult[i].y))
        maxErr = max(maxErr, err)
    }

    // Compute norm for relative error
    var norm: Float = 0
    for i in 0..<N { norm += cpuResult[i].x * cpuResult[i].x + cpuResult[i].y * cpuResult[i].y }
    norm = sqrt(norm)
    let relErr = maxErr / norm

    print("  GPU vs vDSP: max abs err = \(maxErr), rel err = \(relErr)")
    print("  \(relErr < 1e-5 ? "✓ PASS" : relErr < 1e-3 ? "⚠ MARGINAL" : "✗ FAIL")")
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
    for i in 0..<N { ptrIn[i] = SIMD2<Float>(Float(drand48()-0.5), Float(drand48()-0.5)) }

    // Time-domain energy: Σ|x[n]|²
    var timeEnergy: Float = 0
    for i in 0..<N { timeEnergy += ptrIn[i].x * ptrIn[i].x + ptrIn[i].y * ptrIn[i].y }

    // FFT
    runFFT(input: bufIn, output: bufOut, N: N, inverse: false)
    let ptrOut = bufOut.contents().bindMemory(to: SIMD2<Float>.self, capacity: N)

    // Freq-domain energy: (1/N) Σ|X[k]|²
    var freqEnergy: Float = 0
    for i in 0..<N { freqEnergy += ptrOut[i].x * ptrOut[i].x + ptrOut[i].y * ptrOut[i].y }
    freqEnergy /= Float(N)

    let relDiff = abs(timeEnergy - freqEnergy) / timeEnergy

    print("  Time-domain energy:  \(timeEnergy)")
    print("  Freq-domain energy:  \(freqEnergy)  (scaled by 1/N)")
    print("  Relative difference: \(relDiff)")
    print("  \(relDiff < 1e-4 ? "✓ PASS" : "✗ FAIL")  (Parseval: should be equal)")
}

// ═══════════════════════════════════════════════════════════════════════════
// TEST 6: Performance — GPU FFT throughput
// ═══════════════════════════════════════════════════════════════════════════

print("\n── Test 6: FFT throughput ───────────────────────────────────")

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
        let _ = vdspFFT(cpuData, forward: true) // warmup
        let t1 = CFAbsoluteTimeGetCurrent()
        for _ in 0..<reps {
            let _ = vdspFFT(cpuData, forward: true)
        }
        let vdspMs = (CFAbsoluteTimeGetCurrent() - t1) * 1000.0 / Double(reps)

        let ratio = vdspMs / gpuMs
        print(String(format: "  %8d   %5d    %7.3f    %7.3f      %.2fx %s",
                     N, numPasses, gpuMs, vdspMs, ratio,
                     ratio > 1.0 ? "(GPU wins)" : "(vDSP wins)"))
    }

    print("")
    print("  At small N: dispatch overhead kills GPU (one command buffer per pass)")
    print("  At large N: GPU parallelism pays off")
    print("  Optimization: batch all passes into ONE command buffer (huge speedup)")
}

print("""

═══════════════════════════════════════════════════════════════
  Exercise 5 complete.

  What you learned:
    • Stockham FFT: ping-pong between two buffers, natural order output
    • Twiddle factors: W = exp(-2πik/N), applied via cmul
    • Multi-pass dispatch: log₂(N) kernel launches for length-N FFT
    • Parseval's theorem: ||x||² = (1/N)||X||²  (energy conservation)
    • Inverse FFT = forward with conjugate twiddle + scale by 1/N
    • GPU vs vDSP crossover point

  What's slow about this implementation:
    • One command buffer per pass (dispatch overhead × log₂(N))
    • No shared memory — each butterfly reads from global memory
    • Single radix-2 — mixed radix (2,3,5) handles QE's grid sizes

  The path to a fast FFT:
    ┌────────────────────────────────────────────────────────────┐
    │ 1. Batch all passes into ONE command buffer                │
    │    → eliminates per-pass dispatch overhead                 │
    │ 2. Use threadgroup memory for small transforms             │
    │    → length ≤ 256 fits in 32 KB, do all passes in-kernel  │
    │ 3. Mixed radix (2, 3, 4, 5, 8)                            │
    │    → handles QE grid sizes: 36, 48, 54, 60, 72, 96, etc.  │
    │ 4. Batched 1D FFT for 3D composition                      │
    │    → Exercise 6: dispatch N² 1D FFTs along each axis       │
    └────────────────────────────────────────────────────────────┘

  NEXT: Exercise 6 — Batched 3D FFT from 1D transforms.
═══════════════════════════════════════════════════════════════
""")
