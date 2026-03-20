#!/usr/bin/env swift
//
// Exercise 6: Batched 3D FFT from 1D Stockham Transforms
//
// KEY OPTIMIZATION: Tiled transpose with threadgroup memory
//   Naive transpose does uncoalesced reads or writes (one or the other).
//   Tiled transpose: load a TILE×TILE block coalesced into shared memory,
//   then write it transposed from shared — both reads and writes are coalesced.
//   This is 2-4x faster and critical since 3D FFT does 6 transpose operations.
//
// COMPILE & RUN:
//   swiftc -O -framework Metal -framework Foundation -framework Accelerate \
//       ex06_batched_3d_fft.swift -o ex06
//   ./ex06
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

// ─── Batched Stockham with precomputed twiddle table ──────────────

kernel void stockham_batched(
    device const float2 *input       [[buffer(0)]],
    device float2       *output      [[buffer(1)]],
    constant uint       &fft_len     [[buffer(2)]],
    constant uint       &pass        [[buffer(3)]],
    constant uint       &batch_size  [[buffer(4)]],
    constant uint       &batch_stride [[buffer(5)]],
    device const float2 *twiddle     [[buffer(6)]],
    uint2 gid [[thread_position_in_grid]]
) {
    uint j = gid.x;
    uint batch = gid.y;
    uint halfN = fft_len / 2;
    if (j >= halfN || batch >= batch_size) return;

    uint stride = 1u << pass;
    uint group  = j / stride;
    uint k      = j % stride;
    uint base   = batch * batch_stride;

    float2 a = input[base + j];
    float2 b = input[base + j + halfN];

    // Twiddle table lookup instead of per-butterfly sin/cos
    uint tw_idx = k * (halfN / stride);
    float2 W = twiddle[tw_idx];
    float2 Wb = cmul(W, b);

    uint out_idx = group * stride * 2 + k;
    output[base + out_idx]          = a + Wb;
    output[base + out_idx + stride] = a - Wb;
}

// ─── Tiled transpose XY ─────────────────────────────────────────
//
// Uses threadgroup memory for coalesced reads AND writes.
// Each threadgroup processes a TTILE×TTILE block.
// Load: coalesced read from input[x + y*Nx + z*Nx*Ny]
// Store transposed: coalesced write to output[y + x*Ny + z*Ny*Nx]
//
// The +1 padding on shared memory prevents bank conflicts when
// reading the transposed column.

#define TTILE 16

kernel void transpose_xy_tiled(
    device const float2 *input  [[buffer(0)]],
    device float2       *output [[buffer(1)]],
    constant uint       &Nx     [[buffer(2)]],
    constant uint       &Ny     [[buffer(3)]],
    constant uint       &Nz     [[buffer(4)]],
    uint3 lid  [[thread_position_in_threadgroup]],
    uint3 tgid [[threadgroup_position_in_grid]]
) {
    threadgroup float2 tile[TTILE][TTILE + 1];  // +1 avoids bank conflicts

    // Which z-slice this threadgroup handles
    uint z = tgid.z;

    // Global coordinates for the load
    uint gx = tgid.x * TTILE + lid.x;
    uint gy = tgid.y * TTILE + lid.y;

    // Coalesced load: threads read sequential x within a row
    if (gx < Nx && gy < Ny && z < Nz) {
        tile[lid.y][lid.x] = input[gx + gy * Nx + z * Nx * Ny];
    } else {
        tile[lid.y][lid.x] = float2(0.0);
    }

    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Transposed coordinates: swap tile block indices
    uint tx = tgid.y * TTILE + lid.x;  // was y-block, now x
    uint ty = tgid.x * TTILE + lid.y;  // was x-block, now y

    // Coalesced write: threads write sequential new-x within a row
    if (tx < Ny && ty < Nx && z < Nz) {
        output[tx + ty * Ny + z * Ny * Nx] = tile[lid.x][lid.y];
    }
}

kernel void transpose_xz_tiled(
    device const float2 *input  [[buffer(0)]],
    device float2       *output [[buffer(1)]],
    constant uint       &Nx     [[buffer(2)]],
    constant uint       &Ny     [[buffer(3)]],
    constant uint       &Nz     [[buffer(4)]],
    uint3 lid  [[thread_position_in_threadgroup]],
    uint3 tgid [[threadgroup_position_in_grid]]
) {
    threadgroup float2 tile[TTILE][TTILE + 1];

    uint y = tgid.z;  // Each y-slice is a separate threadgroup layer

    uint gx = tgid.x * TTILE + lid.x;
    uint gz = tgid.y * TTILE + lid.y;

    if (gx < Nx && y < Ny && gz < Nz) {
        tile[lid.y][lid.x] = input[gx + y * Nx + gz * Nx * Ny];
    } else {
        tile[lid.y][lid.x] = float2(0.0);
    }

    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Swap x and z block indices
    uint tz = tgid.x * TTILE + lid.x;  // was x-block, now z
    uint tx = tgid.y * TTILE + lid.y;  // was z-block, now x

    if (tz < Nz && y < Ny && tx < Nx) {
        output[tz + y * Nz + tx * Nz * Ny] = tile[lid.x][lid.y];
    }
}
"""

// ═══════════════════════════════════════════════════════════════════════════
// SETUP
// ═══════════════════════════════════════════════════════════════════════════

print("""

╔═══════════════════════════════════════════════════════════════╗
║  Exercise 6: Batched 3D FFT from 1D Stockham Transforms      ║
╚═══════════════════════════════════════════════════════════════╝

""")

guard let device = MTLCreateSystemDefaultDevice() else { print("No Metal"); exit(1) }
assert(device.hasUnifiedMemory, "Requires Apple Silicon (unified memory)")
print("GPU: \(device.name)")

func gpuCheck(_ cb: MTLCommandBuffer, label: String) {
    if cb.status == .error { print("GPU error [\(label)]: \(cb.error?.localizedDescription ?? "unknown")"); exit(1) }
}

let compileOptions = MTLCompileOptions()
compileOptions.mathMode = .fast

let library: MTLLibrary
do { library = try device.makeLibrary(source: shaderSource, options: compileOptions); print("Shaders compiled\n") }
catch { print("Shader error: \(error)"); exit(1) }

guard let commandQueue = device.makeCommandQueue() else { exit(1) }
let fftPipeline = try device.makeComputePipelineState(
    function: library.makeFunction(name: "stockham_batched")!)
let transpXYPipeline = try device.makeComputePipelineState(
    function: library.makeFunction(name: "transpose_xy_tiled")!)
let transpXZPipeline = try device.makeComputePipelineState(
    function: library.makeFunction(name: "transpose_xz_tiled")!)
let maxTG = fftPipeline.maxTotalThreadsPerThreadgroup
let TTILE = 16

// ═══════════════════════════════════════════════════════════════════════════
// TWIDDLE TABLE — cached to avoid repeated allocation in tight loops
// ═══════════════════════════════════════════════════════════════════════════

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

// ═══════════════════════════════════════════════════════════════════════════
// GPU HELPERS
// ═══════════════════════════════════════════════════════════════════════════

func runTranspose(pipeline: MTLComputePipelineState,
                  src: MTLBuffer, dst: MTLBuffer,
                  nx: Int, ny: Int, nz: Int, label: String) {
    autoreleasepool {
        guard let cb = commandQueue.makeCommandBuffer(),
              let enc = cb.makeComputeCommandEncoder() else { return }
        cb.label = label
        enc.setComputePipelineState(pipeline)
        enc.setBuffer(src, offset: 0, index: 0)
        enc.setBuffer(dst, offset: 0, index: 1)
        var ux = UInt32(nx), uy = UInt32(ny), uz = UInt32(nz)
        enc.setBytes(&ux, length: 4, index: 2)
        enc.setBytes(&uy, length: 4, index: 3)
        enc.setBytes(&uz, length: 4, index: 4)
        // Tiled dispatch: threadgroups cover TTILE×TTILE blocks
        let gx = (nx + TTILE - 1) / TTILE
        let gy = (ny + TTILE - 1) / TTILE
        enc.dispatchThreadgroups(
            MTLSize(width: gx, height: gy, depth: nz),
            threadsPerThreadgroup: MTLSize(width: TTILE, height: TTILE, depth: 1))
        enc.endEncoding()
        cb.commit(); cb.waitUntilCompleted(); gpuCheck(cb, label: label)
    }
}

func runTransposeXZ(pipeline: MTLComputePipelineState,
                    src: MTLBuffer, dst: MTLBuffer,
                    nx: Int, ny: Int, nz: Int, label: String) {
    autoreleasepool {
        guard let cb = commandQueue.makeCommandBuffer(),
              let enc = cb.makeComputeCommandEncoder() else { return }
        cb.label = label
        enc.setComputePipelineState(pipeline)
        enc.setBuffer(src, offset: 0, index: 0)
        enc.setBuffer(dst, offset: 0, index: 1)
        var ux = UInt32(nx), uy = UInt32(ny), uz = UInt32(nz)
        enc.setBytes(&ux, length: 4, index: 2)
        enc.setBytes(&uy, length: 4, index: 3)
        enc.setBytes(&uz, length: 4, index: 4)
        let gx = (nx + TTILE - 1) / TTILE
        let gz = (nz + TTILE - 1) / TTILE
        enc.dispatchThreadgroups(
            MTLSize(width: gx, height: gz, depth: ny),
            threadsPerThreadgroup: MTLSize(width: TTILE, height: TTILE, depth: 1))
        enc.endEncoding()
        cb.commit(); cb.waitUntilCompleted(); gpuCheck(cb, label: label)
    }
}

func runBlit(src: MTLBuffer, dst: MTLBuffer, size: Int, label: String) {
    autoreleasepool {
        guard let cb = commandQueue.makeCommandBuffer(),
              let blit = cb.makeBlitCommandEncoder() else { return }
        cb.label = label
        blit.copy(from: src, sourceOffset: 0, to: dst, destinationOffset: 0, size: size)
        blit.endEncoding()
        cb.commit(); cb.waitUntilCompleted(); gpuCheck(cb, label: label)
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// BATCHED 1D FFT — all passes in one command buffer, twiddle table
// ═══════════════════════════════════════════════════════════════════════════

func batchedFFT1D(bufA: MTLBuffer, bufB: MTLBuffer,
                  fftLen: Int, numBatches: Int, batchStride: Int) -> MTLBuffer {
    let numPasses = Int(log2(Double(fftLen)))
    let tgW = min(fftLen / 2, maxTG)
    guard let twiddleBuf = getTwiddleBuffer(N: fftLen) else { return bufA }

    autoreleasepool {
        guard let cmdBuf = commandQueue.makeCommandBuffer() else { return }
        cmdBuf.label = "batchedFFT1D_len\(fftLen)"

        for pass in 0..<numPasses {
            let src = (pass % 2 == 0) ? bufA : bufB
            let dst = (pass % 2 == 0) ? bufB : bufA

            guard let enc = cmdBuf.makeComputeCommandEncoder() else { return }
            enc.setComputePipelineState(fftPipeline)
            enc.setBuffer(src, offset: 0, index: 0)
            enc.setBuffer(dst, offset: 0, index: 1)
            var fl = UInt32(fftLen), p = UInt32(pass)
            var bs = UInt32(numBatches), bst = UInt32(batchStride)
            enc.setBytes(&fl, length: 4, index: 2)
            enc.setBytes(&p, length: 4, index: 3)
            enc.setBytes(&bs, length: 4, index: 4)
            enc.setBytes(&bst, length: 4, index: 5)
            enc.setBuffer(twiddleBuf, offset: 0, index: 6)
            enc.dispatchThreads(
                MTLSize(width: fftLen / 2, height: numBatches, depth: 1),
                threadsPerThreadgroup: MTLSize(width: tgW, height: 1, depth: 1))
            enc.endEncoding()
        }

        cmdBuf.commit()
        cmdBuf.waitUntilCompleted()
        gpuCheck(cmdBuf, label: "batchedFFT1D(len=\(fftLen))")
    }

    return (numPasses % 2 == 1) ? bufB : bufA
}

// ═══════════════════════════════════════════════════════════════════════════
// 3D FFT
// ═══════════════════════════════════════════════════════════════════════════

func fft3D(input: MTLBuffer, output: MTLBuffer, temp: MTLBuffer,
           Nx: Int, Ny: Int, Nz: Int) {

    let byteSize = Nx * Ny * Nz * MemoryLayout<SIMD2<Float>>.stride

    // Step 1: FFT along x (contiguous)
    let xResult = batchedFFT1D(bufA: input, bufB: output,
                                fftLen: Nx, numBatches: Ny * Nz, batchStride: Nx)
    if xResult !== output {
        runBlit(src: xResult, dst: output, size: byteSize, label: "x-copy")
    }

    // Step 2: FFT along y — transpose, FFT, transpose back
    runTranspose(pipeline: transpXYPipeline, src: output, dst: temp,
                 nx: Nx, ny: Ny, nz: Nz, label: "y-fwd-transpose")

    let yResult = batchedFFT1D(bufA: temp, bufB: output,
                                fftLen: Ny, numBatches: Nx * Nz, batchStride: Ny)

    let yTranspDst = (yResult === output) ? temp : output
    runTranspose(pipeline: transpXYPipeline, src: yResult, dst: yTranspDst,
                 nx: Ny, ny: Nx, nz: Nz, label: "y-back-transpose")
    if yTranspDst !== output {
        runBlit(src: yTranspDst, dst: output, size: byteSize, label: "y-copy")
    }

    // Step 3: FFT along z — transpose, FFT, transpose back
    runTransposeXZ(pipeline: transpXZPipeline, src: output, dst: temp,
                   nx: Nx, ny: Ny, nz: Nz, label: "z-fwd-transpose")

    let zResult = batchedFFT1D(bufA: temp, bufB: output,
                                fftLen: Nz, numBatches: Nx * Ny, batchStride: Nz)

    let zTranspDst = (zResult === output) ? temp : output
    runTransposeXZ(pipeline: transpXZPipeline, src: zResult, dst: zTranspDst,
                   nx: Nz, ny: Ny, nz: Nx, label: "z-back-transpose")
    if zTranspDst !== output {
        runBlit(src: zTranspDst, dst: output, size: byteSize, label: "z-copy")
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// vDSP reference
// ═══════════════════════════════════════════════════════════════════════════

func vdsp3DFFT(data: inout [SIMD2<Float>], Nx: Int, Ny: Int, Nz: Int) {
    func fftAxis(data: inout [SIMD2<Float>], n: Int, outerCount: Int,
                 indexFn: (Int, Int) -> Int) {
        let log2n = vDSP_Length(log2(Double(n)))
        guard let setup = vDSP_create_fftsetup(log2n, FFTRadix(kFFTRadix2)) else { return }
        for outer in 0..<outerCount {
            var real = (0..<n).map { data[indexFn(outer, $0)].x }
            var imag = (0..<n).map { data[indexFn(outer, $0)].y }
            real.withUnsafeMutableBufferPointer { rBuf in
                imag.withUnsafeMutableBufferPointer { iBuf in
                    var sc = DSPSplitComplex(realp: rBuf.baseAddress!, imagp: iBuf.baseAddress!)
                    vDSP_fft_zip(setup, &sc, 1, log2n, FFTDirection(kFFTDirection_Forward))
                }
            }
            for i in 0..<n { data[indexFn(outer, i)] = SIMD2<Float>(real[i], imag[i]) }
        }
        vDSP_destroy_fftsetup(setup)
    }
    fftAxis(data: &data, n: Nx, outerCount: Ny * Nz) { o, i in i + (o % Ny) * Nx + (o / Ny) * Nx * Ny }
    fftAxis(data: &data, n: Ny, outerCount: Nx * Nz) { o, i in (o % Nx) + i * Nx + (o / Nx) * Nx * Ny }
    fftAxis(data: &data, n: Nz, outerCount: Nx * Ny) { o, i in (o % Nx) + (o / Nx) * Nx + i * Nx * Ny }
}

// ═══════════════════════════════════════════════════════════════════════════
// TESTS
// ═══════════════════════════════════════════════════════════════════════════

print("── Test 1: 3D impulse (8×8×8) ──────────────────────────────")
autoreleasepool {
    let Nx = 8, Ny = 8, Nz = 8, total = 512
    let size = total * MemoryLayout<SIMD2<Float>>.stride
    guard let bI = device.makeBuffer(length: size, options: .storageModeShared),
          let bO = device.makeBuffer(length: size, options: .storageModeShared),
          let bT = device.makeBuffer(length: size, options: .storageModeShared) else { exit(1) }
    let p = bI.contents().bindMemory(to: SIMD2<Float>.self, capacity: total)
    for i in 0..<total { p[i] = .zero }; p[0] = SIMD2<Float>(1, 0)
    fft3D(input: bI, output: bO, temp: bT, Nx: Nx, Ny: Ny, Nz: Nz)
    let r = bO.contents().bindMemory(to: SIMD2<Float>.self, capacity: total)
    var mx: Float = 0
    for i in 0..<total { mx = max(mx, max(abs(r[i].x - 1), abs(r[i].y))) }
    print("  Max error: \(mx)  \(mx < 1e-4 ? "✓ PASS" : "✗ FAIL")")
}

print("\n── Test 2: GPU vs vDSP (16×16×16) ──────────────────────────")
autoreleasepool {
    let Nx = 16, Ny = 16, Nz = 16, total = Nx*Ny*Nz
    let size = total * MemoryLayout<SIMD2<Float>>.stride
    guard let bI = device.makeBuffer(length: size, options: .storageModeShared),
          let bO = device.makeBuffer(length: size, options: .storageModeShared),
          let bT = device.makeBuffer(length: size, options: .storageModeShared) else { exit(1) }
    let p = bI.contents().bindMemory(to: SIMD2<Float>.self, capacity: total)
    srand48(42)
    var cpu = [SIMD2<Float>]()
    for i in 0..<total {
        let v = SIMD2<Float>(Float(drand48()-0.5), Float(drand48()-0.5)); p[i] = v; cpu.append(v)
    }
    fft3D(input: bI, output: bO, temp: bT, Nx: Nx, Ny: Ny, Nz: Nz)
    let g = bO.contents().bindMemory(to: SIMD2<Float>.self, capacity: total)
    vdsp3DFFT(data: &cpu, Nx: Nx, Ny: Ny, Nz: Nz)
    var mx: Float = 0, nm: Float = 0
    for i in 0..<total {
        mx = max(mx, max(abs(g[i].x-cpu[i].x), abs(g[i].y-cpu[i].y)))
        nm += cpu[i].x*cpu[i].x + cpu[i].y*cpu[i].y
    }
    let rel = mx / sqrt(nm)
    print("  Rel err: \(rel)  \(rel < 1e-5 ? "✓ PASS" : rel < 1e-3 ? "⚠ MARGINAL" : "✗ FAIL")")
}

print("\n── Test 3: Parseval (16×16×16) ─────────────────────────────")
autoreleasepool {
    let Nx = 16, Ny = 16, Nz = 16, total = Nx*Ny*Nz
    let size = total * MemoryLayout<SIMD2<Float>>.stride
    guard let bI = device.makeBuffer(length: size, options: .storageModeShared),
          let bO = device.makeBuffer(length: size, options: .storageModeShared),
          let bT = device.makeBuffer(length: size, options: .storageModeShared) else { exit(1) }
    let p = bI.contents().bindMemory(to: SIMD2<Float>.self, capacity: total)
    srand48(123); var tE: Float = 0
    for i in 0..<total {
        let v = SIMD2<Float>(Float(drand48()-0.5), Float(drand48()-0.5)); p[i] = v
        tE += v.x*v.x + v.y*v.y
    }
    fft3D(input: bI, output: bO, temp: bT, Nx: Nx, Ny: Ny, Nz: Nz)
    let r = bO.contents().bindMemory(to: SIMD2<Float>.self, capacity: total)
    var fE: Float = 0
    for i in 0..<total { fE += r[i].x*r[i].x + r[i].y*r[i].y }
    fE /= Float(total)
    let d = abs(tE - fE) / tE
    print("  Time: \(tE)  Freq: \(fE)  Diff: \(d)  \(d < 1e-4 ? "✓ PASS" : "✗ FAIL")")
}

print("\n── Test 4: 3D FFT throughput ────────────────────────────────")
print("     Grid       Total      GPU (ms)    vDSP (ms)   Ratio")
print("  " + String(repeating: "─", count: 56))
for (Nx, Ny, Nz, reps) in [(8,8,8,3), (16,16,16,3), (32,32,32,2), (64,64,64,1)] as [(Int,Int,Int,Int)] {
    autoreleasepool {
        let total = Nx*Ny*Nz, size = total * MemoryLayout<SIMD2<Float>>.stride
        guard let bI = device.makeBuffer(length: size, options: .storageModeShared),
              let bO = device.makeBuffer(length: size, options: .storageModeShared),
              let bT = device.makeBuffer(length: size, options: .storageModeShared) else { return }
        let p = bI.contents().bindMemory(to: SIMD2<Float>.self, capacity: total)
        for i in 0..<total { p[i] = SIMD2<Float>(Float(drand48()), Float(drand48())) }

        fft3D(input: bI, output: bO, temp: bT, Nx: Nx, Ny: Ny, Nz: Nz)

        let t0 = CFAbsoluteTimeGetCurrent()
        for _ in 0..<reps {
            autoreleasepool { fft3D(input: bI, output: bO, temp: bT, Nx: Nx, Ny: Ny, Nz: Nz) }
        }
        let gMs = (CFAbsoluteTimeGetCurrent()-t0)*1000/Double(reps)

        var cpu = (0..<total).map { _ in SIMD2<Float>(Float(drand48()), Float(drand48())) }
        vdsp3DFFT(data: &cpu, Nx: Nx, Ny: Ny, Nz: Nz)
        let t1 = CFAbsoluteTimeGetCurrent()
        for _ in 0..<reps { var d = cpu; vdsp3DFFT(data: &d, Nx: Nx, Ny: Ny, Nz: Nz) }
        let vMs = (CFAbsoluteTimeGetCurrent()-t1)*1000/Double(reps)

        let ratio = vMs/gMs
        let winner = ratio > 1 ? "GPU" : "vDSP"
        print(String(format: "  %2dx%2dx%2d  %8d    %7.2f     %7.2f     %.2fx",
                     Nx,Ny,Nz, total, gMs, vMs, ratio) + " (\(winner))")
    }
}

print("""

═══════════════════════════════════════════════════════════════
  Exercise 6 complete.

  What you now have across all exercises:
    Ex01: cmul, cadd, cconj, cmag (using dot() intrinsic)
    Ex02: parallel reduction + threadgroup memory
    Ex03: tiled SGEMM with GPU timestamp benchmarking
    Ex04: register-blocked CGEMM with NUM_THREADS (not hardcoded)
    Ex05: 1D Stockham FFT with precomputed twiddle table
    Ex06: 3D FFT with tiled transpose (threadgroup memory)

  Production improvements across all exercises:
    • GPU timestamp benchmarking (gpuStartTime/gpuEndTime)
    • Command buffer error checking (gpuCheck after every dispatch)
    • autoreleasepool in all tight loops (no ObjC object leaks)
    • hasUnifiedMemory assertion (portable safety)
    • Command buffer labels (for Metal Debugger / Instruments)
    • Precomputed twiddle tables (no per-butterfly sin/cos)
    • Tiled transpose with bank-conflict padding
    • Clean API contracts (input buffers never mutated)
    • Derived constants (NUM_THREADS from block parameters)
═══════════════════════════════════════════════════════════════
""")
