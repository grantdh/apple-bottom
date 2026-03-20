#!/usr/bin/env swift
// Exercise 6: Batched 3D FFT from 1D Stockham Transforms
// swiftc -O -Xcc -DACCELERATE_NEW_LAPACK -framework Metal -framework Foundation -framework Accelerate ex06_batched_3d_fft.swift -o ex06
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

kernel void stockham_batched(
    device const float2 *input [[buffer(0)]], device float2 *output [[buffer(1)]],
    constant uint &fft_len [[buffer(2)]], constant uint &pass [[buffer(3)]],
    constant uint &batch_size [[buffer(4)]], constant uint &batch_stride [[buffer(5)]],
    device const float2 *twiddle [[buffer(6)]], uint2 gid [[thread_position_in_grid]]
) {
    uint j = gid.x, batch = gid.y, halfN = fft_len / 2;
    if (j >= halfN || batch >= batch_size) return;
    uint stride = 1u << pass, group = j / stride, k = j % stride;
    uint base = batch * batch_stride;
    float2 a = input[base + j], Wb = cmul(twiddle[k * (halfN / stride)], input[base + j + halfN]);
    uint out = group * stride * 2 + k;
    output[base + out]          = a + Wb;
    output[base + out + stride] = a - Wb;
}

// Tiled transpose with threadgroup memory + bank-conflict padding
#define TT 16

kernel void transpose_xy(
    device const float2 *input [[buffer(0)]], device float2 *output [[buffer(1)]],
    constant uint &Nx [[buffer(2)]], constant uint &Ny [[buffer(3)]], constant uint &Nz [[buffer(4)]],
    uint3 lid [[thread_position_in_threadgroup]], uint3 tgid [[threadgroup_position_in_grid]]
) {
    threadgroup float2 tile[TT][TT + 1];
    uint z = tgid.z, gx = tgid.x * TT + lid.x, gy = tgid.y * TT + lid.y;
    tile[lid.y][lid.x] = (gx < Nx && gy < Ny && z < Nz) ? input[gx + gy*Nx + z*Nx*Ny] : float2(0.0);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    uint tx = tgid.y * TT + lid.x, ty = tgid.x * TT + lid.y;
    if (tx < Ny && ty < Nx && z < Nz) output[tx + ty*Ny + z*Ny*Nx] = tile[lid.x][lid.y];
}

kernel void transpose_xz(
    device const float2 *input [[buffer(0)]], device float2 *output [[buffer(1)]],
    constant uint &Nx [[buffer(2)]], constant uint &Ny [[buffer(3)]], constant uint &Nz [[buffer(4)]],
    uint3 lid [[thread_position_in_threadgroup]], uint3 tgid [[threadgroup_position_in_grid]]
) {
    threadgroup float2 tile[TT][TT + 1];
    uint y = tgid.z, gx = tgid.x * TT + lid.x, gz = tgid.y * TT + lid.y;
    tile[lid.y][lid.x] = (gx < Nx && y < Ny && gz < Nz) ? input[gx + y*Nx + gz*Nx*Ny] : float2(0.0);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    uint tz = tgid.x * TT + lid.x, tx = tgid.y * TT + lid.y;
    if (tz < Nz && y < Ny && tx < Nx) output[tz + y*Nz + tx*Nz*Ny] = tile[lid.x][lid.y];
}
"""

// MARK: - Setup

guard let device = MTLCreateSystemDefaultDevice() else { fatalError("No Metal device") }
assert(device.hasUnifiedMemory, "Requires Apple Silicon")

let opts = MTLCompileOptions(); opts.mathMode = .fast
let library = try device.makeLibrary(source: shaderSource, options: opts)
guard let queue = device.makeCommandQueue() else { fatalError("No command queue") }

let fftPip    = try device.makeComputePipelineState(function: library.makeFunction(name: "stockham_batched")!)
let transpXY  = try device.makeComputePipelineState(function: library.makeFunction(name: "transpose_xy")!)
let transpXZ  = try device.makeComputePipelineState(function: library.makeFunction(name: "transpose_xz")!)
let maxTG     = fftPip.maxTotalThreadsPerThreadgroup
let TT = 16

// MARK: - Resource caches

var twiddleCache: [Int: MTLBuffer] = [:]

func twiddle(_ N: Int) -> MTLBuffer {
    if let c = twiddleCache[N] { return c }
    let h = N / 2, buf = device.makeBuffer(length: h * 8, options: .storageModeShared)!
    let p = buf.contents().bindMemory(to: SIMD2<Float>.self, capacity: h)
    for j in 0..<h { let a = -2.0 * Float.pi * Float(j) / Float(N); p[j] = SIMD2(cos(a), sin(a)) }
    twiddleCache[N] = buf; return buf
}

// MARK: - GPU dispatch helpers

func gpuCheck(_ cb: MTLCommandBuffer) {
    if cb.status == .error { fatalError("GPU [\(cb.label ?? "?")]: \(cb.error?.localizedDescription ?? "?")") }
}

func runTranspose(_ pip: MTLComputePipelineState, src: MTLBuffer, dst: MTLBuffer,
                  nx: Int, ny: Int, nz: Int, xzMode: Bool = false) {
    autoreleasepool {
        guard let cb = queue.makeCommandBuffer(), let enc = cb.makeComputeCommandEncoder() else { return }
        enc.setComputePipelineState(pip)
        enc.setBuffer(src, offset: 0, index: 0); enc.setBuffer(dst, offset: 0, index: 1)
        var ux = UInt32(nx), uy = UInt32(ny), uz = UInt32(nz)
        enc.setBytes(&ux, length: 4, index: 2); enc.setBytes(&uy, length: 4, index: 3); enc.setBytes(&uz, length: 4, index: 4)
        if xzMode {
            enc.dispatchThreadgroups(MTLSize(width: (nx+TT-1)/TT, height: (nz+TT-1)/TT, depth: ny),
                                      threadsPerThreadgroup: MTLSize(width: TT, height: TT, depth: 1))
        } else {
            enc.dispatchThreadgroups(MTLSize(width: (nx+TT-1)/TT, height: (ny+TT-1)/TT, depth: nz),
                                      threadsPerThreadgroup: MTLSize(width: TT, height: TT, depth: 1))
        }
        enc.endEncoding(); cb.commit(); cb.waitUntilCompleted(); gpuCheck(cb)
    }
}

func blit(_ src: MTLBuffer, _ dst: MTLBuffer, _ bytes: Int) {
    autoreleasepool {
        guard let cb = queue.makeCommandBuffer(), let b = cb.makeBlitCommandEncoder() else { return }
        b.copy(from: src, sourceOffset: 0, to: dst, destinationOffset: 0, size: bytes)
        b.endEncoding(); cb.commit(); cb.waitUntilCompleted(); gpuCheck(cb)
    }
}

// MARK: - Batched 1D FFT — all passes in one command buffer

func batchedFFT1D(bufA: MTLBuffer, bufB: MTLBuffer, fftLen: Int, batches: Int, stride: Int) -> MTLBuffer {
    let passes = Int(log2(Double(fftLen)))
    let tg = min(fftLen / 2, maxTG)
    let tw = twiddle(fftLen)

    autoreleasepool {
        guard let cb = queue.makeCommandBuffer() else { return }
        for pass in 0..<passes {
            let src = (pass % 2 == 0) ? bufA : bufB
            let dst = (pass % 2 == 0) ? bufB : bufA
            guard let enc = cb.makeComputeCommandEncoder() else { return }
            enc.setComputePipelineState(fftPip)
            enc.setBuffer(src, offset: 0, index: 0); enc.setBuffer(dst, offset: 0, index: 1)
            var fl = UInt32(fftLen), p = UInt32(pass), bs = UInt32(batches), bst = UInt32(stride)
            enc.setBytes(&fl, length: 4, index: 2); enc.setBytes(&p, length: 4, index: 3)
            enc.setBytes(&bs, length: 4, index: 4); enc.setBytes(&bst, length: 4, index: 5)
            enc.setBuffer(tw, offset: 0, index: 6)
            enc.dispatchThreads(MTLSize(width: fftLen/2, height: batches, depth: 1),
                                threadsPerThreadgroup: MTLSize(width: tg, height: 1, depth: 1))
            enc.endEncoding()
        }
        cb.commit(); cb.waitUntilCompleted(); gpuCheck(cb)
    }
    return (passes % 2 == 1) ? bufB : bufA
}

// MARK: - 3D FFT: FFT each axis via transpose-FFT-transpose

func fft3D(input: MTLBuffer, output: MTLBuffer, temp: MTLBuffer, Nx: Int, Ny: Int, Nz: Int) {
    let bytes = Nx * Ny * Nz * 8

    // X-axis (contiguous)
    let xR = batchedFFT1D(bufA: input, bufB: output, fftLen: Nx, batches: Ny*Nz, stride: Nx)
    if xR !== output { blit(xR, output, bytes) }

    // Y-axis: transpose XY → FFT → transpose back
    runTranspose(transpXY, src: output, dst: temp, nx: Nx, ny: Ny, nz: Nz)
    let yR = batchedFFT1D(bufA: temp, bufB: output, fftLen: Ny, batches: Nx*Nz, stride: Ny)
    let yDst = (yR === output) ? temp : output
    runTranspose(transpXY, src: yR, dst: yDst, nx: Ny, ny: Nx, nz: Nz)
    if yDst !== output { blit(yDst, output, bytes) }

    // Z-axis: transpose XZ → FFT → transpose back
    runTranspose(transpXZ, src: output, dst: temp, nx: Nx, ny: Ny, nz: Nz, xzMode: true)
    let zR = batchedFFT1D(bufA: temp, bufB: output, fftLen: Nz, batches: Nx*Ny, stride: Nz)
    let zDst = (zR === output) ? temp : output
    runTranspose(transpXZ, src: zR, dst: zDst, nx: Nz, ny: Ny, nz: Nx, xzMode: true)
    if zDst !== output { blit(zDst, output, bytes) }
}

// MARK: - vDSP reference

func vdsp3DFFT(data: inout [SIMD2<Float>], Nx: Int, Ny: Int, Nz: Int) {
    func fftAxis(_ data: inout [SIMD2<Float>], _ n: Int, _ outerCount: Int, _ idx: (Int, Int) -> Int) {
        let log2n = vDSP_Length(log2(Double(n)))
        guard let setup = vDSP_create_fftsetup(log2n, FFTRadix(kFFTRadix2)) else { return }
        for o in 0..<outerCount {
            var real = (0..<n).map { data[idx(o, $0)].x }
            var imag = (0..<n).map { data[idx(o, $0)].y }
            real.withUnsafeMutableBufferPointer { r in
                imag.withUnsafeMutableBufferPointer { im in
                    var sc = DSPSplitComplex(realp: r.baseAddress!, imagp: im.baseAddress!)
                    vDSP_fft_zip(setup, &sc, 1, log2n, FFTDirection(kFFTDirection_Forward))
                }
            }
            for i in 0..<n { data[idx(o, i)] = SIMD2(real[i], imag[i]) }
        }
        vDSP_destroy_fftsetup(setup)
    }
    fftAxis(&data, Nx, Ny*Nz) { o, i in i + (o%Ny)*Nx + (o/Ny)*Nx*Ny }
    fftAxis(&data, Ny, Nx*Nz) { o, i in (o%Nx) + i*Nx + (o/Nx)*Nx*Ny }
    fftAxis(&data, Nz, Nx*Ny) { o, i in (o%Nx) + (o/Nx)*Nx + i*Nx*Ny }
}

print("Exercise 6: Batched 3D FFT — GPU: \(device.name)\n")

// MARK: - Test 1: 3D impulse (8×8×8)

autoreleasepool {
    let (Nx, Ny, Nz, total) = (8, 8, 8, 512)
    let sz = total * 8
    let I = device.makeBuffer(length: sz, options: .storageModeShared)!
    let O = device.makeBuffer(length: sz, options: .storageModeShared)!
    let T = device.makeBuffer(length: sz, options: .storageModeShared)!
    let p = I.contents().bindMemory(to: SIMD2<Float>.self, capacity: total)
    for i in 0..<total { p[i] = .zero }; p[0] = SIMD2(1, 0)
    fft3D(input: I, output: O, temp: T, Nx: Nx, Ny: Ny, Nz: Nz)
    let r = O.contents().bindMemory(to: SIMD2<Float>.self, capacity: total)
    var mx: Float = 0
    for i in 0..<total { mx = max(mx, max(abs(r[i].x - 1), abs(r[i].y))) }
    print("  impulse 8³:      err = \(mx)  \(mx < 1e-4 ? "✓" : "✗")")
}

// MARK: - Test 2: GPU vs vDSP (16×16×16)

autoreleasepool {
    let (Nx, Ny, Nz) = (16, 16, 16); let total = Nx*Ny*Nz
    let sz = total * 8
    let I = device.makeBuffer(length: sz, options: .storageModeShared)!
    let O = device.makeBuffer(length: sz, options: .storageModeShared)!
    let T = device.makeBuffer(length: sz, options: .storageModeShared)!
    let p = I.contents().bindMemory(to: SIMD2<Float>.self, capacity: total)
    srand48(42)
    var cpu = [SIMD2<Float>]()
    for i in 0..<total { let v = SIMD2<Float>(Float(drand48()-0.5), Float(drand48()-0.5)); p[i] = v; cpu.append(v) }

    fft3D(input: I, output: O, temp: T, Nx: Nx, Ny: Ny, Nz: Nz)
    let g = O.contents().bindMemory(to: SIMD2<Float>.self, capacity: total)
    vdsp3DFFT(data: &cpu, Nx: Nx, Ny: Ny, Nz: Nz)
    var mx: Float = 0, nm: Float = 0
    for i in 0..<total { mx = max(mx, max(abs(g[i].x-cpu[i].x), abs(g[i].y-cpu[i].y))); nm += cpu[i].x*cpu[i].x + cpu[i].y*cpu[i].y }
    print("  vs vDSP 16³:     rel = \(mx/sqrt(nm))  \(mx/sqrt(nm) < 1e-5 ? "✓" : "✗")")
}

// MARK: - Test 3: Parseval (16×16×16)

autoreleasepool {
    let (Nx, Ny, Nz) = (16, 16, 16); let total = Nx*Ny*Nz
    let sz = total * 8
    let I = device.makeBuffer(length: sz, options: .storageModeShared)!
    let O = device.makeBuffer(length: sz, options: .storageModeShared)!
    let T = device.makeBuffer(length: sz, options: .storageModeShared)!
    let p = I.contents().bindMemory(to: SIMD2<Float>.self, capacity: total)
    srand48(123); var tE: Float = 0
    for i in 0..<total { let v = SIMD2<Float>(Float(drand48()-0.5), Float(drand48()-0.5)); p[i] = v; tE += v.x*v.x + v.y*v.y }

    fft3D(input: I, output: O, temp: T, Nx: Nx, Ny: Ny, Nz: Nz)
    let r = O.contents().bindMemory(to: SIMD2<Float>.self, capacity: total)
    var fE: Float = 0
    for i in 0..<total { fE += r[i].x*r[i].x + r[i].y*r[i].y }
    fE /= Float(total)
    let d = abs(tE - fE) / tE
    print("  Parseval 16³:    diff = \(d)  \(d < 1e-4 ? "✓" : "✗")")
}

// MARK: - Test 4: Throughput

do {
    print("\n  Throughput:")
    print("     Grid      Total     GPU ms    vDSP ms    ratio")

    for (Nx, Ny, Nz, reps) in [(8,8,8,3), (16,16,16,3), (32,32,32,2), (64,64,64,1)] as [(Int,Int,Int,Int)] {
        autoreleasepool {
            let total = Nx*Ny*Nz, sz = total * 8
            let I = device.makeBuffer(length: sz, options: .storageModeShared)!
            let O = device.makeBuffer(length: sz, options: .storageModeShared)!
            let T = device.makeBuffer(length: sz, options: .storageModeShared)!
            let p = I.contents().bindMemory(to: SIMD2<Float>.self, capacity: total)
            for i in 0..<total { p[i] = SIMD2(Float(drand48()), Float(drand48())) }

            fft3D(input: I, output: O, temp: T, Nx: Nx, Ny: Ny, Nz: Nz) // warmup

            let t0 = CFAbsoluteTimeGetCurrent()
            for _ in 0..<reps { autoreleasepool { fft3D(input: I, output: O, temp: T, Nx: Nx, Ny: Ny, Nz: Nz) } }
            let gMs = (CFAbsoluteTimeGetCurrent()-t0)*1000/Double(reps)

            var cpu = (0..<total).map { _ in SIMD2<Float>(Float(drand48()), Float(drand48())) }
            vdsp3DFFT(data: &cpu, Nx: Nx, Ny: Ny, Nz: Nz)
            let t1 = CFAbsoluteTimeGetCurrent()
            for _ in 0..<reps { var d = cpu; vdsp3DFFT(data: &d, Nx: Nx, Ny: Ny, Nz: Nz) }
            let vMs = (CFAbsoluteTimeGetCurrent()-t1)*1000/Double(reps)

            let ratio = vMs/gMs
            let w = ratio > 1 ? "GPU" : "vDSP"
            print(String(format: "  %2dx%2dx%2d  %8d   %7.2f   %7.2f    %5.1fx", Nx,Ny,Nz, total, gMs, vMs, ratio) + " \(w)")
        }
    }
}
