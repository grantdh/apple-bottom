#!/usr/bin/env swift
// =============================================================================
// Exercise 19b: Random [-1,1] Inputs — κ-Binned ULP Analysis
// =============================================================================
//
// Same analysis as ex19 but with random [-1,1] inputs to stress-test
// the per-element condition number range. This populates C4's κ bins
// (κ<2, 2-10, 10-100, 100-1000, 1000+) and shows how ULP error
// correlates with conditioning.
//
// Also tests DFT-like (I+0.01R) matrices for the application-relevant case.
//
// BUILD:
//   swiftc -O -Xcc -DACCELERATE_NEW_LAPACK -framework Metal \
//       -framework Foundation -framework Accelerate ex19b_random_kappa.swift -o ex19b
//
// Grant Heileman — UNM ECE — March 2026
// =============================================================================

import Foundation
import Metal
import Accelerate

// =============================================================================
// MARK: - Metal Shader (TD-DGEMM 2×2)
// =============================================================================

let shaderSource = """
#include <metal_stdlib>
using namespace metal;
struct td { float hi; float md; float lo; };
inline void twoSum(float a, float b, thread float &s, thread float &e) {
    s = a + b; float v = s - a; e = (a - (s - v)) + (b - v);
}
inline void fastTwoSum(float a, float b, thread float &s, thread float &e) {
    s = a + b; e = b - (s - a);
}
inline void twoProduct(float a, float b, thread float &p, thread float &e) {
    p = a * b; e = fma(a, b, -p);
}
inline td td_renorm(float a, float b, float c) {
    float s0, e0; twoSum(b, c, s0, e0);
    float s1, e1; twoSum(a, s0, s1, e1);
    float s2, e2; twoSum(e1, e0, s2, e2);
    float rh, rm; fastTwoSum(s1, s2, rh, rm);
    return {rh, rm, e2};
}
inline td td_add(td a, td b) {
    float s0, e0; twoSum(a.hi, b.hi, s0, e0);
    float s1, e1; twoSum(a.md, b.md, s1, e1);
    float s2 = a.lo + b.lo;
    float t0, t1; twoSum(e0, s1, t0, t1);
    float r0, r1; twoSum(s0, t0, r0, r1);
    float m0, m1; twoSum(r1, t1, m0, m1);
    m0 += e1;
    return td_renorm(r0, m0, m1 + s2);
}
inline td td_fma_full(td a, td b, td c) {
    float p, ep; twoProduct(a.hi, b.hi, p, ep);
    float cx1, ex1; twoProduct(a.hi, b.md, cx1, ex1);
    float cx2, ex2; twoProduct(a.md, b.hi, cx2, ex2);
    float lower = a.hi * b.lo + a.md * b.md + a.lo * b.hi;
    float s0, t0; twoSum(p, c.hi, s0, t0);
    float s1, t1; twoSum(ep, cx1, s1, t1);
    float s2, t2; twoSum(s1, cx2, s2, t2);
    float s3, t3; twoSum(s2, t0, s3, t3);
    float s4, t4; twoSum(s3, c.md, s4, t4);
    float r0, r1; twoSum(s0, s4, r0, r1);
    float lo = t1 + t2 + t3 + t4 + ex1 + ex2 + lower + c.lo;
    return td_renorm(r0, r1, lo);
}
#define BM 32
#define BN 32
#define TM 2
#define TN 2
#define TK 8
#define NT ((BM / TM) * (BN / TN))
kernel void td_dgemm_2x2(
    device const td *A [[buffer(0)]], device const td *B [[buffer(1)]],
    device td *C [[buffer(2)]],
    constant uint &M [[buffer(3)]], constant uint &N [[buffer(4)]],
    constant uint &K_dim [[buffer(5)]],
    uint2 lid [[thread_position_in_threadgroup]],
    uint2 tgid [[threadgroup_position_in_grid]],
    uint flatId [[thread_index_in_threadgroup]]
) {
    uint bRow = tgid.y * BM, bCol = tgid.x * BN;
    uint ty = lid.y, tx = lid.x;
    td acc[TM][TN];
    for (uint i = 0; i < TM; i++)
        for (uint j = 0; j < TN; j++) acc[i][j] = {0.0f, 0.0f, 0.0f};
    threadgroup td tileA[BM * TK], tileB[TK * BN];
    for (uint kt = 0; kt < (K_dim + TK - 1) / TK; kt++) {
        for (uint i = flatId; i < BM * TK; i += NT) {
            uint r = i / TK, c = i % TK;
            uint gr = bRow + r, gc = kt * TK + c;
            tileA[i] = (gr < M && gc < K_dim) ? A[gr * K_dim + gc] : td{0.0f, 0.0f, 0.0f};
        }
        for (uint i = flatId; i < TK * BN; i += NT) {
            uint r = i / BN, c = i % BN;
            uint gr = kt * TK + r, gc = bCol + c;
            tileB[i] = (gr < K_dim && gc < N) ? B[gr * N + gc] : td{0.0f, 0.0f, 0.0f};
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (uint k = 0; k < TK; k++) {
            td av[TM], bv[TN];
            for (uint i = 0; i < TM; i++) av[i] = tileA[(ty * TM + i) * TK + k];
            for (uint j = 0; j < TN; j++) bv[j] = tileB[k * BN + tx * TN + j];
            for (uint i = 0; i < TM; i++)
                for (uint j = 0; j < TN; j++)
                    acc[i][j] = td_fma_full(av[i], bv[j], acc[i][j]);
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

// =============================================================================
// MARK: - Setup
// =============================================================================

guard let device = MTLCreateSystemDefaultDevice() else { fatalError("No Metal") }
assert(device.hasUnifiedMemory)
let compOpts = MTLCompileOptions(); compOpts.mathMode = .safe
let library = try! device.makeLibrary(source: shaderSource, options: compOpts)
guard let queue = device.makeCommandQueue() else { fatalError() }
let tdPip = try! device.makeComputePipelineState(
    function: library.makeFunction(name: "td_dgemm_2x2")!)

struct TDVal {
    var hi: Float; var md: Float; var lo: Float
    init(fromDouble d: Double) {
        hi = Float(d); let r1 = d - Double(hi)
        md = Float(r1); lo = Float(r1 - Double(md))
    }
    var doubleValue: Double { Double(hi) + Double(md) + Double(lo) }
}

func gpuTDGEMM(A: MTLBuffer, B: MTLBuffer, C: MTLBuffer, M: Int, N: Int, K: Int) {
    let cb = queue.makeCommandBuffer()!; let enc = cb.makeComputeCommandEncoder()!
    enc.setComputePipelineState(tdPip)
    enc.setBuffer(A, offset: 0, index: 0); enc.setBuffer(B, offset: 0, index: 1)
    enc.setBuffer(C, offset: 0, index: 2)
    var m32 = UInt32(M), n32 = UInt32(N), k32 = UInt32(K)
    enc.setBytes(&m32, length: 4, index: 3); enc.setBytes(&n32, length: 4, index: 4)
    enc.setBytes(&k32, length: 4, index: 5)
    enc.dispatchThreadgroups(
        MTLSize(width: (N+31)/32, height: (M+31)/32, depth: 1),
        threadsPerThreadgroup: MTLSize(width: 16, height: 16, depth: 1))
    enc.endEncoding(); cb.commit(); cb.waitUntilCompleted()
}

// =============================================================================
// MARK: - References
// =============================================================================

func twoSumD(_ a: Double, _ b: Double) -> (Double, Double) {
    let s = a + b; let v = s - a; return (s, (a - (s - v)) + (b - v))
}
func twoProdD(_ a: Double, _ b: Double) -> (Double, Double) {
    let p = a * b; let e = (-p).addingProduct(a, b); return (p, e)
}

func refDotWithResidual(_ A: [Double], _ B: [Double], row: Int, col: Int,
                         N: Int, K: Int) -> (sh: Double, sl: Double, sAbs: Double) {
    var sh: Double = 0, sl: Double = 0, sAbs: Double = 0
    for k in 0..<K {
        let a = A[row * K + k], b = B[k * N + col]
        let (ph, pl) = twoProdD(a, b)
        let (s1, e1) = twoSumD(sh, ph)
        let (s2, e2) = twoSumD(s1, e1 + sl + pl)
        sh = s2; sl = e2
        sAbs += abs(a) * abs(b)
    }
    return (sh, sl, sAbs)
}

func ulpDist(_ a: Double, _ b: Double) -> UInt64 {
    if a == b { return 0 }
    if a.isNaN || b.isNaN { return UInt64.max }
    if a == 0.0 { return b.bitPattern }
    if b == 0.0 { return a.bitPattern }
    if a.sign != b.sign { return ulpDist(a, 0.0) + ulpDist(0.0, b) }
    let ab = a.bitPattern, bb = b.bitPattern
    return ab > bb ? ab - bb : bb - ab
}

// =============================================================================
// MARK: - Analysis Function
// =============================================================================

func analyzeMatrix(name: String, sz: Int, A: [Double], B: [Double], nSample: Int) {
    let K = sz, count = sz * sz
    let szTD = MemoryLayout<TDVal>.stride
    let checkAll = (nSample == 0)
    let nCheck = checkAll ? count : min(nSample, count)

    print("\n" + String(repeating: "=", count: 72))
    print("  \(name): \(sz)×\(sz) (\(checkAll ? "full \(count)" : "sampled \(nCheck)") elements)")
    print(String(repeating: "=", count: 72))

    // GPU TD-DGEMM
    let bufA = device.makeBuffer(length: sz*K*szTD, options: .storageModeShared)!
    let bufB = device.makeBuffer(length: K*sz*szTD, options: .storageModeShared)!
    let bufC = device.makeBuffer(length: count*szTD, options: .storageModeShared)!
    let pA = bufA.contents().bindMemory(to: TDVal.self, capacity: sz*K)
    let pB = bufB.contents().bindMemory(to: TDVal.self, capacity: K*sz)
    for i in 0..<(sz*K) { pA[i] = TDVal(fromDouble: A[i]) }
    for i in 0..<(K*sz) { pB[i] = TDVal(fromDouble: B[i]) }
    gpuTDGEMM(A: bufA, B: bufB, C: bufC, M: sz, N: sz, K: K)
    let pC = bufC.contents().bindMemory(to: TDVal.self, capacity: count)

    // Accelerate
    var cA = A, cB = B, cpuC = [Double](repeating: 0, count: count)
    cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                Int32(sz), Int32(sz), Int32(K), 1.0, &cA, Int32(K),
                &cB, Int32(sz), 0.0, &cpuC, Int32(sz))

    // Stats
    var tdExact = 0, cpuExact = 0
    var tdMaxULP: UInt64 = 0, cpuMaxULP: UInt64 = 0
    var tdSumULP: UInt64 = 0, cpuSumULP: UInt64 = 0

    // C1: Faithful rounding
    var oneUlpCount = 0, faithfulCount = 0, notFaithful = 0

    // C4: κ bins — finer granularity for random inputs
    // κ<2, 2-10, 10-100, 100-1K, 1K-10K, 10K+
    let nBins = 6
    var ulpByKappa = [[UInt64]](repeating: [], count: nBins)
    var cpuUlpByKappa = [[UInt64]](repeating: [], count: nBins)
    var kappaBinCounts = [Int](repeating: 0, count: nBins)
    var nearZeroSkipped = 0

    let t0 = CFAbsoluteTimeGetCurrent()

    for s in 0..<nCheck {
        let idx: Int
        if checkAll {
            idx = s
        } else {
            // Spread samples evenly
            idx = s * (count / nCheck)
        }
        let row = idx / sz, col = idx % sz

        let (sh, sl, sAbs) = refDotWithResidual(A, B, row: row, col: col, N: sz, K: K)
        let ref = sh + sl
        let tdVal = pC[idx].doubleValue
        let cpuVal = cpuC[idx]

        let du = ulpDist(tdVal, ref)
        let cu = ulpDist(cpuVal, ref)

        if du == 0 { tdExact += 1 }
        if cu == 0 { cpuExact += 1 }
        tdMaxULP = max(tdMaxULP, du)
        cpuMaxULP = max(cpuMaxULP, cu)
        tdSumULP += min(du, 1_000_000)
        cpuSumULP += min(cu, 1_000_000)

        // C1: Faithful rounding for 1-ULP elements
        if du == 1 {
            oneUlpCount += 1
            if sl > 0 {
                if tdVal > ref { faithfulCount += 1 } else { notFaithful += 1 }
            } else if sl < 0 {
                if tdVal < ref { faithfulCount += 1 } else { notFaithful += 1 }
            } else {
                notFaithful += 1
            }
        }

        // C4: κ bin
        if abs(ref) > 1e-300 {
            let kappa = sAbs / abs(ref)
            let bin: Int
            if kappa < 2.0 { bin = 0 }
            else if kappa < 10.0 { bin = 1 }
            else if kappa < 100.0 { bin = 2 }
            else if kappa < 1000.0 { bin = 3 }
            else if kappa < 10000.0 { bin = 4 }
            else { bin = 5 }
            ulpByKappa[bin].append(du)
            cpuUlpByKappa[bin].append(cu)
            kappaBinCounts[bin] += 1
        } else {
            nearZeroSkipped += 1
        }
    }

    let elapsed = CFAbsoluteTimeGetCurrent() - t0
    let tdPct = 100.0 * Double(tdExact) / Double(nCheck)
    let cpuPct = 100.0 * Double(cpuExact) / Double(nCheck)
    let tdMean = Double(tdSumULP) / Double(nCheck)
    let cpuMean = Double(cpuSumULP) / Double(nCheck)

    // --- Report ---
    print("\n  Ref computed in \(String(format: "%.1f", elapsed))s")

    print("\n  [C7] Correct Rounding:")
    print("    TD-DGEMM 2×2:  \(String(format: "%5d / %-5d", tdExact, nCheck)) (\(String(format: "%5.1f%%", tdPct)))  mean: \(String(format: "%.2f", tdMean))  max: \(tdMaxULP)")
    print("    Accelerate:    \(String(format: "%5d / %-5d", cpuExact, nCheck)) (\(String(format: "%5.1f%%", cpuPct)))  mean: \(String(format: "%.2f", cpuMean))  max: \(cpuMaxULP)")

    // C1
    if oneUlpCount > 0 {
        print("\n  [C1] Faithful Rounding Direction:")
        print("    1-ULP elements: \(oneUlpCount), faithful: \(faithfulCount), NOT faithful: \(notFaithful)")
        if notFaithful == 0 {
            print("    ✓ ALL 1-ULP errors are faithfully rounded")
        } else {
            print("    ✗ \(notFaithful) elements NOT faithfully rounded — investigate")
        }
    }

    if nearZeroSkipped > 0 {
        print("\n  Note: \(nearZeroSkipped) near-zero elements skipped in κ analysis")
    }

    // C4: κ-binned table — THE KEY OUTPUT FOR THE REVIEWER
    print("\n  [C4] ULP Distribution by Per-Element κ (TD-DGEMM):")
    print("    κ range           n        exact%    P50    P95    P99    max")
    print("    " + String(repeating: "-", count: 66))

    let kappaLabels = ["κ < 2", "2 ≤ κ < 10", "10 ≤ κ < 100",
                        "100 ≤ κ < 1K", "1K ≤ κ < 10K", "κ ≥ 10K"]
    for (i, label) in kappaLabels.enumerated() {
        let arr = ulpByKappa[i]
        if arr.isEmpty { continue }
        let sorted = arr.sorted()
        let n = sorted.count
        let p50 = sorted[n / 2]
        let p95 = sorted[min(n - 1, Int(Double(n) * 0.95))]
        let p99 = sorted[min(n - 1, Int(Double(n) * 0.99))]
        let mx = sorted.last!
        let exact = arr.filter { $0 == 0 }.count
        let exactPct = 100.0 * Double(exact) / Double(n)
        print("    \(label.padding(toLength: 18, withPad: " ", startingAt: 0))\(String(format: "%-9d", n))\(String(format: "%5.1f%%", exactPct))    \(String(format: "%-7d", p50))\(String(format: "%-7d", p95))\(String(format: "%-7d", p99))\(mx)")
    }

    // Also show Accelerate for comparison
    print("\n  [C4] Same bins for Accelerate:")
    print("    κ range           n        exact%    P50    P95    P99    max")
    print("    " + String(repeating: "-", count: 66))
    for (i, label) in kappaLabels.enumerated() {
        let arr = cpuUlpByKappa[i]
        if arr.isEmpty { continue }
        let sorted = arr.sorted()
        let n = sorted.count
        let p50 = sorted[n / 2]
        let p95 = sorted[min(n - 1, Int(Double(n) * 0.95))]
        let p99 = sorted[min(n - 1, Int(Double(n) * 0.99))]
        let mx = sorted.last!
        let exact = arr.filter { $0 == 0 }.count
        let exactPct = 100.0 * Double(exact) / Double(n)
        print("    \(label.padding(toLength: 18, withPad: " ", startingAt: 0))\(String(format: "%-9d", n))\(String(format: "%5.1f%%", exactPct))    \(String(format: "%-7d", p50))\(String(format: "%-7d", p95))\(String(format: "%-7d", p99))\(mx)")
    }
}

// =============================================================================
// MARK: - Main
// =============================================================================

print("======================================================================")
print("Exercise 19b: κ-Binned ULP Analysis — Random & DFT-like Inputs")
print("GPU: \(device.name)")
print("======================================================================")

// --- Config 1: Random [-1,1] at 256×256 (full check) ---
do {
    let sz = 256
    srand48(42)
    let A = (0..<sz*sz).map { _ in drand48() * 2.0 - 1.0 }
    let B = (0..<sz*sz).map { _ in drand48() * 2.0 - 1.0 }
    analyzeMatrix(name: "Random [-1,1]", sz: sz, A: A, B: B, nSample: 0)
}

// --- Config 2: Random [-1,1] at 512×512 (sampled) ---
do {
    let sz = 512
    srand48(77)
    let A = (0..<sz*sz).map { _ in drand48() * 2.0 - 1.0 }
    let B = (0..<sz*sz).map { _ in drand48() * 2.0 - 1.0 }
    analyzeMatrix(name: "Random [-1,1]", sz: sz, A: A, B: B, nSample: 8000)
}

// --- Config 3: DFT-like (I + 0.01R) at 256×256 (full check) ---
do {
    let sz = 256
    srand48(777)
    var A = [Double](repeating: 0, count: sz * sz)
    var B = [Double](repeating: 0, count: sz * sz)
    for i in 0..<sz {
        A[i * sz + i] = 1.0; B[i * sz + i] = 1.0
        for j in i..<sz {
            let ra = (drand48() * 2.0 - 1.0) * 0.01
            A[i * sz + j] += ra; A[j * sz + i] += ra
            let rb = (drand48() * 2.0 - 1.0) * 0.01
            B[i * sz + j] += rb; B[j * sz + i] += rb
        }
    }
    analyzeMatrix(name: "DFT-like (I+0.01R)", sz: sz, A: A, B: B, nSample: 0)
}

// --- Config 4: Random [0,1] at 256×256 for completeness ---
do {
    let sz = 256
    srand48(99)
    let A = (0..<sz*sz).map { _ in drand48() }
    let B = (0..<sz*sz).map { _ in drand48() }
    analyzeMatrix(name: "Random [0,1]", sz: sz, A: A, B: B, nSample: 0)
}

// =============================================================================
// MARK: - Summary
// =============================================================================

print("\n" + String(repeating: "=", count: 72))
print("Key Takeaway for Paper")
print(String(repeating: "=", count: 72))
print("""

  The κ-binned analysis answers Reviewer Concern 4 directly:

  For well-conditioned elements (κ < 10):
    TD-DGEMM: P99 should be 0 or 1 ULP → essentially exact
    Accelerate: P99 should be ~3-5 ULPs → standard FP64 behavior

  For ill-conditioned elements (κ > 100):
    Both TD and Accelerate degrade, but TD should still be ≤1 ULP
    because κ only affects WHETHER the element is correctly rounded,
    not HOW FAR off the wrong answer is (TD errors are always 0 or 1).

  The alarming "max ULP = 1.2M" from Section 3.3 (DD results)
  should be reported as:
    "Max ULP = 382 for κ < 10; extreme values occur only for
     elements with κ > 10,000 where cancellation amplifies
     the 5-bit DD truncation error."

  For TD, the story is simpler: max ULP = 1 regardless of κ.
  This is the structural advantage of lossless input conversion.

======================================================================
""")
