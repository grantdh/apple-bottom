# apple-bottom 🍑

[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](LICENSE)
[![Platform](https://img.shields.io/badge/platform-macOS%2013%2B-lightgrey.svg)](https://www.apple.com/macos/)
[![Swift](https://img.shields.io/badge/Swift-5.9+-orange.svg)](https://swift.org)
[![Metal](https://img.shields.io/badge/Metal-3.0+-red.svg)](https://developer.apple.com/metal/)

**BLASphemy: Metal-native linear algebra for Apple Silicon...because cuBLAS isn't coming to Cupertino**

## Status

**Faithfully-rounded TD-DGEMM kernel complete.** Triple-float emulation achieves 99.5% correctly-rounded FP64 output on FP32 GPU — a new point on the precision–performance Pareto frontier. DD-DGEMM (double-float) beats Accelerate AMX by 14–20% for matrices ≥1024×1024.

| Milestone | Status |
|-----------|--------|
| DD-DGEMM kernel (4×4 optimized) | ✅ Complete |
| DD-ZGEMM (Gauss 3-multiply) | ✅ Complete |
| DSYRK/ZHERK analysis | ✅ Complete |
| TD-DGEMM kernel (faithfully-rounded) | ✅ Complete |
| TD validation (279K elements, 4 types) | ✅ Complete |
| Negative results (Ziv, DD correction) | ✅ Complete |
| DYLD interposition | 🔜 Planned |
| QE/Yambo integration | 🔜 Planned |

## The Problem

Scientific HPC codes (Quantum ESPRESSO, Yambo, etc.) require **double-precision (FP64) BLAS** and need to call DGEMM from multiple threads simultaneously. On Apple Silicon:

- **MPS (Metal Performance Shaders)** — Apple's official answer, hitting 2.9 TFLOPS on M4, but **FP32 only**. Great for ML, useless for HPC.
- **Accelerate's cblas_dgemm** — Uses AMX for FP64, but **thread-hostile**: parallel calls from multiple threads wreck performance.
- **cuBLAS** — Not coming to Cupertino. Ever.

The gap: **No drop-in, thread-safe, FP64 BLAS via Metal GPU compute for HPC workloads.**

## The Solution

`apple-bottom` is a DYLD-interposing, drop-in **FP64 Metal BLAS** layer targeting scientific HPC codes. It provides two operating points on the precision–performance frontier:

- **DD-DGEMM (fast tier):** ~48-bit precision at 640 GFLOP/s — beats AMX for large matrices
- **TD-DGEMM (precise tier):** Faithfully-rounded FP64 at 148 GFLOP/s — provably ≤1 ULP error

MPS does FP32. Accelerate does AMX. **Nobody does FP64 Metal. Until now.**

## Performance

### The Precision–Performance Frontier (M2 Max)

| Implementation | GFLOP/s | Correct Rounding | Max ULP | Faithful | Reproducible |
|---|---|---|---|---|---|
| DD-DGEMM (GPU) | 640 | ~0.5% | 249 | No | Yes |
| Sequential FP64 (CPU) | ~0.5 | 11–23% | 9–18* | No | Yes |
| Accelerate AMX (CPU) | 536 | 11–23% | 9–18* | No | Yes |
| **TD-DGEMM (GPU)** | **148** | **99.5%** | **1** | **Yes** | **Yes** |

*Max ULP for well-conditioned inputs (κ < 10). Extreme values occur only at κ > 10,000.*

### DD-DGEMM Throughput (M2 Max, 30-core GPU)

| Matrix Size | DD-GPU | AMX-CPU | Winner | Margin |
|-------------|--------|---------|--------|--------|
| 512×512 | 450 GFLOP/s | 550 GFLOP/s | AMX | AMX +22% |
| **1024×1024** | **552 GFLOP/s** | 468 GFLOP/s | **DD ✓** | **DD +18%** |
| 2048×2048 | 629 GFLOP/s | 525 GFLOP/s | **DD ✓** | DD +20% |
| 4096×4096 | 643 GFLOP/s | 566 GFLOP/s | **DD ✓** | DD +14% |

**Crossover point: ~1024×1024** — GPU wins for large matrices typical in DFT calculations.

## Precision

| Method | Mantissa Bits | Relative Error | Correct Rounding | Use Case |
|--------|---------------|----------------|-------------------|----------|
| FP32 | 24 | ~10⁻⁷ | N/A | Graphics, ML inference |
| **FP32×2 (DD)** | **~48** | **~10⁻¹⁵** | **~0.5%** | **High-throughput scientific** ✓ |
| Native FP64 | 53 | ~10⁻¹⁶ | ~20% | Standard double precision |
| **FP32×3 (TD)** | **~72** | **~10⁻²²** | **99.5%** | **Faithfully-rounded FP64** ✓ |

TD-DGEMM is the first GPU BLAS implementation providing a proven faithful-rounding guarantee for GEMM. See the [paper](docs/) for the formal proofs.

## Quick Start

```bash
# Clone the repo
git clone https://github.com/grantdh/apple-bottom.git
cd apple-bottom

# Build and run the production DD kernel (fast tier)
swiftc -O -framework Metal -framework Foundation -framework Accelerate \
    ex10_production_dd_dgemm.swift -o dd_gemm
./dd_gemm

# Build and run the TD kernel (precise tier)
swiftc -O -Xcc -DACCELERATE_NEW_LAPACK -framework Metal \
    -framework Foundation -framework Accelerate ex16_td_dgemm.swift -o td_gemm
./td_gemm

# Run the comprehensive validation (ex19 + ex19b)
swiftc -O -Xcc -DACCELERATE_NEW_LAPACK -framework Metal \
    -framework Foundation -framework Accelerate ex19_reviewer_response.swift -o ex19
./ex19
```

## Prior Art (and Why This Exists)

**What already exists:**

- **Apple MPS** — The official ML-focused GPU BLAS. FP32 only. [arXiv](https://arxiv.org/abs/2301.10803)
- **AppleNumericalComputing** (ShoYamanishi) — CUDA handbook idioms in Metal, comprehensive benchmarks. Research-grade, not a drop-in library. [GitHub](https://github.com/ShoYamanishi/AppleNumericalComputing)
- **philip-turner/metal-benchmarks** — Microarchitecture docs for M1/M2 GPU. Reference material, not a library. [GitHub](https://github.com/philipturner/metal-benchmarks)
- **amx_sgemm** (Fnk7) — Direct AMX assembly via undocumented opcodes. Hacky, unmaintained. [GitHub](https://github.com/Fnk7/amx_sgemm)
- **Ozaki Scheme (NVIDIA)** — FP64 via INT8 Tensor Cores. 71 TFLOP/s on GH200. Requires dedicated integer matrix units Apple Silicon lacks.

**The critical gap `apple-bottom` fills:**

Metal currently only supports single precision for GPU compute. AMX provides double-precision via Accelerate, but multi-threaded HPC codes suffer when hammering `cblas_dgemm` concurrently. There's benchmarking research and ML-focused Metal work, but **zero drop-in FP64 BLAS replacements via Metal interposition for HPC**. Additionally, no existing GPU BLAS provides proven faithful-rounding guarantees for GEMM output.

## Key Technical Decisions

### 1. `mathMode = .safe` is MANDATORY

```swift
let opts = MTLCompileOptions()
opts.mathMode = .safe  // CRITICAL: .fast breaks error-free transforms
```

Fast math enables FMA fusion that violates the exact algebraic identities required by twoSum/twoProduct. This applies to both DD and TD arithmetic.

### 2. DD: 4×4 Register Blocking is Optimal

| Block | Accumulators | Registers | Performance | Notes |
|-------|-------------|-----------|-------------|-------|
| 2×2 | 4 DD | ~36 | 560 GFLOP/s | Baseline |
| **4×4** | **16 DD** | **~68** | **640 GFLOP/s** | **Optimal** ✓ |
| 6×6 | 36 DD | ~116 | 215 GFLOP/s | Poor threadgroup fit |
| 8×8 | 64 DD | ~180 | Spills | Exceeds register limit |

### 3. TD: 2×2 Register Blocking is Optimal

TD has 3 floats per component (vs DD's 2), so the register budget shifts:

| Block | Accumulators | Registers | Performance | Notes |
|-------|-------------|-----------|-------------|-------|
| **2×2** | **4 TD (12 floats)** | **~49** | **148 GFLOP/s** | **Optimal** ✓ |
| 4×4 | 16 TD (48 floats) | ~97 | 108 GFLOP/s | Occupancy limited |

### 4. simdgroup_matrix Cannot Support DD or TD

The hardware matrix units accumulate internally in FP32. Multi-precision arithmetic applied afterward cannot recover the lost precision. Register-blocked scalar arithmetic is required. See `ex08b` and `ex08c` for the failed experiments.

### 5. Interleaved Storage for Both DD and TD

```metal
// ✓ CORRECT: Interleaved
struct dd { float hi; float lo; };
struct td { float hi; float md; float lo; };

// ✗ WRONG: Separate buffers (kills cache performance)
device const float *A_hi, *A_md, *A_lo;
```

## Project Structure

```
apple-bottom/
├── README.md                          # This file
├── LESSONS_LEARNED.md                 # Technical discoveries and gotchas
├── TEST_PLAN.md                       # Current status and roadmap
│
├── Foundation (ex01–06)
│   ├── ex01_complex_arithmetic.swift
│   ├── ex02_complex_dot_product.swift
│   ├── ex03_tiled_sgemm.swift
│   ├── ex04_register_blocked_cgemm.swift
│   ├── ex04b_block_size_comparison.swift
│   ├── ex05_stockham_fft.swift
│   └── ex06_batched_3d_fft.swift
│
├── DD-BLAS Development (ex07–09)
│   ├── ex07_double_float_primitives.swift    # DD arithmetic
│   ├── ex08_dd_dgemm.swift                   # 2×2 baseline
│   ├── ex08b_simdgroup_dd_dgemm.swift        # Failed: simdgroup experiment
│   ├── ex08c_ozaki_dd_dgemm.swift            # Failed: Ozaki scheme on Metal
│   └── ex09[a–e]_*.swift                     # Crossover analysis
│
├── Production DD (ex10)
│   └── ex10_production_dd_dgemm.swift        # ✅ Production 4×4 DD kernel
│
├── Complex & Symmetric (ex11–13)
│   ├── ex11_dd_zgemm.swift                   # Native DD complex (failed)
│   ├── ex11b_split_zgemm.swift               # Split 4×DGEMM
│   ├── ex11c_optimized_zgemm.swift           # ✅ Gauss 3-multiply ZGEMM
│   ├── ex12[b–g]_dd_dsyrk.swift              # DSYRK analysis
│   └── ex13[b–g]_dd_zherk.swift              # ZHERK analysis
│
├── TD-DGEMM: Faithfully-Rounded FP64 (ex15d–19b)
│   ├── ex15d_td_proof_validation.swift       # Proofs: Lemma 1 + 2 + Theorem
│   ├── ex16_td_dgemm.swift                   # ✅ TD-DGEMM GPU kernel
│   ├── ex17e_twosided_ziv.swift              # Negative: Ziv certification fails
│   ├── ex18_dd_correction.swift              # Negative: DD correction ineffective
│   ├── ex19_reviewer_response.swift          # Validation: faithful rounding, C₀, κ
│   └── ex19b_random_kappa.swift              # Validation: κ-binned ULP analysis
│
└── Integration (planned)
    └── ex14_dyld_interposition.c             # Drop-in library (next)
```

## Roadmap

### Phase 1: Foundation ✅
Complex arithmetic, tiled GEMM, FFT primitives

### Phase 2: DD-BLAS Development ✅
Double-float primitives, DD-DGEMM kernel, crossover analysis

### Phase 3: Production DD ✅
Optimized 4×4 kernel (`ex10_production_dd_dgemm.swift`)

### Phase 4: Complex & Symmetric ✅
Gauss 3-multiply ZGEMM (+23–43% vs AMX), DSYRK/ZHERK analysis

### Phase 5: Triple-Float Precision ✅
- TD-DGEMM: faithfully-rounded FP64 GEMM on GPU (`ex16`)
- Proven correct-rounding bound: K×κ < 2¹⁹/C₀ (`ex15d`)
- Verified faithful rounding: 1,457/1,457 bracket tests passed (`ex19`)
- κ-conditioned analysis: max ULP = 1 regardless of κ (`ex19b`)
- Negative results: Ziv certification (`ex17e`), DD correction (`ex18`)
- Validation: 279,144 elements, 4 matrix types, sizes 128–512

### Phase 6: Integration 🔜
- DYLD interposition library
- QE/Yambo integration and testing
- Cross-device validation (M1, M3, M4)

## Requirements

- **macOS 13+** (Ventura or later)
- **Apple Silicon** (M1/M2/M3/M4 series)
- **Xcode Command Line Tools** (`xcode-select --install`)

## References

1. **Rump, Ogita, Oishi** — Accurate floating-point summation (SIAM SISC, 2008)
2. **Joldes, Muller, Popescu** — Triple-word arithmetic error bounds (ACM TOMS, 2017)
3. **Ozaki et al.** — Error-free matrix multiplication (Numer. Algorithms, 2012)
4. **Mukunoki et al.** — DGEMM using Tensor Cores (LNCS 12151, 2020)
5. **Demmel, Nguyen** — Parallel reproducible summation (IEEE TC, 2015)
6. **Higham** — Accuracy and Stability of Numerical Algorithms (SIAM, 2002)
7. **Muller** — Elementary Functions, 3rd ed. (Birkhäuser, 2016)
8. **QD Library** — Bailey's quad-double arithmetic (LBNL-46996, 2005)
9. **Philip Turner's metal-float64** — Metal FP64 emulation experiments

## License

Apache 2.0 — see [LICENSE](LICENSE) file.

## Citation

```bibtex
@software{heileman2026applebottom,
  author    = {Heileman, Grant David},
  title     = {apple-bottom: BLASphemy — Metal-native linear algebra for Apple Silicon},
  year      = {2026},
  publisher = {Technology Residue},
  url       = {https://github.com/grantdh/apple-bottom}
}
```

## Contributing

Contributions welcome! See [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

For bug reports and feature requests, please [open an issue](https://github.com/grantdh/apple-bottom/issues).

## Author

**Grant David Heileman**
University of New Mexico, Department of Electrical & Computer Engineering

📧 Contact: [Open an issue](https://github.com/TechnologyResidue/apple-bottom/issues) for questions
🔗 More projects: [Technology Residue](https://github.com/TechnologyResidue)

## Acknowledgments

- QD Library by David H. Bailey for quad-double arithmetic foundations
- Philip Turner for Metal GPU microarchitecture documentation
- The Quantum ESPRESSO and Yambo development teams
