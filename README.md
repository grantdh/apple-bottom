# apple-bottom

**BLASphemy: Metal-native linear algebra for Apple Silicon...because cuBLAS isn't coming to Cupertino**

## The Problem

Scientific HPC codes (Quantum ESPRESSO, Yambo, etc.) require **double-precision (FP64) BLAS** and need to call DGEMM from multiple threads simultaneously. On Apple Silicon:

- **MPS (Metal Performance Shaders)** — Apple's official answer, hitting 2.9 TFLOPS on M4, but **FP32 only**. Great for ML, useless for HPC.
- **Accelerate's cblas_dgemm** — Uses AMX for FP64, but **thread-hostile**: parallel calls from multiple threads wreck performance.
- **cuBLAS** — Not coming to Cupertino. Ever.

The gap: **No drop-in, thread-safe, FP64 BLAS via Metal GPU compute for HPC workloads.**

## The Solution

`apple-bottom` is a DYLD-interposing, drop-in **FP64 Metal BLAS** layer targeting scientific HPC codes. The goal: outperform Accelerate in multi-threaded FP64 workloads by leveraging Metal GPU compute where AMX falls short.

MPS does FP32. Accelerate does AMX. **Nobody does FP64 Metal. Until now.**

## Prior Art (and Why This Exists)

**What already exists:**

- **Apple MPS** — The official ML-focused GPU BLAS. FP32 only. [arXiv](https://arxiv.org/abs/2301.10803)
- **AppleNumericalComputing** (ShoYamanishi) — CUDA handbook idioms in Metal, comprehensive benchmarks. Research-grade, not a drop-in library. [GitHub](https://github.com/ShoYamanishi/AppleNumericalComputing)
- **philip-turner/metal-benchmarks** — Microarchitecture docs for M1/M2 GPU. Reference material, not a library. [GitHub](https://github.com/philipturner/metal-benchmarks)
- **amx_sgemm** (Fnk7) — Direct AMX assembly via undocumented opcodes. Hacky, unmaintained. [GitHub](https://github.com/Fnk7/amx_sgemm)

**The critical gap `apple-bottom` fills:**

Metal currently only supports single precision for GPU compute. AMX provides double-precision via Accelerate, but multi-threaded HPC codes suffer when hammering `cblas_dgemm` concurrently. [Discussion](https://forums.macrumors.com/threads/apple-silicon-gpu-performance.2306988/)

There's benchmarking research and ML-focused Metal work, but **zero drop-in FP64 BLAS replacements via Metal interposition for HPC**. That's the white space.

## Status

Early development. DGEMM kernel and complex arithmetic exercises in progress.

## License

See LICENSE file.
