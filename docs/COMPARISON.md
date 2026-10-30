# FP64 matrix multiplication on Apple Silicon: the options

Apple Silicon GPUs have no FP64 arithmetic units, so FP64 matrix work on a
Mac runs either on the CPU or on the GPU's FP32 units through some form of
emulation. This note compares the options that exist and says when
apple-bottom is the right one. Status and descriptions of other projects
are as of October 2026, from their repositories.

| | Runs on | Arithmetic | Significand | BLAS coverage |
|---|---|---|---|---|
| **Accelerate** (Apple) | CPU (matrix coprocessor) | IEEE FP64 | 53 bits | Full BLAS and LAPACK |
| **OpenBLAS** and other CPU BLAS | CPU cores | IEEE FP64 | 53 bits | Full BLAS (LAPACK alongside) |
| **apple-bottom** | GPU (Metal) | double-float: a pair of FP32 values per FP64 | ~48 bits | DGEMM, ZGEMM, DSYRK and helpers; `dgemm_`/`zgemm_` through the BLAS ABI |
| [**metal-float64**](https://github.com/philipturner/metal-float64) (Turner) | GPU (Metal) | IEEE FP64 emulated in software | 53 bits (goal) | Scalar arithmetic library; the README describes it as unfinished. Archived. |
| [**MLX**](https://github.com/ml-explore/mlx) (Apple) | CPU for FP64; GPU for FP32/FP16/BF16 | IEEE FP64 on the CPU | 53 bits | An array framework for machine learning, not a BLAS |

## Accelerate

Accelerate is the reference apple-bottom is measured against. It is exact
IEEE FP64, complete, and fast: on an M2 Max it beat the GPU path of
apple-bottom 1.4.0 on every DGEMM measured through the BLAS ABI and on
every Quantum ESPRESSO shape ([README](../README.md#performance)), and
Quantum ESPRESSO forced onto 1.5.0's GPU path is still not faster than with
Accelerate. If your code
calls BLAS one call at a time on host arrays, Accelerate is the right
choice on this hardware today, and apple-bottom's default routing
(`AB_MODE=auto`) sends those calls there anyway.

## apple-bottom

apple-bottom keeps each value as an unevaluated sum of two FP32 numbers and
runs error-free transformations on the GPU's FP32 units, compiled without
fast math. That gives about 10⁻¹⁵ relative error per element instead of
FP64's 10⁻¹⁶, with FP32's exponent range bridged by power-of-two scaling
(and Accelerate for BLAS-ABI calls no scaling keeps in range). Where it helps:

- Large GEMMs on matrices kept on the GPU between calls (the native
  `ABMatrix` API), where it overtakes Accelerate on an M2 Max from about
  N = 1,500 (DGEMM) and N = 1,000 (ZGEMM).
- Codes that want measured routing between GPU and CPU behind the standard
  `dgemm_`/`zgemm_` interface, with `AB_REPORT` accounting.

Where it does not:

- Bit-for-bit IEEE FP64 results, or native-API matrices whose entries span
  more than about 10²³.
- GEMMs issued one call at a time from host memory through the BLAS ABI.
  On an M2 Max, 1.4.0's host path, which packed and converted on the CPU,
  was no more than 1.02× faster than Accelerate for any measured shape
  except ZGEMM with both operands transposed at N ≥ 4096 (1.11–1.20×).
  1.5.0's staged path converts on the GPU and has not been calibrated yet.
  Quantum ESPRESSO forced onto it takes 1.01–1.06× its time on Accelerate,
  against 1.06–1.43× for 1.4.0 on the same QE build
  ([release check](../benchmarks/results/2026-10-28-415cb5d-qe-si64-release-check/),
  [staged-path fix](../benchmarks/results/2026-10-29-351d56c-staged-refusals/)).
  The device-buffer API converts on the CPU on every call, as 1.4.0's BLAS
  ABI did (it has not been benchmarked separately).
- Anything beyond the GEMM family: the rest of BLAS and LAPACK comes from
  your BLAS library.

## metal-float64

metal-float64 set out to emulate IEEE-compliant FP64 on Apple GPUs in
software, as a Metal library of scalar operations. Its README describes it
as still in the planning stage and estimates 1/32 to 1/64 of FP32 throughput
for additions and multiplications; the repository has been archived since
2023. It aims at exact IEEE semantics, which apple-bottom gives up in
exchange for running on the FP32 units directly.

## MLX

MLX is Apple's array framework for machine learning. Its GPU kernels use
FP32, FP16 and BF16; FP64 runs on the CPU. It is the better choice for
machine-learning workloads and not a replacement for BLAS in scientific
codes.

## Choosing

| You need | Use |
|---|---|
| Exact IEEE FP64, all of BLAS/LAPACK | Accelerate (or another CPU BLAS) |
| FP64-class GEMMs on the GPU, matrices kept resident | apple-bottom, native `ABMatrix` API |
| An unchanged Fortran code calling `DGEMM`/`ZGEMM` | Accelerate today on an M2 Max; apple-bottom routes there by default and reports what it did |
| Machine learning | MLX |
