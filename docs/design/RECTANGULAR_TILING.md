# Rectangular Matrix Tile-Dispatch Design Note

This note documents the rationale for apple-bottom's default tile configuration (BM=BN=64, TM=TN=4, TK=16), the tall-skinny variant (BM=128, BN=16) used for QE's Davidson eigensolver shapes and the narrow-output variant (BM=BN=32) for small C. It is intended as a reference for future tile-size tuning on new Apple Silicon generations or workloads with different aspect ratios.

DD error stays within its O(K)·2⁻⁴⁸ componentwise bound for all tested aspect ratios from 121:1 (tall-skinny) to 1:100 (short-wide); `test_rectangular.c` checks this on tall, wide and thin-K shapes up to 18277 × 150 × 18277.

---

## Tile-Dispatch Model

**Default tile configuration** (square and moderate-aspect shapes):
- BM = BN = 64 (block tiles in M, N)
- TM = TN = 4 (thread tiles)
- TK = 16 (K-slab width)
- Tuned for M ≈ N ≈ K

Each threadgroup computes a BM×BN tile of C by iterating TK-wide slabs across the K dimension. The grid dimension is `ceil(M/BM) × ceil(N/BN)` threadgroups; threads within a threadgroup cooperatively load A and B tiles into threadgroup memory before the TK-wide inner product.

**Tall-skinny variant** (deployed; `BM_TS`/`BN_TS` in `src/apple_bottom.m`):
- BM = 128, BN = 16
- TM = 4, TN = 2
- Threadgroup layout 8×32 = 256 threads
- Selected on `M >= 4 * N` (`use_tall_skinny` in `src/apple_bottom.m`) by `ab_dgemm_scaled` (and so the DGEMM BLAS ABI) and by ZGEMM's three real products, unless the narrow-output variant takes the shape; `ab_dgemm` never uses it

The tall-skinny kernel reduces boundary waste for N = 150 from ~22% under BN=64 (3 tiles, 192 columns computed) to ~6% under BN=16 (10 tiles, 160 columns).

**Narrow-output variant** (deployed; `BM_NW`/`BN_NW` in `src/apple_bottom.m`):
- BM = BN = 32, TM = TN = 2, TK = 16
- Threadgroup layout 16×16 = 256 threads
- Selected by `use_narrow_tiles` for a C of at most 480 × 480 elements: in place of the 64 × 64 kernel, and in place of the tall-skinny one only where its 32 columns per tile add no padding to that kernel's 16 (N = 17-32, 49-64, ...) or where the tall-skinny grid would have at most 16 threadgroups, as for Davidson's overlaps with a small subspace (200 × 40 × 18277: 6 threadgroups, 2.1× faster on 32 × 32 tiles)
- Four times the threadgroups of the 64 × 64 tiling: a 160 × 160 C is 25 threadgroups instead of 9, 2.6× faster on 160 × 160 × 18432 on a 38-core M2 Max. 512 × 512 breaks even, so larger outputs keep the 64 × 64 tiles
- Each output gets the same dd_fma sequence and renormalizations as in the other kernels, so the results are bitwise the same

**Register prefetch** (deployed; the `*_pf` instances of every tiled kernel):
- The kernel loads K-tile k+1 from device memory into registers while it computes tile k, and stores it to threadgroup memory at the top of the next iteration; the inner loop and every tile slot are those of the plain kernel, so the bits are the same
- It hides the tile loads' latency where a GPU core holds one or two threadgroups, and costs occupancy otherwise (`maxTotalThreadsPerThreadgroup` 832 → 512 for 64 × 64, 896 → 576 tall-skinny, 1024 → 896 narrow)
- Selected by `use_prefetch`: up to 64 threadgroups for the 64 × 64 kernels and DSYRK, and for 64 × 64 only in a serial encoder; up to 512 for tall-skinny; every narrow grid. On a 38-core M2 Max: 1.43× on 160 × 160 × 18432 (narrow), 1.09× on 512 × 512 × 18432 (64 × 64), up to 1.47× tall-skinny. Re-calibrate the limits on a GPU with another core count or register file

**Tall-skinny stress shape used by the tests:**
- M = 18,277, N = 150, K = 18,277: synthetic. Quantum ESPRESSO's Davidson
  GEMMs share the long M (plane waves) and short N (bands), but their K is a
  few hundred, e.g. 17,846 × 160 × 320 in [VAL-1](../../tests/validation/VAL001_QE_Si64.md)'s
  input C
- Aspect ratio 121:1 (M:N), which selects the tall-skinny kernel where it applies

**Dispatch utilization under the default 64×64 tile (pre-variant, illustrative):**
```
Threadgroups in M dimension: ceil(18277 / 64) = 286
Threadgroups in N dimension: ceil(150 / 64) = 3
Total threadgroups: 286 × 3 = 858
```

Consequences of the default 64×64 tile at 121:1 aspect:
- Good parallelism in M (286 threadgroups saturate the 38-core GPU)
- Poor parallelism in N (only 3 threadgroups; residual partial-block work)
- Many threadgroups carry partial work (M and N not multiples of 64)

These are the conditions that motivated the tall-skinny variant above.

---

## Test Suite: test_rectangular.c

### Correctness Tests

**1. DGEMM Rectangular Tests**
- `test_dgemm_tall_skinny()` — 10000 × 100 × 100 × 100
- `test_dgemm_short_wide()` — 100 × 10000 × 100 × 10000
- `test_dgemm_qe_dimensions()` — 18277 × 150 × 18277 × 150
- `test_dgemm_thin_middle()` — 5000 × 5000 × 10 × 5000

**2. ZGEMM Rectangular Tests**
- `test_zgemm_qe_dimensions()` — 18277 × 150 × 18277 × 150
- `test_zgemm_conjugate_transpose_qe()` — 150 × 150 × 18277 × 150, through `ab_zgemm_ex`

**Validation:**
- All tests validate against reference BLAS (cblas_dgemm, cblas_zgemm)
- Error measure: componentwise, |C − C_ref| / (|op(A)| |B|), against 10·K·u² (u² = 2⁻⁴⁸); measured errors are about 1e-14
- The DGEMM cases call `ab_dgemm`, so they run the 64 × 64 kernel; of the ZGEMM cases only the QE-like one (18277 × 150, M ≥ 4N) runs the tall-skinny kernel, and the conjugate-transpose case (150 × 150 × 18277) runs the 64 × 64 kernel

### Performance Benchmarks

**benchmark_aspect_ratios()** tests:
1. Square: 2048 × 2048 × 2048
2. Tall 4:1: 4096 × 1024 × 2048
3. Tall 16:1: 8192 × 512 × 2048
4. Tall 64:1: 16384 × 256 × 2048
5. QE-like (synthetic stress shape): 18277 × 150 × 18277
6. Wide 1:4: 1024 × 4096 × 2048
7. Wide 1:16: 512 × 8192 × 2048

Metrics: GPU time (ms), BLAS time (ms), speedup ratio.

---

## Measured Dispatch Utilization at Representative QE Shapes

The tables below illustrate the M×N×K trade-space for tile dispatch; they are not a regression baseline. All numbers measured on M2 Max / 38-core GPU / 64 GB unified memory.

### Correctness

Rectangular matrices meet DD precision bounds across the full aspect-ratio sweep:

- Gauss 3-multiply is dimension-agnostic.
- Double-float arithmetic is dimension-agnostic.
- Componentwise error (relative to |A||B|) stays about 1e-14 up to K = 18277 in `test_rectangular`; normwise error grows like √K (5e-14 at K = 4096 in the V-2 study).

### Performance by Aspect Ratio

`make test-rectangular` on an M2 Max, 2026-10-14 (`ab_dgemm`, so the 64 × 64
kernel for every shape; matrices uploaded once and the multiply timed;
Accelerate `cblas_dgemm` on the same data; one run each, so treat
differences under ~10% as noise):

| Shape (M × N × K) | GPU (ms) | Accelerate (ms) | GPU speed vs Accelerate |
|---|---:|---:|---:|
| Square 2048 × 2048 × 2048 | 27.9 | 31.2 | 1.12× |
| Tall 4:1, 4096 × 1024 × 2048 | 35.9 | 26.8 | 0.75× |
| Tall 16:1, 8192 × 512 × 2048 | 36.7 | 27.1 | 0.74× |
| Tall 64:1, 16384 × 256 × 2048 | 36.7 | 33.0 | 0.90× |
| Stress shape, 18277 × 150 × 18277 | 233.8 | 189.1 | 0.81× |
| Wide 1:4, 1024 × 4096 × 2048 | 43.0 | 35.2 | 0.82× |
| Wide 1:16, 512 × 8192 × 2048 | 32.0 | 32.0 | 1.00× |

**Why tall shapes were slow under the default 64 × 64 tile** (the reasoning behind the tall-skinny variant; these causes were not measured separately):

1. **Threadgroup underutilization.** Under BN=64, N=150 creates only 3 threadgroups in N; many threads idle in partial blocks.

2. **Transfers, when matrices are uploaded per call** (not in the resident timings above; stress shape).
   - A upload: 18277 × 18277 × 8 bytes = 2.5 GB
   - B upload: 18277 × 150 × 8 bytes = 21 MB
   - C download: 18277 × 150 × 8 bytes = 21 MB
   - Total per call: 2.5 GB

3. **Cache locality (hypothesis).** B (150 columns) fits in cache; A (18277 rows) does not, so tall shapes walk long strides.

The tall-skinny kernel addresses (1) directly: at BN=16, N=150 yields ceil(150/16) = 10 threadgroups in N rather than 3, and boundary waste drops from ~22% to ~6%. (2) and (3) are upload/download and stride concerns inherent to the per-call API. They motivated the device-buffer API (`apple_bottom_device.h`) and 1.5.0's staged BLAS-ABI path, which keeps its GPU buffers between calls and can read registered host memory in place.

**What this means for Quantum ESPRESSO.**

The numbers above are for the native `ABMatrix` API with one call per
upload. Through the BLAS ABI (`dgemm_`/`zgemm_`, which is what QE
calls) every operand is also converted FP64 → DD and back on each
call. On 1.4.0's host path, which did that on the CPU, every QE si64
Davidson and PAW shape measured by `make calibrate` was slower on the GPU
than on Accelerate (0.09–0.76×,
[`benchmarks/results/2026-10-14-52f1880-calibrate/`](../../benchmarks/results/2026-10-14-52f1880-calibrate/)),
and `AB_MODE=auto` keeps those calls on Accelerate. 1.5.0's staged path
converts on the GPU and has not been calibrated yet; QE forced onto it
takes 1.01–1.06× its time on Accelerate
([`benchmarks/results/2026-10-29-351d56c-staged-refusals/`](../../benchmarks/results/2026-10-29-351d56c-staged-refusals/)).
The tall-skinny kernel still matters for `AB_MODE=gpu` and for callers
that keep matrices resident on the GPU.

---

## Running the Tests

### Build and run
```bash
make test-rectangular
```

Output on an M2 Max (correctness part):
```
=============================================================================
apple-bottom Rectangular Matrix Test Suite
=============================================================================

DGEMM Rectangular Tests:
  DGEMM tall-skinny (10000 × 100 × 100 × 100)     (err 1.3e-14, bound 3.6e-12) ✓ PASS
  DGEMM short-wide (100 × 10000 × 100 × 10000)    (err 1.3e-14, bound 3.6e-12) ✓ PASS
  DGEMM QE-like (18277 × 150 × 18277 × 150)       (err 1.2e-14, bound 6.5e-10) ✓ PASS
  DGEMM thin middle (5000 × 5000 × 10 × 5000)     (err 1.4e-14, bound 3.6e-13) ✓ PASS

ZGEMM Rectangular Tests:
  ZGEMM QE-like (18277 × 150 × 18277 × 150)       (err 1.4e-14, bound 6.5e-10) ✓ PASS
  ZGEMM QE conjugate transpose (150 × 150 × 18277 × 150) (err 7.7e-15, bound 6.5e-10) ✓ PASS

=============================================================================
Summary: 6 passed, 0 failed
=============================================================================
```

---

## References

- Tall-skinny kernel parameters: `BM_TS`, `BN_TS`, `TM_TS`, `TN_TS` in `src/apple_bottom.m`
- Narrow-output kernel parameters: `BM_NW`, `BN_NW`, `TM_NW`, `TN_NW` in `src/apple_bottom.m`
- Dispatch rules: `use_tall_skinny` (`M >= 4 * N`), `use_narrow_tiles` and `use_prefetch` in `src/apple_bottom.m`
- QE integration: [`docs/INTEGRATION.md`](../INTEGRATION.md)
- Main benchmarks: [`benchmarks/`](../../benchmarks/)
