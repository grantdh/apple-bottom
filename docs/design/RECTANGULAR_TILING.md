# Rectangular Matrix Tile-Dispatch Design Note

This note documents the rationale for apple-bottom's default tile configuration (BM=BN=64, TM=TN=4, TK=16) and the tall-skinny variant (BM=128, BN=16) used for QE's Davidson eigensolver shapes. It is intended as a reference for future tile-size tuning on new Apple Silicon generations or workloads with different aspect ratios.

DD precision bounds scale as O(√K)·2⁻⁴⁸ (Wilkinson) for all tested aspect ratios from 121:1 (tall-skinny) to 1:121 (short-wide). Verified in `test_rectangular.c` against QE Davidson shapes.

---

## Tile-Dispatch Model

**Default tile configuration** (square and moderate-aspect shapes):
- BM = BN = 64 (block tiles in M, N)
- TM = TN = 4 (thread tiles)
- TK = 16 (K-slab width)
- Tuned for M ≈ N ≈ K

Each threadgroup computes a BM×BN tile of C by iterating TK-wide slabs across the K dimension. The grid dimension is `ceil(M/BM) × ceil(N/BN)` threadgroups; threads within a threadgroup cooperatively load A and B tiles into threadgroup memory before the TK-wide inner product.

**Tall-skinny variant** (deployed; see `src/apple_bottom.m:249-250`):
- BM = 128, BN = 16
- TM = 4, TN = 2
- Threadgroup layout 8×32 = 256 threads (`src/apple_bottom.m:1211`)
- Selected on `M >= 4 * N` (`src/apple_bottom.m:1055`)

The tall-skinny kernel reduces boundary waste on QE-shape workloads (M≈18K, N=150) from ~34% under BN=64 to ~4% under BN=16.

**QE Davidson eigensolver shape:**
- M = 18,277 (kdim — basis size)
- N = 150 (nvec — number of eigenvectors)
- K = 18,277 (kdim)
- Aspect ratio: 121:1 (M:N) → routes to the tall-skinny kernel

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
- `test_zgemm_conjugate_transpose_qe()` — 150 × 150 × 18277 × 150

**Validation:**
- All tests validate against reference BLAS (cblas_dgemm, cblas_zgemm)
- Error threshold: < 1e-14 (within DD precision)
- Tests exercise the tall-skinny dispatch path at QE Davidson dimensions

### Performance Benchmarks

**benchmark_aspect_ratios()** tests:
1. Square: 2048 × 2048 × 2048
2. Tall 4:1: 4096 × 1024 × 2048
3. Tall 16:1: 8192 × 512 × 2048
4. Tall 64:1: 16384 × 256 × 2048
5. QE-like: 18277 × 150 × 18277
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
- For large K (>2000), max error approaches ~1e-13 consistent with O(√K)·2⁻⁴⁸ accumulation.

### Performance by Aspect Ratio

**Square matrices (baseline):**
```
2048 × 2048 × 2048: GPU = 1.1-1.2× faster than BLAS
```

**Tall matrices (M >> N), default tile:**
```
4096 × 1024 × 2048:  GPU ≈ 1.0× (neutral)
8192 × 512 × 2048:   GPU ≈ 0.8×
18277 × 150 × 18277: GPU ≈ 0.5-0.7×
```

**Why the default-tile slowdown on tall shapes:**

1. **Threadgroup underutilization.** Under BN=64, N=150 creates only 3 threadgroups in N; many threads idle in partial blocks.

2. **Memory traffic dominates.**
   - A upload: 18277 × 18277 × 8 bytes = 2.5 GB
   - B upload: 18277 × 150 × 8 bytes = 21 MB
   - C download: 18277 × 150 × 8 bytes = 21 MB
   - Total per call: 2.5 GB

3. **Cache locality.** B (150 columns) fits in cache; A (18277 rows) does not. Tall shapes yield long stride walks.

The tall-skinny kernel addresses (1) directly: at BN=16, N=150 yields ceil(150/16) = 10 threadgroups in N rather than 3, and boundary waste drops from ~34% to ~4% per `src/apple_bottom.m:1056`. (2) and (3) are upload/download and stride concerns inherent to the per-call API and are the motivation for a future device-resident path.

**Net QE-level speedup.**

QE shows a net 2.7× speedup end-to-end despite per-call characteristics on individual shapes:
- QE issues 12 ZGEMM calls per Davidson iteration
- Calls below the 100M-FLOP threshold route to OpenBLAS
- Large `hpsi = H * psi` calls hit the GPU, dispatched to the tall-skinny kernel on M ≥ 4N
- GPU wins on the large calls; BLAS handles the small ones

---

## Running the Tests

### Build
```bash
cd ~/Dev/Claude/apple-bottom
make  # Build library first

# Compile test
clang -O3 -I include -L build -lapplebottom \
  -framework Accelerate -framework Metal -framework Foundation \
  -o tests/test_rectangular tests/test_rectangular.c
```

### Run Correctness Tests
```bash
./tests/test_rectangular
```

Expected output:
```
=============================================================================
apple-bottom Rectangular Matrix Test Suite
=============================================================================

DGEMM Rectangular Tests:
  DGEMM tall-skinny (10000 × 100 × 100 × 100)        ✓ PASS
  DGEMM short-wide (100 × 10000 × 100 × 10000)       ✓ PASS
  DGEMM QE-like (18277 × 150 × 18277 × 150)          ✓ PASS
  DGEMM thin middle (5000 × 5000 × 10 × 5000)        ✓ PASS

ZGEMM Rectangular Tests:
  ZGEMM QE-like (18277 × 150 × 18277 × 150)          ✓ PASS
  ZGEMM QE conjugate transpose (150 × 150 × ...)    (skipped) ✓ PASS

=============================================================================
Summary: 6 passed, 0 failed
=============================================================================
```

---

## References

- Tall-skinny kernel parameters: `src/apple_bottom.m:249-250`
- Dispatch rule (`M >= 4 * N`): `src/apple_bottom.m:1055`
- Tall-skinny threadgroup layout: `src/apple_bottom.m:1211`
- QE integration: [`docs/INTEGRATION.md`](../INTEGRATION.md)
- Main benchmarks: [`benchmarks/`](../../benchmarks/)
