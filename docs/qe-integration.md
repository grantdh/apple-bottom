# Quantum ESPRESSO Integration Guide

This guide shows how to integrate apple-bottom with Quantum ESPRESSO 7.4.1 to achieve GPU-accelerated ZGEMM operations in the Davidson eigensolver.

**Performance:** 2.7× speedup over single-threaded OpenBLAS, 14% faster than 6-thread OpenBLAS on Si64 benchmark.

---

## Prerequisites

- macOS 14+ (Sonoma) with Apple Silicon (M1/M2/M3/M4)
- Quantum ESPRESSO 7.4.1 source
- apple-bottom library built (`make` in this directory)
- gfortran (from conda or Homebrew)

---

## Quick Start

```bash
# 1. Build apple-bottom
cd ~/Dev/arm/metal-algos  # or wherever you cloned
make
make test

# 2. Set up symlink (QE expects ~/apple-bottom)
ln -s ~/Dev/arm/metal-algos ~/apple-bottom

# 3. Patch QE (ONE file: cegterg.f90)
cd ~/qe-test/q-e-qe-7.4.1/KS_Solvers/Davidson
# Apply patches from section below

# 4. Configure QE
cd ~/qe-test/q-e-qe-7.4.1
# Update make.inc (see configuration section)

# 5. Build QE
make clean
make pw -j8

# 6. Run benchmark
cd ~/qe-test/benchmark
time ~/qe-test/q-e-qe-7.4.1/bin/pw.x < si64.in > si64_gpu.out 2>&1
grep '!' si64_gpu.out  # Should show: -2990.44276157 Ry
```

Expected wall time: ~2:05 (vs 5:43 for single-threaded OpenBLAS).

---

## Integration Architecture

### The EXTERNAL Declaration Approach

apple-bottom uses **EXTERNAL declaration** to replace BLAS calls without modules:

```fortran
! In cegterg.f90, after IMPLICIT NONE:
IMPLICIT NONE
EXTERNAL :: ab_zgemm

! Then replace CALL ZGEMM( with CALL ab_zgemm(
CALL ab_zgemm('N', 'N', m, n, k, alpha, A, lda, B, ldb, beta, C, ldc)
```

**Why this works:**
- Fortran uses sequence association (passes addresses, not descriptors)
- No explicit interface → gfortran doesn't construct array descriptors
- C bridge receives pointers exactly as standard BLAS expects
- QE can pass scalar elements like `hpsi(1,n_start)` without segfaulting

### Call Chain

```
QE cegterg.f90:
  CALL ab_zgemm('N', 'N', m, n, k, ...)
           ↓
fortran_bridge.c: ab_zgemm_()
  - Dereference pointers (char*, int*, double complex*)
  - Calculate FLOPs = M × N × K × 8
  - Check threshold
           ↓
  ├─ < 100M FLOPs → zgemm_() passthrough (OpenBLAS, zero overhead)
  └─ ≥ 100M FLOPs → ab_zgemm_blas() (GPU path)
           ↓
blas_wrapper.c: ab_zgemm_blas()
  - Convert interleaved complex → split-complex (real[], imag[])
  - Upload to GPU (ABMatrix objects)
  - Dispatch Metal kernel (Gauss 3-multiply)
  - Download result
  - Convert split-complex → interleaved complex
           ↓
apple_bottom.m: Metal kernel execution
  - Double-float (DD) arithmetic (~10⁻¹⁵ precision)
  - Register-blocked tile algorithm (BM=BN=32, TM=TN=8)
  - Gauss algorithm: 3 DD multiplies instead of 4
```

**Key insight:** Small calls (< 100M FLOPs) pass directly to OpenBLAS with zero overhead. Only large Davidson subspace updates use GPU.

---

## Step-by-Step Integration

### Step 1: Build apple-bottom

```bash
cd ~/Dev/arm/metal-algos
make clean
make
make test  # Should show 37 tests passing
```

Verify library symbols:
```bash
nm build/libapplebottom.a | grep "ab_[dz]gemm"
# Should show:
#   T _ab_dgemm_       (Fortran callable)
#   T _ab_zgemm_       (Fortran callable)
#   T _ab_dgemm_blas   (C API)
#   T _ab_zgemm_blas   (C API)
```

**Troubleshooting:** If `blas_wrapper.c` symbols are missing, manually compile:
```bash
clang -Wall -O3 -std=c11 -Iinclude -c src/blas_wrapper.c -o build/blas_wrapper.o
ar rcs build/libapplebottom.a build/apple_bottom.o build/blas_wrapper.o build/fortran_bridge.o
```

### Step 2: Set Up Symlink

QE integration expects library at `~/apple-bottom/build/libapplebottom.a`:

```bash
ln -sf ~/Dev/arm/metal-algos ~/apple-bottom
ls -la ~/apple-bottom/build/libapplebottom.a  # Verify exists
```

### Step 3: Patch cegterg.f90

**File:** `~/qe-test/q-e-qe-7.4.1/KS_Solvers/Davidson/cegterg.f90`

**Changes:** 13 lines total (1 declaration + 12 replacements)

#### Change 1: Add EXTERNAL declaration

After `IMPLICIT NONE` (around line 40-41):

```fortran
  IMPLICIT NONE
  EXTERNAL :: ab_zgemm
```

#### Change 2: Replace ZGEMM calls

In the main `cegterg` subroutine (lines 18-701), replace all `CALL ZGEMM(` with `CALL ab_zgemm(`:

**12 replacements total:**

1. Line ~420: `hpsi = spsi * vc` (conjugate transpose)
2. Line ~445: `sc = spsi^H * psi`
3. Line ~458: `hc = hpsi^H * psi`
4. Line ~475: `vc = eigenvectors * ew`
5. Line ~510: `psi = psi * vc` (update eigenvectors)
6. Line ~527: `hpsi = hpsi * vc`
7. Line ~544: `spsi = spsi * vc`
8. Line ~580: `psi_new = hpsi * psi`
9. Line ~595: `hpsi_new = spsi * hc`
10. Line ~615: `overlap = psi^H * psi_new`
11. Line ~635: `psi_new orthogonalization`
12. Line ~670: `final psi update`

**Do NOT modify** the `pcegterg` subroutine (lines ~750+) — it uses different parallelization.

**Example diff:**
```diff
- CALL ZGEMM('N', 'N', kdim, nbase, nbase, ONE, spsi, kdmx, vc, nvecx, ZERO, psi(:,nb1), kdmx)
+ CALL ab_zgemm('N', 'N', kdim, nbase, nbase, ONE, spsi, kdmx, vc, nvecx, ZERO, psi(:,nb1), kdmx)
```

### Step 4: Configure QE

Edit `~/qe-test/q-e-qe-7.4.1/make.inc`:

#### Add preprocessor flag

Find the `DFLAGS` line and add `-D__APPLE_BOTTOM__`:

```makefile
DFLAGS         = -D__GFORTRAN -D__STD_F95 -D__FFTW3 -D__APPLE_BOTTOM__
```

#### Update BLAS_LIBS

Find `BLAS_LIBS` and replace with:

```makefile
BLAS_LIBS      = -L$(HOME)/apple-bottom/build -lapplebottom -framework Accelerate -framework Metal -framework Foundation
```

**Important:** apple-bottom must come BEFORE `-framework Accelerate` so Fortran finds `ab_zgemm_()` before falling back to OpenBLAS `zgemm_()`.

**Complete make.inc example:**
```makefile
F90            = gfortran
CC             = gcc
DFLAGS         = -D__GFORTRAN -D__STD_F95 -D__FFTW3 -D__APPLE_BOTTOM__
BLAS_LIBS      = -L$(HOME)/apple-bottom/build -lapplebottom -framework Accelerate -framework Metal -framework Foundation
LAPACK_LIBS    =
SCALAPACK_LIBS =
FFT_LIBS       = -L/opt/homebrew/lib -lfftw3
```

### Step 5: Build QE

```bash
cd ~/qe-test/q-e-qe-7.4.1

# Clean previous build
make clean

# Rebuild pw.x
make pw -j8

# Verify binary exists
ls -la bin/pw.x
```

**Build time:** ~5-10 minutes on M2 Max.

**Troubleshooting:** If build fails with "undefined reference to `ab_zgemm_`", verify:
1. Symlink: `ls -la ~/apple-bottom/build/libapplebottom.a`
2. Library symbols: `nm ~/apple-bottom/build/libapplebottom.a | grep ab_zgemm_`
3. make.inc has `-lapplebottom` BEFORE `-framework Accelerate`

### Step 6: Validate Integration

Run the integration test (from apple-bottom repo):

```bash
cd ~/Dev/arm/metal-algos
./tests/test_qe_integration.sh
```

This validates:
- ✓ Symlink exists
- ✓ Library has correct symbols
- ✓ No stale references in QE make.depend
- ✓ Source files are clean
- ✓ cegterg.f90 has EXTERNAL declaration
- ✓ cegterg.f90 has 12 ab_zgemm calls
- ✓ make.inc has correct flags
- ✓ pw.x binary exists

### Step 7: Run Benchmark

```bash
cd ~/qe-test/benchmark
rm -rf tmp && mkdir -p tmp
time ~/qe-test/q-e-qe-7.4.1/bin/pw.x < si64.in > si64_gpu.out 2>&1
```

**Expected output:**
```
real    2m5.40s      # Wall time: ~2:05
user    7m18.89s     # CPU time (parallelism from Metal dispatch)
sys     0m21.50s
```

**Verify correctness:**
```bash
grep '!' si64_gpu.out
```

**Expected energy:**
```
!    total energy              =   -2990.44276157 Ry
```

**CRITICAL:** Energy must match `-2990.44276157 Ry` exactly. Any deviation indicates numerical errors.

---

## Performance Results

### Si64 Benchmark (64-atom silicon, SCF calculation)

**Hardware:** M2 Max (38-core GPU, 64 GB RAM)

| Configuration | Wall Time | vs 1-thread | Energy (Ry) |
|--------------|-----------|-------------|-------------|
| OpenBLAS (6 threads) | 2:22 | 2.4× | -2990.44276157 |
| OpenBLAS (1 thread) | 5:43 | 1.0× | -2990.44276157 |
| **apple-bottom GPU** | **2:05** | **2.7×** | **-2990.44276157** ✓ |

**Breakdown by routine:**

| Routine | OpenBLAS 6T | OpenBLAS 1T | GPU | GPU Speedup |
|---------|-------------|-------------|-----|-------------|
| **Total** | 2:22 | 5:43 | **2:05** | **2.7×** |
| c_bands | 109s | 251s | 112s | 2.2× |
| cegterg | 107s | 248s | 110s | 2.3× |
| h_psi | 75.6s | 162s | 73.2s | 2.2× |
| calbec | 27.2s | 59.8s | 21.9s | 2.7× |

**Key metrics:**
- GPU is 14% faster than 6-thread OpenBLAS
- GPU matches single-threaded performance but with 2.7× less wall time
- Correct energy validation proves numerical accuracy

**Why GPU doesn't beat 6-thread by more:**
- Per-call overhead: upload (44 MB) + download (44 MB) per ZGEMM
- Small calls still route to OpenBLAS (< 100M FLOPs threshold)
- Davidson subspace dimension grows during iteration (nvec: 4 → 150)

**Future optimization:** Native API to keep matrices GPU-resident across iterations (expected 40%+ improvement).

---

## Troubleshooting

### Build Failures

#### "undefined reference to `ab_zgemm_`"

**Cause:** Library not linked or missing symbols.

**Fix:**
```bash
# Verify library exists
ls -la ~/apple-bottom/build/libapplebottom.a

# Check symbols
nm ~/apple-bottom/build/libapplebottom.a | grep ab_zgemm_

# If missing, rebuild library
cd ~/Dev/arm/metal-algos
make clean && make

# Verify make.inc
grep "BLAS_LIBS" ~/qe-test/q-e-qe-7.4.1/make.inc
# Should show: -L$(HOME)/apple-bottom/build -lapplebottom ...
```

#### "multiple definition of `ab_print_stats`"

**Cause:** Duplicate symbols in library (blas_wrapper.o compiled twice).

**Fix:**
```bash
cd ~/Dev/arm/metal-algos
make clean
rm -f build/*.o build/*.a
make
```

#### "stale reference to `apple_bottom_mod`"

**Cause:** Old module files from previous integration attempts.

**Fix:**
```bash
cd ~/qe-test/q-e-qe-7.4.1
grep apple_bottom_mod Modules/make.depend
# If found, clean:
rm -f Modules/apple_bottom_mod.* UtilXlib/apple_bottom_mod.*
cd Modules && make clean
cd ../UtilXlib && make clean
cd .. && make clean && make pw
```

### Runtime Errors

#### Wrong energy (e.g., -2989.xx instead of -2990.44)

**Cause:** Numerical errors from incorrect Fortran ABI or beta handling.

**Diagnosis:**
```bash
# Check that cegterg.f90 uses EXTERNAL, not a module
grep "EXTERNAL :: ab_zgemm" ~/qe-test/q-e-qe-7.4.1/KS_Solvers/Davidson/cegterg.f90

# Verify no other files were modified
cd ~/qe-test/q-e-qe-7.4.1
git status  # Or diff against clean QE source
```

**Fix:** Ensure ONLY cegterg.f90 is modified. Clean these files if they have apple-bottom references:
- `Modules/becmod.f90` (must be clean)
- `UtilXlib/device_helper.f90` (must be clean)

#### "S matrix not positive definite" crash

**Cause:** Beta parameter handling error (GPU ignoring beta=1 accumulation).

**Current status:** Fixed in v1.2+ (blas_wrapper.c routes beta≠0 to OpenBLAS fallback until GPU kernel supports it).

**Verify fix:**
```bash
grep "beta" ~/Dev/arm/metal-algos/src/blas_wrapper.c
# Should see: if (beta != 0.0 && beta != 1.0) fallback check
```

#### Slow performance (no speedup)

**Cause:** Calls not reaching GPU (below threshold or incorrect routing).

**Diagnosis:**
```bash
# Run with stats
export AB_PRINT_STATS=1
~/qe-test/q-e-qe-7.4.1/bin/pw.x < si64.in > si64.out 2>&1

# Check for GPU calls in output
grep "ab_zgemm" si64.out
```

**Expected:** Should see calls with kdim=18277, nvec=4-150 routed to GPU.

### Clean Rebuild

If integration is broken, start fresh:

```bash
# 1. Clean QE
cd ~/qe-test/q-e-qe-7.4.1
git checkout KS_Solvers/Davidson/cegterg.f90  # Revert patches
make clean

# 2. Rebuild apple-bottom
cd ~/Dev/arm/metal-algos
make clean && make && make test

# 3. Re-apply patches
# (Follow Step 3 above)

# 4. Rebuild QE
cd ~/qe-test/q-e-qe-7.4.1
make pw -j8

# 5. Validate
cd ~/Dev/arm/metal-algos
./tests/test_qe_integration.sh
```

---

## Files Changed

**In apple-bottom repo:**
- NONE (integration uses library as-is)

**In QE source (`~/qe-test/q-e-qe-7.4.1/`):**
- `KS_Solvers/Davidson/cegterg.f90` — 13 lines (1 EXTERNAL + 12 replacements)
- `make.inc` — 2 lines (DFLAGS + BLAS_LIBS)

**Total QE changes:** 15 lines across 2 files.

---

## Extracting Patches

To create distributable patches:

```bash
cd ~/qe-test/q-e-qe-7.4.1

# Generate patch for cegterg.f90
git diff KS_Solvers/Davidson/cegterg.f90 > cegterg-apple-bottom.patch

# Or manual diff
diff -u cegterg.f90.orig cegterg.f90 > cegterg-apple-bottom.patch
```

---

## Next Steps

### Current Implementation (Per-Call API)
- ✓ Working integration with correct energy
- ✓ 2.7× speedup over single-threaded OpenBLAS
- ✓ Minimal code changes (13 lines)
- ⚠ Per-call overhead: 44 MB upload + 44 MB download per ZGEMM

### Future: Native API (GPU-Resident Matrices)

Keep matrices on GPU across Davidson iterations to eliminate upload/download overhead:

```fortran
! Instead of per-call conversion:
do iter = 1, max_iter
    CALL ab_zgemm(...)  ! Upload + compute + download
end do

! Do this:
CALL ab_zmatrix_upload(hpsi_gpu, hpsi_host)  ! Once
do iter = 1, max_iter
    CALL ab_zgemm_gpu(hpsi_gpu, spsi_gpu, vc_gpu, ...)  ! On GPU
end do
CALL ab_zmatrix_download(hpsi_gpu, hpsi_host)  ! Once
```

**Expected improvement:** 40%+ reduction in wall time (eliminate ~50s of PCIe traffic).

See `PARALLEL_WORKFLOW.md` for native-api branch development setup.

---

## Support

- Integration test: `./tests/test_qe_integration.sh`
- Lessons learned: [`LESSONS_LEARNED.md`](../LESSONS_LEARNED.md)
- General integration: [`docs/INTEGRATION.md`](INTEGRATION.md)
- Issues: https://github.com/grantdh/apple-bottom/issues

---

## License

Integration guide: MIT License (same as apple-bottom)
Quantum ESPRESSO: GPL v2+
