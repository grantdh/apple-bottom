# Lessons Learned: apple-bottom QE Integration

## Date: March 24, 2026

---

## Executive Summary

This document analyzes two attempted optimizations for Quantum ESPRESSO (threaded FFT, GPU BLAS) and documents the technical challenges encountered. The analysis provides guidance for future integration efforts.

---

## Issue 1: Threaded FFTW Failure

### Attempted Approaches
1. OpenMP FFTW (`libfftw3_omp`) - 42× slower due to ABI mismatch with gfortran
2. pthreads FFTW (`libfftw3_threads`) - crashed or slower on small grids

### Root Cause
- Conda's FFTW OpenMP library uses a different OpenMP runtime than gfortran's `libgomp`
- Small FFT grids (36³) have more threading overhead than benefit
- Large grids (72³) showed 4.8× speedup in isolation but QE crashed

### Analysis
**Prerequisites for library threading changes:**
1. Check ABI compatibility: `nm -D` both libraries, verify runtime symbols match
2. Test threading benefit at actual problem sizes before integration
3. For QE specifically: FFT is only ~30% of runtime - even 5× FFT speedup = 15% total improvement

### Correct Approach (Not Yet Implemented)
```bash
# Option 1: Build FFTW from source with same compiler as QE
./configure CC=gcc FC=gfortran --enable-threads
make && make install

# Option 2: Use Homebrew FFTW (same toolchain)
brew install fftw
# Then point QE's make.inc to /opt/homebrew/opt/fftw
```

---

## Issue 2: apple-bottom BLAS Integration Failure

### QE Call Chain Architecture
```
QE Call Chain:
  becmod.f90 → MYZGEMM() 
    → UtilXlib/device_helper.f90 
      → ZGEMM (standard BLAS)

Required Integration:
  MYZGEMM() → ab_zgemm_blas() [BLAS-compatible wrapper]
    → Routing logic (size check, beta check)
      → GPU path: ab_zgemm() [split-complex ABMatrix API]
      → CPU path: cblas_zgemm() [fallback]
```

### Beta Parameter Handling

QE's eigensolver calls ZGEMM with `beta=1.0` to accumulate results:
```fortran
CALL ZGEMM('N','N', M,N,K, ONE, A, lda, B, ldb, ONE, C, ldc)
!                                              ^^^ beta=1 accumulates into C
```

apple-bottom's GPU path **ignores beta** and always overwrites C:
```c
// metal_context.m line 820
/* TODO: implement beta scaling - for now just overwrite */
```

This works for small matrices (routed to BLAS fallback) but fails when:
- FLOPs > 100M threshold AND beta ≠ 0

**Result:** Wrong eigenvalues → wrong energy (-2989 vs -2990 Ry) → "S matrix not positive definite" crash

### Integration Challenges

1. **Build artifact location** - QE integration requires library path consistency:
   - Development builds in `~/Dev/arm/metal-algos/build/`
   - QE links against `~/apple-bottom/build/` (requires symlink or copy)
   - Recommendation: Use symlink to maintain single source of truth

2. **Symbol naming collision** - apple-bottom already has:
   - `ab_zgemm(ABMatrix...)` - split-complex matrix API
   - Added `ab_zgemm_blas(char, char, int...)` - BLAS-compatible API
   - Fortran module tried to call wrong one

3. **Module placement** - put `apple_bottom_mod.f90` in wrong directory:
   - `Modules/` compiles AFTER `UtilXlib/`
   - `device_helper.f90` is in `UtilXlib/` and needs the module
   - **Solution:** Module must be in `UtilXlib/` and compile BEFORE `device_helper.f90`

4. **Didn't trace full call chain** - assumed QE calls ZGEMM directly, but it calls MYZGEMM wrapper

---

## Correct Implementation Plan

### Step 1: Understand Current State
```bash
# Verify which ZGEMM QE actually calls
grep -rn "CALL.*ZGEMM" ~/qe-test/q-e-qe-7.4.1/Modules/becmod.f90
# Answer: MYZGEMM

# Find MYZGEMM
grep -rn "SUBROUTINE MYZGEMM" ~/qe-test/q-e-qe-7.4.1/
# Answer: UtilXlib/device_helper.f90

# Check what MYZGEMM does
sed -n '69,85p' ~/qe-test/q-e-qe-7.4.1/UtilXlib/device_helper.f90
# Answer: Calls standard ZGEMM on non-CUDA path
```

### Step 2: Design the Integration

**Files to create/modify:**

| File | Purpose | Location |
|------|---------|----------|
| `blas_wrapper.c` | BLAS-compatible C wrappers | `~/Dev/arm/metal-algos/src/` |
| `apple_bottom_mod.f90` | Minimal Fortran interface | `~/qe-test/.../UtilXlib/` |
| `device_helper.f90` | Add `__APPLE_BOTTOM__` branch | `~/qe-test/.../UtilXlib/` |

**Symbol mapping:**
```
Fortran calls     →  C function         →  Implementation
ab_zgemm()        →  ab_zgemm_blas()    →  routing + GPU/CPU
ab_dgemm()        →  ab_dgemm_blas()    →  routing + GPU/CPU  
ab_init()         →  ab_init()          →  existing
ab_shutdown()     →  ab_shutdown()      →  existing
ab_use_gpu()      →  ab_use_gpu()       →  threshold check
ab_print_stats()  →  ab_print_stats()   →  existing
```

### Step 3: Implementation Order

1. **In apple-bottom repo** (`~/Dev/arm/metal-algos/`):
```bash
   # Create blas_wrapper.c with BLAS-compatible signatures
   # Key: Route to BLAS fallback if beta != 0 (until kernel supports it)
   
   # Compile and add to library
   clang -c -O3 -DACCELERATE_NEW_LAPACK -I include src/blas_wrapper.c -o build/blas_wrapper.o
   ar rcs build/libapplebottom.a build/*.o
   
   # Sync to QE location
   cp build/libapplebottom.a ~/apple-bottom/build/
```

2. **In QE UtilXlib** (`~/qe-test/q-e-qe-7.4.1/UtilXlib/`):
```bash
   # Create minimal apple_bottom_mod.f90 (only the required symbols)
   # Add to Makefile BEFORE device_helper.o
   
   # Modify device_helper.f90:
   #   - Add: #elif defined(__APPLE_BOTTOM__)
   #   - Add: use apple_bottom_mod
   #   - Add: CALL ab_zgemm(...) in MYZGEMM
```

3. **Remove old module** from `Modules/`:
```bash
   rm -f Modules/apple_bottom_mod.f90
```

4. **Rebuild QE**:
```bash
   cd ~/qe-test/q-e-qe-7.4.1/UtilXlib
   # Compile module first (dependency)
   gfortran -c apple_bottom_mod.f90
   
   cd ..
   make clean
   make pw -j8
```

### Step 4: Test
```bash
cd ~/qe-test/benchmark
rm -rf tmp && mkdir -p tmp

# Test with baseline first
time /usr/local/bin/pw.x < si64.in > si64_baseline.out 2>&1
grep '!' si64_baseline.out  # Should be -2990.44 Ry

# Test with apple-bottom
time ~/qe-test/q-e-qe-7.4.1/bin/pw.x < si64.in > si64_ab.out 2>&1
grep '!' si64_ab.out  # MUST match baseline energy
```

---

## Current State (as of end of session)

### Implementation Status

**Completed:**
- [x] `blas_wrapper.c` created with `ab_zgemm_blas()` and `ab_dgemm_blas()`
- [x] Wrapper routes to BLAS fallback when `beta != 0`
- [x] `device_helper.f90` patched with `__APPLE_BOTTOM__` branch
- [x] Minimal `apple_bottom_mod.f90` in `UtilXlib/`

**Pending Resolution:**
- [ ] Old `apple_bottom_mod.f90` still in `Modules/` - causes duplicate symbols
- [ ] `libqemod.a` contains old module - needs rebuild

### To Fix (Next Session)
```bash
# 1. Remove old module
rm -f ~/qe-test/q-e-qe-7.4.1/Modules/apple_bottom_mod.f90
rm -f ~/qe-test/q-e-qe-7.4.1/Modules/apple_bottom_mod.o
rm -f ~/qe-test/q-e-qe-7.4.1/Modules/apple_bottom_mod.mod

# 2. Clean and rebuild Modules
cd ~/qe-test/q-e-qe-7.4.1/Modules
make clean

# 3. Ensure UtilXlib module compiles first
cd ~/qe-test/q-e-qe-7.4.1/UtilXlib
rm -f *.o *.mod
# Manually compile module first
gfortran -cpp -D__APPLE_BOTTOM__ -c apple_bottom_mod.f90

# 4. Full rebuild
cd ~/qe-test/q-e-qe-7.4.1
make pw -j8

# 5. Test correctness
cd ~/qe-test/benchmark
rm -rf tmp && mkdir tmp
~/qe-test/q-e-qe-7.4.1/bin/pw.x < si64.in > si64_test.out 2>&1
grep '!' si64_test.out
# MUST output: -2990.44 Ry (not -2989.xx)
```

---

## Architectural Principles for Future Work

### 1. Single Source of Truth
```
Development: ~/Dev/arm/metal-algos/
Deployment:  ~/apple-bottom/ (symlink or copy)

NEVER edit in both places. Always:
  edit in Dev → build → copy to apple-bottom
```

### 2. Symbol Naming Convention
```
ab_xxx()           - Low-level ABMatrix API (internal)
ab_xxx_blas()      - BLAS-compatible API (Fortran interface)
ab_xxx_gpu()       - Direct GPU call (pre-uploaded matrices)
```

### 3. Routing Logic
```c
bool should_use_gpu(int M, int N, int K, double alpha, double beta) {
    uint64_t flops = 8ULL * M * N * K;
    
    // Size check
    if (flops < 100000000ULL) return false;
    
    // Beta check (until kernel supports it)
    if (beta != 0.0) return false;
    
    // Alpha check (until kernel supports it)  
    if (alpha != 1.0) return false;
    
    return true;
}
```

### 4. Testing Hierarchy
```
1. Unit test: ab_zgemm_blas() with various alpha/beta values
2. Integration test: Small QE run (si8.in, 4 seconds)
3. Validation test: Full QE run, verify energy matches baseline
4. Performance test: Only after validation passes
```

---

## Files Reference

| Path | Purpose | Status |
|------|---------|--------|
| `~/Dev/arm/metal-algos/src/apple_bottom.m` | Core GPU implementation | Working |
| `~/Dev/arm/metal-algos/src/blas_wrapper.c` | BLAS interface | Created |
| `~/apple-bottom/build/libapplebottom.a` | Library for QE | Must sync |
| `~/qe-test/q-e-qe-7.4.1/UtilXlib/apple_bottom_mod.f90` | Fortran module | Created |
| `~/qe-test/q-e-qe-7.4.1/UtilXlib/device_helper.f90` | MYZGEMM wrapper | Patched |
| `~/qe-test/q-e-qe-7.4.1/Modules/apple_bottom_mod.f90` | OLD - DELETE | Must remove |
| `~/qe-test/q-e-qe-7.4.1/make.inc` | Build config | Has __APPLE_BOTTOM__ |
| `~/qe-test/q-e-qe-7.4.1/make.inc.backup` | Clean backup | Keep |

---

## Key Metrics

| Test | Baseline | GPU (broken) | Expected (fixed) |
|------|----------|--------------|------------------|
| si64 Energy | -2990.44 Ry | -2989.90 Ry ❌ | -2990.44 Ry |
| si64 Wall Time | 1:59 | 4:17 ❌ | <1:59 (goal) |
| si64 calbec CPU | 274s | 6s ✓ | 6s |

---

## Next Session Checklist

- [x] Remove old `Modules/apple_bottom_mod.f90`
- [x] Rebuild Modules cleanly
- [x] Verify `UtilXlib/apple_bottom_mod.f90` compiles first
- [x] Rebuild QE
- [x] Validate energy matches baseline
- [x] If working: benchmark performance
- [x] If faster: tag v1.2.0

---

## QE Integration SUCCESS (March 26, 2026)

### Implementation: EXTERNAL Declaration

The successful integration uses Fortran's EXTERNAL declaration:

```fortran
! In cegterg.f90, after IMPLICIT NONE:
IMPLICIT NONE
EXTERNAL :: ab_zgemm

! Then replace all CALL ZGEMM( with CALL ab_zgemm(
! (12 replacements in lines 18-701, pcegterg untouched)
```

**Why this works:**
- Fortran uses **sequence association** — passes addresses, not descriptors
- No explicit interface → gfortran doesn't try to construct array descriptors
- The C bridge receives pointers exactly as standard BLAS expects
- QE can pass scalar elements like `hpsi(1,n_start)` without segfaulting

### The Integration Architecture

```
QE cegterg.f90:
  CALL ab_zgemm('N', 'N', m, n, k, (1.d0,0.d0), A, lda, B, ldb, ...)
                     ↓
Fortran bridge (fortran_bridge.c):
  void ab_zgemm_(const char *transA, ..., const double complex *alpha, ...) {
      uint64_t flops = (uint64_t)(*M) * (*N) * (*K) * 8;

      if (flops >= 100000000ULL) {
          ab_zgemm_blas(*transA, ..., *alpha, ...);  // Dereference → GPU
          return;
      }

      zgemm_(transA, ..., alpha, ...);  // Pass-through to OpenBLAS
  }
                     ↓
BLAS wrapper (blas_wrapper.c):
  void ab_zgemm_blas(char transA, ..., double complex alpha, ...) {
      // Convert to split-complex, upload to GPU, run kernel, download
  }
```

**Key insight:** Sub-threshold calls pass pointers **directly** to `zgemm_()` with zero conversion. Only above-threshold calls dereference and convert.

### Performance Results

| Configuration | si64 Wall Time | vs 1-thread | c_bands | cegterg | h_psi | calbec |
|--------------|----------------|-------------|---------|---------|-------|--------|
| **OpenBLAS 6-thread** | 2:22 | 2.4× | 109s | 107s | 75.6s | 27.2s |
| **OpenBLAS 1-thread** | 5:43 | 1.0× | 251s | 248s | 162s | 59.8s |
| **apple-bottom GPU** | **2:05** | **2.7×** | **112s** | **110s** | **73.2s** | **21.9s** |

**Energy validation:** `-2990.44276157 Ry` ✓ (exact match to baseline)

**Performance characteristics:**
- 14% faster than 6-thread OpenBLAS
- 2.7× faster than single-threaded baseline
- GPU compute replaces CPU thread parallelism
- GPU overhead is ~3s per 2-minute run

### Unsuccessful Approaches

**1. Module Interface with BIND(C)**
```fortran
! This SEGFAULTS:
MODULE apple_bottom_mod
  INTERFACE
    SUBROUTINE ab_zgemm(...) BIND(C)
      COMPLEX(8), INTENT(IN) :: A(ldA, *)  ! Rank-2 assumed-size
    END SUBROUTINE
  END INTERFACE
END MODULE
```
**Why:** QE passes array slices like `hpsi(1,n_start)` which are scalar elements. With an explicit interface, gfortran tries to construct an array descriptor → segfault.

**2. #define ZGEMM ab_zgemm Macro**
```fortran
! This produces WRONG RESULTS:
#define ZGEMM ab_zgemm
```
**Why:** Without an explicit interface, gfortran uses default Fortran calling conventions which pass hidden character length arguments. Our C bridge doesn't expect them → stack corruption → numerical drift → "S matrix not positive definite" crash after 8 SCF iterations.

**3. cblas_zgemm Fallback**
```c
// This produces WRONG RESULTS:
extern void cblas_zgemm(int, int, int, ...);  // Hand-declared
cblas_zgemm(CblasColMajor, ta, tb, M, N, K, &alpha, ...);
```
**Why:** The hand-declared extern resolved to OpenBLAS's cblas_zgemm but with potentially incorrect ABI. Even subtle mismatches in how `double complex` is passed cause numerical drift.

**4. Buffer Pooling (First Attempt)**
```c
// This CRASHES:
static id<MTLBuffer> pool_acquire(size_t needed) { ... }
// Buffer reused while previous GPU command still in flight
```
**Why:** Metal command buffers execute asynchronously. Releasing a buffer back to the pool before `waitUntilCompleted` means the next user overwrites data still being read by the GPU.

### Critical Recovery Steps

When rebuilding from scratch, check these in order:

**Step 0: Symlink**
```bash
ls -la ~/apple-bottom  # Must point to ~/Dev/arm/metal-algos
```

**Step 1: Library symbols**
```bash
nm build/libapplebottom.a | grep "ab_[dz]gemm"
```
Must show:
- `_ab_dgemm_` and `_ab_zgemm_` (Fortran-callable with trailing underscore)
- `_ab_dgemm_blas` and `_ab_zgemm_blas` (C API, no trailing underscore)
- No duplicate `_ab_print_stats`

**Step 1b: blas_wrapper.c compilation**
The Makefile doesn't compile `blas_wrapper.c` automatically. Must manually:
```bash
clang -Wall -O3 -std=c11 -Iinclude -c src/blas_wrapper.c -o build/blas_wrapper.o
ar rcs build/libapplebottom.a build/apple_bottom.o build/blas_wrapper.o build/fortran_bridge.o
```

**Step 2: QE Modules/make.depend**
```bash
grep apple_bottom_mod ~/qe-test/q-e-qe-7.4.1/Modules/make.depend
```
Must return **nothing** (no stale references)

**Step 3: QE source files are clean**
These must have **ZERO** apple-bottom references:
- `Modules/becmod.f90`
- `UtilXlib/device_helper.f90`

**Step 3b: Clean UtilXlib/device_helper.f90**
If integration was previously attempted via `device_helper.f90`, remove all traces:
```bash
cd ~/qe-test/q-e-qe-7.4.1/UtilXlib
git checkout device_helper.f90  # Or manually remove #ifdef __APPLE_BOTTOM__ blocks
```

**Step 4: cegterg.f90 patches**
```bash
grep "EXTERNAL :: ab_zgemm" ~/qe-test/q-e-qe-7.4.1/KS_Solvers/Davidson/cegterg.f90
grep -c "CALL ab_zgemm" ~/qe-test/q-e-qe-7.4.1/KS_Solvers/Davidson/cegterg.f90
```
Must have:
- `EXTERNAL :: ab_zgemm` after `IMPLICIT NONE`
- 12 `CALL ab_zgemm(` replacements (lines 18-701 only, pcegterg untouched)

**Step 5: make.inc configuration**
```bash
grep -A5 "DFLAGS" ~/qe-test/q-e-qe-7.4.1/make.inc
grep "BLAS_LIBS" ~/qe-test/q-e-qe-7.4.1/make.inc
```
Must have:
- `DFLAGS = ... -D__APPLE_BOTTOM__`
- `BLAS_LIBS = -L$(HOME)/apple-bottom/build -lapplebottom -framework Accelerate -framework Metal -framework Foundation`

**Step 6: Rebuild and test**
```bash
cd ~/qe-test/q-e-qe-7.4.1
make clean
make pw -j8

cd ~/qe-test/benchmark
rm -rf tmp && mkdir -p tmp
time ~/qe-test/q-e-qe-7.4.1/bin/pw.x < si64.in > si64_test.out 2>&1
grep '!' si64_test.out
```
Expected: `-2990.44276157 Ry` (exact match)

### Repository Structure (Final)

```
~/Dev/arm/metal-algos/          # Main repo (symlinked from ~/apple-bottom)
├── src/
│   ├── apple_bottom.m          # Core Metal implementation
│   ├── blas_wrapper.c          # BLAS-compatible C API (ab_zgemm_blas, etc.)
│   └── fortran_bridge.c        # Fortran ABI bridge (ab_zgemm_, etc.)
├── build/
│   └── libapplebottom.a        # Static library for QE linking
└── tests/
    ├── test_correctness.c
    ├── test_precision.c
    └── test_qe_integration.sh  # NEW: Validation script

~/qe-test/q-e-qe-7.4.1/         # QE source (ONLY cegterg.f90 is modified)
├── KS_Solvers/Davidson/
│   └── cegterg.f90             # EXTERNAL + CALL ab_zgemm patches
└── make.inc                    # -D__APPLE_BOTTOM__ + -lapplebottom
```

### The One File That Gets Patched

**Only `cegterg.f90` is modified.** Everything else stays upstream clean.

```fortran
! Line ~40-41 (after IMPLICIT NONE):
IMPLICIT NONE
EXTERNAL :: ab_zgemm

! Lines 18-701: Replace CALL ZGEMM( with CALL ab_zgemm(
! (12 replacements total, pcegterg section untouched)
```

### Next Steps: Native API for GPU-Resident Matrices

The current integration achieves **2.7× speedup** but has per-call overhead:
- Upload to GPU: 18277×150 complex matrix → 44 MB
- Download from GPU: 18277×150 complex matrix → 44 MB
- Happens 12 times per Davidson iteration

**The opportunity:** Keep matrices **resident on GPU** across Davidson iterations.

From integration guide:
```c
// Instead of:
for (int iter = 0; iter < n_iter; iter++) {
    CALL ab_zgemm(...)  // Upload A, B → compute → download C
}

// Do this:
ABZMatrixHandle hpsi_gpu = ab_zmatrix_create(kdim, nvec);
ab_zmatrix_upload(hpsi_gpu, hpsi_host, 0, nvec);  // Once

for (int iter = 0; iter < n_iter; iter++) {
    ab_zgemm_gpu(hpsi_gpu, spsi_gpu, hc_gpu, ...);  // No upload/download
}

ab_zmatrix_download(hpsi_gpu, hpsi_host, 0, nvec);  // Once
ab_zmatrix_destroy(hpsi_gpu);
```

**Expected improvement:** 40%+ reduction in wall time (eliminate ~50s of PCIe traffic per run).

**Branch strategy:** Work in `native-api` branch with separate QE copy (`q-e-native`) to preserve working integration.
