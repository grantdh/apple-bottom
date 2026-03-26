# Lessons Learned: apple-bottom QE Integration

## Date: March 24, 2026

---

## Executive Summary

We attempted two optimizations for QE (threaded FFT, GPU BLAS) and encountered systematic issues due to inadequate upfront analysis. This document captures the lessons to prevent repeating mistakes.

---

## Issue 1: Threaded FFTW Failure

### What We Tried
1. OpenMP FFTW (`libfftw3_omp`) - 42× slower due to ABI mismatch with gfortran
2. pthreads FFTW (`libfftw3_threads`) - crashed or slower on small grids

### Root Cause
- Conda's FFTW OpenMP library uses a different OpenMP runtime than gfortran's `libgomp`
- Small FFT grids (36³) have more threading overhead than benefit
- Large grids (72³) showed 4.8× speedup in isolation but QE crashed

### Lesson
**Before attempting library threading changes:**
1. Check ABI compatibility: `nm -D` both libraries, verify runtime symbols match
2. Test threading benefit at actual problem sizes FIRST (not in isolation)
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

### Architecture We Discovered (Too Late)
```
QE Call Chain:
  becmod.f90 → MYZGEMM() 
    → UtilXlib/device_helper.f90 
      → ZGEMM (standard BLAS)

What We Needed:
  MYZGEMM() → ab_zgemm_blas() [BLAS-compatible wrapper]
    → Routing logic (size check, beta check)
      → GPU path: ab_zgemm() [split-complex ABMatrix API]
      → CPU path: cblas_zgemm() [fallback]
```

### Critical Discovery: The Beta Bug

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

### What We Got Wrong

1. **Two library locations** - confusion between:
   - `~/Dev/arm/metal-algos/build/libapplebottom.a` (development)
   - `~/apple-bottom/build/libapplebottom.a` (QE expects this)
   
2. **Symbol naming collision** - apple-bottom already has:
   - `ab_zgemm(ABMatrix...)` - split-complex matrix API
   - We added `ab_zgemm_blas(char, char, int...)` - BLAS-compatible API
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
   # Create minimal apple_bottom_mod.f90 (ONLY the symbols we need)
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

### What's Done
- [x] `blas_wrapper.c` created with `ab_zgemm_blas()` and `ab_dgemm_blas()`
- [x] Wrapper routes to BLAS fallback when `beta != 0`
- [x] `device_helper.f90` patched with `__APPLE_BOTTOM__` branch
- [x] Minimal `apple_bottom_mod.f90` in `UtilXlib/`

### What's Broken
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

- [ ] Remove old `Modules/apple_bottom_mod.f90`
- [ ] Rebuild Modules cleanly
- [ ] Verify `UtilXlib/apple_bottom_mod.f90` compiles first
- [ ] Rebuild QE
- [ ] Validate energy matches baseline
- [ ] If working: benchmark performance
- [ ] If faster: tag v1.2.0
