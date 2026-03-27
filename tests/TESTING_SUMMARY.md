# Testing Summary: Rectangular Matrices

## Test Results (March 27, 2026)

### Status: ⚠ **Rectangular matrices have correctness issues**

### What We Tested

1. **test_rectangular.c** - Comprehensive rectangular matrix suite
   - Small test (100×50): ✓ **PASSED** (0.00e+00 error)
   - Large tests: ✗ **FAILED** (error too large)

2. **Performance benchmarks** - Ran successfully

### Results

#### Correctness Tests
```
DGEMM tall-skinny (10000 × 100):   ✗ FAIL
DGEMM short-wide (100 × 10000):    ✗ FAIL
DGEMM QE-like (18277 × 150):       ✗ FAIL
ZGEMM QE-like (18277 × 150):       ✗ FAIL

Debug test (100 × 50):             ✓ PASS (perfect accuracy)
```

####Performance (GPU vs BLAS)
```
Square (2048×2048):        44.6ms vs 35.8ms  = 0.80× (slower)
Tall 4:1 (4096×1024):      29.7ms vs 28.6ms  = 0.96×
Tall 16:1 (8192×512):      30.6ms vs 28.9ms  = 0.94×
QE-like (18277×150):      257.3ms vs 202.7ms = 0.79× (slower)
Wide 1:16 (512×8192):      30.1ms vs 34.7ms  = 1.15× (faster!)
```

### Key Findings

1. **Small rectangles work** (100×50 tested)
2. **Large rectangles fail** (10000×100+)
3. **Performance is slower** for most rectangular cases
4. **Wide matrices** (1:16) show promise (1.15× faster)

### Hypothesis

The issue is likely one of:

1. **Threshold routing** - Large calls should go to GPU but might have issues
2. **Memory layout** - Row-major (BLAS) vs column-major (Metal) mismatch at scale
3. **Precision accumulation** - DD arithmetic errors accumulate differently for rectangles
4. **Matrix dimensions** - Leading dimension (lda, ldb, ldc) handling

### Why QE Integration Still Works

The QE integration (cegterg.f90) shows **correct energy** despite these unit tests failing. Possible reasons:

1. **Fortran bridge** handles dimensions differently than C API
2. **QE matrix layout** might match Metal's column-major
3. **Accumulation effects** cancel out in full SCF calculation
4. **Threshold routing** sends most calls to OpenBLAS anyway

### Immediate Actions Needed

1. **Investigate dimension handling**
   - Check if ab_matrix_upload expects row-major or column-major
   - Verify lda/ldb/ldc are correctly passed to Metal kernel

2. **Add dimension logging**
   ```c
   printf("ab_dgemm: M=%d, N=%d, K=%d\n", m->rows, m->cols, ...);
   ```

3. **Test with column-major BLAS**
   ```c
   // Try CblasColMajor instead of CblasRowMajor
   cblas_dgemm(CblasColMajor, ...);
   ```

4. **Check threshold logic**
   - Verify 100M FLOPs calculation
   - Check if large rectangles hit GPU path

### Next Steps

**Option A: Fix rectangular matrices (recommended)**
- Debug dimension handling
- Fix memory layout issues
- Validate with test suite
- **Then** integrate with QE

**Option B: Document limitation**
- Mark rectangular matrices as "known issue"
- Focus on square matrices (2048×2048+)
- Update README with limitations

**Option C: Focus on Native API**
- Current per-call API has fundamental issues
- Native API (GPU-resident matrices) bypasses upload/download
- Might avoid dimension issues entirely

### Files Created

- `tests/test_rectangular.c` - Comprehensive test suite (536 lines)
- `tests/RECTANGULAR_MATRICES.md` - Analysis and optimization guide
- `tests/test_rectangle_debug.c` - Quick diagnostic test
- `tests/TESTING_SUMMARY.md` - This file

### Recommendation

**Prioritize fixing correctness before performance.**

The fact that small rectangles work but large ones fail suggests a fixable bug, not a fundamental limitation. We should:

1. Debug the dimension/layout issue (1-2 hours)
2. Fix it (likely < 50 lines of code)
3. Validate with test suite
4. **Then** optimize performance

The performance slowdown (0.79-0.96×) is expected and documented. The **correctness failure** is the blocker.

---

## How to Reproduce

```bash
cd ~/Dev/arm/metal-algos

# Build library with Fortran bridge
make clean && make
clang -Wall -O3 -std=c11 -Iinclude -c src/blas_wrapper.c -o build/blas_wrapper.o
clang -Wall -O3 -std=c11 -Iinclude -c src/fortran_bridge.c -o build/fortran_bridge.o
ar rcs build/libapplebottom.a build/apple_bottom.o build/blas_wrapper.o build/fortran_bridge.o

# Build and run tests
clang -O3 -DACCELERATE_NEW_LAPACK -Iinclude -Lbuild -lapplebottom -lc++ \
  -framework Accelerate -framework Metal -framework Foundation \
  -o tests/test_rectangular tests/test_rectangular.c

./tests/test_rectangular

# Debug test (small size - should pass)
clang -O3 -DACCELERATE_NEW_LAPACK -Iinclude -Lbuild -lapplebottom -lc++ \
  -framework Accelerate -framework Metal -framework Foundation \
  -o tests/test_rectangle_debug tests/test_rectangle_debug.c

./tests/test_rectangle_debug
```

---

## Update (Current Session)

- Identified issue: rectangular matrices fail at large sizes
- Small rectangles (100×50) work perfectly
- QE integration still works (needs investigation why)
- Performance is 0.79-1.15× vs BLAS (expected for rectangles)
- **Next:** Debug dimension handling in apple_bottom.m
