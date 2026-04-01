# Integration Guide

Complete guide for integrating apple-bottom into scientific computing codes.

---

## Table of Contents

- [Quick Decision](#quick-decision-should-you-use-apple-bottom)
- [C Integration](#c-integration)
- [Fortran Integration](#fortran-integration)
- [Quantum ESPRESSO Integration](#quantum-espresso-integration)

---

## Quick Decision: Should You Use apple-bottom?

**YES** if:
- Your workload is **iterative** (CG, GMRES, SCF, eigensolvers)
- Matrix sizes are **≥ 2048**
- You're already on Apple Silicon

**NO** if:
- Single DGEMM calls scattered throughout code
- Matrix sizes < 1024
- You need cross-platform compatibility

---

## C Integration

### The Key Insight

apple-bottom wins by **amortizing conversion overhead**:

```
Traditional BLAS (per-call):
  [FP64→DD] → [GPU GEMM] → [DD→FP64]   ← conversion every call = slow

apple-bottom (persistent):
  [FP64→DD] → [GPU GEMM] → [GPU GEMM] → ... → [DD→FP64]
       ↑         many iterations              ↑
    once                                    once
```

### Pattern 1: Iterative Solver

If you have a loop that calls DGEMM repeatedly:

```c
// BEFORE (using Accelerate)
void solve_iterative(double* A, double* x, double* b, int N, int max_iter) {
    double* temp = malloc(N * sizeof(double));

    for (int iter = 0; iter < max_iter; iter++) {
        cblas_dgemv(CblasRowMajor, CblasNoTrans, N, N,
                    1.0, A, N, x, 1, 0.0, temp, 1);
        // ... update x based on temp and b ...
    }
    free(temp);
}

// AFTER (using apple-bottom)
void solve_iterative(double* A, double* x, double* b, int N, int max_iter) {
    ab_init();

    // Create GPU matrices ONCE
    ABMatrix mA = ab_matrix_create(N, N);
    ABMatrix mX = ab_matrix_create(N, 1);
    ABMatrix mTemp = ab_matrix_create(N, 1);

    // Upload ONCE
    ab_matrix_upload(mA, A, true);
    ab_matrix_upload(mX, x, true);

    for (int iter = 0; iter < max_iter; iter++) {
        ab_dgemm(mA, mX, mTemp);  // No conversion overhead!

        // Download for CPU scalar operations
        double* temp_cpu = malloc(N * sizeof(double));
        ab_matrix_download(mTemp, temp_cpu, false);

        // ... update x ...

        ab_matrix_upload(mX, x, false);
        free(temp_cpu);
    }

    // Download final result ONCE
    ab_matrix_download(mX, x, true);

    ab_matrix_destroy(mA);
    ab_matrix_destroy(mX);
    ab_matrix_destroy(mTemp);
    ab_shutdown();
}
```

### Pattern 2: SCF Loop (Quantum Chemistry)

```c
// DFT Self-Consistent Field loop
void scf_loop(int nbasis, double* H, double* S, double* P,
              double tol, int max_iter) {
    ab_init();
    ABSession s = ab_session_create();

    // Allocate GPU matrices
    ab_session_add(s, "H", nbasis, nbasis);     // Hamiltonian
    ab_session_add(s, "S", nbasis, nbasis);     // Overlap
    ab_session_add(s, "P", nbasis, nbasis);     // Density
    ab_session_add(s, "F", nbasis, nbasis);     // Fock matrix
    ab_session_add(s, "C", nbasis, nbasis);     // Coefficients
    ab_session_add(s, "T1", nbasis, nbasis);    // Temp
    ab_session_add(s, "T2", nbasis, nbasis);    // Temp

    // Upload constant matrices ONCE
    ab_session_upload(s, "H", H);
    ab_session_upload(s, "S", S);
    ab_session_upload(s, "P", P);

    double energy_old = 0;

    for (int iter = 0; iter < max_iter; iter++) {
        // Build Fock matrix: F = H + G(P)
        // (G is two-electron integrals - may need custom kernel)
        build_fock_matrix(s);

        // Transform to orthogonal basis: F' = S^(-1/2) * F * S^(-1/2)
        ab_session_dgemm(s, "Sinvhalf", "F", "T1");
        ab_session_dgemm(s, "T1", "Sinvhalf", "Fprime");

        // Diagonalize (download for LAPACK, or use GPU eigensolver)
        double* F_cpu = malloc(nbasis * nbasis * sizeof(double));
        ab_session_download(s, "Fprime", F_cpu);

        double* eigenvalues = malloc(nbasis * sizeof(double));
        double* C_cpu = malloc(nbasis * nbasis * sizeof(double));
        diagonalize(F_cpu, eigenvalues, C_cpu, nbasis);  // LAPACK

        // Upload new coefficients
        ab_session_upload(s, "C", C_cpu);

        // Build new density: P = C * Cocc * C^T
        ab_session_dgemm(s, "C", "Cocc", "T1");
        ab_session_dgemm(s, "T1", "Ctrans", "P");

        // Check convergence
        double energy = compute_energy(s);
        if (fabs(energy - energy_old) < tol) {
            printf("Converged at iteration %d\n", iter);
            break;
        }
        energy_old = energy;

        free(F_cpu);
        free(eigenvalues);
        free(C_cpu);
    }

    // Download final density
    ab_session_download(s, "P", P);

    ab_session_destroy(s);
    ab_shutdown();
}
```

### Pattern 3: Batched Operations

If you have many small matrices:

```c
// Process a batch of matrices
void process_batch(double** matrices, int count, int N) {
    ab_init();

    // Allocate all GPU matrices upfront
    ABMatrix* gpu_mats = malloc(count * sizeof(ABMatrix));
    ABMatrix* gpu_results = malloc(count * sizeof(ABMatrix));

    for (int i = 0; i < count; i++) {
        gpu_mats[i] = ab_matrix_create(N, N);
        gpu_results[i] = ab_matrix_create(N, N);
        ab_matrix_upload(gpu_mats[i], matrices[i], true);
    }

    // Process all
    for (int i = 0; i < count; i++) {
        ab_dgemm(gpu_mats[i], gpu_mats[i], gpu_results[i]);  // Square each
    }

    // Download all
    for (int i = 0; i < count; i++) {
        ab_matrix_download(gpu_results[i], matrices[i], true);
        ab_matrix_destroy(gpu_mats[i]);
        ab_matrix_destroy(gpu_results[i]);
    }

    free(gpu_mats);
    free(gpu_results);
    ab_shutdown();
}
```

### Performance Tips

#### 1. Minimize Transfers

```c
// BAD: Upload/download every iteration
for (int i = 0; i < 100; i++) {
    ab_matrix_upload(mA, A, true);   // 100 uploads!
    ab_dgemm(mA, mB, mC);
    ab_matrix_download(mC, C, true); // 100 downloads!
}

// GOOD: Upload once, download once
ab_matrix_upload(mA, A, true);       // 1 upload
for (int i = 0; i < 100; i++) {
    ab_dgemm(mA, mB, mC);            // Pure GPU
}
ab_matrix_download(mC, C, true);     // 1 download
```

#### 2. Reuse Matrix Handles

```c
// BAD: Create/destroy per operation
for (int i = 0; i < 100; i++) {
    ABMatrix m = ab_matrix_create(N, N);  // 100 allocations!
    ab_dgemm(mA, mB, m);
    ab_matrix_destroy(m);                  // 100 frees!
}

// GOOD: Reuse handles
ABMatrix m = ab_matrix_create(N, N);      // 1 allocation
for (int i = 0; i < 100; i++) {
    ab_dgemm(mA, mB, m);
}
ab_matrix_destroy(m);                      // 1 free
```

#### 3. Use Parallel Conversion for Large Matrices

```c
// For N > 512, use parallel conversion
ab_matrix_upload(m, data, true);   // true = parallel

// For small matrices, serial is faster
ab_matrix_upload(m, data, false);  // false = serial
```

#### 4. Check Crossover Points

```c
// Route based on size
void smart_dgemm(int N, double* A, double* B, double* C) {
    if (N >= 2048) {
        gpu_dgemm(A, B, C, N);  // GPU
    } else {
        cblas_dgemm(...);       // AMX
    }
}
```

### Python Integration

```python
import ctypes
import numpy as np

# Load library
_lib = ctypes.CDLL("libapplebottom.dylib")

# Define types
ABMatrix = ctypes.c_void_p
ABStatus = ctypes.c_int

# Bind functions
_lib.ab_init.restype = ABStatus
_lib.ab_matrix_create.argtypes = [ctypes.c_int, ctypes.c_int]
_lib.ab_matrix_create.restype = ABMatrix
_lib.ab_matrix_upload.argtypes = [ABMatrix, ctypes.POINTER(ctypes.c_double), ctypes.c_bool]
_lib.ab_matrix_download.argtypes = [ABMatrix, ctypes.POINTER(ctypes.c_double), ctypes.c_bool]
_lib.ab_dgemm.argtypes = [ABMatrix, ABMatrix, ABMatrix]
_lib.ab_dgemm.restype = ABStatus

class GPUMatrix:
    def __init__(self, rows, cols):
        self.rows = rows
        self.cols = cols
        self._ptr = _lib.ab_matrix_create(rows, cols)

    def upload(self, data):
        data = np.ascontiguousarray(data, dtype=np.float64)
        ptr = data.ctypes.data_as(ctypes.POINTER(ctypes.c_double))
        _lib.ab_matrix_upload(self._ptr, ptr, True)

    def download(self):
        data = np.zeros((self.rows, self.cols), dtype=np.float64)
        ptr = data.ctypes.data_as(ctypes.POINTER(ctypes.c_double))
        _lib.ab_matrix_download(self._ptr, ptr, True)
        return data

def gpu_matmul(A, B):
    """Matrix multiply using apple-bottom GPU"""
    _lib.ab_init()

    M, K = A.shape
    K2, N = B.shape
    assert K == K2

    mA = GPUMatrix(M, K)
    mB = GPUMatrix(K, N)
    mC = GPUMatrix(M, N)

    mA.upload(A)
    mB.upload(B)

    _lib.ab_dgemm(mA._ptr, mB._ptr, mC._ptr)

    return mC.download()
```

### Troubleshooting

**"No Metal GPU available"**
- Check you're on Apple Silicon (M1/M2/M3/M4)
- Run `system_profiler SPDisplaysDataType` to verify GPU

**Slow performance**
1. Check matrix sizes (< 2048 will be slower than AMX)
2. Ensure you're not uploading/downloading every iteration
3. Use `ab_print_stats()` to see time breakdown

**Precision issues**
- Double-float gives ~10⁻¹⁵ precision, not 10⁻¹⁶
- For ~10⁻¹⁶, use native FP64 (AMX)

---

## Fortran Integration

### Overview

apple-bottom provides a Fortran-callable BLAS API using the **EXTERNAL declaration** pattern for drop-in compatibility.

### Supported Operations

| Fortran Call | apple-bottom Equivalent | Status |
|--------------|------------------------|--------|
| `CALL DGEMM(...)` | `CALL ab_dgemm(...)` | ✓ Production |
| `CALL ZGEMM(...)` | `CALL ab_zgemm(...)` | ✓ Production |
| `CALL DSYRK(...)` | `CALL ab_dsyrk(...)` | ✓ Production |
| `CALL ZHERK(...)` | ⚠ Use cblas_zherk (AMX faster) | Not recommended |

### EXTERNAL Declaration Pattern

```fortran
SUBROUTINE your_solver(...)
  IMPLICIT NONE
  EXTERNAL :: ab_zgemm  ! Declare apple-bottom routine

  ! Replace CALL ZGEMM( with CALL ab_zgemm(
  CALL ab_zgemm('N', 'N', m, n, k, alpha, A, lda, B, ldb, beta, C, ldc)
```

**Why EXTERNAL works:**
- Fortran passes addresses (sequence association)
- No explicit interface → no array descriptors
- Compatible with array slices: `A(1,istart)`, `B(i:j,k:l)`, etc.

### Integration Patterns

#### Pattern 1: Simple Replacement

**Before:**
```fortran
SUBROUTINE matrix_multiply(A, B, C, N)
  IMPLICIT NONE
  INTEGER :: N
  COMPLEX(8) :: A(N,N), B(N,N), C(N,N)
  COMPLEX(8) :: alpha, beta

  alpha = (1.d0, 0.d0)
  beta = (0.d0, 0.d0)

  CALL ZGEMM('N', 'N', N, N, N, alpha, A, N, B, N, beta, C, N)
END SUBROUTINE
```

**After:**
```fortran
SUBROUTINE matrix_multiply(A, B, C, N)
  IMPLICIT NONE
  EXTERNAL :: ab_zgemm  ! Add this
  INTEGER :: N
  COMPLEX(8) :: A(N,N), B(N,N), C(N,N)
  COMPLEX(8) :: alpha, beta

  alpha = (1.d0, 0.d0)
  beta = (0.d0, 0.d0)

  CALL ab_zgemm('N', 'N', N, N, N, alpha, A, N, B, N, beta, C, N)  ! Change this
END SUBROUTINE
```

#### Pattern 2: Davidson Eigensolver

```fortran
SUBROUTINE davidson_eigensolver(H, S, psi, eval, ndim, nvec, niter)
  IMPLICIT NONE
  EXTERNAL :: ab_zgemm

  INTEGER :: ndim, nvec, niter, iter
  COMPLEX(8) :: H(ndim,ndim), S(ndim,ndim)
  COMPLEX(8) :: psi(ndim,nvec), hpsi(ndim,nvec), spsi(ndim,nvec)
  REAL(8) :: eval(nvec)
  COMPLEX(8) :: alpha, beta

  alpha = (1.d0, 0.d0)
  beta = (0.d0, 0.d0)

  DO iter = 1, niter
    ! Apply Hamiltonian: hpsi = H * psi
    CALL ab_zgemm('N', 'N', ndim, nvec, ndim, &
                  alpha, H, ndim, psi, ndim, beta, hpsi, ndim)

    ! Apply overlap: spsi = S * psi
    CALL ab_zgemm('N', 'N', ndim, nvec, ndim, &
                  alpha, S, ndim, psi, ndim, beta, spsi, ndim)

    ! ... diagonalize, update psi, check convergence ...
  END DO
END SUBROUTINE
```

#### Pattern 3: Conjugate Transpose

```fortran
SUBROUTINE compute_overlap(psi, overlap, ndim, nvec)
  IMPLICIT NONE
  EXTERNAL :: ab_zgemm

  INTEGER :: ndim, nvec
  COMPLEX(8) :: psi(ndim,nvec), overlap(nvec,nvec)
  COMPLEX(8) :: alpha, beta

  alpha = (1.d0, 0.d0)
  beta = (0.d0, 0.d0)

  ! overlap = psi^H * psi (conjugate transpose)
  CALL ab_zgemm('C', 'N', nvec, nvec, ndim, &
                alpha, psi, ndim, psi, ndim, beta, overlap, nvec)
END SUBROUTINE
```

**Supported transposes:**
- `'N'` — No transpose
- `'T'` — Transpose (real matrices)
- `'C'` — Conjugate transpose (complex matrices)

### Build Integration

#### Step 1: Build apple-bottom

```bash
cd /path/to/apple-bottom
make
make test
```

#### Step 2: Update Your Makefile

```makefile
# Fortran compiler
FC = gfortran

# BLAS/LAPACK libraries
# IMPORTANT: -lapplebottom must come BEFORE -framework Accelerate
LIBS = -L/path/to/apple-bottom/build -lapplebottom \
       -framework Accelerate -framework Metal -framework Foundation

# Link your executable
my_app: my_app.o solver.o
	$(FC) -o my_app my_app.o solver.o $(LIBS)
```

**Critical:** apple-bottom must be listed BEFORE Accelerate so the linker finds `ab_zgemm_()` before falling back to OpenBLAS.

#### Step 3: Verify Linking

```bash
otool -L my_app | grep -E "applebottom|Metal|Accelerate"
```

Should show:
```
/System/Library/Frameworks/Accelerate.framework/Versions/A/Accelerate
/System/Library/Frameworks/Metal.framework/Versions/A/Metal
/System/Library/Frameworks/Foundation.framework/Versions/C/Foundation
```

### Automatic Routing

apple-bottom automatically routes calls based on problem size:

```
┌─────────────────────┐
│  CALL ab_zgemm(...) │
└──────────┬──────────┘
           │
           ▼
   ┌───────────────┐
   │ fortran_bridge│
   │ FLOPs check   │
   └───┬───────┬───┘
       │       │
  < 100M FLOPs │ ≥ 100M FLOPs
       │       │
       ▼       ▼
   ┌─────┐ ┌──────┐
   │BLAS │ │ GPU  │
   │(CPU)│ │(Metal)│
   └─────┘ └──────┘
```

**Threshold:** 100M FLOPs = M × N × K × 8

**Examples:**
- `M=2048, N=2048, K=2048` → 68.7B FLOPs → **GPU**
- `M=512, N=512, K=512` → 1.1B FLOPs → **GPU**
- `M=256, N=256, K=256` → 134M FLOPs → **GPU**
- `M=128, N=128, K=128` → 16.8M FLOPs → **CPU**

### API Reference

#### ZGEMM (Complex Double Precision)

```fortran
EXTERNAL :: ab_zgemm

CALL ab_zgemm(transA, transB, M, N, K, alpha, A, lda, B, ldb, beta, C, ldc)
```

**Arguments:**
- `transA`, `transB` : `CHARACTER` — `'N'`, `'T'`, or `'C'` (conjugate transpose)
- `M`, `N`, `K` : `INTEGER` — matrix dimensions
- `alpha` : `COMPLEX(8)` — scalar multiplier
- `A` : `COMPLEX(8)` — matrix A
- `lda` : `INTEGER` — leading dimension of A
- `B` : `COMPLEX(8)` — matrix B
- `ldb` : `INTEGER` — leading dimension of B
- `beta` : `COMPLEX(8)` — scalar multiplier for C
- `C` : `COMPLEX(8)` — matrix C, overwritten with result
- `ldc` : `INTEGER` — leading dimension of C

**Operation:** `C = alpha * op(A) * op(B) + beta * C`

**Implementation:** Uses Gauss 3-multiply algorithm (25% compute reduction).

### Common Issues

**"undefined reference to `ab_zgemm_`"**

```bash
# Check library exists
ls -la /path/to/apple-bottom/build/libapplebottom.a

# Verify library has symbol
nm /path/to/apple-bottom/build/libapplebottom.a | grep ab_zgemm_

# Update Makefile
LIBS = -L/path/to/apple-bottom/build -lapplebottom -framework Accelerate ...
```

**"S matrix not positive definite" or Wrong Results**

```fortran
! DON'T do this:
USE apple_bottom_mod  ! ✗ Creates array descriptor issues

! DO this:
EXTERNAL :: ab_zgemm  ! ✓ Simple, works with array slices
```

---

## Quantum ESPRESSO Integration

### Overview

Integration with Quantum ESPRESSO 7.4.1 to GPU-accelerate Davidson eigensolver.

**Performance:** 1.22× speedup vs 6-thread OpenBLAS on Si64 benchmark, 11 decimal place energy agreement.

### Prerequisites

- macOS 14+ with Apple Silicon
- Quantum ESPRESSO 7.4.1 source
- apple-bottom library built
- gfortran (conda or Homebrew)

### Quick Start

```bash
# 1. Build apple-bottom
cd /path/to/apple-bottom
make && make test

# 2. Patch QE cegterg.f90
cd /path/to/q-e-qe-7.4.1/KS_Solvers/Davidson
# Add: EXTERNAL :: ab_zgemm
# Replace: ZGEMM → ab_zgemm (12 calls)

# 3. Update make.inc
DFLAGS = -D__GFORTRAN -D__STD_F95 -D__FFTW3
BLAS_LIBS = -L/path/to/apple-bottom/build -lapplebottom \
            -framework Accelerate -framework Metal -framework Foundation

# 4. Build QE
cd /path/to/q-e-qe-7.4.1
make clean && make pw -j8

# 5. Run benchmark
cd /path/to/benchmark
time /path/to/q-e-qe-7.4.1/bin/pw.x < si64.in > si64_gpu.out 2>&1
grep '!' si64_gpu.out  # Should show: -2990.44276157 Ry
```

### Integration Architecture

#### Call Chain

```
QE cegterg.f90:
  CALL ab_zgemm('N', 'N', m, n, k, ...)
           ↓
fortran_bridge.c: ab_zgemm_()
  - Dereference pointers
  - Calculate FLOPs = M × N × K × 8
  - Check threshold
           ↓
  ├─ < 100M FLOPs → zgemm_() (OpenBLAS, zero overhead)
  └─ ≥ 100M FLOPs → ab_zgemm_blas() (GPU)
           ↓
blas_wrapper.c:
  - Convert interleaved → split-complex
  - Upload to GPU
  - Dispatch Metal kernel
  - Download + convert back
           ↓
apple_bottom.m:
  - DD arithmetic (~10⁻¹⁵ precision)
  - Gauss 3-multiply algorithm
```

### Detailed Integration

#### Step 1: Patch cegterg.f90

**File:** `KS_Solvers/Davidson/cegterg.f90`

Add after `IMPLICIT NONE` (line ~41):
```fortran
IMPLICIT NONE
EXTERNAL :: ab_zgemm
```

Replace all 12 `CALL ZGEMM(` with `CALL ab_zgemm(` in the main subroutine.

**Do NOT modify** `pcegterg` subroutine (different parallelization).

#### Step 2: Update make.inc

```makefile
DFLAGS = -D__GFORTRAN -D__STD_F95 -D__FFTW3
BLAS_LIBS = -L/path/to/apple-bottom/build -lapplebottom \
            -framework Accelerate -framework Metal -framework Foundation
```

**Important:** `-lapplebottom` BEFORE `-framework Accelerate`.

#### Step 3: Build and Validate

```bash
cd /path/to/q-e-qe-7.4.1
make clean && make pw -j8

# Verify binary
ls -la bin/pw.x

# Run test
cd /path/to/benchmark
time /path/to/q-e-qe-7.4.1/bin/pw.x < si64.in > si64_gpu.out 2>&1

# Verify correctness
grep '!' si64_gpu.out
# Expected: !    total energy              =   -2990.44276157 Ry
```

### Performance Results

**Si64 Benchmark** (64-atom silicon, M2 Max):

| Configuration | Wall Time | Energy (Ry) | Status |
|--------------|-----------|-------------|--------|
| OpenBLAS (6 threads) | 2m28s | -2990.44276157 | Baseline |
| **apple-bottom GPU** | **2m01s** | **-2990.44276157** | ✓ 1.22× speedup |

**Breakdown:**
- cegterg: 112.3s → 86.5s (30% faster)
- CPU usage: 600% → 340% (47% reduction)
- **Numerical correctness:** 11 decimal place agreement

### Troubleshooting

**"undefined reference to `ab_zgemm_`"**

```bash
# Verify library
ls -la /path/to/apple-bottom/build/libapplebottom.a
nm /path/to/apple-bottom/build/libapplebottom.a | grep ab_zgemm_

# Check make.inc
grep "BLAS_LIBS" make.inc
# Should have: -lapplebottom BEFORE -framework Accelerate
```

**Wrong energy**

```bash
# Verify EXTERNAL declaration (not module)
grep "EXTERNAL :: ab_zgemm" KS_Solvers/Davidson/cegterg.f90

# Ensure ONLY cegterg.f90 modified
git status
```

**Slow performance**

```bash
# Enable stats
export AB_PRINT_STATS=1
/path/to/q-e-qe-7.4.1/bin/pw.x < si64.in > si64.out 2>&1
grep "ab_zgemm" si64.out
```

### Files Changed

**Total:** 15 lines across 2 files

- `KS_Solvers/Davidson/cegterg.f90` — 13 lines (1 EXTERNAL + 12 replacements)
- `make.inc` — 2 lines (DFLAGS + BLAS_LIBS)

---

## Support

- Main documentation: [`README.md`](../README.md)
- V&V documentation: [`docs/vv/VV_REPORT.md`](vv/VV_REPORT.md)
- Issues: https://github.com/grantdh/apple-bottom/issues
