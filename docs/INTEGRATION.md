# Integration Guide

How to integrate apple-bottom into your existing scientific code.

## Quick Decision: Should You Use apple-bottom?

**YES** if:
- Your workload is **iterative** (CG, GMRES, SCF, eigensolvers)
- Matrix sizes are **≥ 2048**
- You're already on Apple Silicon

**NO** if:
- Single DGEMM calls scattered throughout code
- Matrix sizes < 1024
- You need cross-platform compatibility

## The Key Insight

apple-bottom wins by **amortizing conversion overhead**:

```
Traditional BLAS (per-call):
  [FP64→DD] → [GPU GEMM] → [DD→FP64]   ← conversion every call = slow

apple-bottom (persistent):
  [FP64→DD] → [GPU GEMM] → [GPU GEMM] → ... → [DD→FP64]
       ↑         many iterations              ↑
    once                                    once
```

## Integration Patterns

### Pattern 1: Iterative Solver (Easiest)

If you have a loop that calls DGEMM repeatedly:

```c
// BEFORE (using Accelerate)
void solve_iterative(double* A, double* x, double* b, int N, int max_iter) {
    double* temp = malloc(N * sizeof(double));
    
    for (int iter = 0; iter < max_iter; iter++) {
        // Matrix-vector: temp = A * x
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
        // Matrix-vector on GPU (no conversion!)
        ab_dgemm(mA, mX, mTemp);
        
        // Download temp for CPU scalar operations
        // (or implement these on GPU too for max speed)
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

## Fortran Integration

For Fortran codes (common in scientific computing):

```fortran
! fortran_interface.f90
module apple_bottom_fortran
    use iso_c_binding
    implicit none
    
    interface
        function ab_init() bind(c, name='ab_init')
            import :: c_int
            integer(c_int) :: ab_init
        end function
        
        subroutine ab_shutdown() bind(c, name='ab_shutdown')
        end subroutine
        
        function ab_matrix_create(rows, cols) bind(c, name='ab_matrix_create')
            import :: c_ptr, c_int
            integer(c_int), value :: rows, cols
            type(c_ptr) :: ab_matrix_create
        end function
        
        function ab_matrix_upload(m, data, parallel) bind(c, name='ab_matrix_upload')
            import :: c_ptr, c_int, c_double, c_bool
            type(c_ptr), value :: m
            real(c_double), intent(in) :: data(*)
            logical(c_bool), value :: parallel
            integer(c_int) :: ab_matrix_upload
        end function
        
        function ab_dgemm(A, B, C) bind(c, name='ab_dgemm')
            import :: c_ptr, c_int
            type(c_ptr), value :: A, B, C
            integer(c_int) :: ab_dgemm
        end function
    end interface
    
contains
    subroutine gpu_dgemm(A, B, C, N)
        real(8), intent(in) :: A(N,N), B(N,N)
        real(8), intent(out) :: C(N,N)
        integer, intent(in) :: N
        
        type(c_ptr) :: mA, mB, mC
        integer :: status
        
        status = ab_init()
        
        mA = ab_matrix_create(N, N)
        mB = ab_matrix_create(N, N)
        mC = ab_matrix_create(N, N)
        
        status = ab_matrix_upload(mA, A, .true._c_bool)
        status = ab_matrix_upload(mB, B, .true._c_bool)
        
        status = ab_dgemm(mA, mB, mC)
        
        status = ab_matrix_download(mC, C, .true._c_bool)
        
        ! cleanup...
    end subroutine
end module
```

## Python Integration

```python
# apple_bottom.py
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

# Usage
if __name__ == "__main__":
    A = np.random.rand(2048, 2048)
    B = np.random.rand(2048, 2048)
    
    C = gpu_matmul(A, B)
    print(f"Result shape: {C.shape}")
```

## Performance Tips

### 1. Minimize Transfers

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

### 2. Use Parallel Conversion for Large Matrices

```c
// For N > 512, use parallel conversion
ab_matrix_upload(m, data, true);   // true = parallel

// For small matrices, serial is faster (no dispatch overhead)
ab_matrix_upload(m, data, false);  // false = serial
```

### 3. Reuse Matrix Handles

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

### 4. Check Crossover Points

```c
// Route based on size
void smart_dgemm(int N, double* A, double* B, double* C) {
    if (N >= 2048) {
        // Use GPU
        gpu_dgemm(A, B, C, N);
    } else {
        // Use AMX
        cblas_dgemm(...);
    }
}
```

## Troubleshooting

### "No Metal GPU available"

- Check you're on Apple Silicon (M1/M2/M3)
- Ensure Metal framework is linked
- Run `system_profiler SPDisplaysDataType` to verify GPU

### Slow performance

1. Check matrix sizes (< 2048 will be slower than AMX)
2. Ensure you're not uploading/downloading every iteration
3. Use `ab_print_stats()` to see time breakdown

### Precision issues

- Double-float gives ~10⁻¹⁵ precision, not 10⁻¹⁶
- For ~10⁻¹⁶, use native FP64 (AMX)
- Check that inputs aren't denormalized

### Memory errors

- Each N×N matrix uses 8×N² bytes (DD format)
- 8192×8192 = 512MB per matrix
- Check unified memory limits
