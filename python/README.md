# apple-bottom — Python Wrapper

FP64-class BLAS for Apple Silicon GPUs, from Python.

> **Experimental:** the bindings are not covered by `make test`; check them against your build with `make test-python`.

## Features

- **FP64-class arithmetic**: double-float values with about 48 significant bits (~10⁻¹⁵ per value); GEMM results are accurate to the 10⁻¹⁴ range normwise (~5×10⁻¹⁴ at N = 4096)
- **Pythonic API**: numpy array integration, context managers, proper error handling
- **Type hints**: Full PEP 484 annotations for IDE support
- **Multiple APIs**: Low-level ctypes bindings, high-level classes, and functional convenience functions
- **Memory efficiency**: Optional memory pool for iterative codes
- **Session management**: Named matrix groups for complex workflows

## Installation

```bash
cd /path/to/apple-bottom
make                        # builds build/libapplebottom.dylib
pip install -e python/      # the package finds ../build/libapplebottom.dylib
```

To use another copy of the library, set
`AB_LIBRARY=/path/to/libapplebottom.dylib`. Without it the package also
looks in `/usr/local/lib` and `/opt/homebrew/lib`, so `make install`
(`sudo` for the default `/usr/local`) is optional; after an install to
another prefix, point `AB_LIBRARY` at `<prefix>/lib/libapplebottom.dylib`.

## Quick Start

### Basic Matrix Multiplication

```python
import numpy as np
import apple_bottom as ab

# Initialize GPU
ab.init()

# Create test matrices
A = np.random.randn(1024, 1024)
B = np.random.randn(1024, 512)

# Compute C = A @ B (handles upload/compute/download)
C = ab.dgemm(A, B)

print(f"Result shape: {C.shape}")
print(f"Device: {ab.get_device_name()}")

# Cleanup
ab.shutdown()
```

### Moving Data with the Matrix Class

```python
import numpy as np
import apple_bottom as ab

ab.init()

gpu_A = ab.Matrix(2048, 2048)               # DD storage on the GPU
gpu_A.upload(np.random.randn(2048, 2048))   # FP64 -> DD, row-major
A_back = gpu_A.download()                   # DD -> FP64

ab.shutdown()
```

Products between matrices that stay on the GPU go through a `Session`
(next section) or the C API.

### Using Sessions for Complex Workflows

```python
import numpy as np
import apple_bottom as ab

ab.init()

# Create a session to organize related matrices
with ab.Session() as session:
    # Add named matrices
    session.add("A", 512, 512)
    session.add("B", 512, 512)
    session.add("C", 512, 512)

    # Upload data
    A_data = np.random.randn(512, 512)
    B_data = np.random.randn(512, 512)
    session.upload("A", A_data)
    session.upload("B", B_data)

    # Compute C = A @ B
    session.dgemm("A", "B", "C")

    # Download result
    C_result = session.download("C")
    print(f"C shape: {C_result.shape}")

ab.shutdown()
```

### Complex Matrix Multiplication

```python
import numpy as np
import apple_bottom as ab

ab.init()

# Create complex matrices as separate real/imaginary parts
Ar = np.random.randn(256, 256)
Ai = np.random.randn(256, 256)
Br = np.random.randn(256, 256)
Bi = np.random.randn(256, 256)

# Compute complex product
Cr, Ci = ab.zgemm(Ar, Ai, Br, Bi)

print(f"Real part shape: {Cr.shape}")
print(f"Imag part shape: {Ci.shape}")

ab.shutdown()
```

### Memory Pool for Iterative Codes

```python
import numpy as np
import apple_bottom as ab

ab.init()

pool = ab.MemoryPool(size_hint=32)

for iteration in range(100):
    # Get matrices from pool (avoids repeated allocation)
    A = pool.get_matrix(512, 512)
    B = pool.get_matrix(512, 512)
    C = pool.get_matrix(512, 512)

    # Use matrices...
    # (In real code, upload/compute/download here)

    # Reset pool for next iteration
    pool.reset()

ab.shutdown()
```

## API Overview

### Initialization

- `ab.init()` — Initialize GPU device
- `ab.shutdown()` — Cleanup and shutdown
- `ab.get_device_name()` — Get GPU name
- `ab.is_initialized()` — Check initialization status

### Functional API (All-in-One)

- `ab.dgemm(A, B, alpha=1.0, beta=0.0)` — `alpha * A @ B` (the result starts from zero, so `beta` has no effect)
- `ab.zgemm(Ar, Ai, Br, Bi)` — Complex matrix multiply
- `ab.dsyrk(A)` — Symmetric rank-k update: C = A @ A.T

### Matrix Class

```python
m = ab.Matrix(rows, cols)
m.upload(numpy_array, parallel=True)
result = m.download(parallel=True)
m.zero()
m.copy_from(other_matrix)
m.shape  # Returns (rows, cols)
m.size   # Returns total elements
```

### Session Class

```python
with ab.Session() as session:
    session.add(name, rows, cols)
    session.upload(name, numpy_array)
    session.download(name)
    session.dgemm(A_name, B_name, C_name)
    session.zgemm(Ar_name, Ai_name, Br_name, Bi_name, Cr_name, Ci_name)
```

### Memory Pool

```python
pool = ab.MemoryPool(size_hint=16)
matrix = pool.get_matrix(rows, cols)
pool.reset()  # matrices obtained before the reset can no longer be used
```

### Statistics

```python
stats = ab.get_stats()
# Returns dict: upload_time_ms, download_time_ms, kernel_time_ms,
#               dgemm_count, zgemm_count, elements_converted

ab.reset_stats()
ab.print_stats()
```

## Performance Guidance

There are two regimes (measurements on an M2 Max in the main
[README](../README.md#performance)):

- **`ab.dgemm`, `ab.zgemm`, `ab.dsyrk`** allocate, convert FP64 to DD,
  transfer and convert back on every call. That is the cost structure of
  the BLAS ABI, where Accelerate (and so numpy linked to it) is faster for
  nearly every shape; do not expect these calls to beat `A @ B`.
- **Matrices uploaded once and multiplied repeatedly** in a `Session` stay
  on the GPU. There the GPU overtakes Accelerate from about N = 1,500
  (DGEMM) and N = 1,000 (ZGEMM).

For ZHERK, use `cblas_zherk` (or numpy): the GPU version is slower at every
measured size.

## Error Handling

All functions raise descriptive exceptions:

- `AppleBottomError` — General errors
- `DeviceNotFoundError` — No GPU available
- `AllocationError` — Memory allocation failure
- `DimensionMismatchError` — Shape incompatibility
- `NotUploadedError` — Data not on GPU
- `KernelExecutionError` — GPU kernel failure
- `InvalidArgumentError` — Invalid parameters
- `ShaderCompileError` — Metal shader compilation failure

```python
try:
    ab.init()
    C = ab.dgemm(A, B)
except ab.DeviceNotFoundError:
    print("No Apple Silicon GPU available")
except ab.DimensionMismatchError as e:
    print(f"Shape error: {e}")
except ab.AppleBottomError as e:
    print(f"GPU error: {e}")
```

## System Requirements

- macOS 14 (Sonoma) or later, on Apple Silicon (M1 or later)
- Python 3.9+
- numpy 1.20+

## Library Limits

- Maximum matrix size: rows × cols ≤ 2³¹ − 1 elements (`MAX_ELEMENTS`, `AB_MAX_MATRIX_ELEMENTS`), any shape
- Memory pool capacity: 128 matrices
- Session capacity: 64 matrices per session
- Threads: calls on different matrices may come from different threads; one matrix, pool or session must not be used by two threads at once

## Building from Source

```bash
cd /path/to/apple-bottom
make clean
make test         # the C library's test suite
make test-python  # these bindings
```

Then `pip install -e python/` as above.

## License

MIT License — see LICENSE file in the main repository.
