# apple-bottom — Python Wrapper

High-performance FP64-class BLAS for Apple Silicon GPUs, from Python.

## Features

- **High-precision arithmetic**: ~10⁻¹⁵ Frobenius relative error via double-float emulation
- **Pythonic API**: numpy array integration, context managers, proper error handling
- **Type hints**: Full PEP 484 annotations for IDE support
- **Multiple APIs**: Low-level ctypes bindings, high-level classes, and functional convenience functions
- **Memory efficiency**: Optional memory pool for iterative codes
- **Session management**: Named matrix groups for complex workflows

## Installation

Ensure the C library is built first:

```bash
cd /path/to/apple-bottom
make clean && make
make install  # Installs libapplebottom to /usr/local/lib or /opt/homebrew/lib
```

Then install the Python package:

```bash
cd python/
pip install -e .
```

Or install from source directly:

```bash
pip install -e /path/to/apple-bottom/python
```

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

### Fine-Grained Control with Matrix Class

```python
import numpy as np
import apple_bottom as ab

ab.init()

# Create GPU matrices
gpu_A = ab.Matrix(2048, 2048)
gpu_B = ab.Matrix(2048, 2048)
gpu_C = ab.Matrix(2048, 2048)

# Upload data
A_data = np.random.randn(2048, 2048)
B_data = np.random.randn(2048, 2048)
gpu_A.upload(A_data)
gpu_B.upload(B_data)
gpu_C.zero()

# Perform computation
# Note: ab_dgemm computes C = A @ B
gpu_A.dgemm(gpu_B, gpu_C)  # Would be low-level call

# Download result
C_result = gpu_C.download()

ab.shutdown()
```

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

- `ab.dgemm(A, B, alpha=1.0, beta=0.0)` — Double-precision matrix multiply
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
pool.reset()
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

GPU is beneficial when:
- DGEMM: N >= 2048
- ZGEMM: N >= 1024
- DSYRK: N >= 3072

Use CPU (Accelerate framework) for:
- ZHERK: GPU is 20x slower — use cblas_zherk instead

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

- macOS 14+ (Monterey or later)
- Apple Silicon Mac (M1, M2, M3, etc.)
- Python 3.9+
- numpy 1.20+
- Xcode 16+ SDK recommended (for ~10⁻¹⁵ precision; ~10⁻⁸ without)

## Library Limits

- Maximum matrix dimension: 46,340 (AB_MAX_DIMENSION)
- Memory pool capacity: 128 matrices
- Session capacity: 64 matrices per session
- Thread safety: Matrix operations are NOT thread-safe (Metal queue is serialized)

## Building from Source

```bash
cd /path/to/apple-bottom
make clean
make test  # Verify 48/48 tests pass
make install
```

Then install Python package from `python/` directory.

## License

MIT License — see LICENSE file in the main repository.
