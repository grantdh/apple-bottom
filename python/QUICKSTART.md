# apple-bottom Python — Quick Start Guide

## Installation

```bash
# 1. Build the C library
cd /path/to/apple-bottom
make clean && make
make install

# 2. Install Python wrapper
cd python/
pip install -e .
```

## Minimal Example

```python
import numpy as np
import apple_bottom as ab

# Initialize GPU
ab.init()

# Create random matrices
A = np.random.randn(1024, 1024)
B = np.random.randn(1024, 1024)

# Multiply on GPU (handles upload/compute/download)
C = ab.dgemm(A, B)

print(f"Result: {C.shape}")
print(f"Device: {ab.get_device_name()}")

ab.shutdown()
```

## Next Steps

For more examples, see:
- `examples/demo_dgemm.py` — Comprehensive demonstrations
- `README.md` — Full API documentation

## Key APIs

### Simple (One Function)
```python
C = ab.dgemm(A, B)  # or
Cr, Ci = ab.zgemm(Ar, Ai, Br, Bi)
```

### Control (Classes)
```python
gpu_A = ab.Matrix(rows, cols)
gpu_A.upload(cpu_data)
# ... compute ...
cpu_result = gpu_A.download()
```

### Organized (Sessions)
```python
with ab.Session() as s:
    s.add("A", m, k)
    s.add("B", k, n)
    s.add("C", m, n)
    s.upload("A", A_data)
    s.upload("B", B_data)
    s.dgemm("A", "B", "C")
    result = s.download("C")
```

## Error Handling

```python
try:
    ab.init()
    result = ab.dgemm(A, B)
except ab.DeviceNotFoundError:
    print("No Apple Silicon GPU available")
except ab.DimensionMismatchError:
    print("Matrix dimensions incompatible")
except ab.AppleBottomError as e:
    print(f"GPU error: {e}")
```

## Performance Tips

- GPU wins for large matrices: DGEMM N ≥ 2048
- Use memory pool in loops to reduce allocation overhead
- Call `ab.get_stats()` to profile operations
- Complex (ZGEMM) requires separate real/imaginary arrays

## Limits

- Max dimension: 46,340
- Max matrices per pool: 128
- Max matrices per session: 64
- Requires Apple Silicon Mac (M1, M2, M3, etc.)
- Requires Python 3.9+

For full details, see README.md
