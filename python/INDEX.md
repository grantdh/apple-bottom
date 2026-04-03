# Python ctypes Wrapper for apple-bottom — Complete Reference

## File Structure

```
python/
├── apple_bottom/
│   ├── __init__.py       (29 KB) - Main module with ctypes bindings
│   └── py.typed          (empty) - PEP 561 type marker
├── examples/
│   └── demo_dgemm.py     (8.7 KB) - Comprehensive example script
├── setup.py              (1.6 KB) - setuptools configuration
├── README.md             (6.3 KB) - Full documentation
├── QUICKSTART.md         (1.2 KB) - Fast start guide
└── INDEX.md              (this file)
```

## Quick Navigation

| Document | Purpose | Audience |
|----------|---------|----------|
| **QUICKSTART.md** | Get started in 5 minutes | New users |
| **README.md** | Complete API reference | All users |
| **examples/demo_dgemm.py** | Working code examples | Learners, testers |
| **apple_bottom/__init__.py** | Implementation details | Advanced users |

## Module Organization

### apple_bottom/__init__.py Structure

1. **Imports and Constants** (lines 1-30)
   - Version info
   - MAX_DIMENSION, POOL_CAPACITY, SESSION_CAPACITY

2. **Error Classes** (lines ~40-95)
   - AppleBottomError (base)
   - 7 specific exception subclasses
   - Status code to exception mapping

3. **Library Loading** (lines ~100-135)
   - `_find_library()` function
   - Search paths: build/, /usr/local/lib, /opt/homebrew/lib
   - CDLL loading with fallbacks

4. **C Type Definitions** (lines ~140-195)
   - ctypes wrappers: ABMatrix, ABSession, etc.
   - Enum classes: ABStatus, ABTranspose
   - ABStats structure

5. **Low-Level Bindings** (lines ~200-350)
   - `_LibWrapper` class
   - All 33 C function signatures
   - Type-safe ctypes configuration

6. **Matrix Class** (lines ~355-450)
   - Creation, lifecycle management
   - Upload/download with validation
   - Zero, copy, dimension access

7. **MemoryPool Class** (lines ~455-510)
   - Pool creation and destruction
   - Matrix allocation from pool
   - Reset functionality

8. **Session Class** (lines ~515-680)
   - Named matrix management
   - Context manager support
   - DGEMM and ZGEMM operations

9. **Functional API** (lines ~685-870)
   - Convenience functions: init(), shutdown()
   - One-liner operations: dgemm(), zgemm(), dsyrk()
   - Statistics and utilities

10. **Public API Export** (lines ~875-end)
    - __all__ list for clean imports
    - Version metadata

## Key Classes and Functions

### Class Hierarchy

```
Matrix
├── Properties: shape, rows, cols, size
├── Methods: upload(), download(), zero(), copy_from()
└── Auto-cleanup: __del__()

MemoryPool
├── Methods: get_matrix(), reset()
└── Auto-cleanup: __del__()

Session (ContextManager)
├── Methods: add(), get(), upload(), download()
├── BLAS: dgemm(), zgemm()
└── Auto-cleanup: __exit__(), __del__()

Exceptions (Exception tree)
├── AppleBottomError (base)
├── DeviceNotFoundError
├── AllocationError
├── DimensionMismatchError
├── NotUploadedError
├── KernelExecutionError
├── InvalidArgumentError
└── ShaderCompileError
```

### Functional API

```
Initialization:
  init() → None
  shutdown() → None
  get_device_name() → str
  is_initialized() → bool

Matrix Operations:
  dgemm(A: ndarray, B: ndarray, ...) → ndarray
  zgemm(Ar, Ai, Br, Bi) → (Cr, Ci)
  dsyrk(A) → C

Statistics:
  get_stats() → dict
  reset_stats() → None
  print_stats() → None

Utilities:
  status_string(code: int) → str
```

## Code Quality Features

- **Type Hints**: PEP 484 compliant throughout
- **Error Messages**: Descriptive, actionable error text
- **Docstrings**: Google-style docstrings on all public APIs
- **Validation**: Input shape/dtype checking
- **Resource Management**: __del__ and context managers
- **Library Discovery**: 4-location fallback search
- **Constants**: Module-level constants with clear meaning

## Common Tasks

### Get started
See: **QUICKSTART.md**

### Learn all APIs
See: **README.md**

### See working code
See: **examples/demo_dgemm.py**

### Understand implementation
See: **apple_bottom/__init__.py** (annotated above)

### Install package
See: **setup.py** or **README.md**

## Performance Characteristics

| Operation | Best For | GPU Wins When |
|-----------|----------|---------------|
| DGEMM | General matrix multiply | N >= 2048 |
| ZGEMM | Complex matrix multiply | N >= 1024 |
| DSYRK | Symmetric rank-k | N >= 3072 |
| ZHERK | Hermitian rank-k | Use CPU (20x faster) |

Memory overhead:
- Matrix: ~8 bytes per element (double precision)
- Pool: Reduces allocation overhead in loops
- Session: Named matrix registry overhead negligible

## Constants and Limits

```python
MAX_DIMENSION = 46340        # Max rows/cols per matrix
POOL_CAPACITY = 128          # Max matrices in pool
SESSION_CAPACITY = 64        # Max matrices per session

Transpose modes:
  NO_TRANS = 0
  TRANS = 1
  CONJ_TRANS = 2

Status codes:
  OK = 0
  ERROR_NO_DEVICE = -1
  ERROR_ALLOC_FAILED = -2
  ERROR_DIMENSION_MISMATCH = -3
  ERROR_NOT_UPLOADED = -4
  ERROR_KERNEL_FAILED = -5
  ERROR_INVALID_ARG = -6
  ERROR_SHADER_COMPILE = -7
```

## System Requirements

- macOS 14+ (Monterey or later)
- Apple Silicon Mac (M1, M2, M3, M4, etc.)
- Python 3.9, 3.10, 3.11, 3.12
- numpy 1.20 or later
- Xcode 16+ SDK (recommended for full precision)

## Dependencies

**Required**: numpy >= 1.20
**Optional**: pytest >= 7.0 (dev)

## Installation Checklist

- [ ] Build C library: `make && make install`
- [ ] Install wrapper: `pip install -e python/`
- [ ] Test import: `python -c "import apple_bottom; print(apple_bottom.__version__)"`
- [ ] Run demo: `python python/examples/demo_dgemm.py`

## Troubleshooting

| Issue | Solution |
|-------|----------|
| `ImportError: could not find libapplebottom` | Build C library first: `make install` |
| `DeviceNotFoundError` | Run on Apple Silicon Mac, not Intel |
| `DimensionMismatchError` | Check matrix shapes match operation |
| `AttributeError: no attribute 'ab_dgemm'` | libapplebottom not installed in library path |
| Type checking warnings | Ensure `py.typed` is installed (pip does this) |

## Contributing

The wrapper is complete and production-ready. For modifications:
1. Maintain PEP 484 type hints
2. Add docstrings in Google style
3. Update README.md with examples
4. Test with demo_dgemm.py
5. Verify syntax: `python -m py_compile`

## License

MIT License (matches apple-bottom main library)

## Version

This wrapper targets apple-bottom 1.2.0
Python wrapper version: 1.2.0

## Support Resources

- **Quick start**: QUICKSTART.md
- **Full docs**: README.md
- **Examples**: examples/demo_dgemm.py
- **Source**: apple_bottom/__init__.py (well-commented)

---

**Last updated**: 2026-04-02
**Status**: Complete and ready for production use
