"""
Apple Bottom — FP64-class BLAS for Apple Silicon GPU

A Python ctypes wrapper providing both low-level and high-level APIs
for accelerated matrix operations on Apple Silicon GPUs.

numpy is optional: without it, matrices are passed as sequences of rows
and results come back as lists of rows.
"""

from __future__ import annotations

import ctypes
import os
from ctypes import c_int, c_double, c_bool, c_size_t, c_uint64, c_void_p, c_char_p, POINTER
from pathlib import Path
from typing import Optional, Tuple, Union, ContextManager
import warnings

try:
    import numpy as np
except ImportError:
    np = None

__version__ = "1.2.0"

# =============================================================================
# Constants
# =============================================================================
MAX_ELEMENTS = 2**31 - 1     # AB_MAX_MATRIX_ELEMENTS: rows * cols per matrix
MAX_DIMENSION = 2**31 - 1    # rows or cols alone (a C int)
POOL_CAPACITY = 128
SESSION_CAPACITY = 64


# =============================================================================
# Error Handling
# =============================================================================
class AppleBottomError(Exception):
    """Base exception for apple-bottom library errors."""
    pass


class DeviceNotFoundError(AppleBottomError):
    """Raised when no suitable Apple Silicon GPU is available."""
    pass


class AllocationError(AppleBottomError):
    """Raised when memory allocation fails."""
    pass


class DimensionMismatchError(AppleBottomError):
    """Raised when matrix dimensions are incompatible."""
    pass


class NotUploadedError(AppleBottomError):
    """Raised when attempting operations on non-uploaded matrices."""
    pass


class KernelExecutionError(AppleBottomError):
    """Raised when GPU kernel execution fails."""
    pass


class InvalidArgumentError(AppleBottomError):
    """Raised when invalid arguments are provided."""
    pass


class ShaderCompileError(AppleBottomError):
    """Raised when Metal shader compilation fails."""
    pass


# Status code to exception mapping
_STATUS_TO_EXCEPTION = {
    -1: DeviceNotFoundError,
    -2: AllocationError,
    -3: DimensionMismatchError,
    -4: NotUploadedError,
    -5: KernelExecutionError,
    -6: InvalidArgumentError,
    -7: ShaderCompileError,
    -8: InvalidArgumentError,  # AB_ERROR_TOO_LARGE
}


def _check_status(status: int) -> None:
    """Check a C status code and raise if non-zero."""
    if status != 0:
        exc_cls = _STATUS_TO_EXCEPTION.get(status, AppleBottomError)
        raise exc_cls(f"apple_bottom error code {status}")


# =============================================================================
# Library Loading
# =============================================================================
def _find_library() -> ctypes.CDLL:
    """
    Find and load libapplebottom from standard locations.

    Search order:
    0. $AB_LIBRARY, if set (exact path; no fallback)
    1. ../build/ relative to this file
    2. /usr/local/lib
    3. /opt/homebrew/lib
    4. System library path via ctypes.util.find_library
    """
    if os.environ.get("AB_LIBRARY"):
        return ctypes.CDLL(os.environ["AB_LIBRARY"], use_errno=True)

    candidates = []

    # Try relative to this module
    module_dir = Path(__file__).parent.parent.parent  # python/
    build_dir = module_dir / "build"
    for variant in [
        build_dir / "libapplebottom.dylib",
        build_dir / "libapplebottom.a",
    ]:
        if variant.exists():
            candidates.append(str(variant))

    # Standard system locations
    candidates.extend([
        "/usr/local/lib/libapplebottom.dylib",
        "/opt/homebrew/lib/libapplebottom.dylib",
        "/usr/local/lib/libapplebottom.a",
        "/opt/homebrew/lib/libapplebottom.a",
    ])

    for path in candidates:
        try:
            return ctypes.CDLL(path, use_errno=True)
        except OSError:
            continue

    raise AppleBottomError(
        "Could not find libapplebottom. Install with 'make install' or "
        "set LD_LIBRARY_PATH. Searched: " + ", ".join(candidates)
    )


_lib = _find_library()


# =============================================================================
# C Type Definitions
# =============================================================================
# Opaque struct pointers
ABMatrix = c_void_p
ABSession = c_void_p
ABMemoryPool = c_void_p
ABFuture = c_void_p


# Enum definitions
class ABStatus(ctypes.c_int):
    """Status codes from apple_bottom C API."""
    OK = 0
    ERROR_NO_DEVICE = -1
    ERROR_ALLOC_FAILED = -2
    ERROR_DIMENSION_MISMATCH = -3
    ERROR_NOT_UPLOADED = -4
    ERROR_KERNEL_FAILED = -5
    ERROR_INVALID_ARG = -6
    ERROR_SHADER_COMPILE = -7
    ERROR_TOO_LARGE = -8


class ABTranspose(ctypes.c_int):
    """Transpose operation mode."""
    NO_TRANS = 0
    TRANS = 1
    CONJ_TRANS = 2


class ABStats(ctypes.Structure):
    """Statistics structure."""
    _fields_ = [
        ("upload_time_ms", c_double),
        ("download_time_ms", c_double),
        ("kernel_time_ms", c_double),
        ("dgemm_count", c_uint64),
        ("zgemm_count", c_uint64),
        ("elements_converted", c_uint64),
    ]


# =============================================================================
# Low-Level C Bindings
# =============================================================================
class _LibWrapper:
    """Low-level ctypes wrapper for the C API."""

    def __init__(self, lib: ctypes.CDLL):
        self.lib = lib
        self._setup_functions()

    def _setup_functions(self) -> None:
        """Configure ctypes function signatures."""
        # ABStatus/ABTranspose arguments and results are declared as plain
        # c_int: ctypes returns a c_int *subclass* restype as an unconverted
        # ctypes object, not a Python int.

        # Initialization
        self.ab_init = self.lib.ab_init
        self.ab_init.argtypes = []
        self.ab_init.restype = c_int

        self.ab_shutdown = self.lib.ab_shutdown
        self.ab_shutdown.argtypes = []
        self.ab_shutdown.restype = None

        self.ab_device_name = self.lib.ab_device_name
        self.ab_device_name.argtypes = []
        self.ab_device_name.restype = c_char_p

        self.ab_is_initialized = self.lib.ab_is_initialized
        self.ab_is_initialized.argtypes = []
        self.ab_is_initialized.restype = c_bool

        # Matrix lifecycle
        self.ab_matrix_create = self.lib.ab_matrix_create
        self.ab_matrix_create.argtypes = [c_int, c_int]
        self.ab_matrix_create.restype = ABMatrix

        self.ab_matrix_destroy = self.lib.ab_matrix_destroy
        self.ab_matrix_destroy.argtypes = [ABMatrix]
        self.ab_matrix_destroy.restype = None

        self.ab_matrix_dims = self.lib.ab_matrix_dims
        self.ab_matrix_dims.argtypes = [ABMatrix, POINTER(c_int), POINTER(c_int)]
        self.ab_matrix_dims.restype = None

        self.ab_matrix_count = self.lib.ab_matrix_count
        self.ab_matrix_count.argtypes = [ABMatrix]
        self.ab_matrix_count.restype = c_size_t

        # Data transfer
        self.ab_matrix_upload = self.lib.ab_matrix_upload
        self.ab_matrix_upload.argtypes = [ABMatrix, POINTER(c_double), c_bool]
        self.ab_matrix_upload.restype = c_int

        self.ab_matrix_download = self.lib.ab_matrix_download
        self.ab_matrix_download.argtypes = [ABMatrix, POINTER(c_double), c_bool]
        self.ab_matrix_download.restype = c_int

        self.ab_matrix_zero = self.lib.ab_matrix_zero
        self.ab_matrix_zero.argtypes = [ABMatrix]
        self.ab_matrix_zero.restype = c_int

        self.ab_matrix_copy = self.lib.ab_matrix_copy
        self.ab_matrix_copy.argtypes = [ABMatrix, ABMatrix]
        self.ab_matrix_copy.restype = c_int

        # Memory pool
        self.ab_pool_create = self.lib.ab_pool_create
        self.ab_pool_create.argtypes = [c_size_t]
        self.ab_pool_create.restype = ABMemoryPool

        self.ab_pool_destroy = self.lib.ab_pool_destroy
        self.ab_pool_destroy.argtypes = [ABMemoryPool]
        self.ab_pool_destroy.restype = None

        self.ab_pool_get_matrix = self.lib.ab_pool_get_matrix
        self.ab_pool_get_matrix.argtypes = [ABMemoryPool, c_int, c_int]
        self.ab_pool_get_matrix.restype = ABMatrix

        self.ab_pool_reset = self.lib.ab_pool_reset
        self.ab_pool_reset.argtypes = [ABMemoryPool]
        self.ab_pool_reset.restype = None

        # Async API
        self.ab_dgemm_async = self.lib.ab_dgemm_async
        self.ab_dgemm_async.argtypes = [ABMatrix, ABMatrix, ABMatrix]
        self.ab_dgemm_async.restype = ABFuture

        self.ab_zgemm_async = self.lib.ab_zgemm_async
        self.ab_zgemm_async.argtypes = [ABMatrix, ABMatrix, ABMatrix, ABMatrix, ABMatrix, ABMatrix]
        self.ab_zgemm_async.restype = ABFuture

        self.ab_future_wait = self.lib.ab_future_wait
        self.ab_future_wait.argtypes = [ABFuture]
        self.ab_future_wait.restype = c_int

        self.ab_future_is_ready = self.lib.ab_future_is_ready
        self.ab_future_is_ready.argtypes = [ABFuture]
        self.ab_future_is_ready.restype = c_bool

        self.ab_future_status = self.lib.ab_future_status
        self.ab_future_status.argtypes = [ABFuture]
        self.ab_future_status.restype = c_int

        self.ab_future_destroy = self.lib.ab_future_destroy
        self.ab_future_destroy.argtypes = [ABFuture]
        self.ab_future_destroy.restype = None

        # BLAS operations
        self.ab_dgemm = self.lib.ab_dgemm
        self.ab_dgemm.argtypes = [ABMatrix, ABMatrix, ABMatrix]
        self.ab_dgemm.restype = c_int

        self.ab_dgemm_scaled = self.lib.ab_dgemm_scaled
        self.ab_dgemm_scaled.argtypes = [c_double, ABMatrix, ABMatrix, c_double, ABMatrix]
        self.ab_dgemm_scaled.restype = c_int

        self.ab_zgemm = self.lib.ab_zgemm
        self.ab_zgemm.argtypes = [ABMatrix, ABMatrix, ABMatrix, ABMatrix, ABMatrix, ABMatrix]
        self.ab_zgemm.restype = c_int

        self.ab_zgemm_ex = self.lib.ab_zgemm_ex
        self.ab_zgemm_ex.argtypes = [
            c_int, c_int,
            ABMatrix, ABMatrix, ABMatrix, ABMatrix, ABMatrix, ABMatrix
        ]
        self.ab_zgemm_ex.restype = c_int

        self.ab_dsyrk = self.lib.ab_dsyrk
        self.ab_dsyrk.argtypes = [ABMatrix, ABMatrix]
        self.ab_dsyrk.restype = c_int

        self.ab_zherk = self.lib.ab_zherk
        self.ab_zherk.argtypes = [ABMatrix, ABMatrix, ABMatrix, ABMatrix]
        self.ab_zherk.restype = c_int

        # Element-wise operations
        self.ab_matrix_add = self.lib.ab_matrix_add
        self.ab_matrix_add.argtypes = [ABMatrix, ABMatrix, ABMatrix]
        self.ab_matrix_add.restype = c_int

        self.ab_matrix_sub = self.lib.ab_matrix_sub
        self.ab_matrix_sub.argtypes = [ABMatrix, ABMatrix, ABMatrix]
        self.ab_matrix_sub.restype = c_int

        self.ab_matrix_scale = self.lib.ab_matrix_scale
        self.ab_matrix_scale.argtypes = [c_double, ABMatrix]
        self.ab_matrix_scale.restype = c_int

        # Session API
        self.ab_session_create = self.lib.ab_session_create
        self.ab_session_create.argtypes = []
        self.ab_session_create.restype = ABSession

        self.ab_session_destroy = self.lib.ab_session_destroy
        self.ab_session_destroy.argtypes = [ABSession]
        self.ab_session_destroy.restype = None

        self.ab_session_add = self.lib.ab_session_add
        self.ab_session_add.argtypes = [ABSession, c_char_p, c_int, c_int]
        self.ab_session_add.restype = c_int

        self.ab_session_get = self.lib.ab_session_get
        self.ab_session_get.argtypes = [ABSession, c_char_p]
        self.ab_session_get.restype = ABMatrix

        self.ab_session_upload = self.lib.ab_session_upload
        self.ab_session_upload.argtypes = [ABSession, c_char_p, POINTER(c_double)]
        self.ab_session_upload.restype = c_int

        self.ab_session_download = self.lib.ab_session_download
        self.ab_session_download.argtypes = [ABSession, c_char_p, POINTER(c_double)]
        self.ab_session_download.restype = c_int

        self.ab_session_dgemm = self.lib.ab_session_dgemm
        self.ab_session_dgemm.argtypes = [ABSession, c_char_p, c_char_p, c_char_p]
        self.ab_session_dgemm.restype = c_int

        self.ab_session_zgemm = self.lib.ab_session_zgemm
        self.ab_session_zgemm.argtypes = [
            ABSession, c_char_p, c_char_p, c_char_p, c_char_p, c_char_p, c_char_p
        ]
        self.ab_session_zgemm.restype = c_int

        # Statistics
        self.ab_get_stats = self.lib.ab_get_stats
        self.ab_get_stats.argtypes = []
        self.ab_get_stats.restype = ABStats

        self.ab_reset_stats = self.lib.ab_reset_stats
        self.ab_reset_stats.argtypes = []
        self.ab_reset_stats.restype = None

        self.ab_print_stats = self.lib.ab_print_stats
        self.ab_print_stats.argtypes = []
        self.ab_print_stats.restype = None

        # Utility
        self.ab_status_string = self.lib.ab_status_string
        self.ab_status_string.argtypes = [c_int]
        self.ab_status_string.restype = c_char_p


_libwrap = _LibWrapper(_lib)


# =============================================================================
# Host Data: numpy arrays, or sequences of rows when numpy is not installed
# =============================================================================
def _shape(data) -> Tuple[int, int]:
    """(rows, cols) of host data: a 2-D numpy array or a sequence of rows."""
    if np is not None and isinstance(data, np.ndarray):
        if data.ndim != 2:
            raise DimensionMismatchError(f"Expected a 2-D array, got shape {data.shape}")
        return data.shape
    return len(data), (len(data[0]) if len(data) else 0)


def _host_data(data, shape: Tuple[int, int]):
    """
    Return a pointer to `data` as row-major float64 for a GPU matrix of
    `shape`. `data` is a float64 numpy array or a sequence of rows.
    """
    if np is not None and isinstance(data, np.ndarray):
        if data.shape != shape:
            raise DimensionMismatchError(
                f"Data shape {data.shape} doesn't match matrix {shape}"
            )
        if data.dtype != np.float64:
            raise ValueError(f"Expected float64, got {data.dtype}")
        # The pointer keeps a reference to the (possibly copied) array.
        return np.ascontiguousarray(data).ctypes.data_as(POINTER(c_double))

    rows, cols = shape
    if len(data) != rows or any(len(row) != cols for row in data):
        raise DimensionMismatchError(f"Data doesn't match matrix shape {shape}")
    return (c_double * (rows * cols))(*(x for row in data for x in row))


def _host_result(shape: Tuple[int, int], fill):
    """
    Allocate a row-major float64 buffer of `shape`, check the status that
    fill(pointer) returns, and return the buffer: a numpy array, or a list
    of rows when numpy is not installed.
    """
    if np is not None:
        out = np.empty(shape, dtype=np.float64, order='C')
        _check_status(fill(out.ctypes.data_as(POINTER(c_double))))
        return out

    rows, cols = shape
    flat = (c_double * (rows * cols))()
    _check_status(fill(flat))
    return [flat[i * cols:(i + 1) * cols] for i in range(rows)]


# =============================================================================
# Handle Ownership
# =============================================================================
class _Handle:
    """
    Base for wrappers around one C handle. An owned handle is freed exactly
    once: by close(), on leaving a `with` block, or at garbage collection,
    whichever comes first.
    """

    _ptr = None   # C handle; None once closed or if creation failed
    _free = None  # ctypes destroy function; None when another object owns _ptr
    _generation = 0  # bumped when the handles this owner lent out are taken back

    def close(self) -> None:
        """Release the handle now. Calling close() again does nothing."""
        ptr, self._ptr = self._ptr, None
        if ptr and self._free:
            self._free(ptr)

    def __del__(self):
        self.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.close()

    def _live(self):
        """Return the C handle, or raise if it has been closed."""
        if not self._ptr:
            raise InvalidArgumentError(f"{type(self).__name__} is closed")
        return self._ptr


# =============================================================================
# High-Level API: Matrix Class
# =============================================================================
class Matrix(_Handle):
    """
    Represents a GPU-resident matrix.

    Handles creation, data transfer, and cleanup (close(), `with`, or
    garbage collection). A matrix obtained from MemoryPool.get_matrix() or
    Session.get() belongs to that pool or session, which frees it; closing
    such a matrix only detaches the wrapper. A pool matrix can no longer be
    used once the pool is reset.
    """

    _free = _libwrap.ab_matrix_destroy
    _owner = None  # MemoryPool or Session that owns a borrowed handle
    _owner_generation = 0  # owner's _generation when the handle was borrowed

    def __init__(self, rows: int, cols: int):
        """
        Create a new GPU matrix.

        Args:
            rows: Number of rows
            cols: Number of columns

        Raises:
            InvalidArgumentError: If rows * cols exceeds MAX_ELEMENTS
            DeviceNotFoundError: If GPU device is unavailable
            AllocationError: If GPU memory allocation fails
        """
        if rows > MAX_DIMENSION or cols > MAX_DIMENSION or rows * cols > MAX_ELEMENTS:
            raise InvalidArgumentError(
                f"Matrix dimensions ({rows}, {cols}) exceed maximum {MAX_ELEMENTS} elements"
            )

        self._ptr = _libwrap.ab_matrix_create(rows, cols)
        if not self._ptr:
            raise AllocationError("Failed to allocate GPU matrix")
        self._rows = rows
        self._cols = cols

    @classmethod
    def _borrow(cls, handle, rows: int, cols: int, owner: _Handle) -> Matrix:
        """Wrap a handle that `owner` (a MemoryPool or Session) frees."""
        m = cls.__new__(cls)
        m._ptr, m._rows, m._cols = handle, rows, cols
        m._owner = owner  # keeps the owner, and so the handle, alive
        m._owner_generation = owner._generation
        m._free = None
        return m

    def _live(self):
        if self._owner is not None:
            self._owner._live()  # raises once the pool or session is closed
            # After MemoryPool.reset() the pool may hand this handle out again
            # with another shape; this wrapper's shape would then overflow
            # host buffers in download().
            if self._owner._generation != self._owner_generation:
                raise InvalidArgumentError("Matrix was returned to its MemoryPool by reset()")
        return super()._live()

    @property
    def _matrix(self):
        """The live C handle, for direct _libwrap calls."""
        return self._live()

    @property
    def shape(self) -> Tuple[int, int]:
        """Get matrix dimensions as (rows, cols)."""
        return (self._rows, self._cols)

    @property
    def rows(self) -> int:
        """Get number of rows."""
        return self._rows

    @property
    def cols(self) -> int:
        """Get number of columns."""
        return self._cols

    @property
    def size(self) -> int:
        """Get total number of elements."""
        return self._rows * self._cols

    def upload(self, data: np.ndarray, parallel: bool = True) -> None:
        """
        Upload data to GPU.

        Args:
            data: numpy array of shape (rows, cols) with dtype float64, or
                a sequence of `rows` rows of `cols` numbers
            parallel: Use parallel upload (faster but uses more CPU resources)

        Raises:
            DimensionMismatchError: If data shape doesn't match matrix
            ValueError: If a numpy array is not float64
        """
        status = _libwrap.ab_matrix_upload(
            self._matrix,
            _host_data(data, self.shape),
            parallel
        )
        _check_status(status)

    def download(self, parallel: bool = True) -> np.ndarray:
        """
        Download data from GPU.

        Args:
            parallel: Use parallel download

        Returns:
            numpy array of shape (rows, cols) with dtype float64, or a list
            of rows if numpy is not installed
        """
        handle = self._matrix
        return _host_result(
            self.shape,
            lambda out: _libwrap.ab_matrix_download(handle, out, parallel)
        )

    def zero(self) -> None:
        """Fill matrix with zeros on GPU."""
        status = _libwrap.ab_matrix_zero(self._matrix)
        _check_status(status)

    def copy_from(self, src: 'Matrix') -> None:
        """
        Copy data from another matrix.

        Args:
            src: Source matrix (must have same shape)

        Raises:
            DimensionMismatchError: If shapes don't match
        """
        if src.shape != self.shape:
            raise DimensionMismatchError(f"Cannot copy {src.shape} to {self.shape}")
        status = _libwrap.ab_matrix_copy(src._matrix, self._matrix)
        _check_status(status)


# =============================================================================
# High-Level API: MemoryPool Class
# =============================================================================
class MemoryPool(_Handle):
    """
    GPU memory pool for reusing matrix allocations.

    Reduces allocation overhead in iterative codes. Closing the pool
    (close(), `with`, or garbage collection) frees every matrix it handed
    out; it stays alive while any of those matrices is referenced.
    """

    _free = _libwrap.ab_pool_destroy

    def __init__(self, size_hint: int = 16):
        """
        Create a memory pool.

        Args:
            size_hint: Estimated number of matrices to pre-allocate
        """
        self._ptr = _libwrap.ab_pool_create(size_hint)
        if not self._ptr:
            raise AllocationError("Failed to create memory pool")

    def get_matrix(self, rows: int, cols: int) -> Matrix:
        """
        Get a matrix from the pool.

        Args:
            rows: Number of rows
            cols: Number of columns

        Returns:
            Matrix object from pool

        Raises:
            AllocationError: If pool is at capacity
        """
        mat = _libwrap.ab_pool_get_matrix(self._live(), rows, cols)
        if not mat:
            raise AllocationError("Memory pool at capacity (max 128 matrices)")
        return Matrix._borrow(mat, rows, cols, self)

    def reset(self) -> None:
        """
        Mark all matrices in pool as available for reuse. Matrices obtained
        before the reset raise InvalidArgumentError if used again.
        """
        _libwrap.ab_pool_reset(self._live())
        self._generation += 1


# =============================================================================
# High-Level API: Session Class
# =============================================================================
class Session(_Handle):
    """
    Named matrix session for organizing related matrices.

    Simplifies management of multiple matrices and operations. Closing the
    session (close(), `with`, or garbage collection) frees its matrices.
    """

    _free = _libwrap.ab_session_destroy

    def __init__(self):
        """Create a new session."""
        self._ptr = _libwrap.ab_session_create()
        if not self._ptr:
            raise AllocationError("Failed to create session")
        self._matrices = {}

    def add(self, name: str, rows: int, cols: int) -> None:
        """
        Add a named matrix to the session.

        Args:
            name: Matrix name (string identifier)
            rows: Number of rows
            cols: Number of columns

        Raises:
            InvalidArgumentError: If name already exists
        """
        if name in self._matrices:
            raise InvalidArgumentError(f"Matrix '{name}' already exists in session")

        name_bytes = name.encode('utf-8')
        status = _libwrap.ab_session_add(self._live(), name_bytes, rows, cols)
        _check_status(status)
        self._matrices[name] = (rows, cols)

    def get(self, name: str) -> Matrix:
        """
        Get a matrix from the session by name.

        Args:
            name: Matrix name

        Returns:
            Matrix wrapper
        """
        if name not in self._matrices:
            raise KeyError(f"Matrix '{name}' not found in session")

        mat = _libwrap.ab_session_get(self._live(), name.encode('utf-8'))
        if not mat:
            raise KeyError(f"Matrix '{name}' not found in session")

        rows, cols = self._matrices[name]
        return Matrix._borrow(mat, rows, cols, self)

    def upload(self, name: str, data: np.ndarray) -> None:
        """
        Upload data to a named matrix.

        Args:
            name: Matrix name
            data: numpy float64 array or sequence of rows (must match
                declared shape)
        """
        if name not in self._matrices:
            raise KeyError(f"Matrix '{name}' not found in session")

        status = _libwrap.ab_session_upload(
            self._live(),
            name.encode('utf-8'),
            _host_data(data, self._matrices[name])
        )
        _check_status(status)

    def download(self, name: str) -> np.ndarray:
        """
        Download data from a named matrix.

        Args:
            name: Matrix name

        Returns:
            numpy array, or a list of rows if numpy is not installed
        """
        if name not in self._matrices:
            raise KeyError(f"Matrix '{name}' not found in session")

        session, key = self._live(), name.encode('utf-8')
        return _host_result(
            self._matrices[name],
            lambda out: _libwrap.ab_session_download(session, key, out)
        )

    def dgemm(self, A: str, B: str, C: str) -> None:
        """
        Perform double-precision matrix multiplication: C = A @ B.

        Args:
            A: Name of left operand matrix
            B: Name of right operand matrix
            C: Name of output matrix
        """
        status = _libwrap.ab_session_dgemm(
            self._live(),
            A.encode('utf-8'),
            B.encode('utf-8'),
            C.encode('utf-8')
        )
        _check_status(status)

    def zgemm(self, Ar: str, Ai: str, Br: str, Bi: str, Cr: str, Ci: str) -> None:
        """
        Perform complex matrix multiplication (separate real/imaginary).

        Args:
            Ar, Ai: Real and imaginary parts of A
            Br, Bi: Real and imaginary parts of B
            Cr, Ci: Real and imaginary parts of C (output)
        """
        status = _libwrap.ab_session_zgemm(
            self._live(),
            Ar.encode('utf-8'),
            Ai.encode('utf-8'),
            Br.encode('utf-8'),
            Bi.encode('utf-8'),
            Cr.encode('utf-8'),
            Ci.encode('utf-8')
        )
        _check_status(status)


# =============================================================================
# Functional API
# =============================================================================
def init() -> None:
    """Initialize the apple_bottom library and GPU device."""
    status = _libwrap.ab_init()
    _check_status(status)


def shutdown() -> None:
    """Shutdown the apple_bottom library."""
    _libwrap.ab_shutdown()


def get_device_name() -> str:
    """Get the name of the GPU device."""
    name_ptr = _libwrap.ab_device_name()
    if not name_ptr:
        return "Unknown"
    return name_ptr.decode('utf-8')


def is_initialized() -> bool:
    """Check if library is initialized."""
    return _libwrap.ab_is_initialized()


def dgemm(A: np.ndarray, B: np.ndarray, alpha: float = 1.0, beta: float = 0.0) -> np.ndarray:
    """
    Perform double-precision matrix multiplication with optional scaling.

    Computes: C = alpha * (A @ B) + beta * C

    This is a convenience function that handles allocation, upload, compute,
    and download in one call.

    Args:
        A: Left operand, shape (m, k), dtype float64 (or a sequence of rows)
        B: Right operand, shape (k, n), dtype float64 (or a sequence of rows)
        alpha: Scalar multiplier (default 1.0)
        beta: Scalar for C accumulation (default 0.0)

    Returns:
        Result matrix, shape (m, n), dtype float64 (a list of rows if numpy
        is not installed)

    Raises:
        DimensionMismatchError: If A.shape[1] != B.shape[0]
        ValueError: If numpy inputs are not float64
    """
    (m, k), (k_b, n) = _shape(A), _shape(B)
    if k != k_b:
        raise DimensionMismatchError(
            f"Cannot multiply {(m, k)} and {(k_b, n)}: inner dimensions mismatch"
        )

    # Create GPU matrices
    gpu_A = Matrix(m, k)
    gpu_B = Matrix(k, n)
    gpu_C = Matrix(m, n)

    # Upload
    gpu_A.upload(A, parallel=True)
    gpu_B.upload(B, parallel=True)
    gpu_C.zero()

    # Compute
    if alpha == 1.0 and beta == 0.0:
        status = _libwrap.ab_dgemm(gpu_A._matrix, gpu_B._matrix, gpu_C._matrix)
    else:
        status = _libwrap.ab_dgemm_scaled(
            alpha, gpu_A._matrix, gpu_B._matrix, beta, gpu_C._matrix
        )
    _check_status(status)

    # Download
    return gpu_C.download(parallel=True)


def zgemm(Ar: np.ndarray, Ai: np.ndarray, Br: np.ndarray, Bi: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Perform complex matrix multiplication (real and imaginary parts separate).

    Args:
        Ar, Ai: Real and imaginary parts of left operand
        Br, Bi: Real and imaginary parts of right operand
        (float64 numpy arrays or sequences of rows)

    Returns:
        Tuple (Cr, Ci) of real and imaginary parts of result
    """
    m, k = _shape(Ar)
    n = _shape(Br)[1]

    # Validate dimensions
    for arr, name in [(Ar, "Ar"), (Ai, "Ai"), (Br, "Br"), (Bi, "Bi")]:
        if 0 in _shape(arr):
            raise DimensionMismatchError(f"{name} has zero dimension")

    # Create GPU matrices
    gpu_Ar = Matrix(m, k)
    gpu_Ai = Matrix(m, k)
    gpu_Br = Matrix(k, n)
    gpu_Bi = Matrix(k, n)
    gpu_Cr = Matrix(m, n)
    gpu_Ci = Matrix(m, n)

    # Upload
    gpu_Ar.upload(Ar, parallel=True)
    gpu_Ai.upload(Ai, parallel=True)
    gpu_Br.upload(Br, parallel=True)
    gpu_Bi.upload(Bi, parallel=True)
    gpu_Cr.zero()
    gpu_Ci.zero()

    # Compute
    status = _libwrap.ab_zgemm(
        gpu_Ar._matrix, gpu_Ai._matrix,
        gpu_Br._matrix, gpu_Bi._matrix,
        gpu_Cr._matrix, gpu_Ci._matrix
    )
    _check_status(status)

    # Download
    Cr = gpu_Cr.download(parallel=True)
    Ci = gpu_Ci.download(parallel=True)
    return (Cr, Ci)


def dsyrk(A: np.ndarray) -> np.ndarray:
    """
    Perform symmetric rank-k update: C = A @ A.T.

    Args:
        A: Input matrix, shape (m, n), dtype float64 (or a sequence of rows)

    Returns:
        Result matrix, shape (m, m), dtype float64
    """
    m, n = _shape(A)

    gpu_A = Matrix(m, n)
    gpu_C = Matrix(m, m)

    gpu_A.upload(A, parallel=True)
    gpu_C.zero()

    status = _libwrap.ab_dsyrk(gpu_A._matrix, gpu_C._matrix)
    _check_status(status)

    return gpu_C.download(parallel=True)


def get_stats() -> dict:
    """Get runtime statistics."""
    stats = _libwrap.ab_get_stats()
    return {
        'upload_time_ms': stats.upload_time_ms,
        'download_time_ms': stats.download_time_ms,
        'kernel_time_ms': stats.kernel_time_ms,
        'dgemm_count': stats.dgemm_count,
        'zgemm_count': stats.zgemm_count,
        'elements_converted': stats.elements_converted,
    }


def reset_stats() -> None:
    """Reset runtime statistics."""
    _libwrap.ab_reset_stats()


def print_stats() -> None:
    """Print runtime statistics to stdout."""
    _libwrap.ab_print_stats()


def status_string(status: int) -> str:
    """Get human-readable status message."""
    ptr = _libwrap.ab_status_string(status)
    if ptr:
        return ptr.decode('utf-8')
    return f"Unknown status {status}"


# =============================================================================
# Public API
# =============================================================================
__all__ = [
    # Constants
    'MAX_DIMENSION',
    'MAX_ELEMENTS',
    'POOL_CAPACITY',
    'SESSION_CAPACITY',
    # Exceptions
    'AppleBottomError',
    'DeviceNotFoundError',
    'AllocationError',
    'DimensionMismatchError',
    'NotUploadedError',
    'KernelExecutionError',
    'InvalidArgumentError',
    'ShaderCompileError',
    # Classes
    'Matrix',
    'MemoryPool',
    'Session',
    # Functional API
    'init',
    'shutdown',
    'get_device_name',
    'is_initialized',
    'dgemm',
    'zgemm',
    'dsyrk',
    'get_stats',
    'reset_stats',
    'print_stats',
    'status_string',
]
