"""
Apple Bottom — FP64-class BLAS for Apple Silicon GPU

A Python ctypes wrapper providing both low-level and high-level APIs
for accelerated matrix operations on Apple Silicon GPUs.
"""

import ctypes
import numpy as np
from ctypes import c_int, c_double, c_bool, c_size_t, c_uint64, c_void_p, c_char_p, POINTER
from pathlib import Path
from typing import Optional, Tuple, Union, ContextManager
import warnings

__version__ = "1.2.0"

# =============================================================================
# Constants
# =============================================================================
MAX_DIMENSION = 46340
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
    1. ../build/ relative to this file
    2. /usr/local/lib
    3. /opt/homebrew/lib
    4. System library path via ctypes.util.find_library
    """
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
        # Initialization
        self.ab_init = self.lib.ab_init
        self.ab_init.argtypes = []
        self.ab_init.restype = ABStatus

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
        self.ab_matrix_upload.restype = ABStatus

        self.ab_matrix_download = self.lib.ab_matrix_download
        self.ab_matrix_download.argtypes = [ABMatrix, POINTER(c_double), c_bool]
        self.ab_matrix_download.restype = ABStatus

        self.ab_matrix_zero = self.lib.ab_matrix_zero
        self.ab_matrix_zero.argtypes = [ABMatrix]
        self.ab_matrix_zero.restype = ABStatus

        self.ab_matrix_copy = self.lib.ab_matrix_copy
        self.ab_matrix_copy.argtypes = [ABMatrix, ABMatrix]
        self.ab_matrix_copy.restype = ABStatus

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
        self.ab_future_wait.restype = ABStatus

        self.ab_future_is_ready = self.lib.ab_future_is_ready
        self.ab_future_is_ready.argtypes = [ABFuture]
        self.ab_future_is_ready.restype = c_bool

        self.ab_future_status = self.lib.ab_future_status
        self.ab_future_status.argtypes = [ABFuture]
        self.ab_future_status.restype = ABStatus

        self.ab_future_destroy = self.lib.ab_future_destroy
        self.ab_future_destroy.argtypes = [ABFuture]
        self.ab_future_destroy.restype = None

        # BLAS operations
        self.ab_dgemm = self.lib.ab_dgemm
        self.ab_dgemm.argtypes = [ABMatrix, ABMatrix, ABMatrix]
        self.ab_dgemm.restype = ABStatus

        self.ab_dgemm_scaled = self.lib.ab_dgemm_scaled
        self.ab_dgemm_scaled.argtypes = [c_double, ABMatrix, ABMatrix, c_double, ABMatrix]
        self.ab_dgemm_scaled.restype = ABStatus

        self.ab_zgemm = self.lib.ab_zgemm
        self.ab_zgemm.argtypes = [ABMatrix, ABMatrix, ABMatrix, ABMatrix, ABMatrix, ABMatrix]
        self.ab_zgemm.restype = ABStatus

        self.ab_zgemm_ex = self.lib.ab_zgemm_ex
        self.ab_zgemm_ex.argtypes = [
            ABTranspose, ABTranspose,
            ABMatrix, ABMatrix, ABMatrix, ABMatrix, ABMatrix, ABMatrix
        ]
        self.ab_zgemm_ex.restype = ABStatus

        self.ab_dsyrk = self.lib.ab_dsyrk
        self.ab_dsyrk.argtypes = [ABMatrix, ABMatrix]
        self.ab_dsyrk.restype = ABStatus

        self.ab_zherk = self.lib.ab_zherk
        self.ab_zherk.argtypes = [ABMatrix, ABMatrix, ABMatrix, ABMatrix]
        self.ab_zherk.restype = ABStatus

        # Element-wise operations
        self.ab_matrix_add = self.lib.ab_matrix_add
        self.ab_matrix_add.argtypes = [ABMatrix, ABMatrix, ABMatrix]
        self.ab_matrix_add.restype = ABStatus

        self.ab_matrix_sub = self.lib.ab_matrix_sub
        self.ab_matrix_sub.argtypes = [ABMatrix, ABMatrix, ABMatrix]
        self.ab_matrix_sub.restype = ABStatus

        self.ab_matrix_scale = self.lib.ab_matrix_scale
        self.ab_matrix_scale.argtypes = [c_double, ABMatrix]
        self.ab_matrix_scale.restype = ABStatus

        # Session API
        self.ab_session_create = self.lib.ab_session_create
        self.ab_session_create.argtypes = []
        self.ab_session_create.restype = ABSession

        self.ab_session_destroy = self.lib.ab_session_destroy
        self.ab_session_destroy.argtypes = [ABSession]
        self.ab_session_destroy.restype = None

        self.ab_session_add = self.lib.ab_session_add
        self.ab_session_add.argtypes = [ABSession, c_char_p, c_int, c_int]
        self.ab_session_add.restype = ABStatus

        self.ab_session_get = self.lib.ab_session_get
        self.ab_session_get.argtypes = [ABSession, c_char_p]
        self.ab_session_get.restype = ABMatrix

        self.ab_session_upload = self.lib.ab_session_upload
        self.ab_session_upload.argtypes = [ABSession, c_char_p, POINTER(c_double)]
        self.ab_session_upload.restype = ABStatus

        self.ab_session_download = self.lib.ab_session_download
        self.ab_session_download.argtypes = [ABSession, c_char_p, POINTER(c_double)]
        self.ab_session_download.restype = ABStatus

        self.ab_session_dgemm = self.lib.ab_session_dgemm
        self.ab_session_dgemm.argtypes = [ABSession, c_char_p, c_char_p, c_char_p]
        self.ab_session_dgemm.restype = ABStatus

        self.ab_session_zgemm = self.lib.ab_session_zgemm
        self.ab_session_zgemm.argtypes = [
            ABSession, c_char_p, c_char_p, c_char_p, c_char_p, c_char_p, c_char_p
        ]
        self.ab_session_zgemm.restype = ABStatus

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
        self.ab_status_string.argtypes = [ABStatus]
        self.ab_status_string.restype = c_char_p


_libwrap = _LibWrapper(_lib)


# =============================================================================
# High-Level API: Matrix Class
# =============================================================================
class Matrix:
    """
    Represents a GPU-resident matrix.

    Handles creation, data transfer, and automatic cleanup.
    """

    def __init__(self, rows: int, cols: int):
        """
        Create a new GPU matrix.

        Args:
            rows: Number of rows
            cols: Number of columns

        Raises:
            InvalidArgumentError: If dimensions exceed MAX_DIMENSION
            DeviceNotFoundError: If GPU device is unavailable
            AllocationError: If GPU memory allocation fails
        """
        if rows > MAX_DIMENSION or cols > MAX_DIMENSION:
            raise InvalidArgumentError(
                f"Matrix dimensions ({rows}, {cols}) exceed maximum {MAX_DIMENSION}"
            )

        self._matrix = _libwrap.ab_matrix_create(rows, cols)
        if not self._matrix:
            raise AllocationError("Failed to allocate GPU matrix")
        self._rows = rows
        self._cols = cols

    def __del__(self):
        """Automatically destroy GPU matrix on garbage collection."""
        if hasattr(self, '_matrix') and self._matrix:
            _libwrap.ab_matrix_destroy(self._matrix)

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
            data: numpy array of shape (rows, cols) with dtype float64
            parallel: Use parallel upload (faster but uses more CPU resources)

        Raises:
            DimensionMismatchError: If data shape doesn't match matrix
            ValueError: If data is not contiguous float64
        """
        if data.shape != (self._rows, self._cols):
            raise DimensionMismatchError(
                f"Data shape {data.shape} doesn't match matrix {self.shape}"
            )
        if data.dtype != np.float64:
            raise ValueError(f"Expected float64, got {data.dtype}")
        if not data.flags['C_CONTIGUOUS']:
            data = np.ascontiguousarray(data)

        status = _libwrap.ab_matrix_upload(
            self._matrix,
            data.ctypes.data_as(POINTER(c_double)),
            parallel
        )
        _check_status(status)

    def download(self, parallel: bool = True) -> np.ndarray:
        """
        Download data from GPU.

        Args:
            parallel: Use parallel download

        Returns:
            numpy array of shape (rows, cols) with dtype float64
        """
        result = np.empty((self._rows, self._cols), dtype=np.float64, order='C')
        status = _libwrap.ab_matrix_download(
            self._matrix,
            result.ctypes.data_as(POINTER(c_double)),
            parallel
        )
        _check_status(status)
        return result

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
class MemoryPool:
    """
    GPU memory pool for reusing matrix allocations.

    Reduces allocation overhead in iterative codes.
    """

    def __init__(self, size_hint: int = 16):
        """
        Create a memory pool.

        Args:
            size_hint: Estimated number of matrices to pre-allocate
        """
        self._pool = _libwrap.ab_pool_create(size_hint)
        if not self._pool:
            raise AllocationError("Failed to create memory pool")

    def __del__(self):
        """Destroy pool on garbage collection."""
        if hasattr(self, '_pool') and self._pool:
            _libwrap.ab_pool_destroy(self._pool)

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
        mat = _libwrap.ab_pool_get_matrix(self._pool, rows, cols)
        if not mat:
            raise AllocationError("Memory pool at capacity (max 128 matrices)")

        # Wrap the C pointer in our Matrix class
        # Note: We don't own this pointer, pool does
        wrapper = Matrix.__new__(Matrix)
        wrapper._matrix = mat
        wrapper._rows = rows
        wrapper._cols = cols
        return wrapper

    def reset(self) -> None:
        """Mark all matrices in pool as available for reuse."""
        _libwrap.ab_pool_reset(self._pool)


# =============================================================================
# High-Level API: Session Class
# =============================================================================
class Session:
    """
    Named matrix session for organizing related matrices.

    Simplifies management of multiple matrices and operations.
    """

    def __init__(self):
        """Create a new session."""
        self._session = _libwrap.ab_session_create()
        if not self._session:
            raise AllocationError("Failed to create session")
        self._matrices = {}

    def __del__(self):
        """Destroy session on garbage collection."""
        if hasattr(self, '_session') and self._session:
            _libwrap.ab_session_destroy(self._session)

    def __enter__(self) -> 'Session':
        """Context manager entry."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        """Context manager exit."""
        self.__del__()

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
        status = _libwrap.ab_session_add(self._session, name_bytes, rows, cols)
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

        mat = _libwrap.ab_session_get(self._session, name.encode('utf-8'))
        if not mat:
            raise KeyError(f"Matrix '{name}' not found in session")

        rows, cols = self._matrices[name]
        wrapper = Matrix.__new__(Matrix)
        wrapper._matrix = mat
        wrapper._rows = rows
        wrapper._cols = cols
        return wrapper

    def upload(self, name: str, data: np.ndarray) -> None:
        """
        Upload data to a named matrix.

        Args:
            name: Matrix name
            data: numpy array (must match declared shape)
        """
        if name not in self._matrices:
            raise KeyError(f"Matrix '{name}' not found in session")

        rows, cols = self._matrices[name]
        if data.shape != (rows, cols):
            raise DimensionMismatchError(
                f"Data shape {data.shape} doesn't match matrix ({rows}, {cols})"
            )
        if data.dtype != np.float64:
            raise ValueError(f"Expected float64, got {data.dtype}")
        if not data.flags['C_CONTIGUOUS']:
            data = np.ascontiguousarray(data)

        status = _libwrap.ab_session_upload(
            self._session,
            name.encode('utf-8'),
            data.ctypes.data_as(POINTER(c_double))
        )
        _check_status(status)

    def download(self, name: str) -> np.ndarray:
        """
        Download data from a named matrix.

        Args:
            name: Matrix name

        Returns:
            numpy array
        """
        if name not in self._matrices:
            raise KeyError(f"Matrix '{name}' not found in session")

        rows, cols = self._matrices[name]
        result = np.empty((rows, cols), dtype=np.float64, order='C')

        status = _libwrap.ab_session_download(
            self._session,
            name.encode('utf-8'),
            result.ctypes.data_as(POINTER(c_double))
        )
        _check_status(status)
        return result

    def dgemm(self, A: str, B: str, C: str) -> None:
        """
        Perform double-precision matrix multiplication: C = A @ B.

        Args:
            A: Name of left operand matrix
            B: Name of right operand matrix
            C: Name of output matrix
        """
        status = _libwrap.ab_session_dgemm(
            self._session,
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
            self._session,
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
        A: Left operand, shape (m, k), dtype float64
        B: Right operand, shape (k, n), dtype float64
        alpha: Scalar multiplier (default 1.0)
        beta: Scalar for C accumulation (default 0.0)

    Returns:
        Result matrix, shape (m, n), dtype float64

    Raises:
        DimensionMismatchError: If A.shape[1] != B.shape[0]
        ValueError: If inputs are not float64
    """
    if A.dtype != np.float64 or B.dtype != np.float64:
        raise ValueError("Both inputs must be float64")
    if A.shape[1] != B.shape[0]:
        raise DimensionMismatchError(
            f"Cannot multiply {A.shape} and {B.shape}: inner dimensions mismatch"
        )

    A = np.ascontiguousarray(A)
    B = np.ascontiguousarray(B)

    m, k = A.shape
    n = B.shape[1]

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

    Returns:
        Tuple (Cr, Ci) of real and imaginary parts of result
    """
    if Ar.dtype != np.float64 or Ai.dtype != np.float64:
        raise ValueError("A parts must be float64")
    if Br.dtype != np.float64 or Bi.dtype != np.float64:
        raise ValueError("B parts must be float64")

    m, k = Ar.shape
    n = Br.shape[1]

    # Validate dimensions
    for arr, name in [(Ar, "Ar"), (Ai, "Ai"), (Br, "Br"), (Bi, "Bi")]:
        if arr.shape[0] == 0 or arr.shape[1] == 0:
            raise DimensionMismatchError(f"{name} has zero dimension")

    Ar = np.ascontiguousarray(Ar)
    Ai = np.ascontiguousarray(Ai)
    Br = np.ascontiguousarray(Br)
    Bi = np.ascontiguousarray(Bi)

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
        A: Input matrix, shape (m, n), dtype float64

    Returns:
        Result matrix, shape (m, m), dtype float64
    """
    if A.dtype != np.float64:
        raise ValueError("Input must be float64")

    A = np.ascontiguousarray(A)
    m, n = A.shape

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
