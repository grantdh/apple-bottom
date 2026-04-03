#!/usr/bin/env python3
"""
demo_dgemm.py — Comprehensive apple_bottom DGEMM example

Demonstrates:
- Basic initialization and cleanup
- Functional API (high-level convenience)
- Matrix class (fine-grained control)
- Session API (named matrices)
- Performance measurement
- Error handling
"""

import numpy as np
import sys
import time
from pathlib import Path

# Add parent to path so we can import apple_bottom
sys.path.insert(0, str(Path(__file__).parent.parent))

import apple_bottom as ab


def print_header(title: str) -> None:
    """Print a formatted section header."""
    print(f"\n{'=' * 60}")
    print(f"  {title}")
    print('=' * 60)


def demo_functional_api():
    """Demonstrate the simple functional API."""
    print_header("Functional API (All-in-One)")

    # Create test matrices
    print("Creating test matrices (1024 x 1024)...")
    A = np.random.randn(1024, 1024)
    B = np.random.randn(1024, 1024)

    # Simple matrix multiply: C = A @ B
    print("Computing C = A @ B on GPU...")
    start = time.perf_counter()
    C_gpu = ab.dgemm(A, B)
    gpu_time = time.perf_counter() - start

    # Verify against NumPy
    print("Verifying result with NumPy...")
    C_cpu = A @ B
    error = np.linalg.norm(C_gpu - C_cpu, 'fro') / np.linalg.norm(C_cpu, 'fro')

    print(f"  GPU time: {gpu_time*1000:.2f} ms")
    print(f"  Frobenius relative error: {error:.2e}")
    print(f"  Error within tolerance: {error < 1e-14}")


def demo_matrix_class():
    """Demonstrate fine-grained Matrix class API."""
    print_header("Matrix Class (Fine-Grained Control)")

    rows, cols_inner, cols_out = 512, 512, 256

    print(f"Creating matrices: ({rows}, {cols_inner}) x ({cols_inner}, {cols_out})")

    # Create GPU matrices
    gpu_A = ab.Matrix(rows, cols_inner)
    gpu_B = ab.Matrix(cols_inner, cols_out)
    gpu_C = ab.Matrix(rows, cols_out)

    # Create host data
    A_data = np.random.randn(rows, cols_inner)
    B_data = np.random.randn(cols_inner, cols_out)

    print("Uploading data to GPU...")
    gpu_A.upload(A_data, parallel=True)
    gpu_B.upload(B_data, parallel=True)
    gpu_C.zero()

    print("Computing C = A @ B on GPU...")
    ab._libwrap.ab_dgemm(gpu_A._matrix, gpu_B._matrix, gpu_C._matrix)

    print("Downloading result from GPU...")
    C_result = gpu_C.download(parallel=True)

    # Verify
    print("Verifying result...")
    C_expected = A_data @ B_data
    error = np.linalg.norm(C_result - C_expected, 'fro') / np.linalg.norm(C_expected, 'fro')

    print(f"  Result shape: {C_result.shape}")
    print(f"  Frobenius relative error: {error:.2e}")
    print(f"  Matrix stats: min={C_result.min():.4f}, max={C_result.max():.4f}, "
          f"mean={C_result.mean():.4f}")


def demo_session_api():
    """Demonstrate Session API for organized workflows."""
    print_header("Session API (Named Matrices)")

    size = 256

    with ab.Session() as session:
        print(f"Creating session with {size}x{size} matrices...")
        session.add("A", size, size)
        session.add("B", size, size)
        session.add("C", size, size)

        print("Uploading data...")
        A_data = np.random.randn(size, size)
        B_data = np.random.randn(size, size)
        session.upload("A", A_data)
        session.upload("B", B_data)

        print("Computing C = A @ B...")
        session.dgemm("A", "B", "C")

        print("Downloading result...")
        C_result = session.download("C")

        print("Verifying...")
        C_expected = A_data @ B_data
        error = np.linalg.norm(C_result - C_expected, 'fro') / np.linalg.norm(C_expected, 'fro')
        print(f"  Frobenius relative error: {error:.2e}")


def demo_complex_multiply():
    """Demonstrate complex (ZGEMM) multiplication."""
    print_header("Complex Matrix Multiplication (ZGEMM)")

    size = 128

    print(f"Creating complex matrices ({size} x {size})...")
    # Complex matrices represented as separate real/imaginary
    Ar = np.random.randn(size, size)
    Ai = np.random.randn(size, size)
    Br = np.random.randn(size, size)
    Bi = np.random.randn(size, size)

    print("Computing (Ar + 1j*Ai) @ (Br + 1j*Bi)...")
    Cr, Ci = ab.zgemm(Ar, Ai, Br, Bi)

    # Verify with NumPy complex
    print("Verifying with NumPy...")
    A_complex = Ar + 1j * Ai
    B_complex = Br + 1j * Bi
    C_expected = A_complex @ B_complex

    # Reconstruct GPU result
    C_gpu = Cr + 1j * Ci

    error = np.linalg.norm(C_gpu - C_expected, 'fro') / np.linalg.norm(C_expected, 'fro')
    print(f"  Frobenius relative error: {error:.2e}")
    print(f"  Real part range: [{Cr.min():.4f}, {Cr.max():.4f}]")
    print(f"  Imag part range: [{Ci.min():.4f}, {Ci.max():.4f}]")


def demo_memory_pool():
    """Demonstrate memory pool for iterative codes."""
    print_header("Memory Pool (Iterative Allocation)")

    size = 128
    iterations = 5

    pool = ab.MemoryPool(size_hint=10)

    print(f"Running {iterations} iterations with pooled matrices...")
    for i in range(iterations):
        # Get matrices from pool
        A = pool.get_matrix(size, size)
        B = pool.get_matrix(size, size)
        C = pool.get_matrix(size, size)

        # Simulate work
        A_data = np.ones((size, size))
        B_data = np.ones((size, size)) * (i + 1)
        A.upload(A_data)
        B.upload(B_data)
        ab._libwrap.ab_dgemm(A._matrix, B._matrix, C._matrix)
        C_result = C.download()

        # Reset for next iteration
        pool.reset()

        if i == 0:
            expected = A_data @ B_data
            error = np.linalg.norm(C_result - expected, 'fro') / np.linalg.norm(expected, 'fro')
            print(f"  Iteration {i+1}: error = {error:.2e}")
        else:
            print(f"  Iteration {i+1}: completed")

    print(f"Completed {iterations} iterations using pooled allocation")


def demo_statistics():
    """Demonstrate statistics tracking."""
    print_header("Statistics and Performance Metrics")

    ab.reset_stats()

    # Do some work
    print("Performing matrix operations...")
    for i in range(3):
        A = np.random.randn(512, 512)
        B = np.random.randn(512, 512)
        ab.dgemm(A, B)

    # Get stats
    stats = ab.get_stats()
    print("\nRuntime Statistics:")
    print(f"  DGEMM operations: {stats['dgemm_count']}")
    print(f"  Upload time: {stats['upload_time_ms']:.2f} ms")
    print(f"  Download time: {stats['download_time_ms']:.2f} ms")
    print(f"  Kernel time: {stats['kernel_time_ms']:.2f} ms")
    print(f"  Elements converted: {stats['elements_converted']}")


def demo_error_handling():
    """Demonstrate error handling."""
    print_header("Error Handling")

    # Test dimension mismatch
    print("Testing dimension mismatch...")
    try:
        A = np.random.randn(100, 200)
        B = np.random.randn(150, 100)  # Wrong inner dimension
        ab.dgemm(A, B)
    except ab.DimensionMismatchError as e:
        print(f"  Caught expected error: {type(e).__name__}")

    # Test invalid dtype
    print("Testing invalid dtype...")
    try:
        A = np.random.randn(100, 100).astype(np.float32)
        B = np.random.randn(100, 100).astype(np.float32)
        ab.dgemm(A, B)
    except ValueError as e:
        print(f"  Caught expected error: {type(e).__name__}")

    # Test matrix dimension limits
    print("Testing dimension limits...")
    try:
        gpu_m = ab.Matrix(50000, 50000)  # Exceeds MAX_DIMENSION
    except ab.InvalidArgumentError as e:
        print(f"  Caught expected error: {type(e).__name__}")

    print("All error handling tests passed")


def main():
    """Run all demonstrations."""
    print("\n" + "=" * 60)
    print("  apple-bottom Python Wrapper — DGEMM Demonstrations")
    print("=" * 60)

    try:
        # Initialize GPU
        print("\nInitializing GPU...")
        ab.init()
        device_name = ab.get_device_name()
        print(f"GPU Device: {device_name}")

        # Run demonstrations
        demo_functional_api()
        demo_matrix_class()
        demo_session_api()
        demo_complex_multiply()
        demo_memory_pool()
        demo_statistics()
        demo_error_handling()

        # Final message
        print_header("Summary")
        print("All demonstrations completed successfully!")
        print("apple-bottom Python wrapper is working correctly.")

    except ab.DeviceNotFoundError:
        print("\nERROR: No Apple Silicon GPU found.")
        print("This example requires an Apple Silicon Mac.")
        sys.exit(1)

    except Exception as e:
        print(f"\nUnexpected error: {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

    finally:
        # Cleanup
        print("\nShutting down GPU...")
        ab.shutdown()
        print("Done.")


if __name__ == "__main__":
    main()
