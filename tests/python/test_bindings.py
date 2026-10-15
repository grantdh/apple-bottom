#!/usr/bin/env python3
"""
Smoke tests for the apple_bottom Python bindings (H-P1-5).

Standard library only: numpy is not needed, and results are checked against
a pure-Python reference. Run with `make test-python`, or from the repo root:

    AB_LIBRARY=$PWD/build/libapplebottom.dylib python3 -I tests/python/test_bindings.py

`python3 -I` ignores PYTHONPATH, so this file puts python/ on sys.path
itself; AB_LIBRARY pins the dylib under test.
"""

import ctypes
import faulthandler
import gc
import random
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "python"))
import apple_bottom as ab  # noqa: E402

faulthandler.enable()  # a double free kills the process; name the test

RNG = random.Random(20261012)

# DD GEMM error relative to sum_k |a_ik||b_kj|. FP32 would be ~1e-7.
GEMM_TOL = 1e-13
# FP64 -> DD -> FP64 keeps 48 of 53 significand bits.
ROUND_TRIP_TOL = 1e-14


def rand_matrix(rows, cols):
    return [[RNG.uniform(-1.0, 1.0) for _ in range(cols)] for _ in range(rows)]


def rand_complex_matrix(rows, cols):
    return [[complex(RNG.uniform(-1.0, 1.0), RNG.uniform(-1.0, 1.0))
             for _ in range(cols)] for _ in range(rows)]


def apply(f, M):
    return [[f(x) for x in row] for row in M]


def matmul(A, B):
    """Reference product of two lists of rows (real or complex)."""
    cols_b = list(zip(*B))
    return [[sum(a * b for a, b in zip(row, col)) for col in cols_b] for row in A]


class BindingsTest(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        ab.init()

    def assertMatrixClose(self, got, ref, scale, tol):
        """|got - ref| <= tol * scale elementwise; got may be a numpy array."""
        self.assertEqual(len(got), len(ref))
        for i, (g_row, r_row, s_row) in enumerate(zip(got, ref, scale)):
            self.assertEqual(len(g_row), len(r_row))
            for j, (g, r, s) in enumerate(zip(g_row, r_row, s_row)):
                self.assertLessEqual(abs(float(g) - r), tol * s,
                                     f"[{i}][{j}]: got {float(g)!r}, want {r!r}")

    def assertProductClose(self, got, A, B):
        self.assertMatrixClose(got, matmul(A, B),
                               matmul(apply(abs, A), apply(abs, B)), GEMM_TOL)

    def test_init(self):
        ab.init()  # second call returns AB_OK, which must not raise
        self.assertTrue(ab.is_initialized())
        self.assertNotIn(ab.get_device_name(), ("", "Unknown"))
        self.assertEqual(ab.status_string(-3), "Matrix dimensions incompatible")

    def test_restypes_return_python_values(self):
        # A c_int *subclass* restype comes back as a ctypes object, so every
        # status check raised TypeError (H-P1-5).
        allowed = {None, ctypes.c_int, ctypes.c_bool, ctypes.c_size_t,
                   ctypes.c_void_p, ctypes.c_char_p, ab.ABStats}
        for name, fn in vars(ab._libwrap).items():
            if name.startswith("ab_"):
                self.assertIn(fn.restype, allowed, name)

    def test_stats(self):
        ab.reset_stats()
        ab.dgemm([[1.0]], [[2.0]])
        stats = ab.get_stats()
        self.assertEqual(stats["dgemm_count"], 1)
        self.assertEqual(stats["zgemm_count"], 0)

    def test_dgemm(self):
        for m, k, n in [(1, 1, 1), (7, 5, 3), (70, 45, 66)]:
            A, B = rand_matrix(m, k), rand_matrix(k, n)
            self.assertProductClose(ab.dgemm(A, B), A, B)

        # alpha/beta path (ab_dgemm_scaled); C starts at zero, so beta drops out
        A, B = rand_matrix(9, 4), rand_matrix(4, 6)
        self.assertProductClose(ab.dgemm(A, B, alpha=-2.5, beta=0.5),
                                apply(lambda x: -2.5 * x, A), B)

        with self.assertRaises(ab.DimensionMismatchError):
            ab.dgemm(rand_matrix(2, 3), rand_matrix(2, 3))

    def test_zgemm(self):
        A, B = rand_complex_matrix(9, 6), rand_complex_matrix(6, 4)
        re, im = (lambda z: z.real), (lambda z: z.imag)
        Cr, Ci = ab.zgemm(apply(re, A), apply(im, A), apply(re, B), apply(im, B))
        ref = matmul(A, B)
        scale = matmul(apply(abs, A), apply(abs, B))
        self.assertMatrixClose(Cr, apply(re, ref), scale, GEMM_TOL)
        self.assertMatrixClose(Ci, apply(im, ref), scale, GEMM_TOL)

    def test_matrix_upload_download(self):
        data = rand_matrix(33, 17)
        scale = apply(abs, data)
        with ab.Matrix(33, 17) as m:
            m.upload(data)
            parallel = apply(float, m.download())
            self.assertMatrixClose(parallel, data, scale, ROUND_TRIP_TOL)
            # Serial and parallel conversion give identical bits (H-P0-3 fix).
            self.assertEqual(apply(float, m.download(parallel=False)), parallel)
            m.upload(data, parallel=False)
            self.assertEqual(apply(float, m.download()), parallel)
            with self.assertRaises(ab.DimensionMismatchError):
                m.upload(rand_matrix(17, 33))

    def test_pool(self):
        a, b = rand_matrix(5, 4), rand_matrix(4, 3)
        with ab.MemoryPool(4) as pool:
            A, B, C = (pool.get_matrix(5, 4), pool.get_matrix(4, 3),
                       pool.get_matrix(5, 3))
            A.upload(a)
            B.upload(b)
            self.assertEqual(ab._libwrap.ab_dgemm(A._matrix, B._matrix, C._matrix), 0)
            self.assertProductClose(C.download(), a, b)

            # Dropping a borrowed wrapper must not free the pool's matrix:
            # after reset the pool hands the same handle out again.
            handle = A._matrix
            del A
            gc.collect()
            pool.reset()
            A = pool.get_matrix(5, 4)
            self.assertEqual(A._matrix, handle)
            A.upload(a)
            self.assertMatrixClose(A.download(), a, apply(abs, a), ROUND_TRIP_TOL)

        # Closing the pool frees its matrices; their wrappers refuse further use.
        with self.assertRaises(ab.InvalidArgumentError):
            A.download()
        with self.assertRaises(ab.InvalidArgumentError):
            pool.get_matrix(2, 2)
        pool.close()
        del pool, A, B, C
        gc.collect()

    def test_pool_reset_retires_matrices(self):
        # After reset() the pool hands the same handle out again, here with a
        # larger shape. A wrapper kept from before the reset must refuse use:
        # its download() would let C write 1600 values into a 9-value buffer.
        data = rand_matrix(40, 40)
        with ab.MemoryPool(4) as pool:
            big = pool.get_matrix(40, 40)
            pool.reset()
            small = pool.get_matrix(3, 3)
            handle = small._matrix
            pool.reset()
            current = pool.get_matrix(40, 40)
            self.assertEqual(current._matrix, handle)

            for stale in (big, small):
                with self.assertRaises(ab.InvalidArgumentError):
                    stale.download()
                with self.assertRaises(ab.InvalidArgumentError):
                    stale.upload(rand_matrix(*stale.shape))
                with self.assertRaises(ab.InvalidArgumentError):
                    stale.zero()

            current.upload(data)
            self.assertMatrixClose(current.download(), data, apply(abs, data),
                                   ROUND_TRIP_TOL)
        del pool, big, small, current
        gc.collect()

    def test_session(self):
        a, b = rand_matrix(6, 5), rand_matrix(5, 4)
        with ab.Session() as s:
            for name, rows, cols in (("A", 6, 5), ("B", 5, 4), ("C", 6, 4)):
                s.add(name, rows, cols)
            s.upload("A", a)
            s.upload("B", b)
            s.dgemm("A", "B", "C")
            self.assertProductClose(s.download("C"), a, b)

            # The session owns C; dropping a wrapper for it must not free it.
            borrowed = s.get("C")
            self.assertProductClose(borrowed.download(), a, b)
            del borrowed
            gc.collect()
            self.assertProductClose(s.download("C"), a, b)

            with self.assertRaises(ab.DimensionMismatchError):
                s.dgemm("A", "C", "B")  # status -3 from the C library
            with self.assertRaises(ab.InvalidArgumentError):
                s.add("A", 1, 1)
            late = s.get("A")

        with self.assertRaises(ab.InvalidArgumentError):
            late.download()
        with self.assertRaises(ab.InvalidArgumentError):
            s.download("A")
        s.close()
        del s, late
        gc.collect()

    def test_double_close(self):
        m = ab.Matrix(3, 3)
        m.close()
        m.close()
        with self.assertRaises(ab.InvalidArgumentError):
            m.zero()

        with ab.Matrix(2, 2) as m2:
            m2.zero()
        m2.close()
        with self.assertRaises(ab.InvalidArgumentError):
            m2.download()

        with self.assertRaises(ab.InvalidArgumentError):
            ab.Matrix(65536, 32768)  # 2^31 elements; half-built object is collected

        for cls in (ab.MemoryPool, ab.Session):
            with cls() as obj:
                pass
            obj.close()
            obj.__del__()  # the old Session.__exit__ did this, then GC did again
        del m, m2, obj
        gc.collect()


if __name__ == "__main__":
    print(f"apple_bottom: {ab.__file__}\nlibrary: {ab._lib._name}\n"
          f"numpy: {'yes' if ab.np is not None else 'no'}", flush=True)
    unittest.main(verbosity=2)
