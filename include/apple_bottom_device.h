// =============================================================================
// apple_bottom_device.h — Device-resident buffer API
// Copyright (c) 2026 Grant Heileman, Technology Residue. MIT License.
// =============================================================================
//
// This header defines a device-buffer layer: raw FP64 buffers (interleaved
// real/imag for complex) that the caller allocates, fills and reuses, in the
// memory model that cuBLAS-style device APIs and Fortran device-array codes
// expect. Unlike the ABMatrix handles of apple_bottom.h, the data is FP64 and
// column-major:
//
//     host_ptr  ──(ab_dev_memcpy_h2d)──►  ab_dev_buffer_t  ──►  ab_dev_zgemm
//                                                 │
//                                                 └──(ab_dev_memcpy_d2h)──►  host_ptr
//
// Buffers are opaque (id<MTLBuffer> wrappers). All BLAS entry points accept
// offsets measured in *elements of the Fortran datatype* (double for real,
// complex<double> = 2 doubles for complex), because Fortran call sites do:
//
//     CALL DEV_ZGEMM('N','N', n,m,k, one, hpsi(1,n_start), ldhpsi, ...)
//
// and `hpsi(1,n_start)` cannot be expressed as pointer arithmetic on an
// opaque buffer handle: the offset travels as an explicit argument.
//
// What is implemented:
//   - Allocation and deallocation, size query, memset
//   - Registration of memory the caller already owns, without a copy
//   - Host<->device and device<->device copies
//   - One default stream; every operation is synchronous
//   - ab_dev_dgemm / ab_dev_zgemm (column-major, transA/transB, ld, offsets).
//     Each call converts its FP64 operands to DD and back, so it has the
//     per-call costs of the BLAS ABI, not the resident speed of the ABMatrix
//     API.
//   - ab_dev_dgemv / ab_dev_zgemv / ab_dev_zhemv (BLAS-2). These read the
//     FP64 matrix in place on every call, converting to DD in registers, so
//     a matrix filled or registered once runs each product at GPU memory
//     bandwidth: the case of Lanczos, Haydock and other Krylov solvers.
//
// Thread safety: functions are serialized through a single command queue,
// same as the rest of apple-bottom. Concurrent callers from Fortran (OpenMP
// parallel region, for instance) must not share a single buffer.
// =============================================================================
#ifndef APPLE_BOTTOM_DEVICE_H
#define APPLE_BOTTOM_DEVICE_H

#include <stdint.h>
#include <stddef.h>
#include <stdbool.h>
#include "apple_bottom.h"   // ABStatus, ABTranspose

#ifdef __cplusplus
extern "C" {
#endif

// Opaque handle to a device buffer (wraps id<MTLBuffer> internally).
typedef struct ab_dev_buffer_s* ab_dev_buffer_t;

// Opaque handle to a command stream (maps to MTLCommandQueue + an ordering
// mechanism). Reserved for future asynchronous use: today there is a single
// default stream and every operation is synchronous.
typedef struct ab_dev_stream_s* ab_dev_stream_t;

// -----------------------------------------------------------------------------
// Lifecycle
// -----------------------------------------------------------------------------

// Allocate `nbytes` of device memory. Returns NULL on failure. The buffer
// contents are undefined until written. Internally ensures Metal is
// initialized (safe to call before ab_init, will perform init lazily).
ab_dev_buffer_t ab_dev_malloc(size_t nbytes);

// A device buffer over `nbytes` of memory the caller already owns, starting
// at `ptr` (malloc, posix_memalign, mmap, a Fortran ALLOCATABLE): no copy is
// made, GPU work reads and writes the caller's memory in place, and sees
// what the caller writes to it between calls. Offsets count from `ptr`.
// Returns NULL for nbytes == 0, if part of the span is unmapped or
// unreadable, or if Metal refuses it; copy into an ab_dev_malloc() buffer
// instead.
// Call ab_dev_free() on the handle BEFORE the memory is freed, reallocated
// or unmapped: a wrapper that outlives its memory keeps reading the old
// pages, without an error. ab_dev_free() never frees the caller's memory.
ab_dev_buffer_t ab_dev_register(void* ptr, size_t nbytes);

// Free a device buffer (for a registered one, only the handle). Safe to
// call with NULL.
void ab_dev_free(ab_dev_buffer_t buf);

// Query the allocated size in bytes.
size_t ab_dev_buffer_size(ab_dev_buffer_t buf);

// -----------------------------------------------------------------------------
// Data transfer
// -----------------------------------------------------------------------------

// Host → device. `dst_offset` and `nbytes` are in bytes.
ABStatus ab_dev_memcpy_h2d(ab_dev_buffer_t dst, size_t dst_offset,
                           const void* src, size_t nbytes);

// Device → host.
ABStatus ab_dev_memcpy_d2h(void* dst, ab_dev_buffer_t src, size_t src_offset,
                           size_t nbytes);

// Device → device.
ABStatus ab_dev_memcpy_d2d(ab_dev_buffer_t dst, size_t dst_offset,
                           ab_dev_buffer_t src, size_t src_offset,
                           size_t nbytes);

// Asynchronous variants. They take a stream for API compatibility but run
// synchronously on the default stream.
ABStatus ab_dev_memcpy_h2d_async(ab_dev_buffer_t dst, size_t dst_offset,
                                 const void* src, size_t nbytes,
                                 ab_dev_stream_t stream);
ABStatus ab_dev_memcpy_d2h_async(void* dst, ab_dev_buffer_t src, size_t src_offset,
                                 size_t nbytes, ab_dev_stream_t stream);

// Fill a byte range with a constant byte value (memset semantics).
ABStatus ab_dev_memset(ab_dev_buffer_t buf, size_t offset,
                       int value, size_t nbytes);

// -----------------------------------------------------------------------------
// Streams
// -----------------------------------------------------------------------------

// Create a stream. Returns the shared default stream: one stream is all a
// code that uses a single device stream (e.g. OpenACC's acc_stream(0)) needs.
ab_dev_stream_t ab_dev_stream_create(void);

// Destroy a stream. No-op on the shared singleton.
void ab_dev_stream_destroy(ab_dev_stream_t stream);

// Block until all work on `stream` has completed. Pass NULL for the default
// (implicit) stream.
ABStatus ab_dev_stream_sync(ab_dev_stream_t stream);

// -----------------------------------------------------------------------------
// BLAS entry points
// -----------------------------------------------------------------------------
//
// All matrix arguments are *column-major* (Fortran order). Offsets are
// measured in elements of the corresponding Fortran datatype:
//   ab_dev_dgemm: offsets are in doubles
//   ab_dev_zgemm: offsets are in complex-doubles (= 2 × double)
//
// transA/transB follow BLAS: op(X) is X, X^T or X^H (AB_CONJ_TRANS is a
// plain transpose for ab_dev_dgemm). Leading dimensions count the stored
// rows (lda >= rows of A as stored, i.e. m or k). An invalid flag, a short
// leading dimension, m, n or k <= 0, or a matrix extending past its buffer
// returns AB_ERROR_INVALID_ARG. An operand beyond the matrix size limits
// (AB_MAX_MATRIX_ELEMENTS or the device's maxBufferLength, see apple_bottom.h)
// returns AB_ERROR_TOO_LARGE.
//
// The _offset variants exist so Fortran callers can express `A(1,j)` etc.
// without having to reinterpret the opaque handle as a pointer.
// -----------------------------------------------------------------------------

// C := alpha * op(A) * op(B) + beta * C   (real double, column-major)
ABStatus ab_dev_dgemm(ABTranspose transA, ABTranspose transB,
                      int m, int n, int k,
                      double alpha,
                      ab_dev_buffer_t A, int lda,
                      ab_dev_buffer_t B, int ldb,
                      double beta,
                      ab_dev_buffer_t C, int ldc);

ABStatus ab_dev_dgemm_offset(ABTranspose transA, ABTranspose transB,
                             int m, int n, int k,
                             double alpha,
                             ab_dev_buffer_t A, size_t a_offset, int lda,
                             ab_dev_buffer_t B, size_t b_offset, int ldb,
                             double beta,
                             ab_dev_buffer_t C, size_t c_offset, int ldc);

// C := alpha * op(A) * op(B) + beta * C   (complex double, interleaved)
ABStatus ab_dev_zgemm(ABTranspose transA, ABTranspose transB,
                      int m, int n, int k,
                      const double alpha[2],
                      ab_dev_buffer_t A, int lda,
                      ab_dev_buffer_t B, int ldb,
                      const double beta[2],
                      ab_dev_buffer_t C, int ldc);

ABStatus ab_dev_zgemm_offset(ABTranspose transA, ABTranspose transB,
                             int m, int n, int k,
                             const double alpha[2],
                             ab_dev_buffer_t A, size_t a_offset, int lda,
                             ab_dev_buffer_t B, size_t b_offset, int ldb,
                             const double beta[2],
                             ab_dev_buffer_t C, size_t c_offset, int ldc);

// -----------------------------------------------------------------------------
// BLAS-2: matrix-vector products on a resident matrix
// -----------------------------------------------------------------------------
//
//   ab_dev_dgemv: y := alpha * op(A) * x + beta * y, op(A) = A or A^T
//                 (AB_CONJ_TRANS is A^T), A m x n
//   ab_dev_zgemv: the same for complex A, op(A) = A, A^T or A^H
//   ab_dev_zhemv: y := alpha * A * x + beta * y, A n x n Hermitian, only the
//                 `uplo` triangle referenced and the imaginary parts of its
//                 diagonal taken as zero
//
// A is column-major in a device buffer, starting `a_offset` elements (doubles,
// or complex doubles for Z) into it, with leading dimension lda. x and y are
// host arrays with BLAS increments: incx, incy != 0, negative ones step
// backwards from the far end, as in BLAS. Complex scalars and vectors are
// (re, im) pairs of doubles, i.e. double _Complex or COMPLEX*16.
//
// The GPU reads A as FP64 and accumulates in double-float: componentwise
// error of order 1e-15 relative to |alpha| |op(A)| |x| + |beta| |y| (at
// most 1e-14 in tests/test_blas2.c, dot products up to 65536 long), about
// ten times an FP64 BLAS's. Each call checks, at no extra memory traffic,
// that its data stayed within double-float's exponent range at full
// precision; a call that did not, or that involves NaN or Inf, and every
// call with m or n == 0 or alpha == 0, runs on Accelerate instead, on the
// same memory, with Accelerate's results. ab_get_last_dispatch_path() says
// which ran.
//
// Returns AB_OK when y holds the result, AB_ERROR_INVALID_ARG (y unchanged)
// for a bad flag, m, n < 0, lda < max(1, rows of A), incx or incy == 0,
// NULL x or y, or a matrix extending past its buffer.
//
// Speed (M2 Max): each call costs about 0.2 ms of submit and wait on top of
// reading A, which GEMV does at 223-359 GB/s on 4096^2-16384^2 matrices and
// ZHEMV, reading its stored triangle once, at 159-312 GB/s, so the GPU wins
// for large matrices that stay in a device buffer across many products
// (README.md, "BLAS-2").
// -----------------------------------------------------------------------------

ABStatus ab_dev_dgemv(ABTranspose trans, int m, int n, double alpha,
                      ab_dev_buffer_t A, size_t a_offset, int lda,
                      const double* x, int incx,
                      double beta, double* y, int incy);

ABStatus ab_dev_zgemv(ABTranspose trans, int m, int n, const double alpha[2],
                      ab_dev_buffer_t A, size_t a_offset, int lda,
                      const double* x, int incx,
                      const double beta[2], double* y, int incy);

ABStatus ab_dev_zhemv(ABUplo uplo, int n, const double alpha[2],
                      ab_dev_buffer_t A, size_t a_offset, int lda,
                      const double* x, int incx,
                      const double beta[2], double* y, int incy);

// Element-wise utilities of the kind GPU-resident Fortran codes need
// (conjugate, scale, axpy on device buffers).
ABStatus ab_dev_conjg_c(ab_dev_buffer_t buf, size_t offset, size_t n);
ABStatus ab_dev_scale_z(ab_dev_buffer_t buf, size_t offset, size_t n,
                        const double alpha[2]);
ABStatus ab_dev_axpy_z(size_t n,
                       const double alpha[2],
                       ab_dev_buffer_t x, size_t x_offset, int incx,
                       ab_dev_buffer_t y, size_t y_offset, int incy);

#ifdef __cplusplus
}
#endif

#endif // APPLE_BOTTOM_DEVICE_H
