// =============================================================================
// apple_bottom_device.h — Device-resident buffer API
// Copyright (c) 2026 Grant Heileman, Technology Residue. MIT License.
// =============================================================================
//
// This header defines the opaque device-buffer layer that the DevXlib __METAL
// backend will call into. Unlike the matrix-handle API in apple_bottom.h,
// this layer exposes *raw FP64 buffers* (interleaved real/imag for complex)
// that mirror the memory model DevXlib, cuBLAS, and OpenACC expect:
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
// and we cannot express `hpsi(1,n_start)` as pointer arithmetic on an opaque
// buffer handle — the offset has to travel as an explicit argument.
//
// Week-1 scope:
//   - Allocation / deallocation
//   - Host↔device and device↔device memcpy
//   - Memset, size query
//   - Stream create / destroy / sync (single stream today, stub for future)
//   - BLAS entry points declared but NOT implemented (TODO markers in .m)
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
// mechanism). Reserved for future async use — Week-1 has a single default
// stream and all operations are synchronous.
typedef struct ab_dev_stream_s* ab_dev_stream_t;

// -----------------------------------------------------------------------------
// Lifecycle
// -----------------------------------------------------------------------------

// Allocate `nbytes` of device memory. Returns NULL on failure. The buffer
// contents are undefined until written. Internally ensures Metal is
// initialized (safe to call before ab_init, will perform init lazily).
ab_dev_buffer_t ab_dev_malloc(size_t nbytes);

// Free a device buffer. Safe to call with NULL.
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

// Async variants — post-Week-1. Declared here so DevXlib templates can
// include the header; implementations will return AB_ERROR_INVALID_ARG
// until stream support lands.
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

// Create a new stream. Week-1 returns a shared singleton; close-enough for
// DevXlib integration work since QE only uses acc_stream(0).
ab_dev_stream_t ab_dev_stream_create(void);

// Destroy a stream. No-op on the shared singleton.
void ab_dev_stream_destroy(ab_dev_stream_t stream);

// Block until all work on `stream` has completed. Pass NULL for the default
// (implicit) stream.
ABStatus ab_dev_stream_sync(ab_dev_stream_t stream);

// -----------------------------------------------------------------------------
// BLAS entry points (Week-2 scope — declared, not yet implemented)
// -----------------------------------------------------------------------------
//
// All matrix arguments are *column-major* (Fortran order). Offsets are
// measured in elements of the corresponding Fortran datatype:
//   ab_dev_dgemm: offsets are in doubles
//   ab_dev_zgemm: offsets are in complex-doubles (= 2 × double)
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

// Elementwise utilities that DevXlib exposes via device_auxfunc.
// These are direct analogues of the CUF kernels in devxlib.
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
