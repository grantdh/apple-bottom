// =============================================================================
// device_api.m — Implementation of the device-buffer layer
// Copyright (c) 2026 Grant Heileman, Technology Residue. MIT License.
// =============================================================================
//
// Week-1 implementation: lifecycle + host↔device↔device memcpy + memset
// + synchronous single-stream model. BLAS entry points are stubbed with
// TODO markers so the DevXlib __METAL backend can link against the full
// symbol set while we iterate on the kernels.
//
// Design:
//   - ab_dev_buffer_s wraps a single id<MTLBuffer> with shared storage mode.
//     Shared storage is the right default on unified-memory Apple Silicon —
//     the CPU pointer and GPU address point at the same physical page, so
//     a "memcpy_h2d" is really just memcpy() into that page.
//   - The "stream" abstraction currently exposes a singleton. QE's OpenACC
//     residency only ever uses acc_stream(0), so this is sufficient for
//     Month-1 integration work. Full multi-stream support is Month-3 scope.
//   - We get our own MTLDevice + MTLCommandQueue here rather than reaching
//     into apple_bottom.m's ABContextImpl. Metal returns the same system
//     device each call, so memcpy_d2d between a matrix-handle buffer and a
//     dev-buffer is legal at the Metal level. This keeps device_api.m
//     independent of the main translation unit for compile-time hygiene.
// =============================================================================

#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#include <os/lock.h>
#include <string.h>
#include <stdlib.h>

#include "apple_bottom.h"
#include "apple_bottom_device.h"

// -----------------------------------------------------------------------------
// Opaque types
// -----------------------------------------------------------------------------

struct ab_dev_buffer_s {
    id<MTLBuffer> mtl;
    size_t nbytes;
};

struct ab_dev_stream_s {
    id<MTLCommandQueue> queue;
    // Placeholder for future ordering / event support.
};

// -----------------------------------------------------------------------------
// Per-translation-unit context
// -----------------------------------------------------------------------------

static id<MTLDevice>       g_dev_device       = nil;
static id<MTLCommandQueue> g_dev_queue        = nil;
static struct ab_dev_stream_s g_default_stream = { 0 };
static os_unfair_lock      g_dev_init_lock    = OS_UNFAIR_LOCK_INIT;
static bool                g_dev_initialized  = false;

static bool dev_ensure_init(void) {
    if (g_dev_initialized) return true;
    os_unfair_lock_lock(&g_dev_init_lock);
    if (!g_dev_initialized) {
        // Make sure the rest of apple-bottom is also initialized — callers
        // will typically mix ab_dev_* with ab_matrix_* and expect both to
        // share a Metal context.
        (void)ab_init();

        g_dev_device = MTLCreateSystemDefaultDevice();
        if (g_dev_device) {
            g_dev_queue = [g_dev_device newCommandQueue];
            g_default_stream.queue = g_dev_queue;
            g_dev_initialized = (g_dev_queue != nil);
        }
    }
    os_unfair_lock_unlock(&g_dev_init_lock);
    return g_dev_initialized;
}

// -----------------------------------------------------------------------------
// Lifecycle
// -----------------------------------------------------------------------------

ab_dev_buffer_t ab_dev_malloc(size_t nbytes) {
    if (nbytes == 0) return NULL;
    if (!dev_ensure_init()) return NULL;

    @autoreleasepool {
        id<MTLBuffer> mtl = [g_dev_device newBufferWithLength:nbytes
                                                      options:MTLResourceStorageModeShared];
        if (!mtl) return NULL;

        // Under ObjC++ ARC, a struct with a strong `id` field is non-POD;
        // using `new` (rather than calloc) lets the implicit constructor
        // initialize the ARC-managed slot correctly, and `delete` runs the
        // destructor that releases the MTLBuffer.
        struct ab_dev_buffer_s* buf = new struct ab_dev_buffer_s();
        if (!buf) return NULL;
        buf->mtl    = mtl;
        buf->nbytes = nbytes;
        return buf;
    }
}

void ab_dev_free(ab_dev_buffer_t buf) {
    if (!buf) return;
    @autoreleasepool {
        delete buf;  // destructor releases the retained MTLBuffer
    }
}

size_t ab_dev_buffer_size(ab_dev_buffer_t buf) {
    return buf ? buf->nbytes : 0;
}

// -----------------------------------------------------------------------------
// Memcpy — shared storage means no blits are required
// -----------------------------------------------------------------------------

static inline bool range_in_bounds(size_t offset, size_t nbytes, size_t total) {
    if (nbytes == 0) return true;
    if (offset > total) return false;
    return (total - offset) >= nbytes;
}

ABStatus ab_dev_memcpy_h2d(ab_dev_buffer_t dst, size_t dst_offset,
                           const void* src, size_t nbytes) {
    if (!dst || !src) return AB_ERROR_INVALID_ARG;
    if (!range_in_bounds(dst_offset, nbytes, dst->nbytes))
        return AB_ERROR_INVALID_ARG;
    if (nbytes == 0) return AB_OK;

    void* contents = [dst->mtl contents];
    if (!contents) return AB_ERROR_ALLOC_FAILED;
    memcpy((char*)contents + dst_offset, src, nbytes);
    return AB_OK;
}

ABStatus ab_dev_memcpy_d2h(void* dst, ab_dev_buffer_t src, size_t src_offset,
                           size_t nbytes) {
    if (!dst || !src) return AB_ERROR_INVALID_ARG;
    if (!range_in_bounds(src_offset, nbytes, src->nbytes))
        return AB_ERROR_INVALID_ARG;
    if (nbytes == 0) return AB_OK;

    // Ensure any in-flight GPU work on this buffer has finished before the
    // CPU reads it. Week-1 has no async pipeline so this is a no-op, but
    // keeping the sync call here documents the contract for Month-2.
    (void)ab_dev_stream_sync(NULL);

    const void* contents = [src->mtl contents];
    if (!contents) return AB_ERROR_ALLOC_FAILED;
    memcpy(dst, (const char*)contents + src_offset, nbytes);
    return AB_OK;
}

ABStatus ab_dev_memcpy_d2d(ab_dev_buffer_t dst, size_t dst_offset,
                           ab_dev_buffer_t src, size_t src_offset,
                           size_t nbytes) {
    if (!dst || !src) return AB_ERROR_INVALID_ARG;
    if (!range_in_bounds(dst_offset, nbytes, dst->nbytes)) return AB_ERROR_INVALID_ARG;
    if (!range_in_bounds(src_offset, nbytes, src->nbytes)) return AB_ERROR_INVALID_ARG;
    if (nbytes == 0) return AB_OK;

    // Could use a blit encoder here, but with shared storage a direct
    // memcpy between the two backing pages is equivalent and faster for
    // small transfers. Blit becomes preferable once we add private-storage
    // buffers in Month-2.
    (void)ab_dev_stream_sync(NULL);
    void*       d = [dst->mtl contents];
    const void* s = [src->mtl contents];
    if (!d || !s) return AB_ERROR_ALLOC_FAILED;
    memmove((char*)d + dst_offset, (const char*)s + src_offset, nbytes);
    return AB_OK;
}

ABStatus ab_dev_memcpy_h2d_async(ab_dev_buffer_t dst, size_t dst_offset,
                                 const void* src, size_t nbytes,
                                 ab_dev_stream_t stream) {
    (void)stream;
    // Week-1: synchronous fallback. The DevXlib backend's _async macros
    // will still work — they just won't overlap anything yet.
    return ab_dev_memcpy_h2d(dst, dst_offset, src, nbytes);
}

ABStatus ab_dev_memcpy_d2h_async(void* dst, ab_dev_buffer_t src, size_t src_offset,
                                 size_t nbytes, ab_dev_stream_t stream) {
    (void)stream;
    return ab_dev_memcpy_d2h(dst, src, src_offset, nbytes);
}

ABStatus ab_dev_memset(ab_dev_buffer_t buf, size_t offset,
                       int value, size_t nbytes) {
    if (!buf) return AB_ERROR_INVALID_ARG;
    if (!range_in_bounds(offset, nbytes, buf->nbytes)) return AB_ERROR_INVALID_ARG;
    if (nbytes == 0) return AB_OK;
    void* contents = [buf->mtl contents];
    if (!contents) return AB_ERROR_ALLOC_FAILED;
    memset((char*)contents + offset, value, nbytes);
    return AB_OK;
}

// -----------------------------------------------------------------------------
// Streams
// -----------------------------------------------------------------------------

ab_dev_stream_t ab_dev_stream_create(void) {
    if (!dev_ensure_init()) return NULL;
    // Singleton for Week-1. Multi-queue support is Month-3.
    return &g_default_stream;
}

void ab_dev_stream_destroy(ab_dev_stream_t stream) {
    (void)stream;  // Singleton has no cleanup.
}

ABStatus ab_dev_stream_sync(ab_dev_stream_t stream) {
    if (!dev_ensure_init()) return AB_ERROR_NO_DEVICE;
    (void)stream;
    // No outstanding work in Week-1 — every op is synchronous on the CPU
    // side. When async memcpy / BLAS land, this will wait on the stream's
    // most recent command buffer.
    return AB_OK;
}

// -----------------------------------------------------------------------------
// BLAS entry points (Week-2 TODO)
// -----------------------------------------------------------------------------
//
// These forward to the existing ab_dgemm / ab_zgemm implementations once
// we've wired up buffer→matrix-handle conversion. For now they return
// AB_ERROR_INVALID_ARG so the link resolves but callers fail loudly.
// -----------------------------------------------------------------------------

ABStatus ab_dev_dgemm(ABTranspose transA, ABTranspose transB,
                      int m, int n, int k,
                      double alpha,
                      ab_dev_buffer_t A, int lda,
                      ab_dev_buffer_t B, int ldb,
                      double beta,
                      ab_dev_buffer_t C, int ldc) {
    (void)transA; (void)transB; (void)m; (void)n; (void)k;
    (void)alpha; (void)A; (void)lda; (void)B; (void)ldb;
    (void)beta;  (void)C; (void)ldc;
    return AB_ERROR_INVALID_ARG;  // TODO Week-2
}

ABStatus ab_dev_dgemm_offset(ABTranspose transA, ABTranspose transB,
                             int m, int n, int k,
                             double alpha,
                             ab_dev_buffer_t A, size_t a_offset, int lda,
                             ab_dev_buffer_t B, size_t b_offset, int ldb,
                             double beta,
                             ab_dev_buffer_t C, size_t c_offset, int ldc) {
    (void)transA; (void)transB; (void)m; (void)n; (void)k;
    (void)alpha; (void)A; (void)a_offset; (void)lda;
    (void)B; (void)b_offset; (void)ldb;
    (void)beta;  (void)C; (void)c_offset; (void)ldc;
    return AB_ERROR_INVALID_ARG;  // TODO Week-2
}

ABStatus ab_dev_zgemm(ABTranspose transA, ABTranspose transB,
                      int m, int n, int k,
                      const double alpha[2],
                      ab_dev_buffer_t A, int lda,
                      ab_dev_buffer_t B, int ldb,
                      const double beta[2],
                      ab_dev_buffer_t C, int ldc) {
    (void)transA; (void)transB; (void)m; (void)n; (void)k;
    (void)alpha; (void)A; (void)lda; (void)B; (void)ldb;
    (void)beta;  (void)C; (void)ldc;
    return AB_ERROR_INVALID_ARG;  // TODO Week-2
}

ABStatus ab_dev_zgemm_offset(ABTranspose transA, ABTranspose transB,
                             int m, int n, int k,
                             const double alpha[2],
                             ab_dev_buffer_t A, size_t a_offset, int lda,
                             ab_dev_buffer_t B, size_t b_offset, int ldb,
                             const double beta[2],
                             ab_dev_buffer_t C, size_t c_offset, int ldc) {
    (void)transA; (void)transB; (void)m; (void)n; (void)k;
    (void)alpha; (void)A; (void)a_offset; (void)lda;
    (void)B; (void)b_offset; (void)ldb;
    (void)beta;  (void)C; (void)c_offset; (void)ldc;
    return AB_ERROR_INVALID_ARG;  // TODO Week-2
}

ABStatus ab_dev_conjg_c(ab_dev_buffer_t buf, size_t offset, size_t n) {
    (void)buf; (void)offset; (void)n;
    return AB_ERROR_INVALID_ARG;  // TODO Week-2
}

ABStatus ab_dev_scale_z(ab_dev_buffer_t buf, size_t offset, size_t n,
                        const double alpha[2]) {
    (void)buf; (void)offset; (void)n; (void)alpha;
    return AB_ERROR_INVALID_ARG;  // TODO Week-2
}

ABStatus ab_dev_axpy_z(size_t n,
                       const double alpha[2],
                       ab_dev_buffer_t x, size_t x_offset, int incx,
                       ab_dev_buffer_t y, size_t y_offset, int incy) {
    (void)n; (void)alpha;
    (void)x; (void)x_offset; (void)incx;
    (void)y; (void)y_offset; (void)incy;
    return AB_ERROR_INVALID_ARG;  // TODO Week-2
}
