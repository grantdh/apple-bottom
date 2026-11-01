// =============================================================================
// device_api.m — Implementation of the device-buffer layer
// Copyright (c) 2026 Grant Heileman, Technology Residue. MIT License.
// =============================================================================
//
// Lifecycle, host↔device↔device memcpy, memset, a synchronous single-stream
// model, ab_dev_dgemm / ab_dev_zgemm and element-wise helpers.
//
// Design:
//   - ab_dev_buffer_s wraps a single id<MTLBuffer> with shared storage mode.
//     Shared storage is the right default on unified-memory Apple Silicon —
//     the CPU pointer and GPU address point at the same physical page, so
//     a "memcpy_h2d" is really just memcpy() into that page.
//   - ab_dev_register() wraps memory the caller already owns, no copy: a
//     no-copy MTLBuffer over the page-rounded span, the caller's data
//     `base` bytes in (device_internal.h). Every CPU access goes through
//     dev_data(), so registered and allocated buffers behave the same.
//   - The "stream" abstraction currently exposes a singleton, which is all
//     a code that uses one device stream (e.g. OpenACC's acc_stream(0))
//     needs. Multiple streams are not implemented.
//   - We get our own MTLDevice + MTLCommandQueue here rather than reaching
//     into apple_bottom.m's ABContextImpl. Metal returns the same system
//     device each call, so memcpy_d2d between a matrix-handle buffer and a
//     dev-buffer is legal at the Metal level. This keeps device_api.m
//     independent of the main translation unit for compile-time hygiene.
// =============================================================================

#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#include <fenv.h>
#include <mach/mach.h>
#include <mach/mach_vm.h>
#include <os/lock.h>
#include <string.h>
#include <unistd.h>

#include "apple_bottom.h"
#include "apple_bottom_device.h"
#include "ab_internal.h"
#include "device_internal.h"   // struct ab_dev_buffer_s
#include "gpu_stage.h"

// -----------------------------------------------------------------------------
// Opaque types
// -----------------------------------------------------------------------------

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

        fenv_t host_fenv;                 // see ab_init(): keep Metal's FP flags out
        feholdexcept(&host_fenv);
        g_dev_device = MTLCreateSystemDefaultDevice();
        fesetenv(&host_fenv);
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

// The caller's first byte, CPU address.
static inline char* dev_data(ab_dev_buffer_t b) { return (char*)[b->mtl contents] + b->base; }

void* ab_internal_dev_data(ab_dev_buffer_t buf) { return dev_data(buf); }

// A handle for the caller's nbytes, `base` bytes into mtl.
static ab_dev_buffer_t new_handle(id<MTLBuffer> mtl, size_t nbytes, size_t base) {
    // Under ObjC++ ARC, a struct with a strong `id` field is non-POD;
    // using `new` (rather than calloc) lets the implicit constructor
    // initialize the ARC-managed slot correctly, and `delete` runs the
    // destructor that releases the MTLBuffer.
    struct ab_dev_buffer_s* buf = new struct ab_dev_buffer_s();
    buf->mtl    = mtl;
    buf->nbytes = nbytes;
    buf->base   = base;
    return buf;
}

ab_dev_buffer_t ab_dev_malloc(size_t nbytes) {
    if (nbytes == 0) return NULL;
    if (!dev_ensure_init()) return NULL;

    @autoreleasepool {
        id<MTLBuffer> mtl = [g_dev_device newBufferWithLength:nbytes
                                                      options:MTLResourceStorageModeShared];
        return mtl ? new_handle(mtl, nbytes, 0) : NULL;
    }
}

// True if [lo, hi) is mapped readable throughout, without holes. One large
// allocation spans several VM map entries (the kernel caps an anonymous
// entry at 128 MB), so the regions are walked rather than required to be one.
static bool span_is_mapped(uintptr_t lo, uintptr_t hi) {
    for (mach_vm_address_t at = lo; at < hi;) {
        mach_vm_address_t addr = at;
        mach_vm_size_t size = 0;
        vm_region_basic_info_data_64_t info;
        mach_msg_type_number_t count = VM_REGION_BASIC_INFO_COUNT_64;
        mach_port_t object = MACH_PORT_NULL;
        kern_return_t kr = mach_vm_region(mach_task_self(), &addr, &size, VM_REGION_BASIC_INFO_64,
                                          (vm_region_info_t)&info, &count, &object);
        if (object != MACH_PORT_NULL) mach_port_deallocate(mach_task_self(), object);
        // mach_vm_region reports the region at or above `at`: above means a hole.
        if (kr != KERN_SUCCESS || addr > at || !(info.protection & VM_PROT_READ)) return false;
        at = addr + size;
    }
    return true;
}

id<MTLBuffer> ab_internal_wrap_host(id<MTLDevice> dev, const void *ptr, size_t nbytes,
                                    size_t *offset) {
    if (!dev || !ptr || nbytes == 0) return nil;
    uintptr_t page = (uintptr_t)getpagesize();
    uintptr_t lo = (uintptr_t)ptr & ~(page - 1);
    uintptr_t end = (uintptr_t)ptr + nbytes;
    if (end < (uintptr_t)ptr || end > UINTPTR_MAX - page) return nil;
    uintptr_t hi = (end + page - 1) & ~(page - 1);
    if (hi - lo > dev.maxBufferLength || !span_is_mapped(lo, hi)) return nil;
    id<MTLBuffer> mtl = [dev newBufferWithBytesNoCopy:(void *)lo length:hi - lo
                                              options:MTLResourceStorageModeShared
                                          deallocator:nil];
    if (mtl) *offset = (uintptr_t)ptr - lo;
    return mtl;
}

ab_dev_buffer_t ab_dev_register(void* ptr, size_t nbytes) {
    if (!ptr || nbytes == 0 || !dev_ensure_init()) return NULL;
    @autoreleasepool {
        size_t base = 0;
        id<MTLBuffer> mtl = ab_internal_wrap_host(g_dev_device, ptr, nbytes, &base);
        return mtl ? new_handle(mtl, nbytes, base) : NULL;
    }
}

void ab_dev_free(ab_dev_buffer_t buf) {
    if (!buf) return;
    @autoreleasepool {
        delete buf;  // destructor releases the retained MTLBuffer (a registration's
                     // wrapper only: the caller's memory stays theirs)
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

    void* contents = dev_data(dst);
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
    // CPU reads it. With no asynchronous work this is a no-op, but the
    // call documents the contract for when there is.
    (void)ab_dev_stream_sync(NULL);

    const void* contents = dev_data(src);
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
    // small transfers. Blit becomes preferable with private-storage
    // buffers, which are not implemented.
    (void)ab_dev_stream_sync(NULL);
    void*       d = dev_data(dst);
    const void* s = dev_data(src);
    if (!d || !s) return AB_ERROR_ALLOC_FAILED;
    memmove((char*)d + dst_offset, (const char*)s + src_offset, nbytes);
    return AB_OK;
}

ABStatus ab_dev_memcpy_h2d_async(ab_dev_buffer_t dst, size_t dst_offset,
                                 const void* src, size_t nbytes,
                                 ab_dev_stream_t stream) {
    (void)stream;
    // Synchronous: callers of the _async variants work, but nothing
    // overlaps yet.
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
    void* contents = dev_data(buf);
    if (!contents) return AB_ERROR_ALLOC_FAILED;
    memset((char*)contents + offset, value, nbytes);
    return AB_OK;
}

// -----------------------------------------------------------------------------
// Streams
// -----------------------------------------------------------------------------

ab_dev_stream_t ab_dev_stream_create(void) {
    if (!dev_ensure_init()) return NULL;
    // One shared stream; multiple queues are not implemented.
    return &g_default_stream;
}

void ab_dev_stream_destroy(ab_dev_stream_t stream) {
    (void)stream;  // Singleton has no cleanup.
}

ABStatus ab_dev_stream_sync(ab_dev_stream_t stream) {
    if (!dev_ensure_init()) return AB_ERROR_NO_DEVICE;
    (void)stream;
    // No outstanding work: every operation is synchronous on the CPU
    // side. When async memcpy / BLAS land, this will wait on the stream's
    // most recent command buffer.
    return AB_OK;
}

// -----------------------------------------------------------------------------
// BLAS entry points — dispatch via the ABMatrix handle path
// -----------------------------------------------------------------------------
//
// Strategy: device buffers hold raw FP64 in shared Metal storage.  GPU kernels
// operate on DD format (two FP32 per value).  We bridge the gap with
// temporary ABMatrix handles, uploading FP64→DD, dispatching through the
// existing ab_dgemm_scaled / ab_zgemm_ex machinery, then downloading DD→FP64.
// A ZGEMM chooses between Gauss-3M and 4M as the BLAS ABI does, per row of
// op(A) and column of op(B) (ab_internal.h), not per matrix as ab_zgemm_ex.
//
// Each call therefore converts, with the per-call costs of the BLAS ABI. The
// handles are views of buffers the staged path's pool keeps between calls
// (ab_internal_stage_views, gpu_stage.h): fresh ones paid first-touch page
// faults as the conversion wrote them and again before the GPU started.
// DD-native device buffers would remove the per-call conversion; they are not
// implemented.
//
// For _offset variants: we offset into the raw FP64 data in shared memory
// before uploading into the temp ABMatrix.  The Fortran offset is in elements
// of the corresponding type (doubles for DGEMM, complex-doubles for ZGEMM).
//
// Layout: the device buffers are column-major, ABMatrix is row-major, and a
// column-major X read as row-major is X^T.  So both GEMMs compute
// C^T = op(B)^T * op(A)^T on the transposed operands (K-P0-2 / H-P0-4).
// -----------------------------------------------------------------------------

static inline bool dev_trans_valid(ABTranspose t) {
    return t == AB_NO_TRANS || t == AB_TRANS || t == AB_CONJ_TRANS;
}

// Helper: true if a column-major `rows` × `cols` submatrix with leading
// dimension `ld`, starting `offset` elements of `elem_size` bytes into `buf`,
// has ld >= rows (the BLAS rule) and lies inside the buffer.
static bool dev_submatrix_fits(ab_dev_buffer_t buf, size_t offset, int ld,
                               int rows, int cols, size_t elem_size) {
    // Counted in elements, so nothing can overflow.
    size_t span = (size_t)ld * (size_t)(cols - 1) + (size_t)rows;
    return ld >= rows && range_in_bounds(offset, span, buf->nbytes / elem_size);
}

bool ab_internal_dev_matrix_fits(ab_dev_buffer_t buf, size_t offset, int ld, int rows, int cols,
                                 size_t elem_size) {
    if (!buf || ld < rows) return false;
    return rows == 0 || cols == 0 || dev_submatrix_fits(buf, offset, ld, rows, cols, elem_size);
}

// Helper: AB_OK if the m×k, k×n and m×n operands (or their transposes, the
// same sizes) fit the ABMatrix size limits, else the failing check's status.
static ABStatus dev_operands_fit(int m, int n, int k) {
    ABStatus s = ab_matrix_check_size(m, k);
    if (s == AB_OK) s = ab_matrix_check_size(k, n);
    if (s == AB_OK) s = ab_matrix_check_size(m, n);
    return s;
}

// The helpers below convert column-major submatrices X of raw FP64 device
// buffers into and out of ABMatrix handles where they lie, with no staging
// copy, in chunks across cores (ab_internal.h). A submatrix starts `offset`
// elements (doubles, or complex pairs [r0,i0,r1,i1,...]) into its buffer,
// with leading dimension `ld` in the same elements; read as row-major, its
// columns are the rows of X^T. The ABMatrix gives X's shape.

// Helper: upload a submatrix X into an ABMatrix, which receives X^T, or X
// itself when `transpose` is set.
static ABStatus dev_upload_real_submatrix(
    ABMatrix dst, ab_dev_buffer_t src, size_t elem_offset, int ld, bool transpose)
{
    const double* base = (const double*)dev_data(src) + elem_offset;
    return transpose ? ab_internal_upload_cols(dst, base, (size_t)ld)
                     : ab_internal_upload_rows(dst, NULL, base, (size_t)ld, NULL, false);
}

// Helper: download an ABMatrix holding C^T into the submatrix C.
static ABStatus dev_download_real_submatrix(
    ab_dev_buffer_t dst, size_t elem_offset, int ld, ABMatrix src)
{
    double* base = (double*)dev_data(dst) + elem_offset;
    return ab_internal_download_rows(src, NULL, base, (size_t)ld);
}

// Helper: upload a complex submatrix X into separate real and imaginary
// ABMatrix handles, which receive X^T.  If `lines` is not NULL, its rmax and
// imax (zeroed) receive the high words of the largest |Re| and |Im| of each
// column of X (by_col) or each row, the statistics ab_lines_rho reads,
// gathered by the conversion as it goes (a pass of their own cost 9-14% of
// a QE-shaped ab_dev_zgemm).
static ABStatus dev_upload_complex_submatrix(
    ABMatrix dst_r, ABMatrix dst_i,
    ab_dev_buffer_t src, size_t cmplx_offset, int ld,
    ab_line_stats_t* lines, bool by_col)
{
    const double* base = (const double*)dev_data(src) + cmplx_offset * 2;
    return ab_internal_upload_rows(dst_r, dst_i, base, (size_t)ld, lines, by_col);
}

// Helper: download separate real/imag ABMatrix handles holding C^T back
// into the complex submatrix C.
static ABStatus dev_download_complex_submatrix(
    ab_dev_buffer_t dst, size_t cmplx_offset, int ld,
    ABMatrix src_r, ABMatrix src_i)
{
    double* base = (double*)dev_data(dst) + cmplx_offset * 2;
    return ab_internal_download_rows(src_r, src_i, base, (size_t)ld);
}

// Helper: C = alpha * T + beta * C in the complex submatrix C, from separate
// real/imag ABMatrix handles holding T^T (C is read only if beta != 0).
static ABStatus dev_epilogue_complex_submatrix(
    ab_dev_buffer_t dst, size_t cmplx_offset, int ld,
    ABMatrix t_r, ABMatrix t_i, const double alpha[2], const double beta[2])
{
    double* base = (double*)dev_data(dst) + cmplx_offset * 2;
    return ab_internal_zgemm_epilogue(t_r, t_i, base, (size_t)ld, alpha, beta);
}

// ---- DGEMM dispatch --------------------------------------------------------

ABStatus ab_dev_dgemm(ABTranspose transA, ABTranspose transB,
                      int m, int n, int k,
                      double alpha,
                      ab_dev_buffer_t A, int lda,
                      ab_dev_buffer_t B, int ldb,
                      double beta,
                      ab_dev_buffer_t C, int ldc) {
    return ab_dev_dgemm_offset(transA, transB, m, n, k, alpha,
                               A, 0, lda, B, 0, ldb, beta, C, 0, ldc);
}

ABStatus ab_dev_dgemm_offset(ABTranspose transA, ABTranspose transB,
                             int m, int n, int k,
                             double alpha,
                             ab_dev_buffer_t A, size_t a_offset, int lda,
                             ab_dev_buffer_t B, size_t b_offset, int ldb,
                             double beta,
                             ab_dev_buffer_t C, size_t c_offset, int ldc) {
    if (!A || !B || !C) return AB_ERROR_INVALID_ARG;
    if (m <= 0 || n <= 0 || k <= 0) return AB_ERROR_INVALID_ARG;
    if (!dev_trans_valid(transA) || !dev_trans_valid(transB)) return AB_ERROR_INVALID_ARG;

    // Stored (column-major) dimensions of A and B. For real data
    // AB_CONJ_TRANS is a plain transpose.
    bool tA = (transA != AB_NO_TRANS), tB = (transB != AB_NO_TRANS);
    int A_rows = tA ? k : m, A_cols = tA ? m : k;
    int B_rows = tB ? n : k, B_cols = tB ? k : n;
    if (!dev_submatrix_fits(A, a_offset, lda, A_rows, A_cols, sizeof(double)) ||
        !dev_submatrix_fits(B, b_offset, ldb, B_rows, B_cols, sizeof(double)) ||
        !dev_submatrix_fits(C, c_offset, ldc, m, n, sizeof(double)))
        return AB_ERROR_INVALID_ARG;
    ABStatus fit = dev_operands_fit(m, n, k);
    if (fit != AB_OK) return fit;

    // Temporary ABMatrix handles (FP64→DD conversion) for op(A)^T (k×m),
    // op(B)^T (n×k) and C^T (n×m), from the pool.
    const int rows[3] = {k, n, n}, cols[3] = {m, k, m};
    ABMatrix h[3];
    ab_stage_t* slot = ab_internal_stage_views(3, rows, cols, h);
    if (!slot) return AB_ERROR_ALLOC_FAILED;
    ABMatrix mA = h[0], mB = h[1], mC = h[2];

    // Upload from device buffer shared memory into DD-format ABMatrix.
    // op(X)^T is X^T for NO_TRANS and X (transposed on the host as it is
    // converted) otherwise; ab_dgemm_scaled takes no transpose flags.
    ABStatus s = dev_upload_real_submatrix(mA, A, a_offset, lda, tA);
    if (s != AB_OK) goto dgemm_cleanup;
    s = dev_upload_real_submatrix(mB, B, b_offset, ldb, tB);
    if (s != AB_OK) goto dgemm_cleanup;

    // If beta != 0, we need C's current contents
    if (beta != 0.0) {
        s = dev_upload_real_submatrix(mC, C, c_offset, ldc, false);
        if (s != AB_OK) goto dgemm_cleanup;
    }

    // C^T = alpha * op(B)^T * op(A)^T + beta * C^T
    s = ab_dgemm_scaled(alpha, mB, mA, beta, mC);
    if (s != AB_OK) goto dgemm_cleanup;

    // Download DD→FP64 back into device buffer
    s = dev_download_real_submatrix(C, c_offset, ldc, mC);

dgemm_cleanup:
    for (int i = 0; i < 3; i++) ab_matrix_destroy(h[i]);
    ab_internal_stage_release(slot);
    return s;

}

// ---- ZGEMM dispatch --------------------------------------------------------

ABStatus ab_dev_zgemm(ABTranspose transA, ABTranspose transB,
                      int m, int n, int k,
                      const double alpha[2],
                      ab_dev_buffer_t A, int lda,
                      ab_dev_buffer_t B, int ldb,
                      const double beta[2],
                      ab_dev_buffer_t C, int ldc) {
    return ab_dev_zgemm_offset(transA, transB, m, n, k, alpha,
                               A, 0, lda, B, 0, ldb, beta, C, 0, ldc);
}

ABStatus ab_dev_zgemm_offset(ABTranspose transA, ABTranspose transB,
                             int m, int n, int k,
                             const double alpha[2],
                             ab_dev_buffer_t A, size_t a_offset, int lda,
                             ab_dev_buffer_t B, size_t b_offset, int ldb,
                             const double beta[2],
                             ab_dev_buffer_t C, size_t c_offset, int ldc) {
    if (!A || !B || !C || !alpha || !beta) return AB_ERROR_INVALID_ARG;
    if (m <= 0 || n <= 0 || k <= 0) return AB_ERROR_INVALID_ARG;
    if (!dev_trans_valid(transA) || !dev_trans_valid(transB)) return AB_ERROR_INVALID_ARG;

    // Stored (column-major) dimensions of A and B
    int A_rows = (transA == AB_NO_TRANS) ? m : k;
    int A_cols = (transA == AB_NO_TRANS) ? k : m;
    int B_rows = (transB == AB_NO_TRANS) ? k : n;
    int B_cols = (transB == AB_NO_TRANS) ? n : k;
    if (!dev_submatrix_fits(A, a_offset, lda, A_rows, A_cols, 2 * sizeof(double)) ||
        !dev_submatrix_fits(B, b_offset, ldb, B_rows, B_cols, 2 * sizeof(double)) ||
        !dev_submatrix_fits(C, c_offset, ldc, m, n, 2 * sizeof(double)))
        return AB_ERROR_INVALID_ARG;
    bool need_beta = (beta[0] != 0.0 || beta[1] != 0.0);
    ABStatus fit = dev_operands_fit(m, n, k);
    if (fit != AB_OK) return fit;

    // 6 ABMatrix handles from the pool: separate real + imaginary for A^T,
    // B^T, C^T
    const int rows[6] = {A_cols, A_cols, B_cols, B_cols, n, n};
    const int cols[6] = {A_rows, A_rows, B_rows, B_rows, m, m};
    ABMatrix h[6];
    ab_stage_t* slot = ab_internal_stage_views(6, rows, cols, h);
    if (!slot) return AB_ERROR_ALLOC_FAILED;
    ABMatrix mAr = h[0], mAi = h[1], mBr = h[2], mBi = h[3], mCr = h[4], mCi = h[5];
    ab_line_stats_t sa = {}, sb = {};
    ABStatus s = AB_ERROR_ALLOC_FAILED;
    if (!ab_line_parts_init(&sa, m) || !ab_line_parts_init(&sb, n)) goto zgemm_cleanup;

    // Upload: deinterleave complex → separate real/imag, FP64→DD, gathering
    // the parts' largest magnitudes of each row of op(A) (a column of A if
    // transposed) and each column of op(B) (a row of B if transposed)
    s = dev_upload_complex_submatrix(mAr, mAi, A, a_offset, lda, &sa, transA != AB_NO_TRANS);
    if (s != AB_OK) goto zgemm_cleanup;
    s = dev_upload_complex_submatrix(mBr, mBi, B, b_offset, ldb, &sb, transB == AB_NO_TRANS);
    if (s != AB_OK) goto zgemm_cleanup;

    // Dispatch via ab_internal_zgemm (ab_zgemm_ex by the given method, which
    // handles transpose on GPU): T = op(A) * op(B), with alpha and beta
    // applied afterwards. op(X)^T = op(X^T) for every flag ((X^H)^T = conj(X)
    // = (X^T)^H), so C^T = op(B)^T * op(A)^T is that product with the operands
    // and their flags swapped. The alpha/beta epilogue is elementwise, so it
    // runs on C^T unchanged. The method is the BLAS ABI's (ab_lines_method):
    // rho(p, q) is symmetric, and C^T's rows and columns are op(B)'s columns
    // and op(A)'s rows, so C^T takes the decision C = op(A) * op(B) would.
    s = ab_internal_zgemm(transB, transA, mBr, mBi, mAr, mAi, mCr, mCi,
                          ab_lines_method(&sa, &sb));
    if (s != AB_OK) goto zgemm_cleanup;

    // Download (reinterleave real/imag → interleaved complex, DD→FP64) when
    // C = op(A) * op(B), the most common case in QE; else C = alpha * T +
    // beta * C on the host, into C's buffer
    if (!need_beta && alpha[0] == 1.0 && alpha[1] == 0.0)
        s = dev_download_complex_submatrix(C, c_offset, ldc, mCr, mCi);
    else
        s = dev_epilogue_complex_submatrix(C, c_offset, ldc, mCr, mCi, alpha, beta);

zgemm_cleanup:
    ab_line_stats_free(&sa);
    ab_line_stats_free(&sb);
    for (int i = 0; i < 6; i++) ab_matrix_destroy(h[i]);
    ab_internal_stage_release(slot);
    return s;
}

// ---- Element-wise operations --------------------------------------------------

ABStatus ab_dev_conjg_c(ab_dev_buffer_t buf, size_t offset, size_t n) {
    if (!buf) return AB_ERROR_INVALID_ARG;
    if (n == 0) return AB_OK;
    // Negate imaginary parts in-place.  Data is interleaved [r0,i0,r1,i1,...].
    double* base = (double*)dev_data(buf) + offset * 2;
    for (size_t i = 0; i < n; i++)
        base[i * 2 + 1] = -base[i * 2 + 1];
    return AB_OK;
}

ABStatus ab_dev_scale_z(ab_dev_buffer_t buf, size_t offset, size_t n,
                        const double alpha[2]) {
    if (!buf || !alpha) return AB_ERROR_INVALID_ARG;
    if (n == 0) return AB_OK;
    double ar = alpha[0], ai = alpha[1];
    double* base = (double*)dev_data(buf) + offset * 2;
    for (size_t i = 0; i < n; i++) {
        double r = base[i * 2], im = base[i * 2 + 1];
        base[i * 2]     = ar * r - ai * im;
        base[i * 2 + 1] = ar * im + ai * r;
    }
    return AB_OK;
}

ABStatus ab_dev_axpy_z(size_t n,
                       const double alpha[2],
                       ab_dev_buffer_t x, size_t x_offset, int incx,
                       ab_dev_buffer_t y, size_t y_offset, int incy) {
    if (!x || !y || !alpha) return AB_ERROR_INVALID_ARG;
    if (n == 0) return AB_OK;
    double ar = alpha[0], ai = alpha[1];
    const double* xp = (const double*)dev_data(x) + x_offset * 2;
    double* yp = (double*)dev_data(y) + y_offset * 2;
    for (size_t i = 0; i < n; i++) {
        double xr = xp[i * incx * 2],     xi = xp[i * incx * 2 + 1];
        double yr = yp[i * incy * 2],     yi = yp[i * incy * 2 + 1];
        yp[i * incy * 2]     = yr + (ar * xr - ai * xi);
        yp[i * incy * 2 + 1] = yi + (ar * xi + ai * xr);
    }
    return AB_OK;
}
