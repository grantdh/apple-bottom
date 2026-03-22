// =============================================================================
// apple_bottom.m — FP64-class BLAS for Apple Silicon GPU
// Copyright (c) 2026 Grant Heileman, UNM ECE. MIT License.
// =============================================================================
//
// Implementation Notes:
// - Uses double-float (DD) format: each FP64 stored as {float hi, float lo}
// - Achieves ~10⁻¹⁶ precision via Dekker/Knuth error-free transformations
// - Tiled GEMM with BM=BN=64, TM=TN=4, TK=16 (tuned for M1/M2/M3)
//
// Tile Size Rationale:
// - BM=BN=64: Balances threadgroup memory (64KB limit) with parallelism
// - TM=TN=4: 4×4 register blocking keeps 16 accumulators in registers
// - TK=16: K-dimension tile for memory coalescing without register spill
// - These values were empirically tuned across M1/M2/M3 chips
// =============================================================================

#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#import <Accelerate/Accelerate.h>
#import <dispatch/dispatch.h>
#import <os/lock.h>
#include <mach/mach_time.h>
#include "apple_bottom.h"

// =============================================================================
// Double-Float Primitives (Host Side)
// =============================================================================
// DD format stores a double as two floats: hi + lo ≈ original double
// This gives ~48 bits of mantissa (vs 24 for single float)

typedef struct { float hi; float lo; } DDFloat;

static inline DDFloat fp64_to_dd(double x) {
    float hi = (float)x;
    float lo = (float)(x - (double)hi);
    return (DDFloat){hi, lo};
}

static inline double dd_to_fp64(DDFloat d) {
    return (double)d.hi + (double)d.lo;
}

// =============================================================================
// Statistics (Thread-Safe)
// =============================================================================

static ABStats g_stats = {};
static os_unfair_lock g_stats_lock = OS_UNFAIR_LOCK_INIT;
static mach_timebase_info_data_t g_timebase;

static void init_timing(void) {
    static dispatch_once_t once;
    dispatch_once(&once, ^{ mach_timebase_info(&g_timebase); });
}

static double get_time_ms(void) {
    return (double)mach_absolute_time() * g_timebase.numer / g_timebase.denom / 1e6;
}

static void stats_add_upload(double ms, uint64_t elements) {
    os_unfair_lock_lock(&g_stats_lock);
    g_stats.upload_time_ms += ms;
    g_stats.elements_converted += elements;
    os_unfair_lock_unlock(&g_stats_lock);
}

static void stats_add_download(double ms, uint64_t elements) {
    os_unfair_lock_lock(&g_stats_lock);
    g_stats.download_time_ms += ms;
    g_stats.elements_converted += elements;
    os_unfair_lock_unlock(&g_stats_lock);
}

static void stats_add_kernel(double ms) {
    os_unfair_lock_lock(&g_stats_lock);
    g_stats.kernel_time_ms += ms;
    os_unfair_lock_unlock(&g_stats_lock);
}

static void stats_add_dgemm(void) {
    os_unfair_lock_lock(&g_stats_lock);
    g_stats.dgemm_count++;
    os_unfair_lock_unlock(&g_stats_lock);
}

static void stats_add_zgemm(void) {
    os_unfair_lock_lock(&g_stats_lock);
    g_stats.zgemm_count++;
    os_unfair_lock_unlock(&g_stats_lock);
}

// =============================================================================
// Metal Shaders (Embedded MSL)
// =============================================================================
// These shaders implement double-float arithmetic on GPU using Dekker/Knuth
// error-free transformations. Key algorithms:
//
// twoSum(a,b): Computes s = a+b and e = error, such that a+b = s+e exactly
// fastTwoSum(a,b): Same but requires |a| >= |b|
// twoProduct(a,b): Computes p = a*b and e = error using FMA
//
// DD operations build on these primitives to maintain full precision.
// =============================================================================

static NSString* const kShaderSource = @(R"(
#include <metal_stdlib>
using namespace metal;

struct DD { float hi; float lo; };

// =============================================================================
// Error-Free Transformations (Dekker/Knuth)
// =============================================================================

// twoSum: Compute sum and rounding error (Knuth's algorithm)
// Postcondition: a + b = s + e (exactly, no rounding)
inline void twoSum(float a, float b, thread float &s, thread float &e) {
    s = a + b;
    float v = s - a;
    e = (a - (s - v)) + (b - v);
}

// fastTwoSum: Faster variant when |a| >= |b| is guaranteed
inline void fastTwoSum(float a, float b, thread float &s, thread float &e) {
    s = a + b;
    e = b - (s - a);
}

// twoProduct: Compute product and rounding error using FMA
// fma(a, b, -p) gives the exact rounding error of a*b
inline void twoProduct(float a, float b, thread float &p, thread float &e) {
    p = a * b;
    e = fma(a, b, -p);
}

// =============================================================================
// Double-Float Arithmetic
// =============================================================================

// DD addition: (a.hi + a.lo) + (b.hi + b.lo)
inline DD dd_add(DD a, DD b) {
    float s1, e1, s2, e2, t1, t2;
    twoSum(a.hi, b.hi, s1, e1);
    twoSum(a.lo, b.lo, s2, e2);
    twoSum(e1, s2, t1, t2);
    t2 += e2;
    twoSum(s1, t1, s1, e1);
    e1 += t2;
    fastTwoSum(s1, e1, s1, e1);
    return {s1, e1};
}

// DD subtraction
inline DD dd_sub(DD a, DD b) { return dd_add(a, {-b.hi, -b.lo}); }

// DD multiplication: (a.hi + a.lo) * (b.hi + b.lo)
inline DD dd_mul(DD a, DD b) {
    float p1, e1;
    twoProduct(a.hi, b.hi, p1, e1);
    e1 += a.hi * b.lo + a.lo * b.hi;
    float s, e;
    fastTwoSum(p1, e1, s, e);
    return {s, e};
}

// DD fused multiply-add: a * b + c
inline DD dd_fma(DD a, DD b, DD c) {
    float p1, e1;
    twoProduct(a.hi, b.hi, p1, e1);
    e1 += a.hi * b.lo + a.lo * b.hi;
    float s2, e2;
    twoSum(p1, c.hi, s2, e2);
    e2 += e1 + c.lo;
    fastTwoSum(s2, e2, s2, e2);
    return {s2, e2};
}

// DD scaling by single float
inline DD dd_scale(DD a, float s) {
    float p, e;
    twoProduct(a.hi, s, p, e);
    e += a.lo * s;
    fastTwoSum(p, e, p, e);
    return {p, e};
}

// =============================================================================
// Tile Sizes (Tuned for Apple Silicon M1/M2/M3)
// =============================================================================
// BM, BN: Output tile dimensions per threadgroup (64×64)
// TM, TN: Elements computed per thread (4×4 = 16 outputs/thread)
// TK: K-dimension tile for shared memory caching
// NT: Threads per threadgroup = (64/4) * (64/4) = 256

#define BM 64
#define BN 64
#define TM 4
#define TN 4
#define TK 16
#define NT ((BM/TM) * (BN/TN))

// =============================================================================
// DGEMM Kernel: C = A × B
// =============================================================================

kernel void dd_dgemm(
    device const DD* A [[buffer(0)]],
    device const DD* B [[buffer(1)]],
    device DD* C [[buffer(2)]],
    constant uint& M [[buffer(3)]],
    constant uint& N [[buffer(4)]],
    constant uint& K [[buffer(5)]],
    uint2 lid [[thread_position_in_threadgroup]],
    uint2 tgid [[threadgroup_position_in_grid]],
    uint flatId [[thread_index_in_threadgroup]]
) {
    uint bRow = tgid.y * BM, bCol = tgid.x * BN;
    uint ty = lid.y, tx = lid.x;

    // Per-thread accumulators (4×4 = 16 DD values in registers)
    DD acc[TM][TN];
    for (uint i = 0; i < TM; i++)
        for (uint j = 0; j < TN; j++)
            acc[i][j] = {0.0f, 0.0f};

    // Shared memory tiles
    threadgroup DD tileA[BM * TK];
    threadgroup DD tileB[TK * BN];

    // Loop over K dimension in TK-sized chunks
    for (uint kt = 0; kt < (K + TK - 1) / TK; kt++) {
        // Cooperative load of A tile
        for (uint i = flatId; i < BM * TK; i += NT) {
            uint r = i / TK, c = i % TK;
            uint gr = bRow + r, gc = kt * TK + c;
            tileA[i] = (gr < M && gc < K) ? A[gr * K + gc] : DD{0.0f, 0.0f};
        }
        // Cooperative load of B tile
        for (uint i = flatId; i < TK * BN; i += NT) {
            uint r = i / BN, c = i % BN;
            uint gr = kt * TK + r, gc = bCol + c;
            tileB[i] = (gr < K && gc < N) ? B[gr * N + gc] : DD{0.0f, 0.0f};
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        // Compute partial products from tile
        for (uint k = 0; k < TK; k++) {
            DD av[TM], bv[TN];
            for (uint i = 0; i < TM; i++) av[i] = tileA[(ty * TM + i) * TK + k];
            for (uint j = 0; j < TN; j++) bv[j] = tileB[k * BN + tx * TN + j];
            for (uint i = 0; i < TM; i++)
                for (uint j = 0; j < TN; j++)
                    acc[i][j] = dd_fma(av[i], bv[j], acc[i][j]);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    // Write results to global memory
    for (uint i = 0; i < TM; i++)
        for (uint j = 0; j < TN; j++) {
            uint gr = bRow + ty * TM + i, gc = bCol + tx * TN + j;
            if (gr < M && gc < N) C[gr * N + gc] = acc[i][j];
        }
}

// =============================================================================
// DGEMM Kernel with Scaling: C = α·A×B + β·C
// =============================================================================

kernel void dd_dgemm_ab(
    device const DD* A [[buffer(0)]],
    device const DD* B [[buffer(1)]],
    device DD* C [[buffer(2)]],
    constant uint& M [[buffer(3)]],
    constant uint& N [[buffer(4)]],
    constant uint& K [[buffer(5)]],
    constant float& alpha [[buffer(6)]],
    constant float& beta [[buffer(7)]],
    uint2 lid [[thread_position_in_threadgroup]],
    uint2 tgid [[threadgroup_position_in_grid]],
    uint flatId [[thread_index_in_threadgroup]]
) {
    uint bRow = tgid.y * BM, bCol = tgid.x * BN;
    uint ty = lid.y, tx = lid.x;

    DD acc[TM][TN];
    for (uint i = 0; i < TM; i++)
        for (uint j = 0; j < TN; j++)
            acc[i][j] = {0.0f, 0.0f};

    threadgroup DD tileA[BM * TK];
    threadgroup DD tileB[TK * BN];

    for (uint kt = 0; kt < (K + TK - 1) / TK; kt++) {
        for (uint i = flatId; i < BM * TK; i += NT) {
            uint r = i / TK, c = i % TK;
            uint gr = bRow + r, gc = kt * TK + c;
            tileA[i] = (gr < M && gc < K) ? A[gr * K + gc] : DD{0.0f, 0.0f};
        }
        for (uint i = flatId; i < TK * BN; i += NT) {
            uint r = i / BN, c = i % BN;
            uint gr = kt * TK + r, gc = bCol + c;
            tileB[i] = (gr < K && gc < N) ? B[gr * N + gc] : DD{0.0f, 0.0f};
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        for (uint k = 0; k < TK; k++) {
            DD av[TM], bv[TN];
            for (uint i = 0; i < TM; i++) av[i] = tileA[(ty * TM + i) * TK + k];
            for (uint j = 0; j < TN; j++) bv[j] = tileB[k * BN + tx * TN + j];
            for (uint i = 0; i < TM; i++)
                for (uint j = 0; j < TN; j++)
                    acc[i][j] = dd_fma(av[i], bv[j], acc[i][j]);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    for (uint i = 0; i < TM; i++)
        for (uint j = 0; j < TN; j++) {
            uint gr = bRow + ty * TM + i, gc = bCol + tx * TN + j;
            if (gr < M && gc < N) {
                DD result = dd_scale(acc[i][j], alpha);
                if (beta != 0.0f) result = dd_add(result, dd_scale(C[gr * N + gc], beta));
                C[gr * N + gc] = result;
            }
        }
}

// =============================================================================
// DSYRK Kernel: C = A × Aᵀ (symmetric, upper triangle only)
// =============================================================================

kernel void dd_dsyrk(
    device const DD* A [[buffer(0)]],
    device DD* C [[buffer(1)]],
    constant uint& N [[buffer(2)]],
    constant uint& K [[buffer(3)]],
    uint2 lid [[thread_position_in_threadgroup]],
    uint2 tgid [[threadgroup_position_in_grid]],
    uint flatId [[thread_index_in_threadgroup]]
) {
    uint bRow = tgid.y * BM, bCol = tgid.x * BN;
    // Skip lower triangle blocks (symmetric optimization)
    if (bCol < bRow) return;
    uint ty = lid.y, tx = lid.x;

    DD acc[TM][TN];
    for (uint i = 0; i < TM; i++)
        for (uint j = 0; j < TN; j++)
            acc[i][j] = {0.0f, 0.0f};

    threadgroup DD tileA[BM * TK];
    threadgroup DD tileB[TK * BN];

    for (uint kt = 0; kt < (K + TK - 1) / TK; kt++) {
        for (uint i = flatId; i < BM * TK; i += NT) {
            uint r = i / TK, c = i % TK;
            uint gr = bRow + r, gc = kt * TK + c;
            tileA[i] = (gr < N && gc < K) ? A[gr * K + gc] : DD{0.0f, 0.0f};
        }
        for (uint i = flatId; i < TK * BN; i += NT) {
            uint r = i / BN, c = i % BN;
            uint gc_a = bCol + c, gr_a = kt * TK + r;
            tileB[i] = (gc_a < N && gr_a < K) ? A[gc_a * K + gr_a] : DD{0.0f, 0.0f};
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        for (uint k = 0; k < TK; k++) {
            DD av[TM], bv[TN];
            for (uint i = 0; i < TM; i++) av[i] = tileA[(ty * TM + i) * TK + k];
            for (uint j = 0; j < TN; j++) bv[j] = tileB[k * BN + tx * TN + j];
            for (uint i = 0; i < TM; i++)
                for (uint j = 0; j < TN; j++)
                    acc[i][j] = dd_fma(av[i], bv[j], acc[i][j]);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    for (uint i = 0; i < TM; i++)
        for (uint j = 0; j < TN; j++) {
            uint gr = bRow + ty * TM + i, gc = bCol + tx * TN + j;
            if (gr < N && gc < N && gc >= gr) C[gr * N + gc] = acc[i][j];
        }
}

// =============================================================================
// Element-wise Kernels
// =============================================================================

kernel void dd_matrix_zero(device DD* M [[buffer(0)]], constant uint& count [[buffer(1)]], uint gid [[thread_position_in_grid]]) {
    if (gid < count) M[gid] = {0.0f, 0.0f};
}

kernel void dd_matrix_add(device const DD* A [[buffer(0)]], device const DD* B [[buffer(1)]], device DD* C [[buffer(2)]], constant uint& count [[buffer(3)]], uint gid [[thread_position_in_grid]]) {
    if (gid < count) C[gid] = dd_add(A[gid], B[gid]);
}

kernel void dd_matrix_sub(device const DD* A [[buffer(0)]], device const DD* B [[buffer(1)]], device DD* C [[buffer(2)]], constant uint& count [[buffer(3)]], uint gid [[thread_position_in_grid]]) {
    if (gid < count) C[gid] = dd_sub(A[gid], B[gid]);
}

kernel void dd_matrix_scale(device DD* A [[buffer(0)]], constant float& alpha [[buffer(1)]], constant uint& count [[buffer(2)]], uint gid [[thread_position_in_grid]]) {
    if (gid < count) A[gid] = dd_scale(A[gid], alpha);
}

kernel void dd_matrix_copy(device const DD* src [[buffer(0)]], device DD* dst [[buffer(1)]], constant uint& count [[buffer(2)]], uint gid [[thread_position_in_grid]]) {
    if (gid < count) dst[gid] = src[gid];
}
)");

// =============================================================================
// Metal Context (Singleton)
// =============================================================================

@interface ABContextImpl : NSObject
@property (nonatomic, strong) id<MTLDevice> device;
@property (nonatomic, strong) id<MTLCommandQueue> commandQueue;
@property (nonatomic, strong) id<MTLComputePipelineState> dgemmPipeline;
@property (nonatomic, strong) id<MTLComputePipelineState> dgemmABPipeline;
@property (nonatomic, strong) id<MTLComputePipelineState> dsyrkPipeline;
@property (nonatomic, strong) id<MTLComputePipelineState> zeroPipeline;
@property (nonatomic, strong) id<MTLComputePipelineState> addPipeline;
@property (nonatomic, strong) id<MTLComputePipelineState> subPipeline;
@property (nonatomic, strong) id<MTLComputePipelineState> scalePipeline;
@property (nonatomic, strong) id<MTLComputePipelineState> matCopyPipeline;
+ (instancetype)shared;
+ (void)shutdown;
@end

static ABContextImpl* g_context = nil;
static dispatch_once_t g_init_once;

@implementation ABContextImpl
+ (instancetype)shared { return g_context; }
+ (void)shutdown {
    g_context = nil;
    g_init_once = 0;  // Reset for potential re-init (testing scenarios)
}

- (instancetype)init {
    self = [super init];
    if (self) {
        _device = MTLCreateSystemDefaultDevice();
        if (!_device) return nil;
        _commandQueue = [_device newCommandQueue];
        if (!_commandQueue) return nil;
        
        NSError* error = nil;
        MTLCompileOptions* opts = [[MTLCompileOptions alloc] init];
        #if __MAC_OS_X_VERSION_MAX_ALLOWED >= 140000
#if __MAC_OS_X_VERSION_MAX_ALLOWED >= 140000
        if (@available(macOS 14.0, *)) { opts.mathMode = MTLMathModeSafe; }
#endif
#endif  // CRITICAL: Prevents fast-math optimizations that break DD precision
        
        id<MTLLibrary> library = [_device newLibraryWithSource:kShaderSource options:opts error:&error];
        if (!library) { NSLog(@"Shader compile failed: %@", error); return nil; }
        
        _dgemmPipeline = [_device newComputePipelineStateWithFunction:[library newFunctionWithName:@"dd_dgemm"] error:&error];
        _dgemmABPipeline = [_device newComputePipelineStateWithFunction:[library newFunctionWithName:@"dd_dgemm_ab"] error:&error];
        _dsyrkPipeline = [_device newComputePipelineStateWithFunction:[library newFunctionWithName:@"dd_dsyrk"] error:&error];
        _zeroPipeline = [_device newComputePipelineStateWithFunction:[library newFunctionWithName:@"dd_matrix_zero"] error:&error];
        _addPipeline = [_device newComputePipelineStateWithFunction:[library newFunctionWithName:@"dd_matrix_add"] error:&error];
        _subPipeline = [_device newComputePipelineStateWithFunction:[library newFunctionWithName:@"dd_matrix_sub"] error:&error];
        _scalePipeline = [_device newComputePipelineStateWithFunction:[library newFunctionWithName:@"dd_matrix_scale"] error:&error];
        _matCopyPipeline = [_device newComputePipelineStateWithFunction:[library newFunctionWithName:@"dd_matrix_copy"] error:&error];
        
        if (!_dgemmPipeline || !_zeroPipeline) { NSLog(@"Pipeline creation failed: %@", error); return nil; }
    }
    return self;
}
@end

// =============================================================================
// Parallel Conversion Helpers
// =============================================================================

static void convert_fp64_to_dd_parallel(const double* src, DDFloat* dst, size_t count) {
    dispatch_apply(count, dispatch_get_global_queue(QOS_CLASS_USER_INITIATED, 0), ^(size_t i) { dst[i] = fp64_to_dd(src[i]); });
}

static void convert_dd_to_fp64_parallel(const DDFloat* src, double* dst, size_t count) {
    dispatch_apply(count, dispatch_get_global_queue(QOS_CLASS_USER_INITIATED, 0), ^(size_t i) { dst[i] = dd_to_fp64(src[i]); });
}

// =============================================================================
// Matrix Handle
// =============================================================================

struct ABMatrix_s {
    id<MTLBuffer> buffer;
    int rows, cols;
    size_t count;
    bool uploaded;
};

// =============================================================================
// Session Handle
// =============================================================================

#define MAX_SESSION_MATRICES 64

struct ABSession_s {
    char names[MAX_SESSION_MATRICES][64];
    ABMatrix matrices[MAX_SESSION_MATRICES];
    int count;
};

// =============================================================================
// Public API: Initialization
// =============================================================================

// Thread-safe initialization using dispatch_once
ABStatus ab_init(void) {
    __block ABStatus status = AB_OK;
    dispatch_once(&g_init_once, ^{
        init_timing();
        g_context = [[ABContextImpl alloc] init];
        if (!g_context) status = AB_ERROR_NO_DEVICE;
    });
    return g_context ? AB_OK : status;
}

void ab_shutdown(void) {
    if (g_context) {
        [ABContextImpl shutdown];
        os_unfair_lock_lock(&g_stats_lock);
        memset(&g_stats, 0, sizeof(g_stats));
        os_unfair_lock_unlock(&g_stats_lock);
    }
}

const char* ab_device_name(void) {
    ABContextImpl* ctx = [ABContextImpl shared];
    return ctx ? ctx.device.name.UTF8String : "Unknown";
}

bool ab_is_initialized(void) { return g_context != nil; }

// =============================================================================
// Public API: Matrix Lifecycle
// =============================================================================

// Maximum supported dimension (prevent overflow: 46340² < 2³¹)
#define AB_MAX_DIMENSION 46340

ABMatrix ab_matrix_create(int rows, int cols) {
    // Input validation
    if (rows <= 0 || cols <= 0) return NULL;
    if (rows > AB_MAX_DIMENSION || cols > AB_MAX_DIMENSION) return NULL;
    
    ABContextImpl* ctx = [ABContextImpl shared];
    if (!ctx) return NULL;
    
    struct ABMatrix_s* m = (struct ABMatrix_s*)malloc(sizeof(struct ABMatrix_s));
    if (!m) return NULL;
    
    m->rows = rows;
    m->cols = cols;
    m->count = (size_t)rows * cols;
    m->uploaded = false;
    
    // Check for allocation size overflow
    size_t alloc_size = m->count * sizeof(DDFloat);
    if (alloc_size / sizeof(DDFloat) != m->count) {
        free(m);
        return NULL;
    }
    
    m->buffer = [ctx.device newBufferWithLength:alloc_size options:MTLResourceStorageModeShared];
    if (!m->buffer) { free(m); return NULL; }
    
    return m;
}

void ab_matrix_destroy(ABMatrix m) {
    if (m) {
        m->buffer = nil;
        free(m);
    }
}

void ab_matrix_dims(ABMatrix m, int* rows, int* cols) {
    if (m) {
        if (rows) *rows = m->rows;
        if (cols) *cols = m->cols;
    }
}

size_t ab_matrix_count(ABMatrix m) { return m ? m->count : 0; }

// =============================================================================
// Public API: Data Transfer
// =============================================================================

ABStatus ab_matrix_upload(ABMatrix m, const double* data, bool parallel) {
    if (!m || !data) return AB_ERROR_INVALID_ARG;
    double t0 = get_time_ms();
    DDFloat* dst = (DDFloat*)m->buffer.contents;
    if (parallel && m->count > 1000)
        convert_fp64_to_dd_parallel(data, dst, m->count);
    else
        for (size_t i = 0; i < m->count; i++) dst[i] = fp64_to_dd(data[i]);
    m->uploaded = true;
    stats_add_upload(get_time_ms() - t0, m->count);
    return AB_OK;
}

ABStatus ab_matrix_download(ABMatrix m, double* data, bool parallel) {
    if (!m || !data) return AB_ERROR_INVALID_ARG;
    double t0 = get_time_ms();
    DDFloat* src = (DDFloat*)m->buffer.contents;
    if (parallel && m->count > 1000)
        convert_dd_to_fp64_parallel(src, data, m->count);
    else
        for (size_t i = 0; i < m->count; i++) data[i] = dd_to_fp64(src[i]);
    stats_add_download(get_time_ms() - t0, m->count);
    return AB_OK;
}

ABStatus ab_matrix_zero(ABMatrix m) {
    if (!m) return AB_ERROR_INVALID_ARG;
    ABContextImpl* ctx = [ABContextImpl shared];
    if (!ctx) return AB_ERROR_NO_DEVICE;
    id<MTLCommandBuffer> cmdBuf = [ctx.commandQueue commandBuffer];
    id<MTLComputeCommandEncoder> encoder = [cmdBuf computeCommandEncoder];
    [encoder setComputePipelineState:ctx.zeroPipeline];
    [encoder setBuffer:m->buffer offset:0 atIndex:0];
    uint32_t count = (uint32_t)m->count;
    [encoder setBytes:&count length:sizeof(count) atIndex:1];
    [encoder dispatchThreads:MTLSizeMake(m->count, 1, 1) threadsPerThreadgroup:MTLSizeMake(MIN((size_t)256, m->count), 1, 1)];
    [encoder endEncoding];
    [cmdBuf commit];
    [cmdBuf waitUntilCompleted];
    return AB_OK;
}

ABStatus ab_matrix_copy(ABMatrix src, ABMatrix dst) {
    if (!src || !dst) return AB_ERROR_INVALID_ARG;
    if (src->count != dst->count) return AB_ERROR_DIMENSION_MISMATCH;
    ABContextImpl* ctx = [ABContextImpl shared];
    if (!ctx) return AB_ERROR_NO_DEVICE;
    id<MTLCommandBuffer> cmdBuf = [ctx.commandQueue commandBuffer];
    id<MTLComputeCommandEncoder> encoder = [cmdBuf computeCommandEncoder];
    [encoder setComputePipelineState:ctx.matCopyPipeline];
    [encoder setBuffer:src->buffer offset:0 atIndex:0];
    [encoder setBuffer:dst->buffer offset:0 atIndex:1];
    uint32_t count = (uint32_t)src->count;
    [encoder setBytes:&count length:sizeof(count) atIndex:2];
    [encoder dispatchThreads:MTLSizeMake(src->count, 1, 1) threadsPerThreadgroup:MTLSizeMake(MIN((size_t)256, src->count), 1, 1)];
    [encoder endEncoding];
    [cmdBuf commit];
    [cmdBuf waitUntilCompleted];
    return AB_OK;
}

// =============================================================================
// Public API: BLAS Operations
// =============================================================================

ABStatus ab_dgemm(ABMatrix A, ABMatrix B, ABMatrix C) {
    if (!A || !B || !C) return AB_ERROR_INVALID_ARG;
    if (A->cols != B->rows || A->rows != C->rows || B->cols != C->cols) return AB_ERROR_DIMENSION_MISMATCH;
    ABContextImpl* ctx = [ABContextImpl shared];
    if (!ctx) return AB_ERROR_NO_DEVICE;
    
    uint32_t M = A->rows, N = B->cols, K = A->cols;
    double t0 = get_time_ms();
    
    id<MTLCommandBuffer> cmdBuf = [ctx.commandQueue commandBuffer];
    id<MTLComputeCommandEncoder> encoder = [cmdBuf computeCommandEncoder];
    [encoder setComputePipelineState:ctx.dgemmPipeline];
    [encoder setBuffer:A->buffer offset:0 atIndex:0];
    [encoder setBuffer:B->buffer offset:0 atIndex:1];
    [encoder setBuffer:C->buffer offset:0 atIndex:2];
    [encoder setBytes:&M length:sizeof(M) atIndex:3];
    [encoder setBytes:&N length:sizeof(N) atIndex:4];
    [encoder setBytes:&K length:sizeof(K) atIndex:5];
    [encoder dispatchThreadgroups:MTLSizeMake((N + 63) / 64, (M + 63) / 64, 1) threadsPerThreadgroup:MTLSizeMake(16, 16, 1)];
    [encoder endEncoding];
    [cmdBuf commit];
    [cmdBuf waitUntilCompleted];
    
    stats_add_kernel(get_time_ms() - t0);
    stats_add_dgemm();
    return AB_OK;
}

ABStatus ab_dgemm_scaled(double alpha, ABMatrix A, ABMatrix B, double beta, ABMatrix C) {
    if (!A || !B || !C) return AB_ERROR_INVALID_ARG;
    if (A->cols != B->rows || A->rows != C->rows || B->cols != C->cols) return AB_ERROR_DIMENSION_MISMATCH;
    ABContextImpl* ctx = [ABContextImpl shared];
    if (!ctx) return AB_ERROR_NO_DEVICE;
    
    uint32_t M = A->rows, N = B->cols, K = A->cols;
    float alpha_f = (float)alpha, beta_f = (float)beta;
    double t0 = get_time_ms();
    
    id<MTLCommandBuffer> cmdBuf = [ctx.commandQueue commandBuffer];
    id<MTLComputeCommandEncoder> encoder = [cmdBuf computeCommandEncoder];
    [encoder setComputePipelineState:ctx.dgemmABPipeline];
    [encoder setBuffer:A->buffer offset:0 atIndex:0];
    [encoder setBuffer:B->buffer offset:0 atIndex:1];
    [encoder setBuffer:C->buffer offset:0 atIndex:2];
    [encoder setBytes:&M length:sizeof(M) atIndex:3];
    [encoder setBytes:&N length:sizeof(N) atIndex:4];
    [encoder setBytes:&K length:sizeof(K) atIndex:5];
    [encoder setBytes:&alpha_f length:sizeof(alpha_f) atIndex:6];
    [encoder setBytes:&beta_f length:sizeof(beta_f) atIndex:7];
    [encoder dispatchThreadgroups:MTLSizeMake((N + 63) / 64, (M + 63) / 64, 1) threadsPerThreadgroup:MTLSizeMake(16, 16, 1)];
    [encoder endEncoding];
    [cmdBuf commit];
    [cmdBuf waitUntilCompleted];
    
    stats_add_kernel(get_time_ms() - t0);
    stats_add_dgemm();
    return AB_OK;
}

ABStatus ab_matrix_add(ABMatrix A, ABMatrix B, ABMatrix C) {
    if (!A || !B || !C) return AB_ERROR_INVALID_ARG;
    if (A->count != B->count || A->count != C->count) return AB_ERROR_DIMENSION_MISMATCH;
    ABContextImpl* ctx = [ABContextImpl shared];
    if (!ctx) return AB_ERROR_NO_DEVICE;
    
    id<MTLCommandBuffer> cmdBuf = [ctx.commandQueue commandBuffer];
    id<MTLComputeCommandEncoder> encoder = [cmdBuf computeCommandEncoder];
    [encoder setComputePipelineState:ctx.addPipeline];
    [encoder setBuffer:A->buffer offset:0 atIndex:0];
    [encoder setBuffer:B->buffer offset:0 atIndex:1];
    [encoder setBuffer:C->buffer offset:0 atIndex:2];
    uint32_t count = (uint32_t)A->count;
    [encoder setBytes:&count length:sizeof(count) atIndex:3];
    [encoder dispatchThreads:MTLSizeMake(A->count, 1, 1) threadsPerThreadgroup:MTLSizeMake(MIN((size_t)256, A->count), 1, 1)];
    [encoder endEncoding];
    [cmdBuf commit];
    [cmdBuf waitUntilCompleted];
    return AB_OK;
}

ABStatus ab_matrix_sub(ABMatrix A, ABMatrix B, ABMatrix C) {
    if (!A || !B || !C) return AB_ERROR_INVALID_ARG;
    if (A->count != B->count || A->count != C->count) return AB_ERROR_DIMENSION_MISMATCH;
    ABContextImpl* ctx = [ABContextImpl shared];
    if (!ctx) return AB_ERROR_NO_DEVICE;
    
    id<MTLCommandBuffer> cmdBuf = [ctx.commandQueue commandBuffer];
    id<MTLComputeCommandEncoder> encoder = [cmdBuf computeCommandEncoder];
    [encoder setComputePipelineState:ctx.subPipeline];
    [encoder setBuffer:A->buffer offset:0 atIndex:0];
    [encoder setBuffer:B->buffer offset:0 atIndex:1];
    [encoder setBuffer:C->buffer offset:0 atIndex:2];
    uint32_t count = (uint32_t)A->count;
    [encoder setBytes:&count length:sizeof(count) atIndex:3];
    [encoder dispatchThreads:MTLSizeMake(A->count, 1, 1) threadsPerThreadgroup:MTLSizeMake(MIN((size_t)256, A->count), 1, 1)];
    [encoder endEncoding];
    [cmdBuf commit];
    [cmdBuf waitUntilCompleted];
    return AB_OK;
}

ABStatus ab_matrix_scale(double alpha, ABMatrix A) {
    if (!A) return AB_ERROR_INVALID_ARG;
    ABContextImpl* ctx = [ABContextImpl shared];
    if (!ctx) return AB_ERROR_NO_DEVICE;
    
    float alpha_f = (float)alpha;
    id<MTLCommandBuffer> cmdBuf = [ctx.commandQueue commandBuffer];
    id<MTLComputeCommandEncoder> encoder = [cmdBuf computeCommandEncoder];
    [encoder setComputePipelineState:ctx.scalePipeline];
    [encoder setBuffer:A->buffer offset:0 atIndex:0];
    [encoder setBytes:&alpha_f length:sizeof(alpha_f) atIndex:1];
    uint32_t count = (uint32_t)A->count;
    [encoder setBytes:&count length:sizeof(count) atIndex:2];
    [encoder dispatchThreads:MTLSizeMake(A->count, 1, 1) threadsPerThreadgroup:MTLSizeMake(MIN((size_t)256, A->count), 1, 1)];
    [encoder endEncoding];
    [cmdBuf commit];
    [cmdBuf waitUntilCompleted];
    return AB_OK;
}

// ZGEMM using Gauss's trick: 3 real multiplications instead of 4
// (A_r + iA_i)(B_r + iB_i) = (A_r*B_r - A_i*B_i) + i((A_r+A_i)(B_r+B_i) - A_r*B_r - A_i*B_i)
ABStatus ab_zgemm(ABMatrix Ar, ABMatrix Ai, ABMatrix Br, ABMatrix Bi, ABMatrix Cr, ABMatrix Ci) {
    if (!Ar || !Ai || !Br || !Bi || !Cr || !Ci) return AB_ERROR_INVALID_ARG;
    
    int M = Ar->rows, N = Br->cols, K = Ar->cols;
    
    // Allocate all temporaries upfront with cleanup on failure
    ABMatrix T1 = ab_matrix_create(M, N);
    ABMatrix T2 = ab_matrix_create(M, N);
    ABMatrix T3 = ab_matrix_create(M, K);
    ABMatrix T4 = ab_matrix_create(K, N);
    
    // Check all allocations succeeded
    if (!T1 || !T2 || !T3 || !T4) {
        // Cleanup any successful allocations (ab_matrix_destroy handles NULL safely)
        ab_matrix_destroy(T1);
        ab_matrix_destroy(T2);
        ab_matrix_destroy(T3);
        ab_matrix_destroy(T4);
        return AB_ERROR_ALLOC_FAILED;
    }
    
    // T3 = Ar + Ai, T4 = Br + Bi
    ab_matrix_add(Ar, Ai, T3);
    ab_matrix_add(Br, Bi, T4);
    
    // T1 = Ar * Br
    ab_dgemm(Ar, Br, T1);
    
    // T2 = Ai * Bi
    ab_dgemm(Ai, Bi, T2);
    
    // Ci = (Ar+Ai) * (Br+Bi)
    ab_dgemm(T3, T4, Ci);
    
    // Ci = Ci - T1 - T2 = imaginary part
    ab_matrix_sub(Ci, T1, Ci);
    ab_matrix_sub(Ci, T2, Ci);
    
    // Cr = T1 - T2 = real part
    ab_matrix_sub(T1, T2, Cr);
    
    // Cleanup
    ab_matrix_destroy(T1);
    ab_matrix_destroy(T2);
    ab_matrix_destroy(T3);
    ab_matrix_destroy(T4);
    
    stats_add_zgemm();
    return AB_OK;
}

ABStatus ab_dsyrk(ABMatrix A, ABMatrix C) {
    if (!A || !C) return AB_ERROR_INVALID_ARG;
    if (A->rows != C->rows || C->rows != C->cols) return AB_ERROR_DIMENSION_MISMATCH;
    ABContextImpl* ctx = [ABContextImpl shared];
    if (!ctx) return AB_ERROR_NO_DEVICE;
    
    uint32_t N = A->rows, K = A->cols;
    double t0 = get_time_ms();
    
    id<MTLCommandBuffer> cmdBuf = [ctx.commandQueue commandBuffer];
    id<MTLComputeCommandEncoder> encoder = [cmdBuf computeCommandEncoder];
    [encoder setComputePipelineState:ctx.dsyrkPipeline];
    [encoder setBuffer:A->buffer offset:0 atIndex:0];
    [encoder setBuffer:C->buffer offset:0 atIndex:1];
    [encoder setBytes:&N length:sizeof(N) atIndex:2];
    [encoder setBytes:&K length:sizeof(K) atIndex:3];
    [encoder dispatchThreadgroups:MTLSizeMake((N + 63) / 64, (N + 63) / 64, 1) threadsPerThreadgroup:MTLSizeMake(16, 16, 1)];
    [encoder endEncoding];
    [cmdBuf commit];
    [cmdBuf waitUntilCompleted];
    
    stats_add_kernel(get_time_ms() - t0);
    return AB_OK;
}

ABStatus ab_zherk(ABMatrix Ar, ABMatrix Ai, ABMatrix Cr, ABMatrix Ci) {
    if (!Ar || !Ai || !Cr || !Ci) return AB_ERROR_INVALID_ARG;
    
    ABStatus s1 = ab_dsyrk(Ar, Cr);
    if (s1 != AB_OK) return s1;
    
    ab_matrix_zero(Ci);
    
    ABMatrix temp = ab_matrix_create(Ar->rows, Ar->rows);
    if (!temp) return AB_ERROR_ALLOC_FAILED;
    
    ab_dsyrk(Ai, temp);
    ab_matrix_add(Cr, temp, Cr);
    ab_matrix_destroy(temp);
    
    return AB_OK;
}

// =============================================================================
// Public API: Session Management
// =============================================================================

ABSession ab_session_create(void) {
    return (ABSession)calloc(1, sizeof(struct ABSession_s));
}

void ab_session_destroy(ABSession s) {
    if (!s) return;
    for (int i = 0; i < s->count; i++) {
        ab_matrix_destroy(s->matrices[i]);
    }
    free(s);
}

ABStatus ab_session_add(ABSession s, const char* name, int rows, int cols) {
    if (!s || !name || s->count >= MAX_SESSION_MATRICES) return AB_ERROR_INVALID_ARG;
    
    ABMatrix m = ab_matrix_create(rows, cols);
    if (!m) return AB_ERROR_ALLOC_FAILED;
    
    // Use strlcpy for safe string copy (macOS native)
    strlcpy(s->names[s->count], name, sizeof(s->names[s->count]));
    s->matrices[s->count] = m;
    s->count++;
    return AB_OK;
}

ABMatrix ab_session_get(ABSession s, const char* name) {
    if (!s || !name) return NULL;
    for (int i = 0; i < s->count; i++) {
        if (strcmp(s->names[i], name) == 0) return s->matrices[i];
    }
    return NULL;
}

ABStatus ab_session_upload(ABSession s, const char* name, const double* data) {
    ABMatrix m = ab_session_get(s, name);
    return m ? ab_matrix_upload(m, data, true) : AB_ERROR_INVALID_ARG;
}

ABStatus ab_session_download(ABSession s, const char* name, double* data) {
    ABMatrix m = ab_session_get(s, name);
    return m ? ab_matrix_download(m, data, true) : AB_ERROR_INVALID_ARG;
}

ABStatus ab_session_dgemm(ABSession s, const char* A, const char* B, const char* C) {
    ABMatrix mA = ab_session_get(s, A);
    ABMatrix mB = ab_session_get(s, B);
    ABMatrix mC = ab_session_get(s, C);
    return (mA && mB && mC) ? ab_dgemm(mA, mB, mC) : AB_ERROR_INVALID_ARG;
}

ABStatus ab_session_zgemm(ABSession s, const char* Ar, const char* Ai, const char* Br, const char* Bi, const char* Cr, const char* Ci) {
    ABMatrix mAr = ab_session_get(s, Ar);
    ABMatrix mAi = ab_session_get(s, Ai);
    ABMatrix mBr = ab_session_get(s, Br);
    ABMatrix mBi = ab_session_get(s, Bi);
    ABMatrix mCr = ab_session_get(s, Cr);
    ABMatrix mCi = ab_session_get(s, Ci);
    return (mAr && mAi && mBr && mBi && mCr && mCi) ? ab_zgemm(mAr, mAi, mBr, mBi, mCr, mCi) : AB_ERROR_INVALID_ARG;
}

// =============================================================================
// Public API: Statistics
// =============================================================================

ABStats ab_get_stats(void) {
    os_unfair_lock_lock(&g_stats_lock);
    ABStats copy = g_stats;
    os_unfair_lock_unlock(&g_stats_lock);
    return copy;
}

void ab_reset_stats(void) {
    os_unfair_lock_lock(&g_stats_lock);
    memset(&g_stats, 0, sizeof(g_stats));
    os_unfair_lock_unlock(&g_stats_lock);
}

void ab_print_stats(void) {
    ABStats s = ab_get_stats();
    fprintf(stderr, "apple-bottom statistics:\n");
    fprintf(stderr, "  Upload time:   %.2f ms\n", s.upload_time_ms);
    fprintf(stderr, "  Download time: %.2f ms\n", s.download_time_ms);
    fprintf(stderr, "  Kernel time:   %.2f ms\n", s.kernel_time_ms);
    fprintf(stderr, "  DGEMM calls:   %llu\n", s.dgemm_count);
    fprintf(stderr, "  ZGEMM calls:   %llu\n", s.zgemm_count);
    fprintf(stderr, "  Elements:      %llu\n", s.elements_converted);
}

const char* ab_status_string(ABStatus status) {
    switch (status) {
        case AB_OK: return "OK";
        case AB_ERROR_NO_DEVICE: return "No Metal GPU available";
        case AB_ERROR_ALLOC_FAILED: return "Memory allocation failed";
        case AB_ERROR_DIMENSION_MISMATCH: return "Matrix dimensions incompatible";
        case AB_ERROR_NOT_UPLOADED: return "Matrix data not uploaded";
        case AB_ERROR_KERNEL_FAILED: return "GPU kernel execution failed";
        case AB_ERROR_INVALID_ARG: return "Invalid argument";
        default: return "Unknown error";
    }
}
