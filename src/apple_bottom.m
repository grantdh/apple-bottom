// =============================================================================
// apple_bottom.m — FP64-class BLAS for Apple Silicon GPU
// Copyright (c) 2026 Grant Heileman, Technology Residue. MIT License.
// =============================================================================
//
// Implementation Notes:
// - Uses double-float (DD) format: each FP64 stored as {float hi, float lo}
// - Achieves ~10⁻¹⁵ precision via Dekker/Knuth error-free transformations
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
#include <fenv.h>
#include <limits.h>
#include <math.h>
#include <mach/mach_time.h>
#include <pthread.h>
#include "apple_bottom.h"
#include "ab_internal.h"
#include "profiling/blas_profiler.h"

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

// GPU execution time of the command buffers this thread completed, for the
// kernel stage of the BLAS-ABI profile (blas_wrapper.c), which takes it per call.
static thread_local double t_gpu_seconds = 0;

// The rest of the host time of the calls that completed them (temporaries,
// encoding, submit and wake-up latency) and when the GPU ran, for the AB_TRACE
// split of the profile's "other"; taken with t_gpu_seconds.
static thread_local ab_prof_kernel_split_t t_kernel_split;

// The clock of the split: mach_absolute_time in ns, converted as the trace's
// call times are, the time base that GPUStartTime and GPUEndTime give in
// seconds.
static uint64_t host_ns(void) { return ab_prof_ticks_to_ns(ab_prof_now_ticks()); }

static uint64_t gpu_time_ns(CFTimeInterval s) { return (uint64_t)(s * 1e9 + 0.5); }

// cmdBuf was committed at commit_ns, and waitUntilCompleted returned at done_ns.
static void note_gpu_time(id<MTLCommandBuffer> cmdBuf, uint64_t commit_ns, uint64_t done_ns) {
    CFTimeInterval start = cmdBuf.GPUStartTime, end = cmdBuf.GPUEndTime;
    if (!(start > 0 && end > start)) return;
    t_gpu_seconds += end - start;
    uint64_t start_ns = gpu_time_ns(start), end_ns = gpu_time_ns(end);
    ab_prof_kernel_split_t *k = &t_kernel_split;
    k->submit_ns += start_ns > commit_ns ? start_ns - commit_ns : 0;
    k->wake_ns += done_ns > end_ns ? done_ns - end_ns : 0;
    if (!k->gpu_start_ns) k->gpu_start_ns = start_ns;
    k->gpu_end_ns = end_ns;
}

// Commits cmdBuf, which encoding started to build at encode_ns, waits for it
// and notes its times. False if it failed; AB_TEST_FAIL_STAGE=kernel
// (test-only) reports every one as failed.
static bool commit_and_wait(id<MTLCommandBuffer> cmdBuf, uint64_t encode_ns) {
    uint64_t commit_ns = host_ns();
    t_kernel_split.encode_ns += commit_ns - encode_ns;
    [cmdBuf commit];
    [cmdBuf waitUntilCompleted];
    note_gpu_time(cmdBuf, commit_ns, host_ns());
    return cmdBuf.status != MTLCommandBufferStatusError &&
           ab_test_fail_stage() != AB_PROF_STAGE_KERNEL;
}

extern "C" double ab_internal_take_gpu_seconds(ab_prof_kernel_split_t *split) {
    double s = t_gpu_seconds;
    t_gpu_seconds = 0;
    if (split) *split = t_kernel_split;
    t_kernel_split = ab_prof_kernel_split_t();
    return s;
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

// DD multiplication and FMA.
//
// Default build: DWTimesDW2 (Joldes--Muller--Popescu 2017, Algorithm 11).
//   Error bound: < 5u^2 / (1+u)^2 < 5u^2  (Muller--Rideau 2022, Theorem 2.7).
//
// Opt-in build (-DAPPLEBOTTOM_USE_DWTIMESDW3): DWTimesDW3 (JMP 2017, Algorithm 12).
//   Error bound: < 4u^2  (Muller--Rideau 2022, Theorem 2.8).
//   Adds one multiplication (al*bl) and one add vs Alg 11; FMA count
//   unchanged. The tighter bound comes from including the a.lo*b.lo
//   cross-term that Alg 11 drops, which is what costs the extra mul+add.
//
// Compile-flag gating is coordinated via MTLCompileOptions.preprocessorMacros;
// see the init method below. Default remains DWTimesDW2.
#ifdef APPLEBOTTOM_USE_DWTIMESDW3

inline DD dd_mul(DD a, DD b) {
    float ch, cl1;
    twoProduct(a.hi, b.hi, ch, cl1);
    float tl0 = a.lo * b.lo;
    float tl1 = fma(a.hi, b.lo, tl0);
    float cl2 = fma(a.lo, b.hi, tl1);
    float cl3 = cl1 + cl2;
    float s, e;
    fastTwoSum(ch, cl3, s, e);
    return {s, e};
}

inline DD dd_fma(DD a, DD b, DD c) {
    float ch, cl1;
    twoProduct(a.hi, b.hi, ch, cl1);
    float tl0 = a.lo * b.lo;
    float tl1 = fma(a.hi, b.lo, tl0);
    float cl2 = fma(a.lo, b.hi, tl1);
    float cl3 = cl1 + cl2;
    float s2, e2;
    twoSum(ch, c.hi, s2, e2);
    e2 += cl3 + c.lo;
    fastTwoSum(s2, e2, s2, e2);
    return {s2, e2};
}

#else  // DWTimesDW2 (default)

inline DD dd_mul(DD a, DD b) {
    float p1, e1;
    twoProduct(a.hi, b.hi, p1, e1);
    e1 = fma(a.hi, b.lo, fma(a.lo, b.hi, e1));
    float s, e;
    fastTwoSum(p1, e1, s, e);
    return {s, e};
}

inline DD dd_fma(DD a, DD b, DD c) {
    float p1, e1;
    twoProduct(a.hi, b.hi, p1, e1);
    e1 = fma(a.hi, b.lo, fma(a.lo, b.hi, e1));
    float s2, e2;
    twoSum(p1, c.hi, s2, e2);
    e2 += e1 + c.lo;
    fastTwoSum(s2, e2, s2, e2);
    return {s2, e2};
}

#endif  // APPLEBOTTOM_USE_DWTIMESDW3

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

// Square tiling (default): 64×64 blocks, 256 threads
#define BM 64
#define BN 64
#define TM 4
#define TN 4
#define TK 16
#define NT ((BM/TM) * (BN/TN))

// Tall-skinny tiling: 128×16 blocks, 256 threads
// For M >> N with N = 150: boundary waste in N drops from ~22% (3 tiles of 64)
// to ~6% (10 tiles of 16)
#define BM_TS 128
#define BN_TS 16
#define TM_TS 4
#define TN_TS 2
#define TK_TS 16
#define NT_TS ((BM_TS/TM_TS) * (BN_TS/TN_TS))

// Narrow-output tiling: 32×32 blocks, 256 threads with 2×2 outputs each.
// Four times the threadgroups of the square tiling, for grids small enough
// that finer tiles spread better over the GPU's cores (routing:
// use_narrow_tiles, host side)
#define BM_NW 32
#define BN_NW 32
#define TM_NW 2
#define TN_NW 2
#define NT_NW ((BM_NW/TM_NW) * (BN_NW/TN_NW))

// The tiled kernels hold their threadgroup tiles k-major: tileA[k][row] (A is
// transposed on its way in) and tileB[k][col]. A thread's TM values of A and
// TN values of B at one k are then contiguous, and it reads them as float4
// words, two DD each; the tiles are float4 arrays so those reads are 16-byte
// aligned. The tall-skinny kernel keeps A row-major: transposing its 128-row
// tile costs more in conflicting threadgroup stores than the float4 reads
// save (measured on M2 Max).
static_assert(TM % 2 == 0 && TN % 2 == 0 && TM_NW % 2 == 0 && TN_NW % 2 == 0,
              "float4 tile reads take DD values in pairs");
// The narrow kernels stand in for the tall-skinny one: same K-tile, so they
// renormalize at the same k and give the same bits
static_assert(TK_TS == TK, "narrow and tall-skinny kernels share the K tiling");

// Block-wise compensated accumulation: renormalize accumulators every
// RENORM_INTERVAL K-tiles to prevent drift in long dot products.
// Per Wilkinson's probabilistic analysis, Frobenius error scales as O(sqrt(K)).
// Periodic twoSum renormalization ensures the DD representation stays
// well-conditioned, preventing edge-case denormalization when magnitudes
// shift during long accumulations. (Higham 2002, §4.2)
#define RENORM_INTERVAL 8  // Every 8 tiles × TK=16 = 128 K-elements

// =============================================================================
// Morton Z-Order Threadgroup Remapping
// =============================================================================
// Maps linear threadgroup dispatch to a Z-order (Morton) space-filling curve.
// Adjacent threadgroups process spatially adjacent output tiles, maximizing
// SLC (System Level Cache) hit rate by preserving 2D locality.
//
// Without Morton ordering: row-major scan causes cache thrashing on large
// matrices as distant threadgroups evict each other's A/B tile shards.
// With Morton ordering: ~15% bandwidth savings on N >= 2048 (empirical).
//
// Implementation: deinterleave bits of the linear threadgroup index to
// produce (x, y) coordinates following a Z-curve pattern.

inline uint2 morton_remap(uint2 tgid, uint gridW, uint gridH) {
    // Morton Z-order bit-deinterleave is ONLY a bijection for power-of-2 grids.
    // For non-power-of-2 grids (e.g. 12×12 for N=768), many linear indices map
    // to out-of-bounds (x,y), and falling back to identity causes collisions:
    // some output blocks get computed twice, others never — catastrophic errors.
    //
    // Even with power-of-2 dims, W*H = 2^n gives x the ceil(n/2) even bits and
    // y the floor(n/2) odd bits, so the curve covers the grid only when
    // gridW == gridH or gridW == 2*gridH. Other shapes (e.g. the 4×8 grid of
    // a 512×256 C) leave tiles unwritten.
    //
    // Guard: require both dims to be powers of 2, >= 4, and one of those two
    // aspect ratios. This covers the common case (N = 1024, 2048, 4096) where
    // Morton gives ~15% bandwidth savings, and disables it for all other sizes.
    bool wPow2 = (gridW & (gridW - 1)) == 0;
    bool hPow2 = (gridH & (gridH - 1)) == 0;
    if (!wPow2 || !hPow2 || gridW < 4 || gridH < 4) return tgid;
    if (gridW != gridH && gridW != 2 * gridH) return tgid;

    // Linearize the 2D threadgroup position
    uint linear = tgid.y * gridW + tgid.x;
    uint total = gridW * gridH;
    if (linear >= total) return tgid;

    // Deinterleave bits: even bits → x, odd bits → y (Z-order curve)
    uint x = 0, y = 0;
    for (uint i = 0; i < 16; i++) {
        x |= ((linear >> (2 * i + 0)) & 1) << i;
        y |= ((linear >> (2 * i + 1)) & 1) << i;
    }

    return uint2(x, y);
}

// =============================================================================
// Register Prefetch of the Next K-Tile (the *_pf kernels)
// =============================================================================
// The tiled kernels load each K-tile from device memory into threadgroup
// memory and wait at a barrier for those loads. On a small grid, one or two
// threadgroups per GPU core, nothing else runs to hide that latency. Each
// tiled kernel is a template on PF: its PF instance (host name *_pf) loads
// K-tile kt+1 into registers right after the barrier that publishes tile kt,
// so the loads overlap tile kt's arithmetic, and stores those registers to
// threadgroup memory at the top of the next iteration. Every tile element
// reaches the same threadgroup slot as in the plain instance and the inner
// loop is the same code, so each output gets the same dd_fma sequence and the
// same bits. The registers cost occupancy: the PF instances are faster only
// on small grids (routing: use_prefetch, host side).

// One thread's share of a ROWS×COLS tile at (row0, col0) of the row-major
// rows×cols matrix X: element flatId + q·NTH of the tile in q, 0 outside X.
// These are the elements the plain kernels' cooperative loads
// `for (i = flatId; i < ROWS*COLS; i += NTH)` read, in the same order.
template <uint ROWS, uint COLS, uint NTH>
inline void dd_fetch_tile(thread DD (&r)[ROWS * COLS / NTH], device const DD* X,
                          uint rows, uint cols, uint row0, uint col0, uint flatId) {
    static_assert(ROWS * COLS % NTH == 0, "every thread loads the same number of elements");
    for (uint q = 0; q < ROWS * COLS / NTH; q++) {
        uint i = flatId + q * NTH, gr = row0 + i / COLS, gc = col0 + i % COLS;
        r[q] = (gr < rows && gc < cols) ? X[gr * cols + gc] : DD{0.0f, 0.0f};
    }
}

// Stores a fetched share to a row-major tile (tile[i]) or to a k-major one,
// tile[i % COLS][i / COLS] (the A tiles of the float4 kernels, and dd_dsyrk's
// B = Aᵀ tile, fetched as the row block of A it transposes).
template <uint ROWS, uint COLS, uint NTH>
inline void dd_put_tile(threadgroup DD* tile, thread const DD (&r)[ROWS * COLS / NTH], uint flatId) {
    for (uint q = 0; q < ROWS * COLS / NTH; q++) tile[flatId + q * NTH] = r[q];
}
template <uint ROWS, uint COLS, uint NTH>
inline void dd_put_tile_kmajor(threadgroup DD* tile, thread const DD (&r)[ROWS * COLS / NTH], uint flatId) {
    for (uint q = 0; q < ROWS * COLS / NTH; q++) {
        uint i = flatId + q * NTH;
        tile[(i % COLS) * ROWS + i / COLS] = r[q];
    }
}

// =============================================================================
// DGEMM Kernel: C = A × B
// =============================================================================

template <bool PF>
kernel void dd_dgemm_t(
    device const DD* A [[buffer(0)]],
    device const DD* B [[buffer(1)]],
    device DD* C [[buffer(2)]],
    constant uint& M [[buffer(3)]],
    constant uint& N [[buffer(4)]],
    constant uint& K [[buffer(5)]],
    constant uint& gridW [[buffer(6)]],
    constant uint& gridH [[buffer(7)]],
    uint2 lid [[thread_position_in_threadgroup]],
    uint2 tgid [[threadgroup_position_in_grid]],
    uint flatId [[thread_index_in_threadgroup]]
) {
    // Morton Z-order remapping for SLC cache locality
    uint2 mtid = morton_remap(tgid, gridW, gridH);
    uint bRow = mtid.y * BM, bCol = mtid.x * BN;
    uint ty = lid.y, tx = lid.x;

    // Per-thread accumulators (4×4 = 16 DD values in registers)
    DD acc[TM][TN];
    for (uint i = 0; i < TM; i++)
        for (uint j = 0; j < TN; j++)
            acc[i][j] = {0.0f, 0.0f};

    // Shared memory tiles (k-major, see Tile Sizes)
    threadgroup float4 tileA4[BM * TK / 2];
    threadgroup float4 tileB4[TK * BN / 2];
    threadgroup DD* tileA = (threadgroup DD*)tileA4;
    threadgroup DD* tileB = (threadgroup DD*)tileB4;

    // PF: this thread's share of the next K-tile (see Register Prefetch)
    DD ra[BM * TK / NT], rb[TK * BN / NT];
    if (PF) {
        dd_fetch_tile<BM, TK, NT>(ra, A, M, K, bRow, 0, flatId);
        dd_fetch_tile<TK, BN, NT>(rb, B, K, N, 0, bCol, flatId);
    }

    // Loop over K dimension in TK-sized chunks
    for (uint kt = 0; kt < (K + TK - 1) / TK; kt++) {
        if (PF) {
            dd_put_tile_kmajor<BM, TK, NT>(tileA, ra, flatId);
            dd_put_tile<TK, BN, NT>(tileB, rb, flatId);
        } else {
            // Cooperative load of A tile
            for (uint i = flatId; i < BM * TK; i += NT) {
                uint r = i / TK, c = i % TK;
                uint gr = bRow + r, gc = kt * TK + c;
                tileA[c * BM + r] = (gr < M && gc < K) ? A[gr * K + gc] : DD{0.0f, 0.0f};
            }
            // Cooperative load of B tile
            for (uint i = flatId; i < TK * BN; i += NT) {
                uint r = i / BN, c = i % BN;
                uint gr = kt * TK + r, gc = bCol + c;
                tileB[i] = (gr < K && gc < N) ? B[gr * N + gc] : DD{0.0f, 0.0f};
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (PF && (kt + 1) * TK < K) {
            dd_fetch_tile<BM, TK, NT>(ra, A, M, K, bRow, (kt + 1) * TK, flatId);
            dd_fetch_tile<TK, BN, NT>(rb, B, K, N, (kt + 1) * TK, bCol, flatId);
        }

        // Compute partial products from tile
        for (uint k = 0; k < TK; k++) {
            DD av[TM], bv[TN];
            for (uint h = 0; h < TM / 2; h++) {
                float4 a = tileA4[(k * BM + ty * TM) / 2 + h];
                av[2 * h] = {a.x, a.y}; av[2 * h + 1] = {a.z, a.w};
            }
            for (uint h = 0; h < TN / 2; h++) {
                float4 b = tileB4[(k * BN + tx * TN) / 2 + h];
                bv[2 * h] = {b.x, b.y}; bv[2 * h + 1] = {b.z, b.w};
            }
            for (uint i = 0; i < TM; i++)
                for (uint j = 0; j < TN; j++)
                    acc[i][j] = dd_fma(av[i], bv[j], acc[i][j]);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        // Block-wise compensated accumulation: periodic renormalization
        if ((kt & (RENORM_INTERVAL - 1)) == (RENORM_INTERVAL - 1)) {
            for (uint i = 0; i < TM; i++)
                for (uint j = 0; j < TN; j++) {
                    float s, e;
                    twoSum(acc[i][j].hi, acc[i][j].lo, s, e);
                    acc[i][j] = {s, e};
                }
        }
    }

    // Final renormalization before store
    for (uint i = 0; i < TM; i++)
        for (uint j = 0; j < TN; j++) {
            float s, e;
            twoSum(acc[i][j].hi, acc[i][j].lo, s, e);
            acc[i][j] = {s, e};
        }

    // Write results to global memory
    for (uint i = 0; i < TM; i++)
        for (uint j = 0; j < TN; j++) {
            uint gr = bRow + ty * TM + i, gc = bCol + tx * TN + j;
            if (gr < M && gc < N) C[gr * N + gc] = acc[i][j];
        }
}

typedef decltype(dd_dgemm_t<false>) dd_dgemm_fn;
template [[host_name("dd_dgemm")]] kernel dd_dgemm_fn dd_dgemm_t<false>;
template [[host_name("dd_dgemm_pf")]] kernel dd_dgemm_fn dd_dgemm_t<true>;

// =============================================================================
// DGEMM Kernel with Scaling: C = α·A×B + β·C
// =============================================================================

template <bool PF>
kernel void dd_dgemm_ab_t(
    device const DD* A [[buffer(0)]],
    device const DD* B [[buffer(1)]],
    device DD* C [[buffer(2)]],
    constant uint& M [[buffer(3)]],
    constant uint& N [[buffer(4)]],
    constant uint& K [[buffer(5)]],
    constant DD& alpha [[buffer(6)]],
    constant DD& beta [[buffer(7)]],
    constant uint& gridW [[buffer(8)]],
    constant uint& gridH [[buffer(9)]],
    uint2 lid [[thread_position_in_threadgroup]],
    uint2 tgid [[threadgroup_position_in_grid]],
    uint flatId [[thread_index_in_threadgroup]]
) {
    // Morton Z-order remapping for SLC cache locality
    uint2 mtid = morton_remap(tgid, gridW, gridH);
    uint bRow = mtid.y * BM, bCol = mtid.x * BN;
    uint ty = lid.y, tx = lid.x;

    DD acc[TM][TN];
    for (uint i = 0; i < TM; i++)
        for (uint j = 0; j < TN; j++)
            acc[i][j] = {0.0f, 0.0f};

    // Shared memory tiles (k-major, see Tile Sizes)
    threadgroup float4 tileA4[BM * TK / 2];
    threadgroup float4 tileB4[TK * BN / 2];
    threadgroup DD* tileA = (threadgroup DD*)tileA4;
    threadgroup DD* tileB = (threadgroup DD*)tileB4;

    // PF: this thread's share of the next K-tile (see Register Prefetch)
    DD ra[BM * TK / NT], rb[TK * BN / NT];
    if (PF) {
        dd_fetch_tile<BM, TK, NT>(ra, A, M, K, bRow, 0, flatId);
        dd_fetch_tile<TK, BN, NT>(rb, B, K, N, 0, bCol, flatId);
    }

    for (uint kt = 0; kt < (K + TK - 1) / TK; kt++) {
        if (PF) {
            dd_put_tile_kmajor<BM, TK, NT>(tileA, ra, flatId);
            dd_put_tile<TK, BN, NT>(tileB, rb, flatId);
        } else {
            for (uint i = flatId; i < BM * TK; i += NT) {
                uint r = i / TK, c = i % TK;
                uint gr = bRow + r, gc = kt * TK + c;
                tileA[c * BM + r] = (gr < M && gc < K) ? A[gr * K + gc] : DD{0.0f, 0.0f};
            }
            for (uint i = flatId; i < TK * BN; i += NT) {
                uint r = i / BN, c = i % BN;
                uint gr = kt * TK + r, gc = bCol + c;
                tileB[i] = (gr < K && gc < N) ? B[gr * N + gc] : DD{0.0f, 0.0f};
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (PF && (kt + 1) * TK < K) {
            dd_fetch_tile<BM, TK, NT>(ra, A, M, K, bRow, (kt + 1) * TK, flatId);
            dd_fetch_tile<TK, BN, NT>(rb, B, K, N, (kt + 1) * TK, bCol, flatId);
        }

        for (uint k = 0; k < TK; k++) {
            DD av[TM], bv[TN];
            for (uint h = 0; h < TM / 2; h++) {
                float4 a = tileA4[(k * BM + ty * TM) / 2 + h];
                av[2 * h] = {a.x, a.y}; av[2 * h + 1] = {a.z, a.w};
            }
            for (uint h = 0; h < TN / 2; h++) {
                float4 b = tileB4[(k * BN + tx * TN) / 2 + h];
                bv[2 * h] = {b.x, b.y}; bv[2 * h + 1] = {b.z, b.w};
            }
            for (uint i = 0; i < TM; i++)
                for (uint j = 0; j < TN; j++)
                    acc[i][j] = dd_fma(av[i], bv[j], acc[i][j]);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        // Block-wise compensated accumulation: periodic renormalization
        // Prevents accumulator drift for large K by ensuring the DD
        // hi/lo split stays well-conditioned after many FMA iterations.
        if ((kt & (RENORM_INTERVAL - 1)) == (RENORM_INTERVAL - 1)) {
            for (uint i = 0; i < TM; i++)
                for (uint j = 0; j < TN; j++) {
                    float s, e;
                    twoSum(acc[i][j].hi, acc[i][j].lo, s, e);
                    acc[i][j] = {s, e};
                }
        }
    }

    // Final renormalization before epilogue scaling
    for (uint i = 0; i < TM; i++)
        for (uint j = 0; j < TN; j++) {
            float s, e;
            twoSum(acc[i][j].hi, acc[i][j].lo, s, e);
            acc[i][j] = {s, e};
        }

    for (uint i = 0; i < TM; i++)
        for (uint j = 0; j < TN; j++) {
            uint gr = bRow + ty * TM + i, gc = bCol + tx * TN + j;
            if (gr < M && gc < N) {
                DD result = dd_mul(acc[i][j], alpha);
                if (beta.hi != 0.0f || beta.lo != 0.0f) result = dd_add(result, dd_mul(C[gr * N + gc], beta));
                C[gr * N + gc] = result;
            }
        }
}

typedef decltype(dd_dgemm_ab_t<false>) dd_dgemm_ab_fn;
template [[host_name("dd_dgemm_ab")]] kernel dd_dgemm_ab_fn dd_dgemm_ab_t<false>;
template [[host_name("dd_dgemm_ab_pf")]] kernel dd_dgemm_ab_fn dd_dgemm_ab_t<true>;

// =============================================================================
// DGEMM Kernel (Tall-Skinny): C = α·A×B + β·C — optimized for M >> N
// BM_TS=128, BN_TS=16, TM_TS=4, TN_TS=2, 256 threads per threadgroup
// =============================================================================

template <bool PF>
kernel void dd_dgemm_ab_ts_t(
    device const DD* A [[buffer(0)]],
    device const DD* B [[buffer(1)]],
    device DD* C [[buffer(2)]],
    constant uint& M [[buffer(3)]],
    constant uint& N [[buffer(4)]],
    constant uint& K [[buffer(5)]],
    constant DD& alpha [[buffer(6)]],
    constant DD& beta [[buffer(7)]],
    constant uint& gridW [[buffer(8)]],
    constant uint& gridH [[buffer(9)]],
    uint2 lid [[thread_position_in_threadgroup]],
    uint2 tgid [[threadgroup_position_in_grid]],
    uint flatId [[thread_index_in_threadgroup]]
) {
    // Morton Z-order remapping for SLC cache locality
    uint2 mtid = morton_remap(tgid, gridW, gridH);
    uint bRow = mtid.y * BM_TS, bCol = mtid.x * BN_TS;
    uint ty = lid.y, tx = lid.x;

    DD acc[TM_TS][TN_TS];
    for (uint i = 0; i < TM_TS; i++)
        for (uint j = 0; j < TN_TS; j++)
            acc[i][j] = {0.0f, 0.0f};

    // Row-major A tile on purpose: k-major measured slower here (see Tile Sizes)
    threadgroup DD tileA[BM_TS * TK_TS];
    threadgroup DD tileB[TK_TS * BN_TS];

    // PF: this thread's share of the next K-tile (see Register Prefetch)
    DD ra[BM_TS * TK_TS / NT_TS], rb[TK_TS * BN_TS / NT_TS];
    if (PF) {
        dd_fetch_tile<BM_TS, TK_TS, NT_TS>(ra, A, M, K, bRow, 0, flatId);
        dd_fetch_tile<TK_TS, BN_TS, NT_TS>(rb, B, K, N, 0, bCol, flatId);
    }

    for (uint kt = 0; kt < (K + TK_TS - 1) / TK_TS; kt++) {
        if (PF) {
            dd_put_tile<BM_TS, TK_TS, NT_TS>(tileA, ra, flatId);
            dd_put_tile<TK_TS, BN_TS, NT_TS>(tileB, rb, flatId);
        } else {
            for (uint i = flatId; i < BM_TS * TK_TS; i += NT_TS) {
                uint r = i / TK_TS, c = i % TK_TS;
                uint gr = bRow + r, gc = kt * TK_TS + c;
                tileA[i] = (gr < M && gc < K) ? A[gr * K + gc] : DD{0.0f, 0.0f};
            }
            for (uint i = flatId; i < TK_TS * BN_TS; i += NT_TS) {
                uint r = i / BN_TS, c = i % BN_TS;
                uint gr = kt * TK_TS + r, gc = bCol + c;
                tileB[i] = (gr < K && gc < N) ? B[gr * N + gc] : DD{0.0f, 0.0f};
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (PF && (kt + 1) * TK_TS < K) {
            dd_fetch_tile<BM_TS, TK_TS, NT_TS>(ra, A, M, K, bRow, (kt + 1) * TK_TS, flatId);
            dd_fetch_tile<TK_TS, BN_TS, NT_TS>(rb, B, K, N, (kt + 1) * TK_TS, bCol, flatId);
        }

        for (uint k = 0; k < TK_TS; k++) {
            DD av[TM_TS], bv[TN_TS];
            for (uint i = 0; i < TM_TS; i++) av[i] = tileA[(ty * TM_TS + i) * TK_TS + k];
            for (uint j = 0; j < TN_TS; j++) bv[j] = tileB[k * BN_TS + tx * TN_TS + j];
            for (uint i = 0; i < TM_TS; i++)
                for (uint j = 0; j < TN_TS; j++)
                    acc[i][j] = dd_fma(av[i], bv[j], acc[i][j]);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        // Block-wise compensated accumulation (tall-skinny variant)
        if ((kt & (RENORM_INTERVAL - 1)) == (RENORM_INTERVAL - 1)) {
            for (uint i = 0; i < TM_TS; i++)
                for (uint j = 0; j < TN_TS; j++) {
                    float s, e;
                    twoSum(acc[i][j].hi, acc[i][j].lo, s, e);
                    acc[i][j] = {s, e};
                }
        }
    }

    // Final renormalization before epilogue scaling
    for (uint i = 0; i < TM_TS; i++)
        for (uint j = 0; j < TN_TS; j++) {
            float s, e;
            twoSum(acc[i][j].hi, acc[i][j].lo, s, e);
            acc[i][j] = {s, e};
        }

    for (uint i = 0; i < TM_TS; i++)
        for (uint j = 0; j < TN_TS; j++) {
            uint gr = bRow + ty * TM_TS + i, gc = bCol + tx * TN_TS + j;
            if (gr < M && gc < N) {
                DD result = dd_mul(acc[i][j], alpha);
                if (beta.hi != 0.0f || beta.lo != 0.0f) result = dd_add(result, dd_mul(C[gr * N + gc], beta));
                C[gr * N + gc] = result;
            }
        }
}

typedef decltype(dd_dgemm_ab_ts_t<false>) dd_dgemm_ab_ts_fn;
template [[host_name("dd_dgemm_ab_ts")]] kernel dd_dgemm_ab_ts_fn dd_dgemm_ab_ts_t<false>;
template [[host_name("dd_dgemm_ab_ts_pf")]] kernel dd_dgemm_ab_ts_fn dd_dgemm_ab_ts_t<true>;

// =============================================================================
// DGEMM Kernels (Narrow Output): dd_dgemm and dd_dgemm_ab with 32×32 blocks
// BM_NW=32, BN_NW=32, TM_NW=2, TN_NW=2, 256 threads per threadgroup
// =============================================================================

// The main loop of the 64×64 kernels for a 32×32 block: each output gets the
// same dd_fma sequence and renormalizations, so the results are bitwise the
// same as those kernels'. acc ends renormalized.
template <bool PF>
inline void dd_gemm_narrow_mainloop(
    device const DD* A, device const DD* B, uint M, uint N, uint K,
    uint bRow, uint bCol, uint ty, uint tx, uint flatId,
    threadgroup float4* tileA4, threadgroup float4* tileB4,
    thread DD (&acc)[TM_NW][TN_NW]
) {
    threadgroup DD* tileA = (threadgroup DD*)tileA4;
    threadgroup DD* tileB = (threadgroup DD*)tileB4;

    for (uint i = 0; i < TM_NW; i++)
        for (uint j = 0; j < TN_NW; j++)
            acc[i][j] = {0.0f, 0.0f};

    // PF: this thread's share of the next K-tile (see Register Prefetch)
    DD ra[BM_NW * TK / NT_NW], rb[TK * BN_NW / NT_NW];
    if (PF) {
        dd_fetch_tile<BM_NW, TK, NT_NW>(ra, A, M, K, bRow, 0, flatId);
        dd_fetch_tile<TK, BN_NW, NT_NW>(rb, B, K, N, 0, bCol, flatId);
    }

    for (uint kt = 0; kt < (K + TK - 1) / TK; kt++) {
        if (PF) {
            dd_put_tile_kmajor<BM_NW, TK, NT_NW>(tileA, ra, flatId);
            dd_put_tile<TK, BN_NW, NT_NW>(tileB, rb, flatId);
        } else {
            for (uint i = flatId; i < BM_NW * TK; i += NT_NW) {
                uint r = i / TK, c = i % TK;
                uint gr = bRow + r, gc = kt * TK + c;
                tileA[c * BM_NW + r] = (gr < M && gc < K) ? A[gr * K + gc] : DD{0.0f, 0.0f};
            }
            for (uint i = flatId; i < TK * BN_NW; i += NT_NW) {
                uint r = i / BN_NW, c = i % BN_NW;
                uint gr = kt * TK + r, gc = bCol + c;
                tileB[i] = (gr < K && gc < N) ? B[gr * N + gc] : DD{0.0f, 0.0f};
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (PF && (kt + 1) * TK < K) {
            dd_fetch_tile<BM_NW, TK, NT_NW>(ra, A, M, K, bRow, (kt + 1) * TK, flatId);
            dd_fetch_tile<TK, BN_NW, NT_NW>(rb, B, K, N, (kt + 1) * TK, bCol, flatId);
        }

        for (uint k = 0; k < TK; k++) {
            DD av[TM_NW], bv[TN_NW];
            for (uint h = 0; h < TM_NW / 2; h++) {
                float4 a = tileA4[(k * BM_NW + ty * TM_NW) / 2 + h];
                av[2 * h] = {a.x, a.y}; av[2 * h + 1] = {a.z, a.w};
            }
            for (uint h = 0; h < TN_NW / 2; h++) {
                float4 b = tileB4[(k * BN_NW + tx * TN_NW) / 2 + h];
                bv[2 * h] = {b.x, b.y}; bv[2 * h + 1] = {b.z, b.w};
            }
            for (uint i = 0; i < TM_NW; i++)
                for (uint j = 0; j < TN_NW; j++)
                    acc[i][j] = dd_fma(av[i], bv[j], acc[i][j]);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        // Block-wise compensated accumulation: periodic renormalization
        if ((kt & (RENORM_INTERVAL - 1)) == (RENORM_INTERVAL - 1)) {
            for (uint i = 0; i < TM_NW; i++)
                for (uint j = 0; j < TN_NW; j++) {
                    float s, e;
                    twoSum(acc[i][j].hi, acc[i][j].lo, s, e);
                    acc[i][j] = {s, e};
                }
        }
    }

    // Final renormalization
    for (uint i = 0; i < TM_NW; i++)
        for (uint j = 0; j < TN_NW; j++) {
            float s, e;
            twoSum(acc[i][j].hi, acc[i][j].lo, s, e);
            acc[i][j] = {s, e};
        }
}

// C = A × B, as dd_dgemm (same buffers)
template <bool PF>
kernel void dd_dgemm_narrow_t(
    device const DD* A [[buffer(0)]],
    device const DD* B [[buffer(1)]],
    device DD* C [[buffer(2)]],
    constant uint& M [[buffer(3)]],
    constant uint& N [[buffer(4)]],
    constant uint& K [[buffer(5)]],
    constant uint& gridW [[buffer(6)]],
    constant uint& gridH [[buffer(7)]],
    uint2 lid [[thread_position_in_threadgroup]],
    uint2 tgid [[threadgroup_position_in_grid]],
    uint flatId [[thread_index_in_threadgroup]]
) {
    uint2 mtid = morton_remap(tgid, gridW, gridH);
    uint bRow = mtid.y * BM_NW, bCol = mtid.x * BN_NW;
    uint ty = lid.y, tx = lid.x;

    threadgroup float4 tileA4[BM_NW * TK / 2];
    threadgroup float4 tileB4[TK * BN_NW / 2];
    DD acc[TM_NW][TN_NW];
    dd_gemm_narrow_mainloop<PF>(A, B, M, N, K, bRow, bCol, ty, tx, flatId, tileA4, tileB4, acc);

    for (uint i = 0; i < TM_NW; i++)
        for (uint j = 0; j < TN_NW; j++) {
            uint gr = bRow + ty * TM_NW + i, gc = bCol + tx * TN_NW + j;
            if (gr < M && gc < N) C[gr * N + gc] = acc[i][j];
        }
}

typedef decltype(dd_dgemm_narrow_t<false>) dd_dgemm_narrow_fn;
template [[host_name("dd_dgemm_narrow")]] kernel dd_dgemm_narrow_fn dd_dgemm_narrow_t<false>;
template [[host_name("dd_dgemm_narrow_pf")]] kernel dd_dgemm_narrow_fn dd_dgemm_narrow_t<true>;

// C = α·A×B + β·C, as dd_dgemm_ab and dd_dgemm_ab_ts (same buffers)
template <bool PF>
kernel void dd_dgemm_ab_narrow_t(
    device const DD* A [[buffer(0)]],
    device const DD* B [[buffer(1)]],
    device DD* C [[buffer(2)]],
    constant uint& M [[buffer(3)]],
    constant uint& N [[buffer(4)]],
    constant uint& K [[buffer(5)]],
    constant DD& alpha [[buffer(6)]],
    constant DD& beta [[buffer(7)]],
    constant uint& gridW [[buffer(8)]],
    constant uint& gridH [[buffer(9)]],
    uint2 lid [[thread_position_in_threadgroup]],
    uint2 tgid [[threadgroup_position_in_grid]],
    uint flatId [[thread_index_in_threadgroup]]
) {
    uint2 mtid = morton_remap(tgid, gridW, gridH);
    uint bRow = mtid.y * BM_NW, bCol = mtid.x * BN_NW;
    uint ty = lid.y, tx = lid.x;

    threadgroup float4 tileA4[BM_NW * TK / 2];
    threadgroup float4 tileB4[TK * BN_NW / 2];
    DD acc[TM_NW][TN_NW];
    dd_gemm_narrow_mainloop<PF>(A, B, M, N, K, bRow, bCol, ty, tx, flatId, tileA4, tileB4, acc);

    for (uint i = 0; i < TM_NW; i++)
        for (uint j = 0; j < TN_NW; j++) {
            uint gr = bRow + ty * TM_NW + i, gc = bCol + tx * TN_NW + j;
            if (gr < M && gc < N) {
                DD result = dd_mul(acc[i][j], alpha);
                if (beta.hi != 0.0f || beta.lo != 0.0f) result = dd_add(result, dd_mul(C[gr * N + gc], beta));
                C[gr * N + gc] = result;
            }
        }
}

typedef decltype(dd_dgemm_ab_narrow_t<false>) dd_dgemm_ab_narrow_fn;
template [[host_name("dd_dgemm_ab_narrow")]] kernel dd_dgemm_ab_narrow_fn dd_dgemm_ab_narrow_t<false>;
template [[host_name("dd_dgemm_ab_narrow_pf")]] kernel dd_dgemm_ab_narrow_fn dd_dgemm_ab_narrow_t<true>;

// =============================================================================
// DSYRK Kernel: C = A × Aᵀ (symmetric, upper triangle only)
// =============================================================================

template <bool PF>
kernel void dd_dsyrk_t(
    device const DD* A [[buffer(0)]],
    device DD* C [[buffer(1)]],
    constant uint& N [[buffer(2)]],
    constant uint& K [[buffer(3)]],
    constant uint& gridW [[buffer(4)]],
    constant uint& gridH [[buffer(5)]],
    uint2 lid [[thread_position_in_threadgroup]],
    uint2 tgid [[threadgroup_position_in_grid]],
    uint flatId [[thread_index_in_threadgroup]]
) {
    // Morton Z-order remapping for SLC cache locality
    uint2 mtid = morton_remap(tgid, gridW, gridH);
    uint bRow = mtid.y * BM, bCol = mtid.x * BN;
    // Skip lower triangle blocks (symmetric optimization)
    if (bCol < bRow) return;
    uint ty = lid.y, tx = lid.x;

    DD acc[TM][TN];
    for (uint i = 0; i < TM; i++)
        for (uint j = 0; j < TN; j++)
            acc[i][j] = {0.0f, 0.0f};

    // Shared memory tiles (k-major, see Tile Sizes)
    threadgroup float4 tileA4[BM * TK / 2];
    threadgroup float4 tileB4[TK * BN / 2];
    threadgroup DD* tileA = (threadgroup DD*)tileA4;
    threadgroup DD* tileB = (threadgroup DD*)tileB4;

    // PF: this thread's share of the next K-tile (see Register Prefetch); the
    // B tile is rows bCol.. of A, stored transposed like the A tile
    DD ra[BM * TK / NT], rb[BN * TK / NT];
    if (PF) {
        dd_fetch_tile<BM, TK, NT>(ra, A, N, K, bRow, 0, flatId);
        dd_fetch_tile<BN, TK, NT>(rb, A, N, K, bCol, 0, flatId);
    }

    for (uint kt = 0; kt < (K + TK - 1) / TK; kt++) {
        if (PF) {
            dd_put_tile_kmajor<BM, TK, NT>(tileA, ra, flatId);
            dd_put_tile_kmajor<BN, TK, NT>(tileB, rb, flatId);
        } else {
            for (uint i = flatId; i < BM * TK; i += NT) {
                uint r = i / TK, c = i % TK;
                uint gr = bRow + r, gc = kt * TK + c;
                tileA[c * BM + r] = (gr < N && gc < K) ? A[gr * K + gc] : DD{0.0f, 0.0f};
            }
            for (uint i = flatId; i < TK * BN; i += NT) {
                uint r = i / BN, c = i % BN;
                uint gc_a = bCol + c, gr_a = kt * TK + r;
                tileB[i] = (gc_a < N && gr_a < K) ? A[gc_a * K + gr_a] : DD{0.0f, 0.0f};
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (PF && (kt + 1) * TK < K) {
            dd_fetch_tile<BM, TK, NT>(ra, A, N, K, bRow, (kt + 1) * TK, flatId);
            dd_fetch_tile<BN, TK, NT>(rb, A, N, K, bCol, (kt + 1) * TK, flatId);
        }

        for (uint k = 0; k < TK; k++) {
            DD av[TM], bv[TN];
            for (uint h = 0; h < TM / 2; h++) {
                float4 a = tileA4[(k * BM + ty * TM) / 2 + h];
                av[2 * h] = {a.x, a.y}; av[2 * h + 1] = {a.z, a.w};
            }
            for (uint h = 0; h < TN / 2; h++) {
                float4 b = tileB4[(k * BN + tx * TN) / 2 + h];
                bv[2 * h] = {b.x, b.y}; bv[2 * h + 1] = {b.z, b.w};
            }
            for (uint i = 0; i < TM; i++)
                for (uint j = 0; j < TN; j++)
                    acc[i][j] = dd_fma(av[i], bv[j], acc[i][j]);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        // Block-wise compensated accumulation: periodic renormalization
        if ((kt & (RENORM_INTERVAL - 1)) == (RENORM_INTERVAL - 1)) {
            for (uint i = 0; i < TM; i++)
                for (uint j = 0; j < TN; j++) {
                    float s, e;
                    twoSum(acc[i][j].hi, acc[i][j].lo, s, e);
                    acc[i][j] = {s, e};
                }
        }
    }

    // Final renormalization before store
    for (uint i = 0; i < TM; i++)
        for (uint j = 0; j < TN; j++) {
            float s, e;
            twoSum(acc[i][j].hi, acc[i][j].lo, s, e);
            acc[i][j] = {s, e};
        }

    for (uint i = 0; i < TM; i++)
        for (uint j = 0; j < TN; j++) {
            uint gr = bRow + ty * TM + i, gc = bCol + tx * TN + j;
            if (gr < N && gc < N && gc >= gr) C[gr * N + gc] = acc[i][j];
        }
}

typedef decltype(dd_dsyrk_t<false>) dd_dsyrk_fn;
template [[host_name("dd_dsyrk")]] kernel dd_dsyrk_fn dd_dsyrk_t<false>;
template [[host_name("dd_dsyrk_pf")]] kernel dd_dsyrk_fn dd_dsyrk_t<true>;

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

kernel void dd_matrix_scale(device DD* A [[buffer(0)]], constant DD& alpha [[buffer(1)]], constant uint& count [[buffer(2)]], uint gid [[thread_position_in_grid]]) {
    if (gid < count) A[gid] = dd_mul(A[gid], alpha);
}

kernel void dd_matrix_copy(device const DD* src [[buffer(0)]], device DD* dst [[buffer(1)]], constant uint& count [[buffer(2)]], uint gid [[thread_position_in_grid]]) {
    if (gid < count) dst[gid] = src[gid];
}

// =============================================================================
// Transpose Kernels (for ZGEMM transpose variants)
// =============================================================================

// Regular transpose: dst[j, i] = src[i, j]
kernel void dd_transpose(
    device const DD* src [[buffer(0)]],
    device DD* dst [[buffer(1)]],
    constant uint& rows [[buffer(2)]],
    constant uint& cols [[buffer(3)]],
    uint2 gid [[thread_position_in_grid]]
) {
    if (gid.x < cols && gid.y < rows) {
        uint src_idx = gid.y * cols + gid.x;  // row-major [row, col]
        uint dst_idx = gid.x * rows + gid.y;  // transposed [col, row]
        dst[dst_idx] = src[src_idx];
    }
}

// Conjugate transpose for complex: dst[j, i] = conj(src[i, j])
// For complex number a + bi, conjugate is a - bi
kernel void dd_conj_transpose(
    device const DD* src_r [[buffer(0)]],
    device const DD* src_i [[buffer(1)]],
    device DD* dst_r [[buffer(2)]],
    device DD* dst_i [[buffer(3)]],
    constant uint& rows [[buffer(4)]],
    constant uint& cols [[buffer(5)]],
    uint2 gid [[thread_position_in_grid]]
) {
    if (gid.x < cols && gid.y < rows) {
        uint src_idx = gid.y * cols + gid.x;  // [row, col]
        uint dst_idx = gid.x * rows + gid.y;  // [col, row]

        // Transpose: swap (row, col) → (col, row)
        dst_r[dst_idx] = src_r[src_idx];

        // Conjugate: negate imaginary part
        dst_i[dst_idx] = DD{-src_i[src_idx].hi, -src_i[src_idx].lo};
    }
}
)");

// =============================================================================
// Metal Context (Singleton)
// =============================================================================

// AB_PREFETCH at initialization: "0" the plain kernels only, "1" the
// register-prefetch kernels on every grid, unset or anything else
// use_prefetch's rule. The results are the same in all three.
typedef enum { AB_PREFETCH_RULE, AB_PREFETCH_OFF, AB_PREFETCH_ALL } ABPrefetchMode;

@interface ABContextImpl : NSObject
@property (nonatomic, strong) id<MTLDevice> device;
@property (nonatomic, strong) id<MTLCommandQueue> commandQueue;
@property (nonatomic, strong) id<MTLComputePipelineState> dgemmPipeline;
@property (nonatomic, strong) id<MTLComputePipelineState> dgemmABPipeline;
@property (nonatomic, strong) id<MTLComputePipelineState> dgemmABTSPipeline;
@property (nonatomic, strong) id<MTLComputePipelineState> dgemmNarrowPipeline;
@property (nonatomic, strong) id<MTLComputePipelineState> dgemmABNarrowPipeline;
@property (nonatomic, strong) id<MTLComputePipelineState> dsyrkPipeline;
// The same kernels' register-prefetch instances (*_pf; routing: use_prefetch)
@property (nonatomic, strong) id<MTLComputePipelineState> dgemmPFPipeline;
@property (nonatomic, strong) id<MTLComputePipelineState> dgemmABPFPipeline;
@property (nonatomic, strong) id<MTLComputePipelineState> dgemmABTSPFPipeline;
@property (nonatomic, strong) id<MTLComputePipelineState> dgemmNarrowPFPipeline;
@property (nonatomic, strong) id<MTLComputePipelineState> dgemmABNarrowPFPipeline;
@property (nonatomic, strong) id<MTLComputePipelineState> dsyrkPFPipeline;
@property (nonatomic) ABPrefetchMode prefetchMode;
@property (nonatomic, strong) id<MTLComputePipelineState> zeroPipeline;
@property (nonatomic, strong) id<MTLComputePipelineState> addPipeline;
@property (nonatomic, strong) id<MTLComputePipelineState> subPipeline;
@property (nonatomic, strong) id<MTLComputePipelineState> transposePipeline;
@property (nonatomic, strong) id<MTLComputePipelineState> conjTransposePipeline;
@property (nonatomic, strong) id<MTLComputePipelineState> scalePipeline;
@property (nonatomic, strong) id<MTLComputePipelineState> matCopyPipeline;
+ (instancetype)shared;
+ (void)shutdown;
@end

static ABContextImpl* g_context = nil;
static bool g_initialized = false;
static os_unfair_lock g_init_lock = OS_UNFAIR_LOCK_INIT;

// Metal does not survive fork(): a child inherits g_context but crashes on
// first use. ab_init() registers this child handler with the first context,
// so a forked child sees no context and GPU-routed BLAS calls run on cblas.
// It is set only in the single-threaded child, before anything else runs.
static bool g_forked_child = false;
static void ab_atfork_child(void) { g_forked_child = true; }

@implementation ABContextImpl
+ (instancetype)shared { return g_forked_child ? nil : g_context; }
+ (void)shutdown {
    os_unfair_lock_lock(&g_init_lock);
    g_context = nil;
    g_initialized = false;
    os_unfair_lock_unlock(&g_init_lock);
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
        // MTLMathModeSafe is CRITICAL for DD precision - prevents FMA reordering
        // that breaks error-free transformations. Without it, precision degrades
        // from ~10⁻¹⁶ to ~10⁻⁸. KVC fallback @(3) was harmful, causing regression.
        // mathMode needs macOS 15 at run time and a macOS 15+ SDK at build time.
        // Otherwise fastMathEnabled=NO, which gives the same full DD precision.
        // The run-time test is respondsToSelector, not @available: @available
        // below a macOS 15 deployment target emits a call to compiler-rt's
        // __isPlatformVersionAtLeast, which gfortran does not link in when it
        // links the static archive.
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wunguarded-availability"
#pragma clang diagnostic ignored "-Wdeprecated-declarations"
#if defined(__MAC_OS_X_VERSION_MAX_ALLOWED) && __MAC_OS_X_VERSION_MAX_ALLOWED >= 150000
        if ([opts respondsToSelector:@selector(setMathMode:)]) {
            opts.mathMode = MTLMathModeSafe;
        } else
#endif
        {
            opts.fastMathEnabled = NO;
        }
#pragma clang diagnostic pop

        // Propagate the host-side DD-algorithm selector into the Metal compiler so
        // the MSL `#ifdef APPLEBOTTOM_USE_DWTIMESDW3` branch sees the same define.
        // Default build: DWTimesDW2 (JMP 2017 Alg 11 / MR 2022 Thm 2.7, < 5u²).
        // Opt-in build: DWTimesDW3 (JMP 2017 Alg 12 / MR 2022 Thm 2.8, < 4u²).
#ifdef APPLEBOTTOM_USE_DWTIMESDW3
        opts.preprocessorMacros = @{@"APPLEBOTTOM_USE_DWTIMESDW3": @1};
        NSLog(@"apple-bottom: MSL compiled with DWTimesDW3 (JMP 2017 Alg 12)");
#endif

        id<MTLLibrary> library = [_device newLibraryWithSource:kShaderSource options:opts error:&error];
        if (!library) { NSLog(@"Shader compile failed: %@", error); return nil; }
        
        _dgemmPipeline = [_device newComputePipelineStateWithFunction:[library newFunctionWithName:@"dd_dgemm"] error:&error];
        if (!_dgemmPipeline) { NSLog(@"Pipeline creation failed (dd_dgemm): %@", error); return nil; }

        _dgemmABPipeline = [_device newComputePipelineStateWithFunction:[library newFunctionWithName:@"dd_dgemm_ab"] error:&error];
        if (!_dgemmABPipeline) { NSLog(@"Pipeline creation failed (dd_dgemm_ab): %@", error); return nil; }

        _dgemmABTSPipeline = [_device newComputePipelineStateWithFunction:[library newFunctionWithName:@"dd_dgemm_ab_ts"] error:&error];
        if (!_dgemmABTSPipeline) { NSLog(@"Pipeline creation failed (dd_dgemm_ab_ts): %@", error); return nil; }

        _dgemmNarrowPipeline = [_device newComputePipelineStateWithFunction:[library newFunctionWithName:@"dd_dgemm_narrow"] error:&error];
        if (!_dgemmNarrowPipeline) { NSLog(@"Pipeline creation failed (dd_dgemm_narrow): %@", error); return nil; }

        _dgemmABNarrowPipeline = [_device newComputePipelineStateWithFunction:[library newFunctionWithName:@"dd_dgemm_ab_narrow"] error:&error];
        if (!_dgemmABNarrowPipeline) { NSLog(@"Pipeline creation failed (dd_dgemm_ab_narrow): %@", error); return nil; }

        _dsyrkPipeline = [_device newComputePipelineStateWithFunction:[library newFunctionWithName:@"dd_dsyrk"] error:&error];
        if (!_dsyrkPipeline) { NSLog(@"Pipeline creation failed (dd_dsyrk): %@", error); return nil; }

        _dgemmPFPipeline = [_device newComputePipelineStateWithFunction:[library newFunctionWithName:@"dd_dgemm_pf"] error:&error];
        if (!_dgemmPFPipeline) { NSLog(@"Pipeline creation failed (dd_dgemm_pf): %@", error); return nil; }

        _dgemmABPFPipeline = [_device newComputePipelineStateWithFunction:[library newFunctionWithName:@"dd_dgemm_ab_pf"] error:&error];
        if (!_dgemmABPFPipeline) { NSLog(@"Pipeline creation failed (dd_dgemm_ab_pf): %@", error); return nil; }

        _dgemmABTSPFPipeline = [_device newComputePipelineStateWithFunction:[library newFunctionWithName:@"dd_dgemm_ab_ts_pf"] error:&error];
        if (!_dgemmABTSPFPipeline) { NSLog(@"Pipeline creation failed (dd_dgemm_ab_ts_pf): %@", error); return nil; }

        _dgemmNarrowPFPipeline = [_device newComputePipelineStateWithFunction:[library newFunctionWithName:@"dd_dgemm_narrow_pf"] error:&error];
        if (!_dgemmNarrowPFPipeline) { NSLog(@"Pipeline creation failed (dd_dgemm_narrow_pf): %@", error); return nil; }

        _dgemmABNarrowPFPipeline = [_device newComputePipelineStateWithFunction:[library newFunctionWithName:@"dd_dgemm_ab_narrow_pf"] error:&error];
        if (!_dgemmABNarrowPFPipeline) { NSLog(@"Pipeline creation failed (dd_dgemm_ab_narrow_pf): %@", error); return nil; }

        _dsyrkPFPipeline = [_device newComputePipelineStateWithFunction:[library newFunctionWithName:@"dd_dsyrk_pf"] error:&error];
        if (!_dsyrkPFPipeline) { NSLog(@"Pipeline creation failed (dd_dsyrk_pf): %@", error); return nil; }

        const char* prefetch = getenv("AB_PREFETCH");
        _prefetchMode = !prefetch ? AB_PREFETCH_RULE
                      : !strcmp(prefetch, "0") ? AB_PREFETCH_OFF
                      : !strcmp(prefetch, "1") ? AB_PREFETCH_ALL : AB_PREFETCH_RULE;

        _zeroPipeline = [_device newComputePipelineStateWithFunction:[library newFunctionWithName:@"dd_matrix_zero"] error:&error];
        if (!_zeroPipeline) { NSLog(@"Pipeline creation failed (dd_matrix_zero): %@", error); return nil; }

        _addPipeline = [_device newComputePipelineStateWithFunction:[library newFunctionWithName:@"dd_matrix_add"] error:&error];
        if (!_addPipeline) { NSLog(@"Pipeline creation failed (dd_matrix_add): %@", error); return nil; }

        _subPipeline = [_device newComputePipelineStateWithFunction:[library newFunctionWithName:@"dd_matrix_sub"] error:&error];
        if (!_subPipeline) { NSLog(@"Pipeline creation failed (dd_matrix_sub): %@", error); return nil; }

        _scalePipeline = [_device newComputePipelineStateWithFunction:[library newFunctionWithName:@"dd_matrix_scale"] error:&error];
        if (!_scalePipeline) { NSLog(@"Pipeline creation failed (dd_matrix_scale): %@", error); return nil; }

        _matCopyPipeline = [_device newComputePipelineStateWithFunction:[library newFunctionWithName:@"dd_matrix_copy"] error:&error];
        if (!_matCopyPipeline) { NSLog(@"Pipeline creation failed (dd_matrix_copy): %@", error); return nil; }

        _transposePipeline = [_device newComputePipelineStateWithFunction:[library newFunctionWithName:@"dd_transpose"] error:&error];
        if (!_transposePipeline) { NSLog(@"Pipeline creation failed (dd_transpose): %@", error); return nil; }

        _conjTransposePipeline = [_device newComputePipelineStateWithFunction:[library newFunctionWithName:@"dd_conj_transpose"] error:&error];
        if (!_conjTransposePipeline) { NSLog(@"Pipeline creation failed (dd_conj_transpose): %@", error); return nil; }
    }
    return self;
}
@end

// =============================================================================
// Internal Accessors for device_api.m (apple_bottom_internal.h)
// =============================================================================

id<MTLCommandQueue> ab_internal_command_queue(void) {
    ABContextImpl* ctx = [ABContextImpl shared];
    return ctx ? ctx.commandQueue : nil;
}

id<MTLComputePipelineState> ab_internal_dgemm_pipeline(void) {
    ABContextImpl* ctx = [ABContextImpl shared];
    return ctx ? ctx.dgemmPipeline : nil;
}

id<MTLComputePipelineState> ab_internal_dgemm_ab_pipeline(void) {
    ABContextImpl* ctx = [ABContextImpl shared];
    return ctx ? ctx.dgemmABPipeline : nil;
}

id<MTLComputePipelineState> ab_internal_dgemm_ab_ts_pipeline(void) {
    ABContextImpl* ctx = [ABContextImpl shared];
    return ctx ? ctx.dgemmABTSPipeline : nil;
}

id<MTLComputePipelineState> ab_internal_add_pipeline(void) {
    ABContextImpl* ctx = [ABContextImpl shared];
    return ctx ? ctx.addPipeline : nil;
}

id<MTLComputePipelineState> ab_internal_sub_pipeline(void) {
    ABContextImpl* ctx = [ABContextImpl shared];
    return ctx ? ctx.subPipeline : nil;
}

// =============================================================================
// Conversion Helpers
// =============================================================================
// Plain loops the compiler vectorizes (~30 GB/s on one core). Large arrays are
// split into a few contiguous chunks across cores; one dispatch_apply block per
// element ran at ~125 MB/s (H-P0-3). Results are identical either way.

#define AB_CONVERT_PARALLEL_MIN ((size_t)1 << 20)  // elements; serial below this
#define AB_CONVERT_CHUNKS 8

// Runs body(offset, n) over [0, count) units of `work` elements each, in
// contiguous ranges, in parallel when there are enough elements, and returns
// the merge of the ranges the bodies return.
static ab_range_t for_each_chunk(size_t count, size_t work,
                                 ab_range_t (^body)(size_t offset, size_t n)) {
    if (count * work < AB_CONVERT_PARALLEL_MIN || count < 2) return body(0, count);
    ab_range_t part_storage[AB_CONVERT_CHUNKS], *parts = part_storage;  // blocks write through it
    size_t chunk = (count + AB_CONVERT_CHUNKS - 1) / AB_CONVERT_CHUNKS;
    dispatch_apply(AB_CONVERT_CHUNKS, dispatch_get_global_queue(QOS_CLASS_USER_INITIATED, 0), ^(size_t c) {
        size_t offset = c * chunk;
        parts[c] = offset < count ? body(offset, MIN(chunk, count - offset)) : AB_RANGE_EMPTY;
    });
    ab_range_t r = AB_RANGE_EMPTY;
    for (int c = 0; c < AB_CONVERT_CHUNKS; c++) {
        r.max = parts[c].max > r.max ? parts[c].max : r.max;
        r.min = parts[c].min < r.min ? parts[c].min : r.min;
    }
    return r;
}

// src * 2^-e to DD, each value rounded once; with range, also returns the range
// of the converted values (in the same vectorized loop; AB_RANGE_EMPTY without).
static ab_range_t convert_fp64_to_dd(const double* __restrict src, DDFloat* __restrict dst,
                                     size_t count, int e, bool range) {
    uint32_t mx = 0, mn = UINT32_MAX;
    if (e < -1023 || e > 1022) {                // 2^-e is not a normal double
        for (size_t i = 0; i < count; i++) {
            double v = ldexp(src[i], -e);
            if (range) AB_RANGE_ADD(v, mx, mn);
            dst[i] = fp64_to_dd(v);
        }
    } else if (e == 0) {
        if (range) {
            for (size_t i = 0; i < count; i++) {
                AB_RANGE_ADD(src[i], mx, mn);
                dst[i] = fp64_to_dd(src[i]);
            }
        } else {
            for (size_t i = 0; i < count; i++) dst[i] = fp64_to_dd(src[i]);
        }
    } else {
        double f = ab_pow2(-e);
        if (range) {
            for (size_t i = 0; i < count; i++) {
                double v = src[i] * f;
                AB_RANGE_ADD(v, mx, mn);
                dst[i] = fp64_to_dd(v);
            }
        } else {
            for (size_t i = 0; i < count; i++) dst[i] = fp64_to_dd(src[i] * f);
        }
    }
    return range ? (ab_range_t){mx, mn} : AB_RANGE_EMPTY;
}

// src * 2^e to FP64.
static void convert_dd_to_fp64(const DDFloat* src, double* dst, size_t count, int e) {
    if (e == 0)
        for (size_t i = 0; i < count; i++) dst[i] = dd_to_fp64(src[i]);
    else
        for (size_t i = 0; i < count; i++) dst[i] = ab_scale2(dd_to_fp64(src[i]), e);
}

static ab_range_t convert_fp64_to_dd_parallel(const double* src, DDFloat* dst, size_t count,
                                              int e, bool range) {
    return for_each_chunk(count, 1, ^(size_t offset, size_t n) {
        return convert_fp64_to_dd(src + offset, dst + offset, n, e, range);
    });
}

static void convert_dd_to_fp64_parallel(const DDFloat* src, double* dst, size_t count, int e) {
    for_each_chunk(count, 1, ^(size_t offset, size_t n) {
        convert_dd_to_fp64(src + offset, dst + offset, n, e);
        return AB_RANGE_EMPTY;
    });
}

// =============================================================================
// Matrix Handle
// =============================================================================

// A matrix holds value / 2^exp2 in DD: one power-of-two scale per matrix,
// which ab_matrix_upload chooses (native_scale_exp) and the GPU operations
// carry to their results, so that data far outside FP32's exponent range
// keeps DD precision. It is 0 for data inside AB_NATIVE_WINDOW, so such
// data is stored and computed exactly as before the scale existed.
// ab_matrix_upload also keeps the largest magnitude, from which a ZGEMM
// chooses between Gauss-3M and 4M (zgemm_uses_4m). Only an upload, a copy
// and a zero know it; every other write of a matrix's data sets its scale
// through set_scale_unmeasured, never exp2 alone, so that a stale magnitude
// cannot steer that choice.
struct ABMatrix_s {
    id<MTLBuffer> buffer;
    int rows, cols;
    size_t count;     // logical elements, rows * cols
    size_t capacity;  // elements the buffer holds (>= count after pool reuse)
    bool uploaded;
    int exp2;         // power-of-two scale: element = DD * 2^exp2
    uint32_t maxhw;   // high word of the largest |element| (ab_internal.h), or AB_MAXHW_UNKNOWN
};

#define AB_MAXHW_UNKNOWN UINT32_MAX     // above the high word of any |x|, NaN included

// Gives m scale e, its magnitudes not measured.
static inline void set_scale_unmeasured(ABMatrix m, int e) {
    m->exp2 = e;
    m->maxhw = AB_MAXHW_UNKNOWN;
}

// floor(s / q) * q for any int s and q > 0.
static inline int floor_to(int s, int q) { return s >= 0 ? s / q * q : -((q - 1 - s) / q) * q; }

// FP32 holds a DD value with its low part from about 2^-102 up, and its sums
// to 2^AB_SUM_MAX_EXP; a matrix spanning more than the window is kept there.
#define AB_STORE_MIN_EXP (-100)
#define AB_STORE_MAX_EXP 120

// The power-of-two scale e for a matrix whose entries have range r; it stores
// value * 2^-e. The scales that bring every nonzero magnitude into
// 2^±AB_NATIVE_WINDOW are [hi - W, lo + W] (none if the span exceeds 2W):
//   - 0 if it is one of them (data inside the window, as before), or if an
//     entry is NaN or Inf (which no scale helps);
//   - else the multiple of 16 nearest the centre, so that real and imaginary
//     parts of a similar range get the same scale and need no alignment
//     before Gauss-3M's sums; else the centre itself;
//   - a wider span stays at 0 if FP32 holds it (as before), else is centred
//     when FP32 can hold it at all, else keeps its top at 2^AB_STORE_MAX_EXP.
static int native_scale_exp(ab_range_t r) {
    if (r.max == 0 || ab_range_nonfinite(r)) return 0;
    int hi = ab_hiword_ilogb(r.max), lo = ab_hiword_ilogb(r.min);
    int centre = floor_to(lo + hi, 2) / 2;
    if (hi - lo <= 2 * AB_NATIVE_WINDOW) {
        int elo = hi - AB_NATIVE_WINDOW, ehi = lo + AB_NATIVE_WINDOW;
        if (elo <= 0 && 0 <= ehi) return 0;
        int e = floor_to(centre + 8, 16);
        return e >= elo && e <= ehi ? e : centre;
    }
    if (lo >= AB_STORE_MIN_EXP && hi <= AB_STORE_MAX_EXP) return 0;
    if (hi - lo <= AB_STORE_MAX_EXP - AB_STORE_MIN_EXP) return centre;
    return hi - AB_STORE_MAX_EXP;
}

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
// Autorelease Scope for Public Entry Points
// =============================================================================
// Metal returns command buffers and encoders autoreleased, and Fortran or C
// hosts never drain an autorelease pool, so each GPU call used to leak them
// for the life of the process (~170 B per call). Entry points that create
// command buffers or encoders open this scope first. It is @autoreleasepool
// as RAII: push/pop is the runtime ABI @autoreleasepool lowers to (Clang ARC
// spec, "Runtime support"), and the destructor also covers early returns and
// goto-cleanup paths without re-indenting every function body.
extern "C" void *objc_autoreleasePoolPush(void);
extern "C" void objc_autoreleasePoolPop(void *pool);

namespace {
struct ABAutoreleaseScope {
    void *pool = objc_autoreleasePoolPush();
    ABAutoreleaseScope() = default;
    ABAutoreleaseScope(const ABAutoreleaseScope&) = delete;
    ABAutoreleaseScope& operator=(const ABAutoreleaseScope&) = delete;
    ~ABAutoreleaseScope() { objc_autoreleasePoolPop(pool); }
};
}  // namespace

// =============================================================================
// Public API: Initialization
// =============================================================================

// Thread-safe initialization with support for shutdown/reinit
ABStatus ab_init(void) {
    // Checked before the lock, which another parent thread may have held at fork().
    if (g_forked_child) return AB_ERROR_NO_DEVICE;

    ABAutoreleaseScope pool;
    os_unfair_lock_lock(&g_init_lock);

    if (g_initialized) {
        os_unfair_lock_unlock(&g_init_lock);
        return g_context ? AB_OK : AB_ERROR_NO_DEVICE;
    }

    init_timing();
    // Creating the Metal device can raise the CPU's floating-point exception
    // flags (MTLCreateSystemDefaultDevice sets FE_OVERFLOW on its first call
    // in a process on macOS 27). They are Metal's, not the host's: restore
    // the host's flags so Fortran runtimes do not report them at exit.
    fenv_t host_fenv;
    feholdexcept(&host_fenv);
    g_context = [[ABContextImpl alloc] init];
    fesetenv(&host_fenv);
    ABStatus status = g_context ? AB_OK : AB_ERROR_NO_DEVICE;

    if (status == AB_OK) {
        g_initialized = true;
        static bool atfork_registered = false;
        if (!atfork_registered)
            atfork_registered = pthread_atfork(NULL, NULL, ab_atfork_child) == 0;
    }

    os_unfair_lock_unlock(&g_init_lock);
    return status;
}

void ab_shutdown(void) {
    if ([ABContextImpl shared]) {  // a forked child must not release the parent's Metal objects
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

bool ab_is_initialized(void) { return [ABContextImpl shared] != nil; }

// =============================================================================
// Public API: Matrix Lifecycle
// =============================================================================

// Size limits of one matrix. Each kernel indexes a matrix as r * cols + c (or
// the transposed form) in 32-bit unsigned arithmetic, with r and c at most
// one tile past the edge before the bounds check. rows * cols <=
// AB_MAX_MATRIX_ELEMENTS (2^31 - 1) keeps every index exact for any shape,
// and keeps the int index math of the DTRSM and MPIR host loops in range.
// The DD buffer must also fit in one MTLBuffer.
static ABStatus matrix_size_status(ABContextImpl* ctx, int rows, int cols) {
    if (rows <= 0 || cols <= 0) return AB_ERROR_INVALID_ARG;
    size_t count = (size_t)rows * (size_t)cols;
    if (count > AB_MAX_MATRIX_ELEMENTS) return AB_ERROR_TOO_LARGE;
    if (!ctx) return AB_ERROR_NO_DEVICE;
    if (count * sizeof(DDFloat) > ctx.device.maxBufferLength) return AB_ERROR_TOO_LARGE;
    return AB_OK;
}

ABStatus ab_matrix_check_size(int rows, int cols) {
    return matrix_size_status([ABContextImpl shared], rows, cols);
}

ABMatrix ab_matrix_create(int rows, int cols) {
    ABContextImpl* ctx = [ABContextImpl shared];
    if (matrix_size_status(ctx, rows, cols) != AB_OK) return NULL;

    // Use C++ new so ARC-managed id<MTLBuffer> field gets a proper
    // constructor (BUG-8: malloc+free leaked Metal objects under ARC).
    struct ABMatrix_s* m = new struct ABMatrix_s();
    if (!m) return NULL;

    m->rows = rows;
    m->cols = cols;
    m->count = (size_t)rows * cols;
    m->capacity = m->count;
    m->uploaded = false;
    set_scale_unmeasured(m, 0);

    size_t alloc_size = m->count * sizeof(DDFloat);
    // AB_TEST_FAIL_STAGE=alloc (test-only): no buffer, as when Metal refuses one.
    if (ab_test_fail_stage() != AB_PROF_STAGE_ALLOC)
        m->buffer = [ctx.device newBufferWithLength:alloc_size options:MTLResourceStorageModeShared];
    if (!m->buffer) { delete m; return NULL; }
    
    return m;
}

void ab_matrix_destroy(ABMatrix m) {
    if (m) {
        m->buffer = nil;  // explicit nil before delete (ARC destructor would handle it, but be clear)
        delete m;
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

// Converts and measures the data in one pass; data outside AB_NATIVE_WINDOW
// is converted again with its power-of-two scale (native_scale_exp).
ABStatus ab_matrix_upload(ABMatrix m, const double* data, bool parallel) {
    if (!m || !data) return AB_ERROR_INVALID_ARG;
    double t0 = get_time_ms();
    DDFloat* dst = (DDFloat*)m->buffer.contents;
    ab_range_t range = parallel ? convert_fp64_to_dd_parallel(data, dst, m->count, 0, true)
                                : convert_fp64_to_dd(data, dst, m->count, 0, true);
    m->exp2 = native_scale_exp(range);
    m->maxhw = range.max;
    if (m->exp2 != 0) {
        if (parallel)
            convert_fp64_to_dd_parallel(data, dst, m->count, m->exp2, false);
        else
            convert_fp64_to_dd(data, dst, m->count, m->exp2, false);
    }
    m->uploaded = true;
    stats_add_upload(get_time_ms() - t0, m->count);
    return AB_OK;
}

ABStatus ab_matrix_download(ABMatrix m, double* data, bool parallel) {
    if (!m || !data) return AB_ERROR_INVALID_ARG;
    double t0 = get_time_ms();
    DDFloat* src = (DDFloat*)m->buffer.contents;
    if (parallel)
        convert_dd_to_fp64_parallel(src, data, m->count, m->exp2);
    else
        convert_dd_to_fp64(src, data, m->count, m->exp2);
    stats_add_download(get_time_ms() - t0, m->count);
    return AB_OK;
}

// Without exponents this is a flat (parallel) conversion. With row exponents,
// each row is converted as a whole at its exponent. With column exponents
// each element has its own: 2^-(row's + column's) is formed from the
// column's 2^-col_exp[j] (exact) when every such exponent of the row is a
// normal double's, else by ldexp.
extern "C" ABStatus ab_internal_upload_scaled(ABMatrix m, const double* data, const int* row_exp,
                                              const int* col_exp, int shift, ab_range_t* range) {
    if (!m || !data) return AB_ERROR_INVALID_ARG;
    double t0 = get_time_ms();
    DDFloat* dst = (DDFloat*)m->buffer.contents;
    const size_t rows = (size_t)m->rows, cols = (size_t)m->cols;
    const bool want = range != NULL;
    double* colf = NULL;
    int cmin = 0, cmax = 0;
    if (col_exp) {
        if (!(colf = (double*)malloc(cols * sizeof *colf))) return AB_ERROR_ALLOC_FAILED;
        cmin = cmax = col_exp[0];
        for (size_t j = 0; j < cols; j++) {
            cmin = MIN(cmin, col_exp[j]);
            cmax = MAX(cmax, col_exp[j]);
            colf[j] = col_exp[j] >= -1023 && col_exp[j] <= 1022 ? ab_pow2(-col_exp[j]) : 0.0;
        }
    }
    ab_range_t r;
    if (!row_exp && !col_exp) {
        r = convert_fp64_to_dd_parallel(data, dst, m->count, shift, want);
    } else {
        r = for_each_chunk(rows, cols, ^(size_t r0, size_t nr) {
            ab_range_t part = AB_RANGE_EMPTY;
            for (size_t i = r0; i < r0 + nr; i++) {
                const double* x = data + i * cols;
                DDFloat* d = dst + i * cols;
                int ei = (row_exp ? row_exp[i] : 0) + shift;
                ab_range_t ri;
                if (!col_exp) {
                    ri = convert_fp64_to_dd(x, d, cols, ei, want);
                } else {
                    uint32_t mx = 0, mn = UINT32_MAX;
                    bool exact = ei >= -1023 && ei <= 1022 && cmin >= -1023 && cmax <= 1022 &&
                                 ei + cmin >= -1023 && ei + cmax <= 1022;
                    double rf = exact ? ab_pow2(-ei) : 0.0;
                    for (size_t j = 0; j < cols; j++) {
                        double v = exact ? x[j] * (rf * colf[j]) : ldexp(x[j], -(ei + col_exp[j]));
                        if (want) AB_RANGE_ADD(v, mx, mn);
                        d[j] = fp64_to_dd(v);
                    }
                    ri = (ab_range_t){mx, mn};
                }
                part.max = ri.max > part.max ? ri.max : part.max;
                part.min = ri.min < part.min ? ri.min : part.min;
            }
            return part;
        });
    }
    free(colf);
    if (range) *range = r;
    set_scale_unmeasured(m, 0);
    m->uploaded = true;
    stats_add_upload(get_time_ms() - t0, m->count);
    return AB_OK;
}

// dd_matrix_copy (dst = src) and dd_matrix_scale (X *= f), each in an encoder
// of its own in cmdBuf.
static void encode_copy(id<MTLCommandBuffer> cmdBuf, ABContextImpl* ctx, ABMatrix src,
                        ABMatrix dst) {
    id<MTLComputeCommandEncoder> encoder = [cmdBuf computeCommandEncoder];
    [encoder setComputePipelineState:ctx.matCopyPipeline];
    [encoder setBuffer:src->buffer offset:0 atIndex:0];
    [encoder setBuffer:dst->buffer offset:0 atIndex:1];
    uint32_t count = (uint32_t)src->count;
    [encoder setBytes:&count length:sizeof(count) atIndex:2];
    [encoder dispatchThreads:MTLSizeMake(src->count, 1, 1) threadsPerThreadgroup:MTLSizeMake(MIN((size_t)256, src->count), 1, 1)];
    [encoder endEncoding];
}

static void encode_scale(id<MTLCommandBuffer> cmdBuf, ABContextImpl* ctx, ABMatrix X, DDFloat f) {
    id<MTLComputeCommandEncoder> encoder = [cmdBuf computeCommandEncoder];
    [encoder setComputePipelineState:ctx.scalePipeline];
    [encoder setBuffer:X->buffer offset:0 atIndex:0];
    [encoder setBytes:&f length:sizeof(f) atIndex:1];
    uint32_t count = (uint32_t)X->count;
    [encoder setBytes:&count length:sizeof(count) atIndex:2];
    [encoder dispatchThreads:MTLSizeMake(X->count, 1, 1) threadsPerThreadgroup:MTLSizeMake(MIN((size_t)256, X->count), 1, 1)];
    [encoder endEncoding];
}

// Encodes X's DD values *= 2^d, exactly, in passes by powers of two FP32
// holds; a value leaving FP32's range overflows or flushes as it would in
// one step.
static void encode_rescale(id<MTLCommandBuffer> cmdBuf, ABContextImpl* ctx, ABMatrix X, int d) {
    while (d != 0) {
        int step = d < -126 ? -126 : d > 127 ? 127 : d;
        encode_scale(cmdBuf, ctx, X, (DDFloat){ldexpf(1.0f, step), 0.0f});
        d -= step;
    }
}

ABStatus ab_matrix_zero(ABMatrix m) {
    ABAutoreleaseScope pool;
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
    m->exp2 = 0;
    m->maxhw = 0;
    return AB_OK;
}

ABStatus ab_matrix_copy(ABMatrix src, ABMatrix dst) {
    ABAutoreleaseScope pool;
    if (!src || !dst) return AB_ERROR_INVALID_ARG;
    if (src->count != dst->count) return AB_ERROR_DIMENSION_MISMATCH;
    ABContextImpl* ctx = [ABContextImpl shared];
    if (!ctx) return AB_ERROR_NO_DEVICE;
    id<MTLCommandBuffer> cmdBuf = [ctx.commandQueue commandBuffer];
    encode_copy(cmdBuf, ctx, src, dst);
    [cmdBuf commit];
    [cmdBuf waitUntilCompleted];
    dst->exp2 = src->exp2;
    dst->maxhw = src->maxhw;
    return AB_OK;
}

// =============================================================================
// Power-of-Two Scales of GPU Results
// =============================================================================
// Data inside AB_NATIVE_WINDOW has scale 0, so for it these keep every
// operation exactly as it was; they matter only for data that upload scaled.

// The scalars the kernel multiplies by, and the result's scale, for
// C = alpha*A*B + beta*C: alpha and beta themselves, and A's + B's scale, when
// A*B and C have the same scale and alpha and beta are inside the window (as
// before). Otherwise the result takes the scale of the larger of the two
// terms and each scalar absorbs its term's difference from it, so both stay
// in FP32's range; a term below FP32's range relative to the other flushes,
// as it would round away in FP64.
static int scaled_gemm_scalars(ABMatrix A, ABMatrix B, ABMatrix C, double* alpha, double* beta) {
    int ep = A->exp2 + B->exp2, ec = C->exp2;
    if (ep == ec && ab_scalar_exp(*alpha, AB_NATIVE_WINDOW) == 0 &&
        ab_scalar_exp(*beta, AB_NATIVE_WINDOW) == 0)
        return ep;
    bool a = *alpha != 0.0, b = *beta != 0.0;
    int ta = ep + (a && isfinite(*alpha) ? ilogb(*alpha) : 0);
    int tb = ec + (b && isfinite(*beta) ? ilogb(*beta) : 0);
    int e = a && b ? MAX(ta, tb) : a ? ta : b ? tb : ep;
    *alpha = ab_scale2(*alpha, ep - e);
    *beta = ab_scale2(*beta, ec - e);
    return e;
}

// Gives *X and *Y one scale for an elementwise combination: if they differ,
// encodes into cmdBuf a copy of the one with the smaller scale, multiplied by
// 2^(its scale - the other's), and points *X or *Y at it; the caller destroys
// *temp once cmdBuf has completed. Returns the common scale, or INT_MIN if
// the copy could not be allocated.
static int encode_align_pair(id<MTLCommandBuffer> cmdBuf, ABContextImpl* ctx,
                             ABMatrix* X, ABMatrix* Y, ABMatrix* temp) {
    *temp = NULL;
    if ((*X)->exp2 == (*Y)->exp2) return (*X)->exp2;
    ABMatrix* low = (*X)->exp2 < (*Y)->exp2 ? X : Y;
    int high = MAX((*X)->exp2, (*Y)->exp2);
    ABMatrix t = ab_matrix_create((*low)->rows, (*low)->cols);
    if (!t) return INT_MIN;
    encode_copy(cmdBuf, ctx, *low, t);
    encode_rescale(cmdBuf, ctx, t, (*low)->exp2 - high);
    t->exp2 = high;
    t->uploaded = true;
    *low = t;
    *temp = t;
    return high;
}

// The real and imaginary parts of A, and of B, on one scale each for
// Gauss-3M's sums (encode_align_pair); false if a copy could not be
// allocated. The caller destroys temps[0] and temps[1] (NULL if unused) once
// cmdBuf has completed.
static bool encode_align_zgemm(id<MTLCommandBuffer> cmdBuf, ABContextImpl* ctx,
                               ABMatrix* Ar, ABMatrix* Ai, ABMatrix* Br, ABMatrix* Bi,
                               ABMatrix temps[2]) {
    int ea = encode_align_pair(cmdBuf, ctx, Ar, Ai, &temps[0]);
    int eb = encode_align_pair(cmdBuf, ctx, Br, Bi, &temps[1]);
    return ea != INT_MIN && eb != INT_MIN;
}

// =============================================================================
// Encoder-Level Dispatch Helpers
// =============================================================================
// These encode GPU dispatches into an EXISTING encoder without creating or
// committing a command buffer. Used by fused ZGEMM and batch API to amortize
// command buffer overhead across multiple operations.

// Output tiling of C = op(A)*op(B), M×N. Every kernel computes each output
// with the same dd_fma sequence and renormalizations, and a call keeps its
// epilogue (plain C = A*B, or alpha/beta), so the choice never changes results.
//   - 64×64 (dd_dgemm, dd_dgemm_ab): the default.
//   - Tall-skinny 128×16 (dd_dgemm_ab_ts) for M >= 4N.
//   - Narrow 32×32 (dd_dgemm_narrow, dd_dgemm_ab_narrow) for small grids
//     (use_narrow_tiles): at most AB_NARROW_MAX_GROUPS threadgroups alone on
//     the GPU, AB_NARROW_MAX_GROUPS_CONCURRENT in a concurrent encoder
//     (Gauss-3M's and 4M's products), AB_NARROW_MAX_GROUPS_VS_PF against
//     64×64 tiles that prefetch; against tall-skinny ones while taking at most
//     twice their GPU steps. AB_PREFETCH changes the instances, never the tiling.
// Measured on M2 Max (38-core GPU), kernel GPU time: alone on the GPU, every
// kernel's time rises in steps of about AB_GPU_STEP_GROUPS threadgroups. A
// narrow threadgroup's step costs half a tall-skinny one's and 0.29x a 64×64
// one's (~16% more per output at full occupancy), so the finer tiles win
// while a grid takes few steps: 1.04-1.46x from 544×544 to 1024×1024 (but
// 0.92-0.98x at 832×832 and 896×896, whose 64×64 grids fill their last
// step), and 1.1-1.35x over tall-skinny on small grids whose N the 32-column
// tiles pad more (1024×40, 1200×100). Beyond the limits the larger tiles win,
// by up to 12%. Re-calibrate these limits, AB_GPU_STEP_GROUPS and the
// prefetch limits below on a GPU with another core count.
#define AB_NARROW_MAX_GROUPS 1024u
#define AB_NARROW_MAX_GROUPS_CONCURRENT 441u
#define AB_NARROW_MAX_GROUPS_VS_PF 239u
#define AB_GPU_STEP_GROUPS 33u

static bool use_tall_skinny(uint32_t M, uint32_t N) { return M >= 4 * N; }

// Register prefetch: the *_pf instance of a kernel (Register Prefetch in the
// MSL; same bits) instead of the plain one, while the GPU holds few enough
// threadgroups that the tile loads' latency shows. Measured on M2 Max (38-core
// GPU), kernel GPU time against the plain instance:
//   - 64×64 (dd_dgemm, dd_dgemm_ab, dd_dsyrk): 1.05-1.26x up to 64
//     threadgroups (1.09x at 512×512×18432; DSYRK 1.03-1.18x up to N = 512)
//     and within ±2% from 70 on, alone on the GPU, but 0.86-0.91x for three
//     concurrent products of 64: so only in a serial encoder.
//   - Tall-skinny: 1.01-1.47x up to ~700 threadgroups (1.16x at
//     4096×64×4608) and within ±1.5% from 720 on (QE's 18277×150×150
//     rotation: 0.99-1.00x), alone or as Gauss-3M's three concurrent products.
//   - Narrow: 1.00-1.43x on every grid measured, up to 1156 threadgroups,
//     alone or concurrent (1.43x at 160×160×18432): every grid.
#define AB_PREFETCH_MAX_GROUPS 64u
#define AB_PREFETCH_MAX_TS_GROUPS 512u
#define AB_PREFETCH_MAX_NARROW_GROUPS UINT32_MAX

static uint64_t g_prefetch_dispatches = 0;

// The *_pf kernel for a grid of `groups` threadgroups while at most
// max_groups (AB_PREFETCH=1: always, =0: never). Each true counts one dispatch.
static bool use_prefetch(ABContextImpl* ctx, uint32_t groups, uint32_t max_groups) {
    ABPrefetchMode mode = ctx.prefetchMode;
    if (mode == AB_PREFETCH_OFF || (mode == AB_PREFETCH_RULE && groups > max_groups)) return false;
    __atomic_fetch_add(&g_prefetch_dispatches, 1, __ATOMIC_RELAXED);
    return true;
}

// Dispatches of a *_pf kernel since the process started, for the tests.
extern "C" uint64_t ab_internal_prefetch_dispatches(void) {
    return __atomic_load_n(&g_prefetch_dispatches, __ATOMIC_RELAXED);
}

// The 64×64 kernels' prefetch limit: none in a concurrent encoder, whose
// products share the GPU (see above)
static uint32_t square_prefetch_max(id<MTLComputeCommandEncoder> encoder) {
    return encoder.dispatchType == MTLDispatchTypeConcurrent ? 0 : AB_PREFETCH_MAX_GROUPS;
}

// GPU steps of a grid of `groups` threadgroups alone on the GPU (Output tiling)
static uint32_t gpu_steps(uint32_t groups) {
    return (groups + AB_GPU_STEP_GROUPS - 1) / AB_GPU_STEP_GROUPS;
}

// The narrow kernels instead of 64×64 tiles, or of tall-skinny ones when
// tall_skinny, for a dispatch into encoder (Output tiling, above).
static bool use_narrow_tiles(id<MTLComputeCommandEncoder> encoder, uint32_t M, uint32_t N,
                             bool tall_skinny) {
    uint32_t groups = ((M + 31) / 32) * ((N + 31) / 32);
    bool concurrent = encoder.dispatchType == MTLDispatchTypeConcurrent;
    if (groups > (concurrent ? AB_NARROW_MAX_GROUPS_CONCURRENT : AB_NARROW_MAX_GROUPS)) return false;
    if (tall_skinny) {
        uint32_t ts_groups = ((M + 127) / 128) * ((N + 15) / 16);
        return gpu_steps(groups) <= 2 * gpu_steps(ts_groups);
    }
    uint32_t square_groups = ((M + 63) / 64) * ((N + 63) / 64);
    return square_groups > square_prefetch_max(encoder) || groups <= AB_NARROW_MAX_GROUPS_VS_PF;
}

// dd_dgemm (C = A*B) with all of its buffers bound, including the Morton grid
// dims at 6/7, or dd_dgemm_narrow (same buffers) for a small grid, or their
// prefetch instances. ab_dgemm and ab_dgemm_async always come here, so their
// results are bitwise identical.
static void encode_dgemm_square_dispatch(id<MTLComputeCommandEncoder> encoder,
                                          ABContextImpl* ctx,
                                          ABMatrix A, ABMatrix B, ABMatrix C) {
    uint32_t M = A->rows, N = B->cols, K = A->cols;
    bool narrow = use_narrow_tiles(encoder, M, N, false);
    uint32_t tile = narrow ? 32 : 64;
    uint32_t gridW = (N + tile - 1) / tile, gridH = (M + tile - 1) / tile;
    id<MTLComputePipelineState> plain = ctx.dgemmPipeline, pf = ctx.dgemmPFPipeline;
    uint32_t max_groups = square_prefetch_max(encoder);
    if (narrow) {
        plain = ctx.dgemmNarrowPipeline;
        pf = ctx.dgemmNarrowPFPipeline;
        max_groups = AB_PREFETCH_MAX_NARROW_GROUPS;
    }
    [encoder setComputePipelineState:use_prefetch(ctx, gridW * gridH, max_groups) ? pf : plain];
    [encoder setBuffer:A->buffer offset:0 atIndex:0];
    [encoder setBuffer:B->buffer offset:0 atIndex:1];
    [encoder setBuffer:C->buffer offset:0 atIndex:2];
    [encoder setBytes:&M length:sizeof(M) atIndex:3];
    [encoder setBytes:&N length:sizeof(N) atIndex:4];
    [encoder setBytes:&K length:sizeof(K) atIndex:5];
    [encoder setBytes:&gridW length:sizeof(gridW) atIndex:6];
    [encoder setBytes:&gridH length:sizeof(gridH) atIndex:7];
    [encoder dispatchThreadgroups:MTLSizeMake(gridW, gridH, 1)
             threadsPerThreadgroup:MTLSizeMake(16, 16, 1)];
}

// C = alpha*A*B + beta*C: dd_dgemm_ab, dd_dgemm_ab_ts or dd_dgemm_ab_narrow,
// or their prefetch instances (buffer layout A,B,C,M,N,K,alpha,beta,gridW,
// gridH).
static void encode_dgemm_scaled_dispatch(id<MTLComputeCommandEncoder> encoder,
                                          ABContextImpl* ctx,
                                          double alpha, ABMatrix A, ABMatrix B,
                                          double beta, ABMatrix C) {
    uint32_t M = A->rows, N = B->cols, K = A->cols;
    DDFloat alpha_dd = fp64_to_dd(alpha);
    DDFloat beta_dd = fp64_to_dd(beta);
    bool use_ts = use_tall_skinny(M, N);
    bool narrow = use_narrow_tiles(encoder, M, N, use_ts);
    // Tile rows × columns and threadgroup: 64×64 and narrow use 16×16 threads,
    // tall-skinny (TM=4, TN=2) 8×32
    uint32_t bm = 64, bn = 64;
    MTLSize threads = MTLSizeMake(16, 16, 1);
    id<MTLComputePipelineState> plain = ctx.dgemmABPipeline, pf = ctx.dgemmABPFPipeline;
    uint32_t max_groups = square_prefetch_max(encoder);
    if (narrow) {
        bm = bn = 32;
        plain = ctx.dgemmABNarrowPipeline;
        pf = ctx.dgemmABNarrowPFPipeline;
        max_groups = AB_PREFETCH_MAX_NARROW_GROUPS;
    } else if (use_ts) {
        bm = 128;
        bn = 16;
        threads = MTLSizeMake(8, 32, 1);
        plain = ctx.dgemmABTSPipeline;
        pf = ctx.dgemmABTSPFPipeline;
        max_groups = AB_PREFETCH_MAX_TS_GROUPS;
    }
    uint32_t gridW = (N + bn - 1) / bn, gridH = (M + bm - 1) / bm;
    [encoder setComputePipelineState:use_prefetch(ctx, gridW * gridH, max_groups) ? pf : plain];
    [encoder setBuffer:A->buffer offset:0 atIndex:0];
    [encoder setBuffer:B->buffer offset:0 atIndex:1];
    [encoder setBuffer:C->buffer offset:0 atIndex:2];
    [encoder setBytes:&M length:sizeof(M) atIndex:3];
    [encoder setBytes:&N length:sizeof(N) atIndex:4];
    [encoder setBytes:&K length:sizeof(K) atIndex:5];
    [encoder setBytes:&alpha_dd length:sizeof(alpha_dd) atIndex:6];
    [encoder setBytes:&beta_dd length:sizeof(beta_dd) atIndex:7];
    [encoder setBytes:&gridW length:sizeof(gridW) atIndex:8];
    [encoder setBytes:&gridH length:sizeof(gridH) atIndex:9];
    [encoder dispatchThreadgroups:MTLSizeMake(gridW, gridH, 1) threadsPerThreadgroup:threads];
}

// C = A*B. Tall-skinny shapes reuse the scaled kernels with alpha=1, beta=0.
static void encode_dgemm_dispatch(id<MTLComputeCommandEncoder> encoder,
                                   ABContextImpl* ctx,
                                   ABMatrix A, ABMatrix B, ABMatrix C) {
    if (use_tall_skinny(A->rows, B->cols))
        encode_dgemm_scaled_dispatch(encoder, ctx, 1.0, A, B, 0.0, C);
    else
        encode_dgemm_square_dispatch(encoder, ctx, A, B, C);
}

static void encode_elemwise_dispatch(id<MTLComputeCommandEncoder> encoder,
                                      id<MTLComputePipelineState> pipeline,
                                      ABMatrix A, ABMatrix B, ABMatrix C) {
    [encoder setComputePipelineState:pipeline];
    [encoder setBuffer:A->buffer offset:0 atIndex:0];
    [encoder setBuffer:B->buffer offset:0 atIndex:1];
    [encoder setBuffer:C->buffer offset:0 atIndex:2];
    uint32_t count = (uint32_t)A->count;
    [encoder setBytes:&count length:sizeof(count) atIndex:3];
    size_t threads = A->count;
    [encoder dispatchThreads:MTLSizeMake(threads, 1, 1)
             threadsPerThreadgroup:MTLSizeMake(MIN((size_t)256, threads), 1, 1)];
}

// C = A B by Gauss-3M, encoder boundaries ordering the steps:
//   1: T3 = Ar + Ai, T4 = Br + Bi
//   2: T1 = Ar*Br, T2 = Ai*Bi, Ci = T3*T4
//   3: Ci = Ci - T1, Cr = T1 - T2 (distinct outputs)
//   4: Ci = Ci - T2
// T1, T2 are M x N, T3 M x K, T4 K x N (zgemm_temps). A's parts must share a
// scale, and B's (encode_align_zgemm). Step 2's products write distinct
// buffers, so its encoder is concurrent and the products of a small C share
// the GPU; if Ci shares a buffer with Ar, Ai, Br or Bi, writing it could race
// T1's or T2's reads, so those stay serial.
static void encode_zgemm_3m(id<MTLCommandBuffer> cmdBuf, ABContextImpl* ctx,
                            ABMatrix Ar, ABMatrix Ai, ABMatrix Br, ABMatrix Bi,
                            const ABMatrix T[4], ABMatrix Cr, ABMatrix Ci) {
    ABMatrix T1 = T[0], T2 = T[1], T3 = T[2], T4 = T[3];
    int e = Ar->exp2 + Br->exp2;
    id<MTLComputeCommandEncoder> enc = [cmdBuf computeCommandEncoder];
    encode_elemwise_dispatch(enc, ctx.addPipeline, Ar, Ai, T3);
    encode_elemwise_dispatch(enc, ctx.addPipeline, Br, Bi, T4);
    [enc endEncoding];
    bool ci_is_input = Ci->buffer == Ar->buffer || Ci->buffer == Ai->buffer ||
                       Ci->buffer == Br->buffer || Ci->buffer == Bi->buffer;
    enc = [cmdBuf computeCommandEncoderWithDispatchType:
        ci_is_input ? MTLDispatchTypeSerial : MTLDispatchTypeConcurrent];
    encode_dgemm_dispatch(enc, ctx, Ar, Br, T1);
    encode_dgemm_dispatch(enc, ctx, Ai, Bi, T2);
    encode_dgemm_dispatch(enc, ctx, T3, T4, Ci);
    [enc endEncoding];
    enc = [cmdBuf computeCommandEncoder];
    encode_elemwise_dispatch(enc, ctx.subPipeline, Ci, T1, Ci);
    encode_elemwise_dispatch(enc, ctx.subPipeline, T1, T2, Cr);
    [enc endEncoding];
    enc = [cmdBuf computeCommandEncoder];
    encode_elemwise_dispatch(enc, ctx.subPipeline, Ci, T2, Ci);
    [enc endEncoding];
    set_scale_unmeasured(Cr, e);
    set_scale_unmeasured(Ci, e);
}

// C = A B by 4M: T1..T4 = Ar*Br, Ai*Bi, Ai*Br, Ar*Bi (M x N each,
// zgemm_temps) in one concurrent encoder, each on its own scale; then
// Cr = T1 - T2 and Ci = T3 + T4, each pair brought to the larger of its
// scales first. Only the temporaries are written before the last step, so C
// may share buffers with A or B.
static void encode_zgemm_4m(id<MTLCommandBuffer> cmdBuf, ABContextImpl* ctx,
                            ABMatrix Ar, ABMatrix Ai, ABMatrix Br, ABMatrix Bi,
                            const ABMatrix T[4], ABMatrix Cr, ABMatrix Ci) {
    const ABMatrix X[4] = {Ar, Ai, Ai, Ar}, Y[4] = {Br, Bi, Br, Bi};
    id<MTLComputeCommandEncoder> enc =
        [cmdBuf computeCommandEncoderWithDispatchType:MTLDispatchTypeConcurrent];
    for (int t = 0; t < 4; t++) encode_dgemm_dispatch(enc, ctx, X[t], Y[t], T[t]);
    [enc endEncoding];
    for (int t = 0; t < 4; t += 2) {
        int e = MAX(X[t]->exp2 + Y[t]->exp2, X[t + 1]->exp2 + Y[t + 1]->exp2);
        for (int u = t; u < t + 2; u++) {
            encode_rescale(cmdBuf, ctx, T[u], X[u]->exp2 + Y[u]->exp2 - e);
            set_scale_unmeasured(T[u], e);
        }
    }
    enc = [cmdBuf computeCommandEncoder];
    encode_elemwise_dispatch(enc, ctx.subPipeline, T[0], T[1], Cr);
    encode_elemwise_dispatch(enc, ctx.addPipeline, T[2], T[3], Ci);
    [enc endEncoding];
    set_scale_unmeasured(Cr, T[0]->exp2);
    set_scale_unmeasured(Ci, T[2]->exp2);
}

// Whether a ZGEMM runs as 4M: as asked, or for AB_ZGEMM_AUTO when the parts'
// largest magnitudes (known for uploaded and copied matrices; a GPU result's
// is not measured, and counts as balanced) give Gauss-3M a rho above
// AB_GAUSS3M_MAX_RHO. A zero A or B gives NaN, and 3M: there are no products.
static bool zgemm_uses_4m(ab_zgemm_method_t method, ABMatrix Ar, ABMatrix Ai, ABMatrix Br,
                          ABMatrix Bi) {
    if (method != AB_ZGEMM_AUTO) return method == AB_ZGEMM_4M;
    if (Ar->maxhw == AB_MAXHW_UNKNOWN || Ai->maxhw == AB_MAXHW_UNKNOWN ||
        Br->maxhw == AB_MAXHW_UNKNOWN || Bi->maxhw == AB_MAXHW_UNKNOWN)
        return false;
    double p = ab_hiword_value(Ai->maxhw) / ab_hiword_value(Ar->maxhw);
    double q = ab_hiword_value(Bi->maxhw) / ab_hiword_value(Br->maxhw);
    return ab_gauss3m_rho(p, q) > AB_GAUSS3M_MAX_RHO;
}

// One ZGEMM's temporaries, NULL where unused: T1..T4 for encode_zgemm_3m or
// encode_zgemm_4m, and the copies encode_align_zgemm makes for 3M. Blocks
// capture it by value.
typedef struct {
    ABMatrix T[4];
    ABMatrix aligned[2];
} ZgemmTemps;

static void zgemm_temps_free(ZgemmTemps t) {
    for (int i = 0; i < 4; i++) ab_matrix_destroy(t.T[i]);
    for (int i = 0; i < 2; i++) ab_matrix_destroy(t.aligned[i]);
}

// T1..T4 for 4M or 3M into t; false, with t all NULL, if one cannot be
// allocated.
static bool zgemm_temps(bool four_m, int M, int N, int K, ZgemmTemps* t) {
    *t = ZgemmTemps();
    t->T[0] = ab_matrix_create(M, N);
    t->T[1] = ab_matrix_create(M, N);
    t->T[2] = four_m ? ab_matrix_create(M, N) : ab_matrix_create(M, K);
    t->T[3] = four_m ? ab_matrix_create(M, N) : ab_matrix_create(K, N);
    if (t->T[0] && t->T[1] && t->T[2] && t->T[3]) return true;
    zgemm_temps_free(*t);
    *t = ZgemmTemps();
    return false;
}

// C = A B by 4M (four_m) or by Gauss-3M, which first brings A's parts, and
// B's, to one scale into aligned (encode_align_zgemm), with T1..T4 from
// zgemm_temps. False if such a copy could not be allocated. The caller
// destroys T and aligned once cmdBuf has completed.
static bool encode_zgemm(id<MTLCommandBuffer> cmdBuf, ABContextImpl* ctx, bool four_m,
                         ABMatrix Ar, ABMatrix Ai, ABMatrix Br, ABMatrix Bi,
                         const ABMatrix T[4], ABMatrix aligned[2], ABMatrix Cr, ABMatrix Ci) {
    if (four_m) {
        encode_zgemm_4m(cmdBuf, ctx, Ar, Ai, Br, Bi, T, Cr, Ci);
        return true;
    }
    if (!encode_align_zgemm(cmdBuf, ctx, &Ar, &Ai, &Br, &Bi, aligned)) return false;
    encode_zgemm_3m(cmdBuf, ctx, Ar, Ai, Br, Bi, T, Cr, Ci);
    return true;
}

// =============================================================================
// Public API: BLAS Operations
// =============================================================================

ABStatus ab_dgemm(ABMatrix A, ABMatrix B, ABMatrix C) {
    ABAutoreleaseScope pool;
    if (!A || !B || !C) return AB_ERROR_INVALID_ARG;
    if (A->cols != B->rows || A->rows != C->rows || B->cols != C->cols) return AB_ERROR_DIMENSION_MISMATCH;
    ABContextImpl* ctx = [ABContextImpl shared];
    if (!ctx) return AB_ERROR_NO_DEVICE;
    
    double t0 = get_time_ms();
    
    id<MTLCommandBuffer> cmdBuf = [ctx.commandQueue commandBuffer];
    id<MTLComputeCommandEncoder> encoder = [cmdBuf computeCommandEncoder];
    encode_dgemm_square_dispatch(encoder, ctx, A, B, C);
    [encoder endEncoding];
    [cmdBuf commit];
    [cmdBuf waitUntilCompleted];
    set_scale_unmeasured(C, A->exp2 + B->exp2);
    if (cmdBuf.status == MTLCommandBufferStatusError) return AB_ERROR_KERNEL_FAILED;

    stats_add_kernel(get_time_ms() - t0);
    stats_add_dgemm();
    return AB_OK;
}

ABStatus ab_dgemm_scaled(double alpha, ABMatrix A, ABMatrix B, double beta, ABMatrix C) {
    ABAutoreleaseScope pool;
    if (!A || !B || !C) return AB_ERROR_INVALID_ARG;
    if (A->cols != B->rows || A->rows != C->rows || B->cols != C->cols) return AB_ERROR_DIMENSION_MISMATCH;
    ABContextImpl* ctx = [ABContextImpl shared];
    if (!ctx) return AB_ERROR_NO_DEVICE;

    int exp2 = scaled_gemm_scalars(A, B, C, &alpha, &beta);
    double t0 = get_time_ms();

    uint64_t encode_ns = host_ns();
    id<MTLCommandBuffer> cmdBuf = [ctx.commandQueue commandBuffer];
    id<MTLComputeCommandEncoder> encoder = [cmdBuf computeCommandEncoder];
    encode_dgemm_scaled_dispatch(encoder, ctx, alpha, A, B, beta, C);
    [encoder endEncoding];
    bool completed = commit_and_wait(cmdBuf, encode_ns);
    set_scale_unmeasured(C, exp2);
    if (!completed) return AB_ERROR_KERNEL_FAILED;

    stats_add_kernel(get_time_ms() - t0);
    stats_add_dgemm();
    return AB_OK;
}

// C = A + B or A - B elementwise (pipeline), on one power-of-two scale.
static ABStatus elementwise_aligned(id<MTLComputePipelineState> pipeline, ABContextImpl* ctx,
                                    ABMatrix A, ABMatrix B, ABMatrix C) {
    id<MTLCommandBuffer> cmdBuf = [ctx.commandQueue commandBuffer];
    ABMatrix a = A, b = B, temp;
    int e = encode_align_pair(cmdBuf, ctx, &a, &b, &temp);
    if (e == INT_MIN) return AB_ERROR_ALLOC_FAILED;
    id<MTLComputeCommandEncoder> encoder = [cmdBuf computeCommandEncoder];
    encode_elemwise_dispatch(encoder, pipeline, a, b, C);
    [encoder endEncoding];
    [cmdBuf commit];
    [cmdBuf waitUntilCompleted];
    ab_matrix_destroy(temp);
    set_scale_unmeasured(C, e);
    return AB_OK;
}

ABStatus ab_matrix_add(ABMatrix A, ABMatrix B, ABMatrix C) {
    ABAutoreleaseScope pool;
    if (!A || !B || !C) return AB_ERROR_INVALID_ARG;
    if (A->count != B->count || A->count != C->count) return AB_ERROR_DIMENSION_MISMATCH;
    ABContextImpl* ctx = [ABContextImpl shared];
    if (!ctx) return AB_ERROR_NO_DEVICE;
    return elementwise_aligned(ctx.addPipeline, ctx, A, B, C);
}

ABStatus ab_matrix_sub(ABMatrix A, ABMatrix B, ABMatrix C) {
    ABAutoreleaseScope pool;
    if (!A || !B || !C) return AB_ERROR_INVALID_ARG;
    if (A->count != B->count || A->count != C->count) return AB_ERROR_DIMENSION_MISMATCH;
    ABContextImpl* ctx = [ABContextImpl shared];
    if (!ctx) return AB_ERROR_NO_DEVICE;
    return elementwise_aligned(ctx.subPipeline, ctx, A, B, C);
}

// An alpha outside AB_NATIVE_WINDOW goes into A's power-of-two scale, so that
// the DD multiplier stays in FP32's range.
ABStatus ab_matrix_scale(double alpha, ABMatrix A) {
    ABAutoreleaseScope pool;
    if (!A) return AB_ERROR_INVALID_ARG;
    ABContextImpl* ctx = [ABContextImpl shared];
    if (!ctx) return AB_ERROR_NO_DEVICE;

    int e = ab_scalar_exp(alpha, AB_NATIVE_WINDOW);
    set_scale_unmeasured(A, A->exp2 + e);
    id<MTLCommandBuffer> cmdBuf = [ctx.commandQueue commandBuffer];
    encode_scale(cmdBuf, ctx, A, fp64_to_dd(ab_scale2(alpha, -e)));
    [cmdBuf commit];
    [cmdBuf waitUntilCompleted];
    return AB_OK;
}

// =============================================================================
// ZGEMM: Complex Matrix Multiply with Transpose Support
// =============================================================================

// Forward declaration of internal ZGEMM
static ABStatus ab_zgemm_internal(ABMatrix Ar, ABMatrix Ai, ABMatrix Br, ABMatrix Bi,
                                  ABMatrix Cr, ABMatrix Ci, bool four_m);

// Extended ZGEMM with transpose support (for QE compatibility), by the given
// method (ab_internal.h).
static ABStatus zgemm_ex(
    ABTranspose transA, ABTranspose transB,
    ABMatrix Ar, ABMatrix Ai,
    ABMatrix Br, ABMatrix Bi,
    ABMatrix Cr, ABMatrix Ci,
    ab_zgemm_method_t method
) {
    ABAutoreleaseScope pool;
    if (!Ar || !Ai || !Br || !Bi || !Cr || !Ci) return AB_ERROR_INVALID_ARG;

    ABContextImpl* ctx = [ABContextImpl shared];
    if (!ctx) return AB_ERROR_NO_DEVICE;

    // Determine dimensions after transpose
    int M = (transA == AB_NO_TRANS) ? Ar->rows : Ar->cols;
    int K_A = (transA == AB_NO_TRANS) ? Ar->cols : Ar->rows;
    int K_B = (transB == AB_NO_TRANS) ? Br->rows : Br->cols;
    int N = (transB == AB_NO_TRANS) ? Br->cols : Br->rows;

    // Validate dimensions
    if (K_A != K_B) return AB_ERROR_DIMENSION_MISMATCH;
    if (M != Cr->rows || N != Cr->cols) return AB_ERROR_DIMENSION_MISMATCH;

    // 3M or 4M, from the operands as given: a transpose keeps their magnitudes
    const bool four_m = zgemm_uses_4m(method, Ar, Ai, Br, Bi);

    // Working matrices (either original or transposed)
    ABMatrix A_work_r = Ar, A_work_i = Ai;
    ABMatrix B_work_r = Br, B_work_i = Bi;
    bool allocated_A = false, allocated_B = false;

    ABStatus status = AB_OK;

    // Transpose A if needed
    if (transA != AB_NO_TRANS) {
        A_work_r = ab_matrix_create(M, K_A);
        A_work_i = ab_matrix_create(M, K_A);
        if (!A_work_r || !A_work_i) {
            status = AB_ERROR_ALLOC_FAILED;
            goto cleanup;
        }
        allocated_A = true;

        id<MTLCommandBuffer> cmdBuf = [ctx.commandQueue commandBuffer];
        id<MTLComputeCommandEncoder> enc = [cmdBuf computeCommandEncoder];

        if (transA == AB_CONJ_TRANS) {
            // Conjugate transpose
            [enc setComputePipelineState:ctx.conjTransposePipeline];
            [enc setBuffer:Ar->buffer offset:0 atIndex:0];
            [enc setBuffer:Ai->buffer offset:0 atIndex:1];
            [enc setBuffer:A_work_r->buffer offset:0 atIndex:2];
            [enc setBuffer:A_work_i->buffer offset:0 atIndex:3];
            uint32_t rows = Ar->rows, cols = Ar->cols;
            [enc setBytes:&rows length:sizeof(rows) atIndex:4];
            [enc setBytes:&cols length:sizeof(cols) atIndex:5];

            MTLSize grid = MTLSizeMake((Ar->cols + 15) / 16, (Ar->rows + 15) / 16, 1);
            MTLSize block = MTLSizeMake(16, 16, 1);
            [enc dispatchThreadgroups:grid threadsPerThreadgroup:block];
        } else {
            // Regular transpose (AB_TRANS)
            [enc setComputePipelineState:ctx.transposePipeline];

            // Transpose real part
            [enc setBuffer:Ar->buffer offset:0 atIndex:0];
            [enc setBuffer:A_work_r->buffer offset:0 atIndex:1];
            uint32_t rows = Ar->rows, cols = Ar->cols;
            [enc setBytes:&rows length:sizeof(rows) atIndex:2];
            [enc setBytes:&cols length:sizeof(cols) atIndex:3];

            MTLSize grid = MTLSizeMake((Ar->cols + 15) / 16, (Ar->rows + 15) / 16, 1);
            MTLSize block = MTLSizeMake(16, 16, 1);
            [enc dispatchThreadgroups:grid threadsPerThreadgroup:block];

            // Transpose imaginary part
            [enc setBuffer:Ai->buffer offset:0 atIndex:0];
            [enc setBuffer:A_work_i->buffer offset:0 atIndex:1];
            [enc setBytes:&rows length:sizeof(rows) atIndex:2];
            [enc setBytes:&cols length:sizeof(cols) atIndex:3];
            [enc dispatchThreadgroups:grid threadsPerThreadgroup:block];
        }

        [enc endEncoding];
        [cmdBuf commit];
        [cmdBuf waitUntilCompleted];
        if (cmdBuf.status == MTLCommandBufferStatusError) {
            status = AB_ERROR_KERNEL_FAILED;
            goto cleanup;
        }

        A_work_r->uploaded = true;
        A_work_i->uploaded = true;
        A_work_r->exp2 = Ar->exp2;
        A_work_i->exp2 = Ai->exp2;
    }

    // Transpose B if needed
    if (transB != AB_NO_TRANS) {
        B_work_r = ab_matrix_create(K_B, N);
        B_work_i = ab_matrix_create(K_B, N);
        if (!B_work_r || !B_work_i) {
            status = AB_ERROR_ALLOC_FAILED;
            goto cleanup;
        }
        allocated_B = true;

        id<MTLCommandBuffer> cmdBuf = [ctx.commandQueue commandBuffer];
        id<MTLComputeCommandEncoder> enc = [cmdBuf computeCommandEncoder];

        if (transB == AB_CONJ_TRANS) {
            [enc setComputePipelineState:ctx.conjTransposePipeline];
            [enc setBuffer:Br->buffer offset:0 atIndex:0];
            [enc setBuffer:Bi->buffer offset:0 atIndex:1];
            [enc setBuffer:B_work_r->buffer offset:0 atIndex:2];
            [enc setBuffer:B_work_i->buffer offset:0 atIndex:3];
            uint32_t rows = Br->rows, cols = Br->cols;
            [enc setBytes:&rows length:sizeof(rows) atIndex:4];
            [enc setBytes:&cols length:sizeof(cols) atIndex:5];

            MTLSize grid = MTLSizeMake((Br->cols + 15) / 16, (Br->rows + 15) / 16, 1);
            MTLSize block = MTLSizeMake(16, 16, 1);
            [enc dispatchThreadgroups:grid threadsPerThreadgroup:block];
        } else {
            [enc setComputePipelineState:ctx.transposePipeline];

            [enc setBuffer:Br->buffer offset:0 atIndex:0];
            [enc setBuffer:B_work_r->buffer offset:0 atIndex:1];
            uint32_t rows = Br->rows, cols = Br->cols;
            [enc setBytes:&rows length:sizeof(rows) atIndex:2];
            [enc setBytes:&cols length:sizeof(cols) atIndex:3];

            MTLSize grid = MTLSizeMake((Br->cols + 15) / 16, (Br->rows + 15) / 16, 1);
            MTLSize block = MTLSizeMake(16, 16, 1);
            [enc dispatchThreadgroups:grid threadsPerThreadgroup:block];

            [enc setBuffer:Bi->buffer offset:0 atIndex:0];
            [enc setBuffer:B_work_i->buffer offset:0 atIndex:1];
            [enc setBytes:&rows length:sizeof(rows) atIndex:2];
            [enc setBytes:&cols length:sizeof(cols) atIndex:3];
            [enc dispatchThreadgroups:grid threadsPerThreadgroup:block];
        }

        [enc endEncoding];
        [cmdBuf commit];
        [cmdBuf waitUntilCompleted];
        if (cmdBuf.status == MTLCommandBufferStatusError) {
            status = AB_ERROR_KERNEL_FAILED;
            goto cleanup;
        }

        B_work_r->uploaded = true;
        B_work_i->uploaded = true;
        B_work_r->exp2 = Br->exp2;
        B_work_i->exp2 = Bi->exp2;
    }

    // Call internal ZGEMM with (possibly transposed) matrices
    status = ab_zgemm_internal(A_work_r, A_work_i, B_work_r, B_work_i, Cr, Ci, four_m);

cleanup:
    // Cleanup temporary transpose buffers
    if (allocated_A) {
        ab_matrix_destroy(A_work_r);
        ab_matrix_destroy(A_work_i);
    }
    if (allocated_B) {
        ab_matrix_destroy(B_work_r);
        ab_matrix_destroy(B_work_i);
    }

    return status;
}

ABStatus ab_zgemm_ex(ABTranspose transA, ABTranspose transB, ABMatrix Ar, ABMatrix Ai,
                     ABMatrix Br, ABMatrix Bi, ABMatrix Cr, ABMatrix Ci) {
    return zgemm_ex(transA, transB, Ar, Ai, Br, Bi, Cr, Ci, AB_ZGEMM_AUTO);
}

extern "C" ABStatus ab_internal_zgemm(ABMatrix Ar, ABMatrix Ai, ABMatrix Br, ABMatrix Bi,
                                      ABMatrix Cr, ABMatrix Ci, ab_zgemm_method_t method) {
    return zgemm_ex(AB_NO_TRANS, AB_NO_TRANS, Ar, Ai, Br, Bi, Cr, Ci, method);
}

// ZGEMM using Gauss's trick: 3 real multiplications instead of 4
// (A_r + iA_i)(B_r + iB_i) = (A_r*B_r - A_i*B_i) + i((A_r+A_i)(B_r+B_i) - A_r*B_r - A_i*B_i)
// or, for parts whose imbalance would cost 3M's imaginary part its accuracy,
// by 4M (four_m: zgemm_uses_4m); encode_zgemm.
//
// Fused implementation: 1 command buffer (was 8 separate round-trips); this
// eliminates ~350μs of command buffer commit+wait overhead per ZGEMM call.
//
// NOTE: This is the internal implementation - ab_zgemm is now a wrapper
static ABStatus ab_zgemm_internal(ABMatrix Ar, ABMatrix Ai, ABMatrix Br, ABMatrix Bi,
                                  ABMatrix Cr, ABMatrix Ci, bool four_m) {
    if (!Ar || !Ai || !Br || !Bi || !Cr || !Ci) return AB_ERROR_INVALID_ARG;

    int M = Ar->rows, N = Br->cols, K = Ar->cols;
    ABContextImpl* ctx = [ABContextImpl shared];
    if (!ctx) return AB_ERROR_NO_DEVICE;

    // Allocate all temporaries upfront
    uint64_t temps_ns = host_ns();
    ZgemmTemps tmp;
    bool allocated = zgemm_temps(four_m, M, N, K, &tmp);
    t_kernel_split.temps_ns += host_ns() - temps_ns;
    if (!allocated) return AB_ERROR_ALLOC_FAILED;

    double t0 = get_time_ms();

    uint64_t encode_ns = host_ns();
    id<MTLCommandBuffer> cmdBuf = [ctx.commandQueue commandBuffer];
    if (!encode_zgemm(cmdBuf, ctx, four_m, Ar, Ai, Br, Bi, tmp.T, tmp.aligned, Cr, Ci)) {
        zgemm_temps_free(tmp);
        return AB_ERROR_ALLOC_FAILED;
    }
    bool completed = commit_and_wait(cmdBuf, encode_ns);

    stats_add_kernel(get_time_ms() - t0);
    stats_add_zgemm();

    temps_ns = host_ns();
    zgemm_temps_free(tmp);
    t_kernel_split.temps_ns += host_ns() - temps_ns;

    if (!completed) {
        return AB_ERROR_KERNEL_FAILED;
    }
    return AB_OK;
}

// Async variant: returns immediately, GPU runs in background.
// Temporaries are cleaned up by a Metal completion handler.
static ABStatus ab_zgemm_internal_async(ABMatrix Ar, ABMatrix Ai, ABMatrix Br, ABMatrix Bi,
                                         ABMatrix Cr, ABMatrix Ci,
                                         id<MTLCommandBuffer> __strong *outCmdBuf) {
    if (!Ar || !Ai || !Br || !Bi || !Cr || !Ci) return AB_ERROR_INVALID_ARG;

    int M = Ar->rows, N = Br->cols, K = Ar->cols;
    ABContextImpl* ctx = [ABContextImpl shared];
    if (!ctx) return AB_ERROR_NO_DEVICE;

    const bool four_m = zgemm_uses_4m(AB_ZGEMM_AUTO, Ar, Ai, Br, Bi);
    ZgemmTemps tmp;
    if (!zgemm_temps(four_m, M, N, K, &tmp)) return AB_ERROR_ALLOC_FAILED;

    id<MTLCommandBuffer> cmdBuf = [ctx.commandQueue commandBuffer];
    if (!encode_zgemm(cmdBuf, ctx, four_m, Ar, Ai, Br, Bi, tmp.T, tmp.aligned, Cr, Ci)) {
        zgemm_temps_free(tmp);
        return AB_ERROR_ALLOC_FAILED;
    }

    // Completion handler cleans up temporaries when GPU finishes
    [cmdBuf addCompletedHandler:^(id<MTLCommandBuffer> _Nonnull __unused buf) {
        zgemm_temps_free(tmp);
    }];

    [cmdBuf commit];
    stats_add_zgemm();

    *outCmdBuf = cmdBuf;
    return AB_OK;
}

// Public ab_zgemm: wrapper for ab_zgemm_ex with no transpose
ABStatus ab_zgemm(ABMatrix Ar, ABMatrix Ai, ABMatrix Br, ABMatrix Bi, ABMatrix Cr, ABMatrix Ci) {
    return ab_zgemm_ex(AB_NO_TRANS, AB_NO_TRANS, Ar, Ai, Br, Bi, Cr, Ci);
}

ABStatus ab_dsyrk(ABMatrix A, ABMatrix C) {
    ABAutoreleaseScope pool;
    if (!A || !C) return AB_ERROR_INVALID_ARG;
    if (A->rows != C->rows || C->rows != C->cols) return AB_ERROR_DIMENSION_MISMATCH;
    ABContextImpl* ctx = [ABContextImpl shared];
    if (!ctx) return AB_ERROR_NO_DEVICE;
    
    uint32_t N = A->rows, K = A->cols;
    double t0 = get_time_ms();
    
    id<MTLCommandBuffer> cmdBuf = [ctx.commandQueue commandBuffer];
    // The kernel writes C's upper triangle only; the rest is brought to the
    // result's scale first, so that it keeps its values.
    encode_rescale(cmdBuf, ctx, C, C->exp2 - 2 * A->exp2);
    id<MTLComputeCommandEncoder> encoder = [cmdBuf computeCommandEncoder];
    uint32_t gridW = (N + 63) / 64, gridH = (N + 63) / 64;
    bool pf = use_prefetch(ctx, gridW * gridH, square_prefetch_max(encoder));
    [encoder setComputePipelineState:pf ? ctx.dsyrkPFPipeline : ctx.dsyrkPipeline];
    [encoder setBuffer:A->buffer offset:0 atIndex:0];
    [encoder setBuffer:C->buffer offset:0 atIndex:1];
    [encoder setBytes:&N length:sizeof(N) atIndex:2];
    [encoder setBytes:&K length:sizeof(K) atIndex:3];
    [encoder setBytes:&gridW length:sizeof(gridW) atIndex:4];
    [encoder setBytes:&gridH length:sizeof(gridH) atIndex:5];
    [encoder dispatchThreadgroups:MTLSizeMake(gridW, gridH, 1) threadsPerThreadgroup:MTLSizeMake(16, 16, 1)];
    [encoder endEncoding];
    [cmdBuf commit];
    [cmdBuf waitUntilCompleted];
    set_scale_unmeasured(C, 2 * A->exp2);

    stats_add_kernel(get_time_ms() - t0);
    return AB_OK;
}

// =============================================================================
// DTRSM: Triangular Solve — op(A) * X = alpha * B  (or X * op(A) = alpha * B)
// =============================================================================
// Strategy: Blocked forward/back-substitution with GPU DGEMM panel updates.
//   1. Solve a small NB×NRHS block on CPU via cblas_dtrsm (AMX-accelerated)
//   2. Update remaining panels: B -= A_panel * X_solved, using GPU DGEMM
//   3. Repeat until all columns/rows are solved
// This gives FP64-class accuracy via the DD-DGEMM updates while keeping
// the small triangular solves on the fast AMX coprocessor.
// =============================================================================

#define DTRSM_BLOCK_SIZE 64

ABStatus ab_dtrsm(ABSide side, ABUplo uplo, ABTranspose transA, ABDiag diag,
                  double alpha, ABMatrix A, ABMatrix B) {
    if (!A || !B) return AB_ERROR_INVALID_ARG;

    int N = A->rows;  // triangular matrix dimension
    int M = B->rows;
    int NRHS = B->cols;

    // Validate: A must be square
    if (A->cols != N) return AB_ERROR_DIMENSION_MISMATCH;

    // For LEFT side: op(A) is N×N, B is N×NRHS → need A->rows == B->rows
    // For RIGHT side: B is M×N, op(A) is N×N → need A->rows == B->cols
    if (side == AB_LEFT) {
        if (N != M) return AB_ERROR_DIMENSION_MISMATCH;
    } else {
        if (N != NRHS) return AB_ERROR_DIMENSION_MISMATCH;
    }

    int NB = DTRSM_BLOCK_SIZE;

    // Download B to host (row-major) for the blocked solve
    size_t B_count = (size_t)M * NRHS;
    double* B_host = (double*)malloc(B_count * sizeof(double));
    if (!B_host) return AB_ERROR_ALLOC_FAILED;
    ab_matrix_download(B, B_host, true);

    // Download A to host (row-major)
    size_t A_count = (size_t)N * N;
    double* A_host = (double*)malloc(A_count * sizeof(double));
    if (!A_host) { free(B_host); return AB_ERROR_ALLOC_FAILED; }
    ab_matrix_download(A, A_host, true);

    // Apply alpha scaling to B
    if (alpha != 1.0) {
        for (size_t i = 0; i < B_count; i++)
            B_host[i] *= alpha;
    }

    // Convert A to column-major for cblas_dtrsm
    double* A_col = (double*)malloc(A_count * sizeof(double));
    if (!A_col) { free(B_host); free(A_host); return AB_ERROR_ALLOC_FAILED; }
    for (int j = 0; j < N; j++)
        for (int i = 0; i < N; i++)
            A_col[j * N + i] = A_host[i * N + j];

    ABStatus status = AB_OK;

    // Blocked triangular solve: LEFT side, op(A) * X = B
    // Process NB columns at a time
    if (side == AB_LEFT) {
        bool upper = (uplo == AB_UPPER);
        bool no_trans = (transA == AB_NO_TRANS);
        // Effective direction: forward if (lower,no_trans) or (upper,trans)
        bool forward = (upper != no_trans);

        CBLAS_UPLO cblas_uplo = upper ? CblasUpper : CblasLower;
        CBLAS_TRANSPOSE cblas_trans = no_trans ? CblasNoTrans : CblasTrans;
        CBLAS_DIAG cblas_diag = (diag == AB_UNIT_DIAG) ? CblasUnit : CblasNonUnit;

        if (forward) {
            // Forward substitution: blocks 0, NB, 2*NB, ...
            for (int jb = 0; jb < N; jb += NB) {
                int nb = (jb + NB <= N) ? NB : (N - jb);

                // Convert current B panel (nb rows × NRHS) to column-major for cblas
                double* B_panel_col = (double*)malloc((size_t)nb * NRHS * sizeof(double));
                if (!B_panel_col) { status = AB_ERROR_ALLOC_FAILED; goto dtrsm_cleanup; }
                for (int c = 0; c < NRHS; c++)
                    for (int r = 0; r < nb; r++)
                        B_panel_col[c * nb + r] = B_host[(jb + r) * NRHS + c];

                // Solve: op(A_block) * X_panel = B_panel on CPU (AMX)
                cblas_dtrsm(CblasColMajor, CblasLeft, cblas_uplo, cblas_trans, cblas_diag,
                            nb, NRHS, 1.0,
                            &A_col[jb * N + jb], N,  // diagonal block in col-major A
                            B_panel_col, nb);

                // Write solved panel back to B_host (row-major)
                for (int c = 0; c < NRHS; c++)
                    for (int r = 0; r < nb; r++)
                        B_host[(jb + r) * NRHS + c] = B_panel_col[c * nb + r];
                free(B_panel_col);

                // GPU DGEMM update: B_remaining -= A_off * X_solved
                int remaining = N - jb - nb;
                if (remaining > 0) {
                    // Upload solved panel X (nb × NRHS, row-major)
                    ABMatrix mX = ab_matrix_create(nb, NRHS);
                    // Upload the off-diagonal block
                    ABMatrix mA_off = ab_matrix_create(remaining, nb);
                    ABMatrix mUpdate = ab_matrix_create(remaining, NRHS);
                    if (!mX || !mA_off || !mUpdate) {
                        if (mX) ab_matrix_destroy(mX);
                        if (mA_off) ab_matrix_destroy(mA_off);
                        if (mUpdate) ab_matrix_destroy(mUpdate);
                        status = AB_ERROR_ALLOC_FAILED;
                        goto dtrsm_cleanup;
                    }

                    ab_matrix_upload(mX, &B_host[jb * NRHS], true);

                    // Extract off-diagonal block from A_host (row-major)
                    int off_start = jb + nb;
                    double* A_off_data = (double*)malloc((size_t)remaining * nb * sizeof(double));
                    if (!A_off_data) {
                        ab_matrix_destroy(mX);
                        ab_matrix_destroy(mA_off);
                        ab_matrix_destroy(mUpdate);
                        status = AB_ERROR_ALLOC_FAILED;
                        goto dtrsm_cleanup;
                    }

                    // Get the correct off-diagonal block based on transpose
                    if (no_trans) {
                        // A[off_start:N, jb:jb+nb] in row-major
                        for (int r = 0; r < remaining; r++)
                            for (int c = 0; c < nb; c++)
                                A_off_data[r * nb + c] = A_host[(off_start + r) * N + jb + c];
                    } else {
                        // A^T[off_start:N, jb:jb+nb] = A[jb:jb+nb, off_start:N]^T
                        for (int r = 0; r < remaining; r++)
                            for (int c = 0; c < nb; c++)
                                A_off_data[r * nb + c] = A_host[(jb + c) * N + off_start + r];
                    }
                    ab_matrix_upload(mA_off, A_off_data, true);
                    free(A_off_data);

                    // B_remaining -= A_off * X_solved  (via GPU: alpha=-1, beta=1)
                    // First upload current B_remaining
                    ab_matrix_upload(mUpdate, &B_host[off_start * NRHS], true);
                    ab_dgemm_scaled(-1.0, mA_off, mX, 1.0, mUpdate);

                    // Download updated B_remaining
                    ab_matrix_download(mUpdate, &B_host[off_start * NRHS], true);

                    ab_matrix_destroy(mX);
                    ab_matrix_destroy(mA_off);
                    ab_matrix_destroy(mUpdate);
                }
            }
        } else {
            // Backward substitution: blocks N-NB, N-2*NB, ..., 0
            for (int jb_end = N; jb_end > 0; jb_end -= NB) {
                int jb = (jb_end - NB >= 0) ? (jb_end - NB) : 0;
                int nb = jb_end - jb;

                double* B_panel_col = (double*)malloc((size_t)nb * NRHS * sizeof(double));
                if (!B_panel_col) { status = AB_ERROR_ALLOC_FAILED; goto dtrsm_cleanup; }
                for (int c = 0; c < NRHS; c++)
                    for (int r = 0; r < nb; r++)
                        B_panel_col[c * nb + r] = B_host[(jb + r) * NRHS + c];

                cblas_dtrsm(CblasColMajor, CblasLeft, cblas_uplo, cblas_trans, cblas_diag,
                            nb, NRHS, 1.0,
                            &A_col[jb * N + jb], N,
                            B_panel_col, nb);

                for (int c = 0; c < NRHS; c++)
                    for (int r = 0; r < nb; r++)
                        B_host[(jb + r) * NRHS + c] = B_panel_col[c * nb + r];
                free(B_panel_col);

                // GPU DGEMM update: B_above -= A_off * X_solved
                if (jb > 0) {
                    ABMatrix mX = ab_matrix_create(nb, NRHS);
                    ABMatrix mA_off = ab_matrix_create(jb, nb);
                    ABMatrix mUpdate = ab_matrix_create(jb, NRHS);
                    if (!mX || !mA_off || !mUpdate) {
                        if (mX) ab_matrix_destroy(mX);
                        if (mA_off) ab_matrix_destroy(mA_off);
                        if (mUpdate) ab_matrix_destroy(mUpdate);
                        status = AB_ERROR_ALLOC_FAILED;
                        goto dtrsm_cleanup;
                    }

                    ab_matrix_upload(mX, &B_host[jb * NRHS], true);

                    double* A_off_data = (double*)malloc((size_t)jb * nb * sizeof(double));
                    if (!A_off_data) {
                        ab_matrix_destroy(mX);
                        ab_matrix_destroy(mA_off);
                        ab_matrix_destroy(mUpdate);
                        status = AB_ERROR_ALLOC_FAILED;
                        goto dtrsm_cleanup;
                    }

                    if (no_trans) {
                        // A[0:jb, jb:jb+nb] in row-major
                        for (int r = 0; r < jb; r++)
                            for (int c = 0; c < nb; c++)
                                A_off_data[r * nb + c] = A_host[r * N + jb + c];
                    } else {
                        // A^T[0:jb, jb:jb+nb] = A[jb:jb+nb, 0:jb]^T
                        for (int r = 0; r < jb; r++)
                            for (int c = 0; c < nb; c++)
                                A_off_data[r * nb + c] = A_host[(jb + c) * N + r];
                    }
                    ab_matrix_upload(mA_off, A_off_data, true);
                    free(A_off_data);

                    ab_matrix_upload(mUpdate, B_host, true);
                    ab_dgemm_scaled(-1.0, mA_off, mX, 1.0, mUpdate);
                    ab_matrix_download(mUpdate, B_host, true);

                    ab_matrix_destroy(mX);
                    ab_matrix_destroy(mA_off);
                    ab_matrix_destroy(mUpdate);
                }
            }
        }
    } else {
        // RIGHT side: X * op(A) = B — solve column-blocks
        // For now, fall back to cblas_dtrsm on CPU for right-side solves
        // (less common in scientific computing — can be GPU-accelerated later)
        double* B_col = (double*)malloc(B_count * sizeof(double));
        if (!B_col) { status = AB_ERROR_ALLOC_FAILED; goto dtrsm_cleanup; }
        for (int j = 0; j < NRHS; j++)
            for (int i = 0; i < M; i++)
                B_col[j * M + i] = B_host[i * NRHS + j];

        cblas_dtrsm(CblasColMajor, CblasRight,
                    (uplo == AB_UPPER) ? CblasUpper : CblasLower,
                    (transA == AB_NO_TRANS) ? CblasNoTrans : CblasTrans,
                    (diag == AB_UNIT_DIAG) ? CblasUnit : CblasNonUnit,
                    M, N, 1.0, A_col, N, B_col, M);

        // Convert back to row-major
        for (int j = 0; j < NRHS; j++)
            for (int i = 0; i < M; i++)
                B_host[i * NRHS + j] = B_col[j * M + i];
        free(B_col);
    }

    // Upload solved B back to GPU
    ab_matrix_upload(B, B_host, true);

dtrsm_cleanup:
    free(B_host);
    free(A_host);
    free(A_col);
    return status;
}

// =============================================================================
// DGESV via Mixed-Precision Iterative Refinement (MPIR)
// =============================================================================
// Solves A * X = B where A is N×N, B is N×NRHS. B is overwritten with X.
//
// Algorithm (adapted from Trilinos/Belos for Apple Silicon):
//   1. LU factorize A in FP32 via Accelerate LAPACK (O(N³) at full speed)
//   2. Initial solve X₀ = LU⁻¹ B in FP32
//   3. Compute residual R = B - A*X₀ using DD-DGEMM (10⁻¹⁵ fidelity)
//   4. Solve correction ΔX = LU⁻¹ R in FP32
//   5. Update X = X + ΔX
//   6. Repeat 3-5 until ||R||/||B|| < 10⁻¹⁴
//
// This avoids the need for a bespoke DD-DTRSM shader and the catastrophic
// κ(A)² error amplification of explicit inversion (Paper §6).
// Convergence: 1-3 iterations for well-conditioned DFT matrices.
// =============================================================================

ABStatus ab_dgesv_mpir(ABMatrix A, ABMatrix B) {
    if (!A || !B) return AB_ERROR_INVALID_ARG;
    if (A->rows != A->cols) return AB_ERROR_INVALID_ARG;
    if (A->rows != B->rows) return AB_ERROR_DIMENSION_MISMATCH;

    int N = A->rows;
    int NRHS = B->cols;
    ABStatus status = AB_OK;

    // MPIR parameters
    int max_iter = 10;
    double tol = 1e-14;  // Target DD-level precision

    // Host arrays
    size_t A_count = (size_t)N * N;
    size_t B_count = (size_t)N * NRHS;

    double* A_host = NULL;    // row-major A (preserved for residual)
    double* B_orig = NULL;    // row-major B (preserved original RHS)
    double* X_host = NULL;    // row-major current solution
    float*  A_f32  = NULL;    // col-major A for LAPACK
    float*  work_f32 = NULL;  // col-major workspace for LAPACK solves
    int*    ipiv   = NULL;    // pivot indices
    double* R_host = NULL;    // row-major residual

    A_host   = (double*)malloc(A_count * sizeof(double));
    B_orig   = (double*)malloc(B_count * sizeof(double));
    X_host   = (double*)malloc(B_count * sizeof(double));
    A_f32    = (float*)malloc(A_count * sizeof(float));
    work_f32 = (float*)malloc(B_count * sizeof(float));
    ipiv     = (int*)malloc(N * sizeof(int));
    R_host   = (double*)malloc(B_count * sizeof(double));

    if (!A_host || !B_orig || !X_host || !A_f32 || !work_f32 || !ipiv || !R_host) {
        status = AB_ERROR_ALLOC_FAILED;
        goto mpir_cleanup;
    }

    // Download A and B from GPU (row-major doubles)
    ab_matrix_download(A, A_host, true);
    ab_matrix_download(B, B_orig, true);

    // --- Step 1: FP32 LU factorization ---
    // Convert A to FP32 column-major for LAPACK
    for (int i = 0; i < N; i++)
        for (int j = 0; j < N; j++)
            A_f32[j * N + i] = (float)A_host[i * N + j];

    {
        int n = N, lda = N, info = 0;
        sgetrf_(&n, &n, A_f32, &lda, ipiv, &info);
        if (info != 0) {
            status = AB_ERROR_KERNEL_FAILED;  // Singular matrix
            goto mpir_cleanup;
        }
    }

    // --- Step 2: Initial solve X₀ in FP32 ---
    // Convert B to FP32 column-major
    for (int i = 0; i < N; i++)
        for (int j = 0; j < NRHS; j++)
            work_f32[j * N + i] = (float)B_orig[i * NRHS + j];

    {
        char trans = 'N';
        int n = N, nrhs = NRHS, lda = N, ldb = N, info = 0;
        sgetrs_(&trans, &n, &nrhs, A_f32, &lda, ipiv, work_f32, &ldb, &info);
        if (info != 0) {
            status = AB_ERROR_KERNEL_FAILED;
            goto mpir_cleanup;
        }
    }

    // Convert X₀ back to double row-major
    for (int i = 0; i < N; i++)
        for (int j = 0; j < NRHS; j++)
            X_host[i * NRHS + j] = (double)work_f32[j * N + i];

    // --- Steps 3-6: Iterative refinement ---
    for (int iter = 0; iter < max_iter; iter++) {
        // Residual R = B - A*X via DD-DGEMM on GPU
        ABMatrix mA_gpu = ab_matrix_create(N, N);
        ABMatrix mX_gpu = ab_matrix_create(N, NRHS);
        ABMatrix mR_gpu = ab_matrix_create(N, NRHS);

        if (!mA_gpu || !mX_gpu || !mR_gpu) {
            if (mA_gpu) ab_matrix_destroy(mA_gpu);
            if (mX_gpu) ab_matrix_destroy(mX_gpu);
            if (mR_gpu) ab_matrix_destroy(mR_gpu);
            status = AB_ERROR_ALLOC_FAILED;
            goto mpir_cleanup;
        }

        ab_matrix_upload(mA_gpu, A_host, true);
        ab_matrix_upload(mX_gpu, X_host, true);
        ab_matrix_upload(mR_gpu, B_orig, true);  // R starts as B

        // R = B - A*X = -1.0 * A * X + 1.0 * B
        ab_dgemm_scaled(-1.0, mA_gpu, mX_gpu, 1.0, mR_gpu);

        ab_matrix_download(mR_gpu, R_host, true);
        ab_matrix_destroy(mA_gpu);
        ab_matrix_destroy(mX_gpu);
        ab_matrix_destroy(mR_gpu);

        // Check convergence: ||R||_F / ||B||_F
        double r_norm_sq = 0, b_norm_sq = 0;
        for (size_t i = 0; i < B_count; i++) {
            r_norm_sq += R_host[i] * R_host[i];
            b_norm_sq += B_orig[i] * B_orig[i];
        }
        if (b_norm_sq > 0 && sqrt(r_norm_sq / b_norm_sq) < tol)
            break;  // Converged

        // Solve correction ΔX = LU⁻¹ R in FP32
        for (int i = 0; i < N; i++)
            for (int j = 0; j < NRHS; j++)
                work_f32[j * N + i] = (float)R_host[i * NRHS + j];

        {
            char trans = 'N';
            int n = N, nrhs = NRHS, lda = N, ldb = N, info = 0;
            sgetrs_(&trans, &n, &nrhs, A_f32, &lda, ipiv, work_f32, &ldb, &info);
        }

        // Update X = X + ΔX
        for (int i = 0; i < N; i++)
            for (int j = 0; j < NRHS; j++)
                X_host[i * NRHS + j] += (double)work_f32[j * N + i];
    }

    // Upload final solution to B's GPU buffer
    ab_matrix_upload(B, X_host, true);

mpir_cleanup:
    free(A_host);
    free(B_orig);
    free(X_host);
    free(A_f32);
    free(work_f32);
    free(ipiv);
    free(R_host);
    return status;
}

ABStatus ab_zherk(ABMatrix Ar, ABMatrix Ai, ABMatrix Cr, ABMatrix Ci) {
    ABAutoreleaseScope pool;
    if (!Ar || !Ai || !Cr || !Ci) return AB_ERROR_INVALID_ARG;

    int N = Ar->rows;
    int K = Ar->cols;

    // Validate: C must be N×N
    if (N != Cr->rows || N != Cr->cols || N != Ci->rows || N != Ci->cols)
        return AB_ERROR_DIMENSION_MISMATCH;
    if (K != Ai->cols || N != Ai->rows)
        return AB_ERROR_DIMENSION_MISMATCH;

    ABContextImpl* ctx = [ABContextImpl shared];
    if (!ctx) return AB_ERROR_NO_DEVICE;

    ABStatus status = AB_OK;
    ABMatrix temp = NULL;
    ABMatrix ArT = NULL;
    ABMatrix AiT = NULL;

    // ── Real part: Cr = Ar×Arᵀ + Ai×Aiᵀ (two GPU DSYRK calls) ──
    if ((status = ab_dsyrk(Ar, Cr)) != AB_OK) goto cleanup;

    temp = ab_matrix_create(N, N);
    if (!temp) { status = AB_ERROR_ALLOC_FAILED; goto cleanup; }

    if ((status = ab_dsyrk(Ai, temp)) != AB_OK) goto cleanup;
    if ((status = ab_matrix_add(Cr, temp, Cr)) != AB_OK) goto cleanup;

    // ── Imaginary part: Ci = Ai×Arᵀ - Ar×Aiᵀ ──
    // GPU transpose (eliminates CPU download/transpose/upload bottleneck)
    ArT = ab_matrix_create(K, N);
    AiT = ab_matrix_create(K, N);
    if (!ArT || !AiT) { status = AB_ERROR_ALLOC_FAILED; goto cleanup; }

    {
        // Transpose both Ar and Ai on GPU in a single command buffer
        id<MTLCommandBuffer> cmdBuf = [ctx.commandQueue commandBuffer];
        id<MTLComputeCommandEncoder> enc = [cmdBuf computeCommandEncoder];
        [enc setComputePipelineState:ctx.transposePipeline];

        uint32_t rows = (uint32_t)N, cols = (uint32_t)K;
        MTLSize grid = MTLSizeMake((cols + 15) / 16, (rows + 15) / 16, 1);
        MTLSize block = MTLSizeMake(16, 16, 1);

        // Transpose Ar → ArT
        [enc setBuffer:Ar->buffer offset:0 atIndex:0];
        [enc setBuffer:ArT->buffer offset:0 atIndex:1];
        [enc setBytes:&rows length:sizeof(rows) atIndex:2];
        [enc setBytes:&cols length:sizeof(cols) atIndex:3];
        [enc dispatchThreadgroups:grid threadsPerThreadgroup:block];

        // Transpose Ai → AiT (same encoder, same command buffer)
        [enc setBuffer:Ai->buffer offset:0 atIndex:0];
        [enc setBuffer:AiT->buffer offset:0 atIndex:1];
        [enc setBytes:&rows length:sizeof(rows) atIndex:2];
        [enc setBytes:&cols length:sizeof(cols) atIndex:3];
        [enc dispatchThreadgroups:grid threadsPerThreadgroup:block];

        [enc endEncoding];
        [cmdBuf commit];
        [cmdBuf waitUntilCompleted];

        ArT->uploaded = true;
        AiT->uploaded = true;
        ArT->exp2 = Ar->exp2;
        AiT->exp2 = Ai->exp2;
    }

    // Ci = Ai × Arᵀ
    if ((status = ab_dgemm(Ai, ArT, Ci)) != AB_OK) goto cleanup;

    // temp = Ar × Aiᵀ
    if ((status = ab_dgemm(Ar, AiT, temp)) != AB_OK) goto cleanup;

    // Ci = Ci - temp
    if ((status = ab_matrix_sub(Ci, temp, Ci)) != AB_OK) goto cleanup;

cleanup:
    ab_matrix_destroy(temp);
    ab_matrix_destroy(ArT);
    ab_matrix_destroy(AiT);

    return status;
}

// =============================================================================
// Public API: Batched DGEMM
// =============================================================================

ABStatus ab_dgemm_batched(
    int batch_count,
    ABMatrix* As, ABMatrix* Bs, ABMatrix* Cs
) {
    ABAutoreleaseScope pool;
    if (batch_count <= 0) return AB_OK;
    if (!As || !Bs || !Cs) return AB_ERROR_INVALID_ARG;

    ABContextImpl* ctx = [ABContextImpl shared];
    if (!ctx) return AB_ERROR_NO_DEVICE;

    for (int i = 0; i < batch_count; i++) {
        if (!As[i] || !Bs[i] || !Cs[i]) return AB_ERROR_INVALID_ARG;
        if (As[i]->cols != Bs[i]->rows ||
            As[i]->rows != Cs[i]->rows ||
            Bs[i]->cols != Cs[i]->cols) return AB_ERROR_DIMENSION_MISMATCH;
    }

    double t0 = get_time_ms();
    id<MTLCommandBuffer> cmdBuf = [ctx.commandQueue commandBuffer];

    for (int i = 0; i < batch_count; i++) {
        id<MTLComputeCommandEncoder> encoder = [cmdBuf computeCommandEncoder];
        encode_dgemm_dispatch(encoder, ctx, As[i], Bs[i], Cs[i]);
        [encoder endEncoding];
        set_scale_unmeasured(Cs[i], As[i]->exp2 + Bs[i]->exp2);
    }

    [cmdBuf commit];
    [cmdBuf waitUntilCompleted];

    stats_add_kernel(get_time_ms() - t0);
    for (int i = 0; i < batch_count; i++) stats_add_dgemm();

    return AB_OK;
}

ABStatus ab_zgemm_batched(
    int batch_count,
    ABMatrix* Ars, ABMatrix* Ais,
    ABMatrix* Brs, ABMatrix* Bis,
    ABMatrix* Crs, ABMatrix* Cis
) {
    ABAutoreleaseScope pool;
    if (batch_count <= 0) return AB_OK;
    if (!Ars || !Ais || !Brs || !Bis || !Crs || !Cis) return AB_ERROR_INVALID_ARG;

    for (int i = 0; i < batch_count; i++) {
        if (!Ars[i] || !Ais[i] || !Brs[i] || !Bis[i] || !Crs[i] || !Cis[i])
            return AB_ERROR_INVALID_ARG;
    }

    // Fall back to sequential for single ZGEMM
    if (batch_count == 1) {
        return ab_zgemm(Ars[0], Ais[0], Brs[0], Bis[0], Crs[0], Cis[0]);
    }

    ABContextImpl* ctx = [ABContextImpl shared];
    if (!ctx) return AB_ERROR_NO_DEVICE;

    // Optimized batch path: single command buffer, shared temporaries.
    // Each ZGEMM fully consumes its temporaries before the next one starts
    // (encoder boundaries guarantee this), so one set per method (3M's or
    // 4M's, zgemm_uses_4m) serves the batch, allocated when first needed.
    //
    // For mixed-dimension batches, we track the current temp size and reallocate
    // only when dimensions change — QE's uniform workloads never trigger this.

    int cur_M = Ars[0]->rows, cur_N = Brs[0]->cols, cur_K = Ars[0]->cols;
    ZgemmTemps shared[2] = {};          // [four_m]; their aligned slots stay NULL

    double t0 = get_time_ms();
    id<MTLCommandBuffer> cmdBuf = [ctx.commandQueue commandBuffer];
    // Each 3M ZGEMM's alignment copies, destroyed at the end. After an error,
    // the work encoded so far still completes first.
    ABMatrix* aligned = (ABMatrix*)calloc(2 * (size_t)batch_count, sizeof *aligned);
    ABStatus status = aligned ? AB_OK : AB_ERROR_ALLOC_FAILED;
    int encoded = 0;

    for (int i = 0; i < batch_count && status == AB_OK; i++) {
        int M = Ars[i]->rows, N = Brs[i]->cols, K = Ars[i]->cols;

        // Reallocate temps if dimensions changed (rare for QE workloads)
        if (M != cur_M || N != cur_N || K != cur_K) {
            // Insert barrier: previous ZGEMM must finish before we destroy temps
            // (encoder boundary from the loop structure already provides this,
            // but we need to wait for the GPU to finish with the old buffers)
            [cmdBuf commit];
            [cmdBuf waitUntilCompleted];
            cmdBuf = [ctx.commandQueue commandBuffer];

            for (int m = 0; m < 2; m++) {
                zgemm_temps_free(shared[m]);
                shared[m] = ZgemmTemps();
            }
            cur_M = M; cur_N = N; cur_K = K;
        }

        const bool four_m = zgemm_uses_4m(AB_ZGEMM_AUTO, Ars[i], Ais[i], Brs[i], Bis[i]);
        ZgemmTemps* t = &shared[four_m];
        if ((!t->T[0] && !zgemm_temps(four_m, M, N, K, t)) ||
            !encode_zgemm(cmdBuf, ctx, four_m, Ars[i], Ais[i], Brs[i], Bis[i], t->T,
                          &aligned[2 * i], Crs[i], Cis[i])) {
            status = AB_ERROR_ALLOC_FAILED;
            break;
        }
        encoded++;
    }

    [cmdBuf commit];
    [cmdBuf waitUntilCompleted];

    stats_add_kernel(get_time_ms() - t0);
    for (int i = 0; i < encoded; i++) stats_add_zgemm();

    for (int m = 0; m < 2; m++) zgemm_temps_free(shared[m]);
    for (int i = 0; aligned && i < 2 * batch_count; i++) ab_matrix_destroy(aligned[i]);
    free(aligned);

    if (status == AB_OK && cmdBuf.status == MTLCommandBufferStatusError) {
        return AB_ERROR_KERNEL_FAILED;
    }
    return status;
}

// =============================================================================
// =============================================================================
// Public API: Memory Pool
// =============================================================================

#define AB_POOL_MAX_ENTRIES 128

struct ABPoolEntry {
    ABMatrix matrix;
    bool in_use;
};

struct ABMemoryPool_s {
    struct ABPoolEntry entries[AB_POOL_MAX_ENTRIES];
    int count;
    size_t total_allocated;
};

ABMemoryPool ab_pool_create(size_t size_hint) {
    (void)size_hint;  // Reserved for future pre-allocation
    ABMemoryPool pool = new struct ABMemoryPool_s();
    return pool;
}

void ab_pool_destroy(ABMemoryPool pool) {
    if (!pool) return;
    for (int i = 0; i < pool->count; i++) {
        ab_matrix_destroy(pool->entries[i].matrix);
    }
    delete pool;
}

ABMatrix ab_pool_get_matrix(ABMemoryPool pool, int rows, int cols) {
    if (!pool) return ab_matrix_create(rows, cols);
    if (rows <= 0 || cols <= 0) return NULL;
    size_t needed = (size_t)rows * cols;
    
    // Look for available matrix with matching size
    for (int i = 0; i < pool->count; i++) {
        if (!pool->entries[i].in_use && pool->entries[i].matrix) {
            ABMatrix m = pool->entries[i].matrix;
            if (m->rows == rows && m->cols == cols) {
                pool->entries[i].in_use = true;
                m->uploaded = false;
                set_scale_unmeasured(m, 0);
                return m;
            }
        }
    }

    // Look for available matrix with sufficient capacity
    for (int i = 0; i < pool->count; i++) {
        if (!pool->entries[i].in_use && pool->entries[i].matrix) {
            ABMatrix m = pool->entries[i].matrix;
            if (m->capacity >= needed) {
                pool->entries[i].in_use = true;
                m->rows = rows;
                m->cols = cols;
                m->count = needed;  // upload/download/elementwise ops use count
                m->uploaded = false;
                set_scale_unmeasured(m, 0);
                return m;
            }
        }
    }
    
    // Create new matrix and add to pool
    if (pool->count >= AB_POOL_MAX_ENTRIES) {
        // Pool full - return NULL to signal error
        return NULL;
    }

    ABMatrix m = ab_matrix_create(rows, cols);
    if (m) {
        pool->entries[pool->count].matrix = m;
        pool->entries[pool->count].in_use = true;
        pool->count++;
        pool->total_allocated += needed * sizeof(float) * 2;
    }
    return m;
}

void ab_pool_reset(ABMemoryPool pool) {
    if (!pool) return;
    for (int i = 0; i < pool->count; i++) {
        pool->entries[i].in_use = false;
    }
}

// =============================================================================
// Public API: Async Operations
// =============================================================================

struct ABFuture_s {
    id<MTLCommandBuffer> cmdBuffer;
    ABStatus status;
    bool completed;
};

static ABFuture create_future_with_buffer(id<MTLCommandBuffer> cmdBuffer) {
    // BUG-8: use new so ARC-managed id<MTLCommandBuffer> gets proper lifecycle
    ABFuture f = new struct ABFuture_s();
    if (!f) return NULL;
    f->cmdBuffer = cmdBuffer;
    f->status = AB_OK;
    f->completed = false;
    return f;
}

ABFuture ab_dgemm_async(ABMatrix A, ABMatrix B, ABMatrix C) {
    ABAutoreleaseScope pool;
    if (!A || !B || !C) return NULL;
    if (A->cols != B->rows) return NULL;
    if (A->rows != C->rows || B->cols != C->cols) return NULL;
    if (!A->uploaded || !B->uploaded) return NULL;

    ABContextImpl* ctx = [ABContextImpl shared];
    if (!ctx) return NULL;

    id<MTLCommandBuffer> cmdBuf = [ctx.commandQueue commandBuffer];
    id<MTLComputeCommandEncoder> encoder = [cmdBuf computeCommandEncoder];
    encode_dgemm_square_dispatch(encoder, ctx, A, B, C);
    [encoder endEncoding];

    [cmdBuf commit];
    C->uploaded = true;
    set_scale_unmeasured(C, A->exp2 + B->exp2);

    ABFuture f = create_future_with_buffer(cmdBuf);
    return f;
}

ABFuture ab_zgemm_async(ABMatrix Ar, ABMatrix Ai, ABMatrix Br, ABMatrix Bi,
                        ABMatrix Cr, ABMatrix Ci) {
    ABAutoreleaseScope pool;
    id<MTLCommandBuffer> cmdBuf = nil;
    ABStatus s = ab_zgemm_internal_async(Ar, Ai, Br, Bi, Cr, Ci, &cmdBuf);
    if (s != AB_OK || !cmdBuf) {
        // Sync fallback if async setup fails (BUG-8: new, not calloc)
        ABFuture f = new struct ABFuture_s();
        if (f) {
            f->cmdBuffer = nil;
            f->status = s;
            f->completed = true;
        }
        return f;
    }
    return create_future_with_buffer(cmdBuf);
}

ABStatus ab_future_wait(ABFuture f) {
    if (!f) return AB_ERROR_INVALID_ARG;
    if (!f->completed && f->cmdBuffer) {
        [f->cmdBuffer waitUntilCompleted];
        if (f->cmdBuffer.status == MTLCommandBufferStatusError) {
            f->status = AB_ERROR_KERNEL_FAILED;
        }
        f->completed = true;
    }
    return f->status;
}

bool ab_future_is_ready(ABFuture f) {
    if (!f) return true;
    if (f->completed) return true;
    if (f->cmdBuffer) {
        MTLCommandBufferStatus s = f->cmdBuffer.status;
        if (s == MTLCommandBufferStatusCompleted || s == MTLCommandBufferStatusError) {
            f->completed = true;
            if (s == MTLCommandBufferStatusError) f->status = AB_ERROR_KERNEL_FAILED;
            return true;
        }
    }
    return false;
}

ABStatus ab_future_status(ABFuture f) {
    if (!f) return AB_ERROR_INVALID_ARG;
    return f->status;
}

void ab_future_destroy(ABFuture f) {
    if (!f) return;
    if (!f->completed && f->cmdBuffer) {
        [f->cmdBuffer waitUntilCompleted];
    }
    delete f;  // BUG-8: ARC destructor releases cmdBuffer
}


// =============================================================================
// Public API: Batched GEMM Operations
// =============================================================================
// Amortizes Metal command buffer overhead across many GEMMs.
// Uses a single command buffer with encoder boundaries as GPU barriers.
// QE fires hundreds of small GEMMs per SCF iteration; batching eliminates
// the ~50μs per-call commit+wait overhead.

#define AB_BATCH_MAX_ENCODERS 512

struct ABBatch_s {
    id<MTLCommandBuffer> cmdBuffer;
    id<MTLComputeCommandEncoder> currentEncoder;
    ABContextImpl* ctx;
    int op_count;
    bool committed;
};

ABBatch ab_batch_create(void) {
    ABAutoreleaseScope pool;
    ABContextImpl* ctx = [ABContextImpl shared];
    if (!ctx) return NULL;

    // BUG-8: use new so ARC-managed id<MTLCommandBuffer> / id<MTLComputeCommandEncoder>
    // get proper constructors and destructors
    ABBatch batch = new struct ABBatch_s();
    if (!batch) return NULL;

    batch->ctx = ctx;
    batch->cmdBuffer = [ctx.commandQueue commandBuffer];
    batch->currentEncoder = [batch->cmdBuffer computeCommandEncoder];
    batch->op_count = 0;
    batch->committed = false;
    return batch;
}

void ab_batch_destroy(ABBatch batch) {
    if (!batch) return;
    if (!batch->committed && batch->currentEncoder) {
        [batch->currentEncoder endEncoding];
    }
    if (!batch->committed && batch->cmdBuffer) {
        // Discard uncommitted work
        [batch->cmdBuffer commit];
        [batch->cmdBuffer waitUntilCompleted];
    }
    delete batch;  // BUG-8: ARC destructor releases cmdBuffer + encoder
}

ABStatus ab_batch_dgemm(ABBatch batch, ABMatrix A, ABMatrix B, ABMatrix C) {
    if (!batch || !A || !B || !C) return AB_ERROR_INVALID_ARG;
    if (batch->committed) return AB_ERROR_INVALID_ARG;
    if (A->cols != B->rows || A->rows != C->rows || B->cols != C->cols)
        return AB_ERROR_DIMENSION_MISMATCH;

    encode_dgemm_dispatch(batch->currentEncoder, batch->ctx, A, B, C);
    set_scale_unmeasured(C, A->exp2 + B->exp2);
    batch->op_count++;
    stats_add_dgemm();
    return AB_OK;
}

ABStatus ab_batch_dgemm_scaled(ABBatch batch, double alpha, ABMatrix A, ABMatrix B, double beta, ABMatrix C) {
    if (!batch || !A || !B || !C) return AB_ERROR_INVALID_ARG;
    if (batch->committed) return AB_ERROR_INVALID_ARG;
    if (A->cols != B->rows || A->rows != C->rows || B->cols != C->cols)
        return AB_ERROR_DIMENSION_MISMATCH;

    int exp2 = scaled_gemm_scalars(A, B, C, &alpha, &beta);
    encode_dgemm_scaled_dispatch(batch->currentEncoder, batch->ctx, alpha, A, B, beta, C);
    set_scale_unmeasured(C, exp2);
    batch->op_count++;
    stats_add_dgemm();
    return AB_OK;
}

ABStatus ab_batch_zgemm(ABBatch batch, ABMatrix Ar, ABMatrix Ai, ABMatrix Br, ABMatrix Bi,
                         ABMatrix Cr, ABMatrix Ci) {
    ABAutoreleaseScope pool;
    if (!batch || !Ar || !Ai || !Br || !Bi || !Cr || !Ci) return AB_ERROR_INVALID_ARG;
    if (batch->committed) return AB_ERROR_INVALID_ARG;

    int M = Ar->rows, N = Br->cols, K = Ar->cols;
    ABContextImpl* ctx = batch->ctx;

    // Temporaries for Gauss's trick or 4M (zgemm_uses_4m), released by a
    // completion handler; the ZGEMM's encoders go between the batch's.
    const bool four_m = zgemm_uses_4m(AB_ZGEMM_AUTO, Ar, Ai, Br, Bi);
    ZgemmTemps tmp;
    if (!zgemm_temps(four_m, M, N, K, &tmp)) return AB_ERROR_ALLOC_FAILED;
    [batch->currentEncoder endEncoding];
    bool encoded = encode_zgemm(batch->cmdBuffer, ctx, four_m, Ar, Ai, Br, Bi, tmp.T,
                                tmp.aligned, Cr, Ci);
    batch->currentEncoder = [batch->cmdBuffer computeCommandEncoder];
    if (!encoded) {
        zgemm_temps_free(tmp);
        return AB_ERROR_ALLOC_FAILED;
    }

    // Cleanup temporaries via completion handler
    [batch->cmdBuffer addCompletedHandler:^(id<MTLCommandBuffer> _Nonnull __unused buf) {
        zgemm_temps_free(tmp);
    }];

    batch->op_count++;
    stats_add_zgemm();
    return AB_OK;
}

ABStatus ab_batch_barrier(ABBatch batch) {
    ABAutoreleaseScope pool;
    if (!batch) return AB_ERROR_INVALID_ARG;
    if (batch->committed) return AB_ERROR_INVALID_ARG;

    // End current encoder and start a new one — encoder boundary = GPU barrier
    [batch->currentEncoder endEncoding];
    batch->currentEncoder = [batch->cmdBuffer computeCommandEncoder];
    return AB_OK;
}

ABStatus ab_batch_commit(ABBatch batch) {
    if (!batch) return AB_ERROR_INVALID_ARG;
    if (batch->committed) return AB_ERROR_INVALID_ARG;

    [batch->currentEncoder endEncoding];
    batch->currentEncoder = nil;
    [batch->cmdBuffer commit];
    batch->committed = true;
    return AB_OK;
}

ABStatus ab_batch_wait(ABBatch batch) {
    if (!batch) return AB_ERROR_INVALID_ARG;
    if (!batch->committed) return AB_ERROR_INVALID_ARG;

    [batch->cmdBuffer waitUntilCompleted];
    if (batch->cmdBuffer.status == MTLCommandBufferStatusError) {
        return AB_ERROR_KERNEL_FAILED;
    }
    return AB_OK;
}


// Public API: Session Management
// =============================================================================

ABSession ab_session_create(void) {
    return new struct ABSession_s();
}

void ab_session_destroy(ABSession s) {
    if (!s) return;
    for (int i = 0; i < s->count; i++) {
        ab_matrix_destroy(s->matrices[i]);
    }
    delete s;
}

ABStatus ab_session_add(ABSession s, const char* name, int rows, int cols) {
    if (!s || !name || s->count >= MAX_SESSION_MATRICES) return AB_ERROR_INVALID_ARG;
    ABStatus st = ab_matrix_check_size(rows, cols);
    if (st != AB_OK) return st;

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
        case AB_ERROR_SHADER_COMPILE: return "Shader compilation failed";
        case AB_ERROR_TOO_LARGE: return "Matrix exceeds GPU size limits";
        default: return "Unknown error";
    }
}
