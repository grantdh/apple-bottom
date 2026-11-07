// =============================================================================
// strassen.m — one level of Strassen-Winograd over the double-float GEMM (opt-in)
// Copyright (c) 2026 Grant Heileman, Technology Residue. MIT License.
// =============================================================================
//
// The DD GEMM kernel runs close to the GPU's instruction-issue ceiling, so a
// large product gets faster only by doing less work. One level of Winograd's
// variant of Strassen's algorithm computes P = X*Y from 7 products of
// half-size blocks instead of 8, plus 15 block additions (Boyer, Dumas,
// Pernet and Zhou, ISSAC 2009, Alg. 1):
//
//   S1 = A21 + A22   T1 = B12 - B11   P1 = A11 B11   P5 = S1 T1   U2 = P1 + P6
//   S2 = S1 - A11    T2 = B22 - T1    P2 = A12 B21   P6 = S2 T2   U3 = U2 + P7
//   S3 = A11 - A21   T3 = B22 - B12   P3 = S4 B22    P7 = S3 T3   U4 = U2 + P5
//   S4 = A12 - S2    T4 = T2 - B21    P4 = A22 T4
//   C11 = P1 + P2    C12 = U4 + P3    C21 = U3 - P4  C22 = U3 + P5
//
// with A = X, B = Y and C = P. X (M x K), Y (K x N) and P (M x N) split into
// ceil(M/2) x ceil(K/2), ceil(K/2) x ceil(N/2) and ceil(M/2) x ceil(N/2)
// blocks. The trailing blocks of an odd dimension read as zero beyond the
// matrix and are cropped when P is written, so every shape works.
//
// The half-size products run on the library's GEMM kernels, as its GEMMs
// encode their products (ab_internal_encode_product), on whole workspace
// matrices. Every other step is one kernel, dd_block_combine: Z = X, X + Y or
// X - Y on blocks of row-major matrices with offsets and leading dimensions
// of their own, followed, when it writes P, by the GEMM epilogue
// alpha*Z + beta*Z_old. It forms the S and T sums straight from the
// quadrants of X and Y, and the last additions write P's quadrants: only the
// operands of P1-P4 are copied. The steps go in one serial compute encoder,
// so each sees the writes of the steps before it.
//
// Workspace (strassen.h): two ceil(M/2) x ceil(K/2) matrices, two
// ceil(K/2) x ceil(N/2) and three ceil(M/2) x ceil(N/2), 1.75 N^2 DD values
// for a square N. apple_bottom.m takes them from the staging pool, which
// keeps them between calls, as it keeps the ZGEMMs' temporaries.
//
// Scales: the blocks of X share X's power-of-two scale and those of Y share
// Y's, so every sum adds values on one scale and every product has X's plus
// Y's, as the classical product does. The caller gives P that scale (or the
// scale its epilogue's alpha and beta were adjusted to). The sums add rows of
// X, and columns of Y, from different quadrants, so the lines of the DD
// operands must be on one scale too: the BLAS ABI's calls with lines scaled
// one by one take classical products instead (ab_internal.h).
//
// Accuracy (docs/STRASSEN.md): the classical GEMM's error is bounded
// componentwise, |C - Ĉ| <= γ_K |A||B|; Strassen-Winograd's only normwise,
// max|C - Ĉ| <= (4.5 K^2 + 48 K) u max|A| max|B| (Higham, Accuracy and
// Stability of Numerical Algorithms, 2nd ed., Thm 23.3), so rows or columns
// much smaller than the rest lose relative accuracy.
// =============================================================================

#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#include <os/lock.h>
#include <pthread.h>
#include <stdlib.h>
#include <atomic>

#include "apple_bottom.h"
#include "strassen.h"

// Smallest M, N and K that AB_STRASSEN sends here unless AB_STRASSEN_MIN says
// otherwise.
#define AB_STRASSEN_DEFAULT_MIN_DIM 4096

// =============================================================================
// Metal source
// =============================================================================

static NSString *const kStrassenSource = @(R"MSL(
#include <metal_stdlib>
using namespace metal;

struct DD { float hi; float lo; };

// Error-free transformations and DD arithmetic, as in apple_bottom.m.
inline void twoSum(float a, float b, thread float &s, thread float &e) {
    s = a + b;
    float v = s - a;
    e = (a - (s - v)) + (b - v);
}

inline void fastTwoSum(float a, float b, thread float &s, thread float &e) {
    s = a + b;
    e = b - (s - a);
}

inline void twoProduct(float a, float b, thread float &p, thread float &e) {
    p = a * b;
    e = fma(a, b, -p);
}

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

inline DD dd_sub(DD a, DD b) { return dd_add(a, {-b.hi, -b.lo}); }

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
#else
inline DD dd_mul(DD a, DD b) {
    float p1, e1;
    twoProduct(a.hi, b.hi, p1, e1);
    e1 = fma(a.hi, b.lo, fma(a.lo, b.hi, e1));
    float s, e;
    fastTwoSum(p1, e1, s, e);
    return {s, e};
}
#endif

// A block of a row-major matrix: element (i, j) is at offset + i*ld + j for
// i < rows and j < cols, and reads as zero beyond them.
struct Block { uint offset, ld, rows, cols; };

struct CombineArgs {
    Block x, y, z;
    uint op;        // 0: z = x, 1: z = x + y, 2: z = x - y
    uint scale;     // nonzero: z = alpha*z + beta*z_old, dd_dgemm_ab's epilogue
    DD alpha, beta;
};

inline DD block_read(device const DD *p, Block b, uint i, uint j) {
    return i < b.rows && j < b.cols ? p[b.offset + i * b.ld + j] : DD{0.0f, 0.0f};
}

// One element of z per thread, over z's rows x cols. x, y and z may be the
// same block: each thread reads and writes only its own element.
kernel void dd_block_combine(device const DD *X [[buffer(0)]],
                             device const DD *Y [[buffer(1)]],
                             device DD *Z [[buffer(2)]],
                             constant CombineArgs &a [[buffer(3)]],
                             uint2 gid [[thread_position_in_grid]]) {
    uint i = gid.y, j = gid.x;
    if (i >= a.z.rows || j >= a.z.cols) return;
    DD r = block_read(X, a.x, i, j);
    if (a.op == 1) r = dd_add(r, block_read(Y, a.y, i, j));
    else if (a.op == 2) r = dd_sub(r, block_read(Y, a.y, i, j));
    uint k = a.z.offset + i * a.z.ld + j;
    if (a.scale) {
        r = dd_mul(r, a.alpha);
        if (a.beta.hi != 0.0f || a.beta.lo != 0.0f) r = dd_add(r, dd_mul(Z[k], a.beta));
    }
    Z[k] = r;
}
)MSL");

// =============================================================================
// Host side
// =============================================================================

namespace {
// The MSL structs.
struct DDF { float hi, lo; };
struct Block { uint32_t offset, ld, rows, cols; };
struct CombineArgs {
    Block x, y, z;
    uint32_t op, scale;
    DDF alpha, beta;
};

enum : uint32_t { OP_COPY, OP_ADD, OP_SUB };

// Workspace matrices: two ceil(M/2) x ceil(K/2) (S sums, copies of X's
// quadrants), two ceil(K/2) x ceil(N/2) (T sums, copies of Y's) and three
// ceil(M/2) x ceil(N/2) (products and sums of them).
enum { WS_A0, WS_A1, WS_B0, WS_B1, WS_C0, WS_C1, WS_C2, WS_COUNT };
static_assert(WS_COUNT == AB_STRASSEN_WORKSPACE, "strassen.h's workspace size");

// A block of m, as dd_block_combine reads or writes it; a product takes
// whole workspace matrices.
struct Operand {
    ABMatrix m;
    Block b;
};

// The epilogue of a write to P: P = alpha*U + beta*P if scale, else P = U.
struct Epilogue {
    bool scale;
    DDF alpha, beta;
};

struct Combine {
    id<MTLDevice> device;
    id<MTLComputePipelineState> pso;
};
}  // namespace

// x as DD, as fp64_to_dd gives it.
static DDF to_dd(double x) {
    float hi = (float)x;
    return {hi, (float)(x - (double)hi)};
}

// All of m.
static Operand whole(ABMatrix m) {
    int rows, cols;
    ab_matrix_dims(m, &rows, &cols);
    return {m, {0, (uint32_t)cols, (uint32_t)rows, (uint32_t)cols}};
}

// Block (qi, qj) of m split into h x w blocks; it reads as zero beyond m.
static Operand quadrant(const Operand &m, uint32_t h, uint32_t w, uint32_t qi, uint32_t qj) {
    uint32_t r0 = qi * h, c0 = qj * w;
    return {m.m,
            {m.b.offset + r0 * m.b.ld + c0, m.b.ld, MIN(h, m.b.rows - r0), MIN(w, m.b.cols - c0)}};
}

// =============================================================================
// Pipeline
// =============================================================================

// Built once per device and never freed, so a call reads it without a lock.
static std::atomic<const Combine *> g_combine{nullptr};
static os_unfair_lock g_combine_lock = OS_UNFAIR_LOCK_INIT;

static id<MTLComputePipelineState> build_combine(id<MTLDevice> dev) {
    MTLCompileOptions *opts = [[MTLCompileOptions alloc] init];
    // Fast math would let the compiler break the error-free transformations
    // (apple_bottom.m, ABContextImpl -init, has the details).
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
#ifdef APPLEBOTTOM_USE_DWTIMESDW3
    opts.preprocessorMacros = @{@"APPLEBOTTOM_USE_DWTIMESDW3": @1};
#endif
    NSError *error = nil;
    id<MTLLibrary> lib = [dev newLibraryWithSource:kStrassenSource options:opts error:&error];
    id<MTLFunction> fn = [lib newFunctionWithName:@"dd_block_combine"];
    id<MTLComputePipelineState> pso =
        fn ? [dev newComputePipelineStateWithFunction:fn error:&error] : nil;
    if (!pso) NSLog(@"apple-bottom: Strassen shader build failed: %@", error);
    return pso;
}

// The pipeline for dev, built on first use; NULL if it cannot be.
static const Combine *combine_for(id<MTLDevice> dev) {
    const Combine *c = g_combine.load(std::memory_order_acquire);
    if (c && c->device == dev) return c;
    os_unfair_lock_lock(&g_combine_lock);
    c = g_combine.load(std::memory_order_relaxed);
    if (!c || c->device != dev) {
        // A replaced one stays allocated: another thread may still use it.
        id<MTLComputePipelineState> pso = build_combine(dev);
        c = pso ? new Combine{dev, pso} : NULL;
        if (c) g_combine.store(c, std::memory_order_release);
    }
    os_unfair_lock_unlock(&g_combine_lock);
    return c;
}

// =============================================================================
// Encoding
// =============================================================================

namespace {
struct Encoder {
    id ctx;
    id<MTLComputeCommandEncoder> enc;
    id<MTLComputePipelineState> combine;
    const ABMatrix *W;

    // z = x op y, then the epilogue if e says so.
    void op(uint32_t code, const Operand &z, const Operand &x, const Operand &y,
            Epilogue e) const {
        if (!z.b.rows || !z.b.cols) return;  // a trailing block of P with nothing in it
        CombineArgs a = {x.b, y.b, z.b, code, e.scale, e.alpha, e.beta};
        [enc setComputePipelineState:combine];
        [enc setBuffer:ab_internal_matrix_buffer(x.m) offset:0 atIndex:0];
        [enc setBuffer:ab_internal_matrix_buffer(y.m) offset:0 atIndex:1];
        [enc setBuffer:ab_internal_matrix_buffer(z.m) offset:0 atIndex:2];
        [enc setBytes:&a length:sizeof a atIndex:3];
        [enc dispatchThreads:MTLSizeMake(z.b.cols, z.b.rows, 1)
            threadsPerThreadgroup:MTLSizeMake(64, 4, 1)];
    }
    void copy(const Operand &z, const Operand &x) const { op(OP_COPY, z, x, x, {}); }
    void add(const Operand &z, const Operand &x, const Operand &y, Epilogue e = {}) const {
        op(OP_ADD, z, x, y, e);
    }
    void sub(const Operand &z, const Operand &x, const Operand &y, Epilogue e = {}) const {
        op(OP_SUB, z, x, y, e);
    }

    // p = x * y on whole workspace matrices.
    void gemm(const Operand &p, const Operand &x, const Operand &y) const {
        ab_internal_encode_product(ctx, enc, x.m, y.m, p.m);
    }

    Operand slot(int i) const { return whole(W[i]); }
};
}  // namespace

// C = A*B by one level of Strassen-Winograd, through epi when it writes C. The
// order keeps three product blocks live, not seven.
static void encode_winograd(const Encoder &e, const Operand &A, const Operand &B,
                            const Operand &C, Epilogue epi) {
    uint32_t m = (A.b.rows + 1) / 2, k = (A.b.cols + 1) / 2, n = (B.b.cols + 1) / 2;
    Operand a11 = quadrant(A, m, k, 0, 0), a12 = quadrant(A, m, k, 0, 1),
            a21 = quadrant(A, m, k, 1, 0), a22 = quadrant(A, m, k, 1, 1);
    Operand b11 = quadrant(B, k, n, 0, 0), b12 = quadrant(B, k, n, 0, 1),
            b21 = quadrant(B, k, n, 1, 0), b22 = quadrant(B, k, n, 1, 1);
    Operand c11 = quadrant(C, m, n, 0, 0), c12 = quadrant(C, m, n, 0, 1),
            c21 = quadrant(C, m, n, 1, 0), c22 = quadrant(C, m, n, 1, 1);
    Operand x = e.slot(WS_A0), z = e.slot(WS_A1);
    Operand y = e.slot(WS_B0), w = e.slot(WS_B1);
    Operand pa = e.slot(WS_C0), pb = e.slot(WS_C1), pc = e.slot(WS_C2);

    // C11 = P1 + P2 = A11 B11 + A12 B21, on copies of the quadrants.
    e.copy(z, a11); e.copy(w, b11); e.gemm(pa, z, w);             // pa = P1
    e.copy(z, a12); e.copy(w, b21); e.gemm(pb, z, w);             // pb = P2
    e.add(c11, pa, pb, epi);
    // P5 = S1 T1; U2 = P1 + P6, P6 = S2 T2.
    e.add(z, a21, a22); e.sub(w, b12, b11);                        // z = S1, w = T1
    e.sub(x, z, a11); e.sub(y, b22, w);                            // x = S2, y = T2
    e.gemm(pb, z, w);                                              // pb = P5
    e.gemm(pc, x, y); e.add(pa, pa, pc);                           // pa = U2
    // U3 = U2 + P7, P7 = S3 T3; U4 = U2 + P5; C22 = U3 + P5.
    e.sub(z, a11, a21); e.sub(w, b22, b12); e.gemm(pc, z, w);
    e.add(pc, pa, pc);                                             // pc = U3
    e.add(pa, pa, pb);                                             // pa = U4
    e.add(c22, pc, pb, epi);
    // C12 = U4 + P3, P3 = S4 B22, S4 = A12 - S2.
    e.sub(x, a12, x); e.copy(w, b22); e.gemm(pb, x, w);
    e.add(c12, pa, pb, epi);
    // C21 = U3 - P4, P4 = A22 T4, T4 = T2 - B21.
    e.sub(y, y, b21); e.copy(z, a22); e.gemm(pb, z, y);
    e.sub(c21, pc, pb, epi);
}

// =============================================================================
// Entry points (strassen.h)
// =============================================================================

bool ab_internal_strassen_prepare(id<MTLDevice> dev, int M, int N, int K,
                                  int rows[AB_STRASSEN_WORKSPACE],
                                  int cols[AB_STRASSEN_WORKSPACE]) {
    if (!combine_for(dev)) return false;
    int m = (M + 1) / 2, n = (N + 1) / 2, k = (K + 1) / 2;
    const int r[WS_COUNT] = {m, m, k, k, m, m, m}, c[WS_COUNT] = {k, k, n, n, n, n, n};
    for (int i = 0; i < WS_COUNT; i++) {
        rows[i] = r[i];
        cols[i] = c[i];
    }
    return true;
}

void ab_internal_encode_strassen(id ctx, id<MTLCommandBuffer> cmdBuf, double alpha, ABMatrix X,
                                 ABMatrix Y, double beta, ABMatrix P,
                                 const ABMatrix W[AB_STRASSEN_WORKSPACE]) {
    Encoder e = {ctx, [cmdBuf computeCommandEncoder], combine_for(cmdBuf.device)->pso, W};
    encode_winograd(e, whole(X), whole(Y), whole(P),
                    {alpha != 1.0 || beta != 0.0, to_dd(alpha), to_dd(beta)});
    [e.enc endEncoding];
}

// =============================================================================
// Opt-in routing (AB_STRASSEN, AB_STRASSEN_MIN)
// =============================================================================

static bool g_on;
static int g_min_dim = AB_STRASSEN_DEFAULT_MIN_DIM;
static pthread_once_t g_env_once = PTHREAD_ONCE_INIT;

static void read_env(void) {
    const char *on = getenv("AB_STRASSEN"), *min_dim = getenv("AB_STRASSEN_MIN");
    g_on = on && *on && *on != '0';
    int n = min_dim ? atoi(min_dim) : 0;
    if (n > 0) g_min_dim = n;
}

bool ab_internal_strassen_wanted(int M, int N, int K) {
    pthread_once(&g_env_once, read_env);
    return g_on && M >= g_min_dim && N >= g_min_dim && K >= g_min_dim;
}
