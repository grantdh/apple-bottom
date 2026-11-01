// =============================================================================
// strassen.m — one-level Strassen-Winograd over the double-float GEMM (opt-in)
// Copyright (c) 2026 Grant Heileman, Technology Residue. MIT License.
// =============================================================================
//
// The DD GEMM kernel runs close to the GPU's instruction-issue ceiling, so a
// large product gets faster only by doing less work. One level of Winograd's
// variant of Strassen's algorithm computes C = A*B from 7 products of
// half-size blocks instead of 8, plus 15 block additions (Boyer, Dumas,
// Pernet and Zhou, ISSAC 2009, Alg. 1):
//
//   S1 = A21 + A22   T1 = B12 - B11   P1 = A11 B11   P5 = S1 T1   U2 = P1 + P6
//   S2 = S1 - A11    T2 = B22 - T1    P2 = A12 B21   P6 = S2 T2   U3 = U2 + P7
//   S3 = A11 - A21   T3 = B22 - B12   P3 = S4 B22    P7 = S3 T3   U4 = U2 + P5
//   S4 = A12 - S2    T4 = T2 - B21    P4 = A22 T4
//   C11 = P1 + P2    C12 = U4 + P3    C21 = U3 - P4  C22 = U3 + P5
//
// A (M x K), B (K x N) and C (M x N) split into ceil(M/2) x ceil(K/2),
// ceil(K/2) x ceil(N/2) and ceil(M/2) x ceil(N/2) blocks. The trailing blocks
// of an odd dimension read as zero beyond the matrix and are cropped when C
// is written, so every shape works.
//
// The products run on the library's GEMM kernels, dispatched as the classical
// ZGEMM dispatches its products (ab_internal_encode_dgemm), on contiguous
// workspace blocks. Every other step is one kernel, dd_block_combine:
// Z = X, X + Y or X - Y on blocks of row-major matrices with offsets and
// leading dimensions of their own, followed, when it writes C, by the GEMM
// epilogue alpha*Z + beta*Z_old. It forms the S and T sums straight from the
// quadrants of A and B, and the last additions write C's quadrants: only the
// operands of P1-P4 are copied. A call is one serial compute encoder in one
// command buffer, so each step sees the writes of the steps before it.
//
// Workspace: two ceil(M/2) x ceil(K/2) blocks, two ceil(K/2) x ceil(N/2) and
// three ceil(M/2) x ceil(N/2), 1.75 N^2 DD values for a square N; ZGEMM adds
// M x K, K x N and M x N for Gauss-3M's sums and one of its products. It is
// kept for the next call, grown when a call needs more and freed by
// ab_shutdown(): a fresh Metal buffer costs ~12-16 ms per GB the first time
// the GPU uses it. A mutex serializes the calls that use it.
//
// Accuracy (docs/STRASSEN.md): the classical GEMM's error is bounded
// componentwise, |C - Ĉ| <= γ_K |A||B|; Strassen-Winograd's only normwise,
// max|C - Ĉ| <= (4.5 K^2 + 48 K) u max|A| max|B| (Higham, Accuracy and
// Stability of Numerical Algorithms, 2nd ed., Thm 23.3), so rows or columns
// much smaller than the rest lose relative accuracy.
// =============================================================================

#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#include <pthread.h>
#include <stdlib.h>

#include "apple_bottom.h"
#include "apple_bottom_internal.h"
#include "profiling/blas_profiler.h"
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

// The epilogue of a write to C: C = alpha*U + beta*C if scale, else C = U.
struct Epilogue {
    bool scale;
    DDF alpha, beta;
};

// A block of a matrix in buf, as dd_block_combine reads or writes it.
struct Operand {
    __unsafe_unretained id<MTLBuffer> buf;
    Block b;
};

// Workspace buffers: two ceil(M/2) x ceil(K/2) blocks (S sums, copies of A's
// quadrants), two ceil(K/2) x ceil(N/2) (T sums, copies of B's), three
// ceil(M/2) x ceil(N/2) (products and sums of them) and, for ZGEMM, M x K
// (Ar + Ai), K x N (Br + Bi) and M x N (Ai Bi).
enum { WS_A0, WS_A1, WS_B0, WS_B1, WS_C0, WS_C1, WS_C2, WS_ZA, WS_ZB, WS_ZC, WS_COUNT };

struct Workspace {
    id<MTLDevice> device;
    id<MTLComputePipelineState> combine;
    id<MTLBuffer> buf[WS_COUNT];
};
}  // namespace

static DDF to_dd(double x) {
    float hi = (float)x;
    return {hi, (float)(x - (double)hi)};
}

static uint64_t now_ns(void) { return ab_prof_ticks_to_ns(ab_prof_now_ticks()); }

// All of the row-major rows x cols matrix at the start of buf.
static Operand whole(id<MTLBuffer> buf, uint32_t rows, uint32_t cols) {
    return {buf, {0, cols, rows, cols}};
}

static Operand matrix(ABMatrix m) {
    int rows, cols;
    ab_matrix_dims(m, &rows, &cols);
    return whole(ab_internal_matrix_buffer(m), rows, cols);
}

// Block (qi, qj) of m split into h x w blocks; it reads as zero beyond m.
static Operand quadrant(const Operand &m, uint32_t h, uint32_t w, uint32_t qi, uint32_t qj) {
    uint32_t r0 = qi * h, c0 = qj * w;
    return {m.buf, {m.b.offset + r0 * m.b.ld + c0, m.b.ld, MIN(h, m.b.rows - r0),
                    MIN(w, m.b.cols - c0)}};
}

// =============================================================================
// Workspace
// =============================================================================

// Never freed at exit, only by ab_shutdown(): a fork()ed child, which cannot
// use Metal, must not release the parent's buffers. g_lock is held by a call
// from reserving the workspace until its command buffer completes.
static Workspace *g_ws;
static pthread_mutex_t g_lock = PTHREAD_MUTEX_INITIALIZER;

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

// The workspace, its pipeline built on first use; NULL if Metal refuses the
// pipeline. Called with g_lock held.
static Workspace *workspace(id<MTLDevice> dev) {
    if (!g_ws) {
        id<MTLComputePipelineState> pso = build_combine(dev);
        if (!pso) return NULL;
        g_ws = new Workspace();
        g_ws->device = dev;
        g_ws->combine = pso;
    }
    return g_ws;
}

// Makes ws->buf[i] hold n DD values; false if Metal refuses. A buffer that
// grows is released first, so the old and new ones are never held together.
static bool reserve(Workspace *ws, int i, size_t n) {
    size_t bytes = n * sizeof(DDF);
    if (ws->buf[i] && ws->buf[i].length >= bytes) return true;
    ws->buf[i] = nil;
    ws->buf[i] = [ws->device newBufferWithLength:bytes options:MTLResourceStorageModePrivate];
    return ws->buf[i] != nil;
}

void ab_internal_strassen_release(void) {
    pthread_mutex_lock(&g_lock);
    delete g_ws;
    g_ws = NULL;
    pthread_mutex_unlock(&g_lock);
}

// =============================================================================
// Encoding
// =============================================================================

namespace {
struct Encoder {
    id<MTLComputeCommandEncoder> enc;
    const Workspace *ws;

    // z = x op y, then the epilogue if e says so.
    void op(uint32_t code, const Operand &z, const Operand &x, const Operand &y,
            Epilogue e) const {
        if (!z.b.rows || !z.b.cols) return;  // a trailing block of C with nothing in it
        CombineArgs a = {x.b, y.b, z.b, code, e.scale, e.alpha, e.beta};
        [enc setComputePipelineState:ws->combine];
        [enc setBuffer:x.buf offset:0 atIndex:0];
        [enc setBuffer:y.buf offset:0 atIndex:1];
        [enc setBuffer:z.buf offset:0 atIndex:2];
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

    // p = x * y on whole workspace blocks.
    void gemm(const Operand &p, const Operand &x, const Operand &y) const {
        ab_internal_encode_dgemm(enc, x.buf, y.buf, p.buf, p.b.rows, p.b.cols, x.b.cols);
    }

    // Workspace buffer i as a rows x cols block.
    Operand slot(int i, uint32_t rows, uint32_t cols) const {
        return whole(ws->buf[i], rows, cols);
    }
};
}  // namespace

// C = A*B by one level of Strassen-Winograd, through epi when it writes C. The
// order keeps three product blocks live, not seven.
static void encode_winograd(const Encoder &e, const Operand &A, const Operand &B,
                            const Operand &C, Epilogue epi = {}) {
    uint32_t m = (A.b.rows + 1) / 2, k = (A.b.cols + 1) / 2, n = (B.b.cols + 1) / 2;
    Operand a11 = quadrant(A, m, k, 0, 0), a12 = quadrant(A, m, k, 0, 1),
            a21 = quadrant(A, m, k, 1, 0), a22 = quadrant(A, m, k, 1, 1);
    Operand b11 = quadrant(B, k, n, 0, 0), b12 = quadrant(B, k, n, 0, 1),
            b21 = quadrant(B, k, n, 1, 0), b22 = quadrant(B, k, n, 1, 1);
    Operand c11 = quadrant(C, m, n, 0, 0), c12 = quadrant(C, m, n, 0, 1),
            c21 = quadrant(C, m, n, 1, 0), c22 = quadrant(C, m, n, 1, 1);
    Operand x = e.slot(WS_A0, m, k), z = e.slot(WS_A1, m, k);
    Operand y = e.slot(WS_B0, k, n), w = e.slot(WS_B1, k, n);
    Operand pa = e.slot(WS_C0, m, n), pb = e.slot(WS_C1, m, n), pc = e.slot(WS_C2, m, n);

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

// Runs encode(e) on the workspace, reserved for an M x K by K x N product
// (and ZGEMM's sums and product if complex_gemm) and held until the GPU is
// done, in a command buffer of its own.
template <typename F>
static ABStatus run(bool complex_gemm, size_t M, size_t N, size_t K, F encode) {
    id<MTLCommandQueue> queue = ab_internal_command_queue();
    if (!queue) return AB_ERROR_NO_DEVICE;
    size_t m = (M + 1) / 2, n = (N + 1) / 2, k = (K + 1) / 2, sizes[WS_COUNT] = {};
    sizes[WS_A0] = sizes[WS_A1] = m * k;
    sizes[WS_B0] = sizes[WS_B1] = k * n;
    sizes[WS_C0] = sizes[WS_C1] = sizes[WS_C2] = m * n;
    if (complex_gemm) {
        sizes[WS_ZA] = M * K;
        sizes[WS_ZB] = K * N;
        sizes[WS_ZC] = M * N;
    }

    pthread_mutex_lock(&g_lock);
    uint64_t t0 = now_ns();
    Workspace *ws = workspace(queue.device);
    ABStatus status = ws ? AB_OK : AB_ERROR_SHADER_COMPILE;
    for (int i = 0; status == AB_OK && i < WS_COUNT; i++)
        if (sizes[i] && !reserve(ws, i, sizes[i])) status = AB_ERROR_ALLOC_FAILED;
    if (status == AB_OK)
        status = ab_internal_run_gemm(complex_gemm, now_ns() - t0,
                                      ^(id<MTLComputeCommandEncoder> enc) {
                                          encode(Encoder{enc, ws});
                                      });
    pthread_mutex_unlock(&g_lock);
    return status;
}

// =============================================================================
// Entry points
// =============================================================================

ABStatus ab_dgemm_strassen(double alpha, ABMatrix A, ABMatrix B, double beta, ABMatrix C) {
    @autoreleasepool {
        if (!A || !B || !C) return AB_ERROR_INVALID_ARG;
        Operand a = matrix(A), b = matrix(B), c = matrix(C);
        if (a.b.cols != b.b.rows || a.b.rows != c.b.rows || b.b.cols != c.b.cols)
            return AB_ERROR_DIMENSION_MISMATCH;

        // C = A*B needs no epilogue, as in ab_dgemm.
        Epilogue epi = {alpha != 1.0 || beta != 0.0, to_dd(alpha), to_dd(beta)};
        return run(false, a.b.rows, b.b.cols, a.b.cols,
                   [&](const Encoder &e) { encode_winograd(e, a, b, c, epi); });
    }
}

// Gauss-3M as ab_zgemm computes it, each real product by Strassen-Winograd:
// Ci = (Ar + Ai)(Br + Bi), Cr = Ar Br, T = Ai Bi; Ci = (Ci - Cr) - T, Cr = Cr - T.
ABStatus ab_zgemm_strassen(ABMatrix Ar, ABMatrix Ai, ABMatrix Br, ABMatrix Bi,
                           ABMatrix Cr, ABMatrix Ci) {
    @autoreleasepool {
        if (!Ar || !Ai || !Br || !Bi || !Cr || !Ci) return AB_ERROR_INVALID_ARG;
        Operand ar = matrix(Ar), ai = matrix(Ai), br = matrix(Br), bi = matrix(Bi);
        Operand cr = matrix(Cr), ci = matrix(Ci);
        uint32_t M = ar.b.rows, K = ar.b.cols, N = br.b.cols;
        if (ai.b.rows != M || ai.b.cols != K || br.b.rows != K || bi.b.rows != K ||
            bi.b.cols != N || cr.b.rows != M || cr.b.cols != N || ci.b.rows != M ||
            ci.b.cols != N)
            return AB_ERROR_DIMENSION_MISMATCH;

        return run(true, M, N, K, [&](const Encoder &e) {
            Operand sa = e.slot(WS_ZA, M, K), sb = e.slot(WS_ZB, K, N), t = e.slot(WS_ZC, M, N);
            e.add(sa, ar, ai);
            e.add(sb, br, bi);
            encode_winograd(e, sa, sb, ci);
            encode_winograd(e, ar, br, cr);
            encode_winograd(e, ai, bi, t);
            e.sub(ci, ci, cr);
            e.sub(ci, ci, t);
            e.sub(cr, cr, t);
        });
    }
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
