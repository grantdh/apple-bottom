// =============================================================================
// blas2.m — double-float GEMV and HEMV on the GPU, A read as FP64 in place
// Copyright (c) 2026 Grant Heileman, Technology Residue. MIT License.
// =============================================================================
//
// A matrix-vector product uses each element of A once, so it is bound by
// memory bandwidth, where the GPU (~400 GB/s) is well ahead of the CPU and
// double-float arithmetic costs nothing extra. The kernels read A in the
// caller's FP64 column-major layout, convert each element to DD in registers
// (bit-exact with the host's fp64_to_dd for values in FP32's normal range)
// and accumulate in DD. A therefore needs no conversion pass and can stay
// where the caller keeps it: in an ab_dev_malloc() or ab_dev_register()
// buffer for repeated products, or, through the BLAS ABI, behind a no-copy
// wrapper made for that one call.
//
// Kernels, one MSL library compiled on first use; real and complex are one
// template each, instantiated under the names below:
//   *_n      y = A x. A simdgroup covers 32 consecutive rows and reads 256 B
//            (512 B complex) of one column per step. The simdgroups of a
//            threadgroup, and for few rows a grid dimension, split the
//            columns into chunks.
//   *_t, *_c y = A^T x or A^H x. A simdgroup per column, lanes striding the
//            rows; long columns are split into row chunks across the grid.
//   zhemv_*  the stored triangle, each element read once, in (panel,
//            chunk) tiles: an element adds into its row's sum and,
//            conjugated, into its column's (zhemv_tiles); zhemv_reduce_*
//            adds each output's partial results.
//   reduce_* adds a call's partial results, slot by slot.
// Every sum runs in an order fixed by the shape alone, so results are
// bitwise reproducible.
//
// Range guard. Double-float has FP32's exponent range, and the GPU flushes
// FP32 subnormals to zero. The host scales x by an exact power of two so
// that max |x_j| is in [1/2, 1). For each output the kernels also track,
// from the FP64 exponents E(.), pmax = the largest E(a) + E(x_j) over
// nonzero products and lmax = the same over products whose DD form may
// lose bits under flushing (E(a) or E(x_j) < -78). With L = ceil(log2) of
// the number of real products added into one accumulator, the host accepts
// an output if it is finite and either no product was nonzero or
//     L - 75 <= pmax <= 124 - L   and   lmax <= pmax - 51 - L,
// which keeps overflow out and bounds what flushing can lose, at most
// 2^-124 per term and all of a lossy term (< 2^(lmax+2)), by 2^-49 of the
// largest product. Otherwise y is left alone and the caller runs the whole
// call on Accelerate: NaN and Inf therefore propagate as Accelerate does.
// =============================================================================

#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#include <atomic>
#include <initializer_list>
#include <math.h>
#include <os/lock.h>
#include <pthread.h>
#include <string.h>

#include "apple_bottom.h"
#include "blas2.h"
#include "device_internal.h"
#include "profiling/blas_profiler.h"

// apple_bottom.m: the library's command queue, nil without a context.
id<MTLCommandQueue> ab_internal_command_queue(void);

// =============================================================================
// Metal source
// =============================================================================

static NSString *const kBlas2Source = @(R"MSL(
#include <metal_stdlib>
using namespace metal;

struct DD { float hi; float lo; };
struct DD2 { DD re; DD im; };

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

#ifdef APPLEBOTTOM_USE_DWTIMESDW3
inline DD dd_fma(DD a, DD b, DD c) {   // DWTimesDW3 + add
    float ch = a.hi * b.hi;
    float cl1 = fma(a.hi, b.hi, -ch);
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
#else
inline DD dd_fma(DD a, DD b, DD c) {   // DWTimesDW2 + add
    float p1 = a.hi * b.hi;
    float e1 = fma(a.hi, b.hi, -p1);
    e1 = fma(a.hi, b.lo, fma(a.lo, b.hi, e1));
    float s2, e2;
    twoSum(p1, c.hi, s2, e2);
    e2 += e1 + c.lo;
    fastTwoSum(s2, e2, s2, e2);
    return {s2, e2};
}
#endif

inline DD dd_neg(DD a) { return {-a.hi, -a.lo}; }

// The FP64 value with 32-bit words w = (low, high) as DD, hi rounded to
// nearest-even and lo = RN(x - hi), as the host's fp64_to_dd does; +-0
// below FP32's normal range, +-Inf above it, NaN kept. f64_to_dd_fast
// computes the same bits wherever hi is finite: change both together.
inline DD f64_to_dd(uint2 w) {
    uint hw = w.y, lw = w.x, sgn = hw & 0x80000000u;
    int E = int((hw >> 20) & 0x7FFu) - 1023;
    if (E < -126) return {as_type<float>(sgn), 0.0f};
    if (E > 127) {
        bool nan = E == 1024 && ((hw & 0xFFFFFu) | lw) != 0u;
        return {as_type<float>(sgn | (nan ? 0x7FC00000u : 0x7F800000u)), 0.0f};
    }
    uint mant = ((hw & 0xFFFFFu) << 3) | (lw >> 29), r = lw & 0x1FFFFFFFu;
    bool up = r > 0x10000000u || (r == 0x10000000u && (mant & 1u));
    uint hb = (sgn | (uint(E + 127) << 23) | mant) + (up ? 1u : 0u);
    if ((hb << 1) == 0xFF000000u) return {as_type<float>(hb), 0.0f};   // rounded up to Inf
    int rem = up ? int(r) - 0x20000000 : int(r);                        // x - hi in units of 2^(E-52)
    float lo = rem == 0 ? 0.0f : ldexp(float(sgn ? -rem : rem), E - 52);
    return {as_type<float>(hb), lo};
}

// f64_to_dd's bits wherever its hi is finite, without its branches and
// ldexp: lo = float(rem) * 2^-52 * 2^E, two exact multiplies, flushed to
// zero exactly where ldexp flushes (the GPU flushes FP32 subnormal results
// of both). Elsewhere (|x| >= 2^128 or rounding to it, Inf, NaN) hi is NaN
// or Inf, which leaves every output it enters non-finite, even times zero,
// so the call runs on Accelerate. f64_to_dd was the ZHEMV kernel's largest
// cost after the products themselves.
inline DD f64_to_dd_fast(uint2 w) {
    uint hw = w.y, lw = w.x, sgn = hw & 0x80000000u;
    int E = int((hw >> 20) & 0x7FFu) - 1023;
    uint mant = ((hw & 0xFFFFFu) << 3) | (lw >> 29), r = lw & 0x1FFFFFFFu;
    uint up = (r + (mant & 1u) + 0x0FFFFFFFu) >> 29;             // round half to even
    uint hb = (sgn | (uint(E + 127) << 23) | mant) + up;
    int rem = int(r) - int(up << 29), sm = int(hw) >> 31;        // x - hi in units of 2^(E - 52)
    float lo = float((rem ^ sm) - sm) * 0x1p-52f * as_type<float>(uint(E + 127) << 23);
    hb = E < -126 ? sgn : E > 127 ? 0x7FC00000u : hb;           // +-0 below FP32's range, NaN above
    lo = E < -126 ? 0.0f : lo;
    return {as_type<float>(hb), lo};
}

inline int efield(uint2 w) { return int((w.y >> 20) & 0x7FFu); }      // biased FP64 exponent
inline bool nonzero(uint2 w) { return ((w.y << 1) | w.x) != 0u; }

// Range guard (header of blas2.m). xe[j] codes x_j as E(x_j) - 1023, or
// XE_ZERO for 0, so that efield(a) + xe[j] = E(a) + E(x_j).
constant int G_NONE = -(1 << 30);
constant int XE_ZERO = -(1 << 20);
constant int EA_LOSSY = 1023 - 78;
constant int XE_LOSSY = -78 - 1023;

struct Guard { int pmax; int lmax; };

inline void guard_note(thread Guard &g, int ea, bool anz, int xe) {
    int pe = ea + xe;
    bool nz = anz && xe != XE_ZERO;
    bool lossy = ea < EA_LOSSY || xe < XE_LOSSY;
    g.pmax = max(g.pmax, nz ? pe : G_NONE);
    g.lmax = max(g.lmax, nz && lossy ? pe : G_NONE);
}

inline Guard guard_join(Guard a, Guard b) { return {max(a.pmax, b.pmax), max(a.lmax, b.lmax)}; }

// Accumulators: real (x elements DD) and complex (x elements DD2).
struct AccD { DD s; Guard g; };
struct AccZ { DD re; DD im; Guard g; };

template <typename Acc> Acc acc_zero();
template <> inline AccD acc_zero<AccD>() { return {{0.0f, 0.0f}, {G_NONE, G_NONE}}; }
template <> inline AccZ acc_zero<AccZ>() { return {{0.0f, 0.0f}, {0.0f, 0.0f}, {G_NONE, G_NONE}}; }

inline AccD join(AccD a, AccD b) { return {dd_add(a.s, b.s), guard_join(a.g, b.g)}; }
inline AccZ join(AccZ a, AccZ b) { return {dd_add(a.re, b.re), dd_add(a.im, b.im), guard_join(a.g, b.g)}; }

inline DD shuffle_down(DD a, ushort o) { return {simd_shuffle_down(a.hi, o), simd_shuffle_down(a.lo, o)}; }
inline Guard shuffle_down(Guard g, ushort o) { return {simd_shuffle_down(g.pmax, o), simd_shuffle_down(g.lmax, o)}; }
inline AccD shuffle_down(AccD a, ushort o) { return {shuffle_down(a.s, o), shuffle_down(a.g, o)}; }
inline AccZ shuffle_down(AccZ a, ushort o) { return {shuffle_down(a.re, o), shuffle_down(a.im, o), shuffle_down(a.g, o)}; }

// The sum of each aligned group of 2 FIRST lanes in its first lane (the
// simdgroup's in lane 0 for FIRST = 16), by a fixed tree.
template <ushort FIRST = 16, typename Acc> inline Acc simd_tree(Acc a) {
    for (ushort o = FIRST; o > 0; o >>= 1) a = join(a, shuffle_down(a, o));
    return a;
}

// acc += op(a) x_j for the element at a: one word (real) or two (complex,
// a[0] + i a[1]); op(a) = conj(a) if CONJ.
template <bool CONJ> inline void mac(thread AccD &acc, device const uint2 *a, DD x, int xe) {
    uint2 w = a[0];
    guard_note(acc.g, efield(w), nonzero(w), xe);
    acc.s = dd_fma(f64_to_dd(w), x, acc.s);
}

// acc += op(a) x for a = ar + i ai, without the guard.
template <bool CONJ> inline void zmac(thread AccZ &acc, DD ar, DD ai, DD2 x) {
    if (CONJ) ai = dd_neg(ai);
    acc.re = dd_fma(ar, x.re, acc.re);
    acc.re = dd_fma(dd_neg(ai), x.im, acc.re);
    acc.im = dd_fma(ar, x.im, acc.im);
    acc.im = dd_fma(ai, x.re, acc.im);
}

template <bool CONJ> inline void mac(thread AccZ &acc, device const uint2 *a, DD2 x, int xe) {
    uint2 wr = a[0], wi = a[1];
    guard_note(acc.g, max(efield(wr), efield(wi)), nonzero(wr) || nonzero(wi), xe);
    zmac<CONJ>(acc, f64_to_dd(wr), f64_to_dd(wi), x);
}

struct Params {
    ulong a0;     // A(0,0), in 8-byte words from the start of buffer 0
    ulong lda;    // column stride in 8-byte words
    uint m, n;    // A is m x n
    uint chunk;   // columns (rows kernels) or rows (cols kernels) per chunk
    uint nout;    // outputs: m (rows kernels) or n (cols kernels)
};

// Rows kernels: row i over one chunk of columns. The threadgroup's
// simdgroups add their chunks of row i in order and write slot (chunk group).
template <typename Acc, typename X>
[[kernel]] void gemv_rows(device const uint2 *A [[buffer(0)]], device const X *x [[buffer(1)]],
                          device const int *xe [[buffer(2)]], device Acc *out [[buffer(3)]],
                          constant Params &p [[buffer(4)]], threadgroup Acc *tmp [[threadgroup(0)]],
                          uint2 tg [[threadgroup_position_in_grid]],
                          uint lane [[thread_index_in_simdgroup]],
                          uint sg [[simdgroup_index_in_threadgroup]],
                          uint nsg [[simdgroups_per_threadgroup]]) {
    const uint W = sizeof(X) / sizeof(DD);   // words per element
    uint i = tg.x * 32 + lane, c = tg.y * nsg + sg;
    uint j0 = min(c * p.chunk, p.n), j1 = min(j0 + p.chunk, p.n);
    Acc acc = acc_zero<Acc>();
    if (i < p.m && j0 < j1) {
        device const uint2 *a = A + p.a0 + W * (ulong)i + (ulong)j0 * p.lda;
        for (uint j = j0; j < j1; j++, a += p.lda) mac<false>(acc, a, x[j], xe[j]);
    }
    tmp[sg * 32 + lane] = acc;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (sg != 0 || i >= p.m) return;
    for (uint s = 1; s < nsg; s++) acc = join(acc, tmp[s * 32 + lane]);
    out[(ulong)tg.y * p.nout + i] = acc;
}

// Cols kernels: column j over one chunk of rows, op(a) = conj(a) if CONJ.
template <typename Acc, typename X, bool CONJ>
[[kernel]] void gemv_cols(device const uint2 *A [[buffer(0)]], device const X *x [[buffer(1)]],
                          device const int *xe [[buffer(2)]], device Acc *out [[buffer(3)]],
                          constant Params &p [[buffer(4)]],
                          uint2 tg [[threadgroup_position_in_grid]],
                          uint lane [[thread_index_in_simdgroup]],
                          uint sg [[simdgroup_index_in_threadgroup]],
                          uint nsg [[simdgroups_per_threadgroup]]) {
    const uint W = sizeof(X) / sizeof(DD);
    uint j = tg.x * nsg + sg;
    if (j >= p.n) return;   // the whole simdgroup
    uint r0 = min(tg.y * p.chunk, p.m), r1 = min(r0 + p.chunk, p.m);
    Acc acc = acc_zero<Acc>();
    device const uint2 *col = A + p.a0 + (ulong)j * p.lda;
    for (uint k = r0 + lane; k < r1; k += 32) mac<CONJ>(acc, col + W * k, x[k], xe[k]);
    acc = simd_tree(acc);
    if (lane == 0) out[(ulong)tg.y * p.nout + j] = acc;
}

// out[i] = the sum of slots 0..np.y-1 of output i, in slot order.
template <typename Acc>
[[kernel]] void reduce_slots(device const Acc *part [[buffer(0)]], device Acc *out [[buffer(1)]],
                             constant uint2 &np [[buffer(2)]], uint i [[thread_position_in_grid]]) {
    if (i >= np.x) return;
    Acc acc = part[i];
    for (uint s = 1; s < np.y; s++) acc = join(acc, part[(ulong)s * np.x + i]);
    out[i] = acc;
}

// ZHEMV reads each stored element once. The rows split into panels of
// H = 32 R rows, lane l of a simdgroup holding rows l, l + 32, ... of its
// panel, and the columns into chunks of p.chunk, a multiple of H. A
// simdgroup takes one (panel, chunk) and, for each stored a = A(i, j), adds
// a x_j to row i's sum, kept in registers across the chunk, and, unless
// i == j, conj(a) x_i to column j's. A column's sum runs over the lane's R
// rows, then across the lanes, four columns at a time (hemv_cols). The sums
// leave as partial results per (panel, column) and per (chunk, row), packed
// (band_base), and zhemv_reduce adds those of each output in a fixed order.
struct HemvParams {
    ulong a0, lda;    // as Params
    ulong dbase;      // the first (chunk, row) partial, after the (panel, column) ones
    uint n;
    uint chunk;       // columns per chunk W
    uint panel;       // rows per panel H
    uint panels, chunks;
};

// Guard codes. An element's: e = its larger biased FP64 exponent, or
// XE_ZERO if it is zero, and l = e if lossy (e < EA_LOSSY), else XE_ZERO;
// an x element's: e = xe[j] and l = xe[j] if lossy, else XE_ZERO. A
// product's code e_a + e_x is then at least -2046 if both factors are
// nonzero and below G_LOW otherwise, so pmax and lmax are plain maxima
// (hemv_note), mapped back to G_NONE below G_LOW once an output's sum is
// complete (hemv_final): the values guard_note gives. The mapping commutes
// with max, so the partial results keep the raw codes.
constant int G_LOW = -(1 << 19);

struct Code { int e; int l; };

inline Code acode(uint2 wr, uint2 wi) {
    int e = nonzero(wr) || nonzero(wi) ? max(efield(wr), efield(wi)) : XE_ZERO;
    return {e, e < EA_LOSSY ? e : XE_ZERO};
}

inline Code xcode(int xe) { return {xe, xe < XE_LOSSY ? xe : XE_ZERO}; }

inline void hemv_note(thread Guard &g, Code a, Code x) {
    g.pmax = max(g.pmax, a.e + x.e);
    g.lmax = max(g.lmax, max(a.l + x.e, a.e + x.l));
}

inline AccZ hemv_final(AccZ a) {
    a.g.pmax = a.g.pmax < G_LOW ? G_NONE : a.g.pmax;
    a.g.lmax = a.g.lmax < G_LOW ? G_NONE : a.g.lmax;
    return a;
}

// The packed partial results of blocks of B indices (a panel's rows or a
// chunk's columns): block b holds the indices [b B, n) if TAIL, else
// [0, min(n, (b + 1) B)), and its partial for index k sits at
// band_base(b) + k. Upper, panels hold columns [s H, n) and chunks rows
// [0, ...); lower, the other way round.
template <bool TAIL> inline ulong band_base(uint b, uint n, uint B) {
    const ulong head = (ulong)B * b * (b + 1) / 2;
    return TAIL ? (ulong)b * n - head : head;
}

// One simdgroup's threadgroup scratch: four columns' sums per lane, by
// component, in rows of 33 words so that neither the writes (lanes along a
// row) nor the reads (lanes across rows) conflict.
struct HemvTg {
    float rh[4][33], rl[4][33], ih[4][33], il[4][33];
    int pm[4][33], lm[4][33];
};

inline void tg_put(threadgroup HemvTg &t, int c, ushort l, AccZ a) {
    t.rh[c][l] = a.re.hi; t.rl[c][l] = a.re.lo; t.ih[c][l] = a.im.hi; t.il[c][l] = a.im.lo;
    t.pm[c][l] = a.g.pmax; t.lm[c][l] = a.g.lmax;
}

inline AccZ tg_get(threadgroup const HemvTg &t, int c, int l) {
    return {{t.rh[c][l], t.rl[c][l]}, {t.ih[c][l], t.il[c][l]}, {t.pm[c][l], t.lm[c][l]}};
}

// Columns [j, j + 4) of the panel at row r0, adding into the lane's row
// sums `rows`; with MASK, only the stored elements of rows < n and columns
// < jend (the diagonal block, the matrix's last rows or columns). Each
// column's sum over the lane's rows goes to the scratch; then each column's
// eight lanes add four lanes' sums each and add down to the first, which
// writes tp[column].
template <int R, bool UPPER, bool MASK>
inline void hemv_cols(device const uint2 *A, device const DD2 *x, device const int *xe,
                      constant HemvParams &p, uint r0, uint j, uint jend, ushort lane,
                      thread AccZ (&rows)[R], threadgroup HemvTg &tgs, device AccZ *tp) {
    for (int c = 0; c < 4; c++) {
        const uint jc = j + c, jl = !MASK || jc < jend ? jc : j;   // jl: a column to read
        const DD2 xj = x[jl];
        const Code cx = xcode(xe[jl]);
        device const uint2 *col = A + p.a0 + (ulong)jl * p.lda;
        AccZ cs = acc_zero<AccZ>();
        for (int k = 0; k < R; k++) {
            const uint i = r0 + lane + 32 * k, il = MASK ? min(i, p.n - 1) : i;   // il: a row to read
            device const uint2 *a = col + 2 * il;
            DD2 xi = x[il];   // read here rather than kept: fewer registers
            Code ci = xcode(xe[il]);
            uint2 wr, wi;
            if (MASK) {
                bool stored = jl == jc && i < p.n && (UPPER ? i <= jc : i >= jc), diag = i == jc;
                wr = stored ? a[0] : uint2(0u);
                wi = stored && !diag ? a[1] : uint2(0u);   // a diagonal element's imaginary part: unread
                if (diag) {   // counted once, in its row
                    xi = {{0.0f, 0.0f}, {0.0f, 0.0f}};
                    ci = {XE_ZERO, XE_ZERO};
                }
            } else {
                wr = a[0];
                wi = a[1];
            }
            const DD ar = f64_to_dd_fast(wr), ai = f64_to_dd_fast(wi);
            const Code ca = acode(wr, wi);
            zmac<false>(rows[k], ar, ai, xj);   // row i += a x_j
            hemv_note(rows[k].g, ca, cx);
            zmac<true>(cs, ar, ai, xi);         // column j += conj(a) x_i
            hemv_note(cs.g, ca, ci);
        }
        tg_put(tgs, c, lane, cs);
    }
    simdgroup_barrier(mem_flags::mem_threadgroup);
    const ushort cc = lane >> 3, sub = lane & 7;
    AccZ t = tg_get(tgs, cc, 4 * sub);
    for (int u = 1; u < 4; u++) t = join(t, tg_get(tgs, cc, 4 * sub + u));
    t = simd_tree<4>(t);
    const uint jt = j + cc;
    if (sub == 0 && (!MASK || jt < jend)) tp[jt] = t;
    simdgroup_barrier(mem_flags::mem_threadgroup);
}

template <int R, bool UPPER>
[[kernel]] void zhemv_tiles(device const uint2 *A [[buffer(0)]], device const DD2 *x [[buffer(1)]],
                            device const int *xe [[buffer(2)]], device AccZ *part [[buffer(3)]],
                            constant HemvParams &p [[buffer(4)]],
                            threadgroup HemvTg *tgm [[threadgroup(0)]],
                            uint2 tg [[threadgroup_position_in_grid]],
                            ushort lane [[thread_index_in_simdgroup]],
                            ushort sg [[simdgroup_index_in_threadgroup]],
                            ushort nsg [[simdgroups_per_threadgroup]]) {
    const uint H = 32 * R, n = p.n, s = tg.y, q = tg.x * nsg + sg, r0 = s * H;
    uint c0 = q * p.chunk, c1 = min(c0 + p.chunk, n);   // this chunk's stored columns
    if (UPPER) c0 = max(c0, r0);
    else c1 = min(c1, r0 + H);
    if (c0 >= c1) return;   // none, or q past the last chunk

    AccZ rows[R];
    for (int k = 0; k < R; k++) rows[k] = acc_zero<AccZ>();
    threadgroup HemvTg &tgs = tgm[sg];
    device AccZ *tp = part + band_base<UPPER>(s, n, H);
    // The diagonal block [d0, d1), masked, and the rest of the chunk,
    // [f0, f1), where every panel row is stored: unmasked in whole groups of
    // four columns unless the panel runs past row n. Left to right.
    const uint d0 = clamp(r0, c0, c1), d1 = clamp(r0 + H, c0, c1);
    const uint f0 = UPPER ? d1 : c0, f1 = UPPER ? c1 : d0;
    const uint fu = r0 + H <= n ? f0 + ((f1 - f0) & ~3u) : f0;
    if (UPPER) for (uint j = d0; j < d1; j += 4) hemv_cols<R, UPPER, true>(A, x, xe, p, r0, j, d1, lane, rows, tgs, tp);
    for (uint j = f0; j < fu; j += 4) hemv_cols<R, UPPER, false>(A, x, xe, p, r0, j, fu, lane, rows, tgs, tp);
    for (uint j = fu; j < f1; j += 4) hemv_cols<R, UPPER, true>(A, x, xe, p, r0, j, f1, lane, rows, tgs, tp);
    if (!UPPER) for (uint j = d0; j < d1; j += 4) hemv_cols<R, UPPER, true>(A, x, xe, p, r0, j, d1, lane, rows, tgs, tp);
    device AccZ *dp = part + p.dbase + band_base<!UPPER>(q, n, p.chunk);
    for (int k = 0; k < R; k++) {
        const uint i = r0 + lane + 32 * k;
        if (i < n) dp[i] = rows[k];
    }
}

// acc joined with index k's partials in blocks [b0, b1) of a band
// (band_base), in block order, the loads issued four ahead of their joins.
template <bool TAIL>
inline AccZ join_band(AccZ acc, device const AccZ *band, uint b0, uint b1, uint n, uint B, uint k) {
    uint b = b0;
    for (; b + 4 <= b1; b += 4) {
        AccZ v[4];
        for (int u = 0; u < 4; u++) v[u] = band[band_base<TAIL>(b + u, n, B) + k];
        for (int u = 0; u < 4; u++) acc = join(acc, v[u]);
    }
    for (; b < b1; b++) acc = join(acc, band[band_base<TAIL>(b, n, B) + k]);
    return acc;
}

// out[k] = the sum of output k's partials: upper, its (panel, column)
// partials by panel, then its (chunk, row) ones by chunk (row k of the
// Hermitian matrix left to right); lower, the (chunk, row) ones, then the
// (panel, column) ones.
template <bool UPPER>
[[kernel]] void zhemv_reduce(device const AccZ *part [[buffer(0)]], device AccZ *out [[buffer(1)]],
                             constant HemvParams &p [[buffer(2)]], uint k [[thread_position_in_grid]]) {
    if (k >= p.n) return;
    const uint n = p.n, H = p.panel, W = p.chunk;
    device const AccZ *dp = part + p.dbase;
    AccZ acc = acc_zero<AccZ>();
    if (UPPER) {
        acc = join_band<true>(acc, part, 0, k / H + 1, n, H, k);
        acc = join_band<false>(acc, dp, k / W, p.chunks, n, W, k);
    } else {
        acc = join_band<true>(acc, dp, 0, k / W + 1, n, W, k);
        acc = join_band<false>(acc, part, k / H, p.panels, n, H, k);
    }
    out[k] = hemv_final(acc);
}

template [[host_name("dgemv_n")]] [[kernel]] decltype(gemv_rows<AccD, DD>) gemv_rows<AccD, DD>;
template [[host_name("zgemv_n")]] [[kernel]] decltype(gemv_rows<AccZ, DD2>) gemv_rows<AccZ, DD2>;
template [[host_name("dgemv_t")]] [[kernel]] decltype(gemv_cols<AccD, DD, false>) gemv_cols<AccD, DD, false>;
template [[host_name("zgemv_t")]] [[kernel]] decltype(gemv_cols<AccZ, DD2, false>) gemv_cols<AccZ, DD2, false>;
template [[host_name("zgemv_c")]] [[kernel]] decltype(gemv_cols<AccZ, DD2, true>) gemv_cols<AccZ, DD2, true>;
template [[host_name("reduce_d")]] [[kernel]] decltype(reduce_slots<AccD>) reduce_slots<AccD>;
template [[host_name("reduce_z")]] [[kernel]] decltype(reduce_slots<AccZ>) reduce_slots<AccZ>;
template [[host_name("zhemv_u1")]] [[kernel]] decltype(zhemv_tiles<1, true>) zhemv_tiles<1, true>;
template [[host_name("zhemv_l1")]] [[kernel]] decltype(zhemv_tiles<1, false>) zhemv_tiles<1, false>;
template [[host_name("zhemv_u3")]] [[kernel]] decltype(zhemv_tiles<3, true>) zhemv_tiles<3, true>;
template [[host_name("zhemv_l3")]] [[kernel]] decltype(zhemv_tiles<3, false>) zhemv_tiles<3, false>;
template [[host_name("zhemv_reduce_u")]] [[kernel]] decltype(zhemv_reduce<true>) zhemv_reduce<true>;
template [[host_name("zhemv_reduce_l")]] [[kernel]] decltype(zhemv_reduce<false>) zhemv_reduce<false>;
)MSL");

// =============================================================================
// Pipelines
// =============================================================================

typedef enum {
    K_DGEMV_N, K_ZGEMV_N,                  // rows kernels
    K_DGEMV_T, K_ZGEMV_T, K_ZGEMV_C,       // cols kernels
    K_REDUCE_D, K_REDUCE_Z,
    K_ZHEMV_U1, K_ZHEMV_L1, K_ZHEMV_U3, K_ZHEMV_L3, K_ZHEMV_REDUCE_U, K_ZHEMV_REDUCE_L,
    K_COUNT
} KernelId;

static const char *const kKernelNames[K_COUNT] = {
    "dgemv_n", "zgemv_n", "dgemv_t", "zgemv_t", "zgemv_c", "reduce_d", "reduce_z",
    "zhemv_u1", "zhemv_l1", "zhemv_u3", "zhemv_l3", "zhemv_reduce_u", "zhemv_reduce_l",
};

static bool is_rows_kernel(KernelId k) { return k <= K_ZGEMV_N; }

namespace {
struct Pipelines {
    id<MTLDevice> device;
    id<MTLComputePipelineState> pso[K_COUNT];
};
}  // namespace

// Built once per device and never freed, so a call reads them without a lock.
static std::atomic<const Pipelines *> g_pipelines{nullptr};
static os_unfair_lock g_pipelines_lock = OS_UNFAIR_LOCK_INIT;

// Compiles the library for dev into *out; false if Metal refuses.
static bool build_pipelines(id<MTLDevice> dev, Pipelines *out) {
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
    id<MTLLibrary> lib = [dev newLibraryWithSource:kBlas2Source options:opts error:&error];
    if (!lib) {
        NSLog(@"apple-bottom: BLAS-2 shader compile failed: %@", error);
        return false;
    }
    for (int k = 0; k < K_COUNT; k++) {
        id<MTLFunction> fn = [lib newFunctionWithName:@(kKernelNames[k])];
        out->pso[k] = fn ? [dev newComputePipelineStateWithFunction:fn error:&error] : nil;
        if (!out->pso[k]) {
            NSLog(@"apple-bottom: BLAS-2 pipeline %s failed: %@", kKernelNames[k], error);
            return false;
        }
    }
    out->device = dev;
    return true;
}

// The pipelines for dev, built on first use; NULL if they cannot be.
static const Pipelines *pipelines_for(id<MTLDevice> dev) {
    const Pipelines *pl = g_pipelines.load(std::memory_order_acquire);
    if (pl && pl->device == dev) return pl;
    os_unfair_lock_lock(&g_pipelines_lock);
    pl = g_pipelines.load(std::memory_order_relaxed);
    if (!pl || pl->device != dev) {
        // A replaced set stays allocated: another thread may still use it.
        Pipelines *fresh = new Pipelines();
        if (build_pipelines(dev, fresh)) {
            g_pipelines.store(fresh, std::memory_order_release);
            pl = fresh;
        } else {
            delete fresh;
            pl = NULL;
        }
    }
    os_unfair_lock_unlock(&g_pipelines_lock);
    return pl;
}

// =============================================================================
// Per-thread scratch buffers
// =============================================================================
// x and its exponent codes, partial results and outputs, grown as needed
// and kept for the thread's next call (calls wait for their GPU work, so a
// thread never reuses a buffer the GPU still reads); one larger than
// kScratchKeepBytes is dropped after its call. Held through a pthread key
// rather than thread_local: a fork()ed child, which cannot use Metal, must
// not release the parent's buffers when it exits.

namespace {
struct Scratch {
    id<MTLBuffer> x, xe, part, out;
};

const size_t kScratchKeepBytes = (size_t)64 << 20;

// Drops the oversized buffers when a call returns.
struct ScratchTrim {
    Scratch *s;
    ~ScratchTrim() {
        for (id<MTLBuffer> __strong *b : {&s->x, &s->xe, &s->part, &s->out})
            if (*b && (*b).length > kScratchKeepBytes) *b = nil;
    }
};
}  // namespace

static pthread_key_t g_scratch_key;
static pthread_once_t g_scratch_once = PTHREAD_ONCE_INIT;

static void scratch_free(void *p) {
    @autoreleasepool { delete static_cast<Scratch *>(p); }
}

static void scratch_key_init(void) { pthread_key_create(&g_scratch_key, scratch_free); }

static Scratch *thread_scratch(void) {
    pthread_once(&g_scratch_once, scratch_key_init);
    Scratch *s = static_cast<Scratch *>(pthread_getspecific(g_scratch_key));
    if (!s) {
        s = new Scratch();
        pthread_setspecific(g_scratch_key, s);
    }
    return s;
}

// slot, grown to at least bytes on dev; nil if Metal refuses (or under
// AB_TEST_FAIL_STAGE=alloc).
static id<MTLBuffer> scratch_buffer(id<MTLBuffer> __strong &slot, id<MTLDevice> dev, size_t bytes) {
    if (ab_test_fail_stage() == AB_PROF_STAGE_ALLOC) return nil;
    if (slot && slot.device == dev && slot.length >= bytes) return slot;
    slot = nil;
    // Room to grow only in a buffer that ScratchTrim keeps.
    size_t roomy = bytes > kScratchKeepBytes ? bytes
                                             : MIN(MAX(bytes + bytes / 2, (size_t)1 << 16), kScratchKeepBytes);
    slot = [dev newBufferWithLength:roomy options:MTLResourceStorageModeShared];
    if (!slot) slot = [dev newBufferWithLength:bytes options:MTLResourceStorageModeShared];
    return slot;
}

// =============================================================================
// Host side of a call
// =============================================================================

typedef struct { float hi, lo; } DDf;
typedef struct { DDf s; int32_t pmax, lmax; } AccDHost;               // = MSL AccD
typedef struct { DDf re, im; int32_t pmax, lmax; } AccZHost;          // = MSL AccZ

typedef struct {
    uint64_t a0, lda;
    uint32_t m, n, chunk, nout;
} ParamsHost;                                                         // = MSL Params

typedef struct {
    uint64_t a0, lda, dbase;
    uint32_t n, chunk, panel, panels, chunks;
} HemvParamsHost;                                                     // = MSL HemvParams

typedef struct {
    float rh[4][33], rl[4][33], ih[4][33], il[4][33];
    int32_t pm[4][33], lm[4][33];
} HemvTgHost;                                                         // = MSL HemvTg

static const int32_t kGuardNone = -(1 << 30);
static const int32_t kXeZero = -(1 << 20);

// Threads a call aims to keep in flight, and the fewest columns (rows) a
// chunk of a rows (cols) kernel gets.
static const uint32_t kTargetThreads = 1u << 17;
static const uint32_t kMinChunkRows = 64;
static const uint32_t kMinChunkCols = 256;

static inline uint32_t ceil_div(uint64_t a, uint64_t b) { return (uint32_t)((a + b - 1) / b); }

static inline uint64_t bits_of(double v) {
    uint64_t u;
    memcpy(&u, &v, sizeof u);
    return u;
}

// Element i of a BLAS vector argument with increment inc (len elements).
static inline size_t vec_index(int len, int inc, int i) {
    return inc > 0 ? (size_t)i * inc : (size_t)(len - 1 - i) * (size_t)(-(ptrdiff_t)inc);
}

// The exponent sx that scales x (len elements of w doubles, increment inc)
// to max |x| in [1/2, 1): max |x| < 2^(emax - 1022) for the largest biased
// exponent emax, FP64 subnormals included; 0 for x == 0. False if x holds
// NaN or Inf.
static bool x_scale(const ab_b2_call_t *c, int len, int w, int *sx) {
    const size_t step = (size_t)(c->incx < 0 ? -(ptrdiff_t)c->incx : c->incx) * w;
    uint64_t top = 0;   // the largest magnitude's bits; NaN and Inf are above every finite value
    const double *v = c->x;
    for (int i = 0; i < len; i++, v += step)
        for (int k = 0; k < w; k++) {
            uint64_t b = bits_of(v[k]) & ~(1ULL << 63);
            top = b > top ? b : top;
        }
    *sx = top ? (int)(top >> 52) - 1022 : 0;
    return top < 0x7FF0000000000000ULL;
}

// Stages x for the kernels: scaled by 2^-sx (exact: sx is in [-1022, 1024],
// so 2^-sx is a double and the product rounds as ldexp does), as DD, and
// the exponent codes xe = E(x_j) - 1023, or kXeZero for 0.
static void x_stage(const ab_b2_call_t *c, int len, int w, int sx, DDf *xdd, int32_t *xe) {
    const double scale = ldexp(1.0, -sx);
    for (int i = 0; i < len; i++) {
        const double *v = c->x + vec_index(len, c->incx, i) * w;
        int ex = -1;
        for (int k = 0; k < w; k++) {
            double s = v[k] * scale;
            float hi = (float)s;
            xdd[(size_t)i * w + k] = (DDf){hi, (float)(s - (double)hi)};
            int e = s != 0.0 ? (int)((bits_of(s) >> 52) & 0x7FF) : -1;
            ex = e > ex ? e : ex;
        }
        xe[i] = ex < 0 ? kXeZero : ex - 2046;
    }
}

// ceil(log2(terms)).
static int ceil_log2(uint64_t terms) {
    int l = 0;
    while (((uint64_t)1 << l) < terms) l++;
    return l;
}

// The range guard's verdict on one output (header of this file); L is
// ceil(log2) of the real products added into one accumulator.
// Branch-free (int, not bool) so that the check of all outputs vectorizes.
static inline int output_ok(int32_t pmax, int32_t lmax, int L) {
    int none = pmax == kGuardNone, in_range = (pmax >= L - 75) & (pmax <= 124 - L);
    int lossy_ok = (lmax == kGuardNone) | (lmax <= pmax - 51 - L);
    return none | (in_range & lossy_ok);
}

static inline int dd_finite(DDf d) { return (int)isfinite(d.hi) & (int)isfinite(d.lo); }

// 2^sx (hi + lo) in FP64: a multiply by the exact power, which rounds as
// ldexp does, except for 2^1024, which is not a double.
static inline double dd_value(DDf d, int sx, double up) {
    double t = (double)d.hi + (double)d.lo;
    return sx <= 1023 ? t * up : ldexp(t, sx);
}

namespace {
// One dispatch of a rows or cols kernel, writing partial-result slots
// 0 .. groups - 1 (the chunk groups along grid y).
struct Pass {
    KernelId kernel;
    MTLSize grid, group;    // threadgroups, threads per threadgroup
    NSUInteger tg_mem;      // threadgroup memory (rows kernels)
    uint32_t groups;
    ParamsHost p;
};
}  // namespace

// The decomposition of kernel k over an m x n matrix: the columns split into
// chunks (rows kernels) or the rows (cols kernels).
static Pass plan_pass(const Pipelines *pl, KernelId k, uint32_t m, uint32_t n, uint64_t a0,
                      uint64_t ldw, size_t acc_size) {
    bool rows = is_rows_kernel(k);
    Pass ps = {k, {}, {}, 0, 1, {a0, ldw, m, n, 0, rows ? m : n}};
    uint32_t simdgroups =
        (uint32_t)MIN(MAX(pl->pso[k].maxTotalThreadsPerThreadgroup / 32, (NSUInteger)1), (NSUInteger)32);
    if (rows) {
        uint32_t row_blocks = ceil_div(m, 32);
        uint32_t want = ceil_div(kTargetThreads, (uint64_t)row_blocks * 32);
        uint32_t chunks = MAX(1u, MIN(want, ceil_div(n, kMinChunkRows)));
        simdgroups = MIN(chunks, simdgroups);
        ps.groups = ceil_div(chunks, simdgroups);
        ps.p.chunk = ceil_div(n, (uint64_t)simdgroups * ps.groups);
        ps.grid = MTLSizeMake(row_blocks, ps.groups, 1);
        ps.tg_mem = (NSUInteger)simdgroups * 32 * acc_size;
    } else {
        simdgroups = MIN(8u, simdgroups);
        uint32_t want = ceil_div(kTargetThreads, (uint64_t)n * 32);
        uint32_t chunks = MAX(1u, MIN(want, ceil_div(m, kMinChunkCols)));
        ps.p.chunk = ceil_div(m, chunks);
        ps.groups = ceil_div(m, ps.p.chunk);
        ps.grid = MTLSizeMake(ceil_div(n, simdgroups), ps.groups, 1);
    }
    ps.group = MTLSizeMake(32 * simdgroups, 1, 1);
    return ps;
}

// ZHEMV's decomposition (zhemv_tiles): panels of H = 32 R rows and chunks
// of W columns, W a multiple of H. R = 3 adds the column sums across the
// lanes a third as often per element as R = 1, but leaves a small call too
// few (panel, chunk) work items to fill the GPU: kHemvBigN is the measured
// crossover (M2 Max: R = 1 5% faster at n = 4096, R = 3 6% faster at 5120).
// W aims at kTargetThreads / 32 work items, a simdgroup each.
static const uint32_t kHemvBigN = 4608;
static const uint32_t kHemvSimdgroups = 2;   // per threadgroup, a work item each

static thread_local int t_hemv_rows = 0, t_hemv_chunk_panels = 0;   // ab_internal_b2_hemv_override

void ab_internal_b2_hemv_override(int rows_per_lane, int chunk_panels) {
    t_hemv_rows = rows_per_lane;
    t_hemv_chunk_panels = chunk_panels;
}

// = MSL band_base: where block b's packed partial results sit, by row or column.
static inline uint64_t band_base(bool tail, uint32_t b, uint32_t n, uint32_t B) {
    const uint64_t head = (uint64_t)B * b * (b + 1) / 2;
    return tail ? (uint64_t)b * n - head : head;
}

namespace {
struct HemvPlan {
    KernelId kernel, reduce;
    HemvParamsHost p;
    uint64_t parts;         // partial results: the (panel, column) ones, then the (chunk, row) ones
    MTLSize grid, group;    // threadgroups, threads per threadgroup
};
}  // namespace

static HemvPlan plan_hemv(bool up, uint32_t n, uint64_t a0, uint64_t ldw) {
    const uint32_t R = t_hemv_rows == 1 || t_hemv_rows == 3 ? (uint32_t)t_hemv_rows : n < kHemvBigN ? 1 : 3;
    const uint32_t H = 32 * R;
    const uint64_t per_chunk = t_hemv_chunk_panels > 0
                                   ? (uint64_t)t_hemv_chunk_panels
                                   : (uint64_t)n * n / (2 * (uint64_t)H * H * (kTargetThreads / 32));
    const uint32_t W = H * (uint32_t)MAX(per_chunk, (uint64_t)1), P = ceil_div(n, H), Q = ceil_div(n, W);
    // Each band's last block ends at n.
    const uint64_t T = band_base(up, P - 1, n, H) + n, D = band_base(!up, Q - 1, n, W) + n;
    HemvPlan hp;
    hp.kernel = R == 1 ? (up ? K_ZHEMV_U1 : K_ZHEMV_L1) : (up ? K_ZHEMV_U3 : K_ZHEMV_L3);
    hp.reduce = up ? K_ZHEMV_REDUCE_U : K_ZHEMV_REDUCE_L;
    hp.p = HemvParamsHost{a0, ldw, T, n, W, H, P, Q};
    hp.parts = T + D;
    hp.grid = MTLSizeMake(ceil_div(Q, kHemvSimdgroups), P, 1);
    hp.group = MTLSizeMake(32 * kHemvSimdgroups, 1, 1);
    return hp;
}

void ab_internal_b2_hemv_layout(int n, int *panel, int *chunk) {
    HemvPlan hp = plan_hemv(true, (uint32_t)n, 0, 0);
    *panel = (int)hp.p.panel;
    *chunk = (int)hp.p.chunk;
}

// A reduce kernel's dispatch: partial results at 0, outputs at 1, its
// parameters at 2, a thread per output.
static void encode_reduce(id<MTLComputeCommandEncoder> enc, id<MTLComputePipelineState> pso,
                          id<MTLBuffer> part, id<MTLBuffer> out, const void *params, size_t len,
                          uint32_t nout) {
    [enc setComputePipelineState:pso];
    [enc setBuffer:part offset:0 atIndex:0];
    [enc setBuffer:out offset:0 atIndex:1];
    [enc setBytes:params length:len atIndex:2];
    [enc dispatchThreads:MTLSizeMake(nout, 1, 1) threadsPerThreadgroup:MTLSizeMake(64, 1, 1)];
}

// Runs c on the GPU with A at byte a_byte of abuf, leading dimension lda in
// elements, x scaled by 2^-sx (x_scale).
static ab_b2_result_t run_gpu(const ab_b2_call_t *c, id<MTLCommandQueue> queue, id<MTLBuffer> abuf,
                              size_t a_byte, int lda, int sx, uint64_t *stage_ns, uint64_t *mark) {
    if (a_byte % sizeof(double)) return AB_B2_UNSUPPORTED;
    id<MTLDevice> dev = queue.device;
    const Pipelines *pl = pipelines_for(dev);
    if (!pl) return AB_B2_FAILED;
    ab_prof_stage_mark(stage_ns, mark, AB_PROF_STAGE_INIT);

    const bool cplx = c->routine != AB_B2_DGEMV, hemv = c->routine == AB_B2_ZHEMV;
    const bool rows_out = hemv || c->op == 'N';              // outputs indexed by rows of A
    const uint32_t m = (uint32_t)c->m, n = (uint32_t)c->n;
    const int w = cplx ? 2 : 1;
    const uint32_t nx = rows_out ? n : m, nout = rows_out ? m : n;
    const size_t acc_size = cplx ? sizeof(AccZHost) : sizeof(AccDHost);

    Pass pass = {};
    HemvPlan hplan = {};
    uint64_t parts;   // partial results
    if (hemv) {
        hplan = plan_hemv(c->op == 'U', n, a_byte / sizeof(double), (uint64_t)lda * w);
        parts = hplan.parts;
    } else {
        KernelId k = !cplx ? (c->op == 'N' ? K_DGEMV_N : K_DGEMV_T)
                           : (c->op == 'N' ? K_ZGEMV_N : c->op == 'T' ? K_ZGEMV_T : K_ZGEMV_C);
        pass = plan_pass(pl, k, m, n, a_byte / sizeof(double), (uint64_t)lda * w, acc_size);
        parts = pass.groups > 1 ? (uint64_t)pass.groups * nout : 0;
    }

    Scratch *s = thread_scratch();
    ScratchTrim trim = {s};
    id<MTLBuffer> bx = scratch_buffer(s->x, dev, (size_t)nx * w * sizeof(DDf));
    id<MTLBuffer> bxe = scratch_buffer(s->xe, dev, (size_t)nx * sizeof(int32_t));
    id<MTLBuffer> bout = scratch_buffer(s->out, dev, (size_t)nout * acc_size);
    id<MTLBuffer> bpart = parts ? scratch_buffer(s->part, dev, (size_t)parts * acc_size) : bout;
    if (!bx || !bxe || !bout || !bpart) return AB_B2_FAILED;
    ab_prof_stage_mark(stage_ns, mark, AB_PROF_STAGE_ALLOC);

    x_stage(c, (int)nx, w, sx, (DDf *)bx.contents, (int32_t *)bxe.contents);
    ab_prof_stage_mark(stage_ns, mark, AB_PROF_STAGE_PACK);

    id<MTLCommandBuffer> cb = [queue commandBuffer];
    id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
    if (!cb || !enc) return AB_B2_FAILED;
    [enc setBuffer:abuf offset:0 atIndex:0];
    [enc setBuffer:bx offset:0 atIndex:1];
    [enc setBuffer:bxe offset:0 atIndex:2];
    [enc setBuffer:bpart offset:0 atIndex:3];
    if (hemv) {
        [enc setComputePipelineState:pl->pso[hplan.kernel]];
        [enc setBytes:&hplan.p length:sizeof hplan.p atIndex:4];
        [enc setThreadgroupMemoryLength:kHemvSimdgroups * sizeof(HemvTgHost) atIndex:0];
        [enc dispatchThreadgroups:hplan.grid threadsPerThreadgroup:hplan.group];
        encode_reduce(enc, pl->pso[hplan.reduce], bpart, bout, &hplan.p, sizeof hplan.p, nout);
    } else {
        [enc setComputePipelineState:pl->pso[pass.kernel]];
        [enc setBytes:&pass.p length:sizeof pass.p atIndex:4];
        if (pass.tg_mem) [enc setThreadgroupMemoryLength:pass.tg_mem atIndex:0];
        [enc dispatchThreadgroups:pass.grid threadsPerThreadgroup:pass.group];
        if (pass.groups > 1) {
            uint32_t np[2] = {nout, pass.groups};
            encode_reduce(enc, pl->pso[cplx ? K_REDUCE_Z : K_REDUCE_D], bpart, bout, np, sizeof np, nout);
        }
    }
    [enc endEncoding];
    [cb commit];
    [cb waitUntilCompleted];
    if (cb.status == MTLCommandBufferStatusError || ab_test_fail_stage() == AB_PROF_STAGE_KERNEL)
        return AB_B2_FAILED;
    if (stage_ns) {
        // The kernel stage is the GPU time; the rest of the step stays in "other".
        uint64_t now = ab_prof_now_ticks();
        double gpu = cb.GPUEndTime - cb.GPUStartTime;
        stage_ns[AB_PROF_STAGE_KERNEL] += ab_prof_kernel_ns(ab_prof_ticks_to_ns(now - *mark), gpu > 0 ? gpu : 0);
        *mark = now;
    }

    // Every output must pass the guard before y is touched.
    const int L = ceil_log2((uint64_t)nx * w);
    int ok = 1;
    if (cplx) {
        const AccZHost *o = (const AccZHost *)bout.contents;
        for (uint32_t i = 0; i < nout; i++)
            ok &= dd_finite(o[i].re) & dd_finite(o[i].im) & output_ok(o[i].pmax, o[i].lmax, L);
    } else {
        const AccDHost *o = (const AccDHost *)bout.contents;
        for (uint32_t i = 0; i < nout; i++) ok &= dd_finite(o[i].s) & output_ok(o[i].pmax, o[i].lmax, L);
    }
    if (!ok) return AB_B2_RANGE;
    ab_prof_stage_mark(stage_ns, mark, AB_PROF_STAGE_DOWNLOAD);

    // y := alpha t + beta y, t = 2^sx (hi + lo); y is not read for beta == 0.
    const double ar = c->alpha[0], ai = c->alpha[1], br = c->beta[0], bi = c->beta[1];
    const bool beta0 = br == 0.0 && bi == 0.0;
    const double up = sx <= 1023 ? ldexp(1.0, sx) : 0.0;
    if (cplx) {
        const AccZHost *o = (const AccZHost *)bout.contents;
        for (uint32_t i = 0; i < nout; i++) {
            double tr = dd_value(o[i].re, sx, up), ti = dd_value(o[i].im, sx, up);
            double *yi = c->y + 2 * vec_index((int)nout, c->incy, (int)i);
            double nr = ar * tr - ai * ti, ni = ar * ti + ai * tr;
            if (!beta0) {
                nr += br * yi[0] - bi * yi[1];
                ni += br * yi[1] + bi * yi[0];
            }
            yi[0] = nr;
            yi[1] = ni;
        }
    } else {
        const AccDHost *o = (const AccDHost *)bout.contents;
        for (uint32_t i = 0; i < nout; i++) {
            double t = dd_value(o[i].s, sx, up);
            double *yi = c->y + vec_index((int)nout, c->incy, (int)i);
            *yi = beta0 ? ar * t : ar * t + br * *yi;
        }
    }
    ab_prof_stage_mark(stage_ns, mark, AB_PROF_STAGE_EPILOGUE);
    return AB_B2_GPU;
}

// The library queue, creating the context on first use; nil without one
// (no device, or a fork()ed child).
static id<MTLCommandQueue> library_queue(void) {
    return ab_init() == AB_OK ? ab_internal_command_queue() : nil;
}

// The length of x: the columns of A for 'N' (and HEMV), its rows otherwise.
static int x_length(const ab_b2_call_t *c) {
    return c->routine == AB_B2_ZHEMV || c->op == 'N' ? c->n : c->m;
}

ab_b2_result_t ab_internal_b2_host(const ab_b2_call_t *c, const void *A, int lda,
                                   uint64_t stage_ns[AB_PROF_STAGE_COUNT]) {
    @autoreleasepool {
        uint64_t mark = stage_ns ? ab_prof_now_ticks() : 0;
        id<MTLCommandQueue> queue = library_queue();
        if (!queue) return AB_B2_NO_DEVICE;
        ab_prof_stage_mark(stage_ns, &mark, AB_PROF_STAGE_INIT);
        int w = c->routine == AB_B2_DGEMV ? 1 : 2, sx;
        if (!x_scale(c, x_length(c), w, &sx)) return AB_B2_RANGE;   // before mapping A
        ab_prof_stage_mark(stage_ns, &mark, AB_PROF_STAGE_PACK);

        // A's span, wrapped for this call only.
        size_t span = ((size_t)lda * (size_t)(c->n - 1) + (size_t)c->m) * ab_b2_elem_size(c->routine);
        if (span > queue.device.maxBufferLength) return AB_B2_TOO_LARGE;
        size_t offset = 0;
        id<MTLBuffer> abuf = ab_internal_wrap_host(queue.device, A, span, &offset);
        if (!abuf) return AB_B2_UNSUPPORTED;
        ab_prof_stage_mark(stage_ns, &mark, AB_PROF_STAGE_UPLOAD);
        return run_gpu(c, queue, abuf, offset, lda, sx, stage_ns, &mark);
    }
}

ab_b2_result_t ab_internal_b2_dev(const ab_b2_call_t *c, ab_dev_buffer_t A, size_t a_offset,
                                  int lda) {
    @autoreleasepool {
        id<MTLCommandQueue> queue = library_queue();
        if (!queue) return AB_B2_NO_DEVICE;
        int sx;
        if (!x_scale(c, x_length(c), c->routine == AB_B2_DGEMV ? 1 : 2, &sx)) return AB_B2_RANGE;
        size_t a_byte = A->base + a_offset * ab_b2_elem_size(c->routine);
        return run_gpu(c, queue, A->mtl, a_byte, lda, sx, NULL, NULL);
    }
}
