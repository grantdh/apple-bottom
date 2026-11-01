/*
 * fft_device.h -- arithmetic of the double-float FFT, compiled both as Metal
 * (the kernels, src/fft/fft_kernel.metal) and as C++ (the CPU emulator in
 * tests/fft_emu.cpp, which must reproduce the GPU's bits).
 *
 * A value is an unevaluated sum hi + lo of two floats (|lo| <= ulp(hi)/2),
 * about 48 significant bits. Every operation is built from error-free
 * transformations (TwoSum, FastTwoSum, FMA-based TwoProd), so the Metal
 * library MUST be compiled without fast math (MTLMathModeSafe), and the
 * emulator with -ffp-contract=off.
 *
 * Unlike apple-bottom's BLAS, additions use SloppyDWPlusDW: its error is
 * bounded in absolute terms, 2u^2 (|a| + |b|), not relative to a + b under
 * cancellation. An FFT's error bound is normwise anyway, and this bound gives
 * one of the same order; measured, the normwise error rises ~7% against
 * AccurateDWPlusDW (JMP 2017, Alg. 6) for a 20% shorter 3D transform.
 *
 *   1. platform layer
 *   2. double-float add and multiply
 *   3. FP64 <-> double-float on raw FP64 bit patterns (the GPU has no FP64)
 *   4. complex arithmetic
 *   5. radix 2, 3, 4, 5, 8 and odd-prime DFT butterflies
 *   6. the Stockham butterfly (load + twiddle + DFT, store)
 *   7. (Metal only) one Stockham pass of a line held in threadgroup memory
 *
 * Adapted from the QE band-FFT library (libqemetal: qm_device.h), whose
 * GPU results were bit-identical to this arithmetic on the CPU.
 *
 * Twiddle tables hold exp(+2 pi i t/n), t = 0..n-1; sign S = -1 applies the
 * conjugate (forward transform), S = +1 the table value (backward).
 */
#ifndef ABF_DEVICE_H
#define ABF_DEVICE_H

/* ========================================================================== */
/* 1. Platform layer                                                          */
/* ========================================================================== */
#if defined(__METAL_VERSION__)
#include <metal_stdlib>
using namespace metal;
#define ABF_TG  threadgroup
#define ABF_CST constant
#define ABF_THR thread
#define ABF_INL inline
typedef uint  abf_u32;
typedef ulong abf_u64;
typedef long  abf_i64;
ABF_INL float   abf_u2f(abf_u32 u) { return as_type<float>(u); }
ABF_INL abf_u32 abf_f2u(float f) { return as_type<uint>(f); }
ABF_INL float   abf_fma(float a, float b, float c) { return fma(a, b, c); }
ABF_INL float   abf_ldexp(float x, int e) { return ldexp(x, e); }
ABF_INL float   abf_rint(float x) { return rint(x); }
#else
#include <stdint.h>
#include <string.h>
#include <math.h>
#define ABF_TG
#define ABF_CST
#define ABF_THR
#define ABF_INL static inline
typedef uint32_t abf_u32;
typedef uint64_t abf_u64;
typedef int64_t  abf_i64;
ABF_INL float   abf_u2f(abf_u32 u) { float f; memcpy(&f, &u, 4); return f; }
ABF_INL abf_u32 abf_f2u(float f) { abf_u32 u; memcpy(&u, &f, 4); return u; }
ABF_INL float   abf_fma(float a, float b, float c) { return fmaf(a, b, c); }
ABF_INL float   abf_ldexp(float x, int e) { return ldexpf(x, e); }
ABF_INL float   abf_rint(float x) { return rintf(x); }
#endif

/* ========================================================================== */
/* 2. Double-float primitives                                                 */
/* ========================================================================== */
/* Butterfly constants c as hi = RN(c), lo = RN(c - hi), from 60-digit
 * decimal values; |hi + lo - c| / c <= 7.1e-16.                             */
#define ABF_SQRT1_2_HI 7.071067691e-01f
#define ABF_SQRT1_2_LO 1.210161749e-08f
#define ABF_SQRT3_2_HI 8.660253882e-01f
#define ABF_SQRT3_2_LO 1.554362505e-08f
#define ABF_C5_1_HI 3.090170026e-01f   /* cos(2 pi/5) */
#define ABF_C5_1_LO -8.207602598e-09f
#define ABF_C5_2_HI -8.090170026e-01f  /* cos(4 pi/5) */
#define ABF_C5_2_LO 8.207602598e-09f
#define ABF_S5_1_HI 9.510565400e-01f   /* sin(2 pi/5) */
#define ABF_S5_1_LO -2.371720598e-08f
#define ABF_S5_2_HI 5.877852440e-01f   /* sin(4 pi/5) */
#define ABF_S5_2_LO 8.304436250e-09f

struct abf_dd { float hi; float lo; };

ABF_INL abf_dd abf_mkdd(float h, float l) { abf_dd r; r.hi = h; r.lo = l; return r; }
ABF_INL abf_dd abf_ddzero(void) { return abf_mkdd(0.0f, 0.0f); }

/* Knuth TwoSum: s + e == a + b exactly. */
ABF_INL abf_dd abf_two_sum(float a, float b)
{
    const float s = a + b;
    const float v = s - a;
    const float e = (a - (s - v)) + (b - v);
    return abf_mkdd(s, e);
}

/* Dekker FastTwoSum: exact when exponent(a) >= exponent(b). */
ABF_INL abf_dd abf_fast_two_sum(float a, float b)
{
    const float s = a + b;
    const float e = b - (s - a);
    return abf_mkdd(s, e);
}

/* FMA-based TwoProd: p + e == a * b exactly. */
ABF_INL abf_dd abf_two_prod(float a, float b)
{
    const float p = a * b;
    const float e = abf_fma(a, b, -p);
    return abf_mkdd(p, e);
}

ABF_INL abf_dd abf_dd_neg(abf_dd a) { return abf_mkdd(-a.hi, -a.lo); }

/* SloppyDWPlusDW (Joldes-Muller-Popescu 2017, Alg. 4): |error| <= 2u^2 (|a|+|b|),
 * u = 2^-24 (see the top of this file). */
ABF_INL abf_dd abf_dd_add(abf_dd a, abf_dd b)
{
    const abf_dd s = abf_two_sum(a.hi, b.hi);
    return abf_fast_two_sum(s.hi, s.lo + (a.lo + b.lo));
}

ABF_INL abf_dd abf_dd_sub(abf_dd a, abf_dd b) { return abf_dd_add(a, abf_dd_neg(b)); }

/* DWTimesDW2 (JMP 2017, Alg. 11; < 5u^2), apple-bottom's default product. */
ABF_INL abf_dd abf_dd_mul(abf_dd a, abf_dd b)
{
    const abf_dd p = abf_two_prod(a.hi, b.hi);
    const float e = abf_fma(a.hi, b.lo, abf_fma(a.lo, b.hi, p.lo));
    return abf_fast_two_sum(p.hi, e);
}

/* Exact scaling by 1/2. */
ABF_INL abf_dd abf_dd_half(abf_dd a) { return abf_mkdd(0.5f * a.hi, 0.5f * a.lo); }

/* ========================================================================== */
/* 3. FP64 <-> double-float on raw bit patterns                               */
/* ========================================================================== */
#define ABF_F64_MANT 0x000FFFFFFFFFFFFFul

/* FP64 bits -> double-float, bit-identical to the host's hi = (float)x,
 * lo = (float)(x - (double)hi) when lo is a normal float or zero. Values
 * below 2^-126 (and FP64 subnormals) become a signed zero, overflow
 * becomes Inf; Inf/NaN propagate in hi. All in 32-bit integer operations:
 * hi takes the top 24 of the 53 significand bits (rounded to nearest even),
 * lo is the remainder, |rem| <= 2^28 units of 2^(E-52).                    */
ABF_INL abf_dd abf_f64bits_to_dd(abf_u64 b)
{
    const abf_u32 hw = (abf_u32)(b >> 32), lw = (abf_u32)b;
    const abf_u32 sgn = hw >> 31;
    const int e = (int)((hw >> 20) & 0x7FFu);
    if (e == 0x7FF)
        return abf_mkdd(abf_u2f((sgn << 31) | 0x7F800000u | (((hw & 0xFFFFFu) | lw) ? 0x400000u : 0u)), 0.0f);
    const float zero = abf_u2f(sgn << 31);
    if (e == 0)
        return abf_mkdd(zero, 0.0f);
    abf_u32 q = ((0x100000u | (hw & 0xFFFFFu)) << 3) | (lw >> 29);   /* top 24 bits */
    const abf_u32 r = lw & 0x1FFFFFFFu;                              /* low 29 bits */
    const abf_u32 up = (r > 0x10000000u || (r == 0x10000000u && (q & 1u))) ? 1u : 0u;
    q += up;
    const int rem = (int)r - (int)(up << 29);                         /* x - hi */
    const int E0 = e - 1023;
    int E = E0;
    if (q == (1u << 24)) { q >>= 1; E += 1; }
    if (E > 127)
        return abf_mkdd(abf_u2f((sgn << 31) | 0x7F800000u), 0.0f);
    if (E < -126)
        return abf_mkdd(zero, 0.0f);
    const float hi = abf_u2f((sgn << 31) | ((abf_u32)(E + 127) << 23) | (q & 0x7FFFFFu));
    const float l = abf_ldexp((float)rem, E0 - 52);
    return abf_mkdd(hi, sgn ? -l : l);
}

/* FP32 -> FP64 bit pattern, exact for normal values; a subnormal becomes a
 * signed zero, as the GPU flushes it. */
ABF_INL abf_u64 abf_f32_to_f64bits(float f)
{
    const abf_u32 u = abf_f2u(f);
    const abf_u64 s = (abf_u64)(u >> 31) << 63;
    const int e = (int)((u >> 23) & 0xFFu);
    const abf_u32 m = u & 0x7FFFFFu;
    if (e == 0xFF)
        return s | 0x7FF0000000000000ul | ((abf_u64)m << 29);
    if (e == 0)
        return s;
    return s | ((abf_u64)(e - 127 + 1023) << 52) | ((abf_u64)m << 29);
}

/* Double-float -> FP64 bits: RN(hi + lo), the correctly rounded value. Every
 * operation here returns a normalized pair, |lo| <= ulp(hi)/2, and hi is
 * exact in FP64 with its low 29 significand bits zero, so the sum is hi's bit
 * pattern plus lo counted in FP64 ulps of hi, rounded to nearest even:
 * |k| <= 2^28 never carries out of the significand. When lo shrinks a power
 * of two the result lies in the binade below, whose ulp is half as large; the
 * borrow from the exponent field then gives exactly that. A non-finite hi
 * (overflow) is returned as is, without the NaN its lo holds.              */
ABF_INL abf_u64 abf_dd_to_f64bits(abf_dd v)
{
    const abf_u64 h = abf_f32_to_f64bits(v.hi);
    if ((h & 0x7FF0000000000000ul) == 0x7FF0000000000000ul || v.lo == 0.0f) return h;
    const abf_u32 opp = (abf_f2u(v.hi) ^ abf_f2u(v.lo)) >> 31;           /* lo shrinks |hi| */
    const int down = opp && (h & ABF_F64_MANT) == 0ul;
    const int e = (int)((h >> 52) & 0x7FFul) - 1023;
    const abf_i64 k = (int)abf_rint(abf_ldexp(v.lo, 52 - e + down));  /* exact scaling, RNE; |k| <= 2^29 */
    return h + (abf_u64)(v.hi < 0.0f ? -k : k);
}

/* ========================================================================== */
/* 4. Complex double-float arithmetic                                         */
/* ========================================================================== */
struct abf_cplx { abf_dd re; abf_dd im; };

ABF_INL abf_cplx abf_mkc(abf_dd r, abf_dd i) { abf_cplx c; c.re = r; c.im = i; return c; }
ABF_INL abf_cplx abf_czero(void) { return abf_mkc(abf_ddzero(), abf_ddzero()); }
ABF_INL abf_cplx abf_cadd(abf_cplx a, abf_cplx b) { return abf_mkc(abf_dd_add(a.re, b.re), abf_dd_add(a.im, b.im)); }
ABF_INL abf_cplx abf_csub(abf_cplx a, abf_cplx b) { return abf_mkc(abf_dd_sub(a.re, b.re), abf_dd_sub(a.im, b.im)); }
ABF_INL abf_cplx abf_chalf(abf_cplx a) { return abf_mkc(abf_dd_half(a.re), abf_dd_half(a.im)); }
ABF_INL abf_cplx abf_cmul_r(abf_cplx a, abf_dd s) { return abf_mkc(abf_dd_mul(a.re, s), abf_dd_mul(a.im, s)); }

/* a * w (S > 0) or a * conj(w) (S < 0). */
template <int S>
ABF_INL abf_cplx abf_ctw(abf_cplx a, abf_cplx w)
{
    if (S > 0)
        return abf_mkc(abf_dd_sub(abf_dd_mul(a.re, w.re), abf_dd_mul(a.im, w.im)),
                       abf_dd_add(abf_dd_mul(a.re, w.im), abf_dd_mul(a.im, w.re)));
    else
        return abf_mkc(abf_dd_add(abf_dd_mul(a.re, w.re), abf_dd_mul(a.im, w.im)),
                       abf_dd_sub(abf_dd_mul(a.im, w.re), abf_dd_mul(a.re, w.im)));
}

/* Multiply by i*S (exact). */
template <int S>
ABF_INL abf_cplx abf_cmul_i(abf_cplx a)
{
    if (S > 0) return abf_mkc(abf_dd_neg(a.im), a.re);
    else       return abf_mkc(a.im, abf_dd_neg(a.re));
}

/* ========================================================================== */
/* 5. DFT butterflies, W_R = exp(S * 2 pi i / R)                              */
/* ========================================================================== */
template <int S>
ABF_INL void abf_dft2(ABF_THR abf_cplx* v)
{
    const abf_cplx a = v[0], b = v[1];
    v[0] = abf_cadd(a, b);
    v[1] = abf_csub(a, b);
}

template <int S>
ABF_INL void abf_dft3(ABF_THR abf_cplx* v)
{
    const abf_cplx t1 = abf_cadd(v[1], v[2]);
    const abf_cplx t2 = abf_csub(v[0], abf_chalf(t1));
    const abf_cplx t3 = abf_cmul_i<S>(abf_cmul_r(abf_csub(v[1], v[2]), abf_mkdd(ABF_SQRT3_2_HI, ABF_SQRT3_2_LO)));
    v[0] = abf_cadd(v[0], t1);
    v[1] = abf_cadd(t2, t3);
    v[2] = abf_csub(t2, t3);
}

template <int S>
ABF_INL void abf_dft4_vals(ABF_THR abf_cplx& x0, ABF_THR abf_cplx& x1, ABF_THR abf_cplx& x2, ABF_THR abf_cplx& x3)
{
    const abf_cplx t0 = abf_cadd(x0, x2), t1 = abf_csub(x0, x2);
    const abf_cplx t2 = abf_cadd(x1, x3), t3 = abf_cmul_i<S>(abf_csub(x1, x3));
    x0 = abf_cadd(t0, t2);
    x2 = abf_csub(t0, t2);
    x1 = abf_cadd(t1, t3);
    x3 = abf_csub(t1, t3);
}

template <int S>
ABF_INL void abf_dft4(ABF_THR abf_cplx* v) { abf_dft4_vals<S>(v[0], v[1], v[2], v[3]); }

template <int S>
ABF_INL void abf_dft5(ABF_THR abf_cplx* v)
{
    const abf_dd c1 = abf_mkdd(ABF_C5_1_HI, ABF_C5_1_LO), c2 = abf_mkdd(ABF_C5_2_HI, ABF_C5_2_LO);
    const abf_dd s1 = abf_mkdd(ABF_S5_1_HI, ABF_S5_1_LO), s2 = abf_mkdd(ABF_S5_2_HI, ABF_S5_2_LO);
    const abf_cplx b1 = abf_cadd(v[1], v[4]), b2 = abf_cadd(v[2], v[3]);
    const abf_cplx d1 = abf_csub(v[1], v[4]), d2 = abf_csub(v[2], v[3]);
    const abf_cplx t1 = abf_cadd(v[0], abf_cadd(abf_cmul_r(b1, c1), abf_cmul_r(b2, c2)));
    const abf_cplx t2 = abf_cadd(v[0], abf_cadd(abf_cmul_r(b1, c2), abf_cmul_r(b2, c1)));
    const abf_cplx u1 = abf_cmul_i<S>(abf_cadd(abf_cmul_r(d1, s1), abf_cmul_r(d2, s2)));
    const abf_cplx u2 = abf_cmul_i<S>(abf_csub(abf_cmul_r(d1, s2), abf_cmul_r(d2, s1)));
    v[0] = abf_cadd(v[0], abf_cadd(b1, b2));
    v[1] = abf_cadd(t1, u1);
    v[4] = abf_csub(t1, u1);
    v[2] = abf_cadd(t2, u2);
    v[3] = abf_csub(t2, u2);
}

/* z * W8 with W8 = (1 + i S)/sqrt(2). */
template <int S>
ABF_INL abf_cplx abf_mul_w8(abf_cplx z)
{
    const abf_dd h = abf_mkdd(ABF_SQRT1_2_HI, ABF_SQRT1_2_LO);
    if (S > 0) return abf_mkc(abf_dd_mul(abf_dd_sub(z.re, z.im), h), abf_dd_mul(abf_dd_add(z.im, z.re), h));
    else       return abf_mkc(abf_dd_mul(abf_dd_add(z.re, z.im), h), abf_dd_mul(abf_dd_sub(z.im, z.re), h));
}

template <int S>
ABF_INL void abf_dft8(ABF_THR abf_cplx* v)
{
    abf_cplx e0 = v[0], e1 = v[2], e2 = v[4], e3 = v[6];
    abf_cplx o0 = v[1], o1 = v[3], o2 = v[5], o3 = v[7];
    abf_dft4_vals<S>(e0, e1, e2, e3);
    abf_dft4_vals<S>(o0, o1, o2, o3);
    o1 = abf_mul_w8<S>(o1);                     /* W8^1           */
    o2 = abf_cmul_i<S>(o2);                     /* W8^2 = i S     */
    o3 = abf_cmul_i<S>(abf_mul_w8<S>(o3));      /* W8^3 = W8 * iS */
    v[0] = abf_cadd(e0, o0); v[4] = abf_csub(e0, o0);
    v[1] = abf_cadd(e1, o1); v[5] = abf_csub(e1, o1);
    v[2] = abf_cadd(e2, o2); v[6] = abf_csub(e2, o2);
    v[3] = abf_cadd(e3, o3); v[7] = abf_csub(e3, o3);
}

/* Odd prime R (7, 11, 13): out[k] = sum_r v[r] W_R^(S r k), pairing r with
 * R - r. cos/sin(2 pi m/R) come from the line's twiddle table at m * step,
 * step = n/R.                                                               */
template <int S, abf_u32 R>
ABF_INL void abf_dft_odd(ABF_THR abf_cplx* v, ABF_CST const abf_cplx* tw, abf_u32 step)
{
    constexpr abf_u32 H = (R - 1) / 2;
    abf_cplx b[(R + 1) / 2], d[(R + 1) / 2], o[R];
    abf_cplx sum = v[0];
    for (abf_u32 r = 1; r <= H; ++r) {
        b[r - 1] = abf_cadd(v[r], v[R - r]);
        d[r - 1] = abf_csub(v[r], v[R - r]);
        sum = abf_cadd(sum, b[r - 1]);
    }
    o[0] = sum;
    for (abf_u32 k = 1; k <= H; ++k) {
        abf_cplx t = v[0], u = abf_czero();
        for (abf_u32 r = 1; r <= H; ++r) {
            const abf_cplx w = tw[((r * k) % R) * step];
            t = abf_cadd(t, abf_cmul_r(b[r - 1], w.re));
            u = abf_cadd(u, abf_cmul_r(d[r - 1], w.im));
        }
        u = abf_cmul_i<S>(u);
        o[k] = abf_cadd(t, u);
        o[R - k] = abf_csub(t, u);
    }
    for (abf_u32 r = 0; r < R; ++r) v[r] = o[r];
}

template <int S, abf_u32 R>
ABF_INL void abf_dft(ABF_THR abf_cplx* v, ABF_CST const abf_cplx* tw, abf_u32 n)
{
    if (R == 2) abf_dft2<S>(v);
    else if (R == 3) abf_dft3<S>(v);
    else if (R == 4) abf_dft4<S>(v);
    else if (R == 5) abf_dft5<S>(v);
    else if (R == 8) abf_dft8<S>(v);
    else abf_dft_odd<S, R>(v, tw, n / R);
}

/* ========================================================================== */
/* 6. Mixed-radix Stockham butterfly                                          */
/* ========================================================================== */
/* Pass with radix R after earlier passes whose radix product is ns;
 * butterfly j in [0, n/R):
 *   load    v[r] = a[j + r n/R]
 *   twiddle v[r] *= W_n^(S (j mod ns) r n/(ns R))
 *   DFT_R
 *   store   out[(j/ns) ns R + j mod ns + r ns] = v[r]
 * After the last pass the line is in natural order.                        */

/* Twiddles and DFT of butterfly j once its R points are in v. */
template <int S, abf_u32 R>
ABF_INL void abf_bfly(ABF_THR abf_cplx* v, abf_u32 n, abf_u32 ns, abf_u32 j, ABF_CST const abf_cplx* tw)
{
    const abf_u32 jm = j % ns;
    if (jm != 0u) {
        const abf_u32 step = jm * (n / (ns * R));
        for (abf_u32 r = 1; r < R; ++r) v[r] = abf_ctw<S>(v[r], tw[r * step]);
    }
    abf_dft<S, R>(v, tw, n);
}

template <int S, abf_u32 R>
ABF_INL void abf_bfly_load(ABF_TG const abf_cplx* a, abf_u32 n, abf_u32 ns, abf_u32 j,
                           ABF_CST const abf_cplx* tw, ABF_THR abf_cplx* v)
{
    const abf_u32 stride = n / R;
    for (abf_u32 r = 0; r < R; ++r) v[r] = a[j + r * stride];
    abf_bfly<S, R>(v, n, ns, j, tw);
}

template <abf_u32 R>
ABF_INL void abf_bfly_store(ABF_TG abf_cplx* a, abf_u32 ns, abf_u32 j, ABF_THR const abf_cplx* v)
{
    const abf_u32 base = (j / ns) * ns * R + (j % ns);
    for (abf_u32 r = 0; r < R; ++r) a[base + r * ns] = v[r];
}

/* ========================================================================== */
/* 7. One Stockham pass over a line of N points in threadgroup memory: T      */
/*    lanes, each ROUNDS butterflies kept in registers across the barrier,    */
/*    then stored in place. Every lane executes both barriers.                */
/* ========================================================================== */
#if defined(__METAL_VERSION__)
template <int S, abf_u32 N, abf_u32 T, abf_u32 R>
inline void abf_fft_pass(threadgroup abf_cplx* a, abf_u32 lt, abf_u32 ns, constant abf_cplx* tw)
{
    constexpr abf_u32 NB = N / R;
    constexpr abf_u32 ROUNDS = (NB + T - 1) / T;
    abf_cplx v[ROUNDS * R];
    for (abf_u32 q = 0; q < ROUNDS; ++q) {
        const abf_u32 j = lt + q * T;
        if (j < NB) abf_bfly_load<S, R>(a, N, ns, j, tw, v + q * R);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (abf_u32 q = 0; q < ROUNDS; ++q) {
        const abf_u32 j = lt + q * T;
        if (j < NB) abf_bfly_store<R>(a, ns, j, v + q * R);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
}

ABF_INL abf_cplx abf_from4(float4 v) { return abf_mkc(abf_mkdd(v.x, v.y), abf_mkdd(v.z, v.w)); }
ABF_INL float4 abf_to4(abf_cplx c) { return float4(c.re.hi, c.re.lo, c.im.hi, c.im.lo); }

/* Element k of the caller's FP64 array or of the double-float work buffer. */
ABF_INL abf_cplx abf_ld(device const ulong* p, uint k)
{
    return abf_mkc(abf_f64bits_to_dd(p[2 * k]), abf_f64bits_to_dd(p[2 * k + 1]));
}
ABF_INL abf_cplx abf_ld(device const float4* p, uint k) { return abf_from4(p[k]); }
ABF_INL void abf_st(device ulong* p, uint k, abf_cplx c)
{
    p[2 * k] = abf_dd_to_f64bits(c.re);
    p[2 * k + 1] = abf_dd_to_f64bits(c.im);
}
ABF_INL void abf_st(device float4* p, uint k, abf_cplx c) { p[k] = abf_to4(c); }
#endif

#endif /* ABF_DEVICE_H */
