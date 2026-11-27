/*
 * fft_host.h -- host-side rules of the double-float FFT, shared by the
 * library (src/fft.m), the CPU emulator and the high-precision reference in
 * tests/: radix plans, the four-step split of long 1D transforms, and
 * twiddles and scale factors computed in double-double (~106 bits) and then
 * rounded to double-float pairs. Header-only C99, also valid C++.
 */
#ifndef ABF_HOST_H
#define ABF_HOST_H

#include <math.h>
#include <stdlib.h>

/* The double-double helpers below are error-free transformations: keep the
 * compiler from contracting them into FMAs. */
#pragma STDC FP_CONTRACT OFF

#define ABF_MAX_PASSES 24
#define ABF_MAX_LINE   2048   /* longest line one threadgroup holds (32 KB)  */

/* Radix plan of a line length: greedy over 8, 7, 5, 4, 3, 2, 11, 13, e.g.
 * 72 -> 8,3,3; 96 -> 8,4,3; 120 -> 8,5,3; 128 -> 8,8,2. n == 1 has no pass.
 * Returns the number of passes, or -1 for n < 1 or a prime factor > 13.   */
static inline int abf_factor(int n, int radix[ABF_MAX_PASSES])
{
    static const int cand[8] = {8, 7, 5, 4, 3, 2, 11, 13};
    int np = 0;
    if (n < 1) return -1;
    while (n > 1) {
        int k, found = 0;
        for (k = 0; k < 8; ++k) {
            if (n % cand[k] == 0) {
                if (np >= ABF_MAX_PASSES) return -1;
                radix[np++] = cand[k];
                n /= cand[k];
                found = 1;
                break;
            }
        }
        if (!found) return -1;
    }
    return np;
}

static inline int abf_max_radix(const int* radix, int np)
{
    int i, m = 1;
    for (i = 0; i < np; ++i) if (radix[i] > m) m = radix[i];
    return m;
}

/* Four-step split of a rank-1 length n > ABF_MAX_LINE into n1 * n2, both
 * lines a threadgroup can hold, as balanced as possible. Returns 0, or -1. */
static inline int abf_four_step_split(long n, int* n1, int* n2)
{
    int r[ABF_MAX_PASSES], best = -1;
    long d;
    for (d = 2; d <= ABF_MAX_LINE; ++d) {
        if (n % d || n / d > ABF_MAX_LINE) continue;
        if (abf_factor((int)d, r) < 0 || abf_factor((int)(n / d), r) < 0) continue;
        if (best < 0 || labs(d - n / d) < labs((long)best - n / best)) best = (int)d;
    }
    if (best < 0) return -1;
    *n1 = best;
    *n2 = (int)(n / best);
    return 0;
}

/* How a dimension of length n is transformed: ABF_DIRECT, one pass of lines
 * of n points; ABF_FOUR_STEP (rank 1 only, n > ABF_MAX_LINE), two passes of
 * lines of *n1 and *n2 points; or -1 when it cannot be.                     */
enum { ABF_DIRECT = 0, ABF_FOUR_STEP = 1 };
static inline int abf_line_split(int rank, long n, int* n1, int* n2)
{
    int r[ABF_MAX_PASSES];
    if (n <= ABF_MAX_LINE) return abf_factor((int)n, r) < 0 ? -1 : ABF_DIRECT;
    return rank == 1 && abf_four_step_split(n, n1, n2) == 0 ? ABF_FOUR_STEP : -1;
}

/* -------------------------------------------------------------------------- */
/* Double-double (FP64 pairs) for exactly rounded double-float constants.     */
/* -------------------------------------------------------------------------- */
typedef struct { double hi, lo; } abf_hdd;

static inline abf_hdd abf_hdd_make(double hi, double lo)
{
    abf_hdd r;
    r.hi = hi;
    r.lo = lo;
    return r;
}

static inline abf_hdd abf_hdd_two_sum(double a, double b)
{
    abf_hdd r;
    double v;
    r.hi = a + b;
    v = r.hi - a;
    r.lo = (a - (r.hi - v)) + (b - v);
    return r;
}

static inline abf_hdd abf_hdd_fast(double a, double b)
{
    abf_hdd r;
    r.hi = a + b;
    r.lo = b - (r.hi - a);
    return r;
}

static inline abf_hdd abf_hdd_add(abf_hdd a, abf_hdd b)
{
    abf_hdd s = abf_hdd_two_sum(a.hi, b.hi), t = abf_hdd_two_sum(a.lo, b.lo), v;
    v = abf_hdd_fast(s.hi, s.lo + t.hi);
    return abf_hdd_fast(v.hi, t.lo + v.lo);
}

static inline abf_hdd abf_hdd_mul(abf_hdd a, abf_hdd b)
{
    double p = a.hi * b.hi;
    double e = fma(a.hi, b.hi, -p);
    e = fma(a.hi, b.lo, fma(a.lo, b.hi, e));
    return abf_hdd_fast(p, e);
}

static inline abf_hdd abf_hdd_div_d(abf_hdd a, double b)
{
    double q1 = a.hi / b, q2;
    abf_hdd r;
    /* remainder a - q1*b, exact through fma */
    r.hi = fma(-q1, b, a.hi);
    r.lo = a.lo;
    q2 = (r.hi + r.lo) / b;
    return abf_hdd_fast(q1, q2);
}

/* cos and sin of 2 pi k / n to ~1e-32, from the octant symmetries and a
 * Taylor series on [0, pi/4]; k = 0, n/4, n/2, 3n/4 come out exact.        */
static inline void abf_hdd_cis(long k, long n, abf_hdd* c, abf_hdd* s)
{
    const abf_hdd pi = abf_hdd_make(3.141592653589793116e+00, 1.224646799147353207e-16);
    long o, r;
    int i, odd;
    abf_hdd phi, phi2, term, cs, sn, t;
    k %= n;
    if (k < 0) k += n;
    o = (8 * k) / n;               /* octant 0..7 */
    r = 8 * k - o * n;              /* angle within it: (pi/4) r / n */
    odd = (int)(o & 1);
    if (odd) r = n - r;             /* odd octants measure from the far side */
    phi = abf_hdd_div_d(abf_hdd_mul(pi, abf_hdd_make((double)r, 0.0)), 4.0 * (double)n);
    phi2 = abf_hdd_mul(phi, phi);
    /* sin: phi - phi^3/3! + ... ; cos: 1 - phi^2/2! + ... */
    sn = phi; term = phi;
    for (i = 1; i <= 16; ++i) {
        term = abf_hdd_div_d(abf_hdd_mul(term, phi2), -(double)((2 * i) * (2 * i + 1)));
        sn = abf_hdd_add(sn, term);
    }
    cs = abf_hdd_make(1.0, 0.0); term = cs;
    for (i = 1; i <= 16; ++i) {
        term = abf_hdd_div_d(abf_hdd_mul(term, phi2), -(double)((2 * i - 1) * (2 * i)));
        cs = abf_hdd_add(cs, term);
    }
    if (odd) { t = cs; cs = sn; sn = t; }   /* angle = (o+1) pi/4 - phi: swap */
    switch ((int)(o >> 1)) {                 /* quarter turns */
    case 0: *c = cs; *s = sn; break;
    case 1: *c = abf_hdd_make(-sn.hi, -sn.lo); *s = cs; break;
    case 2: *c = abf_hdd_make(-cs.hi, -cs.lo); *s = abf_hdd_make(-sn.hi, -sn.lo); break;
    default: *c = sn; *s = abf_hdd_make(-cs.hi, -cs.lo); break;
    }
}

/* Double-double -> double-float pair: hi = RN(x), lo = RN(x - hi). */
static inline void abf_hdd_split(abf_hdd x, float* hi, float* lo)
{
    float h = (float)x.hi;
    *hi = h;
    *lo = (float)((x.hi - (double)h) + x.lo);
}

/* Twiddle table exp(+2 pi i t/n), t = 0..count-1, as {re.hi, re.lo, im.hi,
 * im.lo} per entry. */
static inline void abf_twiddles(long n, long count, float* out)
{
    long t;
    for (t = 0; t < count; ++t) {
        abf_hdd c, s;
        abf_hdd_cis(t, n, &c, &s);
        abf_hdd_split(c, &out[4 * t], &out[4 * t + 1]);
        abf_hdd_split(s, &out[4 * t + 2], &out[4 * t + 3]);
    }
}

/* Scale for a normalization (0: none, 1: 1/N, 2: 1/sqrt(N)) as a double-double. */
static inline abf_hdd abf_scale_hdd(int norm, double N)
{
    const abf_hdd one = abf_hdd_make(1.0, 0.0);
    if (norm == 1) return abf_hdd_div_d(one, N);
    if (norm == 2) {
        /* sqrt(N) = s0 + (N - s0^2) / (2 s0); N - s0^2 is exact through fma */
        double s0 = sqrt(N);
        abf_hdd s = abf_hdd_fast(s0, fma(-s0, s0, N) / (2.0 * s0));
        double q1 = 1.0 / s.hi;
        /* 1/s = q1 + (1 - q1 s) / s */
        abf_hdd r = abf_hdd_mul(abf_hdd_make(q1, 0.0), s);
        double rem = (1.0 - r.hi) - r.lo;
        return abf_hdd_fast(q1, rem / s.hi);
    }
    return one;
}

#endif /* ABF_HOST_H */
