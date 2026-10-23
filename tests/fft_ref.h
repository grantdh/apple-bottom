/*
 * fft_ref.h -- high-precision reference DFTs for the FFT tests and benchmark.
 *
 * Arithmetic is double-double (FP64 pairs, ~106 bits) with twiddles from
 * src/fft/fft_host.h (~1e-32), so the reference's own error, ~1e-31 log n,
 * is far below what it measures (~1e-15). Two forms:
 *   ref_dft_direct  O(n^2) sum, to validate the fast one on small sizes
 *   ref_fft_axis    mixed-radix decimation in time (O(p^2) butterfly per prime
 *                   factor p) applied along one axis of a packed column-major
 *                   batch of transforms, lines spread over threads
 * plus error measures of an FP64 result against a reference.
 */
#ifndef AB_FFT_REF_H
#define AB_FFT_REF_H

#include <dispatch/dispatch.h>
#include <math.h>
#include <stdlib.h>

#include "../src/fft/fft_host.h"

typedef struct { abf_hdd re, im; } ref_cplx;

static inline abf_hdd ref_neg(abf_hdd a) { return abf_hdd_make(-a.hi, -a.lo); }

static inline ref_cplx ref_add(ref_cplx a, ref_cplx b)
{
    ref_cplx r = {abf_hdd_add(a.re, b.re), abf_hdd_add(a.im, b.im)};
    return r;
}

static inline ref_cplx ref_mul(ref_cplx a, ref_cplx b)
{
    ref_cplx r = {abf_hdd_add(abf_hdd_mul(a.re, b.re), ref_neg(abf_hdd_mul(a.im, b.im))),
                  abf_hdd_add(abf_hdd_mul(a.re, b.im), abf_hdd_mul(a.im, b.re))};
    return r;
}

/* exp(sign 2 pi i t / n), t = 0..n-1 */
static inline ref_cplx* ref_twiddles(long n, int sign)
{
    ref_cplx* w = (ref_cplx*)malloc(sizeof(ref_cplx) * (size_t)n);
    for (long t = 0; t < n; ++t) {
        abf_hdd c, s;
        abf_hdd_cis(t, n, &c, &s);
        w[t].re = c;
        w[t].im = sign > 0 ? s : ref_neg(s);
    }
    return w;
}

static inline int ref_smallest_factor(long n)
{
    for (long p = 2; p * p <= n; ++p) if (n % p == 0) return (int)p;
    return (int)n;
}

/* X[k + q m] = sum_r W_n^(r (k + q m)) Y_r[k] for k in [k0, k1): the p
 * sub-transforms Y_r of length m lie one after another in out. */
static inline void ref_combine(ref_cplx* out, long n, long p, long m, const ref_cplx* tw, long step, long k0, long k1)
{
    ref_cplx sbuf[16];
    ref_cplx* s = p <= 16 ? sbuf : (ref_cplx*)malloc(sizeof(ref_cplx) * (size_t)p);
    for (long k = k0; k < k1; ++k) {
        for (long q = 0; q < p; ++q) {
            const long d = k + q * m;   /* < n: the exponent r d mod n steps by d */
            ref_cplx acc = out[k];
            long e = 0;
            for (long r = 1; r < p; ++r) {
                e += d;
                if (e >= n) e -= n;
                acc = ref_add(acc, ref_mul(out[r * m + k], tw[e * step]));
            }
            s[q] = acc;
        }
        for (long q = 0; q < p; ++q) out[k + q * m] = s[q];
    }
    if (s != sbuf) free(s);
}

/* out[k] = sum_j in[j*istride] W^(jk), W = tw[step] (tw has n*step entries).
 * Long combines run on every core, so one long line does too. */
static inline void ref_fft_rec(const ref_cplx* in, long istride, ref_cplx* out, long n, const ref_cplx* tw, long step)
{
    if (n == 1) { out[0] = in[0]; return; }
    const long p = ref_smallest_factor(n), m = n / p;
    for (long r = 0; r < p; ++r) ref_fft_rec(in + r * istride, istride * p, out + r * m, m, tw, step * p);
    if (m < 4096) { ref_combine(out, n, p, m, tw, step, 0, m); return; }
    const long per = (m + 63) / 64;
    dispatch_apply(64, dispatch_get_global_queue(QOS_CLASS_USER_INITIATED, 0), ^(size_t b) {
        const long k0 = (long)b * per, k1 = k0 + per < m ? k0 + per : m;
        if (k0 < k1) ref_combine(out, n, p, m, tw, step, k0, k1);
    });
}

/* In place along one axis of nb packed column-major n0 x n1 x n2 arrays. */
static inline void ref_fft_axis(ref_cplx* a, const long n[3], long nb, int axis, int sign)
{
    const long len = n[axis];
    if (len == 1) return;
    long stride = 1;
    for (int d = 0; d < axis; ++d) stride *= n[d];
    const long N = n[0] * n[1] * n[2], nlines = nb * (N / len);
    ref_cplx* tw = ref_twiddles(len, sign);
    const long nslices = nlines < 64 ? 1 : 64, per = (nlines + nslices - 1) / nslices;
    dispatch_apply((size_t)nslices, dispatch_get_global_queue(QOS_CLASS_USER_INITIATED, 0), ^(size_t s) {
        const long l0 = (long)s * per, l1 = l0 + per < nlines ? l0 + per : nlines;
        ref_cplx* line = (ref_cplx*)malloc(sizeof(ref_cplx) * (size_t)len);
        ref_cplx* res = (ref_cplx*)malloc(sizeof(ref_cplx) * (size_t)len);
        for (long l = l0; l < l1; ++l) {
            /* line l: batch b, then the indices other than axis, in order */
            const long b = l / (N / len), rest = l % (N / len);
            const long lo = rest % stride, hi = rest / stride;
            ref_cplx* base = a + b * N + lo + hi * stride * len;
            for (long i = 0; i < len; ++i) line[i] = base[i * stride];
            ref_fft_rec(line, 1, res, len, tw, 1);
            for (long i = 0; i < len; ++i) base[i * stride] = res[i];
        }
        free(line); free(res);
    });
    free(tw);
}

/* Reference transform of nb packed transforms of FP64 data, scaled. */
static inline ref_cplx* ref_transform(const double* in, const long n[3], long nb, int sign, int norm)
{
    const long N = n[0] * n[1] * n[2];
    ref_cplx* a = (ref_cplx*)malloc(sizeof(ref_cplx) * (size_t)(N * nb));
    for (long i = 0; i < N * nb; ++i) {
        a[i].re = abf_hdd_make(in[2 * i], 0.0);
        a[i].im = abf_hdd_make(in[2 * i + 1], 0.0);
    }
    for (int d = 0; d < 3; ++d) ref_fft_axis(a, n, nb, d, sign);
    if (norm) {
        const abf_hdd s = abf_scale_hdd(norm, (double)N);
        for (long i = 0; i < N * nb; ++i) {
            a[i].re = abf_hdd_mul(a[i].re, s);
            a[i].im = abf_hdd_mul(a[i].im, s);
        }
    }
    return a;
}

/* O(n^2) DFT of one 1D sequence, for validating ref_fft_rec. */
static inline void ref_dft_direct(const ref_cplx* in, ref_cplx* out, long n, int sign)
{
    ref_cplx* tw = ref_twiddles(n, sign);
    for (long k = 0; k < n; ++k) {
        ref_cplx acc = {abf_hdd_make(0, 0), abf_hdd_make(0, 0)};
        for (long j = 0; j < n; ++j) acc = ref_add(acc, ref_mul(in[j], tw[(j * k) % n]));
        out[k] = acc;
    }
    free(tw);
}

/* Error of FP64 results y against reference r over m complex values:
 * l2 = ||y - r||_2 / ||r||_2, mx = max|y - r| / max|r| (componentwise over
 * real and imaginary parts). */
typedef struct { double l2, mx; } ref_err;

static inline ref_err ref_error(const double* y, const ref_cplx* r, long m)
{
    double dn = 0, rn = 0, dm = 0, rm = 0;
    for (long i = 0; i < m; ++i) {
        const abf_hdd* c[2] = {&r[i].re, &r[i].im};
        for (int k = 0; k < 2; ++k) {
            const double d = (y[2 * i + k] - c[k]->hi) - c[k]->lo;
            const double v = c[k]->hi;
            dn += d * d; rn += v * v;
            if (fabs(d) > dm) dm = fabs(d);
            if (fabs(v) > rm) rm = fabs(v);
        }
    }
    ref_err e = {rn > 0 ? sqrt(dn / rn) : sqrt(dn), rm > 0 ? dm / rm : dm};
    return e;
}

/* Same for two double-double sequences (reference self-check). */
static inline double ref_diff_dd(const ref_cplx* a, const ref_cplx* b, long m)
{
    double dm = 0, bm = 0;
    for (long i = 0; i < m; ++i) {
        const abf_hdd dr = abf_hdd_add(a[i].re, ref_neg(b[i].re)), di = abf_hdd_add(a[i].im, ref_neg(b[i].im));
        dm = fmax(dm, fmax(fabs(dr.hi), fabs(di.hi)));
        bm = fmax(bm, fmax(fabs(b[i].re.hi), fabs(b[i].im.hi)));
    }
    return bm > 0 ? dm / bm : dm;
}

#endif /* AB_FFT_REF_H */
