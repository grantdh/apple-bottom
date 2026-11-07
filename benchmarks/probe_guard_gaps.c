// =============================================================================
// probe_guard_gaps.c — data the BLAS ABI's exponent-range guard lets through
// =============================================================================
//   build/probe_guard_gaps > out.csv
//   AB_MODE=gpu AB_MIN_GPU_DIM=0 build/probe_guard_gaps > out.csv
//   AB_MODE=cpu build/probe_guard_gaps > out.csv
// The calls behind CHANGELOG 1.6.0's Known issues. One CSV row per call: the
// path it took and, for each part of the result, the largest error at sampled
// outputs against a Dot2 reference (tests/test_dot2.h), in u² = 2⁻⁴⁸ of that
// part's own scale: |Ar||Br| + |Ai||Bi| for the real part, |Ar||Bi| + |Ai||Br|
// for the imaginary part, |A||B| + |beta C| for a real result. A correct DD
// result is a few u² at most; about 2.8×10¹⁴ means the part came back zero.
// Alpha is 1 and beta 0 unless a row's data says otherwise. Correctness only,
// so no timing is read from it. The first command is the default routing,
// under which only the large T,T ZGEMM, ZGEMV and ZHEMV rows reach the GPU, and
// only on the M2 Max the routing table was measured on.
//
// Data, for each part of an element of A, B or x, u uniform in [-1, 1):
//   uniform  u
//   imagE    real part 1 + u/4, imaginary part (1 + u/4) 2^E
//   tiny     u 2^-1060 in A, u 2^1000 in B and x: products near 2^-60
//   spread   (1 + |u|/2) 2^-e, e uniform in 0..59, with a random sign
//   smallcE  DGEMM: row i of A nonzero only for k = i (mod 2), column j of B
//            only for k != j (mod 2), so half of C has no products at all and
//            is beta C alone; C = (1 + 2^-29) 2^E, beta 1 unless the row says
// =============================================================================

#include "apple_bottom.h"
#include "apple_bottom_device.h"
#include "test_blas2_ref.h"    // b2_elem, dot2_step
#include "test_blas_check.h"   // op_index, splitmix
#include <complex.h>
#include <math.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

typedef enum { UNIFORM, IMAG80, IMAG100, IMAG130, TINY, SPREAD } Data;
static const char *const DATA_NAME[] = {"uniform", "imag-80", "imag-100", "imag-130", "tiny", "spread"};

#define MAX_SAMPLES 4096

static double uni(uint64_t *s) {
    uint64_t z = splitmix(*s);
    *s += 0x9E3779B97F4A7C15ULL;
    return (double)(z >> 11) * 0x1p-52 - 1.0;
}

// One part of an element; large: B's or x's side of tiny.
static double part(Data d, uint64_t *s, bool large, bool im) {
    switch (d) {
    case IMAG80:  return ldexp(1.0 + 0.25 * uni(s), im ? -80 : 0);
    case IMAG100: return ldexp(1.0 + 0.25 * uni(s), im ? -100 : 0);
    case IMAG130: return ldexp(1.0 + 0.25 * uni(s), im ? -130 : 0);
    case TINY:    return ldexp(uni(s), large ? 1000 : -1060);
    case SPREAD: {
        double mag = ldexp(1.0 + 0.5 * fabs(uni(s)), -(int)(60.0 * fabs(uni(s))));
        return copysign(mag, uni(s));
    }
    default:      return uni(s);
    }
}

static double complex *zdata(Data d, size_t n, uint64_t seed, bool large) {
    double complex *z = malloc(n * sizeof *z);
    for (size_t i = 0; i < n; i++) {
        double re = part(d, &seed, large, false);
        z[i] = CMPLX(re, part(d, &seed, large, true));
    }
    return z;
}

// One complex output: both parts by Dot2, and their scales.
typedef struct { double rs, re, is, ie, sr, si; } ZOut;

static void zout_add(ZOut *o, double complex a, double complex b) {
    double ar = creal(a), ai = cimag(a), br = creal(b), bi = cimag(b);
    dot2_step(&o->rs, &o->re, ar, br);
    dot2_step(&o->rs, &o->re, -ai, bi);
    dot2_step(&o->is, &o->ie, ar, bi);
    dot2_step(&o->is, &o->ie, ai, br);
    o->sr += fabs(ar * br) + fabs(ai * bi);
    o->si += fabs(ar * bi) + fabs(ai * br);
}

// |x - r| in u² of scale s; Inf for a NaN x, which no maximum may drop.
static double part_err(double x, double r, double s) {
    double e = s > 0.0 ? fabs(x - r) / s / U2 : 0.0;
    return isnan(e) ? INFINITY : e;
}

static void zout_err(const ZOut *o, double complex x, double *er, double *ei) {
    *er = fmax(*er, part_err(creal(x), o->rs + o->re, o->sr));
    *ei = fmax(*ei, part_err(cimag(x), o->is + o->ie, o->si));
}

static void row(const char *routine, char ta, char tb, int M, int N, int K, const char *data,
                const char *path, double er, double ei) {
    printf("%s,%c%c,%d,%d,%d,%s,%s,%.3g,%.3g\n", routine, ta, tb, M, N, K, data, path, er, ei);
    fflush(stdout);
}

typedef enum { VIA_BLAS, VIA_DEVICE } Entry;

// C = op(A) op(B) with op(X) X or X^T.
static void zgemm_row(Entry via, char ta, char tb, int M, int N, int K, Data d) {
    const Layout la = stored(ta, M, K, 0), lb = stored(tb, K, N, 0);
    const size_t na = nelem(la), nb = nelem(lb), nc = (size_t)M * N;
    double complex *A = zdata(d, na, 11, false), *B = zdata(d, nb, 22, true);
    double complex *C = calloc(nc, sizeof *C);
    const char *path = "device";
    if (via == VIA_BLAS) {
        ab_zgemm_blas(ta, tb, M, N, K, 1.0, A, la.ld, B, lb.ld, 0.0, C, M);
        path = ab_get_last_dispatch_path();
    } else {
        const double one[2] = {1.0, 0.0}, zero[2] = {0.0, 0.0};
        const ABTranspose xa = ta == 'N' ? AB_NO_TRANS : AB_TRANS, xb = tb == 'N' ? AB_NO_TRANS : AB_TRANS;
        ab_dev_buffer_t dA = ab_dev_malloc(na * sizeof *A), dB = ab_dev_malloc(nb * sizeof *B),
                        dC = ab_dev_malloc(nc * sizeof *C);
        ab_dev_memcpy_h2d(dA, 0, A, na * sizeof *A);
        ab_dev_memcpy_h2d(dB, 0, B, nb * sizeof *B);
        ABStatus st = ab_dev_zgemm(xa, xb, M, N, K, one, dA, la.ld, dB, lb.ld, zero, dC, M);
        if (st != AB_OK) path = ab_status_string(st);
        ab_dev_memcpy_d2h(C, dC, 0, nc * sizeof *C);
        ab_dev_free(dA); ab_dev_free(dB); ab_dev_free(dC);
    }
    double er = 0.0, ei = 0.0;
    for (size_t s = 0, n = nc < MAX_SAMPLES ? nc : MAX_SAMPLES; s < n; s++) {
        size_t q = nc <= MAX_SAMPLES ? s : splitmix(s) % nc;
        int i = (int)(q % (size_t)M), j = (int)(q / (size_t)M);
        ZOut o = {0};
        for (int k = 0; k < K; k++) zout_add(&o, A[op_index(ta, la, i, k)], B[op_index(tb, lb, k, j)]);
        zout_err(&o, C[q], &er, &ei);
    }
    row(via == VIA_BLAS ? "zgemm" : "dev_zgemm", ta, tb, M, N, K, DATA_NAME[d], path, er, ei);
    free(A); free(B); free(C);
}

// y = A x, by ZGEMV 'N' or (hemv) ZHEMV 'U'.
static void zmv_row(bool hemv, int n, Data d) {
    const size_t nn = (size_t)n * n;
    double complex *A = zdata(d, nn, 33, false), *x = zdata(d, (size_t)n, 44, true);
    double complex *y = calloc((size_t)n, sizeof *y);
    if (hemv) ab_zhemv_blas('U', n, 1.0, A, n, x, 1, 0.0, y, 1);
    else      ab_zgemv_blas('N', n, n, 1.0, A, n, x, 1, 0.0, y, 1);
    const char *path = ab_get_last_dispatch_path();
    double er = 0.0, ei = 0.0;
    for (int i = 0; i < n; i++) {
        ZOut o = {0};
        for (int j = 0; j < n; j++) {
            double ar, ai;
            b2_elem(hemv ? B2_ZHEMV : B2_ZGEMV, hemv ? 'U' : 'N', (const double *)A, n, i, j, &ar, &ai);
            zout_add(&o, CMPLX(ar, ai), x[j]);
        }
        zout_err(&o, y[i], &er, &ei);
    }
    row(hemv ? "zhemv" : "zgemv", hemv ? 'U' : 'N', '-', n, n, 0, DATA_NAME[d], path, er, ei);
    free(A); free(x); free(y);
}

// C = A B + beta C0, all no-transpose, 8 x 8 x 2. small_c: C0 at 2^small_c on
// the disjoint supports of smallcE, else (0) the tiny data with beta = 0.
static void dgemm_row(int small_c, double beta) {
    enum { M = 8, N = 8, K = 2 };
    double A[M * K], B[K * N], C0[M * N], C[M * N];
    uint64_t s = 55;
    for (int k = 0; k < K; k++)
        for (int i = 0; i < M; i++) {
            double u = uni(&s);
            A[i + k * M] = small_c ? (k % 2 == i % 2 ? u : 0.0) : ldexp(u, -1060);
        }
    for (int j = 0; j < N; j++)
        for (int k = 0; k < K; k++) {
            double u = uni(&s);
            B[k + j * K] = small_c ? (k % 2 != j % 2 ? u : 0.0) : ldexp(u, 1000);
        }
    for (int i = 0; i < M * N; i++) C0[i] = small_c ? ldexp(1.0 + 0x1p-29, small_c) : 0.0;
    memcpy(C, C0, sizeof C);
    ab_dgemm_blas('N', 'N', M, N, K, 1.0, A, M, B, K, beta, C, M);
    const char *path = ab_get_last_dispatch_path();
    double er = 0.0;
    for (int j = 0; j < N; j++)
        for (int i = 0; i < M; i++) {
            double rs = 0.0, re = 0.0, sc = fabs(beta * C0[i + j * M]);
            for (int k = 0; k < K; k++) {
                dot2_step(&rs, &re, A[i + k * M], B[k + j * K]);
                sc += fabs(A[i + k * M] * B[k + j * K]);
            }
            dot2_step(&rs, &re, beta, C0[i + j * M]);
            er = fmax(er, part_err(C[i + j * M], rs + re, sc));
        }
    char name[32];
    if (small_c) snprintf(name, sizeof name, "smallc%d beta %g", small_c, beta);
    row("dgemm", 'N', 'N', M, N, K, small_c ? name : DATA_NAME[TINY], path, er, 0.0);
}

int main(void) {
    if (ab_init() != AB_OK) {
        fprintf(stderr, "probe_guard_gaps: no Metal device\n");
        return 1;
    }
    const char *mode = getenv("AB_MODE");
    fprintf(stderr, "%s, AB_MODE=%s\n", ab_build_info(), mode ? mode : "(unset)");
    printf("routine,trans,M,N,K,data,path,re_u2,im_u2\n");

    // What AB_MODE=auto sends to the GPU on an M2 Max
    static const Data LARGE[] = {UNIFORM, SPREAD, IMAG80, IMAG100, IMAG130, TINY};
    for (size_t i = 0; i < sizeof LARGE / sizeof *LARGE; i++)
        zgemm_row(VIA_BLAS, 'T', 'T', 4096, 4096, 4096, LARGE[i]);
    for (size_t i = 0; i < sizeof LARGE / sizeof *LARGE; i++) {
        zmv_row(false, 3072, LARGE[i]);
        zmv_row(true, 2048, LARGE[i]);
    }
    // Small shapes, on the GPU only when forced
    for (size_t i = 0; i < sizeof LARGE / sizeof *LARGE; i++)
        zgemm_row(VIA_BLAS, 'N', 'N', 8, 8, 2, LARGE[i]);
    zgemm_row(VIA_BLAS, 'N', 'N', 544, 481, 600, SPREAD);
    zgemm_row(VIA_BLAS, 'N', 'N', 544, 481, 600, IMAG100);
    dgemm_row(0, 0.0);
    dgemm_row(-98, 0x1p20);
    dgemm_row(-98, 1.0);
    dgemm_row(-100, 1.0);
    dgemm_row(-130, 1.0);
    // The device API: no guard and no fallback
    zgemm_row(VIA_DEVICE, 'N', 'N', 544, 481, 600, SPREAD);
    zgemm_row(VIA_DEVICE, 'N', 'N', 544, 481, 600, IMAG100);
    ab_shutdown();
    return 0;
}
