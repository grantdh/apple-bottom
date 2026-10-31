// Exact-arithmetic check of the fused double-float multiply-accumulate.
//
// The shader primitives (struct DD .. dd_fma) are extracted from the MSL source
// in src/apple_bottom.m at build time (build/dd_prims.inc) and compiled here as
// C++ without contraction, so this checks the code the GPU runs, not a copy.
// Every result is compared with MPFR at 400 bits against the bound proven in
// docs/vv/PRECISION_ENVELOPE.md §1.2 (default build, DWTimesDW2-style product):
//
//   (i)   the final fastTwoSum of dd_fma is exact;
//   (ii)  |P - ab| <= (6u^2 + 2u^3) |a.hi b.hi| for the unnormalized product P;
//   (iii) |dd_fma(a,b,c) - (ab+c)| <= (13+43u) u^2 |a||b| + (3+5u) u^2 |c|,
//
// with u = 2^-24, no underflow or overflow. Half the triples are adversarial
// (significands 1 and 2-2^-23, low words at or near half an ulp), and half have
// c.hi within 32 ulps of -fl(a.hi b.hi), where fastTwoSum's |s| >= |w| can fail.
//
// Not part of `make test` (needs MPFR): `make test-dd-bound [DD_BOUND_N=...]`.

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <mpfr.h>

using std::fma;
#define thread
#include "dd_prims.inc"
#undef thread

#ifdef APPLEBOTTOM_USE_DWTIMESDW3
#error "the bound is proven for the default (DWTimesDW2-style) product only"
#endif

// dd_fma step by step, to expose s and w for check (i). It must agree bitwise
// with the extracted dd_fma; the test fails if it does not.
static DD mirror_fma(DD a, DD b, DD c, float *s_out, float *w_out, float *p_out, float *f_out) {
    float p, e;
    twoProduct(a.hi, b.hi, p, e);
    float f = fma(a.hi, b.lo, fma(a.lo, b.hi, e));
    float s, g;
    twoSum(p, c.hi, s, g);
    float w = g + (f + c.lo);
    *s_out = s; *w_out = w; *p_out = p; *f_out = f;
    float hi, lo;
    fastTwoSum(s, w, hi, lo);
    return {hi, lo};
}

static uint64_t rs = 0x9E3779B97F4A7C15ull;
static uint64_t rnd() { rs ^= rs << 13; rs ^= rs >> 7; rs ^= rs << 17; return rs; }
static double unit() { return (double)(rnd() >> 11) * 0x1p-53; }

static float rand_hi(bool adv) {
    uint32_t m;
    int k = adv ? (int)(rnd() % 4) : 3;
    if (k == 0) m = 0;
    else if (k == 1) m = 0x7FFFFF;
    else if (k == 2) m = 0x7FFFFF - (uint32_t)(rnd() % 16);
    else m = (uint32_t)(rnd() & 0x7FFFFF);
    int e = (int)(rnd() % 9) - 4;
    uint32_t bits = ((uint32_t)(127 + e) << 23) | m | (uint32_t)((rnd() & 1) << 31);
    float f; std::memcpy(&f, &bits, 4); return f;
}

static DD rand_dd(bool adv) {
    float h = rand_hi(adv), ulp = std::ldexp(1.0f, std::ilogb(h) - 23);
    double fr = (adv && (rnd() & 1)) ? 1.0 - unit() * 1e-3 : unit();
    float l = (float)(0.5 * ulp * fr);
    if (rnd() & 1) l = -l;
    if (adv && rnd() % 8 == 0) l = (rnd() & 1) ? 0.5f * ulp : -0.5f * ulp;
    float s, e; fastTwoSum(h, l, s, e);
    return {s, e};
}

static DD cancelling_c(DD a, DD b, bool exact) {
    float p = a.hi * b.hi;
    int k = (int)(rnd() % 64) - 32;
    float ch = exact ? -p : -p + (float)k * std::ldexp(1.0f, std::ilogb(p) - 23);
    float ulp = std::ldexp(1.0f, std::ilogb(ch == 0 ? 1.0f : ch) - 23);
    float cl = (float)(0.5 * ulp * unit()) * ((rnd() & 1) ? 1.0f : -1.0f);
    float s, e; fastTwoSum(ch, cl, s, e);
    return {s, e};
}

static void set_dd(mpfr_t r, float hi, float lo) { mpfr_set_flt(r, hi, MPFR_RNDN); mpfr_add_d(r, r, lo, MPFR_RNDN); }

int main(int argc, char **argv) {
    long n = argc > 1 ? atol(argv[1]) : 5000000;
    const double u = 0x1p-24, u2 = 0x1p-48;
    const double alpha = 13 + 43 * u, beta = 3 + 5 * u, pbound = 6 * u2 + 2 * u * u2;
    mpfr_t x, y, c, xy, ex, r, t, err;
    mpfr_inits2(400, x, y, c, xy, ex, r, t, err, (mpfr_ptr)0);

    long mirror_diff = 0, f2s_inexact = 0, prod_fail = 0, step_fail = 0;
    double max_prod = 0, max_ratio = 0, max_beta = 0;
    for (long i = 0; i < n; i++) {
        bool adv = i & 1;
        DD a = rand_dd(adv), b = rand_dd(adv), cc;
        int mode = (int)(rnd() % 4);
        if (mode == 0) cc = {0.0f, 0.0f};
        else if (mode == 1) cc = rand_dd(adv);
        else cc = cancelling_c(a, b, mode == 3);

        DD out = dd_fma(a, b, cc);
        float s, w, p, f;
        DD mo = mirror_fma(a, b, cc, &s, &w, &p, &f);
        if (std::memcmp(&out, &mo, sizeof out) != 0) mirror_diff++;

        set_dd(x, a.hi, a.lo); set_dd(y, b.hi, b.lo); set_dd(c, cc.hi, cc.lo);
        mpfr_mul(xy, x, y, MPFR_RNDN);
        mpfr_add(ex, xy, c, MPFR_RNDN);

        set_dd(r, out.hi, out.lo);                      // (i)
        set_dd(t, s, w);
        if (!mpfr_equal_p(r, t)) f2s_inexact++;

        set_dd(t, p, f);                                 // (ii)
        mpfr_sub(err, t, xy, MPFR_RNDN);
        double pe = std::fabs(mpfr_get_d(err, MPFR_RNDU));
        double ahbh = std::fabs((double)a.hi * (double)b.hi);
        if (pe > pbound * ahbh) prod_fail++;
        if (ahbh > 0 && pe / (u2 * ahbh) > max_prod) max_prod = pe / (u2 * ahbh);

        mpfr_sub(err, r, ex, MPFR_RNDN);                 // (iii)
        double E = std::fabs(mpfr_get_d(err, MPFR_RNDU));
        double ax = std::fabs(mpfr_get_d(xy, MPFR_RNDN)), ac = std::fabs(mpfr_get_d(c, MPFR_RNDN));
        double bound = alpha * u2 * ax + beta * u2 * ac;
        if (E > bound) step_fail++;
        if (bound > 0 && E / bound > max_ratio) max_ratio = E / bound;
        if (ac > 64 * ax && E / (u2 * ac) > max_beta) max_beta = E / (u2 * ac);
    }
    mpfr_clears(x, y, c, xy, ex, r, t, err, (mpfr_ptr)0);

    printf("dd_fma exact-arithmetic check: %ld triples\n", n);
    printf("  mirror differs from extracted dd_fma : %ld\n", mirror_diff);
    printf("  (i)   fastTwoSum inexact             : %ld\n", f2s_inexact);
    printf("  (ii)  product over (6u^2+2u^3)|ahbh| : %ld   (max %.4f u^2)\n", prod_fail, max_prod);
    printf("  (iii) step over proven bound         : %ld   (max %.4f of bound; %.4f u^2|c| when |c| > 64|ab|)\n",
           step_fail, max_ratio, max_beta);
    bool ok = mirror_diff == 0 && f2s_inexact == 0 && prod_fail == 0 && step_fail == 0;
    printf("%s\n", ok ? "PASS" : "FAIL");
    return ok ? 0 : 1;
}
