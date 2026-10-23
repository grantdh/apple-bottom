// =============================================================================
// test_fft.c — the double-float FFT (apple_bottom_fft.h)
// =============================================================================
//
// Every GPU result is checked two ways:
//   - against a double-double reference DFT (tests/fft_ref.h, ~1e-31): the
//     relative L2 and max errors must stay within bounds set from what the
//     DD arithmetic achieves (about ten times FP64's error);
//   - against the CPU emulator (tests/fft_emu.cpp), which runs the same
//     double-float arithmetic from src/fft/fft_device.h: the bits must match,
//     which pins down every kernel's addressing and order of operations.
// Then: lengths (radix 2..13, four-step), 2D/3D batches, FFTW-style strides
// and embeddings with untouched padding, in place, zero copy (ab_fft_malloc,
// ab_fft_register), chunked batches, normalizations, bad arguments, range
// limits, threads and fork.
// =============================================================================

#include <math.h>
#include <pthread.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/wait.h>
#include <unistd.h>

#include "apple_bottom_fft.h"
#include "fft_ref.h"
#include "test_blas_check.h"   // clear_ab_env, splitmix
#include "test_registry.h"

int abf_emu_fft(int rank, const int* n, int howmany, int sign, int norm, const double* in, double* out);
void abf_emu_to_dd(double x, float* hi, float* lo);
double abf_emu_to_f64(float hi, float lo);

static int failures = 0;
static int checks_run = 0;

#define CHECK(cond, ...) do {                                        \
    checks_run++;                                                    \
    if (!(cond)) {                                                   \
        fprintf(stderr, "  ✗ FAIL (%s:%d): ", __FILE__, __LINE__);  \
        fprintf(stderr, __VA_ARGS__);                                \
        fprintf(stderr, "\n");                                       \
        failures++;                                                  \
    }                                                                \
} while (0)

// Error bounds against the reference. The double-float FFT's normwise error
// grows like log2(N) u with u = 2^-48 (~3.6e-15); these allow ~3x over the
// worst value measured on the M2 Max (printed by every case).
static double tol_l2(long N) { return 4e-15 * (1.0 + log2((double)N) / 4.0); }
static double tol_max(long N) { return 2.0 * tol_l2(N); }

static uint64_t g_rng = 0x9E3779B97F4A7C15ull;   // the same sequence as splitmix64 seeded at 0
static double urand(void)   // [-1, 1), 53 random bits
{
    const uint64_t z = splitmix(g_rng);
    g_rng += 0x9E3779B97F4A7C15ull;
    return (double)(z >> 11) * (2.0 / 9007199254740992.0) - 1.0;
}

static const double SENTINEL = -123456.75;

typedef struct {
    int rank, n[3], howmany;
    int iemb[3], oemb[3];         // {0}: packed
    int istride, ostride;         // 0: 1
    int idist, odist;             // 0: packed
    int sign, norm;
    int inplace;                  // out = in (same layout)
    int zero_copy;                // arrays from ab_fft_malloc
} fft_case;

typedef struct { long n[3], emb[3], stride, dist; } layout;

static layout make_layout(const fft_case* c, const int* emb, int stride, int dist)
{
    layout l;
    for (int d = 0; d < 3; ++d) {
        l.n[d] = d < c->rank ? c->n[d] : 1;
        l.emb[d] = (emb[0] && d < c->rank - 1) ? emb[d] : l.n[d];
    }
    l.stride = stride ? stride : 1;
    l.dist = dist ? dist : l.stride * l.emb[0] * l.emb[1] * l.n[2];
    return l;
}

static long lay_off(const layout* l, long b, long i0, long i1, long i2)
{
    return b * l->dist + l->stride * (i0 + l->emb[0] * (i1 + l->emb[1] * i2));
}

static long lay_elems(const layout* l, long howmany)
{
    return lay_off(l, howmany - 1, l->n[0] - 1, l->n[1] - 1, l->n[2] - 1) + 1;
}

// Packed copy of the transform elements (pack = 1) or back (pack = 0).
static void gather(const layout* l, long howmany, double* arr, double* packed, int pack)
{
    long k = 0;
    for (long b = 0; b < howmany; ++b)
        for (long i2 = 0; i2 < l->n[2]; ++i2)
            for (long i1 = 0; i1 < l->n[1]; ++i1)
                for (long i0 = 0; i0 < l->n[0]; ++i0, ++k) {
                    double* e = arr + 2 * lay_off(l, b, i0, i1, i2);
                    if (pack) { packed[2 * k] = e[0]; packed[2 * k + 1] = e[1]; }
                    else { e[0] = packed[2 * k]; e[1] = packed[2 * k + 1]; }
                }
}

typedef struct { ref_err err; long same, total; ab_fft_stats st; } case_result;

// Run one case on the GPU, compare with the reference and the emulator.
static case_result run_case(const fft_case* c, const char* label, int quiet)
{
    case_result res;
    memset(&res, 0, sizeof res);
    const layout li = make_layout(c, c->iemb, c->istride, c->idist);
    const layout lo = c->inplace ? li : make_layout(c, c->oemb, c->ostride, c->odist);
    const long N = li.n[0] * li.n[1] * li.n[2], H = c->howmany;
    const long ein = lay_elems(&li, H), eout = lay_elems(&lo, H);
    double* in = c->zero_copy ? ab_fft_malloc((size_t)ein * 16) : malloc((size_t)ein * 16);
    double* out = c->inplace ? in : (c->zero_copy ? ab_fft_malloc((size_t)eout * 16) : malloc((size_t)eout * 16));
    double* x = malloc((size_t)(N * H) * 16);
    double* y = malloc((size_t)(N * H) * 16);
    double* ye = malloc((size_t)(N * H) * 16);
    for (long i = 0; i < 2 * ein; ++i) in[i] = SENTINEL;
    if (!c->inplace) for (long i = 0; i < 2 * eout; ++i) out[i] = SENTINEL;
    for (long i = 0; i < 2 * N * H; ++i) x[i] = urand();
    gather(&li, H, in, x, 0);

    int iemb[3] = {(int)li.emb[0], (int)li.emb[1], (int)li.emb[2]};
    int oemb[3] = {(int)lo.emb[0], (int)lo.emb[1], (int)lo.emb[2]};
    ABStatus st;
    ab_fft_plan_t p = ab_fft_plan_many(c->rank, c->n, c->howmany, iemb, (int)li.stride, (int)li.dist, oemb,
                                       (int)lo.stride, (int)lo.dist, c->sign, (ab_fft_norm)c->norm, &st);
    if (!p) {
        CHECK(0, "%s: plan failed (%s): %s", label, ab_status_string(st), ab_fft_last_error());
        goto done;
    }
    st = ab_fft_execute(p, in, out);
    ab_fft_get_stats(p, &res.st);
    ab_fft_destroy(p);
    if (st != AB_OK) {
        CHECK(0, "%s: execute failed (%s): %s", label, ab_status_string(st), ab_fft_last_error());
        goto done;
    }
    gather(&lo, H, out, y, 1);

    // Padding between output elements is left as it was.
    if (!c->inplace) {
        double* chk = malloc((size_t)eout * 16);
        memcpy(chk, out, (size_t)eout * 16);
        for (long i = 0; i < 2 * N * H; ++i) ye[i] = SENTINEL;
        gather(&lo, H, chk, ye, 0);
        long bad = 0;
        for (long i = 0; i < 2 * eout; ++i) bad += chk[i] != SENTINEL;
        CHECK(bad == 0, "%s: %ld padding values changed", label, bad);
        free(chk);
    }

    const long nn[3] = {li.n[0], li.n[1], li.n[2]};
    ref_cplx* r = ref_transform(x, nn, H, c->sign, c->norm);
    res.err = ref_error(y, r, N * H);
    free(r);
    abf_emu_fft(c->rank, c->n, c->howmany, c->sign, c->norm, x, ye);
    res.total = 2 * N * H;
    for (long i = 0; i < res.total; ++i) res.same += memcmp(&y[i], &ye[i], 8) == 0;
    const bool l2ok = res.err.l2 <= tol_l2(N), mxok = res.err.mx <= tol_max(N), bitsok = res.same == res.total;
    if (!quiet || !(l2ok && mxok && bitsok))
        printf("    %-44s L2 %.2e max %.2e  emulator bits %ld/%ld%s\n", label, res.err.l2, res.err.mx, res.same,
               res.total, l2ok && mxok && bitsok ? "" : "  <-- FAIL");
    CHECK(l2ok, "%s: L2 error %.3e > %.3e", label, res.err.l2, tol_l2(N));
    CHECK(mxok, "%s: max error %.3e > %.3e", label, res.err.mx, tol_max(N));
    CHECK(bitsok, "%s: %ld of %ld values differ from the CPU emulator", label, res.total - res.same, res.total);
done:
    if (c->zero_copy) { ab_fft_free(in); if (!c->inplace) ab_fft_free(out); }
    else { free(in); if (!c->inplace) free(out); }
    free(x); free(y); free(ye);
    return res;
}

static fft_case packed(int rank, int n0, int n1, int n2, int howmany, int sign, int norm)
{
    fft_case c;
    memset(&c, 0, sizeof c);
    c.rank = rank; c.n[0] = n0; c.n[1] = n1; c.n[2] = n2; c.howmany = howmany;
    c.sign = sign; c.norm = norm;
    return c;
}

// =============================================================================

static void test_reference(void)
{
    // The fast reference against the O(n^2) sum, both in double-double.
    static const long lens[] = {1, 2, 3, 4, 5, 6, 7, 8, 9, 11, 12, 13, 16, 17, 30, 49, 60, 64, 77, 120, 143, 169};
    double worst = 0;
    for (size_t li = 0; li < sizeof lens / sizeof lens[0]; ++li) {
        const long n = lens[li];
        ref_cplx* a = malloc(sizeof(ref_cplx) * (size_t)n);
        ref_cplx* f = malloc(sizeof(ref_cplx) * (size_t)n);
        ref_cplx* d = malloc(sizeof(ref_cplx) * (size_t)n);
        for (long i = 0; i < n; ++i) { a[i].re = abf_hdd_make(urand(), 0); a[i].im = abf_hdd_make(urand(), 0); }
        for (int sign = -1; sign <= 1; sign += 2) {
            ref_cplx* tw = ref_twiddles(n, sign);
            ref_fft_rec(a, 1, f, n, tw, 1);
            ref_dft_direct(a, d, n, sign);
            worst = fmax(worst, ref_diff_dd(f, d, n));
            free(tw);
        }
        free(a); free(f); free(d);
    }
    printf("    double-double FFT vs O(n^2) DFT, worst max rel. difference %.2e\n", worst);
    CHECK(worst < 1e-28, "reference FFT disagrees with the direct DFT: %.3e", worst);
}

// The GPU's conversions, run on the CPU from the same code (bit patterns):
// FP64 -> double-float must equal hi = (float)x, lo = (float)(x - hi), and
// double-float -> FP64 must equal the FP64 sum (double)hi + (double)lo, for
// normalized pairs, ties, powers of two and lo of either sign.
static void test_conversions(void)
{
    long bad_in = 0, bad_out = 0, n = 0;
    for (long t = 0; t < 2000000; ++t) {
        const int ex = (int)(fabs(urand()) * 200) - 100;
        const double x = ldexp(urand(), ex);
        float h, l;
        abf_emu_to_dd(x, &h, &l);
        const float hh = (float)x, ll = (float)(x - (double)hh);
        if (fabsf(ll) >= 1.17549435e-38f || ll == 0) bad_in += memcmp(&h, &hh, 4) || memcmp(&l, &ll, 4);

        // a normalized pair: |lo| <= ulp(hi)/2, sometimes exactly a tie or tiny
        float hi = (float)ldexp(urand(), ex);
        if (t % 7 == 0) hi = (float)ldexp(t % 2 ? 1.0 : -1.0, ex);   // power of two
        int eh;
        frexpf(hi, &eh);
        const int kind = (int)(t % 5);
        float lo = (float)ldexp(urand(), eh - 24 - (kind == 3 ? (int)(fabs(urand()) * 40) : 0));
        if (kind == 4) lo = (float)ldexp(t % 3 ? 0.5 : -0.5, eh - 24 - 29);   // half an FP64 ulp
        if (hi == 0 || fabsf(lo) < 1.17549435e-38f) continue;
        const double ref = (double)hi + (double)lo, got = abf_emu_to_f64(hi, lo);
        bad_out += memcmp(&ref, &got, 8) != 0;
        ++n;
    }
    printf("    FP64 -> double-float: %ld mismatches; double-float -> FP64: %ld of %ld mismatches\n", bad_in, bad_out, n);
    CHECK(bad_in == 0, "FP64 -> double-float differs from the host split in %ld cases", bad_in);
    CHECK(bad_out == 0, "double-float -> FP64 differs from (double)hi + (double)lo in %ld cases", bad_out);
}

static void test_lengths_1d(void)
{
    // Every radix and mixes of them, both directions, batch of 3.
    static const int lens[] = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 18, 20, 21, 22, 24, 25, 26, 27,
                               28, 30, 32, 33, 35, 36, 39, 40, 42, 44, 45, 48, 49, 52, 55, 56, 60, 63, 64, 65, 66, 72,
                               77, 81, 84, 88, 91, 96, 99, 100, 104, 108, 117, 120, 121, 125, 128, 143, 144, 169, 180,
                               192, 210, 216, 240, 243, 256, 343, 384, 480, 500, 512, 625, 720, 729, 1000, 1001, 1024,
                               1331, 1536, 2000, 2048};
    double wl2 = 0, wmx = 0;
    int bad = 0;
    for (size_t i = 0; i < sizeof lens / sizeof lens[0]; ++i)
        for (int sign = -1; sign <= 1; sign += 2) {
            fft_case c = packed(1, lens[i], 1, 1, 3, sign, AB_FFT_NORM_NONE);
            char label[64];
            snprintf(label, sizeof label, "1D n=%d sign %+d", lens[i], sign);
            case_result r = run_case(&c, label, 1);
            wl2 = fmax(wl2, r.err.l2 / tol_l2(lens[i]));
            wmx = fmax(wmx, r.err.mx / tol_max(lens[i]));
            bad += r.err.l2 > tol_l2(lens[i]) || r.err.mx > tol_max(lens[i]) || r.same != r.total;
        }
    printf("    %zu lengths x 2 signs: worst error / bound: L2 %.2f, max %.2f; %d failing\n",
           sizeof lens / sizeof lens[0], wl2, wmx, bad);
}

static void test_four_step(void)
{
    static const struct { int n, howmany, sign, norm; } cases[] = {
        {4096, 2, AB_FFT_FORWARD, AB_FFT_NORM_NONE},   {6000, 2, AB_FFT_BACKWARD, AB_FFT_NORM_N},
        {10000, 2, AB_FFT_FORWARD, AB_FFT_NORM_NONE},  {65536, 2, AB_FFT_BACKWARD, AB_FFT_NORM_NONE},
        {2048 * 2048, 1, AB_FFT_FORWARD, AB_FFT_NORM_SQRT_N},
    };
    for (size_t i = 0; i < sizeof cases / sizeof cases[0]; ++i) {
        fft_case c = packed(1, cases[i].n, 1, 1, cases[i].howmany, cases[i].sign, cases[i].norm);
        char label[64];
        snprintf(label, sizeof label, "1D four-step n=%d x%d", cases[i].n, cases[i].howmany);
        run_case(&c, label, 0);
    }
}

static void test_multi_dim(void)
{
    static const int dims[][4] = {   // rank, n0, n1, n2
        {2, 6, 10, 1}, {2, 48, 36, 1}, {2, 1, 64, 1}, {2, 128, 120, 1}, {2, 2048, 3, 1},
        {3, 8, 6, 5}, {3, 12, 10, 9}, {3, 24, 18, 20}, {3, 1, 1, 7}, {3, 7, 1, 11}, {3, 32, 32, 32},
        {3, 48, 48, 48}, {3, 72, 72, 72}, {3, 3, 2048, 2},
    };
    for (size_t i = 0; i < sizeof dims / sizeof dims[0]; ++i) {
        const int H = dims[i][1] * dims[i][2] * dims[i][3] > 100000 ? 1 : 3;
        for (int sign = -1; sign <= 1; sign += 2) {
            fft_case c = packed(dims[i][0], dims[i][1], dims[i][2], dims[i][3], H, sign,
                                sign < 0 ? AB_FFT_NORM_NONE : AB_FFT_NORM_N);
            char label[80];
            snprintf(label, sizeof label, "%dD %dx%dx%d x%d sign %+d", dims[i][0], dims[i][1], dims[i][2], dims[i][3],
                     H, sign);
            run_case(&c, label, 0);
        }
    }
}

static void test_layouts(void)
{
    fft_case c;
    // QE-style padded grid psic(nr1x, nr2x, nr3x) in a batch with gaps.
    c = packed(3, 10, 12, 7, 3, AB_FFT_FORWARD, AB_FFT_NORM_NONE);
    c.iemb[0] = 11; c.iemb[1] = 13; c.oemb[0] = 12; c.oemb[1] = 12;
    c.idist = 11 * 13 * 7 + 5; c.odist = 12 * 12 * 7 + 3;
    run_case(&c, "3D 10x12x7 embedded, gaps between transforms", 0);
    // Strided elements.
    c = packed(2, 9, 8, 1, 2, AB_FFT_BACKWARD, AB_FFT_NORM_SQRT_N);
    c.istride = 2; c.ostride = 3;
    run_case(&c, "2D 9x8 istride 2 ostride 3", 0);
    // Interleaved batch in (dist 1, stride howmany), packed out.
    c = packed(1, 64, 1, 1, 5, AB_FFT_FORWARD, AB_FFT_NORM_NONE);
    c.istride = 5; c.idist = 1; c.odist = 64;
    run_case(&c, "1D 64 x5 interleaved in, packed out", 0);
    // Packed in, interleaved out, 3D.
    c = packed(3, 6, 5, 4, 4, AB_FFT_BACKWARD, AB_FFT_NORM_N);
    c.ostride = 4; c.odist = 1;
    run_case(&c, "3D 6x5x4 x4 packed in, interleaved out", 0);
    // Four-step with strides.
    c = packed(1, 4096, 1, 1, 2, AB_FFT_FORWARD, AB_FFT_NORM_NONE);
    c.istride = 2; c.idist = 8200; c.ostride = 1; c.odist = 4100;
    run_case(&c, "1D four-step 4096 strided", 0);
}

static void test_in_place(void)
{
    fft_case c = packed(3, 20, 18, 16, 2, AB_FFT_FORWARD, AB_FFT_NORM_NONE);
    c.inplace = 1;
    run_case(&c, "3D 20x18x16 x2 in place, staged", 0);
    c.zero_copy = 1;
    case_result r = run_case(&c, "3D 20x18x16 x2 in place, zero copy", 0);
    CHECK(r.st.zero_copy_in && r.st.zero_copy_out, "in-place ab_fft_malloc arrays not used in place");
    c = packed(1, 1000, 1, 1, 7, AB_FFT_BACKWARD, AB_FFT_NORM_N);
    c.inplace = 1;
    run_case(&c, "1D 1000 x7 in place", 0);
    c = packed(1, 8192, 1, 1, 2, AB_FFT_BACKWARD, AB_FFT_NORM_NONE);
    c.inplace = 1; c.zero_copy = 1;
    run_case(&c, "1D four-step 8192 x2 in place, zero copy", 0);
}

// Zero copy must give the staged path's bits; registration is explicit.
static void test_zero_copy(void)
{
    const int n[3] = {24, 20, 18}, H = 3;
    const long N = 24 * 20 * 18;
    const size_t bytes = (size_t)(N * H) * 16;
    ABStatus st;
    ab_fft_plan_t p = ab_fft_plan_3d(n[0], n[1], n[2], H, AB_FFT_FORWARD, AB_FFT_NORM_NONE, &st);
    CHECK(p != NULL, "plan: %s", ab_fft_last_error());
    if (!p) return;
    double* x = malloc(bytes);
    double* ref = malloc(bytes);
    for (long i = 0; i < 2 * N * H; ++i) x[i] = urand();
    ab_fft_stats s;
    CHECK(ab_fft_execute(p, x, ref) == AB_OK, "staged execute");
    ab_fft_get_stats(p, &s);
    CHECK(!s.zero_copy_in && !s.zero_copy_out, "malloc'd arrays must be staged");

    double* zi = ab_fft_malloc(bytes);
    double* zo = ab_fft_malloc(bytes);
    CHECK(zi && zo && ((uintptr_t)zi % 16384) == 0, "ab_fft_malloc returns page-aligned memory");
    memcpy(zi, x, bytes);
    CHECK(ab_fft_execute(p, zi, zo) == AB_OK, "zero-copy execute");
    ab_fft_get_stats(p, &s);
    CHECK(s.zero_copy_in && s.zero_copy_out && s.copy_in_ms == 0 && s.copy_out_ms == 0,
          "ab_fft_malloc arrays used in place (in %d out %d)", s.zero_copy_in, s.zero_copy_out);
    CHECK(memcmp(zo, ref, bytes) == 0, "zero copy gives the staged bits");
    // Mixed: registered input, staged output.
    double* o2 = malloc(bytes);
    CHECK(ab_fft_execute(p, zi, o2) == AB_OK && memcmp(o2, ref, bytes) == 0, "zero-copy in, staged out");
    ab_fft_get_stats(p, &s);
    CHECK(s.zero_copy_in && !s.zero_copy_out, "mixed path flags");

    // Register plain malloc'd memory (not page-aligned), offset into it.
    double* big = malloc(bytes + 4096);
    double* xr = big + 2 * 37;   // 37 complex elements in
    memcpy(xr, x, bytes);
    CHECK(ab_fft_register(big, bytes + 4096) == AB_OK, "register malloc'd memory: %s", ab_fft_last_error());
    CHECK(ab_fft_register(big + 100, 64) == AB_ERROR_INVALID_ARG, "overlapping registration rejected");
    memset(o2, 0, bytes);
    CHECK(ab_fft_execute(p, xr, o2) == AB_OK && memcmp(o2, ref, bytes) == 0, "registered input");
    ab_fft_get_stats(p, &s);
    CHECK(s.zero_copy_in == 1, "registered input read in place");
    CHECK(ab_fft_unregister(big) == AB_OK, "unregister");
    CHECK(ab_fft_unregister(big) == AB_ERROR_INVALID_ARG, "unregister twice rejected");
    // After unregister, free and a new allocation (possibly at the same
    // address) the data must be read afresh: nothing is cached by address.
    free(big);
    big = malloc(bytes + 4096);
    xr = big + 2 * 37;
    for (long i = 0; i < 2 * N * H; ++i) xr[i] = 0.5 * x[i];
    CHECK(ab_fft_execute(p, xr, o2) == AB_OK, "execute after free/malloc");
    ab_fft_get_stats(p, &s);
    long bad = 0;
    for (long i = 0; i < 2 * N * H; ++i) bad += o2[i] != 0.5 * ref[i];
    CHECK(s.zero_copy_in == 0 && bad == 0, "new data at a reused address (staged, %ld wrong)", bad);
    free(big);
    ab_fft_free(zi);
    ab_fft_free(zo);
    free(o2); free(x); free(ref);
    ab_fft_destroy(p);
}

// A batch split into chunks (AB_FFT_WORK_MB) gives the unsplit bits.
static void test_chunks(void)
{
    const int n[3] = {64, 48, 40}, H = 9;
    const long N = 64 * 48 * 40;
    const size_t bytes = (size_t)(N * H) * 16;
    double* x = malloc(bytes);
    double* a = malloc(bytes);
    double* b = malloc(bytes);
    for (long i = 0; i < 2 * N * H; ++i) x[i] = urand();
    ABStatus st;
    ab_fft_plan_t p1 = ab_fft_plan_3d(n[0], n[1], n[2], H, AB_FFT_BACKWARD, AB_FFT_NORM_N, &st);
    setenv("AB_FFT_WORK_MB", "4", 1);   // 2 transforms per chunk
    ab_fft_plan_t p2 = ab_fft_plan_3d(n[0], n[1], n[2], H, AB_FFT_BACKWARD, AB_FFT_NORM_N, &st);
    unsetenv("AB_FFT_WORK_MB");
    CHECK(p1 && p2, "plans");
    if (!p1 || !p2) return;
    ab_fft_stats s1, s2;
    ab_fft_execute(p1, x, a);
    ab_fft_get_stats(p1, &s1);
    ab_fft_execute(p2, x, b);
    ab_fft_get_stats(p2, &s2);
    printf("    %d transforms of %dx%dx%d: %d chunk(s) vs %d chunks (staged)\n", H, n[0], n[1], n[2], s1.chunks,
           s2.chunks);
    CHECK(s1.chunks == 1 && s2.chunks == 5, "chunk counts %d, %d", s1.chunks, s2.chunks);
    CHECK(memcmp(a, b, bytes) == 0, "chunked staged result differs");
    double* zi = ab_fft_malloc(bytes);
    double* zo = ab_fft_malloc(bytes);
    memcpy(zi, x, bytes);
    ab_fft_execute(p2, zi, zo);
    ab_fft_get_stats(p2, &s2);
    CHECK(s2.zero_copy_in && s2.zero_copy_out && memcmp(a, zo, bytes) == 0, "chunked zero-copy result differs");
    ab_fft_free(zi); ab_fft_free(zo);
    ab_fft_destroy(p1); ab_fft_destroy(p2);
    free(x); free(a); free(b);
}

static void test_normalization(void)
{
    // forward(N) then backward(none) returns the input; sqrt(N) both ways too.
    const int n0 = 30, n1 = 28, n2 = 26, H = 2;
    const long N = (long)n0 * n1 * n2;
    const size_t bytes = (size_t)(N * H) * 16;
    double* x = malloc(bytes);
    double* y = malloc(bytes);
    double* z = malloc(bytes);
    for (long i = 0; i < 2 * N * H; ++i) x[i] = urand();
    static const int pairs[][2] = {{AB_FFT_NORM_N, AB_FFT_NORM_NONE}, {AB_FFT_NORM_SQRT_N, AB_FFT_NORM_SQRT_N},
                                   {AB_FFT_NORM_NONE, AB_FFT_NORM_N}};
    for (int k = 0; k < 3; ++k) {
        ABStatus st;
        ab_fft_plan_t f = ab_fft_plan_3d(n0, n1, n2, H, AB_FFT_FORWARD, (ab_fft_norm)pairs[k][0], &st);
        ab_fft_plan_t b = ab_fft_plan_3d(n0, n1, n2, H, AB_FFT_BACKWARD, (ab_fft_norm)pairs[k][1], &st);
        CHECK(f && b, "plans");
        if (!f || !b) continue;
        ab_fft_execute(f, x, y);
        ab_fft_execute(b, y, z);
        double dn = 0, xn = 0;
        for (long i = 0; i < 2 * N * H; ++i) { dn += (z[i] - x[i]) * (z[i] - x[i]); xn += x[i] * x[i]; }
        printf("    round trip, norms %d then %d: L2 %.2e\n", pairs[k][0], pairs[k][1], sqrt(dn / xn));
        CHECK(sqrt(dn / xn) < 2 * tol_l2(N), "round trip error %.3e", sqrt(dn / xn));
        ab_fft_destroy(f);
        ab_fft_destroy(b);
    }
    free(x); free(y); free(z);
}

static void test_bad_arguments(void)
{
    ABStatus st = AB_OK;
    const int n8[3] = {8, 8, 8}, n17[1] = {17}, nbig[2] = {4096, 2};
    CHECK(!ab_fft_plan_many(0, n8, 1, NULL, 1, 1, NULL, 1, 1, -1, AB_FFT_NORM_NONE, &st) && st == AB_ERROR_INVALID_ARG,
          "rank 0");
    CHECK(!ab_fft_plan_many(4, n8, 1, NULL, 1, 1, NULL, 1, 1, -1, AB_FFT_NORM_NONE, &st), "rank 4");
    CHECK(!ab_fft_plan_many(1, n17, 1, NULL, 1, 17, NULL, 1, 17, -1, AB_FFT_NORM_NONE, &st) &&
              st == AB_ERROR_INVALID_ARG,
          "prime 17 unsupported");
    CHECK(!ab_fft_plan_many(2, nbig, 1, NULL, 1, 1, NULL, 1, 1, -1, AB_FFT_NORM_NONE, &st), "4096 in a 2D plan");
    CHECK(!ab_fft_plan_1d(2049, 1, AB_FFT_FORWARD, AB_FFT_NORM_NONE, &st) && st == AB_ERROR_INVALID_ARG,
          "n = 2049 = 3 * 683 has no four-step split");
    CHECK(!ab_fft_plan_many(1, n8, 2, NULL, 1, 8, NULL, 1, 8, 0, AB_FFT_NORM_NONE, &st), "sign 0");
    CHECK(!ab_fft_plan_many(1, n8, 2, NULL, 1, 8, NULL, 1, 8, -1, (ab_fft_norm)3, &st), "norm 3");
    CHECK(!ab_fft_plan_many(1, n8, 2, NULL, 0, 8, NULL, 1, 8, -1, AB_FFT_NORM_NONE, &st), "stride 0");
    const int emb_small[3] = {7, 8, 8};
    CHECK(!ab_fft_plan_many(3, n8, 1, emb_small, 1, 1, NULL, 1, 1, -1, AB_FFT_NORM_NONE, &st), "inembed < n");
    CHECK(!ab_fft_plan_many(1, n8, 2, NULL, 1, 8, NULL, 1, 1, -1, AB_FFT_NORM_NONE, &st) &&
              st == AB_ERROR_INVALID_ARG,
          "overlapping output transforms (odist 1, ostride 1)");
    ab_fft_plan_t p = ab_fft_plan_1d(64, 2, AB_FFT_FORWARD, AB_FFT_NORM_NONE, &st);
    CHECK(p != NULL, "plan");
    double* a = calloc(256 + 64, 16);
    CHECK(ab_fft_execute(p, NULL, a) == AB_ERROR_INVALID_ARG, "NULL input");
    CHECK(ab_fft_execute(NULL, a, a) == AB_ERROR_INVALID_ARG, "NULL plan");
    CHECK(ab_fft_execute(p, a, a + 2 * 10) == AB_ERROR_INVALID_ARG, "partially overlapping in/out");
    CHECK(ab_fft_execute(p, a, a) == AB_OK, "in place");
    CHECK(ab_fft_unregister(a) == AB_ERROR_INVALID_ARG, "unregister unknown pointer");
    CHECK(ab_fft_register(NULL, 16) == AB_ERROR_INVALID_ARG, "register NULL");
    char d[256];
    ab_fft_describe(p, d, sizeof d);
    CHECK(strstr(d, "N=64") != NULL, "describe: %s", d);
    free(a);
    ab_fft_destroy(p);
    ab_fft_destroy(NULL);
}

// The exponent range is FP32's (documented in apple_bottom_fft.h).
static void test_range(void)
{
    const int n = 16;
    double x[32], y[32];
    ABStatus st;
    ab_fft_plan_t p = ab_fft_plan_1d(n, 1, AB_FFT_FORWARD, AB_FFT_NORM_NONE, &st);
    CHECK(p != NULL, "plan");
    if (!p) return;
    static const double mags[] = {1e30, 1e-25, 1e-300, 1e300};
    for (int k = 0; k < 4; ++k) {
        memset(x, 0, sizeof x);
        x[0] = mags[k];   // delta: every output equals x[0]
        ab_fft_execute(p, x, y);
        long ok = 0;
        for (int i = 0; i < n; ++i) {
            if (k < 2) ok += fabs(y[2 * i] - mags[k]) <= 4e-15 * mags[k] && y[2 * i + 1] == 0;
            else if (k == 2) ok += y[2 * i] == 0 && y[2 * i + 1] == 0;
            else ok += !isfinite(y[2 * i]);
        }
        CHECK(ok == n, "delta of %.0e: %ld of %d outputs as documented (first %.17g)", mags[k], ok, n, y[0]);
    }
    printf("    1e30 and 1e-25 exact to DD precision; 1e-300 flushes to 0; 1e300 overflows\n");
    ab_fft_destroy(p);
}

typedef struct { ab_fft_plan_t shared; const double* x; double* y; double* y2; } thr_arg;

static void* thr_main(void* v)
{
    thr_arg* a = v;
    ABStatus st;
    ab_fft_plan_t own = ab_fft_plan_3d(32, 24, 20, 2, AB_FFT_FORWARD, AB_FFT_NORM_NONE, &st);
    for (int r = 0; r < 4; ++r) {
        ab_fft_execute(a->shared, a->x, a->y);
        if (own) ab_fft_execute(own, a->x, a->y2);
    }
    ab_fft_destroy(own);
    return NULL;
}

static void test_threads(void)
{
    enum { NT = 6 };
    const long N = 32L * 24 * 20 * 2;
    ABStatus st;
    ab_fft_plan_t shared = ab_fft_plan_3d(32, 24, 20, 2, AB_FFT_FORWARD, AB_FFT_NORM_NONE, &st);
    double* x = malloc((size_t)N * 16);
    for (long i = 0; i < 2 * N; ++i) x[i] = urand();
    double* ref = malloc((size_t)N * 16);
    ab_fft_execute(shared, x, ref);
    pthread_t th[NT];
    thr_arg args[NT];
    for (int t = 0; t < NT; ++t) {
        args[t].shared = shared;
        args[t].x = x;
        args[t].y = malloc((size_t)N * 16);
        args[t].y2 = malloc((size_t)N * 16);
        pthread_create(&th[t], NULL, thr_main, &args[t]);
    }
    long bad = 0;
    for (int t = 0; t < NT; ++t) {
        pthread_join(th[t], NULL);
        bad += memcmp(args[t].y, ref, (size_t)N * 16) != 0;
        bad += memcmp(args[t].y2, ref, (size_t)N * 16) != 0;
        free(args[t].y);
        free(args[t].y2);
    }
    CHECK(bad == 0, "%ld of %d threaded results differ", bad, 2 * NT);
    printf("    %d threads, one shared plan and one plan each: bitwise equal\n", NT);
    free(x); free(ref);
    ab_fft_destroy(shared);
}

static void test_fork(void)
{
    ABStatus st;
    ab_fft_plan_t p = ab_fft_plan_1d(32, 1, AB_FFT_FORWARD, AB_FFT_NORM_NONE, &st);
    double x[64] = {1}, y[64];
    CHECK(p && ab_fft_execute(p, x, y) == AB_OK, "execute before fork");
    fflush(stdout);
    pid_t pid = fork();
    if (pid == 0) {
        // Metal is unusable in the child: an error, not a crash.
        _exit(ab_fft_execute(p, x, y) == AB_ERROR_NO_DEVICE &&
              ab_fft_plan_1d(32, 1, AB_FFT_FORWARD, AB_FFT_NORM_NONE, &st) == NULL ? 0 : 1);
    }
    int status = -1;
    waitpid(pid, &status, 0);
    CHECK(WIFEXITED(status) && WEXITSTATUS(status) == 0, "forked child: status %d", status);
    CHECK(ab_fft_execute(p, x, y) == AB_OK, "parent still works after fork");
    ab_fft_destroy(p);
}

static const TestCase TESTS[] = {
    {"test_reference",     test_reference},
    {"test_conversions",   test_conversions},
    {"test_lengths_1d",    test_lengths_1d},
    {"test_four_step",     test_four_step},
    {"test_multi_dim",     test_multi_dim},
    {"test_layouts",       test_layouts},
    {"test_in_place",      test_in_place},
    {"test_zero_copy",     test_zero_copy},
    {"test_chunks",        test_chunks},
    {"test_normalization", test_normalization},
    {"test_bad_arguments", test_bad_arguments},
    {"test_range",         test_range},
    {"test_threads",       test_threads},
    {"test_fork",          test_fork},
};
static const int N_TESTS = (int)TEST_REGISTRY_SIZE(TESTS);

int main(void)
{
    clear_ab_env();   // AB_FFT_WORK_MB would change the chunk counts checked here
    printf("═══════════════════════════════════════════════════════════\n");
    printf("apple-bottom double-float FFT tests\n");
    printf("═══════════════════════════════════════════════════════════\n");
    for (int i = 0; i < N_TESTS; i++) {
        printf("[%d/%d] %s\n", i + 1, N_TESTS, TESTS[i].name);
        fflush(stdout);
        int pre = checks_run;
        TESTS[i].fn();
        if (checks_run == pre) {
            fprintf(stderr, "\nFATAL: ghost test — '%s' ran zero CHECK invocations\n", TESTS[i].name);
            abort();
        }
    }
    printf("═══════════════════════════════════════════════════════════\n");
    printf("Summary: %d test(s) ran, %d CHECK(s), %d failure(s)\n", N_TESTS, checks_run, failures);
    if (failures) {
        printf("✗ %d FFT check(s) failed\n", failures);
        return 1;
    }
    printf("✓ All FFT tests passed\n");
    return 0;
}
