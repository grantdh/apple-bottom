// =============================================================================
// test_split.c — AB_SPLIT: one BLAS-ABI GEMM split between Accelerate and the GPU
// =============================================================================
// The executor (src/blas_wrapper.c, src/split_slice.h) through its seam,
// ab_internal_dgemm_split and ab_internal_zgemm_split, on the GPU, in this
// process (references made here on the same memory):
//   composition  a call cut at g is, bit for bit, cblas on the head slice
//                (the seam with g = 0 on that slice) and the staged path on
//                the tail slice alone (the seam with g = L on it); on the K
//                axis, the head into C plus the GPU's product of the tail
//                added in FP64
//   accuracy     against cblas (TOL) and, for DGEMM with alpha 1 and beta 0,
//                Dot2 within (3K + 13) u^2 |A||B|; ragged ld, NaN padding,
//                beta 0 over NaN in C, signed zeros, cancellation, cuts at
//                1 and L - 1, a long K on the K axis
//   range        NaN, Inf and scaled lines in the head only or the tail only
//   determinism  repeats, after other shapes, registered and misaligned
//                memory, under background load, 8 threads each splitting
//   registered   in place on the N axis, never C on the M axis
// and dgemm_/zgemm_ under AB_SPLIT in child processes (test_spawn.h) with a
// split table covering the calls of TCALLS: AB_SPLIT unset, empty and 0
// alike, AB_MODE=cpu, every AB_TEST_FAIL_STAGE failure, AB_HOST_PACK,
// AB_STRASSEN, AB_STAGING_MB=0, AB_PRECISION=fast, VECLIB_MAXIMUM_THREADS,
// fork(), the reports (AB_REPORT, AB_PROFILE, its JSON, AB_TRACE), the
// other APIs unchanged, memory growth, and the benchmark drivers' refusals.
//
// Usage: test_split                       the test
//        test_split <test_name>...        those tests only
//        test_split --child <scenario>    one child (AB_SPLIT_TABLE from the parent)
// =============================================================================

#include "apple_bottom.h"
#include "apple_bottom_device.h"
#include "gpu_stage.h"
#include "split_slice.h"
#include "test_blas_check.h"
#include "test_dot2.h"
#include "test_registry.h"
#include "test_spawn.h"
#include <Accelerate/Accelerate.h>
#include <complex.h>
#include <limits.h>
#include <mach-o/dyld.h>
#include <malloc/malloc.h>
#include <math.h>
#include <pthread.h>
#include <stdarg.h>
#include <stdatomic.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/wait.h>
#include <unistd.h>

#define CHILD_TIMEOUT_S 300
#define U2_DD 0x1p-48          // u^2 of the DD arithmetic

#ifndef AB_BUILD_DIR
#define AB_BUILD_DIR "build"
#endif

static int tests_passed = 0, tests_failed = 0;
static char g_why[512];        // the current test's first failure

// Records the first failure of the current test; returns cond.
static bool check(bool cond, const char *fmt, ...) {
    if (!cond && !g_why[0]) {
        va_list ap;
        va_start(ap, fmt);
        vsnprintf(g_why, sizeof g_why, fmt, ap);
        va_end(ap);
    }
    return cond;
}

static void finish(void) {
    if (g_why[0]) {
        printf("✗ FAIL (%s)\n", g_why);
        tests_failed++;
        g_why[0] = 0;
    } else {
        printf("✓ PASS\n");
        tests_passed++;
    }
}

// =============================================================================
// Calls and operands
// =============================================================================

// One call as the BLAS ABI takes it; alpha, beta as (real, imaginary).
typedef struct {
    char r, ta, tb;
    int M, N, K, pad;
    double alpha[2], beta[2];
} Call;

// Its operands, column-major with leading dimension rows + pad, padding NaN.
typedef struct {
    int w;                       // doubles per element
    int ldA, ldB, ldC, rA, cA, rB, cB;
    size_t nA, nB, nC;           // doubles
    double *A, *B, *C0;
} Operands;

static uint64_t g_seed = 0x2545F4914F6CDD1DULL;

static double uniform(void) {
    uint64_t z = (g_seed += 0x9E3779B97F4A7C15ULL);
    z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ULL;
    z = (z ^ (z >> 27)) * 0x94D049BB133111EBULL;
    return ((double)((z ^ (z >> 31)) >> 11) * 0x1p-53) * 2.0 - 1.0;
}

static double *stored_operand(int w, int rows, int cols, int ld, size_t *n) {
    *n = (size_t)w * ld * (cols ? cols : 1);
    double *x = malloc(*n * sizeof *x);
    for (size_t i = 0; i < *n; i++) x[i] = NAN;
    for (int j = 0; j < cols; j++)
        for (int i = 0; i < w * rows; i++) x[(size_t)j * w * ld + i] = uniform();
    return x;
}

static Operands operands(const Call *c) {
    Operands o = {.w = c->r == 'z' ? 2 : 1};
    bool na = is_notrans(c->ta), nb = is_notrans(c->tb);
    o.rA = na ? c->M : c->K;
    o.cA = na ? c->K : c->M;
    o.rB = nb ? c->K : c->N;
    o.cB = nb ? c->N : c->K;
    o.ldA = o.rA + c->pad;
    o.ldB = o.rB + c->pad;
    o.ldC = c->M + c->pad;
    o.A = stored_operand(o.w, o.rA, o.cA, o.ldA, &o.nA);
    o.B = stored_operand(o.w, o.rB, o.cB, o.ldB, &o.nB);
    o.C0 = stored_operand(o.w, c->M, c->N, o.ldC, &o.nC);
    return o;
}

static void free_operands(Operands *o) {
    free(o->A);
    free(o->B);
    free(o->C0);
}

// The seam on the call c with A, B and C at the given element offsets.
static int seam(const Call *c, char axis, int g, int order, int M, int N, int K, const Operands *o,
                size_t a, size_t b, double *C, int ldC) {
    ab_split_forced_t f = {axis, g, order, 0};
    if (c->r == 'z')
        return ab_internal_zgemm_split(&f, c->ta, c->tb, M, N, K, CMPLX(c->alpha[0], c->alpha[1]),
                                       (const double complex *)o->A + a, o->ldA,
                                       (const double complex *)o->B + b, o->ldB,
                                       CMPLX(c->beta[0], c->beta[1]), (double complex *)C, ldC);
    return ab_internal_dgemm_split(&f, c->ta, c->tb, M, N, K, c->alpha[0], o->A + a, o->ldA,
                                   o->B + b, o->ldB, c->beta[0], C, ldC);
}

static bool kept(int outcome) { return outcome == AB_SPLIT_KEPT || outcome == AB_SPLIT_KEPT_4M; }

static const char *call_text(const Call *c, char axis, int g) {
    static char s[160];
    snprintf(s, sizeof s, "%cgemm %c %c %dx%dx%d pad %d alpha %g%+gi beta %g%+gi axis %c g=%d", c->r,
             c->ta, c->tb, c->M, c->N, c->K, c->pad, c->alpha[0], c->alpha[1], c->beta[0], c->beta[1],
             axis, g);
    return s;
}

// The composition reference of c cut at g on axis, into ref (C0 on entry):
// the head slice by the seam at g = 0, the tail slice alone by the seam at
// its own full length (the staged path); on K, the tail's product by the
// staged path into P (beta 0, ld M), added in FP64. False if a part did not
// come out as it should (the tail must stand on the GPU).
static bool compose_reference(const Call *c, char axis, int g, const Operands *o, double *ref) {
    const size_t w = (size_t)o->w;
    const bool na = is_notrans(c->ta), nb = is_notrans(c->tb);
    int got_h, got_t;
    if (axis == 'N') {
        size_t cut = (size_t)(c->N - g);
        got_h = cut ? seam(c, 'N', 0, 0, c->M, (int)cut, c->K, o, 0, 0, ref, o->ldC) : AB_SPLIT_CPU;
        got_t = seam(c, 'N', g, 0, c->M, g, c->K, o, 0, nb ? cut * o->ldB : cut, ref + w * cut * o->ldC,
                     o->ldC);
    } else if (axis == 'M') {
        size_t cut = (size_t)(c->M - g);
        got_h = cut ? seam(c, 'M', 0, 0, (int)cut, c->N, c->K, o, 0, 0, ref, o->ldC) : AB_SPLIT_CPU;
        got_t = seam(c, 'M', g, 0, g, c->N, c->K, o, na ? cut : cut * o->ldA, 0, ref + w * cut, o->ldC);
    } else {
        size_t cut = (size_t)(c->K - g);
        got_h = seam(c, 'K', 0, 0, c->M, c->N, (int)cut, o, 0, 0, ref, o->ldC);
        Call t = *c;
        t.beta[0] = t.beta[1] = 0.0;
        double *P = calloc(w * (size_t)c->M * c->N, sizeof *P);
        got_t = seam(&t, 'N', c->N, 0, c->M, c->N, g, o, na ? cut * o->ldA : cut, nb ? cut : cut * o->ldB,
                     P, c->M);
        for (int j = 0; j < c->N; j++)
            for (size_t i = 0; i < w * (size_t)c->M; i++) ref[w * (size_t)j * o->ldC + i] += P[w * (size_t)j * c->M + i];
        free(P);
    }
    return check(got_h == AB_SPLIT_CPU && kept(got_t), "%s: reference parts gave %d, %d", call_text(c, axis, g),
                 got_h, got_t);
}

// c cut at g on axis, in each order, against its composition reference: C
// whole (padding included) bitwise, A and B unchanged, the GPU's part kept.
static void compose_case(const Call *c, char axis, int g) {
    Operands o = operands(c);
    double *ref = malloc(o.nC * sizeof *ref), *C = malloc(o.nC * sizeof *C);
    double *A0 = malloc(o.nA * sizeof *A0), *B0 = malloc(o.nB * sizeof *B0);
    memcpy(A0, o.A, o.nA * sizeof *A0);
    memcpy(B0, o.B, o.nB * sizeof *B0);
    memcpy(ref, o.C0, o.nC * sizeof *ref);
    if (compose_reference(c, axis, g, &o, ref)) {
        for (int order = 0; order < 3; order++) {
            memcpy(C, o.C0, o.nC * sizeof *C);
            int got = seam(c, axis, g, order, c->M, c->N, c->K, &o, 0, 0, C, o.ldC);
            ab_split_record_t rec;
            check(kept(got) && ab_internal_last_split(&rec) && rec.outcome == got && rec.axis == axis &&
                  rec.g == g, "%s order %d: outcome %d", call_text(c, axis, g), order, got);
            check(!memcmp(C, ref, o.nC * sizeof *C), "%s order %d: C is not the composition of its parts",
                  call_text(c, axis, g), order);
            check(!strcmp(ab_get_last_dispatch_path(), "split"), "%s: last path %s", call_text(c, axis, g),
                  ab_get_last_dispatch_path());
        }
        check(!memcmp(o.A, A0, o.nA * sizeof *A0) && !memcmp(o.B, B0, o.nB * sizeof *B0),
              "%s: A or B changed", call_text(c, axis, g));
    }
    free(ref); free(C); free(A0); free(B0);
    free_operands(&o);
}

// =============================================================================
// Tests
// =============================================================================

// The axes this tree's executor cuts.
static const char AXES[] = "NMK";

#define ONE {1, 0}
#define ZERO {0, 0}

// T4: the composition on every axis, routine and TRANS pair, scalar set and
// a range of cuts, with ragged leading dimensions.
static void test_split_composition(void) {
    static const struct { double alpha[2], beta[2]; } S[] = {
        {ONE, ZERO}, {ONE, ONE}, {{-1.7, 0.6}, {0.3, -0.45}}, {{2.0, 0.0}, ZERO},
    };
    static const char TR[] = "NTC";
    for (const char *axis = AXES; *axis; axis++)
        for (int z = 0; z < 2; z++)
            for (int a = 0; a < (z ? 3 : 2); a++)
                for (int b = 0; b < (z ? 3 : 2); b++) {
                    int M = 300, N = 260, K = 200;
                    if (*axis == 'K') K = 2100;
                    int L = *axis == 'N' ? N : *axis == 'M' ? M : K, unit = *axis == 'K' ? 64 : 32;
                    const int G[] = {unit, (L / 2 / unit) * unit, L - unit, 1, L - 1};
                    for (size_t s = 0; s < TEST_REGISTRY_SIZE(S); s++)
                        for (size_t gi = 0; gi < TEST_REGISTRY_SIZE(G); gi++) {
                            Call c = {z ? 'z' : 'd', TR[a], TR[b], M, N, K, (int)((s + gi) % 3),
                                      {S[s].alpha[0], S[s].alpha[1]}, {S[s].beta[0], S[s].beta[1]}};
                            compose_case(&c, *axis, G[gi]);
                        }
                }
    // Quantum ESPRESSO's shapes, smaller: updates, narrow updates, reductions.
    const Call QE[] = {
        {'z', 'N', 'N', 9000, 256, 160, 0, ONE, ZERO},
        {'z', 'N', 'N', 9000, 160, 500, 3, ONE, ONE},
        {'z', 'C', 'N', 320, 320, 9000, 1, ONE, ZERO},
        {'d', 'N', 'N', 2048, 2048, 2048, 0, ONE, ZERO},
    };
    for (size_t i = 0; i < TEST_REGISTRY_SIZE(QE); i++)
        for (const char *axis = AXES; *axis; axis++) {
            int L = *axis == 'N' ? QE[i].N : *axis == 'M' ? QE[i].M : QE[i].K;
            int unit = *axis == 'K' ? 64 : 32;
            compose_case(&QE[i], *axis, (L / 2 / unit) * unit);
        }
    finish();
}

// The parts of c cut at g on axis made one after the other by cblas, into C
// (C0 on entry): what a GPU part that did not stand leaves. K: the head into
// C, then the tail with beta 1.
static enum CBLAS_TRANSPOSE cblas_trans(char t) {
    return t == 'N' ? CblasNoTrans : t == 'T' ? CblasTrans : CblasConjTrans;
}

static void cblas_call(const Call *c, int M, int N, int K, const double *A, int ldA, const double *B,
                       int ldB, const double *beta, double *C, int ldC) {
    if (M == 0 || N == 0) return;
    if (c->r == 'z')
        cblas_zgemm(CblasColMajor, cblas_trans(c->ta), cblas_trans(c->tb), M, N, K,
                    (const void *)c->alpha, (const void *)A, ldA, (const void *)B, ldB,
                    (const void *)beta, (void *)C, ldC);
    else
        cblas_dgemm(CblasColMajor, cblas_trans(c->ta), cblas_trans(c->tb), M, N, K, c->alpha[0], A, ldA,
                    B, ldB, beta[0], C, ldC);
}

static void parts_reference(const Call *c, char axis, int g, const Operands *o, double *C) {
    const size_t w = (size_t)o->w;
    const bool na = is_notrans(c->ta), nb = is_notrans(c->tb);
    static const double one[2] = {1, 0};
    if (axis == 'N') {
        size_t cut = (size_t)(c->N - g);
        cblas_call(c, c->M, (int)cut, c->K, o->A, o->ldA, o->B, o->ldB, c->beta, C, o->ldC);
        cblas_call(c, c->M, g, c->K, o->A, o->ldA, o->B + w * (nb ? cut * o->ldB : cut), o->ldB, c->beta,
                   C + w * cut * o->ldC, o->ldC);
    } else if (axis == 'M') {
        size_t cut = (size_t)(c->M - g);
        cblas_call(c, (int)cut, c->N, c->K, o->A, o->ldA, o->B, o->ldB, c->beta, C, o->ldC);
        cblas_call(c, g, c->N, c->K, o->A + w * (na ? cut : cut * o->ldA), o->ldA, o->B, o->ldB, c->beta,
                   C + w * cut, o->ldC);
    } else {
        size_t cut = (size_t)(c->K - g);
        cblas_call(c, c->M, c->N, (int)cut, o->A, o->ldA, o->B, o->ldB, c->beta, C, o->ldC);
        cblas_call(c, c->M, c->N, g, o->A + w * (na ? cut * o->ldA : cut), o->ldA,
                   o->B + w * (nb ? cut : cut * o->ldB), o->ldB, one, C, o->ldC);
    }
}

static uint64_t hash_doubles(const double *x, size_t n) { return fnv1a(x, n * sizeof *x, FNV_OFFSET); }

// n doubles in whole pages of their own, for ab_register_host_memory; *bytes
// is the size to register.
static double *page_alloc(size_t n, size_t *bytes) {
    void *p = NULL;
    *bytes = (n * sizeof(double) + 16383) & ~(size_t)16383;
    if (posix_memalign(&p, 16384, *bytes)) abort();
    return p;
}

static bool all_finite(const Call *c, const Operands *o, const double *C) {
    for (int j = 0; j < c->N; j++)
        for (int i = 0; i < o->w * c->M; i++)
            if (!isfinite(C[(size_t)j * o->w * o->ldC + i])) return false;
    return true;
}

// T5: against cblas within TOL, and DGEMM (alpha 1, beta 0) against Dot2
// within the DD GEMM bound (3K + 13) u^2 |A||B| (docs/vv/PRECISION_ENVELOPE.md);
// padding untouched; beta = 0 never reads C.
static void accuracy_case(const Call *c, char axis, int g, bool nan_c) {
    Operands o = operands(c);
    if (nan_c)
        for (int j = 0; j < c->N; j++)
            for (int i = 0; i < o.w * c->M; i++) o.C0[(size_t)j * o.w * o.ldC + i] = NAN;
    double *C = malloc(o.nC * sizeof *C);
    memcpy(C, o.C0, o.nC * sizeof *C);
    int got = seam(c, axis, g, 0, c->M, c->N, c->K, &o, 0, 0, C, o.ldC);
    Layout la = {o.rA, o.cA, o.ldA}, lb = {o.rB, o.cB, o.ldB}, lc = {c->M, c->N, o.ldC};
    double complex al = CMPLX(c->alpha[0], c->alpha[1]), be = CMPLX(c->beta[0], c->beta[1]);
    double err = blas_error(c->r == 'z', c->ta, c->tb, c->M, c->N, c->K, al, be, o.A, la, o.B, lb, o.C0,
                            lc, C);
    check(kept(got), "%s: outcome %d", call_text(c, axis, g), got);
    check(err <= TOL, "%s: error %.2e against cblas", call_text(c, axis, g), err);
    check(outside_block_unchanged(C, o.C0, o.nC / o.w, lc, o.w * sizeof(double)),
          "%s: C outside its M x N block changed", call_text(c, axis, g));
    if (nan_c) check(all_finite(c, &o, C), "%s: beta = 0 read the NaN in C", call_text(c, axis, g));
    if (c->r == 'd' && c->alpha[0] == 1.0 && c->beta[0] == 0.0) {
        // op(A), op(B) row-major for Dot2
        double *a = malloc((size_t)c->M * c->K * sizeof *a), *b = malloc((size_t)c->K * c->N * sizeof *b);
        for (int i = 0; i < c->M; i++)
            for (int k = 0; k < c->K; k++) a[(size_t)i * c->K + k] = o.A[op_index(c->ta, la, i, k)];
        for (int k = 0; k < c->K; k++)
            for (int j = 0; j < c->N; j++) b[(size_t)k * c->N + j] = o.B[op_index(c->tb, lb, k, j)];
        double *S = calloc((size_t)c->M * c->N, sizeof *S), *E = calloc((size_t)c->M * c->N, sizeof *E);
        double *L = calloc((size_t)c->M * c->N, sizeof *L);
        dot2_gemm_acc(c->M, c->N, c->K, 1.0, a, b, S, E);
        abs_gemm_acc(c->M, c->N, c->K, a, NULL, b, NULL, L);
        double worst = 0;
        for (int i = 0; i < c->M; i++)
            for (int j = 0; j < c->N; j++) {
                size_t r = (size_t)i * c->N + j;
                double d = fabs(C[(size_t)j * o.ldC + i] - (S[r] + E[r]));
                worst = fmax(worst, L[r] > 0 ? d / L[r] : d);
            }
        check(worst <= (3.0 * c->K + 13.0) * U2_DD, "%s: error %.2f u^2 |A||B| against Dot2, bound %.0f",
              call_text(c, axis, g), worst / U2_DD, 3.0 * c->K + 13.0);
        free(a); free(b); free(S); free(E); free(L);
    }
    free(C);
    free_operands(&o);
}

static void test_split_accuracy(void) {
    static const char TR[] = "NTC";
    for (const char *axis = AXES; *axis; axis++)
        for (int z = 0; z < 2; z++)
            for (int a = 0; a < (z ? 3 : 2); a++)
                for (int b = 0; b < (z ? 3 : 2); b++) {
                    // tall, wide, and long K on K
                    const int SH[2][3] = {{*axis == 'K' ? 96 : 700, 160, *axis == 'K' ? 3000 : 120},
                                          {*axis == 'K' ? 64 : 140, *axis == 'K' ? 80 : 500,
                                           *axis == 'K' ? 1300 : 90}};
                    for (int s = 0; s < 2; s++) {
                        int M = SH[s][0], N = SH[s][1], K = SH[s][2];
                        int L = *axis == 'N' ? N : *axis == 'M' ? M : K;
                        Call c = {z ? 'z' : 'd', TR[a], TR[b], M, N, K, 1 + s, ONE, ZERO};
                        accuracy_case(&c, *axis, L / 2, false);
                        accuracy_case(&c, *axis, 1, false);                  // g = 1
                        accuracy_case(&c, *axis, L - 1, false);              // cut = 1
                        c.alpha[0] = -0.75; c.alpha[1] = z ? 0.5 : 0.0;
                        c.beta[0] = 1.25; c.beta[1] = z ? -0.25 : 0.0;
                        accuracy_case(&c, *axis, L / 3, false);
                        c.beta[0] = c.beta[1] = 0.0;
                        accuracy_case(&c, *axis, L / 3, true);               // beta 0, NaN in C
                    }
                }
    // Signed zeros and cancellation: C0 = -op(A) op(B) by cblas, beta 1.
    for (const char *axis = AXES; *axis; axis++)
        for (int z = 0; z < 2; z++) {
            Call c = {z ? 'z' : 'd', 'N', 'N', 300, 260, *axis == 'K' ? 2100 : 200, 2, ONE, ONE};
            Operands o = operands(&c);
            static const double m1[2] = {-1, 0}, zero[2] = {0, 0};
            Call neg = c;
            memcpy(neg.alpha, m1, sizeof m1);
            cblas_call(&neg, c.M, c.N, c.K, o.A, o.ldA, o.B, o.ldB, zero, o.C0, o.ldC);
            for (int i = 0; i < o.w * c.M; i += 7) o.C0[i] = -0.0;      // and signed zeros
            double *C = malloc(o.nC * sizeof *C);
            memcpy(C, o.C0, o.nC * sizeof *C);
            int L = *axis == 'N' ? c.N : *axis == 'M' ? c.M : c.K;
            int got = seam(&c, *axis, L / 2, 0, c.M, c.N, c.K, &o, 0, 0, C, o.ldC);
            Layout la = {o.rA, o.cA, o.ldA}, lb = {o.rB, o.cB, o.ldB}, lc = {c.M, c.N, o.ldC};
            double err = blas_error(z, 'N', 'N', c.M, c.N, c.K, 1.0, 1.0, o.A, la, o.B, lb, o.C0, lc, C);
            check(kept(got) && err <= TOL, "cancellation %c axis %c: outcome %d error %.2e", c.r, *axis, got,
                  err);
            free(C);
            free_operands(&o);
        }
    // Quantum ESPRESSO's long K on the K axis: newq 'T','N' 171 x 64 x 143666.
    Call qe = {'d', 'T', 'N', 171, 64, 143666, 0, ONE, ZERO};
    accuracy_case(&qe, 'K', (143666 / 2 / 64) * 64, false);
    finish();
}

// A copy of c's operands with values set in one slice: NaN, Inf or a scale.
typedef enum { PLANT_NAN, PLANT_INF, PLANT_SCALE } Plant;

// Plants v in row (M axis) or column (N axis) or K index l of op(A) / op(B):
// N axis: column l of op(B); M axis: row l of op(A); K axis: column l of
// op(A) and row l of op(B).
static void plant(const Call *c, Operands *o, char axis, int l, Plant what) {
    Layout la = {o->rA, o->cA, o->ldA}, lb = {o->rB, o->cB, o->ldB};
    double v = what == PLANT_NAN ? NAN : what == PLANT_INF ? INFINITY : 0;
    for (int t = 0; t < (axis == 'N' ? c->K : axis == 'M' ? c->K : c->M); t++) {
        size_t x = axis == 'N' ? op_index(c->tb, lb, t, l) : op_index(c->ta, la, axis == 'M' ? l : t, axis == 'M' ? t : l);
        double *X = axis == 'N' ? o->B : o->A;
        for (int p = 0; p < o->w; p++) {
            double *e = &X[(size_t)o->w * x + p];
            *e = what == PLANT_SCALE ? *e * 0x1p100 : (t == 3 && p == 0 ? v : *e);
        }
    }
}

static char *run_child(const char *scenario, const char *const *extra, int *status);

// T6: bad data in the head only, or the tail only, on each axis.
static void test_split_exponent_range(void) {
    for (const char *axis = AXES; *axis; axis++)
        for (int z = 0; z < 2; z++)
            for (Plant what = PLANT_NAN; what <= PLANT_SCALE; what++)
                for (int tail_side = 0; tail_side < 2; tail_side++) {
                    // On K the planted index is a column of op(A), not one of the guard's lines.
                    if (*axis == 'K' && what == PLANT_SCALE) continue;
                    // K: a product small enough that malloc hands its buffer on, so a P
                    // added where it should not be holds an earlier call's values.
                    bool k = *axis == 'K';
                    Call c = {z ? 'z' : 'd', 'N', 'N', k ? 40 : 300, k ? 40 : 260, k ? 2100 : 200, 1, ONE, ZERO};
                    int L = *axis == 'N' ? c.N : *axis == 'M' ? c.M : c.K;
                    int g = *axis == 'K' ? 1024 : 128;
                    Operands o = operands(&c);
                    plant(&c, &o, *axis, tail_side ? L - 5 : 5, what);
                    double *C = malloc(o.nC * sizeof *C), *ref = malloc(o.nC * sizeof *ref);
                    memcpy(C, o.C0, o.nC * sizeof *C);
                    memcpy(ref, o.C0, o.nC * sizeof *ref);
                    int got = seam(&c, *axis, g, 0, c.M, c.N, c.K, &o, 0, 0, C, o.ldC);
                    const char *text = call_text(&c, *axis, g);
                    if (what != PLANT_SCALE && tail_side) {
                        // the tail goes to Accelerate: the cblas parts, P not added on K
                        parts_reference(&c, *axis, g, &o, ref);
                        // range, whether the guard or the GPU epilogue met it first
                        check(got == AB_SPLIT_RANGE, "%s, NaN/Inf in the tail: outcome %d", text, got);
                    } else {
                        // the head on cblas, the tail (scaled lines included) on the GPU
                        compose_reference(&c, *axis, g, &o, ref);
                        check(kept(got), "%s, %s in the %s: outcome %d", text,
                              what == PLANT_SCALE ? "scaled line" : "NaN/Inf", tail_side ? "tail" : "head", got);
                    }
                    // NaN payloads aside, the bits: compare as values where both are NaN.
                    bool same = true;
                    for (size_t i = 0; i < o.nC; i++)
                        same = same && (C[i] == ref[i] || (isnan(C[i]) && isnan(ref[i])) ||
                                        !memcmp(&C[i], &ref[i], sizeof C[i]));
                    check(same, "%s, plant %d %s: C is not the expected composition", text, what,
                          tail_side ? "tail" : "head");
                    free(C); free(ref);
                    free_operands(&o);
                }
    // A ZGEMM tail that reads NaN in C goes to Accelerate as out of range:
    // with alpha 2 (the host epilogue) unrun, with alpha 1, beta 1 after the
    // GPU epilogue meets it, as without AB_SPLIT.
    for (int gpu_epi = 0; gpu_epi < 2; gpu_epi++) {
        Call c = {'z', 'N', 'N', 300, 260, 200, 0, {gpu_epi ? 1 : 2, 0}, {gpu_epi ? 1 : 0.5, 0}};
        Operands o = operands(&c);
        o.C0[(size_t)2 * o.ldC * 250 + 4] = NAN;              // column 250: in the tail
        double *C = malloc(o.nC * sizeof *C), *ref = malloc(o.nC * sizeof *ref);
        memcpy(C, o.C0, o.nC * sizeof *C);
        memcpy(ref, o.C0, o.nC * sizeof *ref);
        int got = seam(&c, 'N', 64, 0, c.M, c.N, c.K, &o, 0, 0, C, o.ldC);
        parts_reference(&c, 'N', 64, &o, ref);
        ab_split_record_t rec;
        check(got == AB_SPLIT_RANGE && ab_internal_last_split(&rec) && rec.staged == 8 /* RANGE */,
              "ZGEMM alpha %g beta %g with NaN in the tail's C: outcome %d", c.alpha[0], c.beta[0], got);
        check(!memcmp(C, ref, o.nC * sizeof *C), "ZGEMM with NaN in the tail's C: C is not the cblas parts");
        free(C); free(ref);
        free_operands(&o);
    }
    // A K tail refused after its run: nothing of P reaches C, though P holds
    // 1.0 throughout (AB_TEST_FAIL_STAGE=split_poison, in a child).
    const char *poison[] = {"AB_TEST_FAIL_STAGE=split_poison", NULL};
    int status;
    char *kout = run_child("k_refused", poison, &status);
    check(status == 0 && kout && strstr(kout, "k_refused d 3 1") && strstr(kout, "k_refused z 3 1"),
          "a refused K tail with a poisoned P: %.200s", kout ? kout : "");
    free(kout);
    // Tails refused after their run give their buffers back: the pool keeps
    // its size over repeats.
    Call r = {'d', 'N', 'N', 300, 260, 200, 0, ONE, ZERO};
    Operands ro = operands(&r);
    plant(&r, &ro, 'N', 255, PLANT_NAN);
    double *Cr = malloc(ro.nC * sizeof *Cr);
    size_t bytes = 0;
    for (int k = 0; k < 6; k++) {
        memcpy(Cr, ro.C0, ro.nC * sizeof *Cr);
        int out = seam(&r, 'N', 64, 0, r.M, r.N, r.K, &ro, 0, 0, Cr, ro.ldC);
        check(out == AB_SPLIT_RANGE, "refused tail %d: outcome %d", k, out);
        if (k == 1) bytes = ab_internal_stage_pool_bytes();
        if (k > 1) check(ab_internal_stage_pool_bytes() == bytes, "the pool grew to %zu bytes over refused tails",
                         ab_internal_stage_pool_bytes());
    }
    free(Cr);
    free_operands(&ro);
    finish();
}

// One reference hash of c cut at g, and the split's own, nruns times.
static uint64_t split_hash(const Call *c, char axis, int g, const Operands *o, double *C, int order) {
    memcpy(C, o->C0, o->nC * sizeof *C);
    int got = seam(c, axis, g, order, c->M, c->N, c->K, o, 0, 0, C, o->ldC);
    return kept(got) ? hash_doubles(C, o->nC) : 0;
}

typedef struct {
    const Call *c;
    const Operands *o;
    char axis;
    int g;
    uint64_t hash;
    atomic_int *go;
} SplitThread;

static void *split_thread(void *arg) {
    SplitThread *t = arg;
    double *C = malloc(t->o->nC * sizeof *C);
    while (!atomic_load(t->go)) {}
    t->hash = split_hash(t->c, t->axis, t->g, t->o, C, 0);
    free(C);
    return NULL;
}

static atomic_int g_load_stop;

static void *busy_thread(void *arg) {
    volatile double x = 1.0;
    while (!atomic_load(&g_load_stop)) x = x * 1.0000001 + 1e-9;
    (void)arg;
    return NULL;
}

// T7 (in this process): the bits repeat whatever ran before, on registered
// and misaligned memory, under load, and from 8 threads splitting at once.
static void test_split_determinism(void) {
    const Call QE[] = {
        {'z', 'N', 'N', 9000, 256, 160, 0, ONE, ZERO},
        {'z', 'N', 'N', 9000, 160, 500, 1, ONE, ONE},
        {'z', 'C', 'N', 320, 320, 9000, 0, ONE, ZERO},
    };
    const char AX[] = "NMK";
    const int G[] = {128, 4672, 4544};
    for (size_t i = 0; i < 3; i++) {
        const Call *c = &QE[i];
        Operands o = operands(c);
        double *C = malloc(o.nC * sizeof *C);
        uint64_t h0 = split_hash(c, AX[i], G[i], &o, C, 0);
        check(h0 != 0, "%s: not kept", call_text(c, AX[i], G[i]));
        for (int r = 0; r < 3; r++)
            check(split_hash(c, AX[i], G[i], &o, C, r % 3) == h0, "%s: repeat %d differs", call_text(c, AX[i], G[i]), r);
        // after other shapes, larger and smaller
        Call other = {'z', 'N', 'N', 2000, 700, 300, 0, ONE, ZERO};
        Operands oo = operands(&other);
        double *Co = malloc(oo.nC * sizeof *Co);
        split_hash(&other, 'N', 320, &oo, Co, 0);
        free(Co);
        free_operands(&oo);
        check(split_hash(c, AX[i], G[i], &o, C, 0) == h0, "%s: differs after another shape", call_text(c, AX[i], G[i]));
        // under background load
        atomic_store(&g_load_stop, 0);
        pthread_t busy[8];
        for (int b = 0; b < 8; b++) pthread_create(&busy[b], NULL, busy_thread, NULL);
        check(split_hash(c, AX[i], G[i], &o, C, 0) == h0, "%s: differs under load", call_text(c, AX[i], G[i]));
        atomic_store(&g_load_stop, 1);
        for (int b = 0; b < 8; b++) pthread_join(busy[b], NULL);
        // 8 threads each splitting the same call into its own C
        atomic_int go = 0;
        SplitThread t[8];
        pthread_t tid[8];
        for (int k = 0; k < 8; k++) {
            t[k] = (SplitThread){c, &o, AX[i], G[i], 0, &go};
            pthread_create(&tid[k], NULL, split_thread, &t[k]);
        }
        atomic_store(&go, 1);
        for (int k = 0; k < 8; k++) {
            pthread_join(tid[k], NULL);
            check(t[k].hash == h0, "%s: thread %d of 8 differs", call_text(c, AX[i], G[i]), k);
        }
        // registered operands, then misaligned ones (complex arrays 8 bytes off:
        // staged), each against the same memory unregistered: Accelerate's head
        // depends on where its operands lie, so the bits repeat per address.
        for (int misalign = 0; misalign < 2; misalign++) {
            size_t off = misalign ? 1 : 0, bA, bB, bC;
            double *A = page_alloc(o.nA + 2, &bA), *B = page_alloc(o.nB + 2, &bB), *Cr = page_alloc(o.nC + 2, &bC);
            memcpy(A + off, o.A, o.nA * sizeof *A);
            memcpy(B + off, o.B, o.nB * sizeof *B);
            Operands r = o;
            r.A = A + off;
            r.B = B + off;
            r.C0 = o.C0;
            uint64_t here = split_hash(c, AX[i], G[i], &r, Cr + off, 0);
            bool reg = ab_register_host_memory(A, bA) == AB_OK && ab_register_host_memory(B, bB) == AB_OK &&
                       ab_register_host_memory(Cr, bC) == AB_OK;
            check(reg && here != 0, "registering the operands");
            check(split_hash(c, AX[i], G[i], &r, Cr + off, 0) == here, "%s: differs on %s registered memory",
                  call_text(c, AX[i], G[i]), misalign ? "misaligned" : "aligned");
            ab_unregister_host_memory(A);
            ab_unregister_host_memory(B);
            ab_unregister_host_memory(Cr);
            free(A); free(B); free(Cr);
        }
        free(C);
        free_operands(&o);
    }
    finish();
}

// T12: on registered memory the N axis's tail uses A, B and C in place
// (DGEMM, beta 0), the M axis's A and B but never C.
static void test_split_registered_c(void) {
    Call c = {'d', 'N', 'N', 1000, 400, 300, 0, ONE, ZERO};
    Operands o = operands(&c);
    size_t bA, bB, bC;
    double *A = page_alloc(o.nA, &bA), *B = page_alloc(o.nB, &bB), *C = page_alloc(o.nC, &bC);
    memcpy(A, o.A, o.nA * sizeof *A);
    memcpy(B, o.B, o.nB * sizeof *B);
    free(o.A);
    free(o.B);
    o.A = A;
    o.B = B;
    bool reg = ab_register_host_memory(A, bA) == AB_OK && ab_register_host_memory(B, bB) == AB_OK &&
               ab_register_host_memory(C, bC) == AB_OK;
    check(reg, "registering the operands");
    for (int m = 0; m < 2; m++) {
        char axis = m ? 'M' : 'N';
        memcpy(C, o.C0, o.nC * sizeof *C);
        int got = seam(&c, axis, 128, 0, c.M, c.N, c.K, &o, 0, 0, C, o.ldC);
        ab_split_record_t rec;
        int in_place = -1;
        bool staged = ab_internal_last_staged(&in_place);
        check(kept(got) && ab_internal_last_split(&rec) && staged && rec.in_place == in_place &&
              in_place == (m ? 2 : 3), "axis %c: outcome %d, %d operands in place (expected %d)", axis, got,
              in_place, m ? 2 : 3);
    }
    ab_unregister_host_memory(o.A);
    ab_unregister_host_memory(o.B);
    ab_unregister_host_memory(C);
    free(C);
    free_operands(&o);
    finish();
}

// T8 (in this process): a caller that flushes subnormals is declined, its
// call made whole on cblas.
static void test_split_caller_fpcr(void) {
    Call c = {'d', 'N', 'N', 300, 260, 200, 0, ONE, ZERO};
    Operands o = operands(&c);
    double *C = malloc(o.nC * sizeof *C), *ref = malloc(o.nC * sizeof *ref);
    memcpy(C, o.C0, o.nC * sizeof *C);
    memcpy(ref, o.C0, o.nC * sizeof *ref);
    cblas_call(&c, c.M, c.N, c.K, o.A, o.ldA, o.B, o.ldB, c.beta, ref, o.ldC);
    int got = -1;
#if defined(__aarch64__)
    uint64_t fpcr, fz;
    __asm__ volatile("mrs %0, fpcr" : "=r"(fpcr));
    fz = fpcr | 1ULL << 24;
    __asm__ volatile("msr fpcr, %0" : : "r"(fz));
    got = seam(&c, 'N', 128, 0, c.M, c.N, c.K, &o, 0, 0, C, o.ldC);
    __asm__ volatile("msr fpcr, %0" : : "r"(fpcr));
    ab_split_record_t rec;
    check(got == AB_SPLIT_DECLINED && ab_internal_last_split(&rec) && rec.declined == 3 /* fpcr */,
          "FZ caller: outcome %d", got);
    // cblas under FZ: compare with a reference under FZ too
    __asm__ volatile("msr fpcr, %0" : : "r"(fz));
    memcpy(ref, o.C0, o.nC * sizeof *ref);
    cblas_call(&c, c.M, c.N, c.K, o.A, o.ldA, o.B, o.ldB, c.beta, ref, o.ldC);
    __asm__ volatile("msr fpcr, %0" : : "r"(fpcr));
    check(!memcmp(C, ref, o.nC * sizeof *C), "FZ caller: C is not cblas's whole call");
#endif
    (void)got;
    free(C); free(ref);
    free_operands(&o);
    finish();
}

// =============================================================================
// dgemm_ and zgemm_ under AB_SPLIT, in child processes
// =============================================================================

static char self_path[PATH_MAX], g_dir[PATH_MAX], g_table[PATH_MAX + 32];

// The split table the children run with, and the calls they make: row, axis
// and cut as the plan gives them (-1: no row).
static const char TABLE[] =
    "# AB_SPLIT table for tests/test_split.c\n"
    "# library: none\n"
    "zgemm N N N m=1024: n=96: scalars=u f=0.5\n"      // 0
    "zgemm N N M m=1024: n=33:95 scalars=u f=0.5\n"    // 1
    "zgemm C N K k=2048: scalars=z f=0.5\n"            // 2
    "dgemm N N N m=512: f=0.5\n"                       // 3
    "dgemm T N K k=4096: f=0.5\n"                      // 4
    "zgemm N T N m=200: f=0\n";                        // 5: attributed, never cut

typedef struct {
    Call c;
    int row;
    char axis;
    int g;
    double imag;         // imaginary parts scaled by this (4M below 1)
} TableCall;

static const TableCall TCALLS[] = {
    {{'z', 'N', 'N', 1200, 128, 64, 1, ONE, ZERO}, 0, 'N', 64, 1.0},
    {{'z', 'N', 'N', 1100, 64, 100, 2, ONE, ONE}, 1, 'M', 544, 1.0},
    {{'z', 'C', 'N', 100, 90, 3000, 0, ONE, ZERO}, 2, 'K', 1472, 1.0},
    {{'d', 'N', 'N', 600, 300, 200, 3, {-1.5, 0}, {0.5, 0}}, 3, 'N', 160, 1.0},
    {{'d', 'T', 'N', 64, 64, 5000, 1, ONE, ZERO}, 4, 'K', 2496, 1.0},
    {{'z', 'N', 'T', 300, 300, 300, 0, ONE, ZERO}, 5, 'N', 0, 1.0},
    {{'z', 'T', 'T', 100, 100, 100, 0, ONE, ZERO}, -1, 0, 0, 1.0},
    {{'z', 'N', 'N', 1200, 128, 64, 0, ONE, ZERO}, 0, 'N', 64, 0x1p-20},
    {{'d', 'T', 'N', 64, 64, 5000, 2, ONE, {0.75, 0}}, 4, 'K', 2496, 1.0},   // K with beta != 0
};
#define N_TCALLS ((int)TEST_REGISTRY_SIZE(TCALLS))

// Call i's operands, the same in every process.
static Operands tcall_operands(int i) {
    g_seed = 0x5EED0000ULL + 7919ULL * (uint64_t)i;
    Operands o = operands(&TCALLS[i].c);
    if (o.w == 2 && TCALLS[i].imag != 1.0) {
        for (size_t k = 1; k < o.nA; k += 2) o.A[k] *= TCALLS[i].imag;
        for (size_t k = 1; k < o.nB; k += 2) o.B[k] *= TCALLS[i].imag;
    }
    return o;
}

// The call through the Fortran symbols, into C.
static void abi_call(const Call *c, const Operands *o, double *C) {
    int M = c->M, N = c->N, K = c->K, ldA = o->ldA, ldB = o->ldB, ldC = o->ldC;
    if (c->r == 'z') {
        double complex al = CMPLX(c->alpha[0], c->alpha[1]), be = CMPLX(c->beta[0], c->beta[1]);
        fortran_zgemm(&c->ta, &c->tb, &M, &N, &K, &al, (const double complex *)o->A, &ldA,
                      (const double complex *)o->B, &ldB, &be, (double complex *)C, &ldC);
    } else {
        fortran_dgemm(&c->ta, &c->tb, &M, &N, &K, &c->alpha[0], o->A, &ldA, o->B, &ldB, &c->beta[0], C, &ldC);
    }
}

// Child: each call of TCALLS twice; a line per call: its index, both
// hashes, the split record (outcome -1 without one), the path and the error
// against cblas. Then the GPU's GEMM count and the split table's source.
static void child_calls(void) {
    for (int i = 0; i < N_TCALLS; i++) {
        Operands o = tcall_operands(i);
        double *C = malloc(o.nC * sizeof *C);
        uint64_t h[2];
        ab_split_record_t rec = {.outcome = -1, .declined = -1, .staged = -1};
        for (int r = 0; r < 2; r++) {
            memcpy(C, o.C0, o.nC * sizeof *C);
            abi_call(&TCALLS[i].c, &o, C);
            h[r] = hash_doubles(C, o.nC);
            if (r == 0 && !ab_internal_last_split(&rec))
                rec = (ab_split_record_t){.outcome = -1, .declined = -1, .staged = -1};
        }
        const Call *c = &TCALLS[i].c;
        Layout la = {o.rA, o.cA, o.ldA}, lb = {o.rB, o.cB, o.ldB}, lc = {c->M, c->N, o.ldC};
        double err = blas_error(c->r == 'z', c->ta, c->tb, c->M, c->N, c->K, CMPLX(c->alpha[0], c->alpha[1]),
                                CMPLX(c->beta[0], c->beta[1]), o.A, la, o.B, lb, o.C0, lc, C);
        printf("call %d %016llx %016llx %d %d %d %d %d %d %s %.3e\n", i, (unsigned long long)h[0],
               (unsigned long long)h[1], rec.outcome, rec.declined, rec.staged, rec.retries, rec.retry_kept,
               rec.scales, ab_get_last_dispatch_path(), err);
        free(C);
        free_operands(&o);
    }
    printf("gpu_gemms %llu\nsource %s\n", (unsigned long long)gpu_gemm_count(), ab_split_source());
}

// Child: the first call, then fork(); the forked child makes it again
// (declined: no Metal context there, so cblas's whole call), checks its
// bits and ends with _exit or exit.
static int child_fork(bool use_exit) {
    Operands o = tcall_operands(0);
    double *C = malloc(o.nC * sizeof *C), *ref = malloc(o.nC * sizeof *ref);
    memcpy(C, o.C0, o.nC * sizeof *C);
    abi_call(&TCALLS[0].c, &o, C);
    printf("parent path %s\n", ab_get_last_dispatch_path());
    fflush(stdout);
    pid_t pid = fork();
    if (pid == 0) {
        memcpy(C, o.C0, o.nC * sizeof *C);
        memcpy(ref, o.C0, o.nC * sizeof *ref);
        abi_call(&TCALLS[0].c, &o, C);
        cblas_call(&TCALLS[0].c, TCALLS[0].c.M, TCALLS[0].c.N, TCALLS[0].c.K, o.A, o.ldA, o.B, o.ldB,
                   TCALLS[0].c.beta, ref, o.ldC);
        ab_split_record_t rec = {.outcome = -1};
        ab_internal_last_split(&rec);
        printf("forked outcome %d declined %d same %d path %s\n", rec.outcome, rec.declined,
               !memcmp(C, ref, o.nC * sizeof *C), ab_get_last_dispatch_path());
        fflush(stdout);
        if (use_exit) exit(0);
        _exit(0);
    }
    int status = 0;
    while (waitpid(pid, &status, 0) < 0 && errno == EINTR) {}
    printf("forked status %d\n", WIFEXITED(status) ? WEXITSTATUS(status) : -1);
    return 0;
}

// Child: the other APIs a split never touches, hashed: native ab_dgemm and
// ab_zgemm, a batch, ab_dgemm_async, ab_dev_dgemm on registered memory, and
// dgemv_.
static void child_other(void) {
    const int n = 300;
    double *a = malloc((size_t)n * n * sizeof *a), *b = malloc((size_t)n * n * sizeof *b);
    g_seed = 77;
    for (int i = 0; i < n * n; i++) { a[i] = uniform(); b[i] = uniform(); }
    ABMatrix A = ab_matrix_create(n, n), B = ab_matrix_create(n, n), C = ab_matrix_create(n, n),
             Ai = ab_matrix_create(n, n), Bi = ab_matrix_create(n, n), Ci = ab_matrix_create(n, n);
    ab_matrix_upload(A, a, true);
    ab_matrix_upload(B, b, true);
    ab_matrix_upload(Ai, b, true);
    ab_matrix_upload(Bi, a, true);
    double *c = malloc((size_t)n * n * sizeof *c);
    ab_dgemm(A, B, C);
    ab_matrix_download(C, c, true);
    printf("ab_dgemm %016llx\n", (unsigned long long)hash_doubles(c, (size_t)n * n));
    ab_zgemm(A, Ai, B, Bi, C, Ci);
    ab_matrix_download(Ci, c, true);
    printf("ab_zgemm %016llx\n", (unsigned long long)hash_doubles(c, (size_t)n * n));
    ABBatch batch = ab_batch_create();
    ab_batch_dgemm(batch, B, A, C);
    ab_batch_commit(batch);
    ab_batch_wait(batch);
    ab_batch_destroy(batch);
    ab_matrix_download(C, c, true);
    printf("ab_batch_dgemm %016llx\n", (unsigned long long)hash_doubles(c, (size_t)n * n));
    ABFuture fut = ab_dgemm_async(B, B, C);
    if (fut) { ab_future_wait(fut); ab_future_destroy(fut); }
    ab_matrix_download(C, c, true);
    printf("ab_dgemm_async %016llx\n", (unsigned long long)hash_doubles(c, (size_t)n * n));
    size_t bytes;
    double *da = page_alloc((size_t)n * n, &bytes), *db = page_alloc((size_t)n * n, &bytes),
           *dc = page_alloc((size_t)n * n, &bytes);
    memcpy(da, a, (size_t)n * n * sizeof *da);
    memcpy(db, b, (size_t)n * n * sizeof *db);
    ab_dev_buffer_t DA = ab_dev_register(da, bytes), DB = ab_dev_register(db, bytes),
                    DC = ab_dev_register(dc, bytes);
    ABStatus dst = DA && DB && DC ? ab_dev_dgemm(AB_NO_TRANS, AB_NO_TRANS, n, n, n, 1.0, DA, n, DB, n, 0.0, DC, n)
                                  : AB_ERROR_ALLOC_FAILED;
    printf("ab_dev_dgemm %d %016llx\n", (int)dst, (unsigned long long)hash_doubles(dc, (size_t)n * n));
    ab_dev_free(DA); ab_dev_free(DB); ab_dev_free(DC);
    free(da); free(db); free(dc);
    double *y = calloc((size_t)n, sizeof *y), one = 1.0, zero = 0.0;
    int nn = n, inc = 1;
    fortran_dgemv("N", &nn, &nn, &one, a, &nn, b, &inc, &zero, y, &inc);
    printf("dgemv_ %016llx\n", (unsigned long long)hash_doubles(y, (size_t)n));
    ab_matrix_destroy(A); ab_matrix_destroy(B); ab_matrix_destroy(C);
    ab_matrix_destroy(Ai); ab_matrix_destroy(Bi); ab_matrix_destroy(Ci);
    free(a); free(b); free(c); free(y);
}

// Child: the bytes in use after 60 more split calls of every axis, once
// the pool has grown (MallocNanoZone pinned by the parent).
// Child: a K tail refused after its run (NaN in it) under split_poison, for
// DGEMM with beta 0.75 and ZGEMM with a complex beta: C must be the cblas
// parts, the head with beta then the tail with beta 1.
static void child_k_refused(void) {
    for (int z = 0; z < 2; z++) {
        Call c = {z ? 'z' : 'd', 'N', 'N', 40, 40, 2100, 1, ONE, {0.75, z ? -0.25 : 0}};
        Operands o = operands(&c);
        plant(&c, &o, 'K', 2095, PLANT_NAN);
        double *C = malloc(o.nC * sizeof *C), *ref = malloc(o.nC * sizeof *ref);
        memcpy(C, o.C0, o.nC * sizeof *C);
        memcpy(ref, o.C0, o.nC * sizeof *ref);
        int got = seam(&c, 'K', 1024, 0, c.M, c.N, c.K, &o, 0, 0, C, o.ldC);
        parts_reference(&c, 'K', 1024, &o, ref);
        printf("k_refused %c %d %d\n", c.r, got, !memcmp(C, ref, o.nC * sizeof *C));
        free(C); free(ref);
        free_operands(&o);
    }
}

static void child_memory(void) {
    malloc_statistics_t s0, s1;
    for (int pass = 0; pass < 2; pass++) {
        if (pass) malloc_zone_statistics(NULL, &s0);
        for (int r = 0; r < (pass ? 10 : 2); r++)
            for (int i = 0; i < 5; i++) {
                Operands o = tcall_operands(i);
                double *C = malloc(o.nC * sizeof *C);
                memcpy(C, o.C0, o.nC * sizeof *C);
                abi_call(&TCALLS[i].c, &o, C);
                free(C);
                free_operands(&o);
            }
    }
    malloc_zone_statistics(NULL, &s1);
    printf("memory growth %lld\n", (long long)s1.size_in_use - (long long)s0.size_in_use);
}

static int child_main(const char *scenario) {
    if (ab_init() != AB_OK) {
        printf("no Metal device\n");
        return 1;
    }
    if (!strcmp(scenario, "calls")) child_calls();
    else if (!strcmp(scenario, "fork_exit")) child_fork(true);
    else if (!strcmp(scenario, "fork__exit")) child_fork(false);
    else if (!strcmp(scenario, "other")) child_other();
    else if (!strcmp(scenario, "memory")) child_memory();
    else if (!strcmp(scenario, "k_refused")) child_k_refused();
    else printf("unknown scenario %s\n", scenario);
    fflush(stdout);
    return 0;
}

// Runs the child scenario with the split table and extra settings (NULL-ended).
static char *run_child(const char *scenario, const char *const *extra, int *status) {
    const char *env[24];
    int n = 0;
    for (; extra && extra[n] && n < 20; n++) env[n] = extra[n];
    env[n] = NULL;
    char *argv[] = {self_path, "--child", (char *)scenario, NULL};
    char *out = NULL;
    *status = spawn_capture(argv, env, &out);
    return out;
}

// What a child's call line says.
typedef struct {
    uint64_t h[2];
    int outcome, declined, staged, retries, retry_kept, scales;
    char path[16];
    double err;
} CallLine;

static bool parse_calls(const char *out, CallLine lines[N_TCALLS], uint64_t *gpu_gemms, char *source,
                        size_t source_len) {
    int seen = 0;
    for (const char *p = out; p && *p; p = strchr(p, '\n') ? strchr(p, '\n') + 1 : NULL) {
        int i;
        unsigned long long h0, h1, gg;
        CallLine l;
        char src[256];
        if (sscanf(p, "call %d %llx %llx %d %d %d %d %d %d %15s %lf", &i, &h0, &h1, &l.outcome, &l.declined,
                   &l.staged, &l.retries, &l.retry_kept, &l.scales, l.path, &l.err) == 11 &&
            i >= 0 && i < N_TCALLS) {
            l.h[0] = h0;
            l.h[1] = h1;
            lines[i] = l;
            seen |= 1 << i;
        } else if (sscanf(p, "gpu_gemms %llu", &gg) == 1) {
            *gpu_gemms = gg;
        } else if (sscanf(p, "source %255s", src) == 1) {
            snprintf(source, source_len, "%s", src);
        }
    }
    return seen == (1 << N_TCALLS) - 1;
}

// The parent's references for call i: cblas's whole call, the cblas parts,
// and the composition (the split's bits when its GPU part stands).
typedef struct { uint64_t whole, parts, split; bool done; } Refs;
static Refs g_refs[N_TCALLS];

static const Refs *refs(int i) {
    Refs *r = &g_refs[i];
    if (r->done) return r;
    const TableCall *t = &TCALLS[i];
    Operands o = tcall_operands(i);
    double *C = malloc(o.nC * sizeof *C);
    memcpy(C, o.C0, o.nC * sizeof *C);
    cblas_call(&t->c, t->c.M, t->c.N, t->c.K, o.A, o.ldA, o.B, o.ldB, t->c.beta, C, o.ldC);
    r->whole = hash_doubles(C, o.nC);
    if (t->g) {
        memcpy(C, o.C0, o.nC * sizeof *C);
        parts_reference(&t->c, t->axis, t->g, &o, C);
        r->parts = hash_doubles(C, o.nC);
        memcpy(C, o.C0, o.nC * sizeof *C);
        compose_reference(&t->c, t->axis, t->g, &o, C);
        r->split = hash_doubles(C, o.nC);
    }
    free(C);
    free_operands(&o);
    r->done = true;
    return r;
}

static bool cut_call(int i) { return TCALLS[i].g > 0; }

// What a scenario expects of call i: the outcome (-1 none), the reference
// and the path; NULL ref for "any, but both runs alike".
typedef enum { REF_WHOLE, REF_PARTS, REF_SPLIT, REF_SELF } RefKind;

typedef struct {
    const char *name;
    const char *env[8];
    // for a cut call: outcome, ref, declined (-2 any), staged (-2 any), retries (-2 any), scales
    int outcome;
    RefKind ref;
    int declined, staged, retries, scales;
} Scenario;

#define ANY (-2)
static const Scenario SCENARIOS[] = {
    {"on", {NULL}, AB_SPLIT_KEPT, REF_SPLIT, -1, ANY, 0, 0},
    {"cpu", {"AB_MODE=cpu"}, -1, REF_WHOLE, ANY, ANY, ANY, ANY},
    {"alloc", {"AB_TEST_FAIL_STAGE=alloc"}, AB_SPLIT_GPU_FAILED, REF_PARTS, -1, 5 /* alloc */, 0, 0},
    {"kernel", {"AB_TEST_FAIL_STAGE=kernel"}, AB_SPLIT_GPU_FAILED, REF_PARTS, -1, 6 /* gpu_failed */, 1, 0},
    {"scales", {"AB_TEST_FAIL_STAGE=scales"}, AB_SPLIT_GPU_FAILED, REF_PARTS, -1, 6, 0, 1},
    {"split_thread", {"AB_TEST_FAIL_STAGE=split_thread"}, AB_SPLIT_DECLINED, REF_WHOLE, 6 /* thread */, ANY,
     ANY, ANY},
    {"split_fpcr", {"AB_TEST_FAIL_STAGE=split_fpcr"}, AB_SPLIT_GPU_FAILED, REF_PARTS, -1, 3 /* fpcr */, 0, 0},
    {"host_pack", {"AB_HOST_PACK=1"}, AB_SPLIT_DECLINED, REF_WHOLE, 2 /* host_pack */, ANY, ANY, ANY},
    {"strassen", {"AB_STRASSEN=1", "AB_STRASSEN_MIN=1"}, AB_SPLIT_KEPT, REF_SPLIT, -1, ANY, 0, 0},
    {"staging0", {"AB_STAGING_MB=0"}, AB_SPLIT_KEPT, REF_SPLIT, -1, ANY, 0, 0},
    {"min_gpu_dim", {"AB_MIN_GPU_DIM=0"}, AB_SPLIT_KEPT, REF_SPLIT, -1, ANY, 0, 0},
};

// Checks one child's call lines against what the scenario expects.
static void check_calls(const Scenario *s, const char *out, int status) {
    CallLine l[N_TCALLS];
    uint64_t gpu_gemms = UINT64_MAX;
    char source[256] = "";
    if (!check(status == 0 && parse_calls(out, l, &gpu_gemms, source, sizeof source),
               "scenario %s: child status %d, output: %.300s", s->name, status, out ? out : ""))
        return;
    for (int i = 0; i < N_TCALLS; i++) {
        const Refs *r = refs(i);
        check(l[i].h[0] == l[i].h[1], "scenario %s call %d: the two runs differ", s->name, i);
        if (!cut_call(i) || s->outcome < 0) {
            // no row, or a row that does not cut: AB_SPLIT unset's route; the whole call on cblas
            int want = s->outcome < 0 || TCALLS[i].row < 0 ? -1 : AB_SPLIT_NOT_CUT;
            check(l[i].outcome == want && l[i].h[0] == r->whole && !strcmp(l[i].path, "cpu"),
                  "scenario %s call %d: outcome %d (want %d), path %s, %s cblas's whole call", s->name, i,
                  l[i].outcome, want, l[i].path, l[i].h[0] == r->whole ? "is" : "is not");
            continue;
        }
        int outcome = s->outcome;
        RefKind ref = s->ref;
        int declined = s->declined, staged = s->staged, retries = s->retries;
        if (outcome == AB_SPLIT_KEPT && TCALLS[i].imag != 1.0) outcome = AB_SPLIT_KEPT_4M;
        uint64_t want = ref == REF_WHOLE ? r->whole : ref == REF_PARTS ? r->parts : r->split;
        check(l[i].outcome == outcome && l[i].h[0] == want &&
              (declined == ANY || l[i].declined == declined) && (staged == ANY || l[i].staged == staged) &&
              (retries == ANY || l[i].retries == retries) && (s->scales == ANY || l[i].scales == s->scales),
              "scenario %s call %d: outcome %d declined %d staged %d retries %d scales %d path %s, C %s the "
              "expected bits", s->name, i, l[i].outcome, l[i].declined, l[i].staged, l[i].retries, l[i].scales,
              l[i].path, l[i].h[0] == want ? "has" : "does not have");
        bool gpu = outcome == AB_SPLIT_KEPT || outcome == AB_SPLIT_KEPT_4M;
        check(!strcmp(l[i].path, gpu ? "split" : "cpu"), "scenario %s call %d: path %s", s->name, i, l[i].path);
        check(l[i].err <= TOL, "scenario %s call %d: error %.2e against cblas", s->name, i, l[i].err);
    }
    if (!strcmp(s->name, "cpu")) check(gpu_gemms == 0, "AB_SPLIT=1 AB_MODE=cpu ran %llu GPU GEMMs",
                                       (unsigned long long)gpu_gemms);
    char want_src[sizeof g_table + 8];
    snprintf(want_src, sizeof want_src, "file:%s", g_table);
    check(!strcmp(source, want_src), "scenario %s: ab_split_source() is %s", s->name, source);
}

// The extra settings of a scenario with the split on.
static int split_env(const Scenario *s, const char **env, char *table_env) {
    int n = 0;
    snprintf(table_env, PATH_MAX + 64, "AB_SPLIT_TABLE=%s", g_table);
    env[n++] = "AB_SPLIT=1";
    env[n++] = table_env;
    for (int k = 0; s && k < 8 && s->env[k]; k++) env[n++] = s->env[k];
    env[n] = NULL;
    return n;
}

// T3 and T8-T10: every scenario of SCENARIOS, and those whose failure hits
// one call only.
static void test_split_scenarios(void) {
    char table_env[PATH_MAX + 64];
    for (size_t k = 0; k < TEST_REGISTRY_SIZE(SCENARIOS); k++) {
        const char *env[16];
        split_env(&SCENARIOS[k], env, table_env);
        int status;
        char *out = run_child("calls", env, &status);
        check_calls(&SCENARIOS[k], out, status);
        free(out);
    }
    // One call is hit: kernel_once (the first: retried, kept), redo_4m (the 4M
    // call: failed twice, the parts), split_scratch (the K axis: declined).
    static const struct { const char *fail; int call_mask, outcome, ref, declined, retries; } ONE_HIT[] = {
        {"AB_TEST_FAIL_STAGE=kernel_once", 1 << 0, AB_SPLIT_KEPT, REF_SPLIT, -1, 1},
        {"AB_TEST_FAIL_STAGE=redo_4m", 1 << 7, AB_SPLIT_GPU_FAILED, REF_PARTS, -1, 1},
        {"AB_TEST_FAIL_STAGE=split_scratch", 1 << 2 | 1 << 4 | 1 << 8, AB_SPLIT_DECLINED, REF_WHOLE, 5 /* scratch */, ANY},
    };
    for (size_t k = 0; k < TEST_REGISTRY_SIZE(ONE_HIT); k++) {
        Scenario s = {ONE_HIT[k].fail, {ONE_HIT[k].fail}, 0, 0, 0, 0, 0, 0};
        const char *env[16];
        split_env(&s, env, table_env);
        int status;
        char *out = run_child("calls", env, &status);
        CallLine l[N_TCALLS];
        uint64_t gg;
        char source[256];
        if (check(status == 0 && parse_calls(out, l, &gg, source, sizeof source), "%s: child output %.300s",
                  ONE_HIT[k].fail, out ? out : ""))
            for (int i = 0; i < N_TCALLS; i++) {
                if (!cut_call(i)) continue;
                const Refs *r = refs(i);
                bool hit = ONE_HIT[k].call_mask >> i & 1;
                int want = hit ? ONE_HIT[k].outcome : TCALLS[i].imag != 1.0 ? AB_SPLIT_KEPT_4M : AB_SPLIT_KEPT;
                uint64_t ref = !hit ? r->split : ONE_HIT[k].ref == REF_WHOLE ? r->whole
                             : ONE_HIT[k].ref == REF_PARTS ? r->parts : r->split;
                check(l[i].outcome == want && l[i].h[0] == ref &&
                      (!hit || ONE_HIT[k].declined == ANY || l[i].declined == ONE_HIT[k].declined) &&
                      (!hit || ONE_HIT[k].retries == ANY || l[i].retries == ONE_HIT[k].retries) &&
                      (!hit || k != 0 || l[i].retry_kept == 1),
                      "%s call %d: outcome %d (want %d) declined %d retries %d retry_kept %d, C %s the expected bits",
                      ONE_HIT[k].fail, i, l[i].outcome, want, l[i].declined, l[i].retries, l[i].retry_kept,
                      l[i].h[0] == ref ? "has" : "does not have");
            }
        free(out);
    }
    finish();
}

// T3: AB_SPLIT unset, empty and 0 alike: every call on its own route (the
// whole call on cblas under auto here), no record, and no AB_SPLIT field in
// the reports or the trace.
static void test_split_off(void) {
    char trace[PATH_MAX + 16], trace_env[PATH_MAX + 32];
    snprintf(trace, sizeof trace, "%s/off", g_dir);
    snprintf(trace_env, sizeof trace_env, "AB_TRACE=%s", trace);
    const char *tmp = getenv("TMPDIR");
    char json[PATH_MAX], json_env[PATH_MAX + 32];   // a path without "split" in it
    snprintf(json, sizeof json, "%s/ab_off_%d.json", tmp && *tmp ? tmp : "/tmp", (int)getpid());
    snprintf(json_env, sizeof json_env, "AB_PROFILE_JSON=%s", json);
    const char *settings[] = {NULL, "AB_SPLIT=", "AB_SPLIT=0"};
    uint64_t first[N_TCALLS] = {0};
    for (int k = 0; k < 3; k++) {
        const char *env[8] = {"AB_REPORT=1", "AB_PROFILE=1", trace_env, json_env, settings[k], NULL};
        int status;
        char *out = run_child("calls", env, &status);
        CallLine l[N_TCALLS];
        uint64_t gg;
        char source[256] = "";
        if (check(status == 0 && parse_calls(out, l, &gg, source, sizeof source), "AB_SPLIT %s: output %.300s",
                  settings[k] ? settings[k] : "unset", out ? out : "")) {
            for (int i = 0; i < N_TCALLS; i++) {
                check(l[i].outcome == -1 && l[i].h[0] == refs(i)->whole && strcmp(l[i].path, "split"),
                      "AB_SPLIT %s call %d: outcome %d path %s", settings[k] ? settings[k] : "unset", i,
                      l[i].outcome, l[i].path);
                if (k == 0) first[i] = l[i].h[0];
                else check(l[i].h[0] == first[i], "call %d: AB_SPLIT %s gives other bits", i, settings[k]);
            }
            check(!strcmp(source, "off"), "ab_split_source() is %s with AB_SPLIT %s", source,
                  settings[k] ? settings[k] : "unset");
            check(!strstr(out, "split"), "AB_SPLIT %s: the reports mention the split", settings[k] ? settings[k] : "unset");
        }
        free(out);
        FILE *jf = fopen(json, "r");
        char *js = jf ? slurp(jf) : NULL;
        check(js && strstr(js, "\"routines\"") && !strstr(js, "split"),
              "AB_SPLIT %s: the JSON dump is missing or mentions the split", settings[k] ? settings[k] : "unset");
        free(js);
        unlink(json);
        // the child's trace file: no split field
        char cmd[PATH_MAX * 2];
        snprintf(cmd, sizeof cmd, "/bin/cat %s.* 2>/dev/null | /usr/bin/grep -c split", trace);
        FILE *p = popen(cmd, "r");
        int hits = -1;
        if (p) { if (fscanf(p, "%d", &hits) != 1) hits = -1; pclose(p); }
        check(hits == 0, "AB_SPLIT %s: the trace mentions the split (%d lines)", settings[k] ? settings[k] : "unset",
              hits);
        snprintf(cmd, sizeof cmd, "/bin/rm -f %s.*", trace);
        if (system(cmd)) {}
    }
    finish();
}

// T7 (children): VECLIB_MAXIMUM_THREADS 1, 2 and 8, each twice: every call's
// bits repeat within a child and across the two.
static void test_split_veclib(void) {
    char table_env[PATH_MAX + 64];
    const char *settings[] = {"VECLIB_MAXIMUM_THREADS=1", "VECLIB_MAXIMUM_THREADS=2", "VECLIB_MAXIMUM_THREADS=8"};
    for (int k = 0; k < 3; k++) {
        uint64_t h[2][N_TCALLS];
        for (int run = 0; run < 2; run++) {
            Scenario s = {settings[k], {settings[k]}, 0, 0, 0, 0, 0, 0};
            const char *env[16];
            split_env(&s, env, table_env);
            int status;
            char *out = run_child("calls", env, &status);
            CallLine l[N_TCALLS];
            uint64_t gg;
            char source[256];
            if (check(status == 0 && parse_calls(out, l, &gg, source, sizeof source), "%s: output %.300s",
                      settings[k], out ? out : ""))
                for (int i = 0; i < N_TCALLS; i++) {
                    h[run][i] = l[i].h[0];
                    check(l[i].h[0] == l[i].h[1] && (!cut_call(i) || kept(l[i].outcome)),
                          "%s call %d: outcome %d, runs alike %d", settings[k], i, l[i].outcome, l[i].h[0] == l[i].h[1]);
                }
            free(out);
        }
        for (int i = 0; i < N_TCALLS; i++)
            check(h[0][i] == h[1][i], "%s call %d: two children differ", settings[k], i);
    }
    finish();
}

// T8: fork() after a split call: the forked child is declined (no Metal
// context), gets cblas's bits, and ends with _exit and with exit.
static void test_split_fork(void) {
    char table_env[PATH_MAX + 64];
    const char *env[16];
    split_env(NULL, env, table_env);
    const char *how[] = {"fork__exit", "fork_exit"};
    for (int k = 0; k < 2; k++) {
        int status;
        char *out = run_child(how[k], env, &status);
        int outcome = -1, declined = -1, same = 0, fstatus = -1;
        char path[16] = "";
        const char *f = out ? strstr(out, "forked outcome") : NULL;
        if (f) sscanf(f, "forked outcome %d declined %d same %d path %15s", &outcome, &declined, &same, path);
        const char *s = out ? strstr(out, "forked status") : NULL;
        if (s) sscanf(s, "forked status %d", &fstatus);
        check(status == 0 && out && strstr(out, "parent path split") && outcome == AB_SPLIT_DECLINED &&
              declined == 0 /* no_device */ && same && !strcmp(path, "cpu") && fstatus == 0,
              "%s: status %d, forked outcome %d declined %d same %d path %s status %d", how[k], status, outcome,
              declined, same, path, fstatus);
        free(out);
    }
    finish();
}

// The integer after "key=" in line, or -1.
static long long field(const char *line, const char *key) {
    char k[64];
    snprintf(k, sizeof k, " %s=", key);
    const char *p = strstr(line, k);
    return p ? atoll(p + strlen(k)) : -1;
}

// T9: the reports with AB_SPLIT on.
static void test_split_reports(void) {
    char table_env[PATH_MAX + 64], trace[PATH_MAX + 16], trace_env[PATH_MAX + 32], json[PATH_MAX + 16],
         json_env[PATH_MAX + 32];
    snprintf(trace, sizeof trace, "%s/on", g_dir);
    snprintf(trace_env, sizeof trace_env, "AB_TRACE=%s", trace);
    snprintf(json, sizeof json, "%s/profile.json", g_dir);
    snprintf(json_env, sizeof json_env, "AB_PROFILE_JSON=%s", json);
    Scenario s = {"report", {"AB_REPORT=1", "AB_PROFILE=1", trace_env, json_env}, 0, 0, 0, 0, 0, 0};
    const char *env[16];
    split_env(&s, env, table_env);
    int status;
    char *out = run_child("calls", env, &status);
    check(status == 0 && out, "child status %d", status);
    if (!out) { finish(); return; }
    char want[sizeof g_table + 64];
    snprintf(want, sizeof want, " AB_SPLIT=1 split_table=file:%s", g_table);
    check(strstr(out, want) != NULL, "the report header lacks '%s'", want);
    // routine lines: calls = gpu_calls + cpu_calls + split_calls; split lines; row lines
    long long split_total = 0;
    int routines = 0;
    int rows_seen = 0;
    for (char *line = out; line && *line; line = strchr(line, '\n') ? strchr(line, '\n') + 1 : NULL) {
        char l[2048];
        size_t n = strcspn(line, "\n");
        snprintf(l, sizeof l, "%.*s", (int)(n < sizeof l ? n : sizeof l - 1), line);
        if (!strncmp(l, "AB_REPORT[", 10) && strstr(l, " calls=") && strstr(l, " split_calls=") &&
            !strstr(l, " split calls=")) {
            routines++;
            check(field(l, "calls") == field(l, "gpu_calls") + field(l, "cpu_calls") + field(l, "split_calls"),
                  "calls != gpu + cpu + split: %s", l);
        } else if (!strncmp(l, "AB_REPORT[", 10) && strstr(l, " split calls=")) {   // not AB_PROFILE's copy
            long long calls = field(l, "calls"), sum = field(l, "kept") + field(l, "kept_4m") + field(l, "range") +
                                                       field(l, "gpu_failed");
            for (int d = 0; d < 7; d++) {
                static const char *const D[] = {"no_device", "too_large", "host_pack", "fpcr", "scalar",
                                                "scratch", "thread"};
                char k[32];
                snprintf(k, sizeof k, "declined_%s", D[d]);
                sum += field(l, k);
            }
            check(calls == sum && calls > 0, "split line counts do not add up: %s", l);
            split_total += field(l, "axis_N") + field(l, "axis_M") + field(l, "axis_K");
        } else if (strstr(l, "] split_row i=")) {
            long long i = field(l, "i");
            rows_seen++;
            if (i == 5) check(field(l, "calls") == 2 && field(l, "split") == 0, "the f = 0 row: %s", l);
            if (i == 0) check(field(l, "calls") == 4 && field(l, "split") == 4 && field(l, "kept_4m") == 2,
                              "row 0: %s", l);
        }
    }
    check(routines == 2, "%d routine lines with split_calls (want zgemm, dgemm)", routines);
    check(split_total == 14 && rows_seen == 6, "split calls by axis %lld (want 14), %d row lines (want 6)",
          split_total, rows_seen);
    check(strstr(out, "AB_PROFILE zgemm M=1200 N=128 K=64  split row=0 axis=N g=64") != NULL,
          "no AB_PROFILE line for a split call");
    check(strstr(out, "AB_SPLIT: calls whose GPU part stood") != NULL, "no AB_SPLIT section in AB_PROFILE");
    free(out);
    // JSON
    FILE *f = fopen(json, "r");
    char *js = f ? slurp(f) : NULL;
    check(js && strstr(js, "\"split_table\": \"file:") && strstr(js, "\"split_rows\"") &&
          strstr(js, "\"split_calls\": "), "the JSON dump lacks the split fields");
    free(js);
    // the trace: a split line's stages sum to t1 - t0, its GPU part's to gpu_wall_ns
    char cmd[PATH_MAX * 2];
    snprintf(cmd, sizeof cmd, "/bin/cat %s.*", trace);
    FILE *p = popen(cmd, "r");
    int split_lines = 0, row_lines = 0;
    char line[4096];
    bool header = false;
    while (p && fgets(line, sizeof line, p)) {
        if (line[0] == '#') { header = header || strstr(line, want + 1); continue; }
        if (strstr(line, " path=split") || (strstr(line, " retries=") && strstr(line, " path=fallback:"))) {
            split_lines++;
            long long t = field(line, "t1_ns") - field(line, "t0_ns");
            long long st = field(line, "init_ns") + field(line, "cpu_part_ns") + field(line, "join_ns") +
                           field(line, "redo_ns") + field(line, "sum_ns") + field(line, "other_ns");
            long long gs = field(line, "gpu_other_ns");
            static const char *const S[] = {"init", "alloc", "pack", "upload", "kernel", "download", "epilogue"};
            for (int k = 0; k < 7; k++) {
                char key[32];
                snprintf(key, sizeof key, "gpu_%s_ns", S[k]);
                gs += field(line, key);
            }
            check(st == t && gs == field(line, "gpu_wall_ns") && field(line, "row") >= 0,
                  "trace stages %lld vs %lld, GPU %lld vs %lld: %.200s", st, t, gs, field(line, "gpu_wall_ns"), line);
        } else if (strstr(line, " row=5 axis=N g=0")) {
            row_lines++;
        }
    }
    if (p) pclose(p);
    check(header && split_lines == 14 && row_lines == 2, "trace: header %d, %d split lines (want 14), %d f = 0 "
          "lines (want 2)", header, split_lines, row_lines);
    snprintf(cmd, sizeof cmd, "/bin/rm -f %s.* %s", trace, json);
    if (system(cmd)) {}
    finish();
}

// T10: AB_PRECISION=fast keeps both tokens and stays within 4x the default
// error; the other APIs give the same bits with AB_SPLIT on and off; the
// benchmark drivers refuse to run under AB_SPLIT; memory does not grow (T11).
static void test_split_interactions(void) {
    char table_env[PATH_MAX + 64];
    const char *env[16];
    int status;
    // fast
    Scenario def = {"default", {NULL}, 0, 0, 0, 0, 0, 0}, fast = {"fast", {"AB_PRECISION=fast", "AB_REPORT=1"},
                                                                  0, 0, 0, 0, 0, 0};
    CallLine ld[N_TCALLS], lf[N_TCALLS];
    uint64_t gg;
    char source[256];
    split_env(&def, env, table_env);
    char *out = run_child("calls", env, &status);
    bool ok_d = status == 0 && parse_calls(out, ld, &gg, source, sizeof source);
    free(out);
    split_env(&fast, env, table_env);
    out = run_child("calls", env, &status);
    bool ok_f = status == 0 && parse_calls(out, lf, &gg, source, sizeof source);
    check(ok_d && ok_f && out && strstr(out, "AB_PRECISION=fast") && strstr(out, "AB_SPLIT=1 split_table="),
          "fast: children failed or a token is missing");
    free(out);
    for (int i = 0; ok_d && ok_f && i < N_TCALLS; i++)
        if (cut_call(i))
            check(kept(lf[i].outcome) && lf[i].err <= 4.0 * fmax(ld[i].err, 1e-16),
                  "fast call %d: outcome %d, error %.2e against the default's %.2e", i, lf[i].outcome, lf[i].err,
                  ld[i].err);
    // the other APIs
    const char *off_env[] = {NULL};
    char *off = run_child("other", off_env, &status);
    int st_off = status;
    const char *catch_all[] = {"AB_SPLIT=1", table_env, NULL};
    char *on = run_child("other", catch_all, &status);
    check(st_off == 0 && status == 0 && off && on && strstr(off, "dgemv_ ") && strstr(off, "ab_dev_dgemm 0 ") && !strcmp(off, on),
          "the other APIs differ with AB_SPLIT on:\n%.300s\nvs\n%.300s", off ? off : "", on ? on : "");
    free(off);
    free(on);
    // the drivers refuse
    static const struct { const char *bin; const char *args[4]; } DRIVERS[] = {
        {"calibrate", {"--out", "/nonexistent/calibrate", NULL}},
        {"replay", {"--mode", "cpu", "/nonexistent/trace", NULL}},
        {"bench_split", {"run", "sq1x8", "1", "/nonexistent"}},
    };
    for (size_t k = 0; k < TEST_REGISTRY_SIZE(DRIVERS); k++) {
        char bin[PATH_MAX];
        snprintf(bin, sizeof bin, "%s/%s", AB_BUILD_DIR, DRIVERS[k].bin);
        char *argv[6] = {bin};
        for (int a = 0; a < 4 && DRIVERS[k].args[a]; a++) argv[a + 1] = (char *)DRIVERS[k].args[a];
        char *o = NULL;
        int rc = spawn_capture(argv, catch_all, &o);
        check(rc > 0 && o && strstr(o, "AB_SPLIT is set"), "%s ran under AB_SPLIT (status %d): %.200s",
              DRIVERS[k].bin, rc, o ? o : "");
        free(o);
    }
    // memory growth with MallocNanoZone pinned; not under AddressSanitizer,
    // whose zone statistics count the bytes ever allocated
#if defined(__has_feature)
#if __has_feature(address_sanitizer)
#define SPLIT_UNDER_ASAN 1
#endif
#endif
#ifndef SPLIT_UNDER_ASAN
    const char *mem_env[] = {"AB_SPLIT=1", table_env, "MallocNanoZone=0", NULL};
    out = run_child("memory", mem_env, &status);
    long long growth = LLONG_MAX;
    const char *m = out ? strstr(out, "memory growth ") : NULL;
    if (m) growth = atoll(m + 14);
    check(status == 0 && growth <= (1 << 20), "memory grew %lld bytes over 50 split calls", growth);
    free(out);
#endif
    finish();
}

// =============================================================================
// Main
// =============================================================================

static const TestCase TESTS[] = {
    {"test_split_composition", test_split_composition},
    {"test_split_accuracy", test_split_accuracy},
    {"test_split_exponent_range", test_split_exponent_range},
    {"test_split_determinism", test_split_determinism},
    {"test_split_registered_c", test_split_registered_c},
    {"test_split_caller_fpcr", test_split_caller_fpcr},
    {"test_split_off", test_split_off},
    {"test_split_scenarios", test_split_scenarios},
    {"test_split_veclib", test_split_veclib},
    {"test_split_fork", test_split_fork},
    {"test_split_reports", test_split_reports},
    {"test_split_interactions", test_split_interactions},
};
static const int N_TESTS = (int)TEST_REGISTRY_SIZE(TESTS);

int main(int argc, char **argv) {
    if (argc == 3 && !strcmp(argv[1], "--child")) {
        alarm(CHILD_TIMEOUT_S);
        return child_main(argv[2]);
    }
    uint32_t len = sizeof self_path;
    if (_NSGetExecutablePath(self_path, &len) != 0) {
        fprintf(stderr, "test_split: cannot resolve executable path\n");
        return 1;
    }
    const char *tmp = getenv("TMPDIR");
    snprintf(g_dir, sizeof g_dir, "%s/test_split.XXXXXX", tmp && *tmp ? tmp : "/tmp");
    if (!mkdtemp(g_dir)) {
        fprintf(stderr, "test_split: mkdtemp failed\n");
        return 1;
    }
    snprintf(g_table, sizeof g_table, "%s/split_table.txt", g_dir);
    FILE *t = fopen(g_table, "w");
    if (!t || fputs(TABLE, t) < 0 || fclose(t)) {
        fprintf(stderr, "test_split: cannot write %s\n", g_table);
        return 1;
    }
    clear_ab_env();   // AB_SPLIT and the rest only where a test sets them
    if (ab_init() != AB_OK) {
        fprintf(stderr, "test_split: no Metal device\n");
        return 1;
    }
    printf("╔══════════════════════════════════════════════════════════════════╗\n");
    printf("║  apple-bottom AB_SPLIT: one GEMM on Accelerate and the GPU       ║\n");
    printf("╚══════════════════════════════════════════════════════════════════╝\n");
    for (int i = 0; i < N_TESTS; i++) {
        // test_split <name>...: those tests only (tests/split_defects.py)
        bool wanted = argc < 2;
        for (int a = 1; a < argc; a++) wanted = wanted || !strcmp(argv[a], TESTS[i].name);
        if (!wanted) continue;
        printf("[%d/%d] %-30s ... ", i + 1, N_TESTS, TESTS[i].name);
        fflush(stdout);
        int pre = tests_passed + tests_failed;
        TESTS[i].fn();
        if (tests_passed + tests_failed == pre) {
            fprintf(stderr, "\nFATAL: ghost test — '%s' reported no result\n", TESTS[i].name);
            abort();
        }
    }
    ab_shutdown();

    printf("\n═══════════════════════════════════════════════════════════════════\n");
    printf("Results: %d passed, %d failed\n", tests_passed, tests_failed);
    if (tests_failed == 0) {
        printf("✓ All %d AB_SPLIT tests PASSED\n", tests_passed);
        return 0;
    }
    printf("✗ Some tests FAILED\n");
    return 1;
}
