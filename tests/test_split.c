// =============================================================================
// test_split.c — AB_SPLIT: one BLAS-ABI GEMM split between Accelerate and the GPU
// =============================================================================
// The executor (src/blas_wrapper.c, src/split_slice.h) through its seam,
// ab_internal_dgemm_split and ab_internal_zgemm_split, on the GPU:
//   composition  a call cut at g is, bit for bit, cblas on the head slice
//                (the seam with g = 0 on that slice) and the staged path on
//                the tail slice alone (the seam with g = L on it); on the K
//                axis, the head into C plus the GPU's product of the tail
//                added in FP64.
// References are made in this process on the same memory.
// =============================================================================

#include "apple_bottom.h"
#include "gpu_stage.h"
#include "split_slice.h"
#include "test_blas_check.h"
#include "test_registry.h"
#include <Accelerate/Accelerate.h>
#include <complex.h>
#include <math.h>
#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

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

// =============================================================================
// Main
// =============================================================================

static const TestCase TESTS[] = {
    {"test_split_composition", test_split_composition},
};
static const int N_TESTS = (int)TEST_REGISTRY_SIZE(TESTS);

int main(void) {
    clear_ab_env();   // AB_SPLIT and the rest only where a test sets them
    if (ab_init() != AB_OK) {
        fprintf(stderr, "test_split: no Metal device\n");
        return 1;
    }
    printf("╔══════════════════════════════════════════════════════════════════╗\n");
    printf("║  apple-bottom AB_SPLIT: one GEMM on Accelerate and the GPU       ║\n");
    printf("╚══════════════════════════════════════════════════════════════════╝\n");
    for (int i = 0; i < N_TESTS; i++) {
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
