// Precision Test — apple-bottom
//
// Every GEMM case is compared with a compensated FP64 reference (Dot2: Ogita,
// Rump & Oishi, SIAM J. Sci. Comput. 26(6), 2005), which is as accurate as
// accumulating in twice FP64 precision and rounding once. Its own error is
// below 2·u64·L (u64 = 2⁻⁵³ = u²/32; checked against exact rational
// arithmetic), negligible next to the bounds below.
//
// Units: u = 2⁻²⁴ (FP32 unit roundoff), u² = 2⁻⁴⁸ ≈ 3.55e-15 (DD).
// Two metrics, both must pass:
//
//   componentwise  max_ij |ΔC_ij| / L_ij  <= CW_BOUND · √K · u²
//       L = |A||B| for DGEMM and |α||A||B| + |β||C₀| for the scaled kernels.
//       ZGEMM uses the 3M (Gauss) bounds of Higham, SIAM J. Matrix Anal.
//       Appl. 13(3), 1992: Re |Ar||Br| + |Ai||Bi|, Im (|Ar|+|Ai|)(|Br|+|Bi|)
//       + |Ar||Br| + |Ai||Bi|.
//   normwise       ||ΔC||_F / ||C||_F     <= NW_BOUND · √K · u²
//
// Both use Wilkinson-style probabilistic scaling (Higham 2002, §3.1): the
// worst case grows like K·u², but rounding errors of random sign add up like
// √K. The inputs have mixed magnitudes (10⁻³..10²), so a few large products
// dominate each partial sum and each of the K dd_fma steps rounds an
// accumulator of size ~L.
//
// Thresholds, measured on M2 Max (default and DWTIMESDW3=1 builds agree):
//   componentwise 5.6–19 u² = 0.15–1.02 √K·u²; naive FP64 accumulation on
//   the same data shows the same statistic (10–24 u64). CW_BOUND = 2.
//   normwise 0.21–0.23 √K·u² (DGEMM), 0.32 √K·u² (ZGEMM). NW_BOUND = 1.
//   A build without MTLMathModeSafe exceeds both by more than 10⁶x.
//
// The first case is a MathModeSafe smoke test (exact 1x1 products).

#include "apple_bottom.h"
#include "test_dot2.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <Accelerate/Accelerate.h>

// The error-free transformations in dot2_gemm_acc (test_dot2.h, which turns
// contraction off for them) need each product and sum rounded on its own: no
// FMA contraction in this file either.
#pragma STDC FP_CONTRACT OFF

#define U2       0x1p-48   // u² = 2⁻⁴⁸, DD unit roundoff
#define CW_BOUND 2.0       // componentwise bound, units of √K·u²
#define NW_BOUND 1.0       // normwise bound, units of √K·u²

typedef enum { OP_DGEMM, OP_DGEMM_SCALED, OP_ZGEMM } GemmOp;

typedef struct {
    const char* label;
    GemmOp op;
    int M, K, N;
    double alpha, beta;    // OP_DGEMM_SCALED only
} PrecisionCase;

// Number of real planes per matrix: ZGEMM has real and imaginary parts.
static int n_planes(const PrecisionCase* pc) { return pc->op == OP_ZGEMM ? 2 : 1; }

// Entries (±1)·10^e with e ∈ {-3..2}: mixed magnitudes, random signs.
static double* random_matrix(size_t n) {
    double* x = malloc(n * sizeof(double));
    for (size_t i = 0; i < n; i++)
        x[i] = (drand48() * 2 - 1) * pow(10, (int)(drand48() * 6 - 3));
    return x;
}

// Reference R and componentwise bound L (both zero on entry).
static void reference_gemm(const PrecisionCase* pc, double* const A[2], double* const B[2],
                           const double* C0, double* R[2], double* L[2]) {
    const int M = pc->M, K = pc->K, N = pc->N;
    const size_t nC = (size_t)M * N;
    double* E[2] = {NULL, NULL};    // Dot2 error accumulators
    for (int p = 0; p < n_planes(pc); p++) E[p] = calloc(nC, sizeof(double));

    if (pc->op == OP_ZGEMM) {
        // Cr = Ar·Br − Ai·Bi,  Ci = Ar·Bi + Ai·Br
        dot2_zgemm_acc(M, N, K, A[0], A[1], B[0], B[1], R, E);
        abs_gemm_acc(M, N, K, A[0], NULL, B[0], NULL, L[0]);
        abs_gemm_acc(M, N, K, A[1], NULL, B[1], NULL, L[0]);
        memcpy(L[1], L[0], nC * sizeof(double));
        abs_gemm_acc(M, N, K, A[0], A[1], B[0], B[1], L[1]);
    } else {
        dot2_gemm_acc(M, N, K, 1.0, A[0], B[0], R[0], E[0]);
        abs_gemm_acc(M, N, K, A[0], NULL, B[0], NULL, L[0]);
    }
    for (int p = 0; p < n_planes(pc); p++)
        for (size_t i = 0; i < nC; i++) R[p][i] += E[p][i];

    if (pc->op == OP_DGEMM_SCALED) {
        for (size_t i = 0; i < nC; i++) {
            R[0][i] = pc->alpha * R[0][i] + pc->beta * C0[i];
            L[0][i] = fabs(pc->alpha) * L[0][i] + fabs(pc->beta) * fabs(C0[i]);
        }
    }
    free(E[0]); free(E[1]);
}

// Run the case through the public API into G; returns the first failing status.
static ABStatus gpu_gemm(const PrecisionCase* pc, double* const A[2], double* const B[2],
                         const double* C0, double* G[2]) {
    const int planes = n_planes(pc);
    ABMatrix mA[2] = {NULL, NULL}, mB[2] = {NULL, NULL}, mC[2] = {NULL, NULL};
    ABStatus st = AB_OK;

    for (int p = 0; p < planes; p++) {
        mA[p] = ab_matrix_create(pc->M, pc->K);
        mB[p] = ab_matrix_create(pc->K, pc->N);
        mC[p] = ab_matrix_create(pc->M, pc->N);
        if (!mA[p] || !mB[p] || !mC[p]) { st = AB_ERROR_ALLOC_FAILED; goto cleanup; }
        if ((st = ab_matrix_upload(mA[p], A[p], true)) != AB_OK) goto cleanup;
        if ((st = ab_matrix_upload(mB[p], B[p], true)) != AB_OK) goto cleanup;
    }
    switch (pc->op) {
    case OP_DGEMM:
        st = ab_dgemm(mA[0], mB[0], mC[0]);
        break;
    case OP_DGEMM_SCALED:
        if ((st = ab_matrix_upload(mC[0], C0, true)) != AB_OK) goto cleanup;
        st = ab_dgemm_scaled(pc->alpha, mA[0], mB[0], pc->beta, mC[0]);
        break;
    case OP_ZGEMM:
        st = ab_zgemm(mA[0], mA[1], mB[0], mB[1], mC[0], mC[1]);
        break;
    }
    for (int p = 0; st == AB_OK && p < planes; p++)
        st = ab_matrix_download(mC[p], G[p], true);

cleanup:
    for (int p = 0; p < 2; p++) {
        ab_matrix_destroy(mA[p]);
        ab_matrix_destroy(mB[p]);
        ab_matrix_destroy(mC[p]);
    }
    return st;
}

// Returns 1 if passed, 0 if failed
static int run_case(const PrecisionCase* pc) {
    const int planes = n_planes(pc);
    const size_t nA = (size_t)pc->M * pc->K, nB = (size_t)pc->K * pc->N, nC = (size_t)pc->M * pc->N;
    double *A[2] = {NULL, NULL}, *B[2] = {NULL, NULL}, *C0 = NULL;
    double *G[2] = {NULL, NULL}, *R[2] = {NULL, NULL}, *L[2] = {NULL, NULL};

    srand48(pc->M + pc->K + pc->N);
    for (int p = 0; p < planes; p++) {
        A[p] = random_matrix(nA);
        B[p] = random_matrix(nB);
        G[p] = malloc(nC * sizeof(double));
        R[p] = calloc(nC, sizeof(double));
        L[p] = calloc(nC, sizeof(double));
    }
    if (pc->op == OP_DGEMM_SCALED) C0 = random_matrix(nC);

    int passed = 0;
    ABStatus st = gpu_gemm(pc, A, B, C0, G);
    if (st != AB_OK) {
        printf("  %-34s │ %s │ ✗ FAIL\n", pc->label, ab_status_string(st));
        goto cleanup;
    }
    reference_gemm(pc, A, B, C0, R, L);

    double d2 = 0, r2 = 0, cw = 0;
    for (int p = 0; p < planes; p++) {
        for (size_t i = 0; i < nC; i++) {
            double d = G[p][i] - R[p][i];
            double e = fabs(d) / L[p][i];
            d2 += d * d;
            r2 += R[p][i] * R[p][i];
            if (!(e <= cw)) cw = e;   // NaN-propagating max
        }
    }
    double nw = sqrt(d2 / r2), unit = sqrt((double)pc->K) * U2;
    passed = nw / unit <= NW_BOUND && cw / unit <= CW_BOUND;
    printf("  %-34s │ %.2e %8.3g │ %.2e %8.3g │ %s\n",
           pc->label, nw, nw / unit, cw, cw / unit, passed ? "✓ PASS" : "✗ FAIL");

cleanup:
    for (int p = 0; p < 2; p++) { free(A[p]); free(B[p]); free(G[p]); free(R[p]); free(L[p]); }
    free(C0);
    return passed;
}

// MathModeSafe smoke test (K-P1-6 / W-P1-4): 1x1 ab_dgemm products that are
// exact in DD only if the FP32 product error terms survive compilation. With
// fast math (neither MTLMathModeSafe nor fastMathEnabled=NO) they come back
// as 1.0 and 0x1.000004p+0.
static int run_safe_math_smoke(void) {
    static const struct { double a, b, expect; } cases[] = {
        { 1.0 + 0x1p-24, 1.0 - 0x1p-24, 1.0 - 0x1p-48 },          // (1+2⁻²⁴)(1−2⁻²⁴)
        { 1.0 + 0x1p-23, 1.0 + 0x1p-23, 0x1.000004000004p+0 },    // (1+2⁻²³)²
    };
    int passed = 1;
    ABMatrix mA = ab_matrix_create(1, 1), mB = ab_matrix_create(1, 1), mC = ab_matrix_create(1, 1);
    for (size_t c = 0; c < sizeof(cases) / sizeof(cases[0]); c++) {
        double got = NAN;
        if (ab_matrix_upload(mA, &cases[c].a, false) != AB_OK ||
            ab_matrix_upload(mB, &cases[c].b, false) != AB_OK ||
            ab_dgemm(mA, mB, mC) != AB_OK ||
            ab_matrix_download(mC, &got, false) != AB_OK || got != cases[c].expect) {
            printf("    %a × %a = %a, expected %a\n", cases[c].a, cases[c].b, got, cases[c].expect);
            passed = 0;
        }
    }
    ab_matrix_destroy(mA); ab_matrix_destroy(mB); ab_matrix_destroy(mC);
    printf("  %-34s │ %-37s │ %s\n",
           "MathModeSafe smoke (1x1 ab_dgemm)", "exact DD products, 2 oracles", passed ? "✓ PASS" : "✗ FAIL");
    return passed;
}

int main(void) {
    printf("╔══════════════════════════════════════════════════════════════════════════╗\n");
    printf("║  apple-bottom Precision Test (componentwise + normwise vs Dot2 FP64 ref)  ║\n");
    printf("╚══════════════════════════════════════════════════════════════════════════╝\n\n");

    if (ab_init() != AB_OK) { fprintf(stderr, "Init failed\n"); return 1; }
    printf("Device: %s\n", ab_device_name());
    printf("Bounds: ||ΔC||_F/||C||_F <= %.0f·√K·u²,  max|ΔC|/L <= %.0f·√K·u²,  u² = 2⁻⁴⁸ ≈ %.2e\n\n",
           NW_BOUND, CW_BOUND, U2);

    printf("  Case (kernel)                      │ normwise   /√K·u² │ compwise   /√K·u² │ Status\n");
    printf("─────────────────────────────────────┼───────────────────┼───────────────────┼────────\n");

    static const PrecisionCase cases[] = {
        // ab_dgemm: dd_dgemm, or dd_dgemm_narrow for a small grid (see use_narrow_tiles)
        {"ab_dgemm 64x64 K=64 (narrow)",         OP_DGEMM,   64,   64,   64, 0, 0},
        {"ab_dgemm 128x128 K=128",               OP_DGEMM,  128,  128,  128, 0, 0},
        {"ab_dgemm 256x256 K=256",               OP_DGEMM,  256,  256,  256, 0, 0},
        {"ab_dgemm 512x512 K=512 (dd_dgemm)",    OP_DGEMM,  512,  512,  512, 0, 0},
        {"ab_dgemm 1024x1024 K=1024",            OP_DGEMM, 1024, 1024, 1024, 0, 0},
        {"ab_dgemm 2048x2048 K=2048",            OP_DGEMM, 2048, 2048, 2048, 0, 0},
        {"ab_dgemm 100x100 K=4096 (narrow)",     OP_DGEMM,  100, 4096,  100, 0, 0},
        {"ab_dgemm 1000x50 K=200 (narrow)",      OP_DGEMM, 1000,  200,   50, 0, 0},
        // ab_dgemm_scaled: dd_dgemm_ab, dd_dgemm_ab_ts when M >= 4N, or
        // dd_dgemm_ab_narrow for a small grid (see use_narrow_tiles)
        {"scaled 1100x1000 K=500 (dd_dgemm_ab)", OP_DGEMM_SCALED,  1100, 500, 1000, 0.7, -1.3},
        {"scaled 500x300 K=700 (narrow)",        OP_DGEMM_SCALED,   500, 700, 300, 0.7, -1.3},
        {"scaled 1000x50 K=200 a1b0 (narrow)",   OP_DGEMM_SCALED,  1000, 200,  50, 1.0,  0.0},
        {"scaled 18277x150 K=300 (ts, QE)",      OP_DGEMM_SCALED, 18277, 300, 150, 0.7, -1.3},
        // ab_zgemm: Gauss 3M, three real GEMMs plus DD add/sub
        {"ab_zgemm 256x256 K=256 (narrow)",      OP_ZGEMM,  256,  256,  256, 0, 0},
        {"ab_zgemm 1024x64 K=512 (narrow)",      OP_ZGEMM, 1024,  512,   64, 0, 0},
        {"ab_zgemm 4000x40 K=512 (ts)",          OP_ZGEMM, 4000,  512,   40, 0, 0},
    };
    const int n_cases = (int)(sizeof(cases) / sizeof(cases[0]));
    const int n_total = n_cases + 1;   // + the smoke test

    int n_passed = run_safe_math_smoke();
    for (int c = 0; c < n_cases; c++)
        n_passed += run_case(&cases[c]);

    printf("\n═══════════════════════════════════════════════════════════════════════════\n");
    if (n_passed == n_total) {
        printf("✓ All precision tests PASSED (%d/%d)\n", n_passed, n_total);
    } else {
        printf("✗ Some precision tests FAILED (%d/%d passed)\n", n_passed, n_total);
    }

    ab_shutdown();
    return n_passed == n_total ? 0 : 1;
}
