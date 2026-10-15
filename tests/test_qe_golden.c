// =============================================================================
// apple-bottom golden bits: GPU-routed BLAS-ABI results, bit for bit
// =============================================================================
//
// Each case (CASES) makes one call through the plain Fortran symbols dgemm_ /
// zgemm_ in AB_MODE=gpu, without ab_init() (the first call sets the library
// up, as in QE), and hashes the whole C array, leading-dimension padding
// included, with 64-bit FNV-1a. The hashes must equal the golden file's,
// written by the library its "library:" line names (that library's
// ab_build_info()). Every case also checks that the GPU computed C, that A, B
// and the padding of C are unchanged, and that C is within TOL of cblas
// (test_blas_check.h); --generate writes no file while any case fails these.
// Inputs come from splitmix64 seeded with the case name, so adding a case
// changes no other case's data.
//
// The bits depend on the GPU and the Metal compiler: on another device than
// the file's the hashes are not compared (SKIP; a failure with
// AB_GOLDEN_STRICT=1). When and how to re-baseline: tests/README.md, Golden
// bits.
//
// Usage: test_qe_golden [--generate] [--file PATH]
//   --generate   run every case and rewrite the golden file with the loaded
//                library's results
//   --file PATH  golden file to read or write (default: GOLDEN_FILE below)
// =============================================================================

#include "apple_bottom.h"
#include "test_blas_check.h"
#include <complex.h>
#include <inttypes.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/sysctl.h>

#ifndef AB_SOURCE_ROOT
#define AB_SOURCE_ROOT "."
#endif
#define GOLDEN_FILE AB_SOURCE_ROOT "/tests/golden/qe_golden.txt"
#define MAX_ENTRIES 256

// =============================================================================
// Cases
// =============================================================================

typedef enum { S_A1B0, S_A1B1, S_A1BX, S_AXB0, S_AXB1, S_AXBX, S_A2B0, S_AM1B1 } ScalarSet;

// dgemm_ uses the real parts. The "other" values are not representable in DD
// (FP32 x 2), so dgemm_'s alpha and beta conversion is part of the bits.
static const struct { const char *tag; double complex alpha, beta; } SCALARS[] = {
    [S_A1B0]  = {"a1b0",  CMPLX(1.0, 0.0),   CMPLX(0.0, 0.0)},
    [S_A1B1]  = {"a1b1",  CMPLX(1.0, 0.0),   CMPLX(1.0, 0.0)},
    [S_A1BX]  = {"a1bx",  CMPLX(1.0, 0.0),   CMPLX(0.3, -0.45)},
    [S_AXB0]  = {"axb0",  CMPLX(-1.7, 0.6),  CMPLX(0.0, 0.0)},
    [S_AXB1]  = {"axb1",  CMPLX(-1.7, 0.6),  CMPLX(1.0, 0.0)},
    [S_AXBX]  = {"axbx",  CMPLX(-1.7, 0.6),  CMPLX(0.3, -0.45)},
    [S_A2B0]  = {"a2b0",  CMPLX(2.0, 0.0),   CMPLX(0.0, 0.0)},   // QE PAW newq, gamma point
    [S_AM1B1] = {"am1b1", CMPLX(-1.0, 0.0),  CMPLX(1.0, 0.0)},
};

typedef enum {
    D_UNIFORM,       // A, B, C uniform in [-1, 1)
    D_MIXED,         // A, B, C uniform times 2^e, e in [-12, 6]
    D_NAN_INF_C,     // C cycles NaN, +Inf, -Inf (beta = 0: C must not be read)
    D_NEG_IMAG,      // complex A with Ai = -Ar (1 + 2^-30 r): Ar + Ai cancels
    D_ZERO_SUM,      // each row of op(A) sums to exactly 0, B = 1 + 2^-20 r
    D_SIGNED_ZERO,   // +-0 rows of op(A), columns of op(B), scattered entries, all of C
} DataKind;

static const char *const DATA_TAG[] = {
    [D_UNIFORM] = "uniform", [D_MIXED] = "mixed", [D_NAN_INF_C] = "naninfC",
    [D_NEG_IMAG] = "AiNegAr", [D_ZERO_SUM] = "zerosum", [D_SIGNED_ZERO] = "signedzero",
};

typedef struct {
    char routine;            // 'd' dgemm_, 'z' zgemm_
    char ta, tb;
    int M, N, K, pad;        // pad: ld = stored rows + pad for A, B and C
    ScalarSet sc;
    DataKind data;
} GoldenCase;

// Every dimension is above AB_MIN_GPU_DIM (32), so AB_MODE=gpu sends each
// call to the GPU.
static const GoldenCase CASES[] = {
    // Every TRANS pair
    {'z', 'N', 'N', 75, 53, 47, 3, S_A1B0, D_UNIFORM},
    {'z', 'N', 'T', 75, 53, 47, 3, S_AXB1, D_UNIFORM},
    {'z', 'N', 'C', 75, 53, 47, 3, S_A1BX, D_UNIFORM},
    {'z', 'T', 'N', 75, 53, 47, 3, S_AXB0, D_UNIFORM},
    {'z', 'T', 'T', 75, 53, 47, 3, S_A1B1, D_UNIFORM},
    {'z', 'T', 'C', 75, 53, 47, 3, S_AXBX, D_UNIFORM},
    {'z', 'C', 'N', 75, 53, 47, 3, S_A1B0, D_UNIFORM},
    {'z', 'C', 'T', 75, 53, 47, 3, S_AXB1, D_UNIFORM},
    {'z', 'C', 'C', 75, 53, 47, 3, S_AXBX, D_UNIFORM},
    {'d', 'N', 'N', 75, 53, 47, 3, S_A1B0, D_UNIFORM},
    {'d', 'N', 'T', 75, 53, 47, 3, S_AXB1, D_UNIFORM},
    {'d', 'T', 'N', 75, 53, 47, 3, S_A1BX, D_UNIFORM},
    {'d', 'T', 'T', 75, 53, 47, 3, S_AXB0, D_UNIFORM},
    {'d', 'C', 'N', 75, 53, 47, 3, S_AXBX, D_UNIFORM},
    {'d', 'N', 'C', 75, 53, 47, 3, S_A1B1, D_UNIFORM},
    {'d', 'C', 'C', 75, 53, 47, 3, S_A1B0, D_UNIFORM},
    // Several tiles with ragged edges; the tall-skinny kernel (M >= 4N)
    {'d', 'N', 'T', 513, 257, 129, 2, S_AXBX, D_UNIFORM},
    {'d', 'N', 'N', 600, 48, 40, 3, S_AXBX, D_UNIFORM},
    {'z', 'N', 'N', 600, 48, 40, 3, S_AXBX, D_UNIFORM},
    // Mixed magnitudes: DD low words matter
    {'d', 'T', 'N', 96, 80, 72, 1, S_AXBX, D_MIXED},
    {'z', 'C', 'N', 96, 80, 72, 1, S_AXBX, D_MIXED},
    // beta = 0 with NaN and Inf in C
    {'d', 'N', 'N', 75, 53, 47, 3, S_AXB0, D_NAN_INF_C},
    {'z', 'C', 'N', 75, 53, 47, 3, S_AXB0, D_NAN_INF_C},
    // Cancellation
    {'z', 'C', 'N', 96, 80, 72, 3, S_A1B0, D_NEG_IMAG},
    {'d', 'N', 'N', 96, 80, 72, 3, S_A1B0, D_ZERO_SUM},
    {'z', 'T', 'N', 96, 80, 72, 3, S_A1BX, D_ZERO_SUM},
    // Signed zeros
    {'d', 'T', 'N', 75, 53, 47, 3, S_AM1B1, D_SIGNED_ZERO},
    {'z', 'N', 'C', 75, 53, 47, 3, S_AM1B1, D_SIGNED_ZERO},
    // Quantum ESPRESSO shapes (one k-point of si64 per rank, roughly):
    // cegterg projection psi^H hpsi and update psi vc (ld = npwx > npw), PAW
    // newq ('T','N', 2*ngm_l = 73062, ld = 2*ngm_l, alpha = 2) and addusdens.
    {'z', 'C', 'N', 160, 160, 9140, 7, S_A1B0, D_UNIFORM},
    {'z', 'N', 'N', 9140, 160, 160, 7, S_A1B1, D_UNIFORM},
    {'d', 'T', 'N', 36, 64, 73062, 0, S_A2B0, D_UNIFORM},
    {'d', 'N', 'T', 4000, 171, 64, 0, S_A1B0, D_UNIFORM},
};
#define N_CASES ((int)(sizeof CASES / sizeof CASES[0]))

static void case_name(const GoldenCase *gc, char *buf, size_t n) {
    snprintf(buf, n, "%cgemm_/%c%c/%dx%dx%d/ld+%d/%s/%s", gc->routine, gc->ta, gc->tb,
             gc->M, gc->N, gc->K, gc->pad, SCALARS[gc->sc].tag, DATA_TAG[gc->data]);
}

// =============================================================================
// Data
// =============================================================================

static uint64_t sm_state;
static uint64_t splitmix64(void) {
    uint64_t s = sm_state;
    sm_state += 0x9E3779B97F4A7C15ULL;
    return splitmix(s);
}

// Uniform in [-1, 1) on the 2^-52 grid. All scalings below are exact (powers
// of two), so the inputs are the same on every machine.
static double uniform(void) { return (double)(splitmix64() >> 11) * 0x1.0p-52 - 1.0; }
static double signed_zero(void) { return (splitmix64() & 1) ? -0.0 : 0.0; }

typedef enum { V_UNIFORM, V_MIXED, V_GRID, V_NEAR_ONE, V_NAN_INF, V_SIGNED_ZERO } ValueKind;

static double value(ValueKind v, size_t idx) {
    static const double NAN_INF[] = {NAN, INFINITY, -INFINITY};
    switch (v) {
    case V_MIXED: {
        double x = uniform();       // drawn first: argument order is unspecified
        return ldexp(x, (int)(splitmix64() % 19) - 12);
    }
    case V_GRID:        return ldexp(floor(ldexp(uniform(), 20)), -20);
    case V_NEAR_ONE:    return 1.0 + ldexp(uniform(), -20);
    case V_NAN_INF:     return NAN_INF[idx % 3];
    case V_SIGNED_ZERO: return signed_zero();
    default:            return uniform();
    }
}

// Offset of op(X)(i, j) in doubles; w = 1 (real) or 2 (complex, re then im).
static size_t op_at(char t, Layout l, int w, int i, int j) {
    return (is_notrans(t) ? (size_t)j * l.ld + i : (size_t)i * l.ld + j) * w;
}

// NaN everywhere, then the rows x cols block from v, column by column.
static void fill(double *X, Layout l, int w, ValueKind v) {
    for (size_t i = 0; i < nelem(l) * w; i++) X[i] = NAN;
    size_t idx = 0;
    for (int j = 0; j < l.cols; j++)
        for (int i = 0; i < l.rows; i++)
            for (int p = 0; p < w; p++) X[((size_t)j * l.ld + i) * w + p] = value(v, idx++);
}

typedef struct {
    const GoldenCase *gc;
    int w;                              // doubles per element
    Layout la, lb, lc;
    double *A, *B, *C0, *C, *A0, *B0;
} Problem;

// Inputs of one case; the generator is seeded with the case name.
static void make_problem(Problem *P, const GoldenCase *gc, const char *name) {
    const int M = gc->M, N = gc->N, K = gc->K, w = gc->routine == 'z' ? 2 : 1;
    P->gc = gc;
    P->w = w;
    P->la = stored(gc->ta, M, K, gc->pad);
    P->lb = stored(gc->tb, K, N, gc->pad);
    P->lc = (Layout){M, N, M + gc->pad};
    const size_t na = nelem(P->la) * w, nb = nelem(P->lb) * w, nc = nelem(P->lc) * w;
    P->A = malloc(na * sizeof *P->A);  P->A0 = malloc(na * sizeof *P->A0);
    P->B = malloc(nb * sizeof *P->B);  P->B0 = malloc(nb * sizeof *P->B0);
    P->C0 = malloc(nc * sizeof *P->C0); P->C = malloc(nc * sizeof *P->C);

    sm_state = fnv1a(name, strlen(name), FNV_OFFSET);
    ValueKind va = V_UNIFORM, vb = V_UNIFORM, vc = V_UNIFORM;
    switch (gc->data) {
    case D_MIXED:       va = vb = vc = V_MIXED; break;
    case D_NAN_INF_C:   vc = V_NAN_INF; break;
    case D_ZERO_SUM:    va = V_GRID; vb = V_NEAR_ONE; break;
    case D_SIGNED_ZERO: vc = V_SIGNED_ZERO; break;
    default: break;
    }
    fill(P->A, P->la, w, va);
    fill(P->B, P->lb, w, vb);
    fill(P->C0, P->lc, w, vc);

    switch (gc->data) {
    case D_NEG_IMAG:                    // stored Ai = -Ar (1 + 2^-30 r)
        for (int j = 0; j < P->la.cols; j++)
            for (int i = 0; i < P->la.rows; i++) {
                double *a = &P->A[((size_t)j * P->la.ld + i) * 2];
                a[1] = -a[0] * (1.0 + ldexp(uniform(), -30));
            }
        break;
    case D_ZERO_SUM:                    // grid values: every sum below is exact
        for (int i = 0; i < M; i++)
            for (int p = 0; p < w; p++) {
                double s = 0.0;
                for (int k = 0; k < K - 1; k++) s += P->A[op_at(gc->ta, P->la, w, i, k) + p];
                P->A[op_at(gc->ta, P->la, w, i, K - 1) + p] = -s;
            }
        break;
    case D_SIGNED_ZERO:                 // rows 0-5 of op(A), columns 0-3 of op(B), 1/8 of the rest
        for (int i = 0; i < M; i++)
            for (int k = 0; k < K; k++)
                if (i < 6 || (splitmix64() & 7) == 0)
                    for (int p = 0; p < w; p++) P->A[op_at(gc->ta, P->la, w, i, k) + p] = signed_zero();
        for (int k = 0; k < K; k++)
            for (int j = 0; j < N; j++)
                if (j < 4 || (splitmix64() & 7) == 0)
                    for (int p = 0; p < w; p++) P->B[op_at(gc->tb, P->lb, w, k, j) + p] = signed_zero();
        break;
    default:
        break;
    }
    memcpy(P->A0, P->A, na * sizeof *P->A);
    memcpy(P->B0, P->B, nb * sizeof *P->B);
    memcpy(P->C, P->C0, nc * sizeof *P->C);
}

static void free_problem(Problem *P) {
    free(P->A); free(P->A0); free(P->B); free(P->B0); free(P->C0); free(P->C);
}

// =============================================================================
// The call and its checks
// =============================================================================

static void call_gemm(Problem *P) {
    const GoldenCase *gc = P->gc;
    const int M = gc->M, N = gc->N, K = gc->K;
    const int lda = P->la.ld, ldb = P->lb.ld, ldc = P->lc.ld;
    if (gc->routine == 'd') {
        const double alpha = creal(SCALARS[gc->sc].alpha), beta = creal(SCALARS[gc->sc].beta);
        fortran_dgemm(&gc->ta, &gc->tb, &M, &N, &K, &alpha, P->A, &lda, P->B, &ldb, &beta,
                      P->C, &ldc);
    } else {
        const double complex alpha = SCALARS[gc->sc].alpha, beta = SCALARS[gc->sc].beta;
        fortran_zgemm(&gc->ta, &gc->tb, &M, &N, &K, &alpha, (double complex *)P->A, &lda,
                      (double complex *)P->B, &ldb, &beta, (double complex *)P->C, &ldc);
    }
}

// C's componentwise error against cblas, as test_blas_abi.c measures it;
// NaN if any entry is NaN.
static double error_vs_cblas(const Problem *P) {
    const GoldenCase *gc = P->gc;
    return blas_error(P->w == 2, gc->ta, gc->tb, gc->M, gc->N, gc->K, SCALARS[gc->sc].alpha,
                      SCALARS[gc->sc].beta, P->A, P->la, P->B, P->lb, P->C0, P->lc, P->C);
}

typedef struct {
    char name[96];
    uint64_t hash;
    double err;
    char problem[96];       // empty if every check passed
} Result;

static void run_case(const GoldenCase *gc, Result *r) {
    Problem P;
    case_name(gc, r->name, sizeof r->name);
    make_problem(&P, gc, r->name);

    uint64_t g0 = gpu_gemm_count();
    call_gemm(&P);
    uint64_t gpu_gemms = gpu_gemm_count() - g0;

    const size_t na = nelem(P.la) * P.w, nb = nelem(P.lb) * P.w, nc = nelem(P.lc) * P.w;
    r->hash = fnv1a(P.C, nc * sizeof *P.C, FNV_OFFSET);
    r->err = error_vs_cblas(&P);
    r->problem[0] = '\0';
    if (strcmp(ab_get_last_dispatch_path(), "gpu") || gpu_gemms != 1)
        snprintf(r->problem, sizeof r->problem, "not computed on the GPU (path=%s gpu_gemms=%llu)",
                 ab_get_last_dispatch_path(), (unsigned long long)gpu_gemms);
    else if (memcmp(P.A, P.A0, na * sizeof *P.A) || memcmp(P.B, P.B0, nb * sizeof *P.B))
        snprintf(r->problem, sizeof r->problem, "A or B modified");
    else if (!outside_block_unchanged(P.C, P.C0, nelem(P.lc), P.lc, P.w * sizeof *P.C))
        snprintf(r->problem, sizeof r->problem, "padding of C modified");
    else if (!(r->err <= TOL))
        snprintf(r->problem, sizeof r->problem, "err %.1e vs cblas exceeds %.0e", r->err, TOL);
    free_problem(&P);
}

// =============================================================================
// Golden file
// =============================================================================

typedef struct {
    char device[128], os_build[64], library[128];
    int n;
    struct { char name[96]; uint64_t hash; bool used; } e[MAX_ENTRIES];
} Golden;

static void trim_newline(char *s) { s[strcspn(s, "\r\n")] = '\0'; }

// Header lines "device: ...", "os_build: ...", "library: ...", then one
// "<16 hex digits> <case name>" line per case; '#' starts a comment line.
static bool read_golden(const char *path, Golden *g, char *why, size_t n) {
    FILE *f = fopen(path, "r");
    if (!f) {
        snprintf(why, n, "cannot open %s", path);
        return false;
    }
    memset(g, 0, sizeof *g);
    char line[512];
    int lineno = 0;
    bool ok = true;
    while (ok && fgets(line, sizeof line, f)) {
        lineno++;
        trim_newline(line);
        char name[96];
        uint64_t h;
        int end = 0;
        if (!line[0] || line[0] == '#') continue;
        if (!strncmp(line, "device: ", 8))
            snprintf(g->device, sizeof g->device, "%s", line + 8);
        else if (!strncmp(line, "os_build: ", 10))
            snprintf(g->os_build, sizeof g->os_build, "%s", line + 10);
        else if (!strncmp(line, "library: ", 9))
            snprintf(g->library, sizeof g->library, "%s", line + 9);
        else if (sscanf(line, "%16" SCNx64 " %95s%n", &h, name, &end) == 2 && !line[end] &&
                 g->n < MAX_ENTRIES) {
            g->e[g->n].hash = h;
            snprintf(g->e[g->n].name, sizeof g->e[g->n].name, "%s", name);
            g->n++;
        } else {
            snprintf(why, n, "%s:%d: malformed line", path, lineno);
            ok = false;
        }
    }
    fclose(f);
    if (ok && (!g->device[0] || !g->os_build[0])) {
        snprintf(why, n, "%s: no device or os_build line", path);
        ok = false;
    }
    return ok;
}

static bool write_golden(const char *path, const char *device, const char *os_build,
                         const char *library, const Result *res) {
    char tmp[1024];
    snprintf(tmp, sizeof tmp, "%s.tmp", path);
    FILE *f = fopen(tmp, "w");
    if (!f) return false;
    fprintf(f, "# apple-bottom golden bits (tests/test_qe_golden.c): 64-bit FNV-1a of the whole\n"
               "# C array, ld padding included, after one GPU-routed dgemm_ / zgemm_ call.\n"
               "# library: ab_build_info() of the library that wrote the hashes. Regenerate\n"
               "# (new device or macOS build) only from that library, never from the tree under\n"
               "# test (tests/README.md, Golden bits):\n"
               "#   DYLD_LIBRARY_PATH=<its directory> build/test_qe_golden --generate\n"
               "# The tree's own library replaces it only for an intended change of results.\n");
    fprintf(f, "device: %s\nos_build: %s\nlibrary: %s\n", device, os_build, library);
    for (int i = 0; i < N_CASES; i++) fprintf(f, "%016" PRIx64 " %s\n", res[i].hash, res[i].name);
    bool ok = fclose(f) == 0;
    return ok && rename(tmp, path) == 0;
}

// =============================================================================
// Main
// =============================================================================

int main(int argc, char **argv) {
    bool generate = false;
    const char *path = GOLDEN_FILE;
    for (int i = 1; i < argc; i++) {
        if (!strcmp(argv[i], "--generate")) generate = true;
        else if (!strcmp(argv[i], "--file") && i + 1 < argc) path = argv[++i];
        else {
            fprintf(stderr, "usage: %s [--generate] [--file PATH]\n", argv[0]);
            return 2;
        }
    }
    const char *strict_env = getenv("AB_GOLDEN_STRICT");
    const bool strict = strict_env && !strcmp(strict_env, "1");

    // The routing settings are read at the first BLAS-ABI call: force the GPU
    // route (as test_blas_abi's gpu suites do) before any. No ab_init(): the
    // first call creates the Metal context, as in QE.
    clear_ab_env();
    setenv("AB_MODE", "gpu", 1);

    printf("╔══════════════════════════════════════════════════════════════════╗\n");
    printf("║  apple-bottom golden bits (GPU-routed dgemm_ / zgemm_, FNV-1a)   ║\n");
    printf("╚══════════════════════════════════════════════════════════════════╝\n");

    char os_build[64] = "unknown";
    size_t len = sizeof os_build;
    if (sysctlbyname("kern.osversion", os_build, &len, NULL, 0) != 0)
        snprintf(os_build, sizeof os_build, "unknown");
    // Version and commit ("apple-bottom 1.4.0 (c5a1fa6a8166)", -dirty for a
    // modified tree): a re-baseline from the tree under test shows in the file.
    const char *library = ab_build_info();

    Result *res = calloc(N_CASES, sizeof *res);
    int failed = 0;
    for (int i = 0; i < N_CASES; i++) {
        run_case(&CASES[i], &res[i]);
        failed += res[i].problem[0] != '\0';
    }
    // The first call created the Metal context ("Unknown" if there is none).
    const char *device = ab_device_name();
    printf("device: %s   os_build: %s   library: %s\n", device, os_build, library);

    if (generate) {
        for (int i = 0; i < N_CASES; i++)
            printf("  %-46s %016" PRIx64 "  err=%.1e  %s\n", res[i].name, res[i].hash, res[i].err,
                   res[i].problem[0] ? res[i].problem : "ok");
        if (failed) {
            printf("\n✗ %d case(s) failed their checks: %s not written\n", failed, path);
            return 1;
        }
        if (!write_golden(path, device, os_build, library, res)) {
            printf("\n✗ cannot write %s\n", path);
            return 1;
        }
        printf("\nWrote %d hashes from %s to %s\n", N_CASES, library, path);
        return 0;
    }

    Golden *g = calloc(1, sizeof *g);
    char why[1200] = "", skip[512] = "", other_os[160] = "";
    bool have_file = read_golden(path, g, why, sizeof why);
    if (have_file && strcmp(g->device, device))
        snprintf(skip, sizeof skip, "device '%s', golden file from '%s'", device, g->device);
    else if (have_file && strcmp(g->os_build, os_build))
        snprintf(other_os, sizeof other_os, "macOS build %s, golden file from %s", os_build,
                 g->os_build);
    if (have_file)
        printf("golden file: %s (%d hashes; %s, %s, %s)\n", path, g->n, g->device, g->os_build,
               g->library);
    const bool compare = have_file && !skip[0];

    int passed = 0;
    for (int i = 0; i < N_CASES; i++) {
        Result *r = &res[i];
        bool ok = !r->problem[0];
        const char *reason = r->problem;
        char buf[160];
        if (compare) {
            int k = 0;
            while (k < g->n && strcmp(g->e[k].name, r->name)) k++;
            if (k < g->n) g->e[k].used = true;
            if (ok && k == g->n) {
                ok = false;
                reason = "no golden hash (new case? regenerate with --generate)";
            } else if (ok && g->e[k].hash != r->hash) {
                ok = false;
                snprintf(buf, sizeof buf, "bits differ from golden %016" PRIx64, g->e[k].hash);
                reason = buf;
            }
        }
        passed += ok;
        printf("  %-46s %016" PRIx64 "  err=%.1e  %s%s\n", r->name, r->hash, r->err,
               ok ? "✓" : "✗ ", ok ? "" : reason);
    }
    int stale = 0;
    for (int k = 0; compare && k < g->n; k++)
        if (!g->e[k].used) {
            printf("  %-46s %016" PRIx64 "  ✗ golden entry matches no case\n", g->e[k].name,
                   g->e[k].hash);
            stale++;
        }

    printf("\n═══════════════════════════════════════════════════════════════════\n");
    printf("Results: %d passed, %d failed, %d stale golden entries\n", passed, N_CASES - passed,
           stale);
    int rc = 0;
    if (!have_file) {
        printf("Golden bits: FAIL (%s)\n", why);
        rc = 1;
    } else if (skip[0]) {
        bool bad = passed != N_CASES || strict;
        printf("Golden bits: %s (%s; %d cases checked against cblas only%s)\n",
               bad ? "FAIL" : "SKIP", skip, N_CASES, strict ? ", AB_GOLDEN_STRICT=1" : "");
        rc = bad;
    } else if (passed != N_CASES || stale) {
        printf("Golden bits: FAIL (%d of %d cases differ or fail, %d stale golden entries%s%s)\n",
               N_CASES - passed, N_CASES, stale, other_os[0] ? "; " : "", other_os);
        if (other_os[0])
            printf("If %s gives these bits on this macOS build too, re-baseline from it "
                   "(tests/README.md, Golden bits).\n", g->library);
        rc = 1;
    } else {
        printf("Golden bits: PASS (%d cases bitwise equal to %s%s%s)\n", N_CASES, g->library,
               other_os[0] ? "; " : "", other_os);
    }
    free(g);
    free(res);
    return rc;
}
