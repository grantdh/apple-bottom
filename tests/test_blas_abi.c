// =============================================================================
// apple-bottom BLAS-ABI Conformance Harness
// =============================================================================
//
// Reference: Accelerate cblas_dgemm / cblas_zgemm in CblasColMajor.
// Entry points under test:
//   ab_dgemm_blas / ab_zgemm_blas    C BLAS-ABI entry points (blas_wrapper.c)
//   ab_dgemm_ / ab_zgemm_            Fortran-ABI wrappers (fortran_bridge.c)
//   dgemm_ / zgemm_                  the plain Fortran symbols a gfortran call
//                                    site binds to; whether they bind to
//                                    libapplebottom is the symbol suite's job
//   ab_dgemm, ab_dgemm_scaled, ab_zgemm, ab_batch_dgemm, ab_dgemm_async
//                                    native ABMatrix API on Morton-sensitive grids
//
// A case passes when
//   max_ij |C_ij - Cref_ij| / (|alpha| (|op(A)| |op(B)|)_ij + |beta| |C0_ij| + tiny) <= 1e-11,
// A and B are bitwise unchanged, C outside its M x N block is bitwise
// unchanged, and the case's routing expectation holds (ROUTE_GPU: the GEMM
// counters in ab_get_stats() advanced and ab_get_last_dispatch_path() is
// "gpu"; ROUTE_CPU: the counters did not advance). Leading-dimension padding
// is filled with NaN so that reading it shows up in the result.
//
// Isolation: every suite runs in a child process (this binary re-executed
// with --child lo hi), because
//   - AB_MODE is read once and cached (ab_get_mode in blas_wrapper.c,
//     fb_get_mode in fortran_bridge.c), so each mode needs a fresh process;
//   - the lazy-init suite must run in a process that never called ab_init();
//   - known-bad cases crash (SIGSEGV, Metal API-validation SIGABRT, exit(255)
//     from cblas parameter checks). A crash fails only the case in progress;
//     the parent restarts the child at the next case.
// posix_spawn is used rather than a bare fork(): Metal cannot reach
// MTLCompilerService from a forked child that did not exec.
//
// Known failures live in the XFAIL table below, tagged with audit IDs. An
// XFAIL does not fail the run; an XFAIL case that passes (XPASS), an XFAIL
// entry that matches no case (STALE) and any other failure do. Run with -v
// to list every case.
// =============================================================================

#include "apple_bottom.h"
#include "test_registry.h"
#include <Accelerate/Accelerate.h>
#include <complex.h>
#include <dlfcn.h>
#include <errno.h>
#include <fnmatch.h>
#include <float.h>
#include <limits.h>
#include <mach-o/dyld.h>
#include <math.h>
#include <spawn.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/wait.h>
#include <unistd.h>

extern char **environ;

// Exported by fortran_bridge.c but not declared in apple_bottom.h.
void ab_dgemm_(const char *transA, const char *transB,
               const int *M, const int *N, const int *K,
               const double *alpha, const double *A, const int *ldA,
               const double *B, const int *ldB,
               const double *beta, double *C, const int *ldC);
void ab_zgemm_(const char *transA, const char *transB,
               const int *M, const int *N, const int *K,
               const double complex *alpha, const double complex *A, const int *ldA,
               const double complex *B, const int *ldB,
               const double complex *beta, double complex *C, const int *ldC);

// The plain Fortran BLAS symbols, bound by their exact Mach-O names. Under
// ACCELERATE_NEW_LAPACK a C reference to dgemm_ becomes _dgemm$NEWLAPACK,
// which is not the symbol a gfortran call site links against.
void fortran_dgemm(const char *transA, const char *transB,
                   const int *M, const int *N, const int *K,
                   const double *alpha, const double *A, const int *ldA,
                   const double *B, const int *ldB,
                   const double *beta, double *C, const int *ldC) __asm__("_dgemm_");
void fortran_zgemm(const char *transA, const char *transB,
                   const int *M, const int *N, const int *K,
                   const double complex *alpha, const double complex *A, const int *ldA,
                   const double complex *B, const int *ldB,
                   const double complex *beta, double complex *C, const int *ldC) __asm__("_zgemm_");

#define TOL             1e-11
#define MAX_CASES       1024
#define MIN_GPU_DIM     32      // DEFAULT_MIN_GPU_DIM in blas_wrapper.c
#define CASE_TIMEOUT_S  120

// =============================================================================
// Case model
// =============================================================================

typedef enum {
    OP_DGEMM_BLAS, OP_ZGEMM_BLAS,       // C BLAS-ABI entry points
    OP_DGEMM_F,    OP_ZGEMM_F,          // ab_dgemm_ / ab_zgemm_
    OP_DGEMM_SYM,  OP_ZGEMM_SYM,        // plain dgemm_ / zgemm_
    OP_DGEMM, OP_DGEMM_SCALED, OP_ZGEMM, OP_BATCH_DGEMM, OP_DGEMM_ASYNC,
    OP_SYMBOL,                          // which image a symbol binds to
} Op;

static const char *const OP_NAME[] = {
    [OP_DGEMM_BLAS] = "ab_dgemm_blas", [OP_ZGEMM_BLAS] = "ab_zgemm_blas",
    [OP_DGEMM_F] = "ab_dgemm_",        [OP_ZGEMM_F] = "ab_zgemm_",
    [OP_DGEMM_SYM] = "dgemm_",         [OP_ZGEMM_SYM] = "zgemm_",
    [OP_DGEMM] = "ab_dgemm",           [OP_DGEMM_SCALED] = "ab_dgemm_scaled",
    [OP_ZGEMM] = "ab_zgemm",           [OP_BATCH_DGEMM] = "ab_batch_dgemm",
    [OP_DGEMM_ASYNC] = "ab_dgemm_async", [OP_SYMBOL] = "symbol",
};

typedef enum { ROUTE_ANY, ROUTE_GPU, ROUTE_CPU } Route;
typedef enum { POISON_NONE, POISON_AB, POISON_C } Poison;

typedef struct {
    const char *tag;
    double complex alpha, beta;         // real entry points use the real parts
    Poison poison;
} Scalars;

enum { SC_GEN, SC_A1B0, SC_B1, SC_B0_NAN_C, SC_A0_NAN_AB };

static const Scalars SCALARS[] = {
    [SC_GEN]       = {"gen",     CMPLX(1.25, -0.5), CMPLX(-0.75, 0.25), POISON_NONE},
    [SC_A1B0]      = {"a1b0",    CMPLX(1.0, 0.0),   CMPLX(0.0, 0.0),    POISON_NONE},
    [SC_B1]        = {"b1",      CMPLX(1.25, -0.5), CMPLX(1.0, 0.0),    POISON_NONE},
    [SC_B0_NAN_C]  = {"b0nanC",  CMPLX(1.25, -0.5), CMPLX(0.0, 0.0),    POISON_C},   // C must not be read
    [SC_A0_NAN_AB] = {"a0nanAB", CMPLX(0.0, 0.0),   CMPLX(0.5, -0.25),  POISON_AB},  // A, B must not be read
};

typedef struct {
    char name[80];
    Op op;
    char ta, tb;
    int M, N, K, pad;
    double complex alpha, beta;
    Poison poison;
    Route route;
    const void *sym;                    // OP_SYMBOL only
} AbiCase;

static AbiCase CASES[MAX_CASES];
static int n_cases;

// =============================================================================
// XFAIL table: known failures, first match wins (fnmatch, FNM_PATHNAME).
// Case names: <entry>/<mode>/<transA><transB>/<M>x<N>x<K>/ld+<pad>/<scalars>
// for BLAS-ABI cases, <entry>/<M>x<N>x<K> for native cases.
// =============================================================================

typedef struct {
    const char *pattern;
    const char *id;
} XfailEntry;

static const XfailEntry XFAIL[] = {
    // No lazy init: AB_MODE=gpu without ab_init() silently runs cblas.
    {"*/gpu-noinit/*/*/*/*",                    "H-P1-2"},
};

static int xfail_hits[TEST_REGISTRY_SIZE(XFAIL)];

// Index of the first XFAIL entry matching `name`, or -1.
static int xfail_lookup(const char *name) {
    for (size_t i = 0; i < TEST_REGISTRY_SIZE(XFAIL); i++)
        if (fnmatch(XFAIL[i].pattern, name, FNM_PATHNAME) == 0) return (int)i;
    return -1;
}

// =============================================================================
// Suites (one child process each) and case generation
// =============================================================================

typedef struct Suite Suite;
struct Suite {
    const char *label;          // mode segment of BLAS-ABI case names
    const char *ab_mode;        // AB_MODE for the child (NULL: unset)
    bool init;                  // call ab_init() before the first case
    bool debug_layer;           // MTL_DEBUG_LAYER=1 (Metal API validation)
    Op op;                      // entry point (gen_blas suites only)
    void (*gen)(const Suite *s);
    int lo, hi;                 // case range, set by build_cases()
};

static int imax(int a, int b) { return a > b ? a : b; }

static bool op_is_complex(Op op) {
    return op == OP_ZGEMM_BLAS || op == OP_ZGEMM_F || op == OP_ZGEMM_SYM;
}

static AbiCase *new_case(void) {
    if (n_cases == MAX_CASES) {
        fprintf(stderr, "test_blas_abi: MAX_CASES exceeded\n");
        abort();
    }
    AbiCase *c = &CASES[n_cases++];
    memset(c, 0, sizeof *c);
    return c;
}

static void add_blas(const Suite *s, Op op, char ta, char tb, int M, int N, int K,
                     int pad, int scalars, Route route) {
    const Scalars *sc = &SCALARS[scalars];
    AbiCase *c = new_case();
    snprintf(c->name, sizeof c->name, "%s/%s/%c%c/%dx%dx%d/ld+%d/%s",
             OP_NAME[op], s->label, ta, tb, M, N, K, pad, sc->tag);
    c->op = op; c->ta = ta; c->tb = tb;
    c->M = M; c->N = N; c->K = K; c->pad = pad;
    c->alpha = sc->alpha; c->beta = sc->beta; c->poison = sc->poison;
    c->route = route;
}

// gen_blas suites run with AB_MODE=gpu or cpu. In gpu mode the GPU must be
// used unless a dimension is <= AB_MIN_GPU_DIM or alpha == 0 (BLAS quick
// return is allowed).
static void add_std(const Suite *s, char ta, char tb, int M, int N, int K,
                    int pad, int scalars) {
    Route route = ROUTE_GPU;
    if (!strcmp(s->ab_mode, "cpu"))
        route = ROUTE_CPU;
    else if (SCALARS[scalars].alpha == 0.0 ||
             M <= MIN_GPU_DIM || N <= MIN_GPU_DIM || K <= MIN_GPU_DIM)
        route = ROUTE_ANY;
    add_blas(s, s->op, ta, tb, M, N, K, pad, scalars, route);
}

static void gen_blas(const Suite *s) {
    static const char TRANS[] = "NTCntc";
    for (const char *ta = TRANS; *ta; ta++)
        for (const char *tb = TRANS; *tb; tb++)
            add_std(s, *ta, *tb, 96, 80, 72, 3, SC_GEN);

    add_std(s, 'N', 'N', 96, 80, 72, 0, SC_GEN);                     // tight ld
    add_std(s, op_is_complex(s->op) ? 'C' : 'T', 'N',
            150, 150, 2000, 0, SC_A1B0);                             // QE cegterg
    add_std(s, 'N', 'T', 48, 40, 1500, 3, SC_GEN);                   // K >> M, N
    add_std(s, 'T', 'N', 600, 48, 40, 3, SC_GEN);                    // M >> K
    add_std(s, 'N', 'N', 512, 256, 64, 0, SC_GEN);                   // grid 4x8
    add_std(s, 'N', 'N', 1024, 64, 64, 0, SC_GEN);                   // TS grid 4x8

    for (int sc = SC_A1B0; sc <= SC_A0_NAN_AB; sc++)                 // alpha/beta specials
        add_std(s, 'N', 'N', 96, 80, 72, 3, sc);

    add_std(s, 'N', 'N', 0, 80, 72, 3, SC_GEN);                      // quick returns
    add_std(s, 'N', 'N', 96, 0, 72, 3, SC_GEN);
    add_std(s, 'N', 'N', 96, 80, 0, 3, SC_GEN);
}

// AB_MODE=auto: the QE cegterg call crosses both FLOP thresholds (3.6e8 >= 1e8).
static void gen_auto(const Suite *s) {
    add_blas(s, OP_ZGEMM_BLAS, 'C', 'N', 150, 150, 2000, 0, SC_A1B0, ROUTE_GPU);
    add_blas(s, OP_ZGEMM_F,    'C', 'N', 150, 150, 2000, 0, SC_A1B0, ROUTE_GPU);
    add_blas(s, OP_ZGEMM_SYM,  'C', 'N', 150, 150, 2000, 0, SC_A1B0, ROUTE_GPU);
    add_blas(s, OP_DGEMM_BLAS, 'T', 'N', 150, 150, 2000, 0, SC_A1B0, ROUTE_GPU);
}

// AB_MODE=gpu without ab_init(): the GPU path must still be taken.
static void gen_noinit(const Suite *s) {
    add_blas(s, OP_DGEMM_BLAS, 'N', 'N', 96, 80, 72, 3, SC_GEN, ROUTE_GPU);
    add_blas(s, OP_ZGEMM_BLAS, 'N', 'N', 96, 80, 72, 3, SC_GEN, ROUTE_GPU);
    add_blas(s, OP_DGEMM_F,    'N', 'N', 96, 80, 72, 3, SC_GEN, ROUTE_GPU);
    add_blas(s, OP_ZGEMM_F,    'N', 'N', 96, 80, 72, 3, SC_GEN, ROUTE_GPU);
    add_blas(s, OP_ZGEMM_BLAS, 'C', 'N', 150, 150, 2000, 0, SC_A1B0, ROUTE_GPU);
}

static void add_native(Op op, int M, int N, int K) {
    AbiCase *c = new_case();
    snprintf(c->name, sizeof c->name, "%s/%dx%dx%d", OP_NAME[op], M, N, K);
    c->op = op; c->M = M; c->N = N; c->K = K;
    c->alpha = SCALARS[SC_GEN].alpha; c->beta = SCALARS[SC_GEN].beta;   // ab_dgemm_scaled only
}

// Native ABMatrix GEMMs (row-major) on tile grids that are powers of two but
// neither square nor 2:1 wide (K-P0-1 regressions; 1024x64 and 2048x128 use the
// 128x16 tall-skinny tiles where the entry point routes M >= 4N there).
// 256x512 (grid 8x4) is a control. The suite runs under MTL_DEBUG_LAYER=1,
// which also catches unbound kernel buffers (K-P1-3).
static void gen_native(const Suite *s) {
    (void)s;
    static const int SHAPES[][2] = {
        {512, 256}, {256, 1024}, {1024, 256}, {1024, 64}, {2048, 128}, {256, 512},
    };
    static const Op OPS[] = {
        OP_DGEMM, OP_DGEMM_SCALED, OP_ZGEMM, OP_BATCH_DGEMM, OP_DGEMM_ASYNC,
    };
    for (size_t o = 0; o < TEST_REGISTRY_SIZE(OPS); o++)
        for (size_t i = 0; i < TEST_REGISTRY_SIZE(SHAPES); i++)
            add_native(OPS[o], SHAPES[i][0], SHAPES[i][1], 64);
}

static void add_symbol(const char *sym, const void *addr) {
    AbiCase *c = new_case();
    snprintf(c->name, sizeof c->name, "symbol/%s", sym);
    c->op = OP_SYMBOL;
    c->sym = addr;
}

// The Fortran symbols must bind to libapplebottom; ab_*gemm_ are controls.
static void gen_symbols(const Suite *s) {
    (void)s;
    add_symbol("dgemm_", (const void *)fortran_dgemm);
    add_symbol("zgemm_", (const void *)fortran_zgemm);
    add_symbol("ab_dgemm_", (const void *)ab_dgemm_);
    add_symbol("ab_zgemm_", (const void *)ab_zgemm_);
}

enum {
    SU_DGEMM_BLAS_GPU, SU_DGEMM_BLAS_CPU, SU_ZGEMM_BLAS_GPU, SU_ZGEMM_BLAS_CPU,
    SU_DGEMM_F_GPU, SU_DGEMM_F_CPU, SU_ZGEMM_F_GPU, SU_ZGEMM_F_CPU,
    SU_DGEMM_SYM_GPU, SU_DGEMM_SYM_CPU, SU_ZGEMM_SYM_GPU, SU_ZGEMM_SYM_CPU,
    SU_AUTO, SU_NOINIT, SU_NATIVE, SU_SYMBOLS, SU_COUNT
};

#define BLAS_SUITE(entry, mode) \
    {.label = mode, .ab_mode = mode, .init = true, .op = entry, .gen = gen_blas}

static Suite SUITES[SU_COUNT] = {
    [SU_DGEMM_BLAS_GPU] = BLAS_SUITE(OP_DGEMM_BLAS, "gpu"),
    [SU_DGEMM_BLAS_CPU] = BLAS_SUITE(OP_DGEMM_BLAS, "cpu"),
    [SU_ZGEMM_BLAS_GPU] = BLAS_SUITE(OP_ZGEMM_BLAS, "gpu"),
    [SU_ZGEMM_BLAS_CPU] = BLAS_SUITE(OP_ZGEMM_BLAS, "cpu"),
    [SU_DGEMM_F_GPU]    = BLAS_SUITE(OP_DGEMM_F, "gpu"),
    [SU_DGEMM_F_CPU]    = BLAS_SUITE(OP_DGEMM_F, "cpu"),
    [SU_ZGEMM_F_GPU]    = BLAS_SUITE(OP_ZGEMM_F, "gpu"),
    [SU_ZGEMM_F_CPU]    = BLAS_SUITE(OP_ZGEMM_F, "cpu"),
    [SU_DGEMM_SYM_GPU]  = BLAS_SUITE(OP_DGEMM_SYM, "gpu"),
    [SU_DGEMM_SYM_CPU]  = BLAS_SUITE(OP_DGEMM_SYM, "cpu"),
    [SU_ZGEMM_SYM_GPU]  = BLAS_SUITE(OP_ZGEMM_SYM, "gpu"),
    [SU_ZGEMM_SYM_CPU]  = BLAS_SUITE(OP_ZGEMM_SYM, "cpu"),
    [SU_AUTO]    = {.label = "auto", .ab_mode = "auto", .init = true, .gen = gen_auto},
    [SU_NOINIT]  = {.label = "gpu-noinit", .ab_mode = "gpu", .gen = gen_noinit},
    [SU_NATIVE]  = {.init = true, .debug_layer = true, .gen = gen_native},
    [SU_SYMBOLS] = {.gen = gen_symbols},
};

static void build_cases(void) {
    for (int i = 0; i < SU_COUNT; i++) {
        SUITES[i].lo = n_cases;
        SUITES[i].gen(&SUITES[i]);
        SUITES[i].hi = n_cases;
    }
}

// =============================================================================
// Child side: data, reference, metric
// =============================================================================

typedef struct { int rows, cols, ld; } Layout;   // column-major storage

static bool is_notrans(char t) { return t == 'N' || t == 'n'; }

// Storage of an operand whose op() is op_rows x op_cols.
static Layout stored(char t, int op_rows, int op_cols, int pad) {
    Layout l = {is_notrans(t) ? op_rows : op_cols, is_notrans(t) ? op_cols : op_rows, 0};
    l.ld = imax(1, l.rows + pad);
    return l;
}

static size_t nelem(Layout l) {
    size_t n = (size_t)l.ld * l.cols;
    return n ? n : 1;
}

static enum CBLAS_TRANSPOSE ref_trans(char t, bool cplx) {
    switch (t) {
    case 'N': case 'n': return CblasNoTrans;
    case 'T': case 't': return CblasTrans;
    default:            return cplx ? CblasConjTrans : CblasTrans;   // 'C', 'c'
    }
}

static uint64_t rng_state;
static double rnd(void) {
    rng_state ^= rng_state >> 12;
    rng_state ^= rng_state << 25;
    rng_state ^= rng_state >> 27;
    return (double)((rng_state * 0x2545F4914F6CDD1DULL) >> 11) * 0x1.0p-52 - 1.0;
}

// Valid block uniform in [-1, 1); padding (and everything when all_nan) NaN.
static void fill_real(double *X, Layout l, bool all_nan) {
    for (size_t i = 0; i < nelem(l); i++) X[i] = NAN;
    if (all_nan) return;
    for (int j = 0; j < l.cols; j++)
        for (int i = 0; i < l.rows; i++) X[(size_t)j * l.ld + i] = rnd();
}

static void fill_complex(double complex *X, Layout l, bool all_nan) {
    for (size_t i = 0; i < nelem(l); i++) X[i] = CMPLX(NAN, NAN);
    if (all_nan) return;
    for (int j = 0; j < l.cols; j++)
        for (int i = 0; i < l.rows; i++) {
            double re = rnd();
            X[(size_t)j * l.ld + i] = CMPLX(re, rnd());
        }
}

// True if X equals X0 bitwise everywhere outside the rows x cols block.
static bool outside_block_unchanged(const void *X, const void *X0, size_t total,
                                    Layout l, size_t es) {
    const char *x = X, *x0 = X0;
    size_t pos = 0;
    for (int j = 0; j < l.cols; j++) {
        size_t start = (size_t)j * l.ld;
        if (memcmp(x + pos * es, x0 + pos * es, (start - pos) * es)) return false;
        pos = start + l.rows;
    }
    return !memcmp(x + pos * es, x0 + pos * es, (total - pos) * es);
}

// out (rows x cols, ld rows) = |op(X)| elementwise.
static void abs_op(const double *X, Layout l, char t, int rows, int cols, double *out) {
    for (int j = 0; j < cols; j++)
        for (int i = 0; i < rows; i++)
            out[(size_t)j * rows + i] = fabs(is_notrans(t) ? X[(size_t)j * l.ld + i]
                                                           : X[(size_t)i * l.ld + j]);
}

// out (M x N, ld M) = |op(A)| |op(B)|; M, N, K > 0. Complex operands are
// passed as their elementwise moduli (see moduli()).
static void abs_product(const double *A, Layout la, char ta,
                        const double *B, Layout lb, char tb,
                        int M, int N, int K, double *out) {
    double *pa = malloc((size_t)M * K * sizeof *pa), *pb = malloc((size_t)K * N * sizeof *pb);
    abs_op(A, la, ta, M, K, pa);
    abs_op(B, lb, tb, K, N, pb);
    cblas_dgemm(CblasColMajor, CblasNoTrans, CblasNoTrans, M, N, K,
                1.0, pa, M, pb, K, 0.0, out, M);
    free(pa); free(pb);
}

static double *moduli(const double complex *z, size_t count) {
    double *m = malloc(count * sizeof *m);
    for (size_t k = 0; k < count; k++) m[k] = cabs(z[k]);
    return m;
}

// Componentwise error over the M x N block; NaN if any entry is NaN.
static double err_real(int M, int N, const double *C, const double *Cref, int ldc,
                       const double *absAB, int ldab, const double *C0,
                       double alpha, double beta) {
    double worst = 0.0;
    for (int j = 0; j < N; j++)
        for (int i = 0; i < M; i++) {
            size_t k = (size_t)j * ldc + i;
            double d = DBL_MIN;
            if (alpha != 0.0) d += fabs(alpha) * absAB[(size_t)j * ldab + i];
            if (beta != 0.0) d += fabs(beta) * fabs(C0[k]);
            double e = fabs(C[k] - Cref[k]) / d;
            if (isnan(e)) return NAN;
            if (e > worst) worst = e;
        }
    return worst;
}

static double err_complex(int M, int N, const double complex *C, const double complex *Cref,
                          int ldc, const double *absAB, int ldab, const double complex *C0,
                          double complex alpha, double complex beta) {
    double worst = 0.0;
    for (int j = 0; j < N; j++)
        for (int i = 0; i < M; i++) {
            size_t k = (size_t)j * ldc + i;
            double d = DBL_MIN;
            if (alpha != 0.0) d += cabs(alpha) * absAB[(size_t)j * ldab + i];
            if (beta != 0.0) d += cabs(beta) * cabs(C0[k]);
            double e = cabs(C[k] - Cref[k]) / d;
            if (isnan(e)) return NAN;
            if (e > worst) worst = e;
        }
    return worst;
}

static uint64_t gpu_gemm_count(void) {
    ABStats s = ab_get_stats();
    return s.dgemm_count + s.zgemm_count;
}

static bool blas_verdict(const AbiCase *c, double err, bool inputs_ok, bool outside_ok,
                         uint64_t gpu_gemms, char *detail, size_t n) {
    const char *path = ab_get_last_dispatch_path();
    bool routed = c->route == ROUTE_ANY ||
                  (c->route == ROUTE_GPU && gpu_gemms > 0 && !strcmp(path, "gpu")) ||
                  (c->route == ROUTE_CPU && gpu_gemms == 0);
    snprintf(detail, n, "err=%.1e path=%s gpu_gemms=%llu%s%s%s", err, path,
             (unsigned long long)gpu_gemms,
             inputs_ok ? "" : " [A or B modified]",
             outside_ok ? "" : " [C outside MxN modified]",
             routed ? "" : c->route == ROUTE_GPU ? " [GPU path not taken]"
                                                 : " [GPU used in cpu mode]");
    return err <= TOL && inputs_ok && outside_ok && routed;
}

static bool run_blas_real(const AbiCase *c, char *detail, size_t n) {
    const int M = c->M, N = c->N, K = c->K;
    const double alpha = creal(c->alpha), beta = creal(c->beta);
    const Layout la = stored(c->ta, M, K, c->pad), lb = stored(c->tb, K, N, c->pad);
    const Layout lc = {M, N, imax(1, M + c->pad)};
    const size_t na = nelem(la), nb = nelem(lb), nc = nelem(lc);
    double *A = malloc(na * sizeof *A), *A0 = malloc(na * sizeof *A0);
    double *B = malloc(nb * sizeof *B), *B0 = malloc(nb * sizeof *B0);
    double *C = malloc(nc * sizeof *C), *C0 = malloc(nc * sizeof *C0);
    double *Cref = malloc(nc * sizeof *Cref);
    double *absAB = calloc((size_t)imax(1, M) * imax(1, N), sizeof *absAB);

    fill_real(A, la, c->poison == POISON_AB);
    fill_real(B, lb, c->poison == POISON_AB);
    fill_real(C0, lc, c->poison == POISON_C);
    memcpy(A0, A, na * sizeof *A);
    memcpy(B0, B, nb * sizeof *B);
    memcpy(C, C0, nc * sizeof *C);
    memcpy(Cref, C0, nc * sizeof *Cref);

    if (M > 0 && N > 0) {
        for (int j = 0; j < N; j++)
            for (int i = 0; i < M; i++) {
                size_t k = (size_t)j * lc.ld + i;
                Cref[k] = beta == 0.0 ? 0.0 : beta * C0[k];
            }
        if (alpha != 0.0 && K > 0) {
            cblas_dgemm(CblasColMajor, ref_trans(c->ta, false), ref_trans(c->tb, false),
                        M, N, K, alpha, A, la.ld, B, lb.ld, 1.0, Cref, lc.ld);
            abs_product(A, la, c->ta, B, lb, c->tb, M, N, K, absAB);
        }
    }

    uint64_t g0 = gpu_gemm_count();
    switch (c->op) {
    case OP_DGEMM_BLAS:
        ab_dgemm_blas(c->ta, c->tb, M, N, K, alpha, A, la.ld, B, lb.ld, beta, C, lc.ld);
        break;
    case OP_DGEMM_F:
        ab_dgemm_(&c->ta, &c->tb, &M, &N, &K, &alpha, A, &la.ld, B, &lb.ld, &beta, C, &lc.ld);
        break;
    default:
        fortran_dgemm(&c->ta, &c->tb, &M, &N, &K, &alpha, A, &la.ld, B, &lb.ld, &beta, C, &lc.ld);
        break;
    }
    uint64_t gpu_gemms = gpu_gemm_count() - g0;

    double err = err_real(M, N, C, Cref, lc.ld, absAB, imax(1, M), C0, alpha, beta);
    bool inputs_ok = !memcmp(A, A0, na * sizeof *A) && !memcmp(B, B0, nb * sizeof *B);
    bool outside_ok = outside_block_unchanged(C, C0, nc, lc, sizeof *C);
    bool ok = blas_verdict(c, err, inputs_ok, outside_ok, gpu_gemms, detail, n);

    free(A); free(A0); free(B); free(B0); free(C); free(C0); free(Cref); free(absAB);
    return ok;
}

static bool run_blas_complex(const AbiCase *c, char *detail, size_t n) {
    const int M = c->M, N = c->N, K = c->K;
    const double complex alpha = c->alpha, beta = c->beta;
    const Layout la = stored(c->ta, M, K, c->pad), lb = stored(c->tb, K, N, c->pad);
    const Layout lc = {M, N, imax(1, M + c->pad)};
    const size_t na = nelem(la), nb = nelem(lb), nc = nelem(lc);
    double complex *A = malloc(na * sizeof *A), *A0 = malloc(na * sizeof *A0);
    double complex *B = malloc(nb * sizeof *B), *B0 = malloc(nb * sizeof *B0);
    double complex *C = malloc(nc * sizeof *C), *C0 = malloc(nc * sizeof *C0);
    double complex *Cref = malloc(nc * sizeof *Cref);
    double *absAB = calloc((size_t)imax(1, M) * imax(1, N), sizeof *absAB);

    fill_complex(A, la, c->poison == POISON_AB);
    fill_complex(B, lb, c->poison == POISON_AB);
    fill_complex(C0, lc, c->poison == POISON_C);
    memcpy(A0, A, na * sizeof *A);
    memcpy(B0, B, nb * sizeof *B);
    memcpy(C, C0, nc * sizeof *C);
    memcpy(Cref, C0, nc * sizeof *Cref);

    if (M > 0 && N > 0) {
        for (int j = 0; j < N; j++)
            for (int i = 0; i < M; i++) {
                size_t k = (size_t)j * lc.ld + i;
                Cref[k] = beta == 0.0 ? 0.0 : beta * C0[k];
            }
        if (alpha != 0.0 && K > 0) {
            const double complex one = 1.0;
            cblas_zgemm(CblasColMajor, ref_trans(c->ta, true), ref_trans(c->tb, true),
                        M, N, K, &alpha, A, la.ld, B, lb.ld, &one, Cref, lc.ld);
            double *mA = moduli(A, na), *mB = moduli(B, nb);
            abs_product(mA, la, c->ta, mB, lb, c->tb, M, N, K, absAB);
            free(mA); free(mB);
        }
    }

    uint64_t g0 = gpu_gemm_count();
    switch (c->op) {
    case OP_ZGEMM_BLAS:
        ab_zgemm_blas(c->ta, c->tb, M, N, K, alpha, A, la.ld, B, lb.ld, beta, C, lc.ld);
        break;
    case OP_ZGEMM_F:
        ab_zgemm_(&c->ta, &c->tb, &M, &N, &K, &alpha, A, &la.ld, B, &lb.ld, &beta, C, &lc.ld);
        break;
    default:
        fortran_zgemm(&c->ta, &c->tb, &M, &N, &K, &alpha, A, &la.ld, B, &lb.ld, &beta, C, &lc.ld);
        break;
    }
    uint64_t gpu_gemms = gpu_gemm_count() - g0;

    double err = err_complex(M, N, C, Cref, lc.ld, absAB, imax(1, M), C0, alpha, beta);
    bool inputs_ok = !memcmp(A, A0, na * sizeof *A) && !memcmp(B, B0, nb * sizeof *B);
    bool outside_ok = outside_block_unchanged(C, C0, nc, lc, sizeof *C);
    bool ok = blas_verdict(c, err, inputs_ok, outside_ok, gpu_gemms, detail, n);

    free(A); free(A0); free(B); free(B0); free(C); free(C0); free(Cref); free(absAB);
    return ok;
}

static ABStatus native_dgemm(Op op, double alpha, ABMatrix A, ABMatrix B, double beta,
                             ABMatrix C) {
    ABStatus st;
    switch (op) {
    case OP_DGEMM:
        return ab_dgemm(A, B, C);
    case OP_DGEMM_SCALED:
        return ab_dgemm_scaled(alpha, A, B, beta, C);
    case OP_BATCH_DGEMM: {
        ABBatch batch = ab_batch_create();
        if (!batch) return AB_ERROR_NO_DEVICE;
        st = ab_batch_dgemm(batch, A, B, C);
        if (st == AB_OK) st = ab_batch_commit(batch);
        if (st == AB_OK) st = ab_batch_wait(batch);
        ab_batch_destroy(batch);
        return st;
    }
    default: {
        ABFuture f = ab_dgemm_async(A, B, C);
        if (!f) return AB_ERROR_INVALID_ARG;
        st = ab_future_wait(f);
        ab_future_destroy(f);
        return st;
    }
    }
}

// Row-major M x K, K x N and M x N buffers are column-major K x M, N x K and
// N x M, so the native C = A*B is the column-major GEMM C^T = B^T A^T.
static bool run_native_real(const AbiCase *c, char *detail, size_t n) {
    const int M = c->M, N = c->N, K = c->K;
    const bool scaled = c->op == OP_DGEMM_SCALED;
    const double alpha = scaled ? creal(c->alpha) : 1.0, beta = scaled ? creal(c->beta) : 0.0;
    const Layout la = {K, M, K}, lb = {N, K, N}, lc = {N, M, N};
    double *A = malloc(nelem(la) * sizeof *A), *B = malloc(nelem(lb) * sizeof *B);
    double *C0 = malloc(nelem(lc) * sizeof *C0), *C = malloc(nelem(lc) * sizeof *C);
    double *Cref = malloc(nelem(lc) * sizeof *Cref);
    double *absAB = calloc(nelem(lc), sizeof *absAB);

    fill_real(A, la, false);
    fill_real(B, lb, false);
    fill_real(C0, lc, false);
    memcpy(Cref, C0, nelem(lc) * sizeof *Cref);
    cblas_dgemm(CblasColMajor, CblasNoTrans, CblasNoTrans, N, M, K,
                alpha, B, N, A, K, beta, Cref, N);
    abs_product(B, lb, 'N', A, la, 'N', N, M, K, absAB);

    ABMatrix mA = ab_matrix_create(M, K), mB = ab_matrix_create(K, N);
    ABMatrix mC = ab_matrix_create(M, N);
    ABStatus st = (mA && mB && mC) ? AB_OK : AB_ERROR_ALLOC_FAILED;
    if (st == AB_OK) st = ab_matrix_upload(mA, A, false);
    if (st == AB_OK) st = ab_matrix_upload(mB, B, false);
    if (st == AB_OK) st = ab_matrix_upload(mC, C0, false);
    if (st == AB_OK) st = native_dgemm(c->op, alpha, mA, mB, beta, mC);
    if (st == AB_OK) st = ab_matrix_download(mC, C, false);

    double err = st == AB_OK ? err_real(N, M, C, Cref, N, absAB, N, C0, alpha, beta) : NAN;
    snprintf(detail, n, "err=%.1e status=%s", err, ab_status_string(st));

    ab_matrix_destroy(mA); ab_matrix_destroy(mB); ab_matrix_destroy(mC);
    free(A); free(B); free(C0); free(C); free(Cref); free(absAB);
    return st == AB_OK && err <= TOL;
}

static ABStatus upload_split(ABMatrix re, ABMatrix im, const double complex *z, size_t count) {
    double *r = malloc(count * sizeof *r), *i = malloc(count * sizeof *i);
    for (size_t k = 0; k < count; k++) { r[k] = creal(z[k]); i[k] = cimag(z[k]); }
    ABStatus st = ab_matrix_upload(re, r, false);
    if (st == AB_OK) st = ab_matrix_upload(im, i, false);
    free(r); free(i);
    return st;
}

static bool run_native_complex(const AbiCase *c, char *detail, size_t n) {
    const int M = c->M, N = c->N, K = c->K;
    const Layout la = {K, M, K}, lb = {N, K, N}, lc = {N, M, N};
    const size_t na = nelem(la), nb = nelem(lb), nc = nelem(lc);
    double complex *A = malloc(na * sizeof *A), *B = malloc(nb * sizeof *B);
    double complex *C0 = malloc(nc * sizeof *C0), *C = malloc(nc * sizeof *C);
    double complex *Cref = calloc(nc, sizeof *Cref);
    double *Cr = malloc(nc * sizeof *Cr), *Ci = malloc(nc * sizeof *Ci);
    double *absAB = calloc(nc, sizeof *absAB);
    const double complex one = 1.0, zero = 0.0;

    fill_complex(A, la, false);
    fill_complex(B, lb, false);
    fill_complex(C0, lc, false);
    cblas_zgemm(CblasColMajor, CblasNoTrans, CblasNoTrans, N, M, K,
                &one, B, N, A, K, &zero, Cref, N);
    double *mA = moduli(A, na), *mB = moduli(B, nb);
    abs_product(mB, lb, 'N', mA, la, 'N', N, M, K, absAB);
    free(mA); free(mB);

    ABMatrix mAr = ab_matrix_create(M, K), mAi = ab_matrix_create(M, K);
    ABMatrix mBr = ab_matrix_create(K, N), mBi = ab_matrix_create(K, N);
    ABMatrix mCr = ab_matrix_create(M, N), mCi = ab_matrix_create(M, N);
    ABStatus st = (mAr && mAi && mBr && mBi && mCr && mCi) ? AB_OK : AB_ERROR_ALLOC_FAILED;
    if (st == AB_OK) st = upload_split(mAr, mAi, A, na);
    if (st == AB_OK) st = upload_split(mBr, mBi, B, nb);
    if (st == AB_OK) st = upload_split(mCr, mCi, C0, nc);
    if (st == AB_OK) st = ab_zgemm(mAr, mAi, mBr, mBi, mCr, mCi);
    if (st == AB_OK) st = ab_matrix_download(mCr, Cr, false);
    if (st == AB_OK) st = ab_matrix_download(mCi, Ci, false);
    for (size_t k = 0; k < nc; k++) C[k] = CMPLX(Cr[k], Ci[k]);

    double err = st == AB_OK ? err_complex(N, M, C, Cref, N, absAB, N, C0, one, zero) : NAN;
    snprintf(detail, n, "err=%.1e status=%s", err, ab_status_string(st));

    ab_matrix_destroy(mAr); ab_matrix_destroy(mAi); ab_matrix_destroy(mBr);
    ab_matrix_destroy(mBi); ab_matrix_destroy(mCr); ab_matrix_destroy(mCi);
    free(A); free(B); free(C0); free(C); free(Cref); free(Cr); free(Ci); free(absAB);
    return st == AB_OK && err <= TOL;
}

// The address the static linker bound for this executable is what a Fortran
// call site gets; it must lie inside libapplebottom.
static bool run_symbol(const AbiCase *c, char *detail, size_t n) {
    Dl_info info;
    if (!dladdr(c->sym, &info) || !info.dli_fname) {
        snprintf(detail, n, "dladdr failed");
        return false;
    }
    const char *base = strrchr(info.dli_fname, '/');
    base = base ? base + 1 : info.dli_fname;
    snprintf(detail, n, "bound to %s", base);
    return strstr(base, "libapplebottom") != NULL;
}

static bool run_case(int idx, char *detail, size_t n) {
    const AbiCase *c = &CASES[idx];
    rng_state = 0x9E3779B97F4A7C15ULL * (uint64_t)(idx + 1);
    switch (c->op) {
    case OP_DGEMM_BLAS: case OP_DGEMM_F: case OP_DGEMM_SYM:
        return run_blas_real(c, detail, n);
    case OP_ZGEMM_BLAS: case OP_ZGEMM_F: case OP_ZGEMM_SYM:
        return run_blas_complex(c, detail, n);
    case OP_ZGEMM:
        return run_native_complex(c, detail, n);
    case OP_SYMBOL:
        return run_symbol(c, detail, n);
    default:
        return run_native_real(c, detail, n);
    }
}

static const Suite *suite_of(int idx) {
    for (int i = 0; i < SU_COUNT; i++)
        if (idx >= SUITES[i].lo && idx < SUITES[i].hi) return &SUITES[i];
    return NULL;
}

// Runs cases [lo, hi) of one suite and reports "R <idx> <ok> <detail>" per case.
static int child_main(int lo, int hi) {
    const Suite *s = suite_of(lo);
    if (!s || hi > s->hi || lo >= hi) return 2;
    ABStatus init = s->init ? ab_init() : AB_OK;
    for (int i = lo; i < hi; i++) {
        char detail[512];
        bool ok = false;
        if (init != AB_OK) {
            snprintf(detail, sizeof detail, "ab_init failed: %s", ab_status_string(init));
        } else {
            alarm(CASE_TIMEOUT_S);
            ok = run_case(i, detail, sizeof detail);
            alarm(0);
        }
        printf("R %d %d %s\n", i, ok, detail);
        fflush(stdout);
    }
    return 0;
}

// =============================================================================
// Parent side: spawn, collect, classify
// =============================================================================

static int tests_passed = 0;
static int tests_failed = 0;
static int tests_xfailed = 0;
static int tests_xpassed = 0;
static bool verbose = false;
static FILE *suite_log;              // detail lines, printed after the suite summary
static char self_path[PATH_MAX];

static void record(const AbiCase *c, bool ok, const char *detail) {
    int xi = xfail_lookup(c->name);
    const XfailEntry *x = xi < 0 ? NULL : &XFAIL[xi];
    if (x) xfail_hits[xi]++;
    if (ok && !x) {
        tests_passed++;
        if (verbose) fprintf(suite_log, "      PASS  %s  %s\n", c->name, detail);
    } else if (ok) {
        tests_xpassed++;
        fprintf(suite_log, "      XPASS %s  [%s] %s -- remove it from the XFAIL table\n",
                c->name, x->id, detail);
    } else if (x) {
        tests_xfailed++;
        if (verbose) fprintf(suite_log, "      XFAIL %s  [%s] %s\n", c->name, x->id, detail);
    } else {
        tests_failed++;
        fprintf(suite_log, "      FAIL  %s  %s\n", c->name, detail);
    }
}

// Child environment: the caller's, minus AB_* and MTL_DEBUG_LAYER, plus the suite's.
static char **child_env(const Suite *s, char *mode_buf, size_t mode_len) {
    static char debug_layer[] = "MTL_DEBUG_LAYER=1";
    size_t n = 0;
    while (environ[n]) n++;
    char **env = calloc(n + 3, sizeof *env);
    size_t k = 0;
    for (size_t i = 0; i < n; i++)
        if (strncmp(environ[i], "AB_", 3) && strncmp(environ[i], "MTL_DEBUG_LAYER=", 16))
            env[k++] = environ[i];
    if (s->ab_mode) {
        snprintf(mode_buf, mode_len, "AB_MODE=%s", s->ab_mode);
        env[k++] = mode_buf;
    }
    if (s->debug_layer) env[k++] = debug_layer;
    return env;
}

// Runs cases [lo, s->hi) in one child and records every reported case.
// Returns the first unreported index; *why describes an abnormal exit.
static int run_child(const Suite *s, int lo, char *why, size_t why_len) {
    char mode_buf[32], lo_arg[16], hi_arg[16], child_flag[] = "--child";
    char *argv[] = {self_path, child_flag, lo_arg, hi_arg, NULL};
    int fds[2], next = lo;
    pid_t pid;

    why[0] = '\0';
    snprintf(lo_arg, sizeof lo_arg, "%d", lo);
    snprintf(hi_arg, sizeof hi_arg, "%d", s->hi);
    if (pipe(fds) != 0) {
        snprintf(why, why_len, "pipe: %s", strerror(errno));
        return lo;
    }
    posix_spawn_file_actions_t fa;
    posix_spawn_file_actions_init(&fa);
    posix_spawn_file_actions_adddup2(&fa, fds[1], STDOUT_FILENO);
    posix_spawn_file_actions_adddup2(&fa, fds[1], STDERR_FILENO);
    posix_spawn_file_actions_addclose(&fa, fds[0]);
    posix_spawn_file_actions_addclose(&fa, fds[1]);
    char **env = child_env(s, mode_buf, sizeof mode_buf);
    fflush(stdout);
    int rc = posix_spawn(&pid, self_path, &fa, NULL, argv, env);
    posix_spawn_file_actions_destroy(&fa);
    free(env);
    close(fds[1]);
    if (rc != 0) {
        close(fds[0]);
        snprintf(why, why_len, "posix_spawn: %s", strerror(rc));
        return lo;
    }

    // Result lines advance `next`; anything else (library or Metal
    // diagnostics) is kept so a crash can be explained.
    FILE *in = fdopen(fds[0], "r");
    char line[2048], last_noise[512] = "";
    while (fgets(line, sizeof line, in)) {
        line[strcspn(line, "\n")] = '\0';
        int idx, ok, off = 0;
        if (sscanf(line, "R %d %d %n", &idx, &ok, &off) == 2 && off > 0 && idx == next) {
            record(&CASES[idx], ok != 0, line + off);
            next++;
            last_noise[0] = '\0';
        } else if (line[0]) {
            snprintf(last_noise, sizeof last_noise, "%s", line);
        }
    }
    fclose(in);

    int status = 0;
    while (waitpid(pid, &status, 0) < 0 && errno == EINTR) {}
    if (next < s->hi || !WIFEXITED(status) || WEXITSTATUS(status) != 0) {
        char how[96];
        if (WIFSIGNALED(status))
            snprintf(how, sizeof how, "killed by signal %d (%s)",
                     WTERMSIG(status), strsignal(WTERMSIG(status)));
        else if (WEXITSTATUS(status) != 0)
            snprintf(how, sizeof how, "child exited with status %d", WEXITSTATUS(status));
        else
            snprintf(how, sizeof how, "child exited without reporting");
        snprintf(why, why_len, "%s%s%s", how, last_noise[0] ? ": " : "", last_noise);
    }
    return next;
}

static void run_suite(int si) {
    const Suite *s = &SUITES[si];
    const int p0 = tests_passed, x0 = tests_xfailed, f0 = tests_failed, xp0 = tests_xpassed;
    int crashes = 0;
    char *log_buf = NULL;
    size_t log_len = 0;

    suite_log = open_memstream(&log_buf, &log_len);
    for (int next = s->lo; next < s->hi; ) {
        char why[640];
        next = run_child(s, next, why, sizeof why);
        if (!why[0]) continue;
        crashes++;
        if (next < s->hi) {
            record(&CASES[next], false, why);
            next++;
        } else {
            tests_failed++;
            fprintf(suite_log, "      FAIL  child failed after its last case: %s\n", why);
        }
    }
    fclose(suite_log);

    printf("%d cases: %d pass, %d xfail", s->hi - s->lo, tests_passed - p0, tests_xfailed - x0);
    if (tests_failed > f0) printf(", %d FAIL", tests_failed - f0);
    if (tests_xpassed > xp0) printf(", %d XPASS", tests_xpassed - xp0);
    if (crashes) printf(" (%d child crash%s)", crashes, crashes == 1 ? "" : "es");
    printf("\n%s", log_buf ? log_buf : "");
    free(log_buf);
}

// =============================================================================
// Main
// =============================================================================

static void test_dgemm_blas_gpu(void)   { run_suite(SU_DGEMM_BLAS_GPU); }
static void test_dgemm_blas_cpu(void)   { run_suite(SU_DGEMM_BLAS_CPU); }
static void test_zgemm_blas_gpu(void)   { run_suite(SU_ZGEMM_BLAS_GPU); }
static void test_zgemm_blas_cpu(void)   { run_suite(SU_ZGEMM_BLAS_CPU); }
static void test_ab_dgemm_f_gpu(void)   { run_suite(SU_DGEMM_F_GPU); }
static void test_ab_dgemm_f_cpu(void)   { run_suite(SU_DGEMM_F_CPU); }
static void test_ab_zgemm_f_gpu(void)   { run_suite(SU_ZGEMM_F_GPU); }
static void test_ab_zgemm_f_cpu(void)   { run_suite(SU_ZGEMM_F_CPU); }
static void test_dgemm_sym_gpu(void)    { run_suite(SU_DGEMM_SYM_GPU); }
static void test_dgemm_sym_cpu(void)    { run_suite(SU_DGEMM_SYM_CPU); }
static void test_zgemm_sym_gpu(void)    { run_suite(SU_ZGEMM_SYM_GPU); }
static void test_zgemm_sym_cpu(void)    { run_suite(SU_ZGEMM_SYM_CPU); }
static void test_auto_mode(void)        { run_suite(SU_AUTO); }
static void test_lazy_init(void)        { run_suite(SU_NOINIT); }
static void test_native_morton(void)    { run_suite(SU_NATIVE); }
static void test_fortran_symbols(void)  { run_suite(SU_SYMBOLS); }

static const TestCase TESTS[] = {
    {"test_dgemm_blas_gpu",   test_dgemm_blas_gpu},
    {"test_dgemm_blas_cpu",   test_dgemm_blas_cpu},
    {"test_zgemm_blas_gpu",   test_zgemm_blas_gpu},
    {"test_zgemm_blas_cpu",   test_zgemm_blas_cpu},
    {"test_ab_dgemm_f_gpu",   test_ab_dgemm_f_gpu},
    {"test_ab_dgemm_f_cpu",   test_ab_dgemm_f_cpu},
    {"test_ab_zgemm_f_gpu",   test_ab_zgemm_f_gpu},
    {"test_ab_zgemm_f_cpu",   test_ab_zgemm_f_cpu},
    {"test_dgemm_sym_gpu",    test_dgemm_sym_gpu},
    {"test_dgemm_sym_cpu",    test_dgemm_sym_cpu},
    {"test_zgemm_sym_gpu",    test_zgemm_sym_gpu},
    {"test_zgemm_sym_cpu",    test_zgemm_sym_cpu},
    {"test_auto_mode",        test_auto_mode},
    {"test_lazy_init",        test_lazy_init},
    {"test_native_morton",    test_native_morton},
    {"test_fortran_symbols",  test_fortran_symbols},
};
static const int N_TESTS = (int)TEST_REGISTRY_SIZE(TESTS);

int main(int argc, char **argv) {
    build_cases();
    if (argc == 4 && !strcmp(argv[1], "--child"))
        return child_main(atoi(argv[2]), atoi(argv[3]));
    verbose = argc > 1 && !strcmp(argv[1], "-v");

    uint32_t len = sizeof self_path;
    if (_NSGetExecutablePath(self_path, &len) != 0) {
        fprintf(stderr, "test_blas_abi: cannot resolve executable path\n");
        return 1;
    }

    printf("╔══════════════════════════════════════════════════════════════════╗\n");
    printf("║  apple-bottom BLAS-ABI Conformance (reference: Accelerate cblas) ║\n");
    printf("╚══════════════════════════════════════════════════════════════════╝\n");

    for (int i = 0; i < N_TESTS; i++) {
        printf("[%d/%d] %-22s ... ", i + 1, N_TESTS, TESTS[i].name);
        fflush(stdout);
        int pre = tests_passed + tests_failed + tests_xfailed + tests_xpassed;
        TESTS[i].fn();
        int post = tests_passed + tests_failed + tests_xfailed + tests_xpassed;
        if (post == pre) {
            fprintf(stderr,
                "\nFATAL: ghost test — '%s' reported no result\n",
                TESTS[i].name);
            abort();
        }
    }

    int stale = 0;
    printf("\nXFAIL table (audit ID, cases matched, pattern):\n");
    for (size_t i = 0; i < TEST_REGISTRY_SIZE(XFAIL); i++) {
        printf("  %-14s %4d  %s%s\n", XFAIL[i].id, xfail_hits[i], XFAIL[i].pattern,
               xfail_hits[i] ? "" : "  <-- STALE: matches no case");
        if (!xfail_hits[i]) stale++;
    }

    printf("\n═══════════════════════════════════════════════════════════════════\n");
    printf("Results: %d passed, %d xfailed, %d failed, %d xpassed, %d stale XFAIL entries\n",
           tests_passed, tests_xfailed, tests_failed, tests_xpassed, stale);
    if (tests_failed == 0 && tests_xpassed == 0 && stale == 0) {
        printf("✓ BLAS-ABI conformance: no unexpected results (%d known failures)\n",
               tests_xfailed);
        return 0;
    }
    printf("✗ BLAS-ABI conformance: unexpected results\n");
    return 1;
}
