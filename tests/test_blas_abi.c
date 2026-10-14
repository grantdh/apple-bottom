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
// unchanged, ab_get_last_dispatch_path() names what computed C ("gpu" if the
// GEMM counters in ab_get_stats() advanced, "cpu" otherwise), under
// AB_PROFILE the profiler counts the calls the same way, and the
// case's routing expectation holds (ROUTE_GPU: the counters advanced;
// ROUTE_CPU: they did not). Leading-dimension padding is filled with NaN so
// that reading it shows up in the result. An illegal-argument case passes
// only if the call never returns: cblas must reject it as reference xerbla
// would, naming the bad parameter and ending the process with status 255.
//
// Isolation: every suite runs in a child process (this binary re-executed
// with --child lo hi), because
//   - AB_MODE and the other routing settings are read once and cached
//     (dispatch.c), so each mode needs a fresh process;
//   - the gpu-noinit suites must run in processes that never called ab_init();
//   - known-bad cases crash (SIGSEGV, Metal API-validation SIGABRT), and
//     illegal-argument cases end the child on purpose (exit(255) from the
//     cblas parameter check). A crash fails only the case in progress; the
//     parent restarts the child at the next case.
// posix_spawn is used rather than a bare fork(): Metal cannot reach
// MTLCompilerService from a forked child that did not exec.
//
// The report suite checks the AB_REPORT and AB_PROFILE exit reports of a
// grandchild process (this binary with --report-child) that makes GPU and
// CPU calls and forks.
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
#include <sys/mman.h>
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

// AB_PROFILE table writer, exported by blas_profiler.c (internal header).
void ab_prof_dump(FILE *out);

#define TOL             1e-11
#define MAX_CASES       1024
#define MIN_GPU_DIM     32      // AB_DISPATCH_DEFAULT_MIN_DIM in dispatch.h
#define OLD_MAX_DIM     46340   // v1.3 per-dimension limit, now a per-matrix one
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
    OP_REPORT,                          // AB_REPORT / AB_PROFILE exit reports
} Op;

static const char *const OP_NAME[] = {
    [OP_DGEMM_BLAS] = "ab_dgemm_blas", [OP_ZGEMM_BLAS] = "ab_zgemm_blas",
    [OP_DGEMM_F] = "ab_dgemm_",        [OP_ZGEMM_F] = "ab_zgemm_",
    [OP_DGEMM_SYM] = "dgemm_",         [OP_ZGEMM_SYM] = "zgemm_",
    [OP_DGEMM] = "ab_dgemm",           [OP_DGEMM_SCALED] = "ab_dgemm_scaled",
    [OP_ZGEMM] = "ab_zgemm",           [OP_BATCH_DGEMM] = "ab_batch_dgemm",
    [OP_DGEMM_ASYNC] = "ab_dgemm_async", [OP_SYMBOL] = "symbol",
    [OP_REPORT] = "report",
};

typedef enum { ROUTE_ANY, ROUTE_GPU, ROUTE_CPU } Route;
typedef enum { POISON_NONE, POISON_AB, POISON_C } Poison;

typedef struct {
    const char *tag;
    double complex alpha, beta;         // real entry points use the real parts
    Poison poison;
} Scalars;

enum { SC_GEN, SC_A1B0, SC_B1, SC_B0_NAN_C, SC_A0_NAN_AB, SC_A2B0 };

static const Scalars SCALARS[] = {
    [SC_GEN]       = {"gen",     CMPLX(1.25, -0.5), CMPLX(-0.75, 0.25), POISON_NONE},
    [SC_A1B0]      = {"a1b0",    CMPLX(1.0, 0.0),   CMPLX(0.0, 0.0),    POISON_NONE},
    [SC_B1]        = {"b1",      CMPLX(1.25, -0.5), CMPLX(1.0, 0.0),    POISON_NONE},
    [SC_B0_NAN_C]  = {"b0nanC",  CMPLX(1.25, -0.5), CMPLX(0.0, 0.0),    POISON_C},   // C must not be read
    [SC_A0_NAN_AB] = {"a0nanAB", CMPLX(0.0, 0.0),   CMPLX(0.5, -0.25),  POISON_AB},  // A, B must not be read
    [SC_A2B0]      = {"a2b0",    CMPLX(2.0, 0.0),   CMPLX(0.0, 0.0),    POISON_NONE},  // QE newq, gamma point
};

// Illegal BLAS arguments. Accelerate's cblas rejects each one by printing
// "Parameter number <param> passed to cblas_<x>gemm had an invalid value" and
// exiting with status 255; <param> counts cblas_<x>gemm's arguments, with
// Order as 1.
enum { BAD_NONE, BAD_TRANSA, BAD_TRANSB, BAD_M, BAD_N, BAD_K, BAD_LDA, BAD_LDB, BAD_LDC,
       BAD_COUNT };

static const struct { const char *arg; int param; } BAD_ARGS[] = {
    [BAD_TRANSA] = {"transA", 2}, [BAD_TRANSB] = {"transB", 3},
    [BAD_M] = {"M", 4},   [BAD_N] = {"N", 5},     [BAD_K] = {"K", 6},
    [BAD_LDA] = {"lda", 9}, [BAD_LDB] = {"ldb", 11}, [BAD_LDC] = {"ldc", 14},
};

typedef struct {
    char name[80];
    Op op;
    char ta, tb;
    int M, N, K, pad;
    double complex alpha, beta;
    Poison poison;
    Route route;
    int bad;                            // BAD_*: pass this argument illegal
    bool fork;                          // repeat the call in a fork()ed child
    const void *sym;                    // OP_SYMBOL only
} AbiCase;

static AbiCase CASES[MAX_CASES];
static int n_cases;

// =============================================================================
// XFAIL table: known failures, first match wins (fnmatch, FNM_PATHNAME).
// Case names: <entry>/<mode>/<transA><transB>/<M>x<N>x<K>/ld+<pad>/<scalars>
// for BLAS-ABI cases, .../<M>x<N>x<K>/illegal-<arg> for illegal-argument
// cases, <entry>/<M>x<N>x<K> for native cases.
// =============================================================================

typedef struct {
    const char *pattern;
    const char *id;
} XfailEntry;

static const XfailEntry XFAIL[] = {
    // No known failures.
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
    bool test_table;            // AB_DISPATCH_TABLE=<TEST_TABLE in a temporary file>
    bool init;                  // call ab_init() before the first case
    bool debug_layer;           // MTL_DEBUG_LAYER=1 (Metal API validation)
    bool profile;               // AB_PROFILE=1
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

static AbiCase *add_blas(const Suite *s, Op op, char ta, char tb, int M, int N, int K,
                         int pad, int scalars, Route route) {
    const Scalars *sc = &SCALARS[scalars];
    AbiCase *c = new_case();
    snprintf(c->name, sizeof c->name, "%s/%s/%c%c/%dx%dx%d/ld+%d/%s",
             OP_NAME[op], s->label, ta, tb, M, N, K, pad, sc->tag);
    c->op = op; c->ta = ta; c->tb = tb;
    c->M = M; c->N = N; c->K = K; c->pad = pad;
    c->alpha = sc->alpha; c->beta = sc->beta; c->poison = sc->poison;
    c->route = route;
    return c;
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

    add_std(s, 'N', 'N', OLD_MAX_DIM + 1, 48, 48, 0, SC_GEN);       // past the v1.3 limit
}

// AB_MODE=auto with the built-in table (dispatch_table.h, from `make
// calibrate`). Through every BLAS-ABI entry point, calls the GPU path was
// measured slower on stay on Accelerate: the QE cegterg-like call of the
// gen_blas suites, which v1.3's FLOP thresholds sent to the GPU, and QE si64
// PAW DGEMMs and Davidson ZGEMMs (calibrate_summary.csv; test_dispatch
// checks the whole table against it).
static void gen_auto(const Suite *s) {
    add_blas(s, OP_ZGEMM_BLAS, 'C', 'N', 150, 150, 2000, 0, SC_A1B0, ROUTE_CPU);
    add_blas(s, OP_ZGEMM_F,    'C', 'N', 150, 150, 2000, 0, SC_A1B0, ROUTE_CPU);
    add_blas(s, OP_ZGEMM_SYM,  'C', 'N', 150, 150, 2000, 0, SC_A1B0, ROUTE_CPU);
    add_blas(s, OP_DGEMM_BLAS, 'T', 'N', 150, 150, 2000, 0, SC_A1B0, ROUTE_CPU);
    add_blas(s, OP_DGEMM_F,    'T', 'N', 150, 150, 2000, 0, SC_A1B0, ROUTE_CPU);
    add_blas(s, OP_DGEMM_SYM,  'T', 'N', 171, 64, 143666, 0, SC_A2B0, ROUTE_CPU);
    add_blas(s, OP_DGEMM_SYM,  'N', 'T', 143666, 171, 64, 0, SC_A2B0, ROUTE_CPU);
    add_blas(s, OP_ZGEMM_SYM,  'C', 'N', 1000, 1000, 9000, 0, SC_A1B0, ROUTE_CPU);
    add_blas(s, OP_ZGEMM_SYM,  'N', 'N', 9000, 500, 1000, 0, SC_A1B0, ROUTE_CPU);
}

// AB_DISPATCH_TABLE of the test_table suites: every 'N','N' call on the GPU.
static const char TEST_TABLE[] = "# test_blas_abi\ndgemm N N 0 0 0\nzgemm N N 0 0 0\n";
static char test_table_path[PATH_MAX];

// AB_MODE=auto and AB_PROFILE=1, no ab_init(), with TEST_TABLE. The second
// shape stays on cblas: K = 16 is below AB_MIN_GPU_DIM (the audit's
// 2048x2048x16 call). The third is past the v1.3 limit and must run on the
// GPU; the fourth has no table row for its TRANS class. The profiler must
// report what ran; the fork suite covers a GPU-routed call that falls back.
static void gen_profile(const Suite *s) {
    static const Op OPS[] = {OP_DGEMM_SYM, OP_ZGEMM_SYM};
    for (size_t o = 0; o < TEST_REGISTRY_SIZE(OPS); o++) {
        add_blas(s, OPS[o], 'N', 'N', 256, 256, 2048, 0, SC_GEN, ROUTE_GPU);
        add_blas(s, OPS[o], 'N', 'N', 2048, 2048, 16, 0, SC_GEN, ROUTE_CPU);
        add_blas(s, OPS[o], 'N', 'N', OLD_MAX_DIM + 1, 48, 48, 0, SC_GEN, ROUTE_GPU);
        add_blas(s, OPS[o], op_is_complex(OPS[o]) ? 'C' : 'T', 'N', 256, 256, 2048, 0, SC_GEN,
                 ROUTE_CPU);
    }
}

// Each illegal argument through each BLAS-ABI entry point, in AB_MODE=gpu
// without ab_init() (as in QE), on a shape whose legal call runs on the GPU.
// The legal call is NT (NC for complex) with tight leading dimensions; the
// illegal ld is one less than the stored rows. M > K and N > K, so an ld
// check against the wrong dimension accepts it.
static void gen_illegal(const Suite *s) {
    static const Op OPS[] = {
        OP_DGEMM_BLAS, OP_ZGEMM_BLAS, OP_DGEMM_F, OP_ZGEMM_F, OP_DGEMM_SYM, OP_ZGEMM_SYM,
    };
    for (size_t o = 0; o < TEST_REGISTRY_SIZE(OPS); o++)
        for (int bad = BAD_TRANSA; bad < BAD_COUNT; bad++) {
            AbiCase *c = add_blas(s, OPS[o], 'N', op_is_complex(OPS[o]) ? 'C' : 'T',
                                  96, 80, 72, 0, SC_GEN, ROUTE_ANY);
            c->bad = bad;
            snprintf(c->name, sizeof c->name, "%s/%s/%c%c/%dx%dx%d/illegal-%s",
                     OP_NAME[c->op], s->label, c->ta, c->tb, c->M, c->N, c->K,
                     BAD_ARGS[bad].arg);
        }
}

// AB_MODE=gpu and AB_PROFILE=1 without ab_init(): a GPU call creates the
// Metal context, then a fork()ed child repeats it. Metal does not survive
// fork(), so the child must fall back to cblas, get the same correct result
// and be profiled as a CPU call. (The shapes stay small: Accelerate's own
// threaded cblas does not survive fork() either.)
static void gen_fork(const Suite *s) {
    add_blas(s, OP_DGEMM_SYM, 'N', 'N', 96, 80, 72, 3, SC_GEN, ROUTE_GPU)->fork = true;
    add_blas(s, OP_ZGEMM_SYM, 'N', 'N', 96, 80, 72, 3, SC_GEN, ROUTE_GPU)->fork = true;
}

// The only QE Si64 calls the v1.3 per-dimension limit (46340) kept off the
// GPU: the PAW DGEMMs of newq ('T','N', 171 x 64 x 2*ngm_l, alpha = 2 at the
// gamma point) and addusdens (2*ngm_l x 171 x 64; QE passes 'N','T'). 171 =
// nh(nh+1)/2 projector pairs (nh = 18 per Si atom), 64 = atoms, 2*ngm_l =
// 71834 (gamma point) or 143666 (k-points), the rank's dense-grid G-vectors
// as real pairs; QE's ld is 2*ngm_l, the ld > rows variants are extra. The
// 'C','N' ZGEMM with K = 100000 is synthetic (cegterg's K is ~9000 per rank).
// Through dgemm_/zgemm_ in AB_MODE=gpu without ab_init(), every case must
// run on the GPU. This checks robustness only: these DGEMMs are faster on
// Accelerate (0.09-0.15x on the GPU in make calibrate), where AB_MODE=auto
// keeps them (gen_auto).
static void gen_qe(const Suite *s) {
    add_blas(s, OP_DGEMM_SYM, 'T', 'N', 171, 64, 71834, 0, SC_A2B0, ROUTE_GPU);
    add_blas(s, OP_DGEMM_SYM, 'T', 'N', 171, 64, 143666, 3, SC_A2B0, ROUTE_GPU);
    add_blas(s, OP_DGEMM_SYM, 'N', 'N', 71834, 171, 64, 0, SC_A2B0, ROUTE_GPU);
    add_blas(s, OP_DGEMM_SYM, 'N', 'N', 143666, 171, 64, 3, SC_A2B0, ROUTE_GPU);
    add_blas(s, OP_ZGEMM_SYM, 'C', 'N', 171, 64, 100000, 0, SC_A1B0, ROUTE_GPU);
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

static void gen_report(const Suite *s) {
    (void)s;
    AbiCase *c = new_case();
    snprintf(c->name, sizeof c->name, "report/gpu-noinit/exit-reports");
    c->op = OP_REPORT;
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
    SU_AUTO,
    SU_DGEMM_BLAS_NOINIT, SU_ZGEMM_BLAS_NOINIT, SU_DGEMM_F_NOINIT, SU_ZGEMM_F_NOINIT,
    SU_DGEMM_SYM_NOINIT, SU_ZGEMM_SYM_NOINIT, SU_ILLEGAL, SU_FORK, SU_QE,
    SU_PROFILE, SU_REPORT, SU_NATIVE, SU_SYMBOLS, SU_COUNT
};

#define BLAS_SUITE(entry, mode) \
    {.label = mode, .ab_mode = mode, .init = true, .op = entry, .gen = gen_blas}

// AB_MODE=gpu in a process that never calls ab_init(), as in QE: the first
// GPU-routed call must create the Metal context itself (H-P1-2).
#define NOINIT_SUITE(entry) \
    {.label = "gpu-noinit", .ab_mode = "gpu", .op = entry, .gen = gen_blas}

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
    [SU_DGEMM_BLAS_NOINIT] = NOINIT_SUITE(OP_DGEMM_BLAS),
    [SU_ZGEMM_BLAS_NOINIT] = NOINIT_SUITE(OP_ZGEMM_BLAS),
    [SU_DGEMM_F_NOINIT]    = NOINIT_SUITE(OP_DGEMM_F),
    [SU_ZGEMM_F_NOINIT]    = NOINIT_SUITE(OP_ZGEMM_F),
    [SU_DGEMM_SYM_NOINIT]  = NOINIT_SUITE(OP_DGEMM_SYM),
    [SU_ZGEMM_SYM_NOINIT]  = NOINIT_SUITE(OP_ZGEMM_SYM),
    [SU_ILLEGAL] = {.label = "gpu-noinit", .ab_mode = "gpu", .gen = gen_illegal},
    [SU_FORK]    = {.label = "gpu-noinit-fork", .ab_mode = "gpu", .profile = true,
                    .gen = gen_fork},
    [SU_QE]      = {.label = "gpu-noinit", .ab_mode = "gpu", .gen = gen_qe},
    [SU_PROFILE] = {.label = "auto-table-profile", .ab_mode = "auto", .test_table = true,
                    .profile = true, .gen = gen_profile},
    [SU_REPORT]  = {.label = "gpu-noinit", .ab_mode = "gpu", .gen = gen_report},
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

// Under AB_PROFILE, after every BLAS-ABI call the profiler row for the
// routine must hold all calls so far, gpu_calls the calls that ran on the
// GPU, and gpu_calls% their share. A fork()ed child starts from zero
// (run_fork resets these).
static unsigned long long prof_calls[2], prof_gpu[2];

static bool profile_ok(const AbiCase *c, uint64_t gpu_gemms) {
    if (!getenv("AB_PROFILE")) return true;
    const int z = op_is_complex(c->op);
    prof_calls[z]++;
    prof_gpu[z] += gpu_gemms > 0;

    char *table = NULL;
    size_t len = 0;
    FILE *f = open_memstream(&table, &len);
    ab_prof_dump(f);
    fclose(f);
    bool ok = false;
    char *save = NULL;
    for (char *line = strtok_r(table, "\n", &save); line; line = strtok_r(NULL, "\n", &save)) {
        char name[32];
        unsigned long long n, g;
        double pct;
        int end = 0;
        if (sscanf(line, "%31s %llu %llu %lf%%%n", name, &n, &g, &pct, &end) == 4 && end &&
            !strcmp(name, z ? "zgemm" : "dgemm"))
            ok = n == prof_calls[z] && g == prof_gpu[z] &&
                 fabs(pct - 100.0 * g / n) < 0.051;
    }
    free(table);
    return ok;
}

static bool blas_verdict(const AbiCase *c, double err, bool inputs_ok, bool outside_ok,
                         uint64_t gpu_gemms, char *detail, size_t n) {
    const char *path = ab_get_last_dispatch_path();
    if (c->bad) {
        snprintf(detail, n, "call with illegal %s returned (path=%s gpu_gemms=%llu); "
                 "cblas should have rejected it", BAD_ARGS[c->bad].arg, path,
                 (unsigned long long)gpu_gemms);
        return false;
    }
    bool path_ok = !strcmp(path, gpu_gemms > 0 ? "gpu" : "cpu");
    bool prof_ok = profile_ok(c, gpu_gemms);
    bool routed = c->route == ROUTE_ANY ||
                  (c->route == ROUTE_GPU && gpu_gemms > 0) ||
                  (c->route == ROUTE_CPU && gpu_gemms == 0);
    snprintf(detail, n, "err=%.1e path=%s gpu_gemms=%llu%s%s%s%s%s", err, path,
             (unsigned long long)gpu_gemms,
             inputs_ok ? "" : " [A or B modified]",
             outside_ok ? "" : " [C outside MxN modified]",
             path_ok ? "" : " [path record does not match]",
             prof_ok ? "" : " [AB_PROFILE counts do not match]",
             routed ? "" : c->route == ROUTE_GPU ? " [GPU path not taken]"
                                                 : " [GPU used, expected cblas]");
    return err <= TOL && inputs_ok && outside_ok && path_ok && prof_ok && routed;
}

// The arguments a BLAS-ABI case passes: its legal call, with c->bad made
// illegal. Buffers are always sized for the legal call.
typedef struct { char ta, tb; int M, N, K, lda, ldb, ldc; } BlasArgs;

static BlasArgs blas_args(const AbiCase *c, Layout la, Layout lb, Layout lc) {
    BlasArgs a = {c->ta, c->tb, c->M, c->N, c->K, la.ld, lb.ld, lc.ld};
    switch (c->bad) {
    case BAD_TRANSA: a.ta = 'X'; break;
    case BAD_TRANSB: a.tb = 'X'; break;
    case BAD_M:      a.M = -1; break;
    case BAD_N:      a.N = -1; break;
    case BAD_K:      a.K = -1; break;
    case BAD_LDA:    a.lda = la.rows - 1; break;
    case BAD_LDB:    a.ldb = lb.rows - 1; break;
    case BAD_LDC:    a.ldc = lc.rows - 1; break;
    }
    return a;
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

    const BlasArgs a = blas_args(c, la, lb, lc);
    uint64_t g0 = gpu_gemm_count();
    switch (c->op) {
    case OP_DGEMM_BLAS:
        ab_dgemm_blas(a.ta, a.tb, a.M, a.N, a.K, alpha, A, a.lda, B, a.ldb, beta, C, a.ldc);
        break;
    case OP_DGEMM_F:
        ab_dgemm_(&a.ta, &a.tb, &a.M, &a.N, &a.K, &alpha, A, &a.lda, B, &a.ldb, &beta, C, &a.ldc);
        break;
    default:
        fortran_dgemm(&a.ta, &a.tb, &a.M, &a.N, &a.K, &alpha, A, &a.lda, B, &a.ldb, &beta, C, &a.ldc);
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

    const BlasArgs a = blas_args(c, la, lb, lc);
    uint64_t g0 = gpu_gemm_count();
    switch (c->op) {
    case OP_ZGEMM_BLAS:
        ab_zgemm_blas(a.ta, a.tb, a.M, a.N, a.K, alpha, A, a.lda, B, a.ldb, beta, C, a.ldc);
        break;
    case OP_ZGEMM_F:
        ab_zgemm_(&a.ta, &a.tb, &a.M, &a.N, &a.K, &alpha, A, &a.lda, B, &a.ldb, &beta, C, &a.ldc);
        break;
    default:
        fortran_zgemm(&a.ta, &a.tb, &a.M, &a.N, &a.K, &alpha, A, &a.lda, B, &a.ldb, &beta, C, &a.ldc);
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

// Runs c on the GPU, then fork()s; the child repeats c, which must now run
// on cblas (ROUTE_CPU) and pass the same checks. It reports through a shared
// page and its exit status.
static bool run_fork(const AbiCase *c, char *detail, size_t n) {
    bool (*run)(const AbiCase *, char *, size_t) =
        op_is_complex(c->op) ? run_blas_complex : run_blas_real;
    const size_t page = 4096;
    char before[256], how[64] = "";
    bool ok = run(c, before, sizeof before);

    char *after = mmap(NULL, page, PROT_READ | PROT_WRITE, MAP_SHARED | MAP_ANON, -1, 0);
    if (after == MAP_FAILED) {
        snprintf(detail, n, "mmap: %s", strerror(errno));
        return false;
    }
    snprintf(after, page, "no report");
    fflush(NULL);
    pid_t pid = fork();
    if (pid == 0) {
        AbiCase in_child = *c;
        in_child.route = ROUTE_CPU;
        memset(prof_calls, 0, sizeof prof_calls);
        memset(prof_gpu, 0, sizeof prof_gpu);
        _exit(run(&in_child, after, page) ? 0 : 1);
    }
    int status = 0;
    if (pid < 0)
        snprintf(how, sizeof how, " [fork: %s]", strerror(errno));
    else
        while (waitpid(pid, &status, 0) < 0 && errno == EINTR) {}
    if (pid > 0 && WIFSIGNALED(status))
        snprintf(how, sizeof how, " [killed by signal %d]", WTERMSIG(status));
    ok = ok && pid > 0 && WIFEXITED(status) && WEXITSTATUS(status) == 0;
    snprintf(detail, n, "parent: %s; forked child: %s%s", before, after, how);
    munmap(after, page);
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

// =============================================================================
// Exit reports (AB_REPORT, AB_PROFILE)
// =============================================================================

// The calls the --report-child process makes, in AB_MODE=gpu without
// ab_init(): three run on the GPU and three stay on cblas by routing (a
// dimension <= AB_MIN_GPU_DIM, or alpha == 0). It then fork()s; the child
// repeats the first call, which falls back to cblas (no Metal context after
// fork()), and ends with exit() so that its exit reports run.
static const struct { Op op; int M, N, K; double alpha; bool gpu; } REPORT_CALLS[] = {
    {OP_DGEMM_SYM,  96, 80, 72, 1.25, true},
    {OP_DGEMM_BLAS, 96, 80, 72, 1.25, true},
    {OP_DGEMM_F,    96, 80, 16, 1.25, false},
    {OP_ZGEMM_SYM,  96, 80, 72, 1.25, true},
    {OP_ZGEMM_F,    64, 48, 16, 1.25, false},
    {OP_ZGEMM_BLAS, 96, 80, 72, 0.0,  false},
};

// One 'N','N' call with tight leading dimensions; M, N, K <= 96, 80, 72.
static void report_call(Op op, int M, int N, int K, double alpha) {
    static double complex A[96 * 80], B[96 * 80], C[96 * 80];
    const char t = 'N';
    const double beta = 0.0;
    const double complex za = alpha, zb = 0.0;
    double *a = (double *)A, *b = (double *)B, *c = (double *)C;
    for (size_t i = 0; i < 96 * 80; i++) A[i] = B[i] = CMPLX(1.0 / (i + 1), 0.5);
    switch (op) {
    case OP_DGEMM_SYM:  fortran_dgemm(&t, &t, &M, &N, &K, &alpha, a, &M, b, &K, &beta, c, &M); break;
    case OP_DGEMM_F:    ab_dgemm_(&t, &t, &M, &N, &K, &alpha, a, &M, b, &K, &beta, c, &M); break;
    case OP_DGEMM_BLAS: ab_dgemm_blas(t, t, M, N, K, alpha, a, M, b, K, beta, c, M); break;
    case OP_ZGEMM_SYM:  fortran_zgemm(&t, &t, &M, &N, &K, &za, A, &M, B, &K, &zb, C, &M); break;
    case OP_ZGEMM_F:    ab_zgemm_(&t, &t, &M, &N, &K, &za, A, &M, B, &K, &zb, C, &M); break;
    default:            ab_zgemm_blas(t, t, M, N, K, za, A, M, B, K, zb, C, M); break;
    }
}

// --report-child: REPORT_CALLS, then the fork()ed repeat; prints
// "child <pid> <wait status>".
static int report_child_main(void) {
    for (size_t i = 0; i < TEST_REGISTRY_SIZE(REPORT_CALLS); i++)
        report_call(REPORT_CALLS[i].op, REPORT_CALLS[i].M, REPORT_CALLS[i].N,
                    REPORT_CALLS[i].K, REPORT_CALLS[i].alpha);
    fflush(NULL);
    pid_t pid = fork();
    if (pid == 0) {
        report_call(REPORT_CALLS[0].op, REPORT_CALLS[0].M, REPORT_CALLS[0].N,
                    REPORT_CALLS[0].K, REPORT_CALLS[0].alpha);
        exit(0);
    }
    int status = -1;
    if (pid > 0)
        while (waitpid(pid, &status, 0) < 0 && errno == EINTR) {}
    printf("child %d %d\n", (int)pid, status);
    return 0;
}

// What one process's reports say about one routine (times in ms).
typedef struct {
    int rows, stage_rows, fallback_rows, prof_rows;
    unsigned long long calls, gpu, cpu, fallbacks, reasons[4];
    double calls_pct, total, gpu_ms, cpu_ms, time_pct, stages[8];   // [7]: other
    unsigned long long prof_calls, prof_gpu, prof_fallbacks;
    double prof_time_pct, prof_total, prof_gpu_ms;
} RoutineReport;

typedef struct {
    int pid;
    int headers, prof_headers;       // first lines naming ab_build_info()
    RoutineReport r[2];              // dgemm, zgemm
} ProcessReport;

static RoutineReport *routine_of(ProcessReport *p, const char *name) {
    return !strcmp(name, "dgemm") ? &p->r[0] : !strcmp(name, "zgemm") ? &p->r[1] : NULL;
}

// The report child's process, or the first other pid seen: its fork()ed child.
static ProcessReport *process_of(ProcessReport proc[2], int pid) {
    if (proc[0].pid == pid) return &proc[0];
    if (proc[1].pid == -1) proc[1].pid = pid;
    return proc[1].pid == pid ? &proc[1] : NULL;
}

// Parses one line of the report child's output. Returns false for an
// AB_REPORT line it cannot read.
static bool parse_report_line(const char *line, const char *build, ProcessReport proc[2],
                              ProcessReport **section, int *gpu_lines, int *fallback_lines) {
    char name[16], head[256];
    int pid, off = 0, end = 0;
    RoutineReport *r;
    double st[8], total;
    if (sscanf(line, "AB_REPORT[%d] %n", &pid, &off) == 1 && off) {
        ProcessReport *p = process_of(proc, pid);
        const char *rest = line + off;
        snprintf(head, sizeof head, "%s AB_MODE=gpu", build);
        if (!p) return false;
        if (!strcmp(rest, head)) {
            p->headers++;
            return true;
        }
        if (sscanf(rest, "%15s %n", name, &end) != 1 || !(r = routine_of(p, name))) return false;
        rest += end;
        if (sscanf(rest, "calls=%llu gpu_calls=%llu cpu_calls=%llu gpu_calls%%=%lf total_ms=%lf "
                         "gpu_ms=%lf cpu_ms=%lf gpu_time%%=%lf fallbacks=%llu",
                   &r->calls, &r->gpu, &r->cpu, &r->calls_pct, &r->total, &r->gpu_ms,
                   &r->cpu_ms, &r->time_pct, &r->fallbacks) == 9)
            r->rows++;
        else if (sscanf(rest, "gpu_stages_ms init=%lf alloc=%lf pack=%lf upload=%lf kernel=%lf "
                              "download=%lf epilogue=%lf other=%lf",
                        &r->stages[0], &r->stages[1], &r->stages[2], &r->stages[3],
                        &r->stages[4], &r->stages[5], &r->stages[6], &r->stages[7]) == 8)
            r->stage_rows++;
        else if (sscanf(rest, "fallbacks no_device=%llu too_large=%llu gpu_failed=%llu other=%llu",
                        &r->reasons[0], &r->reasons[1], &r->reasons[2], &r->reasons[3]) == 4)
            r->fallback_rows++;
        else
            return false;
        return true;
    }
    // AB_PROFILE table: its first line names the build and the pid; rows follow.
    snprintf(head, sizeof head, "=== %s BLAS profile, pid %%d ===", build);
    if (sscanf(line, head, &pid) == 1) {
        *section = process_of(proc, pid);
        if (*section) (*section)->prof_headers++;
        return *section != NULL;
    }
    unsigned long long calls, gpu, fb;
    double calls_pct, gpu_ms, cpu_ms, time_pct;
    if (*section && sscanf(line, "%15s %llu %llu %lf%% %lf %lf %lf %lf%% %llu%n", name, &calls,
                           &gpu, &calls_pct, &total, &gpu_ms, &cpu_ms, &time_pct, &fb,
                           &end) == 9 && end && (r = routine_of(*section, name))) {
        r->prof_rows++;
        r->prof_calls = calls; r->prof_gpu = gpu; r->prof_fallbacks = fb;
        r->prof_time_pct = time_pct; r->prof_total = total; r->prof_gpu_ms = gpu_ms;
        return true;
    }
    // AB_PROFILE per-call lines: one per GPU call, its stages within its total.
    if (sscanf(line, "AB_PROFILE %15s M=%*d N=%*d K=%*d init=%lf alloc=%lf pack=%lf upload=%lf "
                     "kernel=%lf download=%lf epilogue=%lf other=%lf total=%lf ms",
               name, &st[0], &st[1], &st[2], &st[3], &st[4], &st[5], &st[6], &st[7],
               &total) == 10) {
        double sum = 0.0;
        for (int i = 0; i < 8; i++) sum += st[i] >= 0.0 ? st[i] : INFINITY;
        if (sum <= total + 0.005) ++*gpu_lines;
        return true;
    }
    if (strstr(line, "AB_PROFILE ") == line && strstr(line, "GPU fallback to cblas: "))
        ++*fallback_lines;
    return true;
}

// The AB_REPORT and AB_PROFILE lines of one process for one routine against
// the calls it made. NULL if they agree.
static const char *check_routine(const RoutineReport *r, unsigned long long calls,
                                 unsigned long long gpu, unsigned long long no_device) {
    if (!calls)
        return r->rows || r->prof_rows ? "a routine without calls is reported" : NULL;
    if (r->rows != 1 || r->prof_rows != 1) return "routine missing from a report";
    if (r->calls != calls || r->gpu != gpu || r->cpu != calls - gpu)
        return "AB_REPORT call counts do not match the calls made";
    if (r->prof_calls != calls || r->prof_gpu != gpu || r->prof_fallbacks != no_device)
        return "AB_PROFILE counts do not match the calls made";
    if (fabs(r->calls_pct - 100.0 * gpu / calls) > 0.051) return "gpu_calls% is not the call share";
    if (!(r->time_pct >= 0.0 && r->time_pct <= 100.0) ||
        !(r->prof_time_pct >= 0.0 && r->prof_time_pct <= 100.0))
        return "gpu_time% outside [0, 100]";
    // gpu_time% is the share of time, not of calls (the two differ by up to 30x in QE).
    if (r->total > 0.0 && fabs(r->time_pct - 100.0 * r->gpu_ms / r->total) > 0.1)
        return "AB_REPORT gpu_time% is not gpu_ms / total_ms";
    if (r->prof_total > 0.0 &&
        fabs(r->prof_time_pct - 100.0 * r->prof_gpu_ms / r->prof_total) > 0.1)
        return "AB_PROFILE gpu_time% is not gpu_ms / total_ms";
    if (fabs(r->total - r->gpu_ms - r->cpu_ms) > 0.002) return "gpu_ms + cpu_ms != total_ms";
    if (!gpu && (r->gpu_ms != 0.0 || r->time_pct != 0.0)) return "GPU time without GPU calls";
    if (r->stage_rows != (gpu > 0)) return "GPU stage line missing or unexpected";
    if (gpu) {
        double sum = 0.0;                    // the named stages; [7] is the remainder
        for (int i = 0; i < 8; i++)
            if (r->stages[i] < 0.0) return "negative stage time";
        for (int i = 0; i < 7; i++) sum += r->stages[i];
        if (r->stages[4] <= 0.0) return "no kernel time";
        if (sum > r->gpu_ms + 0.004) return "GPU stages exceed gpu_ms";
        if (fabs(sum + r->stages[7] - r->gpu_ms) > 0.005) return "stages + other != gpu_ms";
    }
    if (r->fallbacks != no_device || r->fallback_rows != (no_device > 0) ||
        (no_device && (r->reasons[0] != no_device ||
                       r->reasons[1] + r->reasons[2] + r->reasons[3] != 0)))
        return "fallback counts do not match";
    return NULL;
}

// Spawns argv[0] with env, its stdout and stderr on a pipe. Returns the read
// end, or NULL with the failure in why.
static FILE *spawn_piped(char *const argv[], char *const env[], pid_t *pid,
                         char *why, size_t why_len) {
    int fds[2];
    if (pipe(fds) != 0) {
        snprintf(why, why_len, "pipe: %s", strerror(errno));
        return NULL;
    }
    posix_spawn_file_actions_t fa;
    posix_spawn_file_actions_init(&fa);
    posix_spawn_file_actions_adddup2(&fa, fds[1], STDOUT_FILENO);
    posix_spawn_file_actions_adddup2(&fa, fds[1], STDERR_FILENO);
    posix_spawn_file_actions_addclose(&fa, fds[0]);
    posix_spawn_file_actions_addclose(&fa, fds[1]);
    fflush(stdout);
    int rc = posix_spawn(pid, argv[0], &fa, NULL, argv, env);
    posix_spawn_file_actions_destroy(&fa);
    close(fds[1]);
    if (rc != 0) {
        close(fds[0]);
        snprintf(why, why_len, "posix_spawn: %s", strerror(rc));
        return NULL;
    }
    return fdopen(fds[0], "r");
}

// Runs --report-child with AB_REPORT=1 and AB_PROFILE=1 and checks what both
// of its processes report at exit: the build, call counts by route, time
// shares and stage sums, and that the fork()ed child reports only its own call.
static bool run_report(const AbiCase *c, char *detail, size_t n) {
    (void)c;
    static char report[] = "AB_REPORT=1", profile[] = "AB_PROFILE=1", flag[] = "--report-child";
    char self[PATH_MAX];
    uint32_t len = sizeof self;
    if (_NSGetExecutablePath(self, &len) != 0) {
        snprintf(detail, n, "cannot resolve executable path");
        return false;
    }
    size_t ne = 0;
    while (environ[ne]) ne++;
    char **env = calloc(ne + 3, sizeof *env);
    memcpy(env, environ, ne * sizeof *env);
    env[ne] = report;
    env[ne + 1] = profile;
    char *argv[] = {self, flag, NULL};
    pid_t pid;
    FILE *in = spawn_piped(argv, env, &pid, detail, n);
    free(env);
    if (!in) return false;
    char *out = NULL, buf[4096];
    size_t out_len = 0;
    FILE *mem = open_memstream(&out, &out_len);
    for (size_t got; (got = fread(buf, 1, sizeof buf, in)) > 0; ) fwrite(buf, 1, got, mem);
    fclose(in);
    fclose(mem);
    int status = 0;
    while (waitpid(pid, &status, 0) < 0 && errno == EINTR) {}

    // Expected counts: [0] the report child, [1] its fork()ed child.
    unsigned long long calls[2][2] = {{0}}, gpu[2][2] = {{0}}, no_device[2][2] = {{0}};
    for (size_t i = 0; i < TEST_REGISTRY_SIZE(REPORT_CALLS); i++) {
        int z = op_is_complex(REPORT_CALLS[i].op);
        calls[0][z]++;
        gpu[0][z] += REPORT_CALLS[i].gpu;
    }
    calls[1][op_is_complex(REPORT_CALLS[0].op)] = 1;
    no_device[1][op_is_complex(REPORT_CALLS[0].op)] = 1;

    ProcessReport proc[2] = {{.pid = pid}, {.pid = -1}}, *section = NULL;
    int fork_pid = 0, fork_status = -1, gpu_lines = 0, fallback_lines = 0;
    const char *build = ab_build_info(), *why = NULL;
    char bad[160] = "";
    for (char *save = NULL, *line = strtok_r(out, "\n", &save); line;
         line = strtok_r(NULL, "\n", &save))
        if (sscanf(line, "child %d %d", &fork_pid, &fork_status) != 2 &&
            !parse_report_line(line, build, proc, &section, &gpu_lines, &fallback_lines) &&
            !bad[0])
            snprintf(bad, sizeof bad, "%s", line);

    unsigned long long gpu_calls = gpu[0][0] + gpu[0][1];
    if (!WIFEXITED(status) || WEXITSTATUS(status) != 0) why = "report child failed";
    else if (fork_pid <= 0 || !WIFEXITED(fork_status) || WEXITSTATUS(fork_status) != 0)
        why = "forked child failed";
    else if (proc[1].pid != fork_pid) why = "no report from the forked child";
#ifdef AB_EXPECT_BUILD_SHA
    else if (!strstr(build, "(" AB_EXPECT_BUILD_SHA ")")) why = "ab_build_info() lacks the build SHA";
#endif
    else if (bad[0]) why = "unrecognized report line";
    else if (proc[0].headers != 1 || proc[1].headers != 1 ||
             proc[0].prof_headers != 1 || proc[1].prof_headers != 1)
        why = "each process must start each report once with ab_build_info()";
    else if ((unsigned long long)gpu_lines != gpu_calls || fallback_lines != 1)
        why = "AB_PROFILE per-call lines: wrong count, or stages exceed the call's total";
    for (int p = 0; p < 2 && !why; p++)
        for (int z = 0; z < 2 && !why; z++)
            why = check_routine(&proc[p].r[z], calls[p][z], gpu[p][z], no_device[p][z]);

    if (why)
        snprintf(detail, n, "%s%s%s (status %d, fork status %d)", why, bad[0] ? ": " : "", bad,
                 status, fork_status);
    else
        snprintf(detail, n, "%s; pid %d: %llu GPU of %llu calls; forked pid %d: 1 fallback",
                 build, (int)pid, gpu_calls, calls[0][0] + calls[0][1], proc[1].pid);
    free(out);
    return !why;
}

static bool run_case(int idx, char *detail, size_t n) {
    const AbiCase *c = &CASES[idx];
    rng_state = 0x9E3779B97F4A7C15ULL * (uint64_t)(idx + 1);
    if (c->fork) return run_fork(c, detail, n);
    switch (c->op) {
    case OP_DGEMM_BLAS: case OP_DGEMM_F: case OP_DGEMM_SYM:
        return run_blas_real(c, detail, n);
    case OP_ZGEMM_BLAS: case OP_ZGEMM_F: case OP_ZGEMM_SYM:
        return run_blas_complex(c, detail, n);
    case OP_ZGEMM:
        return run_native_complex(c, detail, n);
    case OP_SYMBOL:
        return run_symbol(c, detail, n);
    case OP_REPORT:
        return run_report(c, detail, n);
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
static char **child_env(const Suite *s, char *mode_buf, size_t mode_len,
                        char *table_buf, size_t table_len) {
    static char debug_layer[] = "MTL_DEBUG_LAYER=1";
    static char profile[] = "AB_PROFILE=1";
    size_t n = 0;
    while (environ[n]) n++;
    char **env = calloc(n + 5, sizeof *env);
    size_t k = 0;
    for (size_t i = 0; i < n; i++)
        if (strncmp(environ[i], "AB_", 3) && strncmp(environ[i], "MTL_DEBUG_LAYER=", 16))
            env[k++] = environ[i];
    if (s->ab_mode) {
        snprintf(mode_buf, mode_len, "AB_MODE=%s", s->ab_mode);
        env[k++] = mode_buf;
    }
    if (s->test_table) {
        snprintf(table_buf, table_len, "AB_DISPATCH_TABLE=%s", test_table_path);
        env[k++] = table_buf;
    }
    if (s->debug_layer) env[k++] = debug_layer;
    if (s->profile) env[k++] = profile;
    return env;
}

// Runs cases [lo, s->hi) in one child and records every reported case.
// Returns the first unreported index; *why describes an abnormal exit.
static int run_child(const Suite *s, int lo, char *why, size_t why_len) {
    char mode_buf[32], table_buf[PATH_MAX + 32], lo_arg[16], hi_arg[16], child_flag[] = "--child";
    char *argv[] = {self_path, child_flag, lo_arg, hi_arg, NULL};
    int next = lo;
    pid_t pid;

    why[0] = '\0';
    snprintf(lo_arg, sizeof lo_arg, "%d", lo);
    snprintf(hi_arg, sizeof hi_arg, "%d", s->hi);
    char **env = child_env(s, mode_buf, sizeof mode_buf, table_buf, sizeof table_buf);
    FILE *in = spawn_piped(argv, env, &pid, why, why_len);
    free(env);
    if (!in) return lo;

    // Result lines advance `next`; anything else (library or Metal
    // diagnostics) is kept so a crash can be explained.
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

// An illegal-argument case passes when its child ended the way cblas rejects
// it: exit status 255 after naming the expected parameter.
static bool rejected_by_cblas(const AbiCase *c, const char *why) {
    static const char exited[] = "child exited with status 255: ";
    char want[96];
    snprintf(want, sizeof want, "Parameter number %d passed to cblas_%cgemm had an invalid value",
             BAD_ARGS[c->bad].param, op_is_complex(c->op) ? 'z' : 'd');
    return !strncmp(why, exited, sizeof exited - 1) && strstr(why, want);
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
        if (next < s->hi && CASES[next].bad) {   // the expected end of this child
            record(&CASES[next], rejected_by_cblas(&CASES[next], why), why);
            next++;
            continue;
        }
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
static void test_dgemm_blas_noinit(void) { run_suite(SU_DGEMM_BLAS_NOINIT); }
static void test_zgemm_blas_noinit(void) { run_suite(SU_ZGEMM_BLAS_NOINIT); }
static void test_ab_dgemm_f_noinit(void) { run_suite(SU_DGEMM_F_NOINIT); }
static void test_ab_zgemm_f_noinit(void) { run_suite(SU_ZGEMM_F_NOINIT); }
static void test_dgemm_sym_noinit(void)  { run_suite(SU_DGEMM_SYM_NOINIT); }
static void test_zgemm_sym_noinit(void)  { run_suite(SU_ZGEMM_SYM_NOINIT); }
static void test_illegal_args(void)     { run_suite(SU_ILLEGAL); }
static void test_fork_after_gpu(void)   { run_suite(SU_FORK); }
static void test_qe_shapes(void)        { run_suite(SU_QE); }
static void test_profile_path(void)     { run_suite(SU_PROFILE); }
static void test_exit_reports(void)     { run_suite(SU_REPORT); }
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
    {"test_dgemm_blas_noinit", test_dgemm_blas_noinit},
    {"test_zgemm_blas_noinit", test_zgemm_blas_noinit},
    {"test_ab_dgemm_f_noinit", test_ab_dgemm_f_noinit},
    {"test_ab_zgemm_f_noinit", test_ab_zgemm_f_noinit},
    {"test_dgemm_sym_noinit",  test_dgemm_sym_noinit},
    {"test_zgemm_sym_noinit",  test_zgemm_sym_noinit},
    {"test_illegal_args",     test_illegal_args},
    {"test_fork_after_gpu",   test_fork_after_gpu},
    {"test_qe_shapes",        test_qe_shapes},
    {"test_profile_path",     test_profile_path},
    {"test_exit_reports",     test_exit_reports},
    {"test_native_morton",    test_native_morton},
    {"test_fortran_symbols",  test_fortran_symbols},
};
static const int N_TESTS = (int)TEST_REGISTRY_SIZE(TESTS);

int main(int argc, char **argv) {
    build_cases();
    if (argc == 4 && !strcmp(argv[1], "--child"))
        return child_main(atoi(argv[2]), atoi(argv[3]));
    if (argc == 2 && !strcmp(argv[1], "--report-child"))
        return report_child_main();
    verbose = argc > 1 && !strcmp(argv[1], "-v");

    uint32_t len = sizeof self_path;
    if (_NSGetExecutablePath(self_path, &len) != 0) {
        fprintf(stderr, "test_blas_abi: cannot resolve executable path\n");
        return 1;
    }
    const char *tmp = getenv("TMPDIR");
    snprintf(test_table_path, sizeof test_table_path, "%s/test_blas_abi.XXXXXX",
             tmp && *tmp ? tmp : "/tmp");
    int table_fd = mkstemp(test_table_path);
    if (table_fd < 0 || write(table_fd, TEST_TABLE, sizeof TEST_TABLE - 1) !=
                            (ssize_t)(sizeof TEST_TABLE - 1)) {
        fprintf(stderr, "test_blas_abi: cannot write %s: %s\n", test_table_path, strerror(errno));
        return 1;
    }
    close(table_fd);

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

    unlink(test_table_path);
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
