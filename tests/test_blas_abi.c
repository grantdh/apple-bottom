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
// The long-K, determinism, concurrency, memory and fork-exit suites guard
// properties a faster GPU path must keep (Kind, and the gen_* function of
// each); the bit-for-bit results themselves are test_qe_golden.c's job.
// The exponent-range suite gives the GEMMs data far outside FP32's exponent
// range, which DD shares (gen_range).
//
// Known failures live in the XFAIL table below, tagged with audit IDs. An
// XFAIL does not fail the run; an XFAIL case that passes (XPASS), an XFAIL
// entry that matches no case (STALE) and any other failure do. Run with -v
// to list every case.
// =============================================================================

#include "apple_bottom.h"
#include "test_blas_check.h"
#include "test_dot2.h"
#include "test_registry.h"
#include "test_spawn.h"
#include <Accelerate/Accelerate.h>
#include <complex.h>
#include <dlfcn.h>
#include <errno.h>
#include <fnmatch.h>
#include <float.h>
#include <limits.h>
#include <mach-o/dyld.h>
#include <mach/mach.h>
#include <malloc/malloc.h>
#include <math.h>
#include <pthread.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>
#include <sys/wait.h>
#include <unistd.h>

// AB_PROFILE table writer, exported by blas_profiler.c (internal header).
void ab_prof_dump(FILE *out);

#define MAX_CASES       1280
#define MIN_GPU_DIM     32      // AB_DISPATCH_DEFAULT_MIN_DIM in dispatch.h
#define OLD_MAX_DIM     46340   // v1.3 per-dimension limit, now a per-matrix one
#define CASE_TIMEOUT_S  120
#define FORK_WAIT_S     30      // a fork()ed child still running then is ended by SIGALRM

#define U2              0x1p-48 // u² = 2⁻⁴⁸, DD unit roundoff (u = 2⁻²⁴, FP32)
#define LONG_K_BOUND    8.0     // long-K error bound, units of u² (see gen_long_k)
#define IMAG_BOUND      8.0     // Re/Im error bound under imbalance, units of u² (gen_imbalance)
#define IMAG_RATIO      1e-6    // the smaller part's magnitude, relative (ImagData)
#define N_THREADS       8
#define THREAD_REPS     3       // calls per routine and thread
#define MEM_WARMUP_CALLS 200
#define MEM_CALLS       2000
#define MEM_FOOTPRINT_LIMIT_KB 32768 // growth over MEM_CALLS (see gen_memory)
#define MEM_MALLOC_LIMIT_KB    64    // growth over MEM_CALLS / 2, in at least one half

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

// What a BLAS-ABI case checks.
typedef enum {
    KIND_CALL,          // one call against cblas (blas_verdict); with fork, again in a child
    KIND_LONG_K,        // one call against a Dot2 reference, error in units of u²
    KIND_REPEAT,        // bitwise the same when repeated, also after other GPU calls
    KIND_THREADS,       // N_THREADS threads at once, each bitwise its single-threaded result
    KIND_MEMORY,        // footprint growth over MEM_CALLS GPU calls
    KIND_RANGE,         // one call on data outside FP32's exponent range (RangeData)
    KIND_IMAG,          // one ZGEMM on imbalanced parts, Re and Im each against Dot2 (ImagData)
} Kind;

static const char *const KIND_TAG[] = {
    [KIND_CALL] = "call", [KIND_LONG_K] = "long-k", [KIND_REPEAT] = "repeat",
    [KIND_THREADS] = "threads", [KIND_MEMORY] = "memory", [KIND_RANGE] = "range",
    [KIND_IMAG] = "imag",
};

// KIND_RANGE data. Every entry first becomes a magnitude in [0.5, 1) with a
// random sign (real and imaginary parts alike), so that the pattern alone
// sets how far apart the magnitudes are; then op(A)_ik, op(B)_kj and C0_ij
// are multiplied by the pattern's factors (range_factors). The native-API
// patterns act on row-major A (M x K), B (K x N) and C (M x N).
typedef enum {
    RANGE_TINY,         // every entry x 1e-25, C0 x 1e-50: the products underflow FP32
    RANGE_HUGE,         // every entry x 1e25, C0 x 1e50: the products overflow FP32
    RANGE_ROWCOL,       // row i of op(A) x 2^e_i, column j of op(B) x 2^f_j, C0_ij x 2^(e_i+f_j),
                        // e and f spread evenly over [-120, 120]
    RANGE_ALIGNED,      // op(A)_ik and op(B)_kj x 2^70 for even k and 2^-70 for odd k,
                        // C0 x 2^140: the even-k products dominate
    RANGE_MISALIGNED,   // op(A)_ik x 2^70 for even k and 2^-70 for odd k, op(B)_kj the
                        // reverse: every product is about 1, but scaling op(A)'s rows and
                        // op(B)'s columns to 1 would leave them all at 2^-140
    RANGE_INF,          // op(A)_00 = +Inf: C as cblas gives it (Inf, not DD's Inf - Inf = NaN)
    RANGE_ALPHA_TINY,   // alpha = 1e-45, below FP32's smallest normal; beta = 0
    RANGE_ALPHA_HUGE,   // alpha = 1e45, above FP32's largest; beta = 0
    RANGE_EXACT,        // row i of op(A) x 2^e_i, column j of op(B) x 2^f_j, e and f in
                        // {-100, 0, 100}: C must be 2^(e_i+f_j) times C of the unscaled
                        // data, bit for bit
    RANGE_ROWS_LOW,     // native: row i of A x 2^e_i, e_i spread over [-100, -60], C0 alike
    RANGE_COLS_HIGH,    // native: column j of B x 2^f_j, f_j over [60, 100], A x 2^30, C0 alike
    RANGE_UNIFORM_EXACT,// native: A and B x 2^90, C0 x 2^180: C must be 2^180 times C of the
                        // unscaled data, bit for bit
    RANGE_C_ZERO_ROW,   // row 0 of op(A) zero, row 0 of C0 x 2^-100: there beta C0 is all of C,
                        // below DD's precision (Accelerate must take it)
    RANGE_C_TINY,       // row 0 of op(A) zero, the other rows of C0 x 2^-100: they only round
                        // away beside the products (the GPU must certify it)
    RANGE_C_HUGE,       // C0 x 2^135, beta = 2^-20: beta C0 fits FP32, C0 does not (Accelerate)
} RangeData;

// KIND_IMAG data: entries of magnitude [0.5, 1) with random signs, as for
// KIND_RANGE, then one part of each row of op(A) and each column of op(B)
// multiplied by IMAG_RATIO, which Gauss-3M's imaginary part cannot take.
typedef enum {
    IMAG_SMALL,         // Im x IMAG_RATIO: nearly real A and B
    IMAG_LARGE,         // Re x IMAG_RATIO: nearly imaginary
    IMAG_LINES,         // even rows of op(A) and columns of op(B) nearly real, odd ones nearly
                        // imaginary: every row and column imbalanced, A and B as a whole not
} ImagData;

static const char *const IMAG_TAG[] = {
    [IMAG_SMALL] = "small-im", [IMAG_LARGE] = "small-re", [IMAG_LINES] = "lines",
};

static const char *const RANGE_TAG[] = {
    [RANGE_TINY] = "tiny", [RANGE_HUGE] = "huge", [RANGE_ROWCOL] = "rowcol",
    [RANGE_ALIGNED] = "aligned-span", [RANGE_MISALIGNED] = "misaligned-span",
    [RANGE_INF] = "inf", [RANGE_ALPHA_TINY] = "alpha-tiny", [RANGE_ALPHA_HUGE] = "alpha-huge",
    [RANGE_EXACT] = "exact", [RANGE_ROWS_LOW] = "rows-low", [RANGE_COLS_HIGH] = "cols-high",
    [RANGE_UNIFORM_EXACT] = "uniform-exact", [RANGE_C_ZERO_ROW] = "c-zero-row",
    [RANGE_C_TINY] = "c-tiny", [RANGE_C_HUGE] = "c-huge",
};

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
    bool child_exit;                    // that child ends with exit(), not _exit()
    bool same_shape;                    // KIND_THREADS: every thread makes thread 0's calls
    Kind kind;
    RangeData data;                     // KIND_RANGE only
    ImagData imag;                      // KIND_IMAG only
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
    bool exact_malloc;          // MallocNanoZone=0 (see gen_memory)
    bool show;                  // print every case's detail (measured values), not only failures
    Op op;                      // entry point (gen_blas suites only)
    void (*gen)(const Suite *s);
    int lo, hi;                 // case range, set by build_cases()
};

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

// As gen_fork without AB_PROFILE, but the fork()ed child ends with exit(), as a
// Fortran program does at STOP: atexit handlers and static destructors run in
// a process whose Metal objects did not survive fork(). The child must exit
// with status 0; one that crashes, or is still running after FORK_WAIT_S
// (its alarm ends it), fails the case.
static void gen_fork_exit(const Suite *s) {
    static const Op OPS[] = {OP_DGEMM_SYM, OP_ZGEMM_SYM};
    for (size_t o = 0; o < TEST_REGISTRY_SIZE(OPS); o++) {
        AbiCase *c = add_blas(s, OPS[o], 'N', 'N', 96, 80, 72, 3, SC_GEN, ROUTE_GPU);
        c->fork = c->child_exit = true;
    }
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

// A case of one of the kinds below, named <entry>/<mode>/<transA><transB>/<M>x<N>x<K>/<kind>.
static AbiCase *add_kind(const Suite *s, Kind kind, Op op, char ta, char tb, int M, int N, int K,
                         int pad, int scalars) {
    AbiCase *c = add_blas(s, op, ta, tb, M, N, K, pad, scalars, ROUTE_GPU);
    c->kind = kind;
    snprintf(c->name, sizeof c->name, "%s/%s/%c%c/%dx%dx%d/%s", OP_NAME[op], s->label, ta, tb,
             M, N, K, KIND_TAG[kind]);
    return c;
}

// Long K: QE's PAW and Davidson reductions run over K up to 2*ngm_l =
// 143666 per rank. C = op(A) op(B) (alpha = 1, beta = 0) with inputs uniform
// in [-1, 1) is compared with a compensated FP64 reference (Dot2, as in
// test_precision.c, accurate to ~u64 = u²/32). dgemm_ runs on the 32x32 tiles
// ('N','N' 144x36 is a tall C whose tall-skinny grid would have only 6
// threadgroups), zgemm_ ('C','N') on the Gauss 3M product; the 64x64 and
// tall-skinny kernels compute each output the same way. The error is max_ij |C - R|_ij / (|op(A)||op(B)|)_ij
// (moduli for complex), printed in units of u² = 2⁻⁴⁸ and of √K·u².
//
// Bound: per term, converting both inputs to DD costs ≤ 2u² and the DD
// product (DWTimesDW2) < 5u², at most 7u² of (|A||B|)_ij in all whatever K.
// The K DD additions each round a partial sum of size ~√k by ≤ c·u² of it;
// with random signs these add up like u²·(Σk)^½ ~ u²·K, while (|A||B|)_ij
// grows like K, so they too stay O(u²) as K grows. Measured on 1.4.0 (M2 Max):
// 1.6-2.9 u² for every K and kernel (zgemm_ 1.7-2.0 u²). LONG_K_BOUND = 8 u²
// is 2.8x the largest. The BLAS-ABI tolerance TOL = 1e-11 is ~2800 u², and one
// FP32 rounding in the sum would cost ~u = 2²⁴ u².
static const int LONG_K[] = {4096, 18277, 73062, 143666};

static void gen_long_k(const Suite *s) {
    for (size_t i = 0; i < TEST_REGISTRY_SIZE(LONG_K); i++) {
        add_kind(s, KIND_LONG_K, OP_DGEMM_SYM, 'T', 'N', 40, 36, LONG_K[i], 0, SC_A1B0);
        add_kind(s, KIND_LONG_K, OP_DGEMM_SYM, 'N', 'N', 144, 36, LONG_K[i], 0, SC_A1B0);
        add_kind(s, KIND_LONG_K, OP_ZGEMM_SYM, 'C', 'N', 40, 36, LONG_K[i], 0, SC_A1B0);
    }
}

// Determinism: each call, run three times from the same C0 (the third after
// REPEAT_OTHERS, GPU calls of other shapes and both routines), must give
// bitwise the same C, within TOL of cblas.
static void gen_repeat(const Suite *s) {
    add_kind(s, KIND_REPEAT, OP_DGEMM_SYM, 'T', 'N', 150, 150, 2000, 3, SC_GEN);   // cegterg-like
    add_kind(s, KIND_REPEAT, OP_ZGEMM_SYM, 'C', 'N', 150, 150, 2000, 3, SC_GEN);
    add_kind(s, KIND_REPEAT, OP_DGEMM_SYM, 'N', 'N', 600, 48, 40, 3, SC_GEN);      // tall C
    add_kind(s, KIND_REPEAT, OP_ZGEMM_SYM, 'N', 'T', 96, 80, 72, 3, SC_B1);
}

// Concurrency: N_THREADS threads, released together, each make THREAD_REPS
// dgemm_ and zgemm_ calls on their own data. In the first case each thread
// has its own shapes (thread_calls) and the process has not created the Metal
// context yet; in the second every thread makes thread 0's calls, so state
// the library keeps per shape (a buffer pool, say) is shared between threads.
// Every result must be bitwise the same call made single-threaded afterwards,
// and that within TOL of cblas.
static void gen_threads(const Suite *s) {
    for (int same = 0; same < 2; same++) {
        AbiCase *c = new_case();
        snprintf(c->name, sizeof c->name, "dgemm_+zgemm_/%s/%d-threads/%s", s->label, N_THREADS,
                 same ? "same-shapes" : "own-shapes");
        c->kind = KIND_THREADS;
        c->same_shape = same;
    }
}

// Memory: MEM_CALLS GPU calls (MEM_SHAPES in turn) after MEM_WARMUP_CALLS
// must grow the process's phys_footprint (task_info TASK_VM_INFO; it counts
// the Metal buffers the calls write) by less than MEM_FOOTPRINT_LIMIT_KB, and
// the bytes malloc has handed out (malloc_zone_statistics, all zones) by less
// than MEM_MALLOC_LIMIT_KB over the first or the second half of the calls.
// Measured on 1.4.0 (M2 Max), ~90 runs: footprint -912 to +608 KB, and once
// +11.1 MB (a region Metal maps at some point); malloc -1.5 to +11.5 KB over
// all the calls. Leaks planted in a probe making the same kind of calls: 170 B
// per call (what the autoreleased command buffers and encoders cost before
// ABAutoreleaseScope, apple_bottom.m) +375 KB malloc, which the footprint
// (+400-656 KB) cannot tell from noise; an unwritten 64x64 ABMatrix per call
// +1.4 MB malloc, +1.9-2.2 MB footprint. So malloc catches small leaks, and
// the footprint limit (~3x the outlier) the large ones: a leaked packed
// operand or written GPU buffer costs >= 30 KB per call, >= 60 MB.
//
// Halves: a leak grows both alike (MEM_MALLOC_LIMIT_KB per half is 64 B per
// call), memory allocated once only one. Malloc growth of 3943a96 (M2 Max):
// 0-4 KB in 19 runs, idle or beside CPU, GPU and memory-bandwidth load;
// beside other jobs at load1 23-33, 141-169 KB in 6 of 9 runs and 8-13 KB in
// 3. No call leaves memory to be freed after it returns.
//
// The child runs with MallocNanoZone=0 (Suite.exact_malloc). The default nano
// allocator counts small blocks another thread freed as still in use, in 16 KB
// steps (341 blocks of 48 B), and Metal frees some objects on its own threads:
// 0-32 KB over 2000 calls, no more over 8000. Without Metal, one thread
// allocating 16 blocks of 48 B and another freeing them gains 8.5 KB in 2000
// rounds and 14.5 KB in 32000, and nothing with MallocNanoZone=0.
static void gen_memory(const Suite *s) {
    AbiCase *c = new_case();
    snprintf(c->name, sizeof c->name, "dgemm_+zgemm_/%s/%d-calls/footprint", s->label, MEM_CALLS);
    c->kind = KIND_MEMORY;
}

// Exponent range. DD has FP32's exponent range: entries or products below
// about 2^-126 flush to zero, low parts below about 2^-78 times the result
// are lost, and above about 2^128 they overflow, silently in 1.4.
//   - Through dgemm_ / zgemm_ (AB_MODE=gpu, no ab_init()), each call must
//     run on the GPU where power-of-two scaling of op(A)'s rows and op(B)'s
//     columns puts every product back in range (ROUTE_GPU), and on Accelerate
//     where no such scaling can (ROUTE_CPU: NaN or Inf, or RANGE_MISALIGNED).
//   - The native ABMatrix cases call ab_init() and run on the GPU; a matrix
//     keeps one power-of-two scale there.
// Every case must give C within TOL of cblas, with Inf and NaN where cblas
// has them; the EXACT cases must give C bit for bit as a power of two times
// C of the unscaled data.
static AbiCase *add_range(const Suite *s, Op op, char ta, char tb, RangeData data, Route route) {
    const int M = 96, N = 80, K = 72;
    AbiCase *c = add_blas(s, op, ta, tb, M, N, K, 3, SC_GEN, route);
    c->kind = KIND_RANGE;
    c->data = data;
    if (data == RANGE_ALPHA_TINY || data == RANGE_ALPHA_HUGE) {
        c->alpha = data == RANGE_ALPHA_TINY ? 1e-45 : 1e45;
        c->beta = 0.0;
    }
    if (data == RANGE_C_HUGE) c->beta = 0x1p-20;
    if (op == OP_DGEMM || op == OP_DGEMM_SCALED || op == OP_ZGEMM) {
        c->pad = 0;
        snprintf(c->name, sizeof c->name, "%s/%dx%dx%d/range-%s", OP_NAME[op], M, N, K,
                 RANGE_TAG[data]);
    } else {
        snprintf(c->name, sizeof c->name, "%s/%s/%c%c/%dx%dx%d/range-%s", OP_NAME[op], s->label,
                 ta, tb, M, N, K, RANGE_TAG[data]);
    }
    return c;
}

static void gen_range(const Suite *s) {
    static const struct { Op op; char ta, tb; RangeData data; Route route; } R[] = {
        {OP_DGEMM_SYM, 'N', 'N', RANGE_TINY, ROUTE_GPU},
        {OP_ZGEMM_SYM, 'N', 'N', RANGE_TINY, ROUTE_GPU},
        {OP_DGEMM_SYM, 'T', 'N', RANGE_HUGE, ROUTE_GPU},
        {OP_ZGEMM_SYM, 'C', 'N', RANGE_HUGE, ROUTE_GPU},
        {OP_DGEMM_SYM, 'N', 'N', RANGE_ROWCOL, ROUTE_GPU},
        {OP_DGEMM_SYM, 'T', 'T', RANGE_ROWCOL, ROUTE_GPU},
        {OP_ZGEMM_SYM, 'N', 'C', RANGE_ROWCOL, ROUTE_GPU},
        {OP_ZGEMM_SYM, 'C', 'T', RANGE_ROWCOL, ROUTE_GPU},
        {OP_DGEMM_SYM, 'N', 'T', RANGE_ALIGNED, ROUTE_GPU},
        {OP_ZGEMM_SYM, 'T', 'N', RANGE_ALIGNED, ROUTE_GPU},
        {OP_DGEMM_SYM, 'N', 'N', RANGE_MISALIGNED, ROUTE_CPU},
        {OP_ZGEMM_SYM, 'N', 'N', RANGE_MISALIGNED, ROUTE_CPU},
        {OP_DGEMM_SYM, 'N', 'N', RANGE_INF, ROUTE_CPU},
        {OP_ZGEMM_SYM, 'C', 'N', RANGE_INF, ROUTE_CPU},
        {OP_DGEMM_SYM, 'N', 'N', RANGE_ALPHA_TINY, ROUTE_GPU},
        {OP_DGEMM_SYM, 'T', 'N', RANGE_ALPHA_HUGE, ROUTE_GPU},
        {OP_DGEMM_SYM, 'N', 'N', RANGE_C_ZERO_ROW, ROUTE_CPU},
        {OP_DGEMM_SYM, 'T', 'T', RANGE_C_TINY, ROUTE_GPU},
        {OP_DGEMM_SYM, 'N', 'T', RANGE_C_HUGE, ROUTE_CPU},
        {OP_DGEMM_SYM, 'N', 'N', RANGE_EXACT, ROUTE_GPU},
        {OP_DGEMM_SYM, 'T', 'T', RANGE_EXACT, ROUTE_GPU},
        {OP_ZGEMM_SYM, 'N', 'N', RANGE_EXACT, ROUTE_GPU},
        {OP_ZGEMM_SYM, 'C', 'T', RANGE_EXACT, ROUTE_GPU},
        {OP_DGEMM, 'N', 'N', RANGE_TINY, ROUTE_GPU},
        {OP_DGEMM, 'N', 'N', RANGE_HUGE, ROUTE_GPU},
        {OP_DGEMM_SCALED, 'N', 'N', RANGE_TINY, ROUTE_GPU},
        {OP_DGEMM_SCALED, 'N', 'N', RANGE_HUGE, ROUTE_GPU},
        {OP_DGEMM_SCALED, 'N', 'N', RANGE_ROWS_LOW, ROUTE_GPU},
        {OP_DGEMM_SCALED, 'N', 'N', RANGE_COLS_HIGH, ROUTE_GPU},
        {OP_ZGEMM, 'N', 'N', RANGE_TINY, ROUTE_GPU},
        {OP_ZGEMM, 'N', 'N', RANGE_HUGE, ROUTE_GPU},
        {OP_ZGEMM, 'N', 'N', RANGE_ROWS_LOW, ROUTE_GPU},
        {OP_ZGEMM, 'N', 'N', RANGE_COLS_HIGH, ROUTE_GPU},
        {OP_DGEMM, 'N', 'N', RANGE_UNIFORM_EXACT, ROUTE_GPU},
        {OP_ZGEMM, 'N', 'N', RANGE_UNIFORM_EXACT, ROUTE_GPU},
    };
    for (size_t i = 0; i < TEST_REGISTRY_SIZE(R); i++)
        add_range(s, R[i].op, R[i].ta, R[i].tb, R[i].data, R[i].route);
}

// Gauss-3M's imaginary part under Re/Im imbalance (audit K-P1-2): zgemm_
// with alpha = 1, beta = 0 on ImagData, every TRANS kind of op(). Re(C) and
// Im(C) must each be within IMAG_BOUND u² of a Dot2 reference, relative to
// their own componentwise scales |Ar||Br| + |Ai||Bi| and |Ar||Bi| + |Ai||Br|;
// 3M misses Im by about 1/(2 IMAG_RATIO) of them. IMAG_LINES needs the
// row- and column-wise check: A's and B's largest parts are balanced.
static void gen_imbalance(const Suite *s) {
    static const struct { char ta, tb; ImagData d; } IMB[] = {
        {'N', 'N', IMAG_SMALL}, {'C', 'N', IMAG_SMALL}, {'N', 'T', IMAG_LARGE},
        {'T', 'C', IMAG_LARGE}, {'N', 'N', IMAG_LINES}, {'C', 'T', IMAG_LINES},
    };
    for (size_t i = 0; i < TEST_REGISTRY_SIZE(IMB); i++) {
        AbiCase *c = add_kind(s, KIND_IMAG, OP_ZGEMM_SYM, IMB[i].ta, IMB[i].tb, 96, 80, 72, 3,
                              SC_A1B0);
        c->imag = IMB[i].d;
        snprintf(c->name, sizeof c->name, "zgemm_/%s/%c%c/96x80x72/imag-%s", s->label,
                 IMB[i].ta, IMB[i].tb, IMAG_TAG[IMB[i].d]);
    }
}

static void add_native(Op op, int M, int N, int K) {
    AbiCase *c = new_case();
    snprintf(c->name, sizeof c->name, "%s/%dx%dx%d", OP_NAME[op], M, N, K);
    c->op = op; c->M = M; c->N = N; c->K = K;
    c->alpha = SCALARS[SC_GEN].alpha; c->beta = SCALARS[SC_GEN].beta;   // ab_dgemm_scaled only
}

// Native ABMatrix GEMMs (row-major) on tile grids that are powers of two but
// neither square nor 2:1 wide (K-P0-1 regressions; 2048x128 uses the 128x16
// tall-skinny tiles where the entry point routes M >= 4N there, 512x256 and
// 1024x64 the 32x32 ones).
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
    SU_DGEMM_SYM_NOINIT, SU_ZGEMM_SYM_NOINIT, SU_ILLEGAL, SU_FORK, SU_FORK_EXIT, SU_QE,
    SU_LONG_K, SU_REPEAT, SU_THREADS, SU_MEMORY, SU_RANGE, SU_IMBALANCE,
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
    [SU_FORK_EXIT] = {.label = "gpu-noinit-fork-exit", .ab_mode = "gpu", .gen = gen_fork_exit},
    [SU_QE]      = {.label = "gpu-noinit", .ab_mode = "gpu", .gen = gen_qe},
    [SU_LONG_K]  = {.label = "gpu-noinit", .ab_mode = "gpu", .show = true, .gen = gen_long_k},
    [SU_REPEAT]  = {.label = "gpu-noinit", .ab_mode = "gpu", .gen = gen_repeat},
    [SU_THREADS] = {.label = "gpu-noinit", .ab_mode = "gpu", .show = true, .gen = gen_threads},
    [SU_MEMORY]  = {.label = "gpu-noinit", .ab_mode = "gpu", .exact_malloc = true, .show = true,
                    .gen = gen_memory},
    [SU_RANGE]   = {.label = "gpu-noinit", .ab_mode = "gpu", .show = true, .gen = gen_range},
    [SU_IMBALANCE] = {.label = "gpu-noinit", .ab_mode = "gpu", .show = true, .gen = gen_imbalance},
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

// One call through the real entry point op.
static void call_real(Op op, const BlasArgs *a, double alpha, const double *A, const double *B,
                      double beta, double *C) {
    switch (op) {
    case OP_DGEMM_BLAS:
        ab_dgemm_blas(a->ta, a->tb, a->M, a->N, a->K, alpha, A, a->lda, B, a->ldb, beta, C, a->ldc);
        break;
    case OP_DGEMM_F:
        ab_dgemm_(&a->ta, &a->tb, &a->M, &a->N, &a->K, &alpha, A, &a->lda, B, &a->ldb, &beta, C,
                  &a->ldc);
        break;
    default:
        fortran_dgemm(&a->ta, &a->tb, &a->M, &a->N, &a->K, &alpha, A, &a->lda, B, &a->ldb, &beta,
                      C, &a->ldc);
        break;
    }
}

// One call through the complex entry point op.
static void call_complex(Op op, const BlasArgs *a, double complex alpha, const double complex *A,
                         const double complex *B, double complex beta, double complex *C) {
    switch (op) {
    case OP_ZGEMM_BLAS:
        ab_zgemm_blas(a->ta, a->tb, a->M, a->N, a->K, alpha, A, a->lda, B, a->ldb, beta, C, a->ldc);
        break;
    case OP_ZGEMM_F:
        ab_zgemm_(&a->ta, &a->tb, &a->M, &a->N, &a->K, &alpha, A, &a->lda, B, &a->ldb, &beta, C,
                  &a->ldc);
        break;
    default:
        fortran_zgemm(&a->ta, &a->tb, &a->M, &a->N, &a->K, &alpha, A, &a->lda, B, &a->ldb, &beta,
                      C, &a->ldc);
        break;
    }
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

    blas_reference(false, c->ta, c->tb, M, N, K, alpha, beta, A, la, B, lb, C0, lc.ld, Cref,
                   absAB);

    const BlasArgs a = blas_args(c, la, lb, lc);
    uint64_t g0 = gpu_gemm_count();
    call_real(c->op, &a, alpha, A, B, beta, C);
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

    blas_reference(true, c->ta, c->tb, M, N, K, alpha, beta, A, la, B, lb, C0, lc.ld, Cref,
                   absAB);

    const BlasArgs a = blas_args(c, la, lb, lc);
    uint64_t g0 = gpu_gemm_count();
    call_complex(c->op, &a, alpha, A, B, beta, C);
    uint64_t gpu_gemms = gpu_gemm_count() - g0;

    double err = err_complex(M, N, C, Cref, lc.ld, absAB, imax(1, M), C0, alpha, beta);
    bool inputs_ok = !memcmp(A, A0, na * sizeof *A) && !memcmp(B, B0, nb * sizeof *B);
    bool outside_ok = outside_block_unchanged(C, C0, nc, lc, sizeof *C);
    bool ok = blas_verdict(c, err, inputs_ok, outside_ok, gpu_gemms, detail, n);

    free(A); free(A0); free(B); free(B0); free(C); free(C0); free(Cref); free(absAB);
    return ok;
}

// Runs c on the GPU, then fork()s; the child repeats c, which must now run
// on cblas (ROUTE_CPU) and pass the same checks, and ends with _exit(), or
// exit() for c->child_exit. It reports through a shared page and its exit
// status; an alarm ends it (SIGALRM) if it is still running after FORK_WAIT_S.
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
        alarm(FORK_WAIT_S);             // fork() cleared the parent's CASE_TIMEOUT_S alarm
        AbiCase in_child = *c;
        in_child.route = ROUTE_CPU;
        memset(prof_calls, 0, sizeof prof_calls);
        memset(prof_gpu, 0, sizeof prof_gpu);
        bool child_ok = run(&in_child, after, page);
        if (c->child_exit) exit(child_ok ? 0 : 1);
        _exit(child_ok ? 0 : 1);
    }
    int status = 0;
    if (pid < 0)
        snprintf(how, sizeof how, " [fork: %s]", strerror(errno));
    else
        while (waitpid(pid, &status, 0) < 0 && errno == EINTR) {}
    if (pid > 0 && WIFSIGNALED(status))
        snprintf(how, sizeof how, " [killed by signal %d (%s)]", WTERMSIG(status),
                 strsignal(WTERMSIG(status)));
    else if (pid > 0 && WIFEXITED(status) && WEXITSTATUS(status) != 0)
        snprintf(how, sizeof how, " [exit status %d]", WEXITSTATUS(status));
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
    abs_product(false, B, lb, 'N', A, la, 'N', N, M, K, absAB);

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
    abs_product(true, B, lb, 'N', A, la, 'N', N, M, K, absAB);

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
// Long K, determinism, concurrency, memory
// =============================================================================

// A BLAS-ABI call's entry point, TRANS pair and dimensions.
typedef struct { Op op; char ta, tb; int M, N, K; } Shape;

// A BLAS-ABI call set up for repeated use: A, B and C0 filled as in
// run_blas_real / run_blas_complex (NaN in the padding); every run starts
// from C0.
typedef struct {
    AbiCase c;
    bool cplx;
    Layout la, lb, lc;
    size_t es, nc;                      // element size, elements of C
    void *A, *B, *C0;
} Call;

static void call_init(Call *k, Shape s, int pad, const Scalars *sc) {
    memset(k, 0, sizeof *k);
    k->c.op = s.op; k->c.ta = s.ta; k->c.tb = s.tb;
    k->c.M = s.M; k->c.N = s.N; k->c.K = s.K; k->c.pad = pad;
    k->c.alpha = sc->alpha; k->c.beta = sc->beta;
    k->cplx = op_is_complex(s.op);
    k->es = k->cplx ? sizeof(double complex) : sizeof(double);
    k->la = stored(s.ta, s.M, s.K, pad);
    k->lb = stored(s.tb, s.K, s.N, pad);
    k->lc = (Layout){s.M, s.N, imax(1, s.M + pad)};
    k->nc = nelem(k->lc);
    k->A = malloc(nelem(k->la) * k->es);
    k->B = malloc(nelem(k->lb) * k->es);
    k->C0 = malloc(k->nc * k->es);
    if (k->cplx) {
        fill_complex(k->A, k->la, false);
        fill_complex(k->B, k->lb, false);
        fill_complex(k->C0, k->lc, false);
    } else {
        fill_real(k->A, k->la, false);
        fill_real(k->B, k->lb, false);
        fill_real(k->C0, k->lc, false);
    }
}

static void call_init_case(Call *k, const AbiCase *c) {
    const Scalars sc = {.alpha = c->alpha, .beta = c->beta};
    call_init(k, (Shape){c->op, c->ta, c->tb, c->M, c->N, c->K}, c->pad, &sc);
}

static void call_free(Call *k) {
    free(k->A); free(k->B); free(k->C0);
}

// Runs the call into out, reset from C0 first. True if the GPU computed it
// (the calling thread's path record).
static bool call_run(const Call *k, void *out) {
    const AbiCase *c = &k->c;
    const BlasArgs a = blas_args(c, k->la, k->lb, k->lc);
    memcpy(out, k->C0, k->nc * k->es);
    if (k->cplx)
        call_complex(c->op, &a, c->alpha, k->A, k->B, c->beta, out);
    else
        call_real(c->op, &a, creal(c->alpha), k->A, k->B, creal(c->beta), out);
    return !strcmp(ab_get_last_dispatch_path(), "gpu");
}

// out's componentwise error against cblas, as blas_verdict measures it, in
// *err; true if it is within TOL and C outside its M x N block is unchanged.
static bool call_check(const Call *k, const void *out, double *err) {
    const AbiCase *c = &k->c;
    *err = blas_error(k->cplx, c->ta, c->tb, c->M, c->N, c->K, c->alpha, c->beta, k->A, k->la,
                      k->B, k->lb, k->C0, k->lc, out);
    return *err <= TOL && outside_block_unchanged(out, k->C0, k->nc, k->lc, k->es);
}

// --- Long K -------------------------------------------------------------------

// op(X), rows x cols, as row-major real and imaginary parts (im NULL for a
// real X); 'C' conjugates.
static void pack_rows(const void *X, Layout l, char t, bool cplx, int rows, int cols,
                      double *re, double *im) {
    for (int i = 0; i < rows; i++)
        for (int j = 0; j < cols; j++) {
            size_t x = op_index(t, l, i, j);
            size_t y = (size_t)i * cols + j;
            if (!cplx) {
                re[y] = ((const double *)X)[x];
            } else {
                double complex v = ((const double complex *)X)[x];
                re[y] = creal(v);
                im[y] = t == 'C' || t == 'c' ? -cimag(v) : cimag(v);
            }
        }
}

static bool run_long_k(const AbiCase *c, char *detail, size_t n) {
    const int M = c->M, N = c->N, K = c->K;
    Call k;
    call_init_case(&k, c);
    void *C = malloc(k.nc * k.es);
    uint64_t g0 = gpu_gemm_count();
    bool gpu = call_run(&k, C) && gpu_gemm_count() > g0;

    // R = op(A) op(B) in Dot2: Re = Ar Br - Ai Bi, Im = Ar Bi + Ai Br.
    const size_t na = (size_t)M * K, nb = (size_t)K * N, nr = (size_t)M * N;
    double *Ar = malloc(na * sizeof *Ar), *Ai = k.cplx ? malloc(na * sizeof *Ai) : NULL;
    double *Br = malloc(nb * sizeof *Br), *Bi = k.cplx ? malloc(nb * sizeof *Bi) : NULL;
    double *S[2] = {calloc(nr, sizeof(double)), calloc(nr, sizeof(double))};
    double *E[2] = {calloc(nr, sizeof(double)), calloc(nr, sizeof(double))};
    pack_rows(k.A, k.la, c->ta, k.cplx, M, K, Ar, Ai);
    pack_rows(k.B, k.lb, c->tb, k.cplx, K, N, Br, Bi);
    if (k.cplx)
        dot2_zgemm_acc(M, N, K, Ar, Ai, Br, Bi, S, E);
    else
        dot2_gemm_acc(M, N, K, 1.0, Ar, Br, S[0], E[0]);
    free(Ar); free(Ai); free(Br); free(Bi);

    // R = S + E in C's layout (re, im pairs for complex), and the error
    // against |op(A)||op(B)| (moduli for complex).
    const int w = k.cplx ? 2 : 1;
    double *R = malloc(k.nc * k.es), *absAB = malloc(nr * sizeof *absAB);
    for (int j = 0; j < N; j++)
        for (int i = 0; i < M; i++)
            for (int p = 0; p < w; p++) {
                size_t y = (size_t)i * N + j;
                R[((size_t)j * k.lc.ld + i) * w + p] = S[p][y] + E[p][y];
            }
    abs_product(k.cplx, k.A, k.la, c->ta, k.B, k.lb, c->tb, M, N, K, absAB);
    double err = k.cplx ? err_complex(M, N, C, (double complex *)R, k.lc.ld, absAB, M, k.C0,
                                      1.0, 0.0)
                        : err_real(M, N, C, R, k.lc.ld, absAB, M, k.C0, 1.0, 0.0);
    bool outside_ok = outside_block_unchanged(C, k.C0, k.nc, k.lc, k.es);
    snprintf(detail, n, "max |C-R|/(|A||B|) = %.2f u² = %.4f √K·u² (bound %.0f u²) path=%s%s",
             err / U2, err / (sqrt((double)K) * U2), LONG_K_BOUND, ab_get_last_dispatch_path(),
             outside_ok ? "" : " [C outside MxN modified]");

    free(S[0]); free(S[1]); free(E[0]); free(E[1]); free(absAB); free(R); free(C);
    call_free(&k);
    return gpu && outside_ok && err <= LONG_K_BOUND * U2;
}

// --- Determinism ----------------------------------------------------------------

// The GPU calls a KIND_REPEAT case makes between its second and third run.
static const Shape REPEAT_OTHERS[] = {
    {OP_ZGEMM_SYM, 'N', 'T', 200, 120, 300},
    {OP_DGEMM_SYM, 'N', 'N', 1024, 64, 64},     // 32x32 kernel (scaled), tall C
    {OP_DGEMM_SYM, 'T', 'N', 512, 256, 64},
    {OP_ZGEMM_SYM, 'C', 'N', 64, 64, 4096},
};

static bool run_repeat(const AbiCase *c, char *detail, size_t n) {
    Call k;
    call_init_case(&k, c);
    const size_t bytes = k.nc * k.es;
    void *C[3] = {malloc(bytes), malloc(bytes), malloc(bytes)};
    bool gpu = call_run(&k, C[0]);
    gpu = call_run(&k, C[1]) && gpu;
    for (size_t i = 0; i < TEST_REGISTRY_SIZE(REPEAT_OTHERS); i++) {
        Call o;
        call_init(&o, REPEAT_OTHERS[i], 1, &SCALARS[SC_GEN]);
        void *out = malloc(o.nc * o.es);
        gpu = call_run(&o, out) && gpu;
        free(out);
        call_free(&o);
    }
    gpu = call_run(&k, C[2]) && gpu;

    bool same2 = !memcmp(C[0], C[1], bytes), same3 = !memcmp(C[0], C[2], bytes);
    double err;
    bool ok = call_check(&k, C[0], &err);
    snprintf(detail, n, "err=%.1e%s%s%s%s", err, gpu ? "" : " [a call did not run on the GPU]",
             ok ? "" : " [C outside tolerance or outside MxN modified]",
             same2 ? "" : " [second run differs]",
             same3 ? "" : " [run after other GPU calls differs]");
    free(C[0]); free(C[1]); free(C[2]);
    call_free(&k);
    return gpu && ok && same2 && same3;
}

// --- Concurrency ----------------------------------------------------------------

// Thread t's calls: its own TRANS pair, shape and leading-dimension padding.
static void thread_calls(int t, Call call[2]) {
    static const char DT[][3] = {"NN", "TN", "NT", "TT"}, ZT[][3] = {"CN", "NN", "NC", "TC"};
    const int sd = t % 2 ? SC_B1 : SC_GEN, sz = t % 2 ? SC_GEN : SC_B1;
    call_init(&call[0], (Shape){OP_DGEMM_SYM, DT[t % 4][0], DT[t % 4][1],
                                64 + 24 * t, 48 + 16 * t, 80 + 40 * t},
              t % 4, &SCALARS[sd]);
    call_init(&call[1], (Shape){OP_ZGEMM_SYM, ZT[t % 4][0], ZT[t % 4][1],
                                56 + 16 * t, 40 + 24 * t, 96 + 32 * t},
              (t + 1) % 4, &SCALARS[sz]);
}

typedef struct {
    Call call[2];                       // dgemm_, zgemm_
    void *out[2][THREAD_REPS];
    bool gpu[2][THREAD_REPS];
} ThreadJob;

static pthread_mutex_t go_mutex = PTHREAD_MUTEX_INITIALIZER;
static pthread_cond_t go_cond = PTHREAD_COND_INITIALIZER;
static bool threads_go;

static void *thread_main(void *arg) {
    ThreadJob *job = arg;
    pthread_mutex_lock(&go_mutex);
    while (!threads_go) pthread_cond_wait(&go_cond, &go_mutex);
    pthread_mutex_unlock(&go_mutex);
    for (int r = 0; r < THREAD_REPS; r++)
        for (int z = 0; z < 2; z++)
            job->gpu[z][r] = call_run(&job->call[z], job->out[z][r]);
    return NULL;
}

static bool run_threads(const AbiCase *c, char *detail, size_t n) {
    ThreadJob *jobs = calloc(N_THREADS, sizeof *jobs);
    for (int t = 0; t < N_THREADS; t++) {
        thread_calls(c->same_shape ? 0 : t, jobs[t].call);
        for (int z = 0; z < 2; z++)
            for (int r = 0; r < THREAD_REPS; r++)
                jobs[t].out[z][r] = malloc(jobs[t].call[z].nc * jobs[t].call[z].es);
    }

    pthread_t th[N_THREADS];
    int started = 0;
    threads_go = false;
    while (started < N_THREADS && pthread_create(&th[started], NULL, thread_main, &jobs[started]) == 0)
        started++;
    pthread_mutex_lock(&go_mutex);
    threads_go = true;
    pthread_cond_broadcast(&go_cond);
    pthread_mutex_unlock(&go_mutex);
    for (int t = 0; t < started; t++) pthread_join(th[t], NULL);

    // The same calls single-threaded, after the concurrent ones.
    int not_gpu = 0, differ = 0, bad = 0;
    double worst = 0.0;
    for (int t = 0; t < started; t++)
        for (int z = 0; z < 2; z++) {
            const Call *k = &jobs[t].call[z];
            void *ref = malloc(k->nc * k->es);
            not_gpu += !call_run(k, ref);
            double err;
            bad += !call_check(k, ref, &err);
            if (!(err <= worst)) worst = err;
            for (int r = 0; r < THREAD_REPS; r++) {
                not_gpu += !jobs[t].gpu[z][r];
                differ += memcmp(jobs[t].out[z][r], ref, k->nc * k->es) != 0;
            }
            free(ref);
        }
    snprintf(detail, n, "%d threads x %d dgemm_ + %d zgemm_ calls: %d not on the GPU, "
             "%d differ from the single-threaded result, %d of %d single-threaded results "
             "outside tolerance (worst err=%.1e)",
             started, THREAD_REPS, THREAD_REPS, not_gpu, differ, bad, 2 * started, worst);

    for (int t = 0; t < N_THREADS; t++)
        for (int z = 0; z < 2; z++) {
            for (int r = 0; r < THREAD_REPS; r++) free(jobs[t].out[z][r]);
            call_free(&jobs[t].call[z]);
        }
    free(jobs);
    return started == N_THREADS && !not_gpu && !differ && !bad;
}

// --- Memory -------------------------------------------------------------------

// The calls of the memory case, in turn.
static const Shape MEM_SHAPES[] = {
    {OP_DGEMM_SYM, 'N', 'N', 64, 64, 64},
    {OP_ZGEMM_SYM, 'C', 'N', 48, 40, 300},
    {OP_DGEMM_SYM, 'T', 'N', 96, 40, 200},
    {OP_ZGEMM_SYM, 'N', 'N', 160, 40, 48},      // tall C
};
#define N_MEM_SHAPES ((int)TEST_REGISTRY_SIZE(MEM_SHAPES))

typedef struct { uint64_t phys, resident, malloced; } MemUse;

// The process's physical footprint and resident size, and the bytes malloc
// has handed out (all zones).
static bool mem_use(MemUse *m) {
    task_vm_info_data_t info;
    mach_msg_type_number_t count = TASK_VM_INFO_COUNT;
    if (task_info(mach_task_self(), TASK_VM_INFO, (task_info_t)&info, &count) != KERN_SUCCESS)
        return false;
    malloc_statistics_t st;
    malloc_zone_statistics(NULL, &st);
    m->phys = info.phys_footprint;
    m->resident = info.resident_size;
    m->malloced = st.size_in_use;
    return true;
}

static double kb_change(uint64_t before, uint64_t after) {
    return ((double)after - (double)before) / 1024.0;
}

static bool run_memory(const AbiCase *c, char *detail, size_t n) {
    (void)c;
    Call calls[N_MEM_SHAPES];
    void *out[N_MEM_SHAPES];
    for (int i = 0; i < N_MEM_SHAPES; i++) {
        call_init(&calls[i], MEM_SHAPES[i], 2, &SCALARS[SC_GEN]);
        out[i] = malloc(calls[i].nc * calls[i].es);
    }
    int not_gpu = 0;
    for (int i = 0; i < MEM_WARMUP_CALLS; i++)
        not_gpu += !call_run(&calls[i % N_MEM_SHAPES], out[i % N_MEM_SHAPES]);
    MemUse m0 = {0}, mh = {0}, m1 = {0};    // before, halfway, after
    bool measured = mem_use(&m0);
    uint64_t g0 = gpu_gemm_count();
    for (int i = 0; i < MEM_CALLS; i++) {
        if (i == MEM_CALLS / 2) measured = mem_use(&mh) && measured;
        not_gpu += !call_run(&calls[i % N_MEM_SHAPES], out[i % N_MEM_SHAPES]);
    }
    uint64_t gpu_calls = gpu_gemm_count() - g0;
    measured = mem_use(&m1) && measured;

    const double phys = kb_change(m0.phys, m1.phys);
    const double mall[2] = {kb_change(m0.malloced, mh.malloced), kb_change(mh.malloced, m1.malloced)};
    const char *nano = getenv("MallocNanoZone");
    snprintf(detail, n, "%llu GPU calls after %d warm-up calls: phys_footprint %+.0f KB "
             "(limit %d KB; %.1f MB in all), malloc in use %+.1f then %+.1f KB per half "
             "(limit %d KB in at least one), resident %+.0f KB%s%s%s",
             (unsigned long long)gpu_calls, MEM_WARMUP_CALLS, phys, MEM_FOOTPRINT_LIMIT_KB,
             m1.phys / 1048576.0, mall[0], mall[1], MEM_MALLOC_LIMIT_KB,
             kb_change(m0.resident, m1.resident),
             nano && !strcmp(nano, "0") ? "" : " [not MallocNanoZone=0: inexact malloc counts]",
             measured ? "" : " [task_info failed]",
             not_gpu ? " [a call did not run on the GPU]" : "");
    for (int i = 0; i < N_MEM_SHAPES; i++) {
        free(out[i]);
        call_free(&calls[i]);
    }
    return measured && !not_gpu && gpu_calls == MEM_CALLS && phys < MEM_FOOTPRINT_LIMIT_KB &&
           fmin(mall[0], mall[1]) < MEM_MALLOC_LIMIT_KB;
}

// --- Exponent range -------------------------------------------------------------

// x's sign, with a magnitude in [0.5, 1).
static double half_to_one(double x) { return copysign(0.5 + 0.5 * fabs(x), x); }

// i of n (0 <= i < n) spread evenly over the integers [lo, hi].
static int spread(int i, int n, int lo, int hi) {
    return n > 1 ? lo + (int)lround((double)(hi - lo) * i / (n - 1)) : lo;
}

// RANGE_EXACT's exponents of op(A)'s row i and op(B)'s column j.
static int exact_row_exp(int i) { static const int E[] = {-100, 0, 100}; return E[i % 3]; }
static int exact_col_exp(int j) { static const int E[] = {100, -100, 0}; return E[j % 3]; }

// The factors of pattern d for op(A)_ik (M rows), op(B)_kj (N columns) and C0_ij.
static double range_a(RangeData d, int i, int k, int M) {
    switch (d) {
    case RANGE_TINY:          return 1e-25;
    case RANGE_HUGE:          return 1e25;
    case RANGE_ROWCOL:        return ldexp(1.0, spread(i, M, -120, 120));
    case RANGE_ALIGNED:
    case RANGE_MISALIGNED:    return ldexp(1.0, k % 2 ? -70 : 70);
    case RANGE_EXACT:         return ldexp(1.0, exact_row_exp(i));
    case RANGE_ROWS_LOW:      return ldexp(1.0, spread(i, M, -100, -60));
    case RANGE_COLS_HIGH:     return 0x1p30;
    case RANGE_UNIFORM_EXACT: return 0x1p90;
    case RANGE_C_ZERO_ROW:
    case RANGE_C_TINY:        return i == 0 ? 0.0 : 1.0;
    default:                  return 1.0;
    }
}

static double range_b(RangeData d, int k, int j, int N) {
    switch (d) {
    case RANGE_TINY:          return 1e-25;
    case RANGE_HUGE:          return 1e25;
    case RANGE_ROWCOL:        return ldexp(1.0, spread(j, N, -120, 120));
    case RANGE_ALIGNED:       return ldexp(1.0, k % 2 ? -70 : 70);
    case RANGE_MISALIGNED:    return ldexp(1.0, k % 2 ? 70 : -70);
    case RANGE_EXACT:         return ldexp(1.0, exact_col_exp(j));
    case RANGE_COLS_HIGH:     return ldexp(1.0, spread(j, N, 60, 100));
    case RANGE_UNIFORM_EXACT: return 0x1p90;
    default:                  return 1.0;
    }
}

static double range_c(RangeData d, int i, int j, int M, int N) {
    switch (d) {
    case RANGE_TINY:          return 1e-50;
    case RANGE_HUGE:          return 1e50;
    case RANGE_ALIGNED:       return 0x1p140;
    case RANGE_MISALIGNED:    return 1.0;
    case RANGE_C_ZERO_ROW:    return i == 0 ? 0x1p-100 : 1.0;
    case RANGE_C_TINY:        return i == 0 ? 1.0 : 0x1p-100;
    case RANGE_C_HUGE:        return 0x1p135;
    default:                  return range_a(d, i, 0, M) * range_b(d, 0, j, N);
    }
}

static inline void range_set(double *x, bool base, double f) {
    *x = base ? half_to_one(*x) : *x * f;
}

// Makes the magnitudes of a Call's data [0.5, 1) (base: the M x K, K x N and
// M x N blocks, both parts of complex ones; the padding stays NaN), or else
// multiplies them by pattern d's factors.
static void range_apply(Call *k, RangeData d, bool base) {
    const AbiCase *c = &k->c;
    const int w = k->cplx ? 2 : 1;
    double *A = k->A, *B = k->B, *C0 = k->C0;
    for (int i = 0; i < c->M; i++)
        for (int kk = 0; kk < c->K; kk++)
            for (int p = 0; p < w; p++)
                range_set(&A[op_index(c->ta, k->la, i, kk) * w + p], base, range_a(d, i, kk, c->M));
    for (int kk = 0; kk < c->K; kk++)
        for (int j = 0; j < c->N; j++)
            for (int p = 0; p < w; p++)
                range_set(&B[op_index(c->tb, k->lb, kk, j) * w + p], base, range_b(d, kk, j, c->N));
    for (int j = 0; j < c->N; j++)
        for (int i = 0; i < c->M; i++)
            for (int p = 0; p < w; p++)
                range_set(&C0[((size_t)j * k->lc.ld + i) * w + p], base,
                          range_c(d, i, j, c->M, c->N));
    if (!base && d == RANGE_INF) A[op_index(c->ta, k->la, 0, 0) * w] = INFINITY;
}

// True if x is finite where r is, and else non-finite as r is: the same
// infinity, or NaN.
static bool same_class(double x, double r) {
    return isfinite(r) ? isfinite(x) : isnan(r) ? isnan(x) : x == r;
}

// out's componentwise error against cblas, as blas_error measures it, over
// the entries where cblas's C is finite; *nonfinite_ok: wherever a part of
// cblas's C is not finite, out's is non-finite in the same way.
static double range_error(const Call *k, const void *out, bool *nonfinite_ok) {
    const AbiCase *c = &k->c;
    const int M = c->M, N = c->N, K = c->K;
    const double complex alpha = k->cplx ? c->alpha : creal(c->alpha);
    const double complex beta = k->cplx ? c->beta : creal(c->beta);
    void *Cref = malloc(k->nc * k->es);
    double *absAB = calloc((size_t)M * N, sizeof *absAB);
    memcpy(Cref, k->C0, k->nc * k->es);
    blas_reference(k->cplx, c->ta, c->tb, M, N, K, alpha, beta, k->A, k->la, k->B, k->lb, k->C0,
                   k->lc.ld, Cref, absAB);
    double worst = 0.0;
    *nonfinite_ok = true;
    for (int j = 0; j < N; j++)
        for (int i = 0; i < M; i++) {
            const size_t x = (size_t)j * k->lc.ld + i;
            double complex r, v, c0;
            if (k->cplx) {
                r = ((const double complex *)Cref)[x];
                v = ((const double complex *)out)[x];
                c0 = ((const double complex *)k->C0)[x];
            } else {
                r = ((const double *)Cref)[x];
                v = ((const double *)out)[x];
                c0 = ((const double *)k->C0)[x];
            }
            if (!isfinite(creal(r)) || !isfinite(cimag(r))) {
                *nonfinite_ok = *nonfinite_ok && same_class(creal(v), creal(r)) &&
                                same_class(cimag(v), cimag(r));
                continue;
            }
            double d = DBL_MIN;
            if (alpha != 0.0) d += cabs(alpha) * absAB[(size_t)j * M + i];
            if (beta != 0.0) d += cabs(beta) * cabs(c0);
            double e = cabs(v - r) / d;
            if (isnan(e) || e > worst) worst = e;   // a NaN result stays NaN
        }
    free(Cref);
    free(absAB);
    return worst;
}

// True if out is bit for bit 2^(e_i+f_j) times base in every element of the
// M x N block (RANGE_EXACT's row and column exponents).
static bool exact_multiple(const Call *k, const void *base, const void *out) {
    const AbiCase *c = &k->c;
    const int w = k->cplx ? 2 : 1;
    for (int j = 0; j < c->N; j++)
        for (int i = 0; i < c->M; i++)
            for (int p = 0; p < w; p++) {
                const size_t x = ((size_t)j * k->lc.ld + i) * w + p;
                double want = ldexp(((const double *)base)[x], exact_row_exp(i) + exact_col_exp(j));
                if (memcmp(&want, (const double *)out + x, sizeof want)) return false;
            }
    return true;
}

static bool run_range_blas(const AbiCase *c, char *detail, size_t n) {
    Call k;
    call_init_case(&k, c);
    range_apply(&k, c->data, true);
    const size_t bytes = k.nc * k.es;
    void *C = malloc(bytes), *base = NULL;
    bool base_gpu = true;
    if (c->data == RANGE_EXACT) {               // C of the unscaled data first
        base = malloc(bytes);
        base_gpu = call_run(&k, base);
    }
    range_apply(&k, c->data, false);
    void *A0 = malloc(nelem(k.la) * k.es), *B0 = malloc(nelem(k.lb) * k.es);
    memcpy(A0, k.A, nelem(k.la) * k.es);
    memcpy(B0, k.B, nelem(k.lb) * k.es);

    uint64_t g0 = gpu_gemm_count();
    call_run(&k, C);
    uint64_t gpu_gemms = gpu_gemm_count() - g0;

    bool nonfinite_ok, exact = true;
    double err = range_error(&k, C, &nonfinite_ok);
    if (base) exact = base_gpu && exact_multiple(&k, base, C);
    bool inputs_ok = !memcmp(k.A, A0, nelem(k.la) * k.es) && !memcmp(k.B, B0, nelem(k.lb) * k.es);
    bool outside_ok = outside_block_unchanged(C, k.C0, k.nc, k.lc, k.es);
    bool ok = blas_verdict(c, err, inputs_ok, outside_ok, gpu_gemms, detail, n);
    size_t len = strlen(detail);
    snprintf(detail + len, n - len, "%s%s%s", nonfinite_ok ? "" : " [Inf/NaN differ from cblas]",
             exact ? "" : base_gpu ? " [not 2^(e_i+f_j) times the unscaled C, bitwise]"
                                   : " [unscaled call not on the GPU]",
             err <= TOL ? "" : " [outside tolerance]");
    free(C); free(base); free(A0); free(B0);
    call_free(&k);
    return ok && nonfinite_ok && exact;
}

// The native ABMatrix GEMMs on row-major data shaped as in range_apply, with
// w doubles per entry (2: real and imaginary parts); RANGE_UNIFORM_EXACT also
// runs the unscaled data and compares. C = A*B (ab_dgemm, ab_zgemm) or
// alpha*A*B + beta*C0 (ab_dgemm_scaled).
typedef struct {
    int M, N, K, w;
    double *A, *B, *C0;
} NativeData;

static void native_fill(NativeData *x, RangeData d, bool scale) {
    const int M = x->M, N = x->N, K = x->K, w = x->w;
    for (int i = 0; i < M; i++)
        for (int k = 0; k < K; k++)
            for (int p = 0; p < w; p++)
                x->A[((size_t)i * K + k) * w + p] =
                    half_to_one(rnd()) * (scale ? range_a(d, i, k, M) : 1.0);
    for (int k = 0; k < K; k++)
        for (int j = 0; j < N; j++)
            for (int p = 0; p < w; p++)
                x->B[((size_t)k * N + j) * w + p] =
                    half_to_one(rnd()) * (scale ? range_b(d, k, j, N) : 1.0);
    for (int i = 0; i < M; i++)
        for (int j = 0; j < N; j++)
            for (int p = 0; p < w; p++)
                x->C0[((size_t)i * N + j) * w + p] =
                    half_to_one(rnd()) * (scale ? range_c(d, i, j, M, N) : 1.0);
}

// Uploads count entries of w doubles each into re (and im, for w == 2).
static ABStatus native_upload(ABMatrix re, ABMatrix im, const double *x, size_t count, int w) {
    return w == 2 ? upload_split(re, im, (const double complex *)x, count)
                  : ab_matrix_upload(re, x, false);
}

// Runs the native GEMM of op on x into C (row-major); the GPU must count it.
static ABStatus native_range_call(Op op, double alpha, double beta, const NativeData *x,
                                  double *C, bool *counted) {
    const int M = x->M, N = x->N, K = x->K, w = x->w;
    const size_t nc = (size_t)M * N;
    ABMatrix m[6] = {ab_matrix_create(M, K), ab_matrix_create(M, K), ab_matrix_create(K, N),
                     ab_matrix_create(K, N), ab_matrix_create(M, N), ab_matrix_create(M, N)};
    ABStatus st = AB_OK;
    for (int i = 0; i < 6; i++) if (!m[i]) st = AB_ERROR_ALLOC_FAILED;
    double *re = malloc(nc * sizeof *re), *im = malloc(nc * sizeof *im);
    uint64_t g0 = gpu_gemm_count();
    if (st == AB_OK) st = native_upload(m[0], m[1], x->A, (size_t)M * K, w);
    if (st == AB_OK) st = native_upload(m[2], m[3], x->B, (size_t)K * N, w);
    if (st == AB_OK) st = native_upload(m[4], m[5], x->C0, nc, w);
    if (st == AB_OK)
        st = op == OP_ZGEMM ? ab_zgemm(m[0], m[1], m[2], m[3], m[4], m[5])
           : op == OP_DGEMM ? ab_dgemm(m[0], m[2], m[4])
                            : ab_dgemm_scaled(alpha, m[0], m[2], beta, m[4]);
    *counted = gpu_gemm_count() > g0;
    if (st == AB_OK) st = ab_matrix_download(m[4], re, false);
    if (st == AB_OK && w == 2) st = ab_matrix_download(m[5], im, false);
    for (size_t k = 0; st == AB_OK && k < nc; k++) {
        C[k * w] = re[k];
        if (w == 2) C[k * w + 1] = im[k];
    }
    for (int i = 0; i < 6; i++) ab_matrix_destroy(m[i]);
    free(re); free(im);
    return st;
}

static bool run_range_native(const AbiCase *c, char *detail, size_t n) {
    const int M = c->M, N = c->N, K = c->K, w = c->op == OP_ZGEMM ? 2 : 1;
    const bool scaled = c->op == OP_DGEMM_SCALED;
    const double alpha = scaled ? creal(c->alpha) : 1.0, beta = scaled ? creal(c->beta) : 0.0;
    const size_t nc = (size_t)M * N * w;
    NativeData x = {M, N, K, w, malloc((size_t)M * K * w * sizeof(double)),
                    malloc((size_t)K * N * w * sizeof(double)), malloc(nc * sizeof(double))};
    double *C = malloc(nc * sizeof *C), *base = NULL;
    bool counted = false, base_counted = true, exact = true;
    ABStatus st = ab_init();
    if (c->data == RANGE_UNIFORM_EXACT && st == AB_OK) {
        const uint64_t seed = rng_state;
        native_fill(&x, c->data, false);
        base = malloc(nc * sizeof *base);
        st = native_range_call(c->op, alpha, beta, &x, base, &base_counted);
        rng_state = seed;                           // the same values, scaled
    }
    if (st == AB_OK) {
        native_fill(&x, c->data, true);
        st = native_range_call(c->op, alpha, beta, &x, C, &counted);
    }

    // Reference: row-major M x K, K x N, M x N are column-major K x M, N x K,
    // N x M, so C = A*B is the column-major C^T = B^T A^T.
    double err = NAN;
    if (st == AB_OK) {
        const Layout la = {K, M, K}, lb = {N, K, N}, lc = {N, M, N};
        err = blas_error(w == 2, 'N', 'N', N, M, K, alpha, beta, x.B, lb, x.A, la, x.C0, lc, C);
        for (size_t i = 0; base && exact && i < nc; i++) {
            double want = ldexp(base[i], 180);
            exact = !memcmp(&want, &C[i], sizeof want);
        }
        exact = exact && base_counted;
    }
    snprintf(detail, n, "err=%.1e status=%s%s%s%s", err, ab_status_string(st),
             counted || st != AB_OK ? "" : " [GPU GEMM not counted]",
             exact ? "" : " [not 2^180 times the unscaled C, bitwise]",
             err <= TOL || st != AB_OK ? "" : " [outside tolerance]");
    free(x.A); free(x.B); free(x.C0); free(C); free(base);
    return st == AB_OK && counted && exact && err <= TOL;
}

static bool run_range(const AbiCase *c, char *detail, size_t n) {
    return c->op == OP_DGEMM || c->op == OP_DGEMM_SCALED || c->op == OP_ZGEMM
               ? run_range_native(c, detail, n) : run_range_blas(c, detail, n);
}

// --- Re/Im imbalance ------------------------------------------------------------

// Whether line l of op(A) (rows) or op(B) (columns) gets its real part (else
// its imaginary part) multiplied by IMAG_RATIO.
static bool imag_small_re(ImagData d, int l) {
    return d == IMAG_LARGE || (d == IMAG_LINES && l % 2);
}

static void imag_apply(Call *k, ImagData d) {
    const AbiCase *c = &k->c;
    double *A = k->A, *B = k->B;
    for (int i = 0; i < c->M; i++)
        for (int kk = 0; kk < c->K; kk++)
            A[op_index(c->ta, k->la, i, kk) * 2 + !imag_small_re(d, i)] *= IMAG_RATIO;
    for (int kk = 0; kk < c->K; kk++)
        for (int j = 0; j < c->N; j++)
            B[op_index(c->tb, k->lb, kk, j) * 2 + !imag_small_re(d, j)] *= IMAG_RATIO;
}

static bool run_imag(const AbiCase *c, char *detail, size_t n) {
    const int M = c->M, N = c->N, K = c->K;
    Call k;
    call_init_case(&k, c);
    range_apply(&k, c->data, true);                 // magnitudes [0.5, 1)
    imag_apply(&k, c->imag);
    void *C = malloc(k.nc * k.es);
    uint64_t g0 = gpu_gemm_count();
    bool gpu = call_run(&k, C) && gpu_gemm_count() > g0;

    // R = op(A) op(B) in Dot2, and each part's componentwise scale D:
    // |Ar||Br| + |Ai||Bi| for Re, |Ar||Bi| + |Ai||Br| for Im.
    const size_t na = (size_t)M * K, nb = (size_t)K * N, nr = (size_t)M * N;
    double *Ar = malloc(na * sizeof *Ar), *Ai = malloc(na * sizeof *Ai);
    double *Br = malloc(nb * sizeof *Br), *Bi = malloc(nb * sizeof *Bi);
    double *S[2], *E[2], *D[2];
    for (int p = 0; p < 2; p++) {
        S[p] = calloc(nr, sizeof(double)); E[p] = calloc(nr, sizeof(double));
        D[p] = calloc(nr, sizeof(double));
    }
    pack_rows(k.A, k.la, c->ta, true, M, K, Ar, Ai);
    pack_rows(k.B, k.lb, c->tb, true, K, N, Br, Bi);
    dot2_zgemm_acc(M, N, K, Ar, Ai, Br, Bi, S, E);
    abs_gemm_acc(M, N, K, Ar, NULL, Br, NULL, D[0]);
    abs_gemm_acc(M, N, K, Ai, NULL, Bi, NULL, D[0]);
    abs_gemm_acc(M, N, K, Ar, NULL, Bi, NULL, D[1]);
    abs_gemm_acc(M, N, K, Ai, NULL, Br, NULL, D[1]);
    free(Ar); free(Ai); free(Br); free(Bi);

    double err[2] = {0.0, 0.0};
    for (int j = 0; j < N; j++)
        for (int i = 0; i < M; i++) {
            const double complex v = ((const double complex *)C)[(size_t)j * k.lc.ld + i];
            const double got[2] = {creal(v), cimag(v)};
            for (int p = 0; p < 2; p++) {
                size_t y = (size_t)i * N + j;
                double e = fabs(got[p] - (S[p][y] + E[p][y])) / (D[p][y] + DBL_MIN);
                if (isnan(e) || e > err[p]) err[p] = e;
            }
        }
    bool outside_ok = outside_block_unchanged(C, k.C0, k.nc, k.lc, k.es);
    snprintf(detail, n, "Re err=%.2f u² Im err=%.2f u² (bound %.0f u²) path=%s%s%s", err[0] / U2,
             err[1] / U2, IMAG_BOUND, ab_get_last_dispatch_path(),
             gpu ? "" : " [GPU path not taken]", outside_ok ? "" : " [C outside MxN modified]");

    for (int p = 0; p < 2; p++) { free(S[p]); free(E[p]); free(D[p]); }
    free(C);
    call_free(&k);
    return gpu && outside_ok && err[0] <= IMAG_BOUND * U2 && err[1] <= IMAG_BOUND * U2;
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

// Whether a gpu_time% printed with %.1f is 100 g / t for the times g and t
// printed with %.3f ms: within their rounding, 0.05 for the share and
// 100 d (t + g) / (t (t - d)) for the ratio, d = 0.0005 ms (wide for a
// routine whose calls took under a millisecond in all).
static bool time_share_matches(double pct, double g, double t) {
    const double d = 0.0005;
    return t <= d || fabs(pct - 100.0 * g / t) <= 0.05 + 100.0 * d * (t + g) / (t * (t - d)) + 1e-9;
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
    if (!time_share_matches(r->time_pct, r->gpu_ms, r->total))
        return "AB_REPORT gpu_time% is not gpu_ms / total_ms";
    if (!time_share_matches(r->prof_time_pct, r->prof_gpu_ms, r->prof_total))
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
    char *out = slurp(in);
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
    switch (c->kind) {
    case KIND_LONG_K:  return run_long_k(c, detail, n);
    case KIND_REPEAT:  return run_repeat(c, detail, n);
    case KIND_THREADS: return run_threads(c, detail, n);
    case KIND_MEMORY:  return run_memory(c, detail, n);
    case KIND_RANGE:   return run_range(c, detail, n);
    case KIND_IMAG:    return run_imag(c, detail, n);
    case KIND_CALL:    break;
    }
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
static bool show_passes = false;     // the current suite prints its passing cases too
static FILE *suite_log;              // detail lines, printed after the suite summary
static char self_path[PATH_MAX];

static void record(const AbiCase *c, bool ok, const char *detail) {
    int xi = xfail_lookup(c->name);
    const XfailEntry *x = xi < 0 ? NULL : &XFAIL[xi];
    if (x) xfail_hits[xi]++;
    if (ok && !x) {
        tests_passed++;
        if (verbose || show_passes) fprintf(suite_log, "      PASS  %s  %s\n", c->name, detail);
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

// Child environment: the caller's, minus AB_*, MTL_DEBUG_LAYER and
// MallocNanoZone, plus the suite's.
static char **child_env(const Suite *s, char *mode_buf, size_t mode_len,
                        char *table_buf, size_t table_len) {
    static char debug_layer[] = "MTL_DEBUG_LAYER=1";
    static char profile[] = "AB_PROFILE=1";
    static char nano_off[] = "MallocNanoZone=0";
    size_t k;
    char **env = child_environ(5, &k);
    if (s->exact_malloc) env[k++] = nano_off;
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

    show_passes = s->show;
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
static void test_fork_exit(void)        { run_suite(SU_FORK_EXIT); }
static void test_qe_shapes(void)        { run_suite(SU_QE); }
static void test_long_k_precision(void) { run_suite(SU_LONG_K); }
static void test_determinism(void)      { run_suite(SU_REPEAT); }
static void test_concurrency(void)      { run_suite(SU_THREADS); }
static void test_memory_growth(void)    { run_suite(SU_MEMORY); }
static void test_exponent_range(void)   { run_suite(SU_RANGE); }
static void test_zgemm_imbalance(void)  { run_suite(SU_IMBALANCE); }
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
    {"test_fork_exit",        test_fork_exit},
    {"test_qe_shapes",        test_qe_shapes},
    {"test_long_k_precision", test_long_k_precision},
    {"test_determinism",      test_determinism},
    {"test_concurrency",      test_concurrency},
    {"test_memory_growth",    test_memory_growth},
    {"test_exponent_range",   test_exponent_range},
    {"test_zgemm_imbalance",  test_zgemm_imbalance},
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
    // test_blas_abi [-v] [test_name ...]: with names, only those TESTS[] entries.
    verbose = argc > 1 && !strcmp(argv[1], "-v");
    char **only = argv + 1 + verbose;
    const int n_only = argc - 1 - verbose;

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
        bool selected = n_only == 0;
        for (int o = 0; o < n_only; o++) selected = selected || !strcmp(only[o], TESTS[i].name);
        if (!selected) continue;
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
        bool is_stale = !xfail_hits[i] && n_only == 0;     // a partial run cannot tell
        printf("  %-14s %4d  %s%s\n", XFAIL[i].id, xfail_hits[i], XFAIL[i].pattern,
               is_stale ? "  <-- STALE: matches no case" : "");
        if (is_stale) stale++;
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
