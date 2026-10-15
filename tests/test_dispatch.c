// =============================================================================
// test_dispatch.c — unit tests of the BLAS-ABI routing decision (dispatch.c)
// =============================================================================
//
// Pure: built from src/dispatch.c alone, without the library or a GPU. Checks
// the built-in table against the measurements it was derived from
// (AB_DISPATCH_BUILTIN_DATA, relative to AB_SOURCE_ROOT), the Quantum ESPRESSO
// shapes, the precedence of AB_MODE, AB_MIN_GPU_DIM and
// AB_CROSSOVER_FLOPS[_REAL] over the table, TRANS classes, and the
// AB_DISPATCH_TABLE format and loading.
// =============================================================================

#include "dispatch.h"
#include "dispatch_table.h"
#include "test_registry.h"

#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

#ifndef AB_SOURCE_ROOT
#define AB_SOURCE_ROOT "."
#endif

static int tests_passed = 0;
static int tests_failed = 0;
static char first_failure[512];

#define PASS() do { printf("✓ PASS\n"); tests_passed++; } while (0)
#define FAIL(msg) do { printf("✗ FAIL (%s)\n", msg); tests_failed++; } while (0)

// Records the first failed expectation of the running test.
static bool expect(bool cond, const char *fmt, ...) {
    if (!cond && !first_failure[0]) {
        va_list ap;
        va_start(ap, fmt);
        vsnprintf(first_failure, sizeof first_failure, fmt, ap);
        va_end(ap);
    }
    return cond;
}

static void begin(void) { first_failure[0] = '\0'; }
static void finish(void) { if (first_failure[0]) FAIL(first_failure); else PASS(); }

static const char *const ENV_VARS[] = {
    "AB_MODE", "AB_MIN_GPU_DIM", "AB_CROSSOVER_FLOPS", "AB_CROSSOVER_FLOPS_REAL",
    "AB_DISPATCH_TABLE",
};

static void clear_env(void) {
    for (size_t i = 0; i < TEST_REGISTRY_SIZE(ENV_VARS); i++) unsetenv(ENV_VARS[i]);
}

// What a host gets with none of the routing variables set.
static ab_dispatch_config_t builtin_auto(void) {
    ab_dispatch_config_t c;
    clear_env();
    ab_dispatch_config_from_env(&c);
    return c;
}

static ab_dispatch_config_t with_rows(ab_dispatch_mode_t mode, const char *rows) {
    ab_dispatch_config_t c = {.mode = mode, .min_dim = AB_DISPATCH_DEFAULT_MIN_DIM};
    char buf[512], *save = NULL;
    snprintf(buf, sizeof buf, "%s", rows);
    for (char *line = strtok_r(buf, ";", &save); line; line = strtok_r(NULL, ";", &save))
        if (ab_dispatch_parse_row(line, &c.rows[c.nrows]) == 1) c.nrows++;
    return c;
}

static bool route(const ab_dispatch_config_t *c, const char *call) {
    char r[8], ta, tb;
    int M, N, K;
    if (sscanf(call, "%7s %c %c %d %d %d", r, &ta, &tb, &M, &N, &K) != 6) {
        fprintf(stderr, "bad call spec: %s\n", call);
        abort();
    }
    return ab_dispatch_gpu(c, r[0], ta, tb, M, N, K);
}

// Expects each "<dgemm|zgemm> <ta> <tb> M N K" call to route to want_gpu.
static void expect_routes(const ab_dispatch_config_t *c, bool want_gpu,
                          const char *const *calls, size_t n, const char *what) {
    for (size_t i = 0; i < n; i++)
        expect(route(c, calls[i]) == want_gpu, "%s: %s should run on %s", what, calls[i],
               want_gpu ? "the GPU" : "Accelerate");
}

#define EXPECT_ROUTES(c, gpu, what, ...) do { \
        static const char *const calls_[] = {__VA_ARGS__}; \
        expect_routes(c, gpu, calls_, TEST_REGISTRY_SIZE(calls_), what); \
    } while (0)

static bool write_temp(const char *content, char *path, size_t len) {
    const char *tmp = getenv("TMPDIR");
    snprintf(path, len, "%s/test_dispatch.XXXXXX", tmp && *tmp ? tmp : "/tmp");
    int fd = mkstemp(path);
    if (fd < 0) return false;
    size_t n = strlen(content);
    bool ok = write(fd, content, n) == (ssize_t)n;
    close(fd);
    return ok;
}

// =============================================================================
// The calibration: built-in table and its measurements
// =============================================================================

typedef struct {
    char routine, ta, tb;
    int M, N, K;
    double speedup;              // Accelerate time / GPU time
    int win;
    char auto_route[4];          // what the calibration says auto does
} Measured;

static Measured measured[256];
static int n_measured = -1;

// The calibrate_summary.csv rows behind the built-in table; -1 if unreadable.
static int load_measured(void) {
    if (n_measured >= 0) return n_measured;
    char path[1024], line[512];
    const char *data = AB_DISPATCH_BUILTIN_DATA;
    snprintf(path, sizeof path, "%s%s%s", data[0] == '/' ? "" : AB_SOURCE_ROOT,
             data[0] == '/' ? "" : "/", data);
    FILE *f = fopen(path, "r");
    if (!f) return -1;
    n_measured = 0;
    while (fgets(line, sizeof line, f) && n_measured < (int)TEST_REGISTRY_SIZE(measured)) {
        Measured m;
        char routine[8];
        if (sscanf(line, "%7[^,],%c,%c,%d,%d,%d,%*[^,],%*f,%*f,%*f,%lf,%*f,%*f,%*f,%*d,%d,%3s",
                   routine, &m.ta, &m.tb, &m.M, &m.N, &m.K, &m.speedup, &m.win,
                   m.auto_route) == 9) {
            m.routine = routine[0];
            measured[n_measured++] = m;
        }
    }
    fclose(f);
    return n_measured;
}

static const Measured *find_measured(char routine, char ta, char tb, int M, int N, int K) {
    for (int i = 0; i < n_measured; i++) {
        const Measured *m = &measured[i];
        if (m->routine == routine && m->ta == ta && m->tb == tb && m->M == M && m->N == N &&
            m->K == K)
            return m;
    }
    return NULL;
}

static void test_builtin_table_loads(void) {
    begin();
    ab_dispatch_config_t c = builtin_auto();
    int n = 0;
    while (AB_DISPATCH_BUILTIN[n].routine) n++;
    expect(c.mode == AB_DISPATCH_AUTO, "AB_MODE unset is not auto");
    expect(c.min_dim == AB_DISPATCH_DEFAULT_MIN_DIM, "default AB_MIN_GPU_DIM is %d", c.min_dim);
    expect(!c.crossover_flops && !c.crossover_flops_real, "FLOP overrides set by default");
    expect(c.nrows == n && !memcmp(c.rows, AB_DISPATCH_BUILTIN, (size_t)n * sizeof c.rows[0]),
           "config has %d rows, the built-in table %d", c.nrows, n);
    for (int i = 0; i < n; i++) {
        const ab_dispatch_row_t *r = &AB_DISPATCH_BUILTIN[i];
        expect(strchr("dz", r->routine) && strchr("NT", r->ta) && strchr("NT", r->tb),
               "built-in row %d is not a routine and TRANS classes", i);
    }
    finish();
}

// Every measured shape routes as the calibration recorded, and none that
// was not >= 1.10x faster on the GPU goes there.
static void test_builtin_matches_measurements(void) {
    begin();
    ab_dispatch_config_t c = builtin_auto();
    int n = load_measured();
    expect(n > 0, "cannot read %s/%s", AB_SOURCE_ROOT, AB_DISPATCH_BUILTIN_DATA);
    for (int i = 0; i < n; i++) {
        const Measured *m = &measured[i];
        bool gpu = ab_dispatch_gpu(&c, m->routine, m->ta, m->tb, m->M, m->N, m->K);
        expect(gpu == !strcmp(m->auto_route, "gpu"),
               "%cgemm %c %c %dx%dx%d routes to %s, the calibration says %s", m->routine,
               m->ta, m->tb, m->M, m->N, m->K, gpu ? "gpu" : "cpu", m->auto_route);
        expect(m->win || !gpu, "%cgemm %c %c %dx%dx%d measured %.2fx but routes to the GPU",
               m->routine, m->ta, m->tb, m->M, m->N, m->K, m->speedup);
    }
    finish();
}

// The Quantum ESPRESSO si64 shapes: each must have been measured, and each
// that was not >= 1.10x faster on the GPU must run on Accelerate under auto.
static void test_qe_shapes(void) {
    static const struct { char routine, ta, tb; int M, N, K; } QE[] = {
        // PAW: newq 'T','N' and addusdens 'N','T' (also measured 'N','N');
        // 171 projector pairs, 64 atoms, 2*ngm_l = 71834 (gamma) / 143666 (k).
        {'d', 'T', 'N', 171, 64, 71834},    {'d', 'T', 'N', 171, 64, 143666},
        {'d', 'N', 'N', 71834, 171, 64},    {'d', 'N', 'N', 143666, 171, 64},
        {'d', 'N', 'T', 71834, 171, 64},    {'d', 'N', 'T', 143666, 171, 64},
        // Davidson projections 'C','N': bands x bands x npw.
        {'z', 'C', 'N', 160, 160, 9000},    {'z', 'C', 'N', 320, 320, 9000},
        {'z', 'C', 'N', 500, 500, 9000},    {'z', 'C', 'N', 1000, 1000, 9000},
        {'z', 'C', 'N', 160, 160, 18277},   {'z', 'C', 'N', 320, 320, 18277},
        {'z', 'C', 'N', 500, 500, 18277},   {'z', 'C', 'N', 1000, 1000, 18277},
        // Davidson updates 'N','N': npw x bands x basis.
        {'z', 'N', 'N', 9000, 160, 160},    {'z', 'N', 'N', 9000, 160, 500},
        {'z', 'N', 'N', 9000, 160, 1000},   {'z', 'N', 'N', 9000, 500, 160},
        {'z', 'N', 'N', 9000, 500, 500},    {'z', 'N', 'N', 9000, 500, 1000},
        {'z', 'N', 'N', 18277, 160, 160},   {'z', 'N', 'N', 18277, 160, 500},
        {'z', 'N', 'N', 18277, 160, 1000},  {'z', 'N', 'N', 18277, 500, 160},
        {'z', 'N', 'N', 18277, 500, 500},   {'z', 'N', 'N', 18277, 500, 1000},
        // calbec / add_vuspsi with 1152 projectors (2-rank k-point run).
        {'z', 'C', 'N', 1152, 160, 9000},   {'z', 'N', 'N', 9000, 160, 1152},
    };
    begin();
    ab_dispatch_config_t c = builtin_auto();
    expect(load_measured() > 0, "cannot read the calibration data");
    for (size_t i = 0; i < TEST_REGISTRY_SIZE(QE); i++) {
        const char r = QE[i].routine, ta = QE[i].ta, tb = QE[i].tb;
        const int M = QE[i].M, N = QE[i].N, K = QE[i].K;
        const Measured *m = find_measured(r, ta, tb, M, N, K);
        if (!expect(m != NULL, "QE shape %cgemm %c %c %dx%dx%d was not calibrated", r, ta, tb,
                    M, N, K))
            continue;
        bool gpu = ab_dispatch_gpu(&c, r, ta, tb, M, N, K);
        expect(m->win || !gpu, "QE shape %cgemm %c %c %dx%dx%d measured %.2fx but routes to "
               "the GPU", r, ta, tb, M, N, K, m->speedup);
        // Hosts pass 'T' or 'C' for a transposed complex operand: same class.
        expect(ab_dispatch_gpu(&c, r, ta == 'C' ? 'T' : ta, tb, M, N, K) == gpu,
               "'T' and 'C' route differently for %dx%dx%d", M, N, K);
    }
    finish();
}

// =============================================================================
// Precedence: AB_MODE, AB_MIN_GPU_DIM, FLOP overrides, table
// =============================================================================

static void test_mode_precedence(void) {
    begin();
    const char *all_nn = "zgemm N N 0 0 0; dgemm N N 0 0 0";
    ab_dispatch_config_t cpu = with_rows(AB_DISPATCH_CPU, all_nn);
    ab_dispatch_config_t gpu = with_rows(AB_DISPATCH_GPU, "");
    ab_dispatch_config_t aut = with_rows(AB_DISPATCH_AUTO, all_nn);
    EXPECT_ROUTES(&cpu, false, "AB_MODE=cpu", "zgemm N N 4096 4096 4096", "dgemm N N 64 64 64");
    EXPECT_ROUTES(&gpu, true, "AB_MODE=gpu", "zgemm N N 33 33 33", "dgemm C T 33 4096 33",
                  "dgemm T N 171 64 143666");
    EXPECT_ROUTES(&gpu, false, "AB_MODE=gpu, a dimension <= AB_MIN_GPU_DIM",
                  "zgemm N N 32 4096 4096", "dgemm N N 4096 32 4096", "dgemm N N 4096 4096 32");
    EXPECT_ROUTES(&aut, true, "auto, table row", "zgemm N N 33 33 33", "dgemm n n 64 64 64");
    EXPECT_ROUTES(&aut, false, "auto, no row for the class or below the floor",
                  "zgemm C N 4096 4096 4096", "dgemm N T 4096 4096 4096",
                  "zgemm N N 32 4096 4096", "dgemm N N 4096 4096 16");
    gpu.min_dim = aut.min_dim = 0;
    EXPECT_ROUTES(&gpu, true, "AB_MIN_GPU_DIM=0", "zgemm N N 1 1 1");
    EXPECT_ROUTES(&aut, true, "AB_MIN_GPU_DIM=0", "dgemm N N 1 1 1");
    finish();
}

static void test_row_bounds_and_classes(void) {
    begin();
    ab_dispatch_config_t c = with_rows(AB_DISPATCH_AUTO, "zgemm C N 1000 1000 9000");
    EXPECT_ROUTES(&c, true, "at or above the row's minimums",
                  "zgemm C N 1000 1000 9000", "zgemm T N 1000 1000 9000",
                  "zgemm c n 1000 1000 9000", "zgemm t N 5000 1000 9000",
                  "zgemm C N 1000 1000 100000");
    EXPECT_ROUTES(&c, false, "below a minimum, or another routine or class",
                  "zgemm C N 999 1000 9000", "zgemm C N 1000 999 9000",
                  "zgemm C N 1000 1000 8999", "dgemm T N 1000 1000 9000",
                  "zgemm N N 1000 1000 9000", "zgemm C C 1000 1000 9000",
                  "zgemm C T 1000 1000 9000");
    finish();
}

static void test_flop_overrides(void) {
    begin();
    ab_dispatch_config_t c = with_rows(AB_DISPATCH_AUTO, "");
    c.crossover_flops = 100000000;              // AB_CROSSOVER_FLOPS=1e8: both routines
    EXPECT_ROUTES(&c, true, "AB_CROSSOVER_FLOPS=1e8",
                  "zgemm C N 150 150 2000",      // 8*M*N*K = 3.6e8
                  "dgemm T N 200 200 2000");     // 2*M*N*K = 1.6e8
    EXPECT_ROUTES(&c, false, "AB_CROSSOVER_FLOPS=1e8",
                  "zgemm N N 100 100 100", "dgemm T N 150 150 2000",   // 9e7
                  "zgemm N N 16 4096 4096");     // AB_MIN_GPU_DIM still applies
    c.crossover_flops_real = 10000000;          // AB_CROSSOVER_FLOPS_REAL=1e7: DGEMM only
    EXPECT_ROUTES(&c, true, "AB_CROSSOVER_FLOPS_REAL=1e7", "dgemm T N 150 150 2000");
    EXPECT_ROUTES(&c, false, "AB_CROSSOVER_FLOPS_REAL=1e7", "zgemm N N 100 100 100");

    c = with_rows(AB_DISPATCH_AUTO, "zgemm N N 0 0 0");
    c.crossover_flops = 1000000000000ULL;       // replaces the table
    EXPECT_ROUTES(&c, false, "AB_CROSSOVER_FLOPS over the table", "zgemm N N 64 64 64");
    c.crossover_flops = UINT64_MAX;             // 8*M*N*K ~ 8e28 must not wrap
    EXPECT_ROUTES(&c, true, "FLOPs beyond 2^64", "zgemm N N 2147483647 2147483647 2147483647");
    c = with_rows(AB_DISPATCH_AUTO, "");
    c.crossover_flops_real = 1;                 // DGEMM override alone leaves ZGEMM on the table
    EXPECT_ROUTES(&c, false, "AB_CROSSOVER_FLOPS_REAL alone", "zgemm N N 4096 4096 4096");
    EXPECT_ROUTES(&c, true, "AB_CROSSOVER_FLOPS_REAL alone", "dgemm N N 64 64 64");
    finish();
}

// =============================================================================
// AB_DISPATCH_TABLE format, loading, environment
// =============================================================================

static void test_parse_row(void) {
    static const char *const BAD[] = {
        "sgemm N N 1 1 1", "zgemm X N 1 1 1", "zgemm NN N 1 1 1", "zgemm N N 1 1",
        "zgemm N N 1 1 1 1", "zgemm N N -1 1 1", "zgemm N N 1x 1 1", "zgemm N N 1 1 3000000000",
        "zgemm,N,N,1,1,1",
    };
    begin();
    ab_dispatch_row_t r;
    expect(ab_dispatch_parse_row("zgemm C N 1000 1000 9000\n", &r) == 1 && r.routine == 'z' &&
           r.ta == 'T' && r.tb == 'N' && r.m == 1000 && r.n == 1000 && r.k == 9000,
           "zgemm C N 1000 1000 9000");
    expect(ab_dispatch_parse_row("  DGEMM\tn t 0 0 0   # comment", &r) == 1 && r.routine == 'd' &&
           r.ta == 'N' && r.tb == 'T' && !r.m && !r.n && !r.k, "DGEMM n t 0 0 0 # comment");
    expect(ab_dispatch_parse_row("", &r) == 0, "empty line");
    expect(ab_dispatch_parse_row("  \t\r\n", &r) == 0, "blank line");
    expect(ab_dispatch_parse_row("# zgemm N N 1 1 1", &r) == 0, "comment line");
    for (size_t i = 0; i < TEST_REGISTRY_SIZE(BAD); i++)
        expect(ab_dispatch_parse_row(BAD[i], &r) == -1, "'%s' should be malformed", BAD[i]);
    finish();
}

static void test_load_table(void) {
    char path[1024], err[160], longline[400];
    begin();
    ab_dispatch_config_t c = with_rows(AB_DISPATCH_AUTO, "dgemm T T 7 7 7");

    expect(write_temp("# header\nzgemm C N 1000 1000 9000\n\ndgemm N N 4096 4096 4096 # x",
                      path, sizeof path), "mkstemp");
    expect(ab_dispatch_load_table(&c, path, err, sizeof err) && c.nrows == 2 &&
           c.rows[0].routine == 'z' && c.rows[1].k == 4096, "valid file: %s", err);
    unlink(path);

    expect(write_temp("zgemm N N 1 1 1\ndgemm N N 1 1 1\nzgemm N Q 1 1 1\n", path, sizeof path),
           "mkstemp");
    err[0] = '\0';
    expect(!ab_dispatch_load_table(&c, path, err, sizeof err) && strstr(err, "line 3") &&
           c.nrows == 2 && c.rows[0].routine == 'z' && c.rows[0].m == 1000,
           "a malformed line must fail with its number and leave the config alone: %s", err);
    unlink(path);

    memset(longline, 'x', sizeof longline - 2);
    longline[sizeof longline - 2] = '\n';
    longline[sizeof longline - 1] = '\0';
    expect(write_temp(longline, path, sizeof path), "mkstemp");
    expect(!ab_dispatch_load_table(&c, path, err, sizeof err), "an over-long line must fail");
    unlink(path);
    char long_comment[sizeof longline + 32];      // e.g. a long "# Data: <path>" line
    longline[0] = '#';
    snprintf(long_comment, sizeof long_comment, "%szgemm T T 4096 4096 4096\n", longline);
    expect(write_temp(long_comment, path, sizeof path), "mkstemp");
    expect(ab_dispatch_load_table(&c, path, err, sizeof err) && c.nrows == 1 &&
           c.rows[0].k == 4096, "an over-long comment line must be skipped: %s", err);
    unlink(path);

    char many[AB_DISPATCH_MAX_ROWS * 24 + 32] = "";
    for (int i = 0; i <= AB_DISPATCH_MAX_ROWS; i++) strcat(many, "zgemm N N 64 64 64\n");
    expect(write_temp(many, path, sizeof path), "mkstemp");
    expect(!ab_dispatch_load_table(&c, path, err, sizeof err) && strstr(err, "too many rows"),
           "more than %d rows must fail: %s", AB_DISPATCH_MAX_ROWS, err);
    unlink(path);

    expect(!ab_dispatch_load_table(&c, "/nonexistent/dispatch_table.txt", err, sizeof err),
           "a missing file must fail");

    expect(write_temp("# no rows: everything on Accelerate\n", path, sizeof path), "mkstemp");
    expect(ab_dispatch_load_table(&c, path, err, sizeof err) && c.nrows == 0,
           "a file without rows is valid and empties the table");
    EXPECT_ROUTES(&c, false, "empty table", "zgemm N N 8192 8192 8192");
    unlink(path);
    finish();
}

static void test_config_from_env(void) {
    char path[1024];
    ab_dispatch_config_t c;
    begin();
    static const struct { const char *value; ab_dispatch_mode_t mode; } MODES[] = {
        {"cpu", AB_DISPATCH_CPU}, {"CPU", AB_DISPATCH_CPU}, {"gpu", AB_DISPATCH_GPU},
        {"Gpu", AB_DISPATCH_GPU}, {"auto", AB_DISPATCH_AUTO}, {"", AB_DISPATCH_AUTO},
        {"amx", AB_DISPATCH_AUTO},
    };
    for (size_t i = 0; i < TEST_REGISTRY_SIZE(MODES); i++) {
        clear_env();
        setenv("AB_MODE", MODES[i].value, 1);
        ab_dispatch_config_from_env(&c);
        expect(c.mode == MODES[i].mode, "AB_MODE=%s", MODES[i].value);
    }
    static const struct { const char *value; int min_dim; } DIMS[] = {
        {"0", 0}, {"100", 100}, {"-5", AB_DISPATCH_DEFAULT_MIN_DIM},
        {"", AB_DISPATCH_DEFAULT_MIN_DIM},
    };
    for (size_t i = 0; i < TEST_REGISTRY_SIZE(DIMS); i++) {
        clear_env();
        setenv("AB_MIN_GPU_DIM", DIMS[i].value, 1);
        ab_dispatch_config_from_env(&c);
        expect(c.min_dim == DIMS[i].min_dim, "AB_MIN_GPU_DIM=%s gave %d", DIMS[i].value,
               c.min_dim);
    }
    clear_env();
    setenv("AB_CROSSOVER_FLOPS", "100000000", 1);
    setenv("AB_CROSSOVER_FLOPS_REAL", "", 1);
    ab_dispatch_config_from_env(&c);
    expect(c.crossover_flops == 100000000 && c.crossover_flops_real == 0,
           "AB_CROSSOVER_FLOPS=100000000, AB_CROSSOVER_FLOPS_REAL empty");

    expect(write_temp("dgemm T N 1 2 3\n", path, sizeof path), "mkstemp");
    clear_env();
    setenv("AB_DISPATCH_TABLE", path, 1);
    ab_dispatch_config_from_env(&c);
    expect(c.nrows == 1 && c.rows[0].routine == 'd' && c.rows[0].k == 3,
           "AB_DISPATCH_TABLE replaces the built-in rows");
    unlink(path);

    ab_dispatch_config_t builtin = builtin_auto();
    setenv("AB_DISPATCH_TABLE", "/nonexistent/dispatch_table.txt", 1);
    fprintf(stderr, "(expected warning follows) ");
    ab_dispatch_config_from_env(&c);
    expect(c.nrows == builtin.nrows &&
           !memcmp(c.rows, builtin.rows, (size_t)c.nrows * sizeof c.rows[0]),
           "an unreadable AB_DISPATCH_TABLE must keep the built-in rows");
    clear_env();
    finish();
}

// =============================================================================
// Main
// =============================================================================

static const TestCase TESTS[] = {
    {"test_builtin_table_loads",          test_builtin_table_loads},
    {"test_builtin_matches_measurements", test_builtin_matches_measurements},
    {"test_qe_shapes",                    test_qe_shapes},
    {"test_mode_precedence",              test_mode_precedence},
    {"test_row_bounds_and_classes",       test_row_bounds_and_classes},
    {"test_flop_overrides",               test_flop_overrides},
    {"test_parse_row",                    test_parse_row},
    {"test_load_table",                   test_load_table},
    {"test_config_from_env",              test_config_from_env},
};
static const int N_TESTS = (int)TEST_REGISTRY_SIZE(TESTS);

int main(void) {
    printf("╔══════════════════════════════════════════════════════════════════╗\n");
    printf("║  apple-bottom BLAS-ABI Routing (dispatch.c, no GPU)              ║\n");
    printf("╚══════════════════════════════════════════════════════════════════╝\n");

    for (int i = 0; i < N_TESTS; i++) {
        printf("[%d/%d] %-34s ... ", i + 1, N_TESTS, TESTS[i].name);
        fflush(stdout);
        int pre = tests_passed + tests_failed;
        TESTS[i].fn();
        int post = tests_passed + tests_failed;
        if (post == pre) {
            fprintf(stderr, "\nFATAL: ghost test — '%s' reported no result\n", TESTS[i].name);
            abort();
        }
    }

    printf("\n═══════════════════════════════════════════════════════════════════\n");
    printf("Results: %d passed, %d failed\n", tests_passed, tests_failed);
    if (tests_failed == 0) {
        printf("✓ All %d tests PASSED (%d calibrated shapes checked)\n", tests_passed, n_measured);
        return 0;
    }
    printf("✗ Some tests FAILED\n");
    return 1;
}
