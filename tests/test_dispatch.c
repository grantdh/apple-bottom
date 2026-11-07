// =============================================================================
// test_dispatch.c — unit tests of the BLAS-ABI routing decision (dispatch.c)
// =============================================================================
//
// Pure: built from src/dispatch.c alone, without the library or a GPU. Checks
// every built-in table against the measurements it was derived from (its
// `data`, relative to AB_SOURCE_ROOT), the Quantum ESPRESSO shapes, the
// choice of table by chip and GPU core count (AB_CHIP, AB_GPU_CORES), the
// precedence of AB_MODE, AB_MIN_GPU_DIM and AB_CROSSOVER_FLOPS[_REAL] over the
// table, TRANS classes, and the AB_DISPATCH_TABLE format, header keys and
// loading. The host this runs on may have no table (a CI runner): nothing
// here assumes one.
//
// AB_SPLIT: the split table's row format and file loading, the plan (window
// edges, exact TRANS, scalar classes, GFLOP and ratio limits, the code's K
// limits, first match, the cut and its floors, overflow-safe huge shapes),
// its place ahead of AB_MODE=gpu and the FLOP overrides and behind AB_MODE=cpu
// and AB_MIN_GPU_DIM, the choice of split table by chip, the slices of each
// axis, and every built-in split table against its filed evidence (the rules
// of scripts/ingest_split.py, also checked here on tables made to break them).
// =============================================================================

#include "dispatch.h"
#include "dispatch_table.h"
#include "split_slice.h"
#include "split_table.h"
#include "test_registry.h"

#include <Accelerate/Accelerate.h>
#include <complex.h>
#include <limits.h>
#include <math.h>
#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <strings.h>
#include <sys/sysctl.h>
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
    "AB_MODE", "AB_MIN_GPU_DIM", "AB_BLAS2_MIN_DIM", "AB_CROSSOVER_FLOPS", "AB_CROSSOVER_FLOPS_REAL",
    "AB_DISPATCH_TABLE", "AB_CHIP", "AB_GPU_CORES", "AB_SPLIT", "AB_SPLIT_TABLE",
};

static void clear_env(void) {
    for (size_t i = 0; i < TEST_REGISTRY_SIZE(ENV_VARS); i++) unsetenv(ENV_VARS[i]);
}

// What this host gets with none of the routing variables set.
static ab_dispatch_config_t builtin_auto(void) {
    ab_dispatch_config_t c;
    clear_env();
    ab_dispatch_config_from_env(&c);
    return c;
}

// What a machine with table t's chip gets with none of the routing variables set.
static ab_dispatch_config_t auto_for(const ab_dispatch_table_t *t) {
    ab_dispatch_config_t c;
    char cores[16];
    clear_env();
    setenv("AB_CHIP", t->chip, 1);
    snprintf(cores, sizeof cores, "%d", t->gpu_cores);
    setenv("AB_GPU_CORES", cores, 1);
    ab_dispatch_config_from_env(&c);
    clear_env();
    return c;
}

static int n_tables(void) {
    int n = 0;
    while (AB_DISPATCH_BUILTIN_TABLES[n].chip) n++;
    return n;
}

static int n_rows(const ab_dispatch_row_t *rows) {
    int n = 0;
    while (rows[n].routine) n++;
    return n;
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
// The calibrations: built-in tables and their measurements
// =============================================================================

typedef struct {
    char routine, ta, tb;
    int M, N, K;
    double speedup;              // Accelerate time / GPU time
    int win;
    char auto_route[4];          // what the calibration says auto does
} Measured;

static Measured measured[256];
static int n_measured;
static int shapes_checked;       // over every table, for the summary line

// The calibrate_summary.csv rows behind table t; -1 if unreadable.
static int load_measured(const ab_dispatch_table_t *t) {
    char path[1024], line[512];
    const char *data = t->data ? t->data : "";
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

// Every built-in table names a chip, a matrix unit and its data, has valid
// rows, and is what a machine with that chip gets by default.
static void test_builtin_tables_load(void) {
    begin();
    int n = n_tables();
    expect(n >= 1, "no built-in table at all (benchmarks/calibration/ empty?)");
    for (int i = 0; i < n; i++) {
        const ab_dispatch_table_t *t = &AB_DISPATCH_BUILTIN_TABLES[i];
        expect(t->chip && !strncmp(t->chip, "Apple ", 6), "table %d: chip '%s' is not an Apple chip", i,
               t->chip ? t->chip : "(null)");
        expect(t->gpu_cores >= 0, "table %d: negative core count", i);
        expect(t->cpu_matrix && (!strcmp(t->cpu_matrix, "AMX") || !strcmp(t->cpu_matrix, "SME")),
               "table %d: cpu_matrix '%s' is not AMX or SME", i, t->cpu_matrix ? t->cpu_matrix : "(null)");
        expect(t->data && strstr(t->data, "calibrate_summary.csv"), "table %d names no summary CSV", i);
        for (const ab_dispatch_row_t *r = t->rows; r->routine; r++)
            expect(strchr("dz", r->routine) && strchr("NT", r->ta) && strchr("NT", r->tb) &&
                   r->m >= 0 && r->n >= 0 && r->k >= 0, "table %d: a row is not a routine, TRANS classes "
                   "and minimums", i);
        for (int j = 0; j < i; j++)
            expect(strcasecmp(t->chip, AB_DISPATCH_BUILTIN_TABLES[j].chip) ||
                   t->gpu_cores != AB_DISPATCH_BUILTIN_TABLES[j].gpu_cores,
                   "tables %d and %d are both %s/%d", j, i, t->chip, t->gpu_cores);
        ab_dispatch_config_t c = auto_for(t);
        char slug[AB_DISPATCH_CHIP_LEN], label[AB_DISPATCH_LABEL_LEN];
        ab_dispatch_chip_slug(t->chip, t->gpu_cores, slug, sizeof slug);
        snprintf(label, sizeof label, "builtin:%s", slug);
        expect(c.mode == AB_DISPATCH_AUTO && c.min_dim == AB_DISPATCH_DEFAULT_MIN_DIM &&
               !c.crossover_flops && !c.crossover_flops_real, "%s: defaults are not auto/32/no override", slug);
        expect(c.nrows == n_rows(t->rows) && !memcmp(c.rows, t->rows, (size_t)c.nrows * sizeof c.rows[0]),
               "%s: a machine with this chip gets %d rows, the table has %d", slug, c.nrows, n_rows(t->rows));
        expect(!strcmp(c.table, label) && !strcmp(c.table_chip, t->chip) && c.table_gpu_cores == t->gpu_cores,
               "%s: table label '%s', chip '%s'/%d", slug, c.table, c.table_chip, c.table_gpu_cores);
    }
    finish();
}

// Every measured shape routes as its calibration recorded, and none that was
// not >= 1.10x faster on the GPU goes there, for every built-in table.
static void test_builtin_matches_measurements(void) {
    begin();
    shapes_checked = 0;
    for (int ti = 0; ti < n_tables(); ti++) {
        const ab_dispatch_table_t *t = &AB_DISPATCH_BUILTIN_TABLES[ti];
        ab_dispatch_config_t c = auto_for(t);
        int n = load_measured(t);
        if (!expect(n > 0, "cannot read %s/%s", AB_SOURCE_ROOT, t->data)) continue;
        shapes_checked += n;
        for (int i = 0; i < n; i++) {
            const Measured *m = &measured[i];
            bool gpu = ab_dispatch_gpu(&c, m->routine, m->ta, m->tb, m->M, m->N, m->K);
            expect(gpu == !strcmp(m->auto_route, "gpu"),
                   "%s: %cgemm %c %c %dx%dx%d routes to %s, the calibration says %s", t->chip,
                   m->routine, m->ta, m->tb, m->M, m->N, m->K, gpu ? "gpu" : "cpu", m->auto_route);
            expect(m->win || !gpu, "%s: %cgemm %c %c %dx%dx%d measured %.2fx but routes to the GPU",
                   t->chip, m->routine, m->ta, m->tb, m->M, m->N, m->K, m->speedup);
        }
    }
    finish();
}

// The Quantum ESPRESSO si64 shapes: each must have been measured for every
// table, and each that was not >= 1.10x faster on the GPU must run on
// Accelerate under auto.
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
    for (int ti = 0; ti < n_tables(); ti++) {
        const ab_dispatch_table_t *t = &AB_DISPATCH_BUILTIN_TABLES[ti];
        ab_dispatch_config_t c = auto_for(t);
        if (!expect(load_measured(t) > 0, "cannot read the calibration data of %s", t->chip)) continue;
        for (size_t i = 0; i < TEST_REGISTRY_SIZE(QE); i++) {
            const char r = QE[i].routine, ta = QE[i].ta, tb = QE[i].tb;
            const int M = QE[i].M, N = QE[i].N, K = QE[i].K;
            const Measured *m = find_measured(r, ta, tb, M, N, K);
            if (!expect(m != NULL, "%s: QE shape %cgemm %c %c %dx%dx%d was not calibrated", t->chip, r,
                        ta, tb, M, N, K))
                continue;
            bool gpu = ab_dispatch_gpu(&c, r, ta, tb, M, N, K);
            expect(m->win || !gpu, "%s: QE shape %cgemm %c %c %dx%dx%d measured %.2fx but routes to "
                   "the GPU", t->chip, r, ta, tb, M, N, K, m->speedup);
            // Hosts pass 'T' or 'C' for a transposed complex operand: same class.
            expect(ab_dispatch_gpu(&c, r, ta == 'C' ? 'T' : ta, tb, M, N, K) == gpu,
                   "%s: 'T' and 'C' route differently for %dx%dx%d", t->chip, M, N, K);
        }
    }
    finish();
}

// =============================================================================
// Choosing the table: chip, core count, slug
// =============================================================================

static void test_select_table(void) {
    static const ab_dispatch_row_t R1[] = {{'z', 'T', 'T', 4096, 4096, 4096}, {0, 0, 0, 0, 0, 0}};
    static const ab_dispatch_row_t R0[] = {{0, 0, 0, 0, 0, 0}};
    static const ab_dispatch_table_t T[] = {
        {"Apple M2 Max", 38, "AMX", "", R1},
        {"Apple M2 Max", 30, "AMX", "", R0},
        {"Apple M4 Pro", 0, "SME", "", R0},        // calibrated without a core count: every variant
        {"Apple M3", 10, "AMX", "", R1},
        {NULL, 0, NULL, NULL, NULL},
    };
    begin();
    expect(ab_dispatch_select_table(T, "Apple M2 Max", 38) == &T[0], "M2 Max/38 -> its table");
    expect(ab_dispatch_select_table(T, "Apple M2 Max", 30) == &T[1], "M2 Max/30 -> its table");
    expect(ab_dispatch_select_table(T, "apple m2 max", 38) == &T[0], "chip names compare case-insensitively");
    expect(ab_dispatch_select_table(T, "Apple M2 Max", 40) == NULL, "M2 Max/40: a sibling's table is not used");
    expect(ab_dispatch_select_table(T, "Apple M2 Max", 0) == NULL, "M2 Max, unknown cores, two tables: none");
    expect(ab_dispatch_select_table(T, "Apple M4 Pro", 16) == &T[2], "M4 Pro/16 -> the any-core table");
    expect(ab_dispatch_select_table(T, "Apple M4 Pro", 0) == &T[2], "M4 Pro, unknown cores -> the any-core table");
    expect(ab_dispatch_select_table(T, "Apple M3", 0) == &T[3], "M3, unknown cores, one table -> it");
    expect(ab_dispatch_select_table(T, "Apple M3", 8) == NULL, "M3/8: the 10-core table is not used");
    expect(ab_dispatch_select_table(T, "Apple M5 Max", 40) == NULL, "an uncalibrated chip gets no table");
    expect(ab_dispatch_select_table(T, "", 38) == NULL && ab_dispatch_select_table(T, NULL, 38) == NULL,
           "an unknown chip gets no table");
    // The built-in set is a static array in a header, so this file holds its own copy:
    // compare the chosen table's identity, not its address.
    for (int i = 0; i < n_tables(); i++) {
        const ab_dispatch_table_t *w = &AB_DISPATCH_BUILTIN_TABLES[i];
        const ab_dispatch_table_t *g = ab_dispatch_builtin_table(w->chip, w->gpu_cores ? w->gpu_cores : 1);
        expect(g && !strcasecmp(g->chip, w->chip) && g->gpu_cores == w->gpu_cores,
               "ab_dispatch_builtin_table(%s, %d) is not that table", w->chip, w->gpu_cores);
    }
    expect(ab_dispatch_builtin_table("Apple M0 Nano", 1) == NULL, "the built-in set has no M0 Nano");
    finish();
}

static void test_host_chip_and_slug(void) {
    static const struct { const char *chip; int cores; const char *slug; } SLUGS[] = {
        {"Apple M2 Max", 38, "M2-Max-38c"}, {"Apple M4 Pro", 0, "M4-Pro"}, {"Apple M5 Max", 40, "M5-Max-40c"},
        {"apple M1", 8, "M1-8c"}, {"Apple M5 Max (Virtual)", 10, "M5-Max-Virtual-10c"}, {"", 0, "unknown"},
        {"", 38, "unknown"}, {NULL, 0, "unknown"}, {"Apple ", 4, "unknown"},
    };
    char chip[AB_DISPATCH_CHIP_LEN], slug[AB_DISPATCH_CHIP_LEN], brand[256] = "";
    int cores = -1;
    begin();
    for (size_t i = 0; i < TEST_REGISTRY_SIZE(SLUGS); i++) {
        ab_dispatch_chip_slug(SLUGS[i].chip, SLUGS[i].cores, slug, sizeof slug);
        expect(!strcmp(slug, SLUGS[i].slug), "slug of '%s'/%d is '%s', expected '%s'",
               SLUGS[i].chip ? SLUGS[i].chip : "(null)", SLUGS[i].cores, slug, SLUGS[i].slug);
    }
    ab_dispatch_chip_slug("Apple M2 Max", 38, slug, 6);              // a short buffer truncates, no overrun
    expect(strlen(slug) < 6, "slug truncation");

    clear_env();
    setenv("AB_CHIP", "  Apple M9 Ultra  ", 1);
    setenv("AB_GPU_CORES", "80", 1);
    ab_dispatch_host_chip(chip, sizeof chip, &cores);
    expect(!strcmp(chip, "Apple M9 Ultra") && cores == 80, "AB_CHIP/AB_GPU_CORES override: '%s'/%d", chip, cores);
    setenv("AB_GPU_CORES", "-3", 1);
    ab_dispatch_host_chip(chip, sizeof chip, &cores);
    expect(cores == 0, "AB_GPU_CORES=-3 reads as unknown, got %d", cores);
    setenv("AB_GPU_CORES", "many", 1);
    ab_dispatch_host_chip(chip, sizeof chip, &cores);
    expect(cores == 0, "AB_GPU_CORES=many reads as unknown, got %d", cores);

    clear_env();
    ab_dispatch_host_chip(chip, sizeof chip, &cores);
    size_t n = sizeof brand;
    if (sysctlbyname("machdep.cpu.brand_string", brand, &n, NULL, 0) != 0) brand[0] = '\0';
    expect(!strcmp(chip, brand), "host chip '%s' is not machdep.cpu.brand_string '%s'", chip, brand);
    expect(cores >= 0, "host core count %d", cores);
    // The IOKit count must be what `ioreg -rc AGXAccelerator -d1` lists (the Makefile's
    // GPU_CORES); a host without an AGXAccelerator (a CI runner) lists nothing.
    int ioreg_cores = -1;
    FILE *io = popen("/usr/sbin/ioreg -rc AGXAccelerator -d1 2>/dev/null", "r");
    if (io) {
        char line[4096];
        while (fgets(line, sizeof line, io)) {
            const char *p = strstr(line, "\"gpu-core-count\"");
            if (p && (p = strchr(p, '=')) && ioreg_cores < 0) ioreg_cores = atoi(p + 1);
        }
        pclose(io);
    }
    expect(ioreg_cores < 0 ? cores == 0 : cores == ioreg_cores,
           "IOKit reports %d GPU cores, ioreg lists %d", cores, ioreg_cores);
    // Whatever this host is, the default configuration is its table or none.
    ab_dispatch_config_t c = builtin_auto();
    const ab_dispatch_table_t *t = ab_dispatch_builtin_table(chip, cores);
    expect(!strcmp(c.chip, chip) && c.gpu_cores == cores, "config records the host as '%s'/%d", c.chip, c.gpu_cores);
    if (t)
        expect(c.nrows == n_rows(t->rows) && !strncmp(c.table, "builtin:", 8),
               "host %s/%d has a table but the config has %d rows (%s)", chip, cores, c.nrows, c.table);
    else
        expect(c.nrows == 0 && !strncmp(c.table, "none:", 5),
               "host %s/%d has no table but the config has %d rows (%s)", chip, cores, c.nrows, c.table);
    finish();
}

// An uncalibrated chip keeps every auto call on Accelerate; AB_MODE=gpu and
// the FLOP overrides do not depend on a table.
static void test_uncalibrated_chip(void) {
    ab_dispatch_config_t c;
    begin();
    clear_env();
    setenv("AB_CHIP", "Apple M9 Ultra", 1);
    setenv("AB_GPU_CORES", "99", 1);
    ab_dispatch_config_from_env(&c);
    expect(c.nrows == 0 && !strcmp(c.table, "none:uncalibrated-chip") && !c.table_chip[0] && !c.table_gpu_cores,
           "uncalibrated chip: %d rows, table '%s'", c.nrows, c.table);
    EXPECT_ROUTES(&c, false, "uncalibrated chip, auto", "zgemm T T 8192 8192 8192", "dgemm N N 6144 6144 6144");
    setenv("AB_MODE", "gpu", 1);
    ab_dispatch_config_from_env(&c);
    EXPECT_ROUTES(&c, true, "uncalibrated chip, AB_MODE=gpu", "zgemm T T 8192 8192 8192");
    setenv("AB_MODE", "auto", 1);
    setenv("AB_CROSSOVER_FLOPS", "1000", 1);
    ab_dispatch_config_from_env(&c);
    EXPECT_ROUTES(&c, true, "uncalibrated chip, AB_CROSSOVER_FLOPS", "zgemm N N 64 64 64");
    ab_dispatch_config_t host = builtin_auto();
    setenv("AB_CHIP", "", 1);                      // empty: as unset
    setenv("AB_GPU_CORES", "", 1);
    ab_dispatch_config_from_env(&c);
    expect(!strcmp(c.chip, host.chip) && c.gpu_cores == host.gpu_cores && !strcmp(c.table, host.table),
           "AB_CHIP= must read the host chip '%s' (%s), got '%s' (%s)", host.chip, host.table, c.chip, c.table);
    clear_env();
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
// AB_DISPATCH_TABLE format, header keys, loading, environment
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
    expect(ab_dispatch_parse_row("# chip: Apple M2 Max", &r) == 0, "a header key is a comment to the row parser");
    for (size_t i = 0; i < TEST_REGISTRY_SIZE(BAD); i++)
        expect(ab_dispatch_parse_row(BAD[i], &r) == -1, "'%s' should be malformed", BAD[i]);
    finish();
}

static void test_parse_header(void) {
    char key[32], value[64];
    begin();
    expect(ab_dispatch_parse_header("# chip: Apple M2 Max\n", key, sizeof key, value, sizeof value) == 1 &&
           !strcmp(key, "chip") && !strcmp(value, "Apple M2 Max"), "'# chip: Apple M2 Max' -> '%s'='%s'", key, value);
    expect(ab_dispatch_parse_header("#gpu_cores:38", key, sizeof key, value, sizeof value) == 1 &&
           !strcmp(key, "gpu_cores") && !strcmp(value, "38"), "'#gpu_cores:38' -> '%s'='%s'", key, value);
    expect(ab_dispatch_parse_header("  # cpu_matrix:  AMX   # inferred\r\n", key, sizeof key, value, sizeof value) == 1 &&
           !strcmp(key, "cpu_matrix") && !strcmp(value, "AMX"), "a trailing note is not the value: '%s'", value);
    expect(ab_dispatch_parse_header("# Machine:  Apple M2 Max, 38-core GPU", key, sizeof key, value, sizeof value) == 0,
           "'# Machine:' (capitalized prose) is not a key");
    expect(ab_dispatch_parse_header("# Accelerate time / GPU time per class (a win is >= 1.10x):", key, sizeof key,
                                    value, sizeof value) == 0, "prose ending in a colon is not a key");
    expect(ab_dispatch_parse_header("zgemm T T 4096 4096 4096", key, sizeof key, value, sizeof value) == 0,
           "a row is not a header");
    expect(ab_dispatch_parse_header("#", key, sizeof key, value, sizeof value) == 0 &&
           ab_dispatch_parse_header("# :x", key, sizeof key, value, sizeof value) == 0, "empty comment, empty key");
    expect(ab_dispatch_parse_header("# chip: Apple M2 Max with a very long name indeed", key, sizeof key, value, 8) == 1 &&
           strlen(value) == 7, "a long value is truncated to the buffer");
    finish();
}

static void test_load_table(void) {
    char path[1024], err[160], longline[400];
    begin();
    ab_dispatch_config_t c = with_rows(AB_DISPATCH_AUTO, "dgemm T T 7 7 7");
    snprintf(c.table_chip, sizeof c.table_chip, "stale");
    c.table_gpu_cores = 99;

    expect(write_temp("# header\nzgemm C N 1000 1000 9000\n\ndgemm N N 4096 4096 4096 # x",
                      path, sizeof path), "mkstemp");
    expect(ab_dispatch_load_table(&c, path, err, sizeof err) && c.nrows == 2 &&
           c.rows[0].routine == 'z' && c.rows[1].k == 4096, "valid file: %s", err);
    expect(!c.table_chip[0] && !c.table_gpu_cores, "a file without identity keys clears the table's chip");
    unlink(path);

    expect(write_temp("# apple-bottom table\n# chip: Apple M5 Max\n# gpu_cores: 40\n# cpu_matrix: SME\n"
                      "# Machine:  Apple M5 Max, 40-core GPU\nzgemm T T 2048 2048 2048\n", path, sizeof path), "mkstemp");
    expect(ab_dispatch_load_table(&c, path, err, sizeof err) && c.nrows == 1 && c.rows[0].m == 2048 &&
           !strcmp(c.table_chip, "Apple M5 Max") && c.table_gpu_cores == 40,
           "identity keys are read: '%s'/%d (%s)", c.table_chip, c.table_gpu_cores, err);
    unlink(path);

    expect(write_temp("zgemm N N 1 1 1\ndgemm N N 1 1 1\nzgemm N Q 1 1 1\n", path, sizeof path),
           "mkstemp");
    err[0] = '\0';
    expect(!ab_dispatch_load_table(&c, path, err, sizeof err) && strstr(err, "line 3") &&
           c.nrows == 1 && c.rows[0].routine == 'z' && c.rows[0].m == 2048 &&
           !strcmp(c.table_chip, "Apple M5 Max"),
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
    static const struct { const char *value; int min_dim, blas2_min_dim; } DIMS[] = {
        {"0", 0, 0}, {"100", 100, 100},
        {"-5", AB_DISPATCH_DEFAULT_MIN_DIM, AB_DISPATCH_DEFAULT_BLAS2_MIN_DIM},
        {"", AB_DISPATCH_DEFAULT_MIN_DIM, AB_DISPATCH_DEFAULT_BLAS2_MIN_DIM},
    };
    for (size_t i = 0; i < TEST_REGISTRY_SIZE(DIMS); i++) {
        clear_env();
        setenv("AB_MIN_GPU_DIM", DIMS[i].value, 1);
        setenv("AB_BLAS2_MIN_DIM", DIMS[i].value, 1);
        ab_dispatch_config_from_env(&c);
        expect(c.min_dim == DIMS[i].min_dim, "AB_MIN_GPU_DIM=%s gave %d", DIMS[i].value,
               c.min_dim);
        expect(c.blas2_min_dim == DIMS[i].blas2_min_dim, "AB_BLAS2_MIN_DIM=%s gave %d",
               DIMS[i].value, c.blas2_min_dim);
    }
    clear_env();
    ab_dispatch_config_from_env(&c);
    expect(c.blas2_min_dim == AB_DISPATCH_DEFAULT_BLAS2_MIN_DIM, "AB_BLAS2_MIN_DIM unset gave %d",
           c.blas2_min_dim);
    clear_env();
    setenv("AB_CROSSOVER_FLOPS", "100000000", 1);
    setenv("AB_CROSSOVER_FLOPS_REAL", "", 1);
    ab_dispatch_config_from_env(&c);
    expect(c.crossover_flops == 100000000 && c.crossover_flops_real == 0,
           "AB_CROSSOVER_FLOPS=100000000, AB_CROSSOVER_FLOPS_REAL empty");

    // A table file replaces the built-in rows, whatever chip this is.
    expect(write_temp("# chip: Apple M9 Ultra\n# gpu_cores: 80\ndgemm T N 1 2 3\n", path, sizeof path), "mkstemp");
    clear_env();
    setenv("AB_DISPATCH_TABLE", path, 1);
    ab_dispatch_config_from_env(&c);
    char want[AB_DISPATCH_LABEL_LEN];
    snprintf(want, sizeof want, "file:%s", path);
    expect(c.nrows == 1 && c.rows[0].routine == 'd' && c.rows[0].k == 3 && !strcmp(c.table, want) &&
           !strcmp(c.table_chip, "Apple M9 Ultra") && c.table_gpu_cores == 80,
           "AB_DISPATCH_TABLE replaces the built-in rows (%d rows, %s)", c.nrows, c.table);
    unlink(path);

    ab_dispatch_config_t builtin = builtin_auto();
    setenv("AB_DISPATCH_TABLE", "/nonexistent/dispatch_table.txt", 1);
    fprintf(stderr, "(expected warning follows) ");
    ab_dispatch_config_from_env(&c);
    expect(c.nrows == builtin.nrows &&
           !memcmp(c.rows, builtin.rows, (size_t)c.nrows * sizeof c.rows[0]) && !strcmp(c.table, builtin.table),
           "an unreadable AB_DISPATCH_TABLE must keep the built-in rows and label");
    clear_env();
    finish();
}

// =============================================================================
// AB_SPLIT: rows, plan, cut and slices, tables and their evidence
// =============================================================================

// The provisional M2 Max rows of benchmarks/split/start_table.txt.
static const char *const START_ROWS =
    "zgemm N N N m=2048: n=2048: k=2048: ratio=2 gf=0.5 scalars=z f=0.60;"   // 0 squares
    "dgemm N N N m=2048: n=2048: k=2048: ratio=2 gf=0.5 scalars=z f=0.55;"   // 1 squares
    "zgemm N N N m=8192: n=256:  k=33:           gf=0.5 scalars=u f=0.45;"   // 2 updates
    "zgemm N N M m=8192: n=33:255 k=33:          gf=0.5 scalars=u f=0.52;"   // 3 narrow updates
    "zgemm C N K m=33:   n=33:   k=8192:         gf=0.5 scalars=z f=0.51";   // 4 reductions

static ab_dispatch_config_t with_split_rows(ab_dispatch_mode_t mode, const char *rows) {
    ab_dispatch_config_t c = {.mode = mode, .min_dim = AB_DISPATCH_DEFAULT_MIN_DIM, .split_on = true};
    char buf[1024], *save = NULL;
    snprintf(buf, sizeof buf, "%s", rows);
    for (char *line = strtok_r(buf, ";", &save); line; line = strtok_r(NULL, ";", &save)) {
        int got = ab_split_parse_row(line, &c.split_rows[c.split_nrows]);
        if (got < 0) {
            fprintf(stderr, "bad split row: %s\n", line);
            abort();
        }
        c.split_nrows += got;
    }
    return c;
}

// The plan of "<dgemm|zgemm> <ta> <tb> M N K <scalars>", scalars one of a1b0
// (alpha 1, beta 0), a1b1, a1bx (alpha 1, another beta), axb0, axbx, nan (a
// scalar not finite): its row, or -1 with no plan.
static int split_plan(const ab_dispatch_config_t *c, const char *call, ab_split_plan_t *p) {
    char r[8], ta, tb, sc[8];
    int M, N, K;
    if (sscanf(call, "%7s %c %c %d %d %d %7s", r, &ta, &tb, &M, &N, &K, sc) != 7) {
        fprintf(stderr, "bad call spec: %s\n", call);
        abort();
    }
    bool a1 = sc[1] == '1', b0 = !strcmp(sc + 2, "b0"), b1 = !strcmp(sc + 2, "b1");
    bool finite = strcmp(sc, "nan") != 0;
    if (!finite) a1 = b0 = b1 = false;
    *p = (ab_split_plan_t){-1, 0, 0, 0};
    return ab_dispatch_split_plan(c, r[0], ta, tb, M, N, K, a1, b0, b1, finite, p) ? p->row : -1;
}

// Expects the call's plan: row (-1: none), and for a row its axis, L and g.
static void expect_plan(const ab_dispatch_config_t *c, const char *call, int row, char axis, int L,
                        int g) {
    ab_split_plan_t p;
    int got = split_plan(c, call, &p);
    if (row < 0)
        expect(got < 0, "%s: planned on row %d, expected no row", call, got);
    else
        expect(got == row && p.axis == axis && p.L == L && p.g == g,
               "%s: row %d axis %c L %d g %d, expected row %d axis %c L %d g %d", call, got,
               p.axis ? p.axis : '-', p.L, p.g, row, axis, L, g);
}

static void test_split_parse_row(void) {
    static const char *const BAD[] = {
        "zgemm N N N", "zgemm N N N m=33: n=33:", "zgemm N N X f=0.5", "zgemm N N N m=1 f=0.5",
        "zgemm N N N m=5:3 f=0.5", "zgemm N N N m=-1: f=0.5", "zgemm N N N f=1.5",
        "zgemm N N N f=-0.1", "zgemm N N N f=nan", "zgemm N N N f=inf", "zgemm N N N gf=1 m=1:",
        "zgemm N N N f=0.5 f=0.6", "zgemm N N N q=1: f=0.5", "zgemm N N N scalars=x f=0.5",
        "zgemm N N N scalars=uz f=0.5", "zgemm N N N ratio=0.5 f=0.5", "sgemm N N N f=0.5",
        "zgemm NN N N f=0.5", "zgemm N N N =1 f=0.5", "zgemm N N N m=1:2:3 f=0.5",
        "zgemm N N N k=3000000000: f=0.5", "zgemm N Q N f=0.5", "zgemm N N NK f=0.5",
        "zgemm N N N mn=9:1 f=0.5", "zgemm N N N gf=-1 f=0.5", "zgemm,N,N,N,f=0.5",
        "zgemm N N N m=1: m=2: n=1: k=1: mn=1: ratio=2 gf=1 scalars=u f=0.5 x=1",
    };
    begin();
    ab_split_row_t r;
    expect(ab_split_parse_row("zgemm N N M m=8192: n=33:255 k=33: gf=0.5 scalars=u f=0.52  # x\n", &r) == 1 &&
           r.routine == 'z' && r.ta == 'N' && r.tb == 'N' && r.axis == 'M' && r.min_m == 8192 &&
           !r.max_m && r.min_n == 33 && r.max_n == 255 && r.min_k == 33 && !r.max_k && !r.min_mn &&
           !r.max_mn && !r.ratio && r.min_gf == 0.5 && r.scalars == 'u' && r.share == 0.52,
           "the narrow-updates row");
    expect(ab_split_parse_row("zgemm c n k f=0", &r) == 1 && r.ta == 'C' && r.tb == 'N' && r.axis == 'K' &&
           r.scalars == 'a' && r.share == 0.0 && !r.min_gf && !r.ratio, "zgemm c n k f=0 (defaults)");
    expect(ab_split_parse_row("DGEMM C t m f=1", &r) == 1 && r.routine == 'd' && r.ta == 'T' &&
           r.tb == 'T' && r.axis == 'M' && r.share == 1.0, "a DGEMM 'C' is 'T'");
    expect(ab_split_parse_row("zgemm T C N mn=1e6:2.5e7 ratio=2 m=:4096 f=.5", &r) == 1 && r.ta == 'T' &&
           r.tb == 'C' && r.min_mn == 1e6 && r.max_mn == 2.5e7 && r.ratio == 2 && !r.min_m &&
           r.max_m == 4096 && r.share == 0.5, "mn, ratio, an upper bound alone");
    expect(ab_split_parse_row("zgemm N N N m=: f=0.5", &r) == 1 && !r.min_m && !r.max_m, "m=: is any M");
    expect(ab_split_parse_row("", &r) == 0 && ab_split_parse_row(" \t\r\n", &r) == 0 &&
           ab_split_parse_row("# zgemm N N N f=0.5", &r) == 0 &&
           ab_split_parse_row("# chip: Apple M2 Max", &r) == 0, "blank and comment lines");
    for (size_t i = 0; i < TEST_REGISTRY_SIZE(BAD); i++)
        expect(ab_split_parse_row(BAD[i], &r) == -1, "'%s' should be malformed", BAD[i]);
    // Each provisional row parses.
    ab_dispatch_config_t c = with_split_rows(AB_DISPATCH_AUTO, START_ROWS);
    expect(c.split_nrows == 5 && c.split_rows[4].axis == 'K' && c.split_rows[4].ta == 'C' &&
           c.split_rows[1].routine == 'd' && c.split_rows[0].ratio == 2, "the start table's rows");
    finish();
}

// Windows and their edges, exact TRANS, scalar classes, GFLOP and ratio,
// the code's K limits and first match, on the provisional rows.
static void test_split_plan_rows(void) {
    begin();
    ab_dispatch_config_t c = with_split_rows(AB_DISPATCH_AUTO, START_ROWS);
    // updates: N axis, g = round(0.45 L / 32) 32
    expect_plan(&c, "zgemm N N 9000 256 160 a1b0", 2, 'N', 256, 128);
    expect_plan(&c, "zgemm N N 8192 500 33 a1b1", 2, 'N', 500, 224);
    expect_plan(&c, "zgemm n n 18277 1000 1000 a1b0", 2, 'N', 1000, 448);
    expect_plan(&c, "zgemm N N 8191 500 160 a1b0", -1, 0, 0, 0);           // below m=8192:
    expect_plan(&c, "zgemm N N 9000 500 32 a1b0", -1, 0, 0, 0);            // below k=33: (and the floor)
    // narrow updates: M axis below N = 256
    expect_plan(&c, "zgemm N N 9000 255 160 a1b0", 3, 'M', 9000, 4672);
    expect_plan(&c, "zgemm N N 9000 33 1000 a1b1", 3, 'M', 9000, 4672);
    expect_plan(&c, "zgemm N N 4096 1024 200 a1b0", -1, 0, 0, 0);          // never the M axis
    // reductions: K axis, exact TRANS
    expect_plan(&c, "zgemm C N 500 500 18277 a1b0", 4, 'K', 18277, 9344);
    expect_plan(&c, "zgemm c N 500 500 18277 a1b0", 4, 'K', 18277, 9344);
    expect_plan(&c, "zgemm T N 500 500 18277 a1b0", -1, 0, 0, 0);          // 'T' is not 'C' here
    expect_plan(&c, "zgemm C C 500 500 18277 a1b0", -1, 0, 0, 0);
    expect_plan(&c, "zgemm C N 500 500 8191 a1b0", -1, 0, 0, 0);           // below k=8192:
    expect_plan(&c, "zgemm C N 1200 1200 8192 a1b0", -1, 0, 0, 0);         // K < 7 max(M, N)
    // scalar classes: 'z' alpha 1, beta 0; 'u' beta 0 or 1
    expect_plan(&c, "zgemm C N 500 500 18277 a1b1", -1, 0, 0, 0);
    expect_plan(&c, "zgemm C N 500 500 18277 axb0", -1, 0, 0, 0);
    expect_plan(&c, "zgemm N N 9000 256 160 a1bx", -1, 0, 0, 0);
    expect_plan(&c, "zgemm N N 9000 256 160 axbx", -1, 0, 0, 0);
    expect_plan(&c, "zgemm N N 9000 256 160 nan", -1, 0, 0, 0);
    // squares: ratio 2, K <= M on the N axis, DGEMM by its own row
    expect_plan(&c, "zgemm N N 2048 2048 2048 a1b0", 0, 'N', 2048, 1216);
    expect_plan(&c, "zgemm N N 2048 4096 2048 a1b0", 0, 'N', 4096, 2464);
    expect_plan(&c, "zgemm N N 2048 4097 2048 a1b0", -1, 0, 0, 0);         // ratio > 2
    expect_plan(&c, "zgemm N N 2048 2048 4096 a1b0", -1, 0, 0, 0);         // K > M
    expect_plan(&c, "zgemm N N 2047 2048 2047 a1b0", -1, 0, 0, 0);         // below m=2048:
    expect_plan(&c, "dgemm N N 3000 3000 3000 a1b0", 1, 'N', 3000, 1664);
    expect_plan(&c, "dgemm N T 3000 3000 3000 a1b0", -1, 0, 0, 0);
    expect_plan(&c, "dgemm N N 3000 3000 3000 a1b1", -1, 0, 0, 0);

    // GFLOP: 8 M N K (complex), 2 M N K (real) against gf
    c = with_split_rows(AB_DISPATCH_AUTO, "zgemm N N N gf=1 f=0.5; dgemm N N N gf=1 f=0.5");
    expect_plan(&c, "zgemm N N 500 500 500 axbx", 0, 'N', 500, 256);
    expect_plan(&c, "zgemm N N 499 500 500 axbx", -1, 0, 0, 0);
    expect_plan(&c, "dgemm N N 794 794 794 axbx", 1, 'N', 794, 384);       // 1.0011 GF
    expect_plan(&c, "dgemm N N 793 794 794 axbx", -1, 0, 0, 0);
    // First match wins; DGEMM 'C' matches a 'T' row; upper bounds and mn.
    c = with_split_rows(AB_DISPATCH_AUTO, "dgemm T N K f=0.25; dgemm T N M f=0.75; "
                                          "zgemm N N N m=:999 n=100:200 mn=:150000 f=0.5");
    expect_plan(&c, "dgemm C N 100 100 7000 a1b0", 0, 'K', 7000, 1728);
    expect_plan(&c, "dgemm t n 1000 1000 6999 a1b0", -1, 0, 0, 0);         // K < 7 max; row 1 needs K <= M
    expect_plan(&c, "dgemm T N 1000 100 100 a1b0", 1, 'M', 1000, 736);
    expect_plan(&c, "zgemm N N 999 150 150 a1b0", 2, 'N', 150, 64);
    expect_plan(&c, "zgemm N N 1000 150 150 a1b0", -1, 0, 0, 0);
    expect_plan(&c, "zgemm N N 999 201 150 a1b0", -1, 0, 0, 0);
    expect_plan(&c, "zgemm N N 800 190 150 a1b0", -1, 0, 0, 0);           // M N = 152000
    // f = 0: the call is attributed to its row, g = 0; too short to cut: g = 0
    c = with_split_rows(AB_DISPATCH_AUTO, "zgemm N N N f=0; zgemm N N M f=0.5");
    expect_plan(&c, "zgemm N N 5000 5000 500 a1b0", 0, 'N', 5000, 0);
    c = with_split_rows(AB_DISPATCH_AUTO, "zgemm N N N f=0.5");
    expect_plan(&c, "zgemm N N 1000 63 100 a1b0", 0, 'N', 63, 0);
    finish();
}

static void test_split_cut(void) {
    static const struct { char axis; int L; double f; int min_dim, g; } CUTS[] = {
        {'N', 63, 0.5, 0, 0},       {'N', 64, 0.5, 0, 32},      {'N', 64, 0.5, 32, 0},
        {'N', 1000, 0.0, 32, 0},    {'N', 1000, -0.5, 32, 0},   {'N', 1000, 1.0, 32, 968},
        {'N', 1000, 0.999, 0, 968}, {'N', 1000, 0.001, 0, 32},  {'N', 1000, 0.001, 32, 64},
        {'M', 9000, 0.52, 32, 4672}, {'M', 9000, 0.52, 5000, 5024}, {'M', 9000, 0.1, 8980, 0},
        {'K', 127, 0.5, 0, 0},      {'K', 128, 0.5, 0, 64},     {'K', 128, 0.5, 64, 0},
        {'K', 18277, 0.51, 32, 9344}, {'K', 18277, 0.99, 32, 18112}, {'K', 200, 0.5, 70, 128},
        {'N', 256, 0.45, 32, 128},  {'N', 256, 0.45, 128, 160}, {'N', 96, 0.5, 32, 64},
        {'N', INT_MAX, 0.5, 32, 1073741824}, {'K', INT_MAX, 1.0, 32, INT_MAX - 64},
    };
    begin();
    for (size_t i = 0; i < TEST_REGISTRY_SIZE(CUTS); i++) {
        int g = ab_split_cut(CUTS[i].axis, CUTS[i].L, CUTS[i].f, CUTS[i].min_dim);
        expect(g == CUTS[i].g, "cut %c L=%d f=%g floor %d: g=%d, expected %d", CUTS[i].axis, CUTS[i].L,
               CUTS[i].f, CUTS[i].min_dim, g, CUTS[i].g);
        // Every cut leaves both parts at least one unit, and the GPU's above the floor.
        int a = ab_split_align(CUTS[i].axis);
        expect(g == 0 || (g >= a && CUTS[i].L - g >= a && g > CUTS[i].min_dim),
               "cut %c L=%d: g=%d leaves a part below %d or g at the floor", CUTS[i].axis, CUTS[i].L, g, a);
    }
    // The plan applies the configured floor to g.
    ab_dispatch_config_t c = with_split_rows(AB_DISPATCH_AUTO, "zgemm N N N f=0.1");
    c.min_dim = 500;
    expect_plan(&c, "zgemm N N 1000 1000 600 a1b0", 0, 'N', 1000, 512);
    finish();
}

// The slices of each axis and TRANS: shapes and offsets, as split_slice.h
// documents them, also where an offset exceeds 2^31 elements.
static void test_split_parts(void) {
    begin();
    ab_split_part_t c, g;
    ab_split_parts('N', 96, 'N', 'N', 500, 256, 160, 600, 170, 510, &c, &g);
    expect(c.M == 500 && c.N == 160 && c.K == 160 && !c.a && !c.b && !c.c && !c.into_p &&
           g.M == 500 && g.N == 96 && g.K == 160 && !g.a && g.b == 160u * 170 && g.c == 160u * 510 &&
           !g.into_p, "N axis, tb N");
    ab_split_parts('N', 96, 'C', 't', 500, 256, 160, 600, 300, 510, &c, &g);
    expect(g.N == 96 && !g.a && g.b == 160 && g.c == 160u * 510, "N axis, tb T: B from row cut");
    ab_split_parts('M', 4672, 'n', 'N', 9000, 200, 160, 9100, 170, 9050, &c, &g);
    expect(c.M == 4328 && g.M == 4672 && g.N == 200 && g.a == 4328 && !g.b && g.c == 4328,
           "M axis, ta N");
    ab_split_parts('M', 4672, 'C', 'N', 9000, 200, 160, 170, 170, 9050, &c, &g);
    expect(g.a == 4328u * 170 && !g.b && g.c == 4328, "M axis, ta C: A from column cut");
    ab_split_parts('K', 9344, 'C', 'N', 500, 500, 18277, 18300, 18290, 510, &c, &g);
    expect(c.K == 8933 && g.K == 9344 && c.M == 500 && g.N == 500 && g.a == 8933 && g.b == 8933 &&
           !g.c && g.into_p && !c.into_p, "K axis, ta C, tb N");
    ab_split_parts('K', 9344, 'N', 'T', 500, 500, 18277, 510, 520, 510, &c, &g);
    expect(g.a == 8933u * 510 && g.b == 8933u * 520 && g.into_p, "K axis, ta N, tb T");
    ab_split_parts('N', 64, 'N', 'N', 64, INT_MAX, 64, 64, INT_MAX, INT_MAX, &c, &g);
    size_t cut = (size_t)INT_MAX - 64;
    expect(g.b == cut * INT_MAX && g.c == cut * INT_MAX, "offsets beyond 2^31 elements");
    finish();
}

// AB_MODE=cpu and AB_MIN_GPU_DIM come first; a covered call is planned under
// gpu and whatever the FLOP overrides say.
static void test_split_precedence(void) {
    begin();
    ab_dispatch_config_t c = with_split_rows(AB_DISPATCH_CPU, START_ROWS);
    expect_plan(&c, "zgemm N N 9000 256 160 a1b0", -1, 0, 0, 0);
    c = with_split_rows(AB_DISPATCH_GPU, START_ROWS);
    expect_plan(&c, "zgemm N N 9000 256 160 a1b0", 2, 'N', 256, 128);
    c = with_split_rows(AB_DISPATCH_AUTO, START_ROWS);
    c.crossover_flops = UINT64_MAX;            // would keep every call on Accelerate
    expect_plan(&c, "zgemm N N 9000 256 160 a1b0", 2, 'N', 256, 128);
    c.crossover_flops = 1;                     // would send every call to the GPU
    expect_plan(&c, "zgemm C N 500 500 18277 a1b0", 4, 'K', 18277, 9344);
    c = with_split_rows(AB_DISPATCH_AUTO, "zgemm N N N f=0.5");
    expect_plan(&c, "zgemm N N 1000 1000 32 a1b0", -1, 0, 0, 0);    // a dimension at the floor
    expect_plan(&c, "zgemm N N 1000 32 1000 a1b0", -1, 0, 0, 0);
    c.min_dim = 0;
    expect_plan(&c, "zgemm N N 1000 1000 32 a1b0", 0, 'N', 1000, 512);
    c.split_nrows = 0;                         // no rows: no plan
    expect_plan(&c, "zgemm N N 1000 1000 1000 a1b0", -1, 0, 0, 0);
    // Huge shapes: GFLOP, ratio and the K limit in double, no overflow.
    c = with_split_rows(AB_DISPATCH_AUTO, "zgemm N N N gf=1e9 f=0.5; zgemm C N K f=0.5; "
                                          "dgemm N N N ratio=2 f=0.5");
    expect_plan(&c, "zgemm N N 2147483647 2147483647 2147483647 a1b0", 0, 'N', INT_MAX, 1073741824);
    expect_plan(&c, "zgemm C N 1000 1000 2147483647 a1b0", 1, 'K', INT_MAX, 1073741824);
    expect_plan(&c, "zgemm C N 2147483647 2147483647 2147483647 a1b0", -1, 0, 0, 0);
    expect_plan(&c, "dgemm N N 2147483647 64 64 a1b0", -1, 0, 0, 0);
    finish();
}

static void test_split_select_table(void) {
    static const ab_split_row_t R1[] = {{'z', 'N', 'N', 'N', 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 'u', 0.5},
                                        {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0}};
    static const ab_split_table_t T[] = {
        {"Apple M2 Max", 38, "AMX", "abc1234", "", R1},
        {"Apple M2 Max", 30, "AMX", "abc1234", "", R1},
        {"Apple M4 Pro", 0, "SME", "abc1234", "", R1},
        {"Apple M3", 10, "AMX", "abc1234", "", R1},
        {NULL, 0, NULL, NULL, NULL, NULL},
    };
    begin();
    expect(ab_split_select_table(T, "Apple M2 Max", 38) == &T[0], "M2 Max/38 -> its table");
    expect(ab_split_select_table(T, "apple m2 max", 30) == &T[1], "M2 Max/30 -> its table");
    expect(ab_split_select_table(T, "Apple M2 Max", 40) == NULL, "M2 Max/40: a sibling's table is not used");
    expect(ab_split_select_table(T, "Apple M2 Max", 0) == NULL, "M2 Max, unknown cores, two tables: none");
    expect(ab_split_select_table(T, "Apple M4 Pro", 16) == &T[2], "M4 Pro/16 -> the any-core table");
    expect(ab_split_select_table(T, "Apple M3", 0) == &T[3], "M3, unknown cores, one table -> it");
    expect(ab_split_select_table(T, "Apple M3", 8) == NULL, "M3/8: the 10-core table is not used");
    expect(ab_split_select_table(T, "", 38) == NULL && ab_split_select_table(T, NULL, 38) == NULL,
           "an unknown chip gets no table");
    for (int i = 0; AB_SPLIT_BUILTIN_TABLES[i].chip; i++) {
        const ab_split_table_t *w = &AB_SPLIT_BUILTIN_TABLES[i];
        const ab_split_table_t *g = ab_split_builtin_table(w->chip, w->gpu_cores);
        expect(g && !strcasecmp(g->chip, w->chip) && g->gpu_cores == w->gpu_cores,
               "ab_split_builtin_table(%s, %d) is not that table", w->chip, w->gpu_cores);
    }
    expect(ab_split_builtin_table("Apple M0 Nano", 1) == NULL, "the built-in split set has no M0 Nano");
    finish();
}

static void test_split_load_table(void) {
    char path[1024], err[256];
    begin();
    ab_dispatch_config_t c = with_split_rows(AB_DISPATCH_AUTO, "dgemm T T K f=0.5");
    snprintf(c.split_table_chip, sizeof c.split_table_chip, "stale");
    c.split_table_gpu_cores = 99;
    expect(write_temp("# AB_SPLIT table\n# chip: Apple M5 Max\n# gpu_cores: 40\n# library: abc1234\n"
                      "zgemm N N N m=8192: f=0.45  # updates\n\nzgemm C N K f=0.51\n", path, sizeof path),
           "mkstemp");
    expect(ab_split_load_table(&c, path, err, sizeof err) && c.split_nrows == 2 &&
           c.split_rows[0].min_m == 8192 && c.split_rows[1].axis == 'K' &&
           !strcmp(c.split_table_chip, "Apple M5 Max") && c.split_table_gpu_cores == 40,
           "valid file: %d rows, '%s'/%d (%s)", c.split_nrows, c.split_table_chip, c.split_table_gpu_cores, err);
    unlink(path);

    expect(write_temp("zgemm N N N f=0.5\nzgemm N N N f=0.5 f=0.4\n", path, sizeof path), "mkstemp");
    err[0] = '\0';
    expect(!ab_split_load_table(&c, path, err, sizeof err) && strstr(err, "line 2") && c.split_nrows == 2 &&
           c.split_rows[0].min_m == 8192 && !strcmp(c.split_table_chip, "Apple M5 Max"),
           "a malformed line must fail with its number and leave the config alone: %s", err);
    unlink(path);
    // A dispatch-table row is not a split row.
    expect(write_temp("zgemm T T 4096 4096 4096\n", path, sizeof path), "mkstemp");
    expect(!ab_split_load_table(&c, path, err, sizeof err), "a routing-table row must be refused");
    unlink(path);

    char many[(AB_SPLIT_MAX_ROWS + 1) * 24 + 32] = "";
    for (int i = 0; i <= AB_SPLIT_MAX_ROWS; i++) strcat(many, "zgemm N N N f=0.5\n");
    expect(write_temp(many, path, sizeof path), "mkstemp");
    expect(!ab_split_load_table(&c, path, err, sizeof err) && strstr(err, "too many rows"),
           "more than %d rows must fail: %s", AB_SPLIT_MAX_ROWS, err);
    unlink(path);

    char longline[700], doc[800];
    memset(longline, 'x', sizeof longline - 2);
    longline[0] = '#';
    longline[sizeof longline - 2] = '\n';
    longline[sizeof longline - 1] = '\0';
    snprintf(doc, sizeof doc, "%szgemm C N K f=0.5\n", longline);
    expect(write_temp(doc, path, sizeof path), "mkstemp");
    expect(ab_split_load_table(&c, path, err, sizeof err) && c.split_nrows == 1 && !c.split_table_chip[0] &&
           !c.split_table_gpu_cores, "an over-long comment is skipped, no identity keys clear the chip: %s", err);
    unlink(path);
    longline[0] = 'z';
    expect(write_temp(longline, path, sizeof path), "mkstemp");
    expect(!ab_split_load_table(&c, path, err, sizeof err), "an over-long row must fail");
    unlink(path);

    expect(!ab_split_load_table(&c, "/nonexistent/split_table.txt", err, sizeof err), "a missing file must fail");
    expect(write_temp("# no rows: nothing is split\n", path, sizeof path), "mkstemp");
    expect(ab_split_load_table(&c, path, err, sizeof err) && c.split_nrows == 0, "a file without rows is valid");
    unlink(path);
    // The provisional table loads with its five rows.
    expect(ab_split_load_table(&c, AB_SOURCE_ROOT "/benchmarks/split/start_table.txt", err, sizeof err) &&
           c.split_nrows == 5 && !strcmp(c.split_table_chip, "Apple M2 Max") && c.split_table_gpu_cores == 38,
           "benchmarks/split/start_table.txt: %s", err);
    finish();
}

// AB_SPLIT and AB_SPLIT_TABLE in ab_dispatch_config_from_env.
static void test_split_config_from_env(void) {
    static const struct { const char *value; bool on; } VALUES[] = {
        {NULL, false}, {"", false}, {"0", false}, {"00", false}, {"0yes", false},
        {"1", true}, {"yes", true}, {"on", true},
    };
    char path[1024], want[AB_DISPATCH_LABEL_LEN];
    ab_dispatch_config_t c;
    begin();
    for (size_t i = 0; i < TEST_REGISTRY_SIZE(VALUES); i++) {
        clear_env();
        setenv("AB_CHIP", "Apple M9 Ultra", 1);
        setenv("AB_GPU_CORES", "99", 1);
        if (VALUES[i].value) setenv("AB_SPLIT", VALUES[i].value, 1);
        if (VALUES[i].on) fprintf(stderr, "(expected note follows) ");
        ab_dispatch_config_from_env(&c);
        expect(c.split_on == VALUES[i].on && c.split_nrows == 0 &&
               !strcmp(c.split_table, VALUES[i].on ? "none:uncalibrated-chip" : "off"),
               "AB_SPLIT=%s: on %d, %d rows, '%s'", VALUES[i].value ? VALUES[i].value : "(unset)",
               c.split_on, c.split_nrows, c.split_table);
    }
    // A file replaces the rows; one from another chip is used.
    expect(write_temp("# chip: Apple M2 Max\n# gpu_cores: 38\nzgemm N N N m=8192: scalars=u f=0.45\n", path,
                      sizeof path), "mkstemp");
    clear_env();
    setenv("AB_CHIP", "Apple M9 Ultra", 1);
    setenv("AB_GPU_CORES", "99", 1);
    setenv("AB_SPLIT", "1", 1);
    setenv("AB_SPLIT_TABLE", path, 1);
    fprintf(stderr, "(expected note follows) ");
    ab_dispatch_config_from_env(&c);
    snprintf(want, sizeof want, "file:%s", path);
    expect(c.split_on && c.split_nrows == 1 && c.split_rows[0].share == 0.45 && !strcmp(c.split_table, want) &&
           !strcmp(c.split_table_chip, "Apple M2 Max") && c.split_table_gpu_cores == 38,
           "AB_SPLIT_TABLE replaces the rows (%d rows, %s)", c.split_nrows, c.split_table);
    // Off, the file is not read.
    unsetenv("AB_SPLIT");
    ab_dispatch_config_from_env(&c);
    expect(!c.split_on && !c.split_nrows && !strcmp(c.split_table, "off"), "AB_SPLIT unset reads no table");
    unlink(path);
    // A file that cannot be read keeps what the chip has.
    setenv("AB_SPLIT", "1", 1);
    setenv("AB_SPLIT_TABLE", "/nonexistent/split_table.txt", 1);
    fprintf(stderr, "(expected notes follow) ");
    ab_dispatch_config_from_env(&c);
    expect(c.split_on && !c.split_nrows && !strcmp(c.split_table, "none:uncalibrated-chip"),
           "an unreadable AB_SPLIT_TABLE keeps the chip's (no) table: %s", c.split_table);
    // The split leaves the routing table alone.
    ab_dispatch_config_t off = builtin_auto();
    setenv("AB_SPLIT", "1", 1);
    if (!ab_split_builtin_table(off.chip, off.gpu_cores)) fprintf(stderr, "(expected note follows) ");
    ab_dispatch_config_from_env(&c);
    expect(c.nrows == off.nrows && !memcmp(c.rows, off.rows, (size_t)c.nrows * sizeof c.rows[0]) &&
           !strcmp(c.table, off.table) && c.mode == off.mode && c.min_dim == off.min_dim,
           "AB_SPLIT changed the routing table");
    // Each built-in split table is what a machine with its chip gets.
    for (int i = 0; AB_SPLIT_BUILTIN_TABLES[i].chip; i++) {
        const ab_split_table_t *t = &AB_SPLIT_BUILTIN_TABLES[i];
        char slug[AB_DISPATCH_CHIP_LEN], cores[16];
        int n = 0;
        while (t->rows[n].routine) n++;
        clear_env();
        setenv("AB_CHIP", t->chip, 1);
        snprintf(cores, sizeof cores, "%d", t->gpu_cores);
        setenv("AB_GPU_CORES", cores, 1);
        setenv("AB_SPLIT", "1", 1);
        ab_dispatch_config_from_env(&c);
        ab_dispatch_chip_slug(t->chip, t->gpu_cores, slug, sizeof slug);
        snprintf(want, sizeof want, "builtin:%s@%s", slug, t->library);
        expect(c.split_nrows == n && !memcmp(c.split_rows, t->rows, (size_t)n * sizeof *t->rows) &&
               !strcmp(c.split_table, want), "%s: %d split rows, '%s'", slug, c.split_nrows, c.split_table);
    }
    clear_env();
    finish();
}

// The rules of scripts/ingest_split.py for a table (rows, library) against
// its split_evidence.csv: NULL if they hold, else why.
static const char *split_evidence_why(const ab_split_row_t *rows, int nrows, const char *library,
                                      const char *csv_path, char *why, size_t len) {
    static const char HEADER[] = "row,source,input,layout,library,share,pairs,split_s,base_s,ratio,"
                                 "ratio_lo,ratio_hi,calls,kept,kept_4m,range,gpu_failed";
    FILE *f = fopen(csv_path, "r");
    if (!f) {
        snprintf(why, len, "cannot read %s", csv_path);
        return why;
    }
    char line[1024];
    double base[AB_SPLIT_MAX_ROWS] = {0}, split[AB_SPLIT_MAX_ROWS] = {0};
    int lines[AB_SPLIT_MAX_ROWS] = {0};
    unsigned have[AB_SPLIT_MAX_ROWS] = {0};      // bits: pw.x A 8x1, pw.x B 8x1, bench 1x8, bench 4x2, pw.x, bench
    bool ok = fgets(line, sizeof line, f) && !strncmp(line, HEADER, sizeof HEADER - 1) &&
              strchr("\r\n", line[sizeof HEADER - 1]);
    if (!ok) snprintf(why, len, "%s: the columns are not %s", csv_path, HEADER);
    for (int n = 2; ok && fgets(line, sizeof line, f); n++) {
        char *field[18], *p = line;
        int k = 0;
        line[strcspn(line, "\r\n")] = '\0';
        for (char *t = strsep(&p, ","); t && k < 18; t = strsep(&p, ",")) field[k++] = t;
        int row = atoi(field[0]);
        if (k != 17 || row < 0 || row >= nrows) {
            snprintf(why, len, "%s:%d: not a line of 17 fields for one of the %d rows", csv_path, n, nrows);
            ok = false;
        } else if (strcmp(field[4], library) || fabs(atof(field[5]) - rows[row].share) > 1e-9) {
            snprintf(why, len, "%s:%d: library %s share %s, the table's are %s and %g (rule 1)", csv_path, n,
                     field[4], field[5], library, rows[row].share);
            ok = false;
        } else if (atof(field[10]) < 1.00) {
            snprintf(why, len, "%s:%d: ratio_lo %s < 1.00 (rule 2)", csv_path, n, field[10]);
            ok = false;
        } else if (!(atof(field[7]) > 0 && atof(field[8]) > 0)) {
            snprintf(why, len, "%s:%d: split_s and base_s must be positive", csv_path, n);
            ok = false;
        } else {
            bool pw = !strcmp(field[1], "pw.x"), bench = !strcmp(field[1], "bench_split");
            if (!pw && !bench) {
                snprintf(why, len, "%s:%d: source %s (rule 4)", csv_path, n, field[1]);
                ok = false;
            }
            have[row] |= (pw && !strcmp(field[2], "A") && !strcmp(field[3], "8x1")) << 0 |
                         (pw && !strcmp(field[2], "B") && !strcmp(field[3], "8x1")) << 1 |
                         (bench && !strcmp(field[3], "1x8")) << 2 | (bench && !strcmp(field[3], "4x2")) << 3 |
                         pw << 4 | bench << 5;
            split[row] += atof(field[7]);
            base[row] += atof(field[8]);
            lines[row]++;
        }
    }
    fclose(f);
    for (int i = 0; ok && i < nrows; i++) {
        if (!lines[i]) snprintf(why, len, "row %d has no evidence (rule 1)", i);
        else if (base[i] / split[i] < 1.10)
            snprintf(why, len, "row %d: pooled ratio %.3f < 1.10 (rule 3)", i, base[i] / split[i]);
        else if ((have[i] & 16) && (have[i] & 3) != 3)
            snprintf(why, len, "row %d: no pw.x evidence on A and B at 8x1 (rule 4)", i);
        else if ((have[i] & 32) && (have[i] & 12) != 12)
            snprintf(why, len, "row %d: no bench_split evidence at 1x8 and 4x2 (rule 4)", i);
        else if (rows[i].scalars == 'a')
            snprintf(why, len, "row %d: scalars=a in a built-in table (rule 5)", i);
        else continue;
        ok = false;
    }
    return ok ? NULL : why;
}

// Every built-in split table: its rows are those of its filed
// split_table.txt and they keep the evidence rules; and the rules catch a
// table made to break each of them.
static void test_split_builtin_matches_evidence(void) {
    char why[512], path[1024], err[256];
    begin();
    for (int i = 0; AB_SPLIT_BUILTIN_TABLES[i].chip; i++) {
        const ab_split_table_t *t = &AB_SPLIT_BUILTIN_TABLES[i];
        int n = 0;
        while (t->rows[n].routine) n++;
        snprintf(path, sizeof path, "%s/%s", AB_SOURCE_ROOT, t->data);
        expect(!split_evidence_why(t->rows, n, t->library, path, why, sizeof why), "%s: %s", t->chip, why);
        char *slash = strrchr(path, '/');
        snprintf(slash + 1, sizeof path - (size_t)(slash + 1 - path), "split_table.txt");
        ab_dispatch_config_t c = {0};
        expect(ab_split_load_table(&c, path, err, sizeof err) && c.split_nrows == n &&
               !memcmp(c.split_rows, t->rows, (size_t)n * sizeof *t->rows),
               "%s: the built-in rows are not those of %s (%s)", t->chip, path, err);
    }
    // The checker itself, on a table of two rows.
    ab_dispatch_config_t c = with_split_rows(AB_DISPATCH_AUTO, "zgemm N N N m=8192: scalars=u f=0.45; "
                                                               "zgemm N N N m=2048: ratio=2 scalars=z f=0.6");
    static const char *const H = "row,source,input,layout,library,share,pairs,split_s,base_s,ratio,"
                                 "ratio_lo,ratio_hi,calls,kept,kept_4m,range,gpu_failed\n";
    static const char *const GOOD =
        "0,pw.x,A,8x1,abc1234,0.45,6,80,95,1.19,1.12,1.25,100,100,0,0,0\n"
        "0,pw.x,B,8x1,abc1234,0.45,6,70,80,1.14,1.08,1.20,100,99,0,1,0\n"
        "1,bench_split,sq4096,1x8,abc1234,0.6,20,1.0,1.3,1.3,1.2,1.4,20,20,0,0,0\n"
        "1,bench_split,sq4096,4x2,abc1234,0.6,20,1.0,1.2,1.2,1.1,1.3,20,20,0,0,0\n";
    static const struct { const char *body; const char *rule; } BAD[] = {
        {"0,pw.x,A,8x1,abc1234,0.45,6,80,95,1.19,1.12,1.25,100,100,0,0,0\n"
         "0,pw.x,B,8x1,abc1234,0.45,6,70,80,1.14,1.08,1.20,100,99,0,1,0\n", "rule 1"},           // row 1 none
        {"0,pw.x,A,8x1,abc9999,0.45,6,80,95,1.19,1.12,1.25,100,100,0,0,0\n", "rule 1"},        // library
        {"0,pw.x,A,8x1,abc1234,0.50,6,80,95,1.19,1.12,1.25,100,100,0,0,0\n", "rule 1"},        // share
        {"0,pw.x,A,8x1,abc1234,0.45,6,80,95,1.19,0.99,1.25,100,100,0,0,0\n", "rule 2"},
        {"0,pw.x,A,8x1,abc1234,0.45,6,80,84,1.05,1.01,1.10,100,100,0,0,0\n"
         "0,pw.x,B,8x1,abc1234,0.45,6,70,80,1.14,1.08,1.20,100,99,0,1,0\n", "rule 3"},
        {"0,pw.x,A,8x1,abc1234,0.45,6,80,95,1.19,1.12,1.25,100,100,0,0,0\n"
         "0,pw.x,B,4x2,abc1234,0.45,6,70,80,1.14,1.08,1.20,100,99,0,1,0\n", "rule 4"},
        {"0,replay,A,8x1,abc1234,0.45,6,80,95,1.19,1.12,1.25,100,100,0,0,0\n", "rule 4"},
    };
    char doc[2048];
    snprintf(doc, sizeof doc, "%s%s", H, GOOD);
    expect(write_temp(doc, path, sizeof path), "mkstemp");
    expect(!split_evidence_why(c.split_rows, 2, "abc1234", path, why, sizeof why), "good evidence: %s", why);
    c.split_rows[0].scalars = 'a';
    expect(split_evidence_why(c.split_rows, 2, "abc1234", path, why, sizeof why) && strstr(why, "rule 5"),
           "scalars=a must break rule 5");
    c.split_rows[0].scalars = 'u';
    unlink(path);
    for (size_t i = 0; i < TEST_REGISTRY_SIZE(BAD); i++) {
        // the bad lines, then (except for the first case) the good lines of row 1
        snprintf(doc, sizeof doc, "%s%s%s", H, BAD[i].body,
                 i ? "1,bench_split,s,1x8,abc1234,0.6,20,1.0,1.3,1.3,1.2,1.4,20,20,0,0,0\n"
                     "1,bench_split,s,4x2,abc1234,0.6,20,1.0,1.2,1.2,1.1,1.3,20,20,0,0,0\n" : "");
        expect(write_temp(doc, path, sizeof path), "mkstemp");
        const char *w = split_evidence_why(c.split_rows, 2, "abc1234", path, why, sizeof why);
        expect(w && strstr(w, BAD[i].rule), "bad evidence %zu: expected %s, got %s", i, BAD[i].rule,
               w ? w : "a pass");
        unlink(path);
    }
    // A table without rows needs no evidence.
    expect(write_temp(H, path, sizeof path), "mkstemp");
    expect(!split_evidence_why(c.split_rows, 0, "none", path, why, sizeof why), "an empty table: %s", why);
    unlink(path);
    finish();
}

// =============================================================================
// AB_SPLIT: the slices on Accelerate (the seam, engine 1)
// =============================================================================
// The library's executor cuts a call and runs both parts on cblas, the tail
// on its own thread, in each completion order. C must be bitwise the two
// parts made one after the other by cblas on slices this file works out
// itself (the K axis: the head into C, the tail into a scratch product added
// in FP64), with A, B and C's padding untouched.

static uint64_t t2_state = 0x9E3779B97F4A7C15ULL;

static double t2_uniform(void) {
    uint64_t z = (t2_state += 0x9E3779B97F4A7C15ULL);
    z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ULL;
    z = (z ^ (z >> 27)) * 0x94D049BB133111EBULL;
    return ((double)((z ^ (z >> 31)) >> 11) * 0x1p-53) * 2.0 - 1.0;
}

// An operand stored rows x cols with leading dimension ld (w doubles per
// element), its padding NaN.
static double *t2_operand(int w, int rows, int cols, int ld) {
    size_t n = (size_t)w * ld * (cols ? cols : 1);
    double *x = malloc(n * sizeof *x);
    for (size_t i = 0; i < n; i++) x[i] = NAN;
    for (int j = 0; j < cols; j++)
        for (int i = 0; i < w * rows; i++) x[(size_t)j * w * ld + i] = t2_uniform();
    return x;
}

static enum CBLAS_TRANSPOSE t2_trans(char t) {
    return t == 'N' ? CblasNoTrans : t == 'T' ? CblasTrans : CblasConjTrans;
}

// cblas on one part; alpha and beta (real, imaginary).
static void t2_cblas(char r, char ta, char tb, int M, int N, int K, const double *alpha,
                     const double *A, int ldA, const double *B, int ldB, const double *beta,
                     double *C, int ldC) {
    if (r == 'z')
        cblas_zgemm(CblasColMajor, t2_trans(ta), t2_trans(tb), M, N, K, (const void *)alpha,
                    (const void *)A, ldA, (const void *)B, ldB, (const void *)beta, (void *)C, ldC);
    else
        cblas_dgemm(CblasColMajor, t2_trans(ta), t2_trans(tb), M, N, K, alpha[0], A, ldA, B, ldB,
                    beta[0], C, ldC);
}

// The reference of a call cut at g on axis: the head, then the tail.
static void t2_reference(char r, char axis, int g, char ta, char tb, int M, int N, int K,
                         const double *alpha, const double *A, int ldA, const double *B, int ldB,
                         const double *beta, double *C, int ldC) {
    const size_t w = r == 'z' ? 2 : 1;
    const bool na = ta == 'N', nb = tb == 'N';
    if (axis == 'N') {
        size_t c = (size_t)(N - g);
        if (c) t2_cblas(r, ta, tb, M, (int)c, K, alpha, A, ldA, B, ldB, beta, C, ldC);
        t2_cblas(r, ta, tb, M, g, K, alpha, A, ldA, B + w * (nb ? c * ldB : c), ldB, beta,
                 C + w * c * ldC, ldC);
    } else if (axis == 'M') {
        size_t c = (size_t)(M - g);
        if (c) t2_cblas(r, ta, tb, (int)c, N, K, alpha, A, ldA, B, ldB, beta, C, ldC);
        t2_cblas(r, ta, tb, g, N, K, alpha, A + w * (na ? c : c * ldA), ldA, B, ldB, beta, C + w * c,
                 ldC);
    } else {
        size_t c = (size_t)(K - g);
        static const double zero[2] = {0, 0};
        double *P = calloc(w * (size_t)M * N, sizeof *P);
        t2_cblas(r, ta, tb, M, N, (int)c, alpha, A, ldA, B, ldB, beta, C, ldC);
        t2_cblas(r, ta, tb, M, N, g, alpha, A + w * (na ? c * ldA : c), ldA, B + w * (nb ? c : c * ldB),
                 ldB, zero, P, M);
        for (int j = 0; j < N; j++)
            for (size_t i = 0; i < w * (size_t)M; i++) C[w * (size_t)j * ldC + i] += P[w * (size_t)j * M + i];
        free(P);
    }
}

// One case: every order against the reference, C whole (padding included)
// bitwise, A and B unchanged.
static void t2_case(char r, char axis, char ta, char tb, int M, int N, int K, int pad, int g,
                    const double *alpha, const double *beta, bool nan_c) {
    const int w = r == 'z' ? 2 : 1;
    const bool na = ta == 'N', nb = tb == 'N';
    const int rA = na ? M : K, cA = na ? K : M, rB = nb ? K : N, cB = nb ? N : K;
    const int ldA = rA + pad, ldB = rB + pad, ldC = M + pad;
    double *A = t2_operand(w, rA, cA, ldA), *B = t2_operand(w, rB, cB, ldB),
           *C0 = t2_operand(w, M, N, ldC);
    size_t nA = (size_t)w * ldA * cA, nB = (size_t)w * ldB * cB, nC = (size_t)w * ldC * N;
    if (nan_c)                               // beta = 0 must not read C
        for (int j = 0; j < N; j++) for (int i = 0; i < w * M; i++) C0[(size_t)j * w * ldC + i] = NAN;
    double *A0 = malloc(nA * sizeof *A0), *B0 = malloc(nB * sizeof *B0);
    double *ref = malloc(nC * sizeof *ref), *C = malloc(nC * sizeof *C);
    memcpy(A0, A, nA * sizeof *A);
    memcpy(B0, B, nB * sizeof *B);
    memcpy(ref, C0, nC * sizeof *C0);
    t2_reference(r, axis, g, ta, tb, M, N, K, alpha, A, ldA, B, ldB, beta, ref, ldC);
    for (int order = 0; order < 3; order++) {
        ab_split_forced_t f = {axis, g, order, 1};
        memcpy(C, C0, nC * sizeof *C);
        int got = r == 'z'
            ? ab_internal_zgemm_split(&f, ta, tb, M, N, K, CMPLX(alpha[0], alpha[1]), (const void *)A, ldA,
                                      (const void *)B, ldB, CMPLX(beta[0], beta[1]), (void *)C, ldC)
            : ab_internal_dgemm_split(&f, ta, tb, M, N, K, alpha[0], A, ldA, B, ldB, beta[0], C, ldC);
        expect(got == AB_SPLIT_CPU, "%cgemm %c %c %dx%dx%d axis %c g=%d order %d: outcome %d", r, ta, tb, M,
               N, K, axis, g, order, got);
        expect(!memcmp(C, ref, nC * sizeof *C), "%cgemm %c %c %dx%dx%d axis %c g=%d order %d alpha %g%+gi "
               "beta %g%+gi: C is not the two cblas parts", r, ta, tb, M, N, K, axis, g, order, alpha[0],
               alpha[1], beta[0], beta[1]);
        expect(!memcmp(A, A0, nA * sizeof *A) && !memcmp(B, B0, nB * sizeof *B),
               "%cgemm axis %c: A or B changed", r, axis);
    }
    free(A); free(B); free(C0); free(A0); free(B0); free(ref); free(C);
}

// The axes this tree's executor cuts.
static const char T2_AXES[] = "N";

static void test_split_slices_cblas(void) {
    static const double SCALARS[][4] = {{1, 0, 0, 0}, {1, 0, 1, 0}, {-1.7, 0.6, 0.3, -0.45}};
    static const char TR[] = "NTC";
    begin();
    for (const char *axis = T2_AXES; *axis; axis++)
        for (int z = 0; z < 2; z++)
            for (int a = 0; a < (z ? 3 : 2); a++)
                for (int b = 0; b < (z ? 3 : 2); b++)
                    for (size_t s = 0; s < TEST_REGISTRY_SIZE(SCALARS); s++) {
                        char r = z ? 'z' : 'd';
                        int M = 70, N = 97, K = 53;
                        if (*axis == 'K') K = 700;
                        int L = *axis == 'N' ? N : *axis == 'M' ? M : K;
                        const int G[] = {1, 32, L / 2, L - 1, L};
                        for (size_t gi = 0; gi < TEST_REGISTRY_SIZE(G); gi++)
                            t2_case(r, *axis, TR[a], TR[b], M, N, K, (int)(gi % 3), G[gi], SCALARS[s],
                                    SCALARS[s] + 2, false);
                    }
    // beta = 0 with NaN in C, and the call cut at g = 0 (cblas on the whole call).
    static const double one[2] = {1, 0}, zero[2] = {0, 0};
    for (const char *axis = T2_AXES; *axis; axis++) {
        t2_case('z', *axis, 'C', 'N', 64, 64, 512, 1, 32, one, zero, true);
        t2_case('d', *axis, 'N', 'T', 64, 64, 512, 2, 32, one, zero, true);
        t2_case('z', *axis, 'N', 'N', 40, 50, 450, 0, 0, SCALARS[2], SCALARS[2] + 2, false);
    }
    // Arguments the seam does not cut leave C alone.
    double A[64], B[64], C[64];
    for (int i = 0; i < 64; i++) A[i] = B[i] = C[i] = i;
    ab_split_forced_t bad[] = {{'N', 9, 0, 1}, {'Q', 1, 0, 1}, {'N', -1, 0, 1}, {'N', 1, 3, 1}, {'N', 1, 0, 2}};
    for (size_t i = 0; i < TEST_REGISTRY_SIZE(bad); i++)
        expect(ab_internal_dgemm_split(&bad[i], 'N', 'N', 8, 8, 8, 1.0, A, 8, B, 8, 0.0, C, 8) ==
               AB_SPLIT_INVALID, "forced cut %zu should be refused", i);
    expect(ab_internal_dgemm_split(NULL, 'N', 'N', 8, 8, 8, 1.0, A, 8, B, 8, 0.0, C, 8) == AB_SPLIT_INVALID &&
           ab_internal_dgemm_split(&(ab_split_forced_t){'N', 4, 0, 1}, 'X', 'N', 8, 8, 8, 1.0, A, 8, B, 8,
                                   0.0, C, 8) == AB_SPLIT_INVALID &&
           ab_internal_dgemm_split(&(ab_split_forced_t){'N', 4, 0, 1}, 'N', 'N', 8, 8, 8, 0.0, A, 8, B, 8,
                                   0.0, C, 8) == AB_SPLIT_INVALID &&
           ab_internal_dgemm_split(&(ab_split_forced_t){'N', 4, 0, 1}, 'N', 'N', 8, 8, 8, 1.0, A, 7, B, 8,
                                   0.0, C, 8) == AB_SPLIT_INVALID, "invalid BLAS arguments should be refused");
    bool same = true;
    for (int i = 0; i < 64; i++) same = same && C[i] == i;
    expect(same, "a refused cut wrote C");
    ab_split_record_t rec;
    expect(!ab_internal_last_split(&rec), "a refused cut left a record");
    finish();
}

// =============================================================================
// Main
// =============================================================================

static const TestCase TESTS[] = {
    {"test_builtin_tables_load",          test_builtin_tables_load},
    {"test_builtin_matches_measurements", test_builtin_matches_measurements},
    {"test_qe_shapes",                    test_qe_shapes},
    {"test_select_table",                 test_select_table},
    {"test_host_chip_and_slug",           test_host_chip_and_slug},
    {"test_uncalibrated_chip",            test_uncalibrated_chip},
    {"test_mode_precedence",              test_mode_precedence},
    {"test_row_bounds_and_classes",       test_row_bounds_and_classes},
    {"test_flop_overrides",               test_flop_overrides},
    {"test_parse_row",                    test_parse_row},
    {"test_parse_header",                 test_parse_header},
    {"test_load_table",                   test_load_table},
    {"test_config_from_env",              test_config_from_env},
    {"test_split_parse_row",              test_split_parse_row},
    {"test_split_plan_rows",              test_split_plan_rows},
    {"test_split_cut",                    test_split_cut},
    {"test_split_parts",                  test_split_parts},
    {"test_split_precedence",             test_split_precedence},
    {"test_split_select_table",           test_split_select_table},
    {"test_split_load_table",             test_split_load_table},
    {"test_split_config_from_env",        test_split_config_from_env},
    {"test_split_builtin_matches_evidence", test_split_builtin_matches_evidence},
    {"test_split_slices_cblas",           test_split_slices_cblas},
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
        int n_split = 0;
        while (AB_SPLIT_BUILTIN_TABLES[n_split].chip) n_split++;
        printf("✓ All %d tests PASSED (%d built-in table%s, %d calibrated shapes checked, %d built-in split "
               "table%s)\n", tests_passed, n_tables(), n_tables() == 1 ? "" : "s", shapes_checked, n_split,
               n_split == 1 ? "" : "s");
        return 0;
    }
    printf("✗ Some tests FAILED\n");
    return 1;
}
