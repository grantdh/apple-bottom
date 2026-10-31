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
// =============================================================================

#include "dispatch.h"
#include "dispatch_table.h"
#include "test_registry.h"

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
    "AB_MODE", "AB_MIN_GPU_DIM", "AB_CROSSOVER_FLOPS", "AB_CROSSOVER_FLOPS_REAL",
    "AB_DISPATCH_TABLE", "AB_CHIP", "AB_GPU_CORES",
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
        printf("✓ All %d tests PASSED (%d built-in table%s, %d calibrated shapes checked)\n", tests_passed,
               n_tables(), n_tables() == 1 ? "" : "s", shapes_checked);
        return 0;
    }
    printf("✗ Some tests FAILED\n");
    return 1;
}
