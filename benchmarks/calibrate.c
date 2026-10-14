// =============================================================================
// calibrate.c — BLAS-ABI GPU path vs Accelerate, and the AB_MODE=auto table
// =============================================================================
//
// Run by `make calibrate` (under caffeinate -is). For every shape of the
// grid below, in one process: the end-to-end ab_dgemm_blas / ab_zgemm_blas
// call with the GPU path forced (AB_MODE=gpu, AB_MIN_GPU_DIM=0) against
// Accelerate cblas_dgemm / cblas_zgemm on the same column-major operands
// (alpha = 1, beta = 0, tight leading dimensions). Each side gets --warmup
// untimed calls, then --runs timed calls; the sides alternate and swap order
// every run. A side's time is its median. Every GPU call must report
// dispatch path "gpu" and the GPU result must match cblas (relative
// Frobenius error <= 1e-12), or the shape counts as a GPU loss.
//
// The rule, per routine and TRANS class ('N', or 'T' for 'T' and 'C'): a
// shape is a win if Accelerate time / GPU time >= 1.10. A win becomes a row
// (the GPU for M, N, K at least its own) unless a shape of the class that is
// not a win is at least as large in M, N and K; rows inside another row's
// range are dropped. So auto never sends a measured non-win to the GPU, nor
// anything smaller in some dimension than every measured win.
//
// Writes into --out DIR (which must exist):
//   calibrate_raw.csv      every timed run
//   calibrate_summary.csv  per shape: medians, speedup, error, win, auto route
//   dispatch_table.txt     the rule, loadable with AB_DISPATCH_TABLE=<path>
// and with --source PATH the same rule as the built-in table (dispatch_table.h).
// =============================================================================

#include "apple_bottom.h"
#include "blas_trans.h"
#include "dispatch.h"

#include <Accelerate/Accelerate.h>
#include <complex.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/sysctl.h>
#include <time.h>

#ifndef AB_CAL_SDK
#define AB_CAL_SDK "unknown"
#endif

#define WIN_SPEEDUP 1.10
#define MAX_REL_ERR 1e-12
#define MAX_SHAPES  128

typedef struct {
    char routine, ta, tb;       // 'd'/'z'; actual TRANS flags passed
    int M, N, K;
    const char *source;
} Shape;

typedef struct {
    Shape s;
    double gpu_s, cpu_s, speedup, rel_err;
    bool gpu_path, win, route_gpu;
} Result;

static Shape grid[MAX_SHAPES];
static int n_grid;

static void add(char routine, char ta, char tb, int M, int N, int K, const char *source) {
    if (n_grid == MAX_SHAPES) { fprintf(stderr, "calibrate: MAX_SHAPES exceeded\n"); exit(1); }
    grid[n_grid++] = (Shape){routine, ta, tb, M, N, K, source};
}

// Quantum ESPRESSO si64 shapes first, then square and transposed-square GEMMs.
static void build_grid(void) {
    // PAW DGEMMs (newq 'T','N'; addusdens, which QE calls 'N','T'): 171 projector
    // pairs x 64 atoms; 2*ngm_l = 71834 (gamma point) or 143666 (k-points).
    static const int NGM2[] = {71834, 143666};
    for (int i = 0; i < 2; i++) {
        add('d', 'T', 'N', 171, 64, NGM2[i], "qe-paw");
        add('d', 'N', 'N', NGM2[i], 171, 64, "qe-paw");
        add('d', 'N', 'T', NGM2[i], 171, 64, "qe-paw");
    }
    // Davidson ZGEMMs of si64 k-point runs with up to 500 bands: projections
    // 'C','N' (bands x bands x npw) and updates 'N','N' (npw x bands x basis);
    // npw = 18277 on 1 rank, ~9000 per rank on 2.
    static const int BANDS[] = {160, 320, 500, 1000}, NPW[] = {9000, 18277};
    static const int UPD_N[] = {160, 500}, UPD_K[] = {160, 500, 1000};
    for (int k = 0; k < 2; k++)
        for (int b = 0; b < 4; b++)
            add('z', 'C', 'N', BANDS[b], BANDS[b], NPW[k], "qe-davidson");
    for (int m = 0; m < 2; m++)
        for (int n = 0; n < 2; n++)
            for (int k = 0; k < 3; k++)
                add('z', 'N', 'N', NPW[m], UPD_N[n], UPD_K[k], "qe-davidson");
    // 2-rank k-point run (AB_PROFILE): calbec and add_vuspsi with 1152 projectors.
    add('z', 'C', 'N', 1152, 160, 9000, "qe-k222");
    add('z', 'N', 'N', 9000, 160, 1152, "qe-k222");

    static const int SQ[] = {256, 512, 1024, 2048, 3072, 4096, 6144};
    static const int SQT[] = {1024, 2048, 4096, 6144};
    for (int r = 0; r < 2; r++) {
        char routine = r ? 'z' : 'd', t = r ? 'C' : 'T';
        for (int i = 0; i < 7; i++)
            add(routine, 'N', 'N', SQ[i], SQ[i], SQ[i], "square");
        for (int i = 0; i < 4; i++) {
            add(routine, t, 'N', SQT[i], SQT[i], SQT[i], "square-trans");
            add(routine, 'N', t, SQT[i], SQT[i], SQT[i], "square-trans");
            add(routine, t, t, SQT[i], SQT[i], SQT[i], "square-trans");
        }
    }
}

// =============================================================================
// Measurement
// =============================================================================

static double now_s(void) { return clock_gettime_nsec_np(CLOCK_UPTIME_RAW) * 1e-9; }

static uint64_t rng = 0x9E3779B97F4A7C15ULL;
static double rnd(void) {                   // uniform in [-1, 1)
    rng ^= rng >> 12; rng ^= rng << 25; rng ^= rng >> 27;
    return (double)((rng * 0x2545F4914F6CDD1DULL) >> 11) * 0x1.0p-52 - 1.0;
}

typedef struct {
    double *A, *B, *Cg, *Cc;   // complex data: interleaved re/im pairs
    int lda, ldb;
} Operands;

static double time_call(const Shape *s, const Operands *o, bool gpu) {
    const int M = s->M, N = s->N, K = s->K;
    double t0 = now_s();
    if (s->routine == 'd') {
        if (gpu)
            ab_dgemm_blas(s->ta, s->tb, M, N, K, 1.0, o->A, o->lda, o->B, o->ldb, 0.0, o->Cg, M);
        else
            cblas_dgemm(CblasColMajor, ab_blas_trans(s->ta, false), ab_blas_trans(s->tb, false),
                        M, N, K, 1.0, o->A, o->lda, o->B, o->ldb, 0.0, o->Cc, M);
    } else {
        const double complex one = 1.0, zero = 0.0;
        const double complex *A = (double complex *)o->A, *B = (double complex *)o->B;
        if (gpu)
            ab_zgemm_blas(s->ta, s->tb, M, N, K, one, A, o->lda, B, o->ldb,
                          zero, (double complex *)o->Cg, M);
        else
            cblas_zgemm(CblasColMajor, ab_blas_trans(s->ta, true), ab_blas_trans(s->tb, true),
                        M, N, K, &one, A, o->lda, B, o->ldb, &zero, (double complex *)o->Cc, M);
    }
    return now_s() - t0;
}

// One timed call; a GPU call that did not run on the GPU clears *gpu_path.
static double timed(const Shape *s, const Operands *o, bool gpu, bool *gpu_path) {
    double t = time_call(s, o, gpu);
    if (gpu && strcmp(ab_get_last_dispatch_path(), "gpu")) *gpu_path = false;
    return t;
}

static int cmp_double(const void *a, const void *b) {
    double x = *(const double *)a, y = *(const double *)b;
    return (x > y) - (x < y);
}

static double median(double *v, int n) {
    qsort(v, (size_t)n, sizeof *v, cmp_double);
    return n % 2 ? v[n / 2] : 0.5 * (v[n / 2 - 1] + v[n / 2]);
}

static Result measure(const Shape *s, int warmup, int runs, FILE *raw) {
    const int w = s->routine == 'z' ? 2 : 1;           // doubles per element
    const int rowsA = s->ta == 'N' ? s->M : s->K, colsA = s->ta == 'N' ? s->K : s->M;
    const int rowsB = s->tb == 'N' ? s->K : s->N, colsB = s->tb == 'N' ? s->N : s->K;
    const size_t na = (size_t)w * rowsA * colsA, nb = (size_t)w * rowsB * colsB;
    const size_t nc = (size_t)w * s->M * s->N;
    Operands o = {malloc(na * sizeof(double)), malloc(nb * sizeof(double)),
                  calloc(nc, sizeof(double)), calloc(nc, sizeof(double)), rowsA, rowsB};
    if (!o.A || !o.B || !o.Cg || !o.Cc) {
        fprintf(stderr, "calibrate: out of memory\n");
        exit(1);
    }
    for (size_t i = 0; i < na; i++) o.A[i] = rnd();
    for (size_t i = 0; i < nb; i++) o.B[i] = rnd();

    Result r = {.s = *s, .gpu_path = true};
    double *tg = malloc((size_t)runs * sizeof *tg), *tc = malloc((size_t)runs * sizeof *tc);
    for (int i = 0; i < warmup; i++) {
        timed(s, &o, true, &r.gpu_path);
        timed(s, &o, false, &r.gpu_path);
    }
    for (int i = 0; i < runs; i++) {
        bool gpu_first = i % 2 == 0;
        double first = timed(s, &o, gpu_first, &r.gpu_path);
        double second = timed(s, &o, !gpu_first, &r.gpu_path);
        tg[i] = gpu_first ? first : second;
        tc[i] = gpu_first ? second : first;
        fprintf(raw, "%cgemm,%c,%c,%d,%d,%d,%s,%d,%.9f,%.9f\n", s->routine, s->ta, s->tb,
                s->M, s->N, s->K, s->source, i, tg[i], tc[i]);
    }
    fflush(raw);

    double diff = 0.0, ref = 0.0;
    for (size_t i = 0; i < nc; i++) {
        diff += (o.Cg[i] - o.Cc[i]) * (o.Cg[i] - o.Cc[i]);
        ref += o.Cc[i] * o.Cc[i];
    }
    r.rel_err = ref > 0.0 ? sqrt(diff / ref) : sqrt(diff);
    r.gpu_s = median(tg, runs);
    r.cpu_s = median(tc, runs);
    r.speedup = r.cpu_s / r.gpu_s;
    r.win = r.gpu_path && r.rel_err <= MAX_REL_ERR && r.speedup >= WIN_SPEEDUP;
    free(tg); free(tc);
    free(o.A); free(o.B); free(o.Cg); free(o.Cc);
    return r;
}

// =============================================================================
// Rule
// =============================================================================

// The row shape s would become: its routine and TRANS class, M, N, K at least its own.
static ab_dispatch_row_t row_of(const Shape *s) {
    return (ab_dispatch_row_t){s->routine, s->ta == 'N' ? 'N' : 'T', s->tb == 'N' ? 'N' : 'T',
                               s->M, s->N, s->K};
}

static bool covers(ab_dispatch_row_t r, const Shape *s) {
    return ab_dispatch_row_covers(&r, s->routine, s->ta, s->tb, s->M, s->N, s->K);
}

// Fills c (auto mode, default AB_MIN_GPU_DIM, no FLOP override) with the
// rule's rows, src[r] the result row r came from, and every route_gpu.
static void derive_rule(Result *res, int n, ab_dispatch_config_t *c, const Result **src) {
    bool clean[MAX_SHAPES];
    for (int i = 0; i < n; i++) {
        clean[i] = res[i].win;
        for (int j = 0; j < n && clean[i]; j++)
            if (!res[j].win && covers(row_of(&res[i].s), &res[j].s)) clean[i] = false;
    }
    memset(c, 0, sizeof *c);
    c->mode = AB_DISPATCH_AUTO;
    c->min_dim = AB_DISPATCH_DEFAULT_MIN_DIM;
    for (int i = 0; i < n; i++) {
        bool covered = !clean[i];
        for (int j = 0; j < n && !covered; j++)   // by a smaller row, or an equal earlier one
            covered = j != i && clean[j] && covers(row_of(&res[j].s), &res[i].s) &&
                      (j < i || !covers(row_of(&res[i].s), &res[j].s));
        if (covered) continue;
        if (c->nrows == AB_DISPATCH_MAX_ROWS) {
            fprintf(stderr, "calibrate: more than %d table rows\n", AB_DISPATCH_MAX_ROWS);
            exit(1);
        }
        src[c->nrows] = &res[i];
        c->rows[c->nrows++] = row_of(&res[i].s);
    }
    for (int i = 0; i < n; i++) {
        const Shape *s = &res[i].s;
        res[i].route_gpu = ab_dispatch_gpu(c, s->routine, s->ta, s->tb, s->M, s->N, s->K);
        if (res[i].route_gpu && !res[i].win) {
            fprintf(stderr, "calibrate: rule routes a measured non-win to the GPU\n");
            exit(1);
        }
    }
}

// =============================================================================
// Output
// =============================================================================

typedef struct {
    char date[32], machine[192], os[96], sdk[128], library[128], data[512];
    int warmup, runs;
} Meta;

static void sysctl_str(const char *name, char *buf, size_t len) {
    size_t n = len;
    if (sysctlbyname(name, buf, &n, NULL, 0) != 0) snprintf(buf, len, "unknown");
}

static long long sysctl_ll(const char *name) {
    long long v = 0;
    size_t n = sizeof v;
    return sysctlbyname(name, &v, &n, NULL, 0) == 0 ? v : 0;
}

static void collect_meta(Meta *m, const char *out, int gpu_cores, int warmup, int runs) {
    char cpu[96], ver[32], build[32], cores[32] = "";
    time_t t = time(NULL);
    strftime(m->date, sizeof m->date, "%Y-%m-%dT%H:%M:%SZ", gmtime(&t));
    sysctl_str("machdep.cpu.brand_string", cpu, sizeof cpu);
    if (gpu_cores > 0) snprintf(cores, sizeof cores, "%d-core ", gpu_cores);
    snprintf(m->machine, sizeof m->machine, "%s, %sGPU, %lld GB, %lldP+%lldE CPU cores",
             cpu, cores, sysctl_ll("hw.memsize") >> 30,
             sysctl_ll("hw.perflevel0.physicalcpu"), sysctl_ll("hw.perflevel1.physicalcpu"));
    sysctl_str("kern.osproductversion", ver, sizeof ver);
    sysctl_str("kern.osversion", build, sizeof build);
    snprintf(m->os, sizeof m->os, "macOS %s (%s)", ver, build);
    snprintf(m->sdk, sizeof m->sdk, "macOS SDK %s, clang %s", AB_CAL_SDK, __clang_version__);
    snprintf(m->library, sizeof m->library, "%s", ab_build_info());
    snprintf(m->data, sizeof m->data, "%s/calibrate_summary.csv", out);
    m->warmup = warmup;
    m->runs = runs;
}

// The provenance and result summary, each line starting with comment prefix p
// ("# " or "// "); blank lines get p without its trailing space.
static void write_header(FILE *f, const char *p, const Meta *m, const Result *res, int n) {
    const int bare = (int)strlen(p) - 1;
    fprintf(f, "%sDate:     %s\n%sMachine:  %s\n%sOS:       %s\n%sSDK:      %s\n"
               "%sLibrary:  %s\n%sData:     %s\n"
               "%sMethod:   ab_*gemm_blas, GPU path forced, vs Accelerate cblas in one\n"
               "%s          process; alpha 1, beta 0, tight ld; %d warmups, median of %d\n"
               "%s          timed runs; GPU results cross-checked against cblas\n%.*s\n",
            p, m->date, p, m->machine, p, m->os, p, m->sdk, p, m->library, p, m->data,
            p, p, m->warmup, m->runs, p, bare, p);
    fprintf(f, "%sAccelerate time / GPU time per routine and TRANS class (a win is >= %.2fx):\n",
            p, WIN_SPEEDUP);
    static const char *const CLASSES[] = {"dNN", "dNT", "dTN", "dTT", "zNN", "zNT", "zTN", "zTT"};
    for (int c = 0; c < 8; c++) {
        const ab_dispatch_row_t cls = {CLASSES[c][0], CLASSES[c][1], CLASSES[c][2], 0, 0, 0};
        int count = 0, wins = 0;
        const Result *best = NULL;
        for (int j = 0; j < n; j++) {
            if (!covers(cls, &res[j].s)) continue;
            count++;
            wins += res[j].win;
            if (!best || res[j].speedup > best->speedup) best = &res[j];
        }
        if (best)
            fprintf(f, "%s  %cgemm %c %c  %2d shape%s, %d win%s; best %.2fx at %dx%dx%d\n", p,
                    cls.routine, cls.ta, cls.tb, count, count == 1 ? "" : "s", wins,
                    wins == 1 ? "" : "s", best->speedup, best->s.M, best->s.N, best->s.K);
    }
    fprintf(f, "%.*s\n%sA call goes to the GPU iff M >= min_M, N >= min_N and K >= min_K for a\n"
               "%srow of its routine and TRANS class ('T' rows also cover 'C'); no row:\n"
               "%sAccelerate. Each row is a measured win with no slower shape of its class\n"
               "%sat least as large in M, N and K.\n", bare, p, p, p, p, p);
}

static FILE *open_out(const char *dir, const char *name) {
    char path[1024];
    snprintf(path, sizeof path, "%s/%s", dir, name);
    FILE *f = fopen(path, "w");
    if (!f) {
        perror(path);
        exit(1);
    }
    return f;
}

static void write_summary(const char *dir, const Result *res, int n) {
    FILE *f = open_out(dir, "calibrate_summary.csv");
    fprintf(f, "routine,transA,transB,M,N,K,source,gflop,gpu_median_s,cpu_median_s,speedup,"
               "gpu_gflops,cpu_gflops,rel_err,gpu_path,win,auto_route\n");
    for (int i = 0; i < n; i++) {
        const Result *r = &res[i];
        double gflop = (r->s.routine == 'z' ? 8.0 : 2.0) * r->s.M * r->s.N * r->s.K * 1e-9;
        fprintf(f, "%cgemm,%c,%c,%d,%d,%d,%s,%.3f,%.6f,%.6f,%.3f,%.1f,%.1f,%.2e,%d,%d,%s\n",
                r->s.routine, r->s.ta, r->s.tb, r->s.M, r->s.N, r->s.K, r->s.source, gflop,
                r->gpu_s, r->cpu_s, r->speedup, gflop / r->gpu_s, gflop / r->cpu_s,
                r->rel_err, r->gpu_path, r->win, r->route_gpu ? "gpu" : "cpu");
    }
    fclose(f);
}

static void write_table(const char *dir, const Meta *m, const Result *res, int n,
                        const ab_dispatch_config_t *c, const Result **src) {
    FILE *f = open_out(dir, "dispatch_table.txt");
    fprintf(f, "# apple-bottom AB_MODE=auto dispatch table, for AB_DISPATCH_TABLE=<this file>\n"
               "# Written by `make calibrate` (benchmarks/calibrate.c).\n#\n");
    write_header(f, "# ", m, res, n);
    fprintf(f, "#\n# routine transA transB min_M min_N min_K\n");
    for (int r = 0; r < c->nrows; r++)
        fprintf(f, "%cgemm %c %c %d %d %d   # %.2fx\n", c->rows[r].routine, c->rows[r].ta,
                c->rows[r].tb, c->rows[r].m, c->rows[r].n, c->rows[r].k, src[r]->speedup);
    if (!c->nrows) fprintf(f, "# (no rows: every call stays on Accelerate)\n");
    fclose(f);
}

static void write_source(const char *path, const Meta *m, const Result *res, int n,
                         const ab_dispatch_config_t *c, const Result **src) {
    FILE *f = fopen(path, "w");
    if (!f) {
        perror(path);
        exit(1);
    }
    fprintf(f, "// =============================================================================\n"
               "// dispatch_table.h — built-in AB_MODE=auto rule for BLAS-ABI calls (dispatch.h)\n"
               "// =============================================================================\n"
               "// Generated by `make calibrate` (benchmarks/calibrate.c); do not edit. To\n"
               "// recalibrate, rerun it and commit this file with its results directory.\n//\n");
    write_header(f, "// ", m, res, n);
    fprintf(f, "// =============================================================================\n"
               "#ifndef AB_DISPATCH_TABLE_H\n#define AB_DISPATCH_TABLE_H\n\n"
               "#include \"dispatch.h\"\n\n"
               "// The measurements behind the table (tests/test_dispatch.c checks them).\n"
               "#define AB_DISPATCH_BUILTIN_DATA \"%s\"\n\n"
               "// routine, transA, transB, min_M, min_N, min_K\n"
               "static const ab_dispatch_row_t AB_DISPATCH_BUILTIN[] = {\n", m->data);
    for (int r = 0; r < c->nrows; r++)
        fprintf(f, "    {'%c', '%c', '%c', %d, %d, %d},   // %.2fx\n", c->rows[r].routine,
                c->rows[r].ta, c->rows[r].tb, c->rows[r].m, c->rows[r].n, c->rows[r].k,
                src[r]->speedup);
    fprintf(f, "    {0, 0, 0, 0, 0, 0},%s\n};\n\n#endif // AB_DISPATCH_TABLE_H\n",
            c->nrows ? "" : "   // no rows: every call stays on Accelerate");
    fclose(f);
}

// =============================================================================
// Main
// =============================================================================

int main(int argc, char **argv) {
    const char *out = NULL, *source = NULL, *only = NULL;
    int warmup = 3, runs = 7, gpu_cores = 0;
    for (int i = 1; i < argc; i++) {
        const char *v = i + 1 < argc ? argv[i + 1] : NULL;
        if (!strcmp(argv[i], "--out") && v) out = argv[++i];
        else if (!strcmp(argv[i], "--source") && v) source = argv[++i];
        else if (!strcmp(argv[i], "--only") && v) only = argv[++i];
        else if (!strcmp(argv[i], "--warmup") && v) warmup = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--runs") && v) runs = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--gpu-cores") && v) gpu_cores = atoi(argv[++i]);
        else { out = NULL; break; }
    }
    if (!out || warmup < 3 || runs < 7) {
        fprintf(stderr, "usage: %s --out DIR [--source dispatch_table.h] [--only SOURCE_TAG]\n"
                        "       [--warmup N>=3] [--runs N>=7] [--gpu-cores N]\n", argv[0]);
        return 2;
    }
    // A table derived from part of the grid would drop the rows of the rest.
    if (only && source) {
        fprintf(stderr, "calibrate: --only %s measures part of the grid; %s is left unchanged\n",
                only, source);
        source = NULL;
    }

    // ab_*gemm_blas read these at their first call: every call takes the GPU path.
    setenv("AB_MODE", "gpu", 1);
    setenv("AB_MIN_GPU_DIM", "0", 1);
    unsetenv("AB_PROFILE");
    unsetenv("AB_REPORT");
    if (ab_init() != AB_OK) {
        fprintf(stderr, "calibrate: no Metal device\n");
        return 1;
    }
    build_grid();

    FILE *raw = open_out(out, "calibrate_raw.csv");
    fprintf(raw, "routine,transA,transB,M,N,K,source,run,gpu_s,cpu_s\n");
    static Result res[MAX_SHAPES];
    int n = 0;
    for (int i = 0; i < n_grid; i++) {
        if (only && !strstr(grid[i].source, only)) continue;
        const Shape *s = &grid[i];
        res[n] = measure(s, warmup, runs, raw);
        const Result *r = &res[n++];
        printf("%cgemm %c %c %6d x %6d x %6d  %-12s GPU %9.4f s  Accelerate %9.4f s  "
               "%5.2fx  err %.1e%s\n", s->routine, s->ta, s->tb, s->M, s->N, s->K, s->source,
               r->gpu_s, r->cpu_s, r->speedup, r->rel_err,
               !r->gpu_path ? "  [GPU path not taken]"
               : r->rel_err > MAX_REL_ERR ? "  [GPU result differs]" : "");
        fflush(stdout);
    }
    fclose(raw);

    ab_dispatch_config_t rule;
    const Result *src[AB_DISPATCH_MAX_ROWS];
    derive_rule(res, n, &rule, src);
    Meta meta;
    collect_meta(&meta, out, gpu_cores, warmup, runs);
    write_summary(out, res, n);
    write_table(out, &meta, res, n, &rule, src);
    if (source) write_source(source, &meta, res, n, &rule, src);

    int wins = 0;
    for (int i = 0; i < n; i++) wins += res[i].win;
    printf("\n%d of %d shapes >= %.2fx faster on the GPU; %d table row%s:\n",
           wins, n, WIN_SPEEDUP, rule.nrows, rule.nrows == 1 ? "" : "s");
    for (int r = 0; r < rule.nrows; r++)
        printf("  %cgemm %c %c  M >= %d, N >= %d, K >= %d  (%.2fx)\n", rule.rows[r].routine,
               rule.rows[r].ta, rule.rows[r].tb, rule.rows[r].m, rule.rows[r].n,
               rule.rows[r].k, src[r]->speedup);
    printf("Wrote %s/{calibrate_raw.csv,calibrate_summary.csv,dispatch_table.txt}%s%s\n",
           out, source ? " and " : "", source ? source : "");
    ab_shutdown();
    return 0;
}
