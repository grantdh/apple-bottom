// =============================================================================
// replay.c — replay the BLAS calls of a traced Quantum ESPRESSO run
// =============================================================================
//
// Usage: replay <trace-dir> --mode cpu|gpu|nohost [--pools P] [--gap-scale X]
//               [--no-gaps] [--timeout S] [--out DIR]
//
// <trace-dir> holds the AB_TRACE files (trace.<pid>) of one run. Built by
// `make bench-replay`. What it replays and how, its output and its limits:
// "Replay" in benchmarks/qe/README.md.
// =============================================================================

#include "apple_bottom.h"
#include "blas_trans.h"
#include "dispatch.h"
#include "profiling/blas_profiler.h"
#include "test_blas_check.h"    // fortran_dgemm/zgemm (dgemm_, zgemm_), is_notrans, splitmix

#include <complex.h>
#include <dirent.h>
#include <errno.h>
#include <fcntl.h>
#include <limits.h>
#include <mach-o/dyld.h>
#include <signal.h>
#include <spawn.h>
#include <stdarg.h>
#include <stdatomic.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <sys/wait.h>
#include <unistd.h>

#define MAX_RANKS      256
#define START_DELAY_NS 5000000ULL   // the common start, after the last worker is ready

typedef enum { MODE_CPU, MODE_GPU, MODE_NOHOST } Mode;
static const char *const MODE_NAMES[] = {"cpu", "gpu", "nohost"};

// One traced call, and how the replay makes it.
typedef struct {
    uint64_t t0, t1;            // traced start and end, ns
    uint64_t gap;               // ns to wait before the call, before --gap-scale
    double alpha[2], beta[2];   // real, imaginary (zgemm)
    int m, n, k, lda, ldb, ldc;
    char routine, ta, tb;       // 'd' or 'z'; TRANS as traced
    bool barrier;               // the pool meets after the call
} Call;

typedef struct {
    size_t first, n;            // the rank's calls: Shared.calls[first, first + n)
    uint32_t candidates;        // calls QE may reduce over the pool (needs_barrier)
    uint32_t barriers;          // calls followed by a pool barrier
    uint64_t removed_ns;        // traced pool waits taken out of the gaps
} RankPlan;

// A sense-reversing spin barrier, on a cache line of its own.
typedef struct {
    _Alignas(128) _Atomic uint32_t arrived;
    _Atomic uint32_t sense;
} Barrier;

// A worker's totals, stored once when it is done; a cache line of its own.
typedef struct {
    _Alignas(128) int32_t pid;
    uint64_t gpu_calls;             // gpu: ABI calls the GPU computed; nohost: native calls
    uint64_t first_ns, last_ns;     // first call start, last call end
    uint64_t call_ns, gap_ns, offset_ns, barrier_ns;
} WorkerSummary;

// The memory the parent shares with the workers.
typedef struct {
    int32_t n_ranks, pool_size, mode;
    double gap_scale;
    uint64_t timeout_ns;
    _Atomic uint32_t ready;
    _Atomic uint64_t start_ns;
    Barrier pool[MAX_RANKS];
    RankPlan plan[MAX_RANKS];
    WorkerSummary sum[MAX_RANKS];
    Call calls[];
} Shared;

static uint64_t now_ns(void) { return ab_prof_ticks_to_ns(ab_prof_now_ticks()); }

static double sec(uint64_t ns) { return ns / 1e9; }

static void die(const char *fmt, ...) __attribute__((format(printf, 1, 2), noreturn));
static void die(const char *fmt, ...) {
    va_list ap;
    va_start(ap, fmt);
    fprintf(stderr, "replay: ");
    vfprintf(stderr, fmt, ap);
    fprintf(stderr, "\n");
    va_end(ap);
    exit(2);
}

// =============================================================================
// Reading a trace directory
// =============================================================================

typedef struct {
    char path[PATH_MAX];
    int pid, rank, size, headers;
    int min_dim;                // AB_MIN_GPU_DIM of the traced process; -1 if not in the header
    char mode[16];              // AB_MODE of the traced process
    Call *calls;
    size_t n, cap;
    size_t overlaps;            // calls that started before the previous one ended
} TraceFile;

static bool parse_int(const char *s, int *out) {
    char *end;
    errno = 0;
    long v = strtol(s, &end, 10);
    if (!*s || *end || errno || v < INT_MIN || v > INT_MAX) return false;
    *out = (int)v;
    return true;
}

static bool parse_u64(const char *s, uint64_t *out) {
    char *end;
    errno = 0;
    unsigned long long v = strtoull(s, &end, 10);
    if (*s < '0' || *s > '9' || *end || errno) return false;
    *out = v;
    return true;
}

// "%a" or, for zgemm, "%a,%a".
static bool parse_scalar(const char *s, bool z, double out[2]) {
    char *end;
    out[0] = strtod(s, &end);
    out[1] = 0.0;
    if (end == s) return false;
    if (z) {
        if (*end != ',') return false;
        s = end + 1;
        out[1] = strtod(s, &end);
        if (end == s) return false;
    }
    return !*end;
}

static bool parse_trans(const char *s, char *out) {
    if (!s[0] || s[1] || ab_blas_trans(s[0], true) == AB_TRANS_INVALID) return false;
    *out = s[0];
    return true;
}

// One call line into c; false with the reason in err.
static bool parse_call(char *line, Call *c, char *err, size_t errlen) {
    enum { F_T0, F_T1, F_ROUTINE, F_TA, F_TB, F_M, F_N, F_K, F_LDA, F_LDB, F_LDC,
           F_ALPHA, F_BETA, F_COUNT };
    static const char *const KEYS[F_COUNT] = {"t0_ns", "t1_ns", "routine", "ta", "tb", "M", "N",
                                              "K", "lda", "ldb", "ldc", "alpha", "beta"};
    const char *val[F_COUNT] = {0};
    char *save = NULL;
    for (char *tok = strtok_r(line, " \n", &save); tok; tok = strtok_r(NULL, " \n", &save)) {
        char *eq = strchr(tok, '=');
        if (!eq) {
            snprintf(err, errlen, "field '%s' is not key=value", tok);
            return false;
        }
        *eq = '\0';
        for (int f = 0; f < F_COUNT; f++)
            if (!strcmp(tok, KEYS[f])) val[f] = eq + 1;
    }
    for (int f = 0; f < F_COUNT; f++)
        if (!val[f]) {
            snprintf(err, errlen, "no %s field", KEYS[f]);
            return false;
        }
    memset(c, 0, sizeof *c);
    if (!strcmp(val[F_ROUTINE], "dgemm")) c->routine = 'd';
    else if (!strcmp(val[F_ROUTINE], "zgemm")) c->routine = 'z';
    else {
        snprintf(err, errlen, "routine %s is not replayed", val[F_ROUTINE]);
        return false;
    }
    bool z = c->routine == 'z';
    int *dims[] = {&c->m, &c->n, &c->k, &c->lda, &c->ldb, &c->ldc};
    for (int d = 0; d < 6; d++)
        if (!parse_int(val[F_M + d], dims[d])) {
            snprintf(err, errlen, "bad %s=%s", KEYS[F_M + d], val[F_M + d]);
            return false;
        }
    if (!parse_u64(val[F_T0], &c->t0) || !parse_u64(val[F_T1], &c->t1) || c->t1 < c->t0) {
        snprintf(err, errlen, "bad t0_ns=%s t1_ns=%s", val[F_T0], val[F_T1]);
        return false;
    }
    if (!parse_trans(val[F_TA], &c->ta) || !parse_trans(val[F_TB], &c->tb)) {
        snprintf(err, errlen, "bad ta=%s tb=%s", val[F_TA], val[F_TB]);
        return false;
    }
    if (!parse_scalar(val[F_ALPHA], z, c->alpha) || !parse_scalar(val[F_BETA], z, c->beta)) {
        snprintf(err, errlen, "bad alpha=%s beta=%s", val[F_ALPHA], val[F_BETA]);
        return false;
    }
    return true;
}

static int cmp_t0(const void *a, const void *b) {
    uint64_t x = ((const Call *)a)->t0, y = ((const Call *)b)->t0;
    return (x > y) - (x < y);
}

static void read_trace(TraceFile *f) {
    FILE *in = fopen(f->path, "r");
    if (!in) die("cannot read %s: %s", f->path, strerror(errno));
    f->rank = f->size = f->min_dim = -1;
    char *line = NULL, err[256];
    size_t len = 0, lineno = 0;
    bool sorted = true;
    while (getline(&line, &len, in) > 0) {
        lineno++;
        if (line[0] == '#') {
            int r, s;
            if (strstr(line, " AB_TRACE pid=")) f->headers++;
            else if (sscanf(line, "# mpi_rank=%d mpi_size=%d", &r, &s) == 2) {
                f->rank = r;
                f->size = s;
            } else if (sscanf(line, "# AB_MODE=%15s", f->mode) == 1) {
                const char *d = strstr(line, " AB_MIN_GPU_DIM=");
                if (d) f->min_dim = atoi(d + 16);
            }
            continue;
        }
        if (line[0] == '\n') continue;
        if (f->n == f->cap) {
            f->cap = f->cap ? 2 * f->cap : 1024;
            f->calls = realloc(f->calls, f->cap * sizeof *f->calls);
            if (!f->calls) die("out of memory reading %s", f->path);
        }
        Call *c = &f->calls[f->n];
        if (!parse_call(line, c, err, sizeof err)) die("%s:%zu: %s", f->path, lineno, err);
        if (f->n && c->t0 < f->calls[f->n - 1].t0) sorted = false;
        f->n++;
    }
    free(line);
    fclose(in);
    if (f->headers != 1)
        die("%s holds %d trace headers; expected one (a pid reused by another run "
            "appends to the file)", f->path, f->headers);
    // Lines are written as calls return; a multi-threaded caller's calls can
    // end out of order. The replay makes them in start order, one at a time.
    if (!sorted) qsort(f->calls, f->n, sizeof *f->calls, cmp_t0);
    for (size_t i = 1; i < f->n; i++)
        if (f->calls[i].t0 < f->calls[i - 1].t1) f->overlaps++;
}

// The trace files of dir, read, in rank order; returns how many.
static int read_trace_dir(const char *dir, TraceFile **out) {
    DIR *d = opendir(dir);
    if (!d) die("cannot open %s: %s", dir, strerror(errno));
    int n = 0, cap = 0;
    TraceFile *f = NULL;
    for (struct dirent *e; (e = readdir(d)); ) {
        const char *s = e->d_name;
        if (strncmp(s, "trace.", 6) || !s[6] || strspn(s + 6, "0123456789") != strlen(s + 6))
            continue;
        if (n == cap) {
            cap = cap ? 2 * cap : 16;
            f = realloc(f, (size_t)cap * sizeof *f);
            if (!f) die("out of memory");
        }
        memset(&f[n], 0, sizeof f[n]);
        snprintf(f[n].path, sizeof f[n].path, "%s/%s", dir, s);
        f[n].pid = atoi(s + 6);
        read_trace(&f[n]);
        n++;
    }
    closedir(d);
    if (!n) die("no trace.<pid> files in %s", dir);
    if (n == 1 && f[0].rank == -1) {
        f[0].rank = 0;          // a run without MPI
        *out = f;
        return 1;
    }
    if (n > MAX_RANKS) die("%d ranks; at most %d", n, MAX_RANKS);
    TraceFile *by_rank = calloc((size_t)n, sizeof *by_rank);
    for (int i = 0; i < n; i++) {
        if (f[i].rank < 0)
            die("%s has no MPI rank (no '# mpi_rank=' header line, or -1), but %s holds %d "
                "trace files; put one process's file in a directory of its own",
                f[i].path, dir, n);
        if (f[i].size != n)
            die("%s: mpi_size=%d, but %s holds %d trace files (a rank that made no BLAS call "
                "writes none)", f[i].path, f[i].size, dir, n);
        if (f[i].rank >= n || by_rank[f[i].rank].path[0])
            die("%s: rank %d is out of range or repeated", f[i].path, f[i].rank);
        by_rank[f[i].rank] = f[i];
    }
    free(f);
    *out = by_rank;
    return n;
}

// The AB_MIN_GPU_DIM the traced run routed with, from its headers; the
// default for a trace whose header does not say (*traced = false).
static int traced_min_dim(const TraceFile *f, int n, bool *traced) {
    for (int r = 1; r < n; r++)
        if (f[r].min_dim != f[0].min_dim)
            die("%s and %s name different AB_MIN_GPU_DIM (%d, %d)", f[0].path, f[r].path,
                f[0].min_dim, f[r].min_dim);
    *traced = f[0].min_dim >= 0;
    return *traced ? f[0].min_dim : AB_DISPATCH_DEFAULT_MIN_DIM;
}

// =============================================================================
// The plan
// =============================================================================

// QE sums the result over the pool (mp_sum) after its projections:
// op(A) transposed or conjugated, op(B) not.
static bool needs_barrier(const Call *c) { return !is_notrans(c->ta) && is_notrans(c->tb); }

// The same reduction on every rank of a pool: same routine, TRANS classes
// and M, N (the array summed); K is the rank's share of the plane waves.
static bool same_site(const Call *a, const Call *b) {
    return a->routine == b->routine && is_notrans(a->ta) == is_notrans(b->ta) &&
           is_notrans(a->tb) == is_notrans(b->tb) && a->m == b->m && a->n == b->n;
}

// The time after call i of a rank in which a reduction could have waited:
// from its end to the start of the rank's next call.
static uint64_t window_end(const TraceFile *f, size_t i) {
    return i + 1 < f->n ? f->calls[i + 1].t0 : UINT64_MAX;
}

// A barrier after the calls at[r0 .. r1) of the pool. Each rank's traced
// wait for the last of them is taken out of its next gap; the windows share
// a point, so the wait fits in the gap.
static void place_barrier(TraceFile *f, int r0, int r1, const size_t *at, RankPlan *plan) {
    uint64_t latest = 0;
    for (int r = r0; r < r1; r++)
        if (f[r].calls[at[r]].t1 > latest) latest = f[r].calls[at[r]].t1;
    for (int r = r0; r < r1; r++) {
        Call *c = &f[r].calls[at[r]];
        c->barrier = true;
        plan[r].barriers++;
        if (at[r] + 1 == f[r].n) continue;
        uint64_t wait = latest - c->t1;
        if (wait > c[1].gap) wait = c[1].gap;
        c[1].gap -= wait;
        plan[r].removed_ns += wait;
    }
}

// Gaps and pool barriers of every call. A barrier follows a call QE may
// reduce over the pool (needs_barrier) where the trace shows the whole pool
// waited there: every rank of the pool made the same call (same routine,
// TRANS classes, M and N), and no rank started its next call before the last
// of them had ended, i.e. the windows from each call's end to the rank's
// next call share a point. Such barriers come in the same order on every
// rank. Calls without one: PAW's per-atom DGEMMs (atoms are spread over the
// ranks, and summed after the loop), and the reductions of a subgroup, such
// as the parallel subspace diagonalization's.
static void plan_calls(TraceFile *f, int n, int pools, RankPlan *plan) {
    int pool_size = n / pools;
    uint64_t t_min = UINT64_MAX;
    for (int r = 0; r < n; r++)
        if (f[r].n && f[r].calls[0].t0 < t_min) t_min = f[r].calls[0].t0;
    for (int r = 0; r < n; r++) {
        for (size_t i = 0; i < f[r].n; i++) {
            Call *c = &f[r].calls[i];
            c->gap = i ? (c->t0 > c[-1].t1 ? c->t0 - c[-1].t1 : 0) : c->t0 - t_min;
            plan[r].candidates += needs_barrier(c);
        }
        if (f[r].overlaps)
            fprintf(stderr, "replay: warning: rank %d: %zu calls overlap the previous one "
                            "(several threads); they are replayed one after another\n",
                    r, f[r].overlaps);
    }
    if (pool_size == 1) return;

    for (int p = 0; p < pools; p++) {
        int r0 = p * pool_size, r1 = r0 + pool_size;
        // Each barrier includes a call of rank r0: take them in order, and
        // look on every other rank, from its cursor on, for the same call
        // whose window meets the common part of those found so far.
        size_t at[MAX_RANKS], cursor[MAX_RANKS];
        for (int r = r0; r < r1; r++) cursor[r] = 0;
        for (size_t i = 0; i < f[r0].n; i++) {
            const Call *c0 = &f[r0].calls[i];
            if (!needs_barrier(c0)) continue;
            uint64_t lo = c0->t1, hi = window_end(&f[r0], i);
            if (hi < lo) continue;      // overlaps the next call (several threads)
            at[r0] = i;
            bool found = true;
            for (int r = r0 + 1; r < r1 && found; r++) {
                while (cursor[r] < f[r].n && window_end(&f[r], cursor[r]) < c0->t1) cursor[r]++;
                found = false;
                for (size_t j = cursor[r]; j < f[r].n && f[r].calls[j].t1 <= hi; j++) {
                    const Call *c = &f[r].calls[j];
                    if (!needs_barrier(c) || !same_site(c, c0) || window_end(&f[r], j) < lo)
                        continue;
                    if (c->t1 > lo) lo = c->t1;
                    if (window_end(&f[r], j) < hi) hi = window_end(&f[r], j);
                    at[r] = j;
                    found = true;
                    break;
                }
            }
            if (!found) continue;
            place_barrier(f, r0, r1, at, plan);
            for (int r = r0 + 1; r < r1; r++) cursor[r] = at[r] + 1;
        }
    }
}

// =============================================================================
// Worker
// =============================================================================

static void worker_fail(int rank, const char *fmt, ...) __attribute__((format(printf, 2, 3), noreturn));
static void worker_fail(int rank, const char *fmt, ...) {
    va_list ap;
    va_start(ap, fmt);
    fprintf(stderr, "replay: rank %d: ", rank);
    vfprintf(stderr, fmt, ap);
    fprintf(stderr, "\n");
    va_end(ap);
    _exit(3);
}

// Spins on the clock, keeping the core busy as QE's own work does.
static void spin_until(uint64_t target) {
    while (now_ns() < target) {}
}

// Waits for the other size - 1 workers of b; fails after timeout_ns.
static void barrier_wait(Barrier *b, uint32_t size, uint32_t *sense, int rank, uint32_t count,
                         uint64_t timeout_ns) {
    *sense ^= 1;
    if (atomic_fetch_add_explicit(&b->arrived, 1, memory_order_acq_rel) == size - 1) {
        atomic_store_explicit(&b->arrived, 0, memory_order_relaxed);
        atomic_store_explicit(&b->sense, *sense, memory_order_release);
        return;
    }
    uint64_t t0 = now_ns();
    for (uint32_t i = 1; atomic_load_explicit(&b->sense, memory_order_acquire) != *sense; i++)
        if (!(i & 0xfff) && now_ns() - t0 > timeout_ns)
            worker_fail(rank, "barrier %u of its pool: the others did not arrive within %g s",
                        count, sec(timeout_ns));
}

// A host buffer of at least one double, filled with values in [-1, 1)
// (splitmix64), never subnormal.
static double *host_operand(size_t doubles, uint64_t seed, int rank) {
    if (!doubles) doubles = 1;
    double *x = malloc(doubles * sizeof *x);
    if (!x) worker_fail(rank, "cannot allocate %zu MB of operands", doubles * 8 >> 20);
    for (size_t i = 0; i < doubles; i++)
        x[i] = (double)(splitmix(seed + i * 0x9E3779B97F4A7C15ULL) >> 11) * 0x1p-52 - 1.0;
    return x;
}

// nohost: one resident matrix per operand part, A, B and C, real and
// imaginary, reshaped per call by its own memory pool.
enum { ROLE_A, ROLE_B, ROLE_C, ROLES };

typedef struct {
    ABMemoryPool pool[ROLES][2];
    ABMatrix m[ROLES][2];
} Resident;

static void resident_create(Resident *res, const size_t cap[ROLES], bool complex_calls, int rank) {
    for (int role = 0; role < ROLES; role++)
        for (int part = 0; part < (complex_calls ? 2 : 1); part++) {
            int cols = (int)cap[role];
            res->pool[role][part] = ab_pool_create(0);
            res->m[role][part] = ab_pool_get_matrix(res->pool[role][part], 1, cols);
            if (!res->m[role][part])
                worker_fail(rank, "cannot create a resident GPU matrix of %d elements", cols);
            double *x = host_operand(cap[role], 1000 + 2 * (uint64_t)role + (uint64_t)part, rank);
            if (ab_matrix_upload(res->m[role][part], x, true) != AB_OK)
                worker_fail(rank, "cannot upload a resident GPU matrix");
            free(x);
        }
}

// The resident matrix of (role, part) as rows x cols.
static ABMatrix resident_shape(Resident *res, int role, int part, int rows, int cols, int rank) {
    ab_pool_reset(res->pool[role][part]);
    ABMatrix m = ab_pool_get_matrix(res->pool[role][part], rows, cols);
    if (m != res->m[role][part])
        worker_fail(rank, "resident matrix %d.%d cannot take %d x %d", role, part, rows, cols);
    return m;
}

// One call as QE makes it, through the Fortran ABI.
static void abi_call(const Call *c, double *A, double *B, double *C) {
    if (c->routine == 'd')
        fortran_dgemm(&c->ta, &c->tb, &c->m, &c->n, &c->k, &c->alpha[0], A, &c->lda, B, &c->ldb,
                      &c->beta[0], C, &c->ldc);
    else
        fortran_zgemm(&c->ta, &c->tb, &c->m, &c->n, &c->k, (const double complex *)c->alpha,
                      (double complex *)A, &c->lda, (double complex *)B, &c->ldb,
                      (const double complex *)c->beta, (double complex *)C, &c->ldc);
}

// One GPU call on resident operands, traced as the BLAS-ABI layer traces a
// GPU call: the kernel stage as ab_prof_take_kernel_ns() gives it, every
// other stage zero, and the rest of the call in "other", with its split.
static void native_call(const Call *c, Resident *res, int rank) {
    int parts = c->routine == 'z' ? 2 : 1;
    ABMatrix A[2] = {0}, B[2] = {0}, C[2] = {0};
    for (int p = 0; p < parts; p++) {
        A[p] = resident_shape(res, ROLE_A, p, c->m, c->k, rank);
        B[p] = resident_shape(res, ROLE_B, p, c->k, c->n, rank);
        C[p] = resident_shape(res, ROLE_C, p, c->m, c->n, rank);
    }
    uint64_t t0 = now_ns();
    ABStatus st = parts == 1 ? ab_dgemm_scaled(c->alpha[0], A[0], B[0], c->beta[0], C[0])
                             : ab_zgemm(A[0], A[1], B[0], B[1], C[0], C[1]);
    uint64_t elapsed = now_ns() - t0;
    if (st != AB_OK)
        worker_fail(rank, "native %cgemm %d x %d x %d failed: %s", c->routine, c->m, c->n, c->k,
                    ab_status_string(st));

    ab_trace_call_t tc = {.routine = parts == 1 ? "dgemm" : "zgemm", .ta = c->ta, .tb = c->tb,
                          .m = c->m, .n = c->n, .k = c->k,
                          .lda = c->lda, .ldb = c->ldb, .ldc = c->ldc};
    memcpy(tc.alpha, c->alpha, sizeof tc.alpha);
    memcpy(tc.beta, c->beta, sizeof tc.beta);
    uint64_t stage[AB_PROF_STAGE_COUNT] = {0};
    stage[AB_PROF_STAGE_KERNEL] = ab_prof_take_kernel_ns(elapsed, &tc.split);
    if (ab_trace_enabled())
        ab_trace_record(&tc, t0, elapsed, stage, AB_PROF_FB_NONE, AB_PROF_STAGED_NONE);
}

static int worker_main(int fd, int rank) {
    struct stat stt;
    if (fstat(fd, &stt) != 0) return 2;
    Shared *sh = mmap(NULL, (size_t)stt.st_size, PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0);
    if (sh == MAP_FAILED) return 2;
    close(fd);
    if (rank < 0 || rank >= sh->n_ranks) return 2;
    const Mode mode = (Mode)sh->mode;
    const uint32_t pool_size = (uint32_t)sh->pool_size;
    const double gap_scale = sh->gap_scale;
    const uint64_t timeout_ns = sh->timeout_ns;
    const Call *calls = &sh->calls[sh->plan[rank].first];
    const size_t n = sh->plan[rank].n;
    Barrier *pool = &sh->pool[rank / pool_size];

    // REPLAY_TEST_SIGNAL=<rank>:<call>:<KILL|STOP> (test-only): that worker
    // sends itself the signal before that call.
    size_t test_call = SIZE_MAX;
    int test_signal = SIGKILL, test_rank;
    unsigned long long call;
    char sig[8];
    const char *ts = getenv("REPLAY_TEST_SIGNAL");
    if (ts && sscanf(ts, "%d:%llu:%7s", &test_rank, &call, sig) == 3 && test_rank == rank) {
        test_call = (size_t)call;
        if (!strcmp(sig, "STOP")) test_signal = SIGSTOP;
    }

    // Metal set up before the start, not in the first GPU call.
    if (mode != MODE_CPU && ab_init() != AB_OK) worker_fail(rank, "no Metal device");

    // Operands: host buffers for the calls through the ABI, sized for the
    // largest; nohost's resident GPU matrices for the calls AB_MODE=gpu sends
    // to the GPU (this worker runs with AB_MODE=gpu and the trace's
    // AB_MIN_GPU_DIM, so the library's own routing picks them).
    bool *native = calloc(n ? n : 1, sizeof *native);
    size_t host[ROLES] = {0}, cap[ROLES] = {0};
    bool any_native = false, complex_native = false;
    for (size_t i = 0; i < n; i++) {
        const Call *c = &calls[i];
        native[i] = mode == MODE_NOHOST &&
                    ab_internal_blas_routes_gpu(c->routine, c->ta, c->tb, c->m, c->n, c->k,
                                                c->lda, c->ldb, c->ldc,
                                                c->alpha[0] != 0.0 || c->alpha[1] != 0.0);
        if (native[i]) {
            if (ab_matrix_check_size(c->m, c->k) != AB_OK ||
                ab_matrix_check_size(c->k, c->n) != AB_OK ||
                ab_matrix_check_size(c->m, c->n) != AB_OK)
                worker_fail(rank, "call %zu (%cgemm %d x %d x %d) exceeds the GPU matrix "
                            "limits", i, c->routine, c->m, c->n, c->k);
            size_t need[ROLES] = {(size_t)c->m * c->k, (size_t)c->k * c->n, (size_t)c->m * c->n};
            for (int role = 0; role < ROLES; role++)
                if (need[role] > cap[role]) cap[role] = need[role];
            any_native = true;
            complex_native |= c->routine == 'z';
            continue;
        }
        size_t w = c->routine == 'z' ? 2 : 1;
        size_t need[ROLES] = {w * c->lda * (size_t)(is_notrans(c->ta) ? c->k : c->m),
                              w * c->ldb * (size_t)(is_notrans(c->tb) ? c->n : c->k),
                              w * c->ldc * (size_t)c->n};
        for (int role = 0; role < ROLES; role++)
            if (need[role] > host[role]) host[role] = need[role];
    }
    double *op[ROLES];
    for (int role = 0; role < ROLES; role++)
        op[role] = host_operand(host[role], 0x5EED0000ULL + (uint64_t)role * 0x1000000ULL, rank);
    Resident res = {0};
    if (any_native) {
        resident_create(&res, cap, complex_native, rank);
        (void)ab_internal_take_gpu_seconds(NULL);
    }

    // Start together.
    if (atomic_fetch_add(&sh->ready, 1) + 1 == (uint32_t)sh->n_ranks)
        atomic_store(&sh->start_ns, now_ns() + START_DELAY_NS);
    uint64_t start, waited = now_ns();
    while (!(start = atomic_load(&sh->start_ns))) {
        if (now_ns() - waited > timeout_ns)
            worker_fail(rank, "the other workers were not ready within %g s", sec(timeout_ns));
        usleep(100);
    }
    spin_until(start);

    WorkerSummary s = {.pid = (int32_t)getpid()};
    uint32_t sense = 0, barriers = 0;
    uint64_t prev = start;
    for (size_t i = 0; i < n; i++) {
        const Call *c = &calls[i];
        if (i == test_call) raise(test_signal);
        uint64_t s0 = now_ns();
        spin_until(prev + (uint64_t)((double)c->gap * gap_scale));
        uint64_t t0 = now_ns();
        if (native[i]) native_call(c, &res, rank);
        else abi_call(c, op[ROLE_A], op[ROLE_B], op[ROLE_C]);
        uint64_t t1 = now_ns();
        if (native[i]) {
            s.gpu_calls++;
        } else {
            const char *path = ab_get_last_dispatch_path();
            if (!strcmp(path, "none"))
                worker_fail(rank, "dgemm_/zgemm_ did not reach libapplebottom");
            s.gpu_calls += !strcmp(path, "gpu");
        }
        if (i) {
            s.gap_ns += t0 - s0;
        } else {
            s.offset_ns = t0 - s0;
            s.first_ns = t0;
        }
        s.last_ns = prev = t1;
        s.call_ns += t1 - t0;
        if (c->barrier) {
            barrier_wait(pool, pool_size, &sense, rank, ++barriers, timeout_ns);
            prev = now_ns();
            s.barrier_ns += prev - t1;
        }
    }
    sh->sum[rank] = s;
    return 0;
}

// =============================================================================
// Parent
// =============================================================================

// The workers' environment: the caller's with this run's settings.
static char **worker_environ(const char *out, int n, Mode mode, int min_dim) {
    static const char *const DROP[] = {"AB_MODE=", "AB_MIN_GPU_DIM=", "AB_TRACE=",
                                       "AB_TRACE_HASH="};
    size_t ne = 0;
    while (environ[ne]) ne++;
    char **env = calloc(ne + 6, sizeof *env);
    size_t k = 0;
    bool veclib = false, openblas = false;
    for (size_t i = 0; i < ne; i++) {
        bool drop = false;
        for (size_t d = 0; d < sizeof DROP / sizeof *DROP; d++)
            drop |= !strncmp(environ[i], DROP[d], strlen(DROP[d]));
        veclib |= !strncmp(environ[i], "VECLIB_MAXIMUM_THREADS=", 23);
        openblas |= !strncmp(environ[i], "OPENBLAS_NUM_THREADS=", 21);
        if (!drop) env[k++] = environ[i];
    }
    char buf[PATH_MAX + 32];
    env[k++] = mode == MODE_CPU ? "AB_MODE=cpu" : "AB_MODE=gpu";
    snprintf(buf, sizeof buf, "AB_MIN_GPU_DIM=%d", min_dim);
    env[k++] = strdup(buf);
    snprintf(buf, sizeof buf, "AB_TRACE=%s/trace", out);
    env[k++] = strdup(buf);
    if (n > 1 && !veclib) env[k++] = "VECLIB_MAXIMUM_THREADS=1";
    if (n > 1 && !openblas) env[k++] = "OPENBLAS_NUM_THREADS=1";
    return env;
}

// --out, or <trace-dir>/replay_<mode>[_k]: a directory without trace files.
static void out_dir(const char *given, const char *src, Mode mode, char *out, size_t len) {
    if (given) {
        snprintf(out, len, "%s", given);
        if (mkdir(out, 0755) != 0 && errno != EEXIST) die("cannot create %s: %s", out, strerror(errno));
    } else {
        for (int k = 1; ; k++) {
            if (k == 1) snprintf(out, len, "%s/replay_%s", src, MODE_NAMES[mode]);
            else snprintf(out, len, "%s/replay_%s_%d", src, MODE_NAMES[mode], k);
            if (mkdir(out, 0755) == 0) break;
            if (errno != EEXIST) die("cannot create %s: %s", out, strerror(errno));
        }
    }
    DIR *d = opendir(out);
    if (!d) die("cannot open %s: %s", out, strerror(errno));
    for (struct dirent *e; (e = readdir(d)); )
        if (!strncmp(e->d_name, "trace.", 6) || !strcmp(e->d_name, "replay.txt"))
            die("%s already holds a replay (%s); choose another --out", out, e->d_name);
    closedir(d);
    char abs[PATH_MAX];
    if (realpath(out, abs)) snprintf(out, len, "%s", abs);
}

static void usage(const char *argv0) {
    fprintf(stderr,
            "usage: %s <trace-dir> --mode cpu|gpu|nohost [--pools P] [--gap-scale X]\n"
            "       [--no-gaps] [--timeout S] [--out DIR]\n"
            "  <trace-dir>  the AB_TRACE files (trace.<pid>) of one run\n"
            "  --mode       cpu, gpu: every call through dgemm_/zgemm_ in that AB_MODE;\n"
            "               nohost: GPU-routed calls as native GEMMs on resident operands\n"
            "  --pools P    QE's -nk: ranks r/(N/P) form a pool (default 1)\n"
            "  --gap-scale  scale the traced CPU time between calls (default 1)\n"
            "  --no-gaps    no CPU time between calls\n"
            "  --timeout S  longest wait for the other workers (default 300)\n"
            "  --out DIR    output directory (default <trace-dir>/replay_<mode>)\n"
            "See \"Replay\" in benchmarks/qe/README.md.\n", argv0);
    exit(2);
}

int main(int argc, char **argv) {
    if (argc == 4 && !strcmp(argv[1], "--worker")) return worker_main(atoi(argv[2]), atoi(argv[3]));

    const char *src = NULL, *given_out = NULL;
    int mode = -1, pools = 1;
    double gap_scale = 1.0, timeout_s = 300.0;
    for (int i = 1; i < argc; i++) {
        const char *v = i + 1 < argc ? argv[i + 1] : NULL;
        if (!strcmp(argv[i], "--mode") && v) {
            mode = !strcmp(v, "cpu") ? MODE_CPU : !strcmp(v, "gpu") ? MODE_GPU
                 : !strcmp(v, "nohost") ? MODE_NOHOST : -1;
            i++;
        } else if (!strcmp(argv[i], "--pools") && v) pools = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--gap-scale") && v) gap_scale = strtod(argv[++i], NULL);
        else if (!strcmp(argv[i], "--no-gaps")) gap_scale = 0.0;
        else if (!strcmp(argv[i], "--timeout") && v) timeout_s = strtod(argv[++i], NULL);
        else if (!strcmp(argv[i], "--out") && v) given_out = argv[++i];
        else if (argv[i][0] != '-' && !src) src = argv[i];
        else usage(argv[0]);
    }
    if (!src || mode < 0 || pools < 1 || !(gap_scale >= 0.0) || !(timeout_s > 0.0))
        usage(argv[0]);

    TraceFile *f;
    int n = read_trace_dir(src, &f);
    if (n % pools) die("%d ranks cannot form %d pools of equal size", n, pools);
    bool min_dim_traced;
    int min_dim = traced_min_dim(f, n, &min_dim_traced);
    RankPlan plan[MAX_RANKS] = {{0}};
    plan_calls(f, n, pools, plan);

    char src_abs[PATH_MAX], out[PATH_MAX];
    if (!realpath(src, src_abs)) die("%s: %s", src, strerror(errno));
    out_dir(given_out, src_abs, mode, out, sizeof out);

    // The shared memory: a POSIX shared memory object, unlinked at once, whose
    // descriptor the workers inherit (shm_open sets FD_CLOEXEC).
    size_t total = 0;
    for (int r = 0; r < n; r++) total += f[r].n;
    size_t size = sizeof(Shared) + total * sizeof(Call);
    char name[32];
    snprintf(name, sizeof name, "/ab-replay.%d", (int)getpid());
    int fd = shm_open(name, O_RDWR | O_CREAT | O_EXCL, 0600);
    if (fd < 0 || shm_unlink(name) != 0 || ftruncate(fd, (off_t)size) != 0 ||
        fcntl(fd, F_SETFD, 0) != 0)
        die("cannot create %zu bytes of shared memory: %s", size, strerror(errno));
    Shared *sh = mmap(NULL, size, PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0);
    if (sh == MAP_FAILED) die("mmap: %s", strerror(errno));
    sh->n_ranks = n;
    sh->pool_size = n / pools;
    sh->mode = mode;
    sh->gap_scale = gap_scale;
    sh->timeout_ns = (uint64_t)(timeout_s * 1e9);
    for (int r = 0, at = 0; r < n; r++) {
        plan[r].first = (size_t)at;
        plan[r].n = f[r].n;
        memcpy(&sh->calls[at], f[r].calls, f[r].n * sizeof(Call));
        at += (int)f[r].n;
        sh->plan[r] = plan[r];
    }

    printf("replay %s: %s mode, %d rank%s in %d pool%s, gaps %s", src_abs, MODE_NAMES[mode], n,
           n == 1 ? "" : "s", pools, pools == 1 ? "" : "s",
           gap_scale == 0.0 ? "none" : "as traced less pool waits");
    if (gap_scale != 0.0 && gap_scale != 1.0) printf(" x %g", gap_scale);
    printf("\nreplay %s, AB_MIN_GPU_DIM=%d (%s), output %s\n", ab_build_info(), min_dim,
           min_dim_traced ? "as traced" : "default; not in the trace header", out);
    printf(" rank    calls  barriers  skipped  removed_s\n");
    for (int r = 0; r < n; r++)
        printf("%5d %8zu %9u %8u %10.3f\n", r, f[r].n, plan[r].barriers,
               plan[r].candidates - plan[r].barriers, sec(plan[r].removed_ns));
    fflush(stdout);

    char self[PATH_MAX], fd_arg[16];
    uint32_t self_len = sizeof self;
    if (_NSGetExecutablePath(self, &self_len) != 0) die("cannot find this executable");
    snprintf(fd_arg, sizeof fd_arg, "%d", fd);
    // Workers are posix_spawn()ed copies of this binary: a fork()ed child
    // cannot reach MTLCompilerService (tests/test_spawn.h).
    char **env = worker_environ(out, n, mode, min_dim);
    pid_t pids[MAX_RANKS];
    int started = 0;
    for (; started < n; started++) {
        char rank_arg[16];
        snprintf(rank_arg, sizeof rank_arg, "%d", started);
        char *wargv[] = {self, "--worker", fd_arg, rank_arg, NULL};
        int rc = posix_spawn(&pids[started], self, NULL, NULL, wargv, env);
        if (rc != 0) {
            fprintf(stderr, "replay: cannot start the worker of rank %d: %s\n", started,
                    strerror(rc));
            break;
        }
    }
    close(fd);

    // Wait for every worker. The first failure is reported and the workers
    // still running are killed; how they end is not reported. A hung worker
    // is found by the others' timeouts.
    bool failed = started < n;
    for (int r = 0; failed && r < started; r++) kill(pids[r], SIGKILL);
    for (int left = started; left > 0; ) {
        int st, r = 0;
        pid_t p = waitpid(-1, &st, 0);
        if (p < 0 && errno == EINTR) continue;
        if (p < 0) break;
        while (r < started && pids[r] != p) r++;
        if (r == started) continue;
        pids[r] = 0;            // reaped: never signalled again
        left--;
        if (failed || (WIFEXITED(st) && WEXITSTATUS(st) == 0)) continue;
        failed = true;
        if (WIFSIGNALED(st))
            fprintf(stderr, "replay: rank %d (pid %d) died of signal %d\n", r, (int)p, WTERMSIG(st));
        else
            fprintf(stderr, "replay: rank %d (pid %d) failed (exit status %d)\n", r, (int)p,
                    WEXITSTATUS(st));
        for (int q = 0; q < started; q++)
            if (pids[q]) kill(pids[q], SIGKILL);
    }
    if (failed) {
        fprintf(stderr, "replay: stopped; no summary written\n");
        return 1;
    }

    // replay.txt: the settings, then a line per rank, as key=value tokens.
    char path[PATH_MAX + 16];
    snprintf(path, sizeof path, "%s/replay.txt", out);
    FILE *txt = fopen(path, "w");
    if (!txt) die("cannot write %s: %s", path, strerror(errno));
    fprintf(txt, "# benchmarks/replay.c, %s: the BLAS calls of the run in source, replayed\n"
                 "source=%s mode=%s ranks=%d pools=%d gap_scale=%g min_gpu_dim=%d source_mode=%s\n",
            ab_build_info(), src_abs, MODE_NAMES[mode], n, pools, gap_scale, min_dim,
            f[0].mode[0] ? f[0].mode : "unset");
    printf("\n rank     pid   calls  gpu_calls    wall_s   calls_s    gaps_s  barrier_s"
           "   traced_wall_s\n");
    uint64_t wall = 0, calls_ns = 0, gaps_ns = 0, barrier_ns = 0, traced_wall = 0, traced_calls = 0;
    for (int r = 0; r < n; r++) {
        const WorkerSummary *s = &sh->sum[r];
        uint64_t w = s->last_ns - s->first_ns;
        uint64_t tw = f[r].n ? f[r].calls[f[r].n - 1].t1 - f[r].calls[0].t0 : 0;
        for (size_t i = 0; i < f[r].n; i++) traced_calls += f[r].calls[i].t1 - f[r].calls[i].t0;
        if (w > wall) wall = w;
        if (tw > traced_wall) traced_wall = tw;
        calls_ns += s->call_ns;
        gaps_ns += s->gap_ns;
        barrier_ns += s->barrier_ns;
        printf("%5d %7d %7zu %10llu %9.3f %9.3f %9.3f %10.3f %15.3f\n", r, s->pid, f[r].n,
               (unsigned long long)s->gpu_calls, sec(w), sec(s->call_ns), sec(s->gap_ns),
               sec(s->barrier_ns), sec(tw));
        fprintf(txt, "rank=%d pid=%d traced_pid=%d calls=%zu gpu_calls=%llu wall_s=%.6f "
                     "calls_s=%.6f gaps_s=%.6f offset_s=%.6f barrier_s=%.6f barriers=%u "
                     "skipped=%u removed_s=%.6f traced_wall_s=%.6f\n",
                r, s->pid, f[r].pid, f[r].n, (unsigned long long)s->gpu_calls, sec(w),
                sec(s->call_ns), sec(s->gap_ns), sec(s->offset_ns), sec(s->barrier_ns),
                plan[r].barriers, plan[r].candidates - plan[r].barriers, sec(plan[r].removed_ns),
                sec(tw));
    }
    fclose(txt);
    printf("wall_s %.3f (max over ranks), calls_s %.3f, gaps_s %.3f, barrier_s %.3f (sums)\n"
           "traced: wall_s %.3f (max over ranks), calls_s %.3f (sum)\nwrote %s\n",
           sec(wall), sec(calls_ns), sec(gaps_ns), sec(barrier_ns), sec(traced_wall),
           sec(traced_calls), path);
    return 0;
}
