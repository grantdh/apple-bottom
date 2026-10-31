// =============================================================================
// test_trace.c — the AB_TRACE per-call trace and AB_TEST_FAIL_STAGE
// =============================================================================
//
// Each test re-executes this binary as "--child <scenario> <dir>" with the
// environment it needs (test_spawn.h), with stdout and stderr on a pipe, and
// checks what the child printed and the files it left in <dir>. Every AB_*
// variable of the caller's environment, and MTL_DEBUG_LAYER, is dropped first.
// A child still running after CHILD_TIMEOUT_S, or its fork()ed child after
// FORK_WAIT_S, is ended by SIGALRM, which fails the test.
//
//   lines   AB_MODE=gpu, AB_TRACE=<dir>/trace: GPU and CPU calls through the
//           six BLAS-ABI entry points, then a fork()ed child puts a file of
//           its own on the descriptor of the trace file it inherited and
//           repeats a GPU call, which falls back to cblas (no Metal after
//           fork()). One file per pid with a '#' header, one line per call,
//           fields in the documented order that give back the call's
//           arguments exactly, GPU stages and the split of "other" that sum
//           exactly to the call's time, GPUStartTime / GPUEndTime inside the
//           call, the fork()ed child's line after its parent's on the shared
//           clock, and the child's own file intact.
//   hash    AB_TRACE_HASH=1: hashes of exactly the operands each call reads,
//           equal to the FNV-1a over 64-bit words computed here, so equal
//           operands hash equal whatever their leading dimension.
//   off     AB_TRACE unset, then set but empty: no file appears.
//   fail    AB_TEST_FAIL_STAGE=<stage>, AB_REPORT=1, for alloc and kernel:
//           GPU-routed DGEMM and ZGEMM calls fall back; C is bitwise cblas's
//           result, the path record says "cpu", ab_get_stats() shows the GPU
//           work done before the failure, AB_REPORT counts gpu_failed and the
//           trace says fallback:gpu_failed.
// =============================================================================

#include "apple_bottom.h"
#include "test_blas_check.h"
#include "test_registry.h"
#include "test_spawn.h"
#include <Accelerate/Accelerate.h>
#include <complex.h>
#include <dirent.h>
#include <errno.h>
#include <fcntl.h>
#include <limits.h>
#include <mach-o/dyld.h>
#include <math.h>
#include <signal.h>
#include <stdarg.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <sys/wait.h>
#include <unistd.h>

#define CHILD_TIMEOUT_S 120     // a --child still running then is ended by SIGALRM
#define FORK_WAIT_S     30      // and so is the lines scenario's fork()ed child

static int tests_passed = 0;
static int tests_failed = 0;
static char first_failure[512];
static char self_path[PATH_MAX];
static char root_dir[PATH_MAX];      // per-run scratch directory, one subdirectory per test

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

// =============================================================================
// Calls
// =============================================================================

typedef enum { E_DGEMM_SYM, E_DGEMM_F, E_DGEMM_BLAS, E_ZGEMM_SYM, E_ZGEMM_F, E_ZGEMM_BLAS } Entry;

static bool entry_complex(Entry e) { return e >= E_ZGEMM_SYM; }

typedef struct {
    Entry entry;
    char ta, tb;                    // passed as is (lower case too)
    int M, N, K, pad;               // each ld is the stored rows + pad
    double complex alpha, beta;     // real entries use the real parts
    unsigned seed;                  // operands A, B, C: seeds 3*seed, +1, +2
    bool tweak_a;                   // add 1 to A(0,0)
    const char *path;               // where AB_MODE=gpu runs it
} Call;

// In AB_MODE=gpu: dimensions <= AB_MIN_GPU_DIM (32) and alpha == 0 stay on cblas.
static const Call LINE_CALLS[] = {
    {E_DGEMM_SYM,  'N', 'T', 96, 80, 72, 3, CMPLX(1.25, 0.0),    CMPLX(-0.75, 0.0),  1, false, "gpu"},
    {E_ZGEMM_BLAS, 'C', 'N', 96, 80, 72, 2, CMPLX(1.25, -0.5),   CMPLX(-0.75, 0.25), 2, false, "gpu"},
    {E_DGEMM_F,    'N', 'N', 16, 16, 16, 0, CMPLX(1.0, 0.0),     CMPLX(0.0, 0.0),    3, false, "cpu"},
    {E_ZGEMM_SYM,  'N', 'N', 96, 80, 72, 0, CMPLX(0.0, 0.0),     CMPLX(0.5, -0.25),  4, false, "cpu"},
    {E_ZGEMM_F,    't', 'c', 64, 48, 40, 1, CMPLX(1.0, 0.0),     CMPLX(0.0, 0.0),    5, false, "gpu"},
    {E_DGEMM_BLAS, 'T', 'n', 72, 96, 80, 5, CMPLX(1.0 / 3, 0.0), CMPLX(1.0, 0.0),    6, false, "gpu"},
};

// Calls 0 and 1 read the same op(A) and op(B), stored with different leading
// dimensions; call 2 changes one element of A.
static const Call HASH_CALLS[] = {
    {E_DGEMM_SYM,  'N', 'N', 96, 80, 72, 0, CMPLX(1.25, 0.0),  CMPLX(0.0, 0.0),   1, false, "gpu"},
    {E_DGEMM_BLAS, 'N', 'N', 96, 80, 72, 7, CMPLX(1.25, 0.0),  CMPLX(0.5, 0.0),   1, false, "gpu"},
    {E_DGEMM_F,    'N', 'N', 96, 80, 72, 0, CMPLX(1.25, 0.0),  CMPLX(0.0, 0.0),   1, true,  "gpu"},
    {E_ZGEMM_SYM,  'C', 'T', 64, 48, 40, 3, CMPLX(1.0, -0.5),  CMPLX(0.0, 0.0),   2, false, "gpu"},
    {E_ZGEMM_BLAS, 'N', 'N', 64, 48, 40, 0, CMPLX(0.0, 0.0),   CMPLX(1.0, 0.0),   3, false, "cpu"},
    {E_DGEMM_F,    'T', 'N', 16, 16, 16, 2, CMPLX(1.0, 0.0),   CMPLX(-1.0, 0.0),  4, false, "cpu"},
};

// The GPU-routed calls the fail scenario makes: LINE_CALLS[0] and [1].
#define N_FAIL_CALLS 2

// The GPU steps AB_TEST_FAIL_STAGE can fail, in the order the GPU path reaches them.
static const char *const FAIL_STAGES[] = {"alloc", "kernel"};

// Storage of A, B and C.
static void layouts(const Call *c, Layout l[3]) {
    l[0] = stored(c->ta, c->M, c->K, c->pad);
    l[1] = stored(c->tb, c->K, c->N, c->pad);
    l[2] = stored('N', c->M, c->N, c->pad);
}

// Part p (0 real, 1 imaginary) of element (i, j) of operand `seed`, in
// [-1, 1). It does not depend on the leading dimension.
static double value(unsigned seed, int i, int j, int p, bool tweak) {
    uint64_t h = splitmix(((uint64_t)seed << 42) ^ ((uint64_t)j << 21) ^ ((uint64_t)i << 1) ^
                          (uint64_t)p);
    double v = (double)(h >> 11) * 0x1p-52 - 1.0;
    return tweak && i == 0 && j == 0 && p == 0 ? v + 1.0 : v;
}

// A stored operand: l.rows x l.cols values, column-major with leading
// dimension l.ld; the padding rows hold NaN, which a call must neither read
// nor hash.
static double *fill(unsigned seed, Layout l, bool z, bool tweak) {
    size_t parts = z ? 2 : 1, n = nelem(l) * parts;
    double *x = malloc(n * sizeof *x);
    for (int j = 0; j < l.cols; j++)
        for (int i = 0; i < l.ld; i++)
            for (size_t p = 0; p < parts; p++)
                x[((size_t)j * l.ld + i) * parts + p] =
                    i < l.rows ? value(seed, i, j, (int)p, tweak) : NAN;
    return x;
}

// The trace's operand hash, from value() rather than from memory: FNV-1a over
// the 64-bit words of the elements (a complex one is two, real then
// imaginary), column by column.
static uint64_t fnv_operand(unsigned seed, Layout l, bool z, bool tweak) {
    uint64_t h = FNV_OFFSET;
    for (int j = 0; j < l.cols; j++)
        for (int i = 0; i < l.rows; i++)
            for (int p = 0; p < (z ? 2 : 1); p++) {
                double v = value(seed, i, j, p, tweak);
                uint64_t word;
                memcpy(&word, &v, sizeof word);
                h = (h ^ word) * FNV_PRIME;
            }
    return h;
}

typedef struct {
    double *X[3];                   // A, B, C
    Layout l[3];
    size_t c_bytes;
} Operands;

static void make_operands(const Call *c, Operands *o) {
    bool z = entry_complex(c->entry);
    layouts(c, o->l);
    for (int x = 0; x < 3; x++)
        o->X[x] = fill(3 * c->seed + (unsigned)x, o->l[x], z, x == 0 && c->tweak_a);
    o->c_bytes = nelem(o->l[2]) * (z ? sizeof(double complex) : sizeof(double));
}

static void free_operands(Operands *o) {
    for (int x = 0; x < 3; x++) free(o->X[x]);
}

// Runs c through its entry point on o.
static void run_entry(const Call *c, Operands *o) {
    int M = c->M, N = c->N, K = c->K, lda = o->l[0].ld, ldb = o->l[1].ld, ldc = o->l[2].ld;
    double ar = creal(c->alpha), br = creal(c->beta);
    double complex za = c->alpha, zb = c->beta;
    double *a = o->X[0], *b = o->X[1], *cc = o->X[2];
    double complex *A = (double complex *)a, *B = (double complex *)b, *C = (double complex *)cc;
    switch (c->entry) {
    case E_DGEMM_SYM:  fortran_dgemm(&c->ta, &c->tb, &M, &N, &K, &ar, a, &lda, b, &ldb, &br, cc, &ldc); break;
    case E_DGEMM_F:    ab_dgemm_(&c->ta, &c->tb, &M, &N, &K, &ar, a, &lda, b, &ldb, &br, cc, &ldc); break;
    case E_DGEMM_BLAS: ab_dgemm_blas(c->ta, c->tb, M, N, K, ar, a, lda, b, ldb, br, cc, ldc); break;
    case E_ZGEMM_SYM:  fortran_zgemm(&c->ta, &c->tb, &M, &N, &K, &za, A, &lda, B, &ldb, &zb, C, &ldc); break;
    case E_ZGEMM_F:    ab_zgemm_(&c->ta, &c->tb, &M, &N, &K, &za, A, &lda, B, &ldb, &zb, C, &ldc); break;
    case E_ZGEMM_BLAS: ab_zgemm_blas(c->ta, c->tb, M, N, K, za, A, lda, B, ldb, zb, C, ldc); break;
    }
}

// =============================================================================
// Child side: --child <scenario> <dir>
// =============================================================================

// Runs c on fresh operands; prints "call <tag> path=<path record>".
static void child_call(const char *tag, const Call *c) {
    Operands o;
    make_operands(c, &o);
    run_entry(c, &o);
    printf("call %s path=%s\n", tag, ab_get_last_dispatch_path());
    free_operands(&o);
}

static void child_calls(const Call *calls, size_t n) {
    for (size_t i = 0; i < n; i++) {
        char tag[16];
        snprintf(tag, sizeof tag, "%zu", i);
        child_call(tag, &calls[i]);
    }
}

// Each fail-scenario call against cblas on a copy of C; prints
// "call <i> path=<path record> bitwise=<0|1> converted=<n> gemms=<n>", the
// last two what ab_get_stats() counted during the call.
static void child_fail_calls(void) {
    for (int i = 0; i < N_FAIL_CALLS; i++) {
        const Call *c = &LINE_CALLS[i];
        bool z = entry_complex(c->entry);
        Operands o;
        make_operands(c, &o);
        double *ref = malloc(o.c_bytes);
        memcpy(ref, o.X[2], o.c_bytes);
        enum CBLAS_TRANSPOSE ta = ref_trans(c->ta, z), tb = ref_trans(c->tb, z);
        if (z) {
            double complex alpha = c->alpha, beta = c->beta;
            cblas_zgemm(CblasColMajor, ta, tb, c->M, c->N, c->K, &alpha,
                        (double complex *)o.X[0], o.l[0].ld, (double complex *)o.X[1], o.l[1].ld,
                        &beta, (double complex *)ref, o.l[2].ld);
        } else {
            cblas_dgemm(CblasColMajor, ta, tb, c->M, c->N, c->K, creal(c->alpha), o.X[0],
                        o.l[0].ld, o.X[1], o.l[1].ld, creal(c->beta), ref, o.l[2].ld);
        }
        ABStats s0 = ab_get_stats();
        run_entry(c, &o);
        ABStats s1 = ab_get_stats();
        printf("call %d path=%s bitwise=%d converted=%llu gemms=%llu\n", i,
               ab_get_last_dispatch_path(), memcmp(ref, o.X[2], o.c_bytes) == 0,
               (unsigned long long)(s1.elements_converted - s0.elements_converted),
               (unsigned long long)(s1.dgemm_count + s1.zgemm_count - s0.dgemm_count -
                                    s0.zgemm_count));
        free(ref);
        free_operands(&o);
    }
}

// The descriptor this process's trace file is open on, found by its path;
// -1 if there is none.
static int trace_fd(void) {
    char suffix[32], path[PATH_MAX];
    snprintf(suffix, sizeof suffix, "/trace.%d", (int)getpid());
    size_t ls = strlen(suffix), lp;
    for (int fd = 3; fd < 256; fd++)
        if (fcntl(fd, F_GETPATH, path) == 0 && (lp = strlen(path)) >= ls &&
            !strcmp(path + lp - ls, suffix))
            return fd;
    return -1;
}

// Creates path on descriptor fd, as a program that closes and reopens its
// descriptors may; returns fd, or -1.
static int open_on(const char *path, int fd) {
    int data = fd < 0 ? -1 : open(path, O_WRONLY | O_CREAT | O_TRUNC, 0644);
    if (data < 0 || data == fd) return data;
    int got = dup2(data, fd);
    close(data);
    return got;
}

static int child_main(const char *scenario, const char *dir) {
    if (!strcmp(scenario, "lines")) {
        child_calls(LINE_CALLS, TEST_REGISTRY_SIZE(LINE_CALLS));
        int fd = trace_fd();
        fflush(NULL);
        pid_t pid = fork();
        if (pid == 0) {
            alarm(FORK_WAIT_S);         // fork() cleared the CHILD_TIMEOUT_S alarm
            // A file of the child's own on the descriptor number of the trace
            // file it inherited: the library must not close it or write to it.
            char path[PATH_MAX];
            snprintf(path, sizeof path, "%s/fork.data", dir);
            int data = open_on(path, fd);
            if (data < 0) printf("fork: no trace descriptor (%d) to reuse\n", fd);
            child_call("fork", &LINE_CALLS[0]);
            if (data >= 0) (void)write(data, "fork data\n", 10);
            fflush(NULL);
            _exit(data >= 0 ? 0 : 3);
        }
        int status = -1;
        if (pid > 0)
            while (waitpid(pid, &status, 0) < 0 && errno == EINTR) {}
        printf("fork %d %d\n", (int)pid, status);
        return 0;
    }
    if (!strcmp(scenario, "hash")) {
        child_calls(HASH_CALLS, TEST_REGISTRY_SIZE(HASH_CALLS));
        return 0;
    }
    if (!strcmp(scenario, "off")) {
        if (chdir(dir) != 0) return 2;
        child_calls(LINE_CALLS, 3);
        return 0;
    }
    if (!strcmp(scenario, "fail")) {
        child_fail_calls();
        return 0;
    }
    return 2;
}

// =============================================================================
// Parent side: running a child, reading its files
// =============================================================================

typedef struct {
    pid_t pid;
    int status;
    char *out;                      // its stdout and stderr
} ChildRun;

// Runs this binary as --child scenario dir, with the caller's environment
// less the variables child_environ() drops, plus env_add (NULL-terminated).
static bool run_child(const char *scenario, const char *dir, const char *const *env_add,
                      ChildRun *r) {
    size_t na = 0, k;
    while (env_add[na]) na++;
    char **env = child_environ(na, &k);
    for (size_t i = 0; i < na; i++) env[k++] = (char *)env_add[i];
    char *argv[] = {self_path, "--child", (char *)scenario, (char *)dir, NULL}, why[256];
    FILE *in = spawn_piped(argv, env, &r->pid, why, sizeof why);
    free(env);
    if (!in) return expect(false, "%s", why);
    r->out = slurp(in);
    while (waitpid(r->pid, &r->status, 0) < 0 && errno == EINTR) {}
    bool timed_out = WIFSIGNALED(r->status) && WTERMSIG(r->status) == SIGALRM;
    return expect(WIFEXITED(r->status) && WEXITSTATUS(r->status) == 0,
                  "child '%s' failed (status %d%s): %.200s", scenario, r->status,
                  timed_out ? ", timed out" : "", r->out);
}

// The first line of out that starts with prefix, copied to buf without its
// newline; NULL if there is none.
static const char *find_line(const char *out, const char *prefix, char *buf, size_t n) {
    size_t lp = strlen(prefix);
    for (const char *p = out; p && *p; p = strchr(p, '\n'), p = p ? p + 1 : NULL)
        if (!strncmp(p, prefix, lp)) {
            snprintf(buf, n, "%.*s", (int)strcspn(p, "\n"), p);
            return buf;
        }
    return NULL;
}

// The path record a child printed for call <tag> ("call <tag> path=<path>"),
// or "" if none.
static const char *printed_path(const char *out, const char *tag, char path[32]) {
    char want[32], line[256];
    snprintf(want, sizeof want, "call %s path=", tag);
    path[0] = '\0';
    if (find_line(out, want, line, sizeof line)) sscanf(line + strlen(want), "%31[^ ]", path);
    return path;
}

static bool printed_paths_match(const char *out, const Call *calls, size_t n) {
    for (size_t i = 0; i < n; i++) {
        char tag[16], path[32];
        snprintf(tag, sizeof tag, "%zu", i);
        if (!expect(!strcmp(printed_path(out, tag, path), calls[i].path),
                    "call %zu: path record '%s', expected '%s'", i, path, calls[i].path))
            return false;
    }
    return true;
}

// Fresh scratch directory for one test.
static void test_dir(const char *name, char *dir, size_t n) {
    snprintf(dir, n, "%s/%s", root_dir, name);
    mkdir(dir, 0755);
}

// Names of the files in dir; returns how many (at most max).
static int list_dir(const char *dir, char names[][NAME_MAX + 1], int max) {
    int n = 0;
    DIR *d = opendir(dir);
    if (!d) return -1;
    for (struct dirent *e; (e = readdir(d)); )
        if (strcmp(e->d_name, ".") && strcmp(e->d_name, "..") && n < max)
            snprintf(names[n++], NAME_MAX + 1, "%s", e->d_name);
    closedir(d);
    return n;
}

// Removes dir and its files, if the test passed; keeps them for a failure.
static void clean_dir(const char *dir) {
    if (first_failure[0]) {
        size_t n = strlen(first_failure);
        snprintf(first_failure + n, sizeof first_failure - n, " [files in %s]", dir);
        return;
    }
    char names[16][NAME_MAX + 1], path[PATH_MAX];
    int n = list_dir(dir, names, 16);
    for (int i = 0; i < n; i++) {
        snprintf(path, sizeof path, "%s/%s", dir, names[i]);
        unlink(path);
    }
    rmdir(dir);
}

static char *read_file(const char *path) {
    FILE *f = fopen(path, "r");
    return f ? slurp(f) : NULL;
}

// =============================================================================
// Parent side: trace lines
// =============================================================================

static const char *const BASE_KEYS[] = {
    "seq", "t0_ns", "t1_ns", "routine", "ta", "tb", "M", "N", "K", "lda", "ldb", "ldc",
    "alpha", "beta", "path",
};
// The stages (the AB_PROFILE order, then other), the split of other, the GPU times.
static const char *const GPU_KEYS[] = {
    "init_ns", "alloc_ns", "pack_ns", "upload_ns", "kernel_ns", "download_ns", "epilogue_ns",
    "other_ns", "temps_ns", "encode_ns", "submit_ns", "wake_ns", "rest_ns", "gpu_start_ns",
    "gpu_end_ns",
};
static const char *const HASH_KEYS[] = {"hash_a", "hash_b", "hash_c"};

#define N_BASE TEST_REGISTRY_SIZE(BASE_KEYS)
#define N_GPU  TEST_REGISTRY_SIZE(GPU_KEYS)

typedef struct {
    char val[N_BASE + N_GPU + 3][96];   // in key order: base, GPU (if any), hashes
    bool gpu;
    char staged[64];                    // the staged field, "" if none
    bool hashed[3];
    unsigned long long hash[3];
} TraceLine;

// Splits a line into key=value fields and checks that the keys are exactly
// the documented ones in order: the base keys, the GPU keys on a gpu line,
// staged on a call that reached the staged path, then any hashes.
static bool parse_trace_line(const char *line, TraceLine *t) {
    memset(t, 0, sizeof *t);
    char copy[2048], *save = NULL;
    snprintf(copy, sizeof copy, "%s", line);
    const char *keys[N_BASE + N_GPU + 4];
    char *vals[N_BASE + N_GPU + 4];
    size_t n = 0;
    for (char *tok = strtok_r(copy, " ", &save); tok; tok = strtok_r(NULL, " ", &save)) {
        char *eq = strchr(tok, '=');
        if (!eq || n == TEST_REGISTRY_SIZE(keys))
            return expect(false, "bad field '%s' in: %s", tok, line);
        *eq = '\0';
        keys[n] = tok;
        vals[n++] = eq + 1;
    }
    size_t i = 0;
    for (; i < N_BASE; i++)
        if (i >= n || strcmp(keys[i], BASE_KEYS[i]))
            return expect(false, "field %zu is not %s in: %s", i, BASE_KEYS[i], line);
    t->gpu = !strcmp(vals[N_BASE - 1], "gpu");
    if (t->gpu)
        for (; i < N_BASE + N_GPU; i++)
            if (i >= n || strcmp(keys[i], GPU_KEYS[i - N_BASE]))
                return expect(false, "field %zu is not %s in: %s", i, GPU_KEYS[i - N_BASE], line);
    size_t fixed = i;
    if (i < n && !strcmp(keys[i], "staged")) snprintf(t->staged, sizeof t->staged, "%s", vals[i++]);
    for (int h = 0; h < 3 && i < n; h++)
        if (!strcmp(keys[i], HASH_KEYS[h])) {
            char *end;
            t->hash[h] = strtoull(vals[i], &end, 16);
            t->hashed[h] = strlen(vals[i]) == 16 && !*end;
            if (!expect(t->hashed[h], "bad %s in: %s", HASH_KEYS[h], line)) return false;
            i++;
        }
    if (i != n) return expect(false, "unexpected field %s in: %s", keys[i], line);
    for (size_t f = 0; f < fixed; f++) snprintf(t->val[f], sizeof t->val[f], "%s", vals[f]);
    return true;
}

static const char *get(const TraceLine *t, const char *key) {
    for (size_t i = 0; i < N_BASE; i++)
        if (!strcmp(key, BASE_KEYS[i])) return t->val[i];
    for (size_t i = 0; i < N_GPU; i++)
        if (!strcmp(key, GPU_KEYS[i])) return t->gpu ? t->val[N_BASE + i] : "";
    return "";
}

static bool get_u64(const TraceLine *t, const char *key, unsigned long long *v) {
    const char *s = get(t, key);
    char *end;
    errno = 0;
    *v = strtoull(s, &end, 10);
    return expect(*s >= '0' && *s <= '9' && !*end && !errno, "%s=%s is not a count", key, s);
}

static bool get_int(const TraceLine *t, const char *key, int want) {
    const char *s = get(t, key);
    char *end;
    long v = strtol(s, &end, 10);
    return expect(*s && !*end && v == want, "%s=%s, expected %d", key, s, want);
}

// alpha / beta: one %a value, or real,imaginary for zgemm; exact.
static bool get_scalar(const TraceLine *t, const char *key, double complex want, bool z) {
    const char *s = get(t, key);
    char *end;
    double re = strtod(s, &end), im = 0.0;
    bool parsed = z ? *end == ',' : !*end;
    if (z && parsed) {
        im = strtod(end + 1, &end);
        parsed = !*end;
    }
    double wr = creal(want), wi = cimag(want);
    return expect(parsed && !memcmp(&re, &wr, sizeof re) && !memcmp(&im, &wi, sizeof im),
                  "%s=%s, expected %a%s%a", key, s, wr, z ? "," : "", z ? wi : 0.0);
}

// Whether the children run with AB_HOST_PACK=1 (the host path, whose ZGEMM
// times its GPU temporaries in temps_ns) rather than on the staged path.
static bool g_host_pack;

// Under AB_TEST_FAIL_STAGE, the staged field of the calls that fall back:
// the staged path fails first, at the same step.
static const char *g_fail_staged = "";

// One line against the call it records: the arguments exactly, the path, and
// for a GPU call stages that sum to the call's time and a split that sums to
// other_ns, with the GPU's start and end inside the call.
static bool check_line(const TraceLine *t, const Call *c, const char *path,
                       unsigned long long *t0, unsigned long long *t1) {
    bool z = entry_complex(c->entry);
    Layout l[3];
    layouts(c, l);
    char ta[2] = {c->ta, 0}, tb[2] = {c->tb, 0};
    if (!expect(!strcmp(get(t, "routine"), z ? "zgemm" : "dgemm"), "routine=%s", get(t, "routine")) ||
        !expect(!strcmp(get(t, "ta"), ta) && !strcmp(get(t, "tb"), tb), "ta=%s tb=%s, expected %s %s",
                get(t, "ta"), get(t, "tb"), ta, tb) ||
        !get_int(t, "M", c->M) || !get_int(t, "N", c->N) || !get_int(t, "K", c->K) ||
        !get_int(t, "lda", l[0].ld) || !get_int(t, "ldb", l[1].ld) || !get_int(t, "ldc", l[2].ld) ||
        !get_scalar(t, "alpha", c->alpha, z) || !get_scalar(t, "beta", c->beta, z) ||
        !expect(!strcmp(get(t, "path"), path), "path=%s, expected %s", get(t, "path"), path) ||
        !get_u64(t, "t0_ns", t0) || !get_u64(t, "t1_ns", t1) ||
        !expect(*t0 > 0 && *t0 <= *t1, "t0_ns=%llu t1_ns=%llu", *t0, *t1))
        return false;
    // These operands need no scaling and suit Gauss-3M: a GPU call keeps the
    // staged result unless AB_HOST_PACK=1; a call the routing keeps on cblas,
    // or one without Metal, never reaches the staged path.
    const char *staged = !strcmp(path, "gpu") ? (g_host_pack ? "host_pack" : "kept")
                       : !strcmp(path, "fallback:gpu_failed") ? g_fail_staged : "";
    if (!expect(!strcmp(t->staged, staged), "staged=%s, expected '%s'", t->staged, staged))
        return false;
    if (!t->gpu) return true;

    // The stages are the GPU keys before other_ns.
    unsigned long long named = 0, v, other, temps, encode, submit, wake, kernel, gs, ge;
    for (size_t i = 0; strcmp(GPU_KEYS[i], "other_ns"); i++) {
        if (!get_u64(t, GPU_KEYS[i], &v)) return false;
        named += v;
    }
    const char *rest_s = get(t, "rest_ns");
    char *end;
    long long rest = strtoll(rest_s, &end, 10);
    if (!get_u64(t, "other_ns", &other) || !get_u64(t, "temps_ns", &temps) ||
        !get_u64(t, "encode_ns", &encode) || !get_u64(t, "submit_ns", &submit) ||
        !get_u64(t, "wake_ns", &wake) || !get_u64(t, "kernel_ns", &kernel) ||
        !get_u64(t, "gpu_start_ns", &gs) || !get_u64(t, "gpu_end_ns", &ge) ||
        !expect(*rest_s && !*end, "rest_ns=%s", rest_s))
        return false;
    return expect(named + other == *t1 - *t0,
                  "stages %llu + other %llu != t1_ns - t0_ns %llu", named, other, *t1 - *t0) &&
           expect((long long)(temps + encode + submit + wake) + rest == (long long)other,
                  "temps+encode+submit+wake+rest != other_ns %llu", other) &&
           expect(kernel > 0, "no kernel time") &&
           expect(encode > 0, "no encode time") &&
           expect(g_host_pack ? temps > 0 || !z : temps == 0,
                  "temps_ns=%llu for %s (only the host path's ZGEMM times GPU temporaries)",
                  temps, get(t, "routine")) &&
           expect(gs > 0 && *t0 <= gs && gs <= ge && ge <= *t1,
                  "GPU start %llu / end %llu outside the call [%llu, %llu]", gs, ge, *t0, *t1) &&
           expect(ge - gs <= kernel + 1000 && kernel <= ge - gs + 1000,
                  "kernel_ns %llu is not GPU end - start %llu", kernel, ge - gs);
}

// Checks the trace file of one process: the header, then one line per call
// in seq order, each matching its call and starting after the one before.
// paths[i] overrides calls[i].path when not NULL. Fills lines[], the first
// line's t0_ns and the last line's t1_ns, each unless NULL, and returns how
// many lines there are, or -1 after a failed expectation.
static int check_file(const char *dir, pid_t pid, const Call *calls, const char *const *paths,
                      size_t n, bool hashes, TraceLine *lines, unsigned long long *first_t0,
                      unsigned long long *last_t1) {
    char path[PATH_MAX];
    snprintf(path, sizeof path, "%s/trace.%d", dir, (int)pid);
    char *s = read_file(path);
    if (!expect(s != NULL, "no trace file for pid %d", (int)pid)) return -1;

    char head[512], mode[64];
    snprintf(head, sizeof head, "# %s AB_TRACE pid=%d", ab_build_info(), (int)pid);
    // The children run without AB_MIN_GPU_DIM: the routing's default, 32.
    snprintf(mode, sizeof mode, "# AB_MODE=gpu AB_TRACE_HASH=%d AB_MIN_GPU_DIM=32", hashes);
    int nhead = 0, nlines = 0;
    bool legend = false, rank = false, ok = true;
    unsigned long long prev_t1 = 0;
    TraceLine scratch;
    char *save = NULL;
    for (char *line = strtok_r(s, "\n", &save); line && ok; line = strtok_r(NULL, "\n", &save)) {
        if (line[0] == '#') {
            const char *want = nhead == 0 ? head : nhead == 1 ? mode : line;
            ok = expect(nlines == 0, "header line after a call line: %s", line) &&
                 expect(!strcmp(line, want), "header '%s', expected '%s'", line, want);
            legend = legend || strstr(line, "# line: seq t0_ns t1_ns routine ta tb M N K "
                                            "lda ldb ldc alpha beta path");
            rank = rank || !strncmp(line, "# mpi_rank=", 11);
            nhead++;
            continue;
        }
        ok = expect(nhead >= 2 && legend && rank,
                    "no header, mpi_rank line or legend before the first line");
        if (!ok || !expect((size_t)nlines < n, "more lines than calls (%zu)", n)) {
            ok = false;
            break;
        }
        TraceLine *t = lines ? &lines[nlines] : &scratch;
        unsigned long long seq = 0, t0 = 0, t1 = 0;
        const Call *c = &calls[nlines];
        ok = parse_trace_line(line, t) && get_u64(t, "seq", &seq) &&
             expect(seq == (unsigned long long)nlines + 1, "seq=%llu on line %d", seq, nlines + 1) &&
             check_line(t, c, paths && paths[nlines] ? paths[nlines] : c->path, &t0, &t1) &&
             expect(t0 >= prev_t1, "line %d starts before the previous call ended", nlines + 1);
        if (nlines == 0 && first_t0) *first_t0 = t0;
        if (last_t1) *last_t1 = t1;
        prev_t1 = t1;
        nlines++;
    }
    free(s);
    if (ok) ok = expect((size_t)nlines == n, "%d lines for %zu calls", nlines, n);
    return ok ? nlines : -1;
}

// =============================================================================
// Tests
// =============================================================================

// AB_MODE=gpu, AB_TRACE: GPU and CPU calls, then a fork()ed child's fallback
// with a file of its own on the inherited trace file's descriptor; on the
// staged path, then on the host path (AB_HOST_PACK=1).
static void trace_lines(void) {
    char dir[PATH_MAX], trace[PATH_MAX + 16], path[32], data_path[PATH_MAX];
    char names[8][NAME_MAX + 1], want[3][64];
    test_dir("lines", dir, sizeof dir);
    snprintf(trace, sizeof trace, "AB_TRACE=%s/trace", dir);
    const char *env[] = {"AB_MODE=gpu", trace, g_host_pack ? "AB_HOST_PACK=1" : NULL, NULL};
    const char *fork_path[] = {"fallback:no_device"};
    const size_t n = TEST_REGISTRY_SIZE(LINE_CALLS);
    ChildRun r = {0};
    const char *f;
    char *data = NULL;
    int fork_pid = 0, fork_status = -1, nf, found = 0;
    unsigned long long t1 = 0, fork_t0 = 0;

    if (!run_child("lines", dir, env, &r) || !printed_paths_match(r.out, LINE_CALLS, n))
        goto out;
    f = strstr(r.out, "\nfork ");
    if (!expect(f && sscanf(f, "\nfork %d %d", &fork_pid, &fork_status) == 2 && fork_pid > 0 &&
                    WIFEXITED(fork_status) && WEXITSTATUS(fork_status) == 0,
                "fork()ed child failed: %s", r.out))
        goto out;
    if (!expect(!strcmp(printed_path(r.out, "fork", path), "cpu"),
                "fork()ed child: path record '%s', expected 'cpu'", path))
        goto out;

    snprintf(want[0], sizeof want[0], "trace.%d", (int)r.pid);
    snprintf(want[1], sizeof want[1], "trace.%d", fork_pid);
    snprintf(want[2], sizeof want[2], "fork.data");
    nf = list_dir(dir, names, 8);
    for (int i = 0; i < nf; i++)
        for (int w = 0; w < 3; w++) found += !strcmp(names[i], want[w]);
    if (!expect(nf == 3 && found == 3, "expected exactly %s, %s and %s in %s (%d files)",
                want[0], want[1], want[2], dir, nf))
        goto out;
    snprintf(data_path, sizeof data_path, "%s/fork.data", dir);
    data = read_file(data_path);
    if (!expect(data && !strcmp(data, "fork data\n"), "the fork()ed child's file on the "
                "inherited trace descriptor holds '%s', not its write", data ? data : ""))
        goto out;
    if (check_file(dir, r.pid, LINE_CALLS, NULL, n, false, NULL, NULL, &t1) < 0 ||
        check_file(dir, fork_pid, LINE_CALLS, fork_path, 1, false, NULL, &fork_t0, NULL) < 0)
        goto out;
    expect(fork_t0 >= t1, "the fork()ed child's call starts before its parent's last call "
                          "ended (%llu < %llu)", fork_t0, t1);
out:
    free(data);
    free(r.out);
    clean_dir(dir);
}

static void test_trace_lines(void) {
    begin();
    for (int host = 0; host < 2 && !first_failure[0]; host++) {
        g_host_pack = host;
        trace_lines();
    }
    g_host_pack = false;
    finish();
}

// AB_TRACE_HASH=1: the hashes of what each call reads.
static void test_trace_hash(void) {
    begin();
    char dir[PATH_MAX], trace[PATH_MAX + 16];
    test_dir("hash", dir, sizeof dir);
    snprintf(trace, sizeof trace, "AB_TRACE=%s/trace", dir);
    const char *env[] = {"AB_MODE=gpu", "AB_TRACE_HASH=1", trace, NULL};
    ChildRun r = {0};
    const size_t n = TEST_REGISTRY_SIZE(HASH_CALLS);
    TraceLine lines[TEST_REGISTRY_SIZE(HASH_CALLS)];
    if (run_child("hash", dir, env, &r) && printed_paths_match(r.out, HASH_CALLS, n) &&
        check_file(dir, r.pid, HASH_CALLS, NULL, n, true, lines, NULL, NULL) >= 0) {
        for (size_t i = 0; i < n && !first_failure[0]; i++) {
            const Call *c = &HASH_CALLS[i];
            bool z = entry_complex(c->entry);
            Layout l[3];
            layouts(c, l);
            bool reads[3] = {c->alpha != 0.0, c->alpha != 0.0, c->beta != 0.0};
            for (int h = 0; h < 3; h++) {
                uint64_t want = reads[h] ? fnv_operand(3 * c->seed + (unsigned)h, l[h], z,
                                                       h == 0 && c->tweak_a) : 0;
                if (!expect(lines[i].hashed[h] == reads[h], "call %zu: %s %s", i, HASH_KEYS[h],
                            reads[h] ? "missing" : "for an operand the call does not read") ||
                    !expect(!reads[h] || lines[i].hash[h] == want, "call %zu: %s=%016llx, expected %016llx",
                            i, HASH_KEYS[h], lines[i].hash[h], (unsigned long long)want))
                    break;
            }
        }
        expect(lines[0].hash[0] == lines[1].hash[0] && lines[0].hash[1] == lines[1].hash[1],
               "equal operands stored with other leading dimensions hash differently");
        expect(lines[2].hash[0] != lines[0].hash[0] && lines[2].hash[1] == lines[0].hash[1],
               "changing one element of A must change hash_a only");
    }
    free(r.out);
    clean_dir(dir);
    finish();
}

// No AB_TRACE, or an empty one: no file, whatever else is set.
static void test_trace_off(void) {
    begin();
    char dir[PATH_MAX];
    test_dir("off", dir, sizeof dir);
    const char *unset[] = {"AB_MODE=gpu", "AB_TRACE_HASH=1", NULL};
    const char *empty[] = {"AB_MODE=gpu", "AB_TRACE_HASH=1", "AB_TRACE=", NULL};
    const char *const *envs[] = {unset, empty};
    for (int e = 0; e < 2 && !first_failure[0]; e++) {
        ChildRun r = {0};
        char names[4][NAME_MAX + 1];
        if (run_child("off", dir, envs[e], &r) && printed_paths_match(r.out, LINE_CALLS, 3)) {
            int nf = list_dir(dir, names, 4);
            expect(nf == 0, "AB_TRACE %s: %s appeared in the working directory",
                   e ? "empty" : "unset", nf > 0 ? names[0] : "?");
        }
        free(r.out);
    }
    clean_dir(dir);
    finish();
}

// What ab_get_stats() counts for call c when it fails at `stage`: the
// elements converted to or from DD and the GEMM kernels counted. alloc fails
// before any GPU work. kernel fails after the uploads of op(A), op(B) and,
// for a DGEMM with beta != 0, C (ZGEMM applies beta on the host); ab_zgemm
// counts its kernel before it reports the failure, ab_dgemm_scaled does not.
static void fail_stats(const Call *c, const char *stage, unsigned long long *converted,
                       unsigned long long *gemms) {
    bool z = entry_complex(c->entry), kernel = !strcmp(stage, "kernel");
    unsigned long long parts = z ? 2 : 1, a = parts * c->M * c->K, b = parts * c->K * c->N,
                       in = a + b + (!z && c->beta != 0.0 ? (unsigned long long)c->M * c->N : 0);
    *converted = kernel ? in : 0;
    *gemms = kernel && z;
}

// AB_TEST_FAIL_STAGE=FAIL_STAGES[stage_index]: GPU-routed calls fall back to
// cblas from that step, after the GPU work done before it.
static void test_fail_stage(int stage_index) {
    begin();
    const char *stage = FAIL_STAGES[stage_index];
    char dir[PATH_MAX], trace[PATH_MAX + 16], fail[64];
    test_dir(stage, dir, sizeof dir);
    snprintf(trace, sizeof trace, "AB_TRACE=%s/trace", dir);
    snprintf(fail, sizeof fail, "AB_TEST_FAIL_STAGE=%s", stage);
    const char *env[] = {"AB_MODE=gpu", "AB_REPORT=1", trace, fail, NULL};
    ChildRun r = {0};
    g_fail_staged = stage_index == 0 ? "alloc" : "gpu_failed";
    if (run_child("fail", dir, env, &r)) {
        for (int i = 0; i < N_FAIL_CALLS && !first_failure[0]; i++) {
            const Call *c = &LINE_CALLS[i];
            const char *routine = entry_complex(c->entry) ? "zgemm" : "dgemm";
            char prefix[64], want[128], line[512] = "";
            unsigned long long converted, gemms;
            fail_stats(c, stage, &converted, &gemms);
            snprintf(prefix, sizeof prefix, "call %d ", i);
            snprintf(want, sizeof want, "call %d path=cpu bitwise=1 converted=%llu gemms=%llu", i,
                     converted, gemms);
            if (!expect(find_line(r.out, prefix, line, sizeof line) && !strcmp(line, want),
                        "%s: '%s', expected '%s' (C bitwise cblas's, the path record cpu, the "
                        "GPU work before %s fails): %s", routine, line, want, stage, r.out))
                break;
            // AB_REPORT: one CPU call, one fallback, and its reason gpu_failed.
            snprintf(prefix, sizeof prefix, "AB_REPORT[%d] %s calls=", (int)r.pid, routine);
            snprintf(want, sizeof want, "AB_REPORT[%d] %s calls=1 gpu_calls=0 cpu_calls=1 ",
                     (int)r.pid, routine);
            const char *l = find_line(r.out, prefix, line, sizeof line);
            size_t n = l ? strlen(l) : 0, tail = strlen(" fallbacks=1");
            if (!expect(l && !strncmp(l, want, strlen(want)) && n > tail &&
                        !strcmp(l + n - tail, " fallbacks=1"),
                        "%s: AB_REPORT does not count one CPU call with one fallback: %s",
                        routine, r.out))
                break;
            snprintf(prefix, sizeof prefix, "AB_REPORT[%d] %s fallbacks ", (int)r.pid, routine);
            snprintf(want, sizeof want, "AB_REPORT[%d] %s fallbacks no_device=0 too_large=0 "
                     "gpu_failed=1 other=0 range=0", (int)r.pid, routine);
            if (!expect(find_line(r.out, prefix, line, sizeof line) && !strcmp(line, want),
                        "%s: AB_REPORT does not give the fallback as gpu_failed: %s", routine,
                        r.out))
                break;
            // The staged path failed first, at the same step.
            snprintf(prefix, sizeof prefix, "AB_REPORT[%d] %s staged ", (int)r.pid, routine);
            snprintf(want, sizeof want, "AB_REPORT[%d] %s staged kept=0 kept_4m=0 host_pack=0 "
                     "fpcr=0 scalar=0 alloc=%d gpu_failed=%d nonfinite=0 range=0",
                     (int)r.pid, routine, stage_index == 0, stage_index == 1);
            expect(find_line(r.out, prefix, line, sizeof line) && !strcmp(line, want),
                   "%s: AB_REPORT does not give the staged path's failure as %s: %s", routine,
                   g_fail_staged, r.out);
        }
        const char *paths[] = {"fallback:gpu_failed", "fallback:gpu_failed"};
        if (!first_failure[0])
            check_file(dir, r.pid, LINE_CALLS, paths, N_FAIL_CALLS, false, NULL, NULL, NULL);
    }
    free(r.out);
    clean_dir(dir);
    finish();
}

static void test_fail_stage_alloc(void)  { test_fail_stage(0); }
static void test_fail_stage_kernel(void) { test_fail_stage(1); }

// =============================================================================
// Main
// =============================================================================

static const TestCase TESTS[] = {
    {"test_trace_lines",       test_trace_lines},
    {"test_trace_hash",        test_trace_hash},
    {"test_trace_off",         test_trace_off},
    {"test_fail_stage_alloc",  test_fail_stage_alloc},
    {"test_fail_stage_kernel", test_fail_stage_kernel},
};
static const int N_TESTS = (int)TEST_REGISTRY_SIZE(TESTS);

int main(int argc, char **argv) {
    if (argc == 4 && !strcmp(argv[1], "--child")) {
        alarm(CHILD_TIMEOUT_S);
        return child_main(argv[2], argv[3]);
    }

    uint32_t len = sizeof self_path;
    if (_NSGetExecutablePath(self_path, &len) != 0) {
        fprintf(stderr, "test_trace: cannot resolve executable path\n");
        return 1;
    }
    const char *tmp = getenv("TMPDIR");
    snprintf(root_dir, sizeof root_dir, "%s/test_trace.XXXXXX", tmp && *tmp ? tmp : "/tmp");
    if (!mkdtemp(root_dir)) {
        fprintf(stderr, "test_trace: cannot create %s: %s\n", root_dir, strerror(errno));
        return 1;
    }

    printf("╔══════════════════════════════════════════════════════════════════╗\n");
    printf("║  apple-bottom AB_TRACE and AB_TEST_FAIL_STAGE                    ║\n");
    printf("╚══════════════════════════════════════════════════════════════════╝\n");

    for (int i = 0; i < N_TESTS; i++) {
        printf("[%d/%d] %-26s ... ", i + 1, N_TESTS, TESTS[i].name);
        fflush(stdout);
        int pre = tests_passed + tests_failed;
        TESTS[i].fn();
        int post = tests_passed + tests_failed;
        if (post == pre) {
            fprintf(stderr, "\nFATAL: ghost test — '%s' reported no result\n", TESTS[i].name);
            abort();
        }
    }
    rmdir(root_dir);                 // left in place, with the failing tests' files, on a failure

    printf("\n═══════════════════════════════════════════════════════════════════\n");
    printf("Results: %d passed, %d failed\n", tests_passed, tests_failed);
    if (tests_failed == 0) {
        printf("✓ All %d trace tests PASSED\n", tests_passed);
        return 0;
    }
    printf("✗ Some tests FAILED\n");
    return 1;
}
