// bench_split: one GEMM split between Accelerate (the CPU's matrix units) and apple-bottom's GPU
// path, against either engine alone. Results and method: benchmarks/results/*-hybrid-split/README.md;
// how to run: benchmarks/split/README.md.
//
// CPU slice: cblas_[dz]gemm (Accelerate, VECLIB_MAXIMUM_THREADS threads) on the caller's
//            sub-block, on the calling thread.
// GPU slice: ab_[dz]gemm_blas with AB_MODE=gpu (staged path: GPU-side FP64->DD conversion and
//            epilogue, pooled buffers) on the rest, on a second thread. A, B, C and the K-split
//            partial P are registered once with ab_register_host_memory, so the GPU reads them
//            in place (registration is setup, not timed; SPLIT_NOREG=1 skips it).
// Axes: N splits columns of C, M splits rows of C, K splits the sum:
//       C = fl(fl(beta C0 + P_cpu) + P_gpu), P_gpu computed into P with beta = 0, added after both.
// np processes (1 to NPMAX = 8) run every repeat in lockstep (barrier before each); np = 1 is one process.
// Per case: warm-up, a sweep over the GPU share (3 rounds, configs interleaved), then a fixed block
// at the sweep's best share (SPLIT_ROUNDS rounds, default 3, of Accelerate / GPU / split, interleaved)
// whose medians are the headline (so the choice of f does not bias them), bit checks, and Dot2 errors
// on sampled columns. SPLIT_REPS=n gives every config of the fixed block n repeats per round.
// SPLIT_G=g, or SPLIT_F=f (cut as the library cuts it, ab_split_cut with the AB_MIN_GPU_DIM floor),
// skips the sweep and runs the fixed block at that GPU part.
// SPLIT_POWER_S=s adds runs of s seconds per engine for wall power (SPLIT_IDLE_S: idle before).
// The fixed block also runs four arms at the split's GPU part g (SPLIT_ARMS=0 leaves them out):
//   SEAM   the library's own split of the same call through its seam (ab_internal_[dz]gemm_split,
//          staged path, free completion order), with its outcome and timeline per call;
//   S<g>b  a second, identical user-code split, so the user split can be timed against itself;
//   HEAD   the CPU slice alone, on the calling thread;
//   TAIL   the GPU slice alone, on the GPU thread, the caller waiting.
// Power: the parent samples SMC keys (PDTR, the DC input = wall side on AC power, PSTR, PPBR and
// the GPU rails) at 20 Hz and IOReport's GPU energy, clock residency and busy fraction, both
// without root through undocumented interfaces; without them the power columns read nan.
// SPLIT_SAMPLER=0 turns that sampler off (power columns nan), e.g. to run under benchmarks/powerlog.
// Outside load, run mode: other processes' CPU per round (host ticks minus this run's own CPU time)
// and the GPU's busy fraction while this run left it idle; .rounds.tsv has both.
//
// usage: bench_split run <case> <np> <outdir> [data]    data 0 = uniform [-1,1), 1 = QE-like row decay
//        bench_split child <case> <np> <id> <shmfile> <data>   (internal)
//        bench_split sustain <case> <outdir> <g> [secs=80] [idle=30] [steady_after=20]
//               one process, back-to-back calls in runs of `secs` after `idle` seconds of idle each:
//               Accelerate alone, then in a shuffled order (SPLIT_SEED) the split at GPU part g, the GPU
//               alone, the CPU slice alone and the GPU slice alone, then the split at the share that
//               balances the split run's steady state, and Accelerate alone again; C is not reset
//               between calls; per-call throughput, SMC wall power and IOReport GPU clock and energy
//               over time
//        bench_split list
#include <IOKit/IOKitLib.h>
#include <complex.h>
#include <dispatch/dispatch.h>
#include <dlfcn.h>
#include <fcntl.h>
#include <libproc.h>
#include <mach/mach.h>
#include <mach/mach_time.h>
#include <math.h>
#include <pthread.h>
#include <spawn.h>
#include <stdatomic.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>
#include <sys/wait.h>
#include <unistd.h>
#ifndef ACCELERATE_NEW_LAPACK
#define ACCELERATE_NEW_LAPACK
#endif
#include <Accelerate/Accelerate.h>
#include "apple_bottom.h"
#include "dispatch.h"    // src/: AB_DISPATCH_DEFAULT_MIN_DIM, the default AB_MIN_GPU_DIM
#include "split_slice.h" // src/: the library's cut (ab_split_cut)
#include "test_dot2.h"   // tests/: Dot2 reference, U2 = 2^-48

extern char **environ;
typedef double _Complex zc;
#define NPMAX 8
#define MAXCFG 16
#define MAXREC 2048   // timed calls per process: the sweep, SPLIT_ROUNDS fixed rounds, 2 orders
#define MAXROUND 512  // rounds of the fixed block
#define PG 16384

// ---------------------------------------------------------------- utilities
static double mach_s(uint64_t ticks) {   // mach_absolute_time ticks in s
    static mach_timebase_info_data_t tb;
    if (!tb.denom) mach_timebase_info(&tb);
    return (double)ticks * tb.numer / tb.denom * 1e-9;
}
static double now(void) { return mach_s(mach_absolute_time()); }
static uint64_t splitmix64(uint64_t *s) {
    uint64_t z = (*s += 0x9E3779B97F4A7C15ull);
    z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ull; z = (z ^ (z >> 27)) * 0x94D049BB133111EBull;
    return z ^ (z >> 31);
}
static uint64_t sm;
static double rnd(void) { return (double)(splitmix64(&sm) >> 11) * 0x1p-53 * 2.0 - 1.0; }   // [-1, 1), identical in every process
// result hashes in SUMMARY (h_split, h_gpu, h_acc): this program's own, over 64-bit words; not
// comparable with AB_TRACE_HASH or the golden files' hashes
static uint64_t h64(const void *p, size_t n) {
    const uint64_t *q = p; uint64_t h = 1469598103934665603ull;
    for (size_t i = 0; i < n / 8; i++) { h ^= q[i]; h *= 1099511628211ull; h ^= h >> 29; }
    return h;
}
static void *palloc(size_t bytes) {  // page-aligned, page-rounded (clean registration)
    void *p = NULL;
    if (posix_memalign(&p, PG, (bytes + PG - 1) / PG * PG)) { fprintf(stderr, "alloc %zu failed\n", bytes); exit(1); }
    return p;
}
static int cmpd(const void *a, const void *b) { double x = *(const double *)a, y = *(const double *)b; return x < y ? -1 : x > y; }
static double median(const double *v, int n) {
    if (n <= 0) return NAN;
    double *t = malloc(sizeof(double) * n); memcpy(t, v, sizeof(double) * n); qsort(t, n, sizeof(double), cmpd);
    double m = n % 2 ? t[n / 2] : 0.5 * (t[n / 2 - 1] + t[n / 2]); free(t); return m;
}
static double envd(const char *name, double d) { const char *v = getenv(name); return v ? atof(v) : d; }
static void sleep_s(double s) { usleep((useconds_t)(s * 1e6)); }
static const char *on_off(const char *name) { const char *v = getenv(name); return v && *v && *v != '0' ? "on" : "off"; }   // as the library reads it
static bool noreg(void) { return envd("SPLIT_NOREG", 0) != 0; }   // SPLIT_NOREG=1: the operands are not registered
static void sh_line(const char *cmd, char *out, size_t n) {
    FILE *f = popen(cmd, "r"); out[0] = 0;
    if (f) { if (!fgets(out, (int)n, f)) out[0] = 0; pclose(f); }
    size_t l = strlen(out); if (l && out[l - 1] == '\n') out[l - 1] = 0;
}

// ---------------------------------------------------------------- cases
typedef struct {
    const char *name; char prec, ta, axis;   // prec 'd'/'z'; op(A) = A ('N') or A^H ('C')
    int M, N, K, lda, ldb, ldc, beta, align;
    int nf; double f[10];                    // GPU shares swept (cut aligned to `align`)
    int sweep_reps, fx_acc, fx_gpu, fx_split; // reps per config per round
    int big;                                 // warm up only the largest split; long alarm
} Case;
#define SQF 9, {0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70}
#define QEF 9, {0.25, 0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60, 0.65}
static const Case CASES[] = {
    // large squares, 'N','N', beta = 0, split by columns of C
    {"D2048", 'd', 'N', 'N', 2048, 2048, 2048, 2048, 2048, 2048, 0, 32, SQF, 3, 3, 3, 4, 0},
    {"D4096", 'd', 'N', 'N', 4096, 4096, 4096, 4096, 4096, 4096, 0, 32, SQF, 2, 3, 3, 4, 0},
    {"D8192", 'd', 'N', 'N', 8192, 8192, 8192, 8192, 8192, 8192, 0, 32, SQF, 1, 2, 2, 4, 1},
    {"Z2048", 'z', 'N', 'N', 2048, 2048, 2048, 2048, 2048, 2048, 0, 32, SQF, 3, 3, 3, 4, 0},
    {"Z4096", 'z', 'N', 'N', 4096, 4096, 4096, 4096, 4096, 4096, 0, 32, SQF, 2, 3, 3, 4, 0},
    {"Z8192", 'z', 'N', 'N', 8192, 8192, 8192, 8192, 8192, 8192, 0, 32, SQF, 1, 2, 2, 4, 1},
    // QE updates 'N','N', M = npw: A = psi(npwx, K), B = vc(ldb, N), C = hpsi(npwx, N), beta = 1
    {"UN1000", 'z', 'N', 'N', 18277, 500, 1000, 18336, 1000, 18336, 1, 32, 8, {0.26, 0.32, 0.38, 0.45, 0.51, 0.58, 0.64, 0.70}, 2, 3, 3, 4, 0},
    {"UN512",  'z', 'N', 'N', 18277, 500,  512, 18336,  512, 18336, 1, 32, 8, {0.26, 0.32, 0.38, 0.45, 0.51, 0.58, 0.64, 0.70}, 2, 3, 3, 4, 0},
    // the same with small GPU shares (32-160 columns), for GPU slices that take the host path
    {"UN1000L", 'z', 'N', 'N', 18277, 500, 1000, 18336, 1000, 18336, 1, 32, 5, {0.064, 0.128, 0.192, 0.256, 0.32}, 2, 3, 3, 4, 0},
    {"UN512L",  'z', 'N', 'N', 18277, 500,  512, 18336,  512, 18336, 1, 32, 5, {0.064, 0.128, 0.192, 0.256, 0.32}, 2, 3, 3, 4, 0},
    {"UM160",  'z', 'N', 'M', 18277, 160,  160, 18336, 1000, 18336, 1, 32, QEF, 3, 3, 3, 4, 0},
    {"UM320",  'z', 'N', 'M', 18277, 160,  320, 18336, 1000, 18336, 1, 32, QEF, 3, 3, 3, 4, 0},
    {"UM1000", 'z', 'N', 'M', 18277, 160, 1000, 18336, 1000, 18336, 1, 32, QEF, 2, 3, 3, 4, 0},
    // QE overlaps / reductions 'C','N', K = npw: A(npwx, M), B(npwx, N), C(M, N), beta = 0
    {"RK500",  'z', 'C', 'K',   500, 500, 18277, 18336, 18336,  500, 0, 64, QEF, 2, 3, 3, 4, 0},
    {"RK160",  'z', 'C', 'K',   160, 160, 18432, 18432, 18432,  160, 0, 64, QEF, 3, 3, 3, 4, 0},
    // pw.x's own calls at 8 ranks x 1 thread with -nk 4 (si64 inputs A and B; shapes, leading dimensions
    // and scalars from 1.5.1's AB_TRACE of those runs, one rank's of each class): A's zUN.k1 update and
    // zRK.t1, zRK.t2 overlaps; B's zUM.k2 update and zRK.t1 overlap. Axes as design section 5's rows.
    {"A8_UN512",  'z', 'N', 'N', 9139,  500,  512, 9160,  512, 9160, 1, 32, QEF, 2, 3, 3, 4, 0},
    {"A8_RK512",  'z', 'C', 'K',  512,  500, 9139, 9160, 9160,  512, 0, 64, QEF, 2, 3, 3, 4, 0},
    {"A8_RK1000", 'z', 'C', 'K',  500, 1000, 9176, 9176, 9176, 1000, 0, 64, QEF, 2, 3, 3, 4, 0},
    {"B8_UM1152", 'z', 'N', 'M', 8927,  160, 1152, 9014, 1152, 9014, 1, 32, QEF, 2, 3, 3, 4, 0},
    {"B8_RK1152", 'z', 'C', 'K', 1152,  160, 8988, 9014, 9014, 1152, 0, 64, QEF, 3, 3, 3, 4, 0},
};
#define NCASES (int)(sizeof CASES / sizeof CASES[0])
static const Case *find_case(const char *n) { for (int i = 0; i < NCASES; i++) if (!strcmp(CASES[i].name, n)) return &CASES[i]; return NULL; }
// the case with SPLIT_REPS=n repeats per round for every config of the fixed block, if set
static const Case *with_reps(const Case *c) {
    static Case cc;
    int n = (int)envd("SPLIT_REPS", 0);
    if (!c || n <= 0) return c;
    cc = *c; cc.fx_acc = cc.fx_gpu = cc.fx_split = n;
    return &cc;
}
static int axis_len(const Case *c) { return c->axis == 'N' ? c->N : c->axis == 'M' ? c->M : c->K; }
// the GPU part at share f: the library's cut with no AB_MIN_GPU_DIM floor, round(f L / a) a clamped
// to [a, L - a]; a is the axis's unit (ab_split_align), which every case's `align` states
static int gpu_part(const Case *c, double f) { return ab_split_cut(c->axis, axis_len(c), f, 0); }
static double flops(const Case *c) { return (c->prec == 'z' ? 8.0 : 2.0) * c->M * c->N * c->K; }

// ---------------------------------------------------------------- shared memory
enum { V_TOTAL, V_CPU, V_GPU, V_JOIN, V_WAKE, V_KADD, V_T0, V_T1, NV };
enum { PH_SWEEP, PH_FIXED, PH_ORDER };
enum { CF_ACC, CF_GPU, CF_SPLIT0 };   // config indices; the sweep's splits from CF_SPLIT0, the arms after them
enum { KD_ACC, KD_GPU, KD_SPLIT, KD_SPLIT2, KD_SEAM, KD_HEAD, KD_TAIL };
#define NARM 4                         // KD_SEAM, KD_SPLIT2, KD_HEAD, KD_TAIL
// one timed call: its times (s, V_*) and, for a SEAM call, the library's record of it
typedef struct {
    double v[NV];
    int prev;                          // the config of this process's call before it: PREV_IDLE after the
                                       // idle gap, PREV_NONE if none; untimed calls count
    int outcome, retries, in_place;    // ab_split_outcome_t, -1 if not a SEAM call
    ab_split_times_t tm;               // the seam's timeline (ns, mach_absolute_time); zero otherwise
} Call;
typedef struct { int phase, round, cfg, rep; float load1; Call k; } Rec;   // load1: the load average after the call
typedef struct {
    _Atomic int bcount, bgen;
    int np, ncfg, g[MAXCFG], kind[MAXCFG], best;   // ncfg: ACC, GPU and the sweep's splits
    int fixed, rounds;   // fixed: SPLIT_G or SPLIT_F set, no sweep; rounds of the fixed block
    int nwarm;           // untimed split calls before the fixed block
    // outside load per round of the fixed block, from process 0's snapshots (load_snap): the round's
    // start and end, and with the arms the end of its GPU-idle stretch (the idle gap and the CPU-only
    // configs, when none of this run's processes uses the GPU); other processes' CPU over the stretch
    // and over the round (cores), and the GPU's busy fraction over the stretch
    double rt0[MAXROUND], idle_t1[MAXROUND], rt1[MAXROUND];
    double fcpu_idle[MAXROUND], fcpu_round[MAXROUND], gpu_idle[MAXROUND];
    pid_t pid[NPMAX];
    int narm, k_seam, k_split2, k_head, k_tail;    // the arms (0 or NARM), at indices ncfg..ncfg+narm-1
    int nrec[NPMAX]; Rec rec[NPMAX][MAXREC];
    int reg_ok[NPMAX], notgpu[NPMAX], gpucalls[NPMAX];
    int split_reps[NPMAX], split_diff_reps[NPMAX], gpu_reps[NPMAX], gpu_diff_reps[NPMAX], acc_reps[NPMAX], acc_diff_reps[NPMAX];
    long split_diff_max[NPMAX], order_diff[NPMAX][2];
    int seam_reps[NPMAX], seam_diff_reps[NPMAX], split2_reps[NPMAX], split2_diff_reps[NPMAX];
    long seam_vs_split[NPMAX], head_vs_split[NPMAX], tail_vs_split[NPMAX];   // elements differing; -1 not compared
    uint64_t hash_split[NPMAX], hash_gpu[NPMAX], hash_acc[NPMAX];
    long cpu_vs_acc, gpu_vs_gpu, part_n_cpu, part_n_gpu, split_vs_acc, split_vs_gpu, all_n;
    double err_split_cpu, err_split_gpu, err_gpu, err_acc; int err_cols;
    double tsetup[NPMAX], ttotal[NPMAX];
    int pw_n[3]; double pw_t0[3][3], pw_t1[3][3];   // sustained power phase: calls per run; [round][config] windows
} Shm;
static Shm *shm;
static int myid;
static void barrier(void) {
    int gen = atomic_load(&shm->bgen);
    if (atomic_fetch_add(&shm->bcount, 1) == shm->np - 1) { atomic_store(&shm->bcount, 0); atomic_fetch_add(&shm->bgen, 1); }
    else while (atomic_load(&shm->bgen) == gen) usleep(30);
}

// ---------------------------------------------------------------- the operands and the split
typedef struct {
    const Case *c; size_t es;           // element size, 8 or 16
    void *A, *B, *C, *C0, *P;           // P: K-split GPU partial, M x N, ld M
    size_t nA, nB, nC;                  // elements
    int g, cut;                         // GPU part [cut, cut + g) of the split axis
} Ctx;
static Ctx X;
static dispatch_semaphore_t sem_go, sem_done, sem_cpu_done;
static int g_delay;                     // 0 none, 1 GPU slice starts after the CPU slice, 2 the reverse
enum { PREV_NONE = -2, PREV_IDLE = -1 };
static int g_prev = PREV_NONE;          // the config of this process's last call (Call.prev)
static bool g_noreset;                  // power phase: C is not reset between calls
static double g_tg0, g_tg1;

static void cpu_slice(Ctx *x) {
    const Case *c = x->c;
    if (c->prec == 'd') {   // only N splits for DGEMM
        cblas_dgemm(CblasColMajor, CblasNoTrans, CblasNoTrans, c->M, x->cut, c->K, 1.0, x->A, c->lda, x->B, c->ldb, c->beta, x->C, c->ldc);
        return;
    }
    zc one = 1, beta = c->beta;
    if (c->axis == 'N') cblas_zgemm(CblasColMajor, CblasNoTrans, CblasNoTrans, c->M, x->cut, c->K, &one, x->A, c->lda, x->B, c->ldb, &beta, x->C, c->ldc);
    else if (c->axis == 'M') cblas_zgemm(CblasColMajor, CblasNoTrans, CblasNoTrans, x->cut, c->N, c->K, &one, x->A, c->lda, x->B, c->ldb, &beta, x->C, c->ldc);
    else cblas_zgemm(CblasColMajor, CblasConjTrans, CblasNoTrans, c->M, c->N, x->cut, &one, x->A, c->lda, x->B, c->ldb, &beta, x->C, c->ldc);
}
static void gpu_slice(Ctx *x) {
    const Case *c = x->c; int cut = x->cut, g = x->g;
    if (c->prec == 'd') {
        const double *A = x->A, *B = x->B; double *C = x->C;
        ab_dgemm_blas('N', 'N', c->M, g, c->K, 1.0, A, c->lda, B + (size_t)cut * c->ldb, c->ldb, c->beta, C + (size_t)cut * c->ldc, c->ldc);
        return;
    }
    const zc *A = x->A, *B = x->B; zc *C = x->C, *P = x->P;
    if (c->axis == 'N') ab_zgemm_blas('N', 'N', c->M, g, c->K, 1, A, c->lda, B + (size_t)cut * c->ldb, c->ldb, c->beta, C + (size_t)cut * c->ldc, c->ldc);
    else if (c->axis == 'M') ab_zgemm_blas('N', 'N', g, c->N, c->K, 1, A + cut, c->lda, B, c->ldb, c->beta, C + cut, c->ldc);
    else ab_zgemm_blas('C', 'N', c->M, c->N, g, 1, A + cut, c->lda, B + cut, c->ldb, 0, P, c->M);
}
static void acc_full(Ctx *x) {
    const Case *c = x->c;
    if (c->prec == 'd') { cblas_dgemm(CblasColMajor, CblasNoTrans, CblasNoTrans, c->M, c->N, c->K, 1.0, x->A, c->lda, x->B, c->ldb, c->beta, x->C, c->ldc); return; }
    zc one = 1, beta = c->beta;
    cblas_zgemm(CblasColMajor, c->ta == 'C' ? CblasConjTrans : CblasNoTrans, CblasNoTrans, c->M, c->N, c->K, &one, x->A, c->lda, x->B, c->ldb, &beta, x->C, c->ldc);
}
static void gpu_full(Ctx *x) {
    const Case *c = x->c;
    if (c->prec == 'd') ab_dgemm_blas('N', 'N', c->M, c->N, c->K, 1.0, x->A, c->lda, x->B, c->ldb, c->beta, x->C, c->ldc);
    else ab_zgemm_blas(c->ta, 'N', c->M, c->N, c->K, 1, x->A, c->lda, x->B, c->ldb, c->beta, x->C, c->ldc);
    shm->gpucalls[myid]++;
    if (strcmp(ab_get_last_dispatch_path(), "gpu")) shm->notgpu[myid]++;
}
static void *gpu_thread(void *arg) {
    (void)arg;
    for (;;) {
        dispatch_semaphore_wait(sem_go, DISPATCH_TIME_FOREVER);
        if (g_delay == 1) dispatch_semaphore_wait(sem_cpu_done, DISPATCH_TIME_FOREVER);
        g_tg0 = now();
        gpu_slice(&X);
        g_tg1 = now();
        shm->gpucalls[myid]++;
        if (strcmp(ab_get_last_dispatch_path(), "gpu")) shm->notgpu[myid]++;
        dispatch_semaphore_signal(sem_done);
    }
    return NULL;
}
static void kadd(Ctx *x) {  // C = fl(C_cpu + P_gpu)
    const Case *c = x->c; zc *C = x->C; const zc *P = x->P;
    for (int j = 0; j < c->N; j++) for (int i = 0; i < c->M; i++) C[(size_t)j * c->ldc + i] += P[(size_t)j * c->M + i];
}
// the library's split of the whole call at x->g through its seam: staged path, free order
static int seam_call(Ctx *x) {
    const Case *c = x->c; const ab_split_forced_t f = {c->axis, x->g, 0, 0};
    if (c->prec == 'd') return ab_internal_dgemm_split(&f, 'N', 'N', c->M, c->N, c->K, 1.0, x->A, c->lda, x->B, c->ldb, c->beta, x->C, c->ldc);
    return ab_internal_zgemm_split(&f, c->ta, 'N', c->M, c->N, c->K, 1, x->A, c->lda, x->B, c->ldb, c->beta, x->C, c->ldc);
}
static double ns_s(uint64_t ns) { return (double)ns * 1e-9; }
// one timed call of config k; C is reset from C0 first (untimed). V_CPU, V_GPU: the slices (or the
// whole call); V_JOIN: the caller's wait for the GPU slice after its own; V_WAKE: the GPU slice's start
// after the call's; V_KADD: the K axis's sum (the seam: everything after the join)
static void one_call(Ctx *x, int k, Call *o) {
    if (!g_noreset) memcpy(x->C, x->C0, x->nC * x->es);
    double *v = o->v; int kd = shm->kind[k];
    memset(o, 0, sizeof *o); o->outcome = -1; o->prev = g_prev; g_prev = k;
    double t0 = now(), tc0 = t0, tc1, tj, tk;
    if (kd == KD_ACC) { acc_full(x); tc1 = tj = tk = now(); v[V_CPU] = tc1 - t0; }
    else if (kd == KD_GPU) { gpu_full(x); tc1 = tj = tk = now(); v[V_GPU] = tc1 - t0; }
    else if (kd == KD_HEAD) { cpu_slice(x); tc1 = tj = tk = now(); v[V_CPU] = tc1 - t0; }
    else if (kd == KD_SEAM) {
        o->outcome = seam_call(x); tk = now();
        ab_split_record_t r;
        if (ab_internal_last_split(&r)) {
            const ab_split_times_t *m = &r.times; o->tm = *m; o->retries = r.retries; o->in_place = r.in_place;
            if (m->cpu1_ns) v[V_CPU] = ns_s(m->cpu1_ns - m->cpu0_ns);
            if (m->gpu1_ns) { v[V_GPU] = ns_s(m->gpu1_ns - m->gpu0_ns); v[V_WAKE] = ns_s(m->gpu0_ns) - t0; }
            if (m->join_ns) { v[V_JOIN] = ns_s(m->join_ns - m->cpu1_ns); v[V_KADD] = ns_s(m->end_ns - m->join_ns); }
        }
    } else {   // a user-code split (KD_SPLIT, KD_SPLIT2), or its GPU slice alone (KD_TAIL)
        dispatch_semaphore_signal(sem_go);
        if (g_delay == 2) dispatch_semaphore_wait(sem_done, DISPATCH_TIME_FOREVER);
        tc0 = now();
        if (kd != KD_TAIL) cpu_slice(x);
        tc1 = now();
        if (g_delay == 1) dispatch_semaphore_signal(sem_cpu_done);
        if (g_delay != 2) dispatch_semaphore_wait(sem_done, DISPATCH_TIME_FOREVER);
        tj = now();
        if (x->c->axis == 'K' && kd != KD_TAIL) kadd(x);
        tk = now();
        if (kd != KD_TAIL) v[V_CPU] = tc1 - tc0;
        v[V_GPU] = g_tg1 - g_tg0; v[V_JOIN] = tj - tc1; v[V_WAKE] = g_tg0 - t0; v[V_KADD] = tk - tj;
    }
    v[V_TOTAL] = tk - t0; v[V_T0] = t0; v[V_T1] = tk;
}
static int cfg_g(const Case *c, int k) { return shm->kind[k] == KD_ACC ? 0 : shm->kind[k] == KD_GPU ? axis_len(c) : shm->g[k]; }
static void set_cfg(Ctx *x, int k) {
    if (shm->kind[k] != KD_ACC && shm->kind[k] != KD_GPU) { x->g = shm->g[k]; x->cut = axis_len(x->c) - x->g; }
}
static void record(int phase, int round, int cfg, int rep, const Call *o) {
    int n = shm->nrec[myid];
    if (n >= MAXREC) { fprintf(stderr, "too many records (MAXREC %d)\n", MAXREC); exit(1); }
    Rec *r = &shm->rec[myid][n]; r->phase = phase; r->round = round; r->cfg = cfg; r->rep = rep;
    double la; r->load1 = getloadavg(&la, 1) == 1 ? (float)la : NAN;
    r->k = *o; shm->nrec[myid] = n + 1;
}

// ---------------------------------------------------------------- data
// X: cols x ld, column-major, w doubles per element, uniform [-1,1); with decay, row i scaled by 10^(-9 i/ld)
static void fill_mat(double *X, int cols, int ld, int w, bool decay) {
    double *sc = malloc(sizeof(double) * ld);
    for (int i = 0; i < ld; i++) sc[i] = decay ? pow(10.0, -9.0 * i / ld) : 1.0;
    for (int j = 0; j < cols; j++) for (int i = 0; i < ld; i++)
        for (int t = 0; t < w; t++) X[((size_t)j * ld + i) * w + t] = rnd() * sc[i];
    free(sc);
}
// QE-like data (1): each plane-wave row (G index) of psi-like arrays decays over 9 decades;
// U: A = psi rows, C0 = hpsi rows, B = vc uniform; R: A and B are both psi-like.
static void fill(Ctx *x, int data) {
    const Case *c = x->c; int w = c->prec == 'z' ? 2 : 1;
    sm = 0x9E3779B97F4A7C15ull;
    fill_mat(x->A, c->ta == 'C' ? c->M : c->K, c->lda, w, data);
    fill_mat(x->B, c->N, c->ldb, w, data && c->ta == 'C');
    fill_mat(x->C0, c->N, c->ldc, w, data && c->ta == 'N' && c->axis != 'K' && c->M > 4096);
}

// the case's operands (A, B, C, C0 and the K axis's partial P), filled with `data`, C = C0, and the
// library initialised; registered with ab_register_host_memory unless SPLIT_NOREG=1 (then the staged
// path copies them). Returns how many of the four arrays (A, B, C, P) registered, or -1.
static int setup_operands(Ctx *x, const Case *c, int data) {
    x->c = c; x->es = c->prec == 'z' ? 16 : 8;
    int acols = c->ta == 'C' ? c->M : c->K;
    x->nA = (size_t)c->lda * acols; x->nB = (size_t)c->ldb * c->N; x->nC = (size_t)c->ldc * c->N;
    size_t es = x->es, nP = (size_t)c->M * c->N;
    x->A = palloc(x->nA * es); x->B = palloc(x->nB * es); x->C = palloc(x->nC * es); x->C0 = palloc(x->nC * es); x->P = palloc(nP * es);
    fill(x, data);
    memcpy(x->C, x->C0, x->nC * es); memset(x->P, 0, nP * es);
    if (ab_init() != AB_OK) { fprintf(stderr, "ab_init failed\n"); return -1; }
    int ok = 0;
    if (!noreg()) {
        ok += ab_register_host_memory(x->A, x->nA * es) == AB_OK;
        ok += ab_register_host_memory(x->B, x->nB * es) == AB_OK;
        ok += ab_register_host_memory(x->C, x->nC * es) == AB_OK;
        ok += ab_register_host_memory(x->P, nP * es) == AB_OK;
    }
    return ok;
}

// ---------------------------------------------------------------- checks
static long cmp_block(const Ctx *x, const void *a, const void *b, int r0, int r1, int c0, int c1) {
    long d = 0; const char *p = a, *q = b; size_t es = x->es;
    for (int j = c0; j < c1; j++) for (int i = r0; i < r1; i++) {
        size_t o = ((size_t)j * x->c->ldc + i) * es; d += memcmp(p + o, q + o, es) != 0;
    }
    return d;
}
// Dot2 reference R = beta C0 + op(A) op(B) on sampled columns; max |C - R| / (|op(A)||op(B)| + |beta C0|)
typedef struct { int K, ns; const double *Ar, *Ai, *mb; double *den; bool cplx; } AbsJob;
static void abs_row(void *ctx, size_t i) {
    const AbsJob *j = ctx; int K = j->K, ns = j->ns; double *d = j->den + i * ns;
    for (int k = 0; k < K; k++) {
        double ma = j->cplx ? hypot(j->Ar[i * K + k], j->Ai[i * K + k]) : fabs(j->Ar[i * K + k]);
        const double *b = j->mb + (size_t)k * ns;
        for (int c = 0; c < ns; c++) d[c] += ma * b[c];
    }
}
typedef struct { const Ctx *x; const void *C[3]; double err[3][2]; int ns; const int *cols; } ErrJob;
// err[r][0] CPU part (or all, K split), err[r][1] GPU part, for each of up to 3 results
static void ref_errors(ErrJob *ej) {
    const Ctx *x = ej->x; const Case *c = x->c; int M = c->M, K = c->K, ns = ej->ns; bool z = c->prec == 'z';
    double *Ar = malloc(sizeof(double) * (size_t)M * K), *Ai = z ? malloc(sizeof(double) * (size_t)M * K) : NULL;
    for (int i = 0; i < M; i++) for (int k = 0; k < K; k++) {
        if (z) {
            const zc *A = x->A; zc v = c->ta == 'C' ? conj(A[(size_t)i * c->lda + k]) : A[(size_t)k * c->lda + i];
            Ar[(size_t)i * K + k] = creal(v); Ai[(size_t)i * K + k] = cimag(v);
        } else Ar[(size_t)i * K + k] = ((const double *)x->A)[(size_t)k * c->lda + i];
    }
    double *Br = malloc(sizeof(double) * (size_t)K * ns), *Bi = z ? malloc(sizeof(double) * (size_t)K * ns) : NULL;
    for (int k = 0; k < K; k++) for (int s = 0; s < ns; s++) {
        if (z) { zc v = ((const zc *)x->B)[(size_t)ej->cols[s] * c->ldb + k]; Br[(size_t)k * ns + s] = creal(v); Bi[(size_t)k * ns + s] = cimag(v); }
        else Br[(size_t)k * ns + s] = ((const double *)x->B)[(size_t)ej->cols[s] * c->ldb + k];
    }
    size_t nr = (size_t)M * ns;
    double *S0 = calloc(nr, 8), *E0 = calloc(nr, 8), *S1 = z ? calloc(nr, 8) : NULL, *E1 = z ? calloc(nr, 8) : NULL, *den = calloc(nr, 8);
    if (c->beta) for (int i = 0; i < M; i++) for (int s = 0; s < ns; s++) {
        size_t y = (size_t)i * ns + s, o = (size_t)ej->cols[s] * c->ldc + i;
        if (z) { zc v = ((const zc *)x->C0)[o]; S0[y] = creal(v); S1[y] = cimag(v); den[y] = cabs(v); }
        else { double v = ((const double *)x->C0)[o]; S0[y] = v; den[y] = fabs(v); }
    }
    if (z) dot2_zgemm_acc(M, ns, K, Ar, Ai, Br, Bi, (double *const[2]){S0, S1}, (double *const[2]){E0, E1});
    else dot2_gemm_acc(M, ns, K, 1.0, Ar, Br, S0, E0);
    double *mb = malloc(sizeof(double) * (size_t)K * ns);
    for (size_t t = 0; t < (size_t)K * ns; t++) mb[t] = z ? hypot(Br[t], Bi[t]) : fabs(Br[t]);
    AbsJob aj = {K, ns, Ar, Ai, mb, den, z};
    dispatch_apply_f((size_t)M, DISPATCH_APPLY_AUTO, &aj, abs_row);
    for (int r = 0; r < 3; r++) {
        for (int i = 0; i < M; i++) for (int s = 0; s < ns; s++) {
            size_t y = (size_t)i * ns + s, o = (size_t)ej->cols[s] * c->ldc + i; double e;
            if (z) { zc v = ((const zc *)ej->C[r])[o]; e = hypot(creal(v) - (S0[y] + E0[y]), cimag(v) - (S1[y] + E1[y])) / den[y]; }
            else e = fabs(((const double *)ej->C[r])[o] - (S0[y] + E0[y])) / den[y];
            int part = c->axis == 'N' ? ej->cols[s] >= x->cut : c->axis == 'M' ? i >= x->cut : 0;
            if (e > ej->err[r][part]) ej->err[r][part] = e;
        }
    }
    free(Ar); free(Ai); free(Br); free(Bi); free(S0); free(E0); free(S1); free(E1); free(den); free(mb);
}

// ---------------------------------------------------------------- child
// lockstep (the slowest process) or mean-over-process value of record i
static double lockv(int i, int fld) { double m = 0; for (int p = 0; p < shm->np; p++) if (shm->rec[p][i].k.v[fld] > m) m = shm->rec[p][i].k.v[fld]; return m; }
static double meanv(int i, int fld) { double m = 0; for (int p = 0; p < shm->np; p++) m += shm->rec[p][i].k.v[fld] / shm->np; return m; }
// median over the records of (phase, cfg[, round]) of the lockstep (max over procs) or mean-over-procs value
static double rstat(int phase, int cfg, int round, int fld, bool lock, int *count) {
    double L[MAXREC]; int n = 0;
    for (int i = 0; i < shm->nrec[0]; i++) {
        const Rec *r = &shm->rec[0][i];
        if (r->phase != phase || r->cfg != cfg || (round >= 0 && r->round != round)) continue;
        L[n++] = lock ? lockv(i, fld) : meanv(i, fld);
    }
    if (count) *count = n;
    return median(L, n);
}
static void order_of(int r, int n, int *ord) {  // interleave: rotate by r, reverse on odd rounds
    for (int i = 0; i < n; i++) ord[i] = (i + r) % n;
    if (r % 2) for (int i = 0; i < n / 2; i++) { int t = ord[i]; ord[i] = ord[n - 1 - i]; ord[n - 1 - i] = t; }
}
// One call of a fixed-block round: config k's repeat rep, or an untimed call of k (rep -1).
typedef struct { int k, rep; } Step;
#define MAXSTEP 512   // calls per round (main checks the cases' repeats against it)
static int add_reps(Step *p, int n, int k, int reps) { for (int r = 0; r < reps; r++) p[n++] = (Step){k, r}; return n; }
// The trio's orders, a Williams design for three (0 S<g>, 1 SEAM, 2 S<g>b): over NW6 rounds each arm
// runs first, and right before and right after each other arm, equally often.
#define NW6 6
static const int W6[NW6][3] = {{1, 0, 2}, {2, 0, 1}, {0, 2, 1}, {1, 2, 0}, {2, 1, 0}, {0, 1, 2}};
// Round r of the fixed block, the user split at config bk. Without the arms, as before: Accelerate,
// GPU and the split with their repeats back to back, rotated and reversed by round (order_of). With
// them (after the idle gap child() runs first): the CPU-only configs ACC and HEAD, then the GPU alone
// and TAIL, each pair's order alternating by round; then one untimed call of the user split and the
// trio SEAM, S<g>, S<g>b, repeat by repeat, in W6[r % NW6]'s order. So ACC and HEAD never follow a
// GPU config, no arm of the trio follows ACC, GPU, HEAD or TAIL, and pairs by (round, rep) stay
// within one repeat of the trio. Round 0 runs the user split before its twin, so its first result
// (the bit checks' reference) comes first.
static int fixed_plan(const Case *c, int r, int bk, Step *p) {
    int n = 0;
    if (!shm->narm) {
        const int cf[3] = {CF_ACC, CF_GPU, bk}, reps[3] = {c->fx_acc, c->fx_gpu, c->fx_split}; int ord[3];
        order_of(r, 3, ord);
        for (int i = 0; i < 3; i++) n = add_reps(p, n, cf[ord[i]], reps[ord[i]]);
        return n;
    }
    const int cf[2][2] = {{CF_ACC, shm->k_head}, {CF_GPU, shm->k_tail}}, rp[2][2] = {{c->fx_acc, c->fx_split}, {c->fx_gpu, c->fx_split}};
    for (int e = 0; e < 2; e++) for (int i = 0; i < 2; i++) n = add_reps(p, n, cf[e][i ^ (r & 1)], rp[e][i ^ (r & 1)]);
    p[n++] = (Step){bk, -1};
    const int trio[3] = {bk, shm->k_seam, shm->k_split2};
    for (int rep = 0; rep < c->fx_split; rep++)
        for (int j = 0; j < 3; j++) p[n++] = (Step){trio[W6[r % NW6][j]], rep};
    return n;
}
// outside load: process 0's snapshots at the fixed block's round boundaries (defined with the samplers)
typedef struct { double t, busy, total, own; CFDictionaryRef ior; } LoadSnap;   // a snapshot (load_snap)
static void load_init(void);
static void load_snap(LoadSnap *s);
static double load_fcpu(const LoadSnap *a, const LoadSnap *b);
static double load_gpu_busy(const LoadSnap *a, const LoadSnap *b);
static void load_free(LoadSnap *s);
// Watchdogs, so a hang never holds the machine lock: setup, warm-up and sweep, then each round of the
// fixed block, then the rest (orders, power runs, checks) each get this long.
static unsigned watchdog(const Case *c) { return c->big ? 2400 : 600; }
static int child(const Case *c, int id, const char *shmpath, int data) {   // np: shm->np
    double T0 = now();
    alarm(watchdog(c));
    myid = id;
    int fd = open(shmpath, O_RDWR); if (fd < 0) { perror("open shm"); return 1; }
    shm = mmap(NULL, sizeof(Shm), PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0); if (shm == MAP_FAILED) { perror("mmap"); return 1; }
    Ctx *x = &X;
    int ok = setup_operands(x, c, data);
    if (ok < 0) return 1;
    size_t es = x->es;
    void *Cs0 = palloc(x->nC * es), *Cg0 = palloc(x->nC * es), *Ca0 = palloc(x->nC * es), *Cm0 = shm->narm ? palloc(x->nC * es) : NULL;
    shm->reg_ok[id] = ok;
    shm->pid[id] = getpid();
    sem_go = dispatch_semaphore_create(0); sem_done = dispatch_semaphore_create(0); sem_cpu_done = dispatch_semaphore_create(0);
    pthread_t th; pthread_create(&th, NULL, gpu_thread, NULL);
    Call v;
    // warm-up (untimed): Accelerate's threads, the library's pipelines and staging pool
    barrier(); one_call(x, CF_ACC, &v);
    barrier(); one_call(x, CF_GPU, &v);
    for (int k = shm->ncfg - 1; k >= CF_SPLIT0; k--) {
        set_cfg(x, k); barrier(); one_call(x, k, &v);
        if (c->big) break;
    }
    shm->tsetup[id] = now() - T0;
    // ---- sweep: every config, 3 rounds, interleaved (none at a fixed share)
    int ord[MAXCFG];
    for (int r = 0; r < (shm->fixed ? 0 : 3); r++) {
        order_of(r, shm->ncfg, ord);
        for (int i = 0; i < shm->ncfg; i++) {
            int k = ord[i]; set_cfg(x, k);
            for (int rep = 0; rep < c->sweep_reps; rep++) { barrier(); one_call(x, k, &v); record(PH_SWEEP, r, k, rep, &v); }
        }
    }
    barrier();
    if (id == 0) {  // best split = lowest median lockstep time over the sweep
        double best = 1e30; int bk = CF_SPLIT0;
        for (int k = CF_SPLIT0; k < shm->ncfg; k++) {
            double md = rstat(PH_SWEEP, k, -1, V_TOTAL, true, NULL);
            if (md < best) { best = md; bk = k; }
        }
        shm->best = bk;
        for (int a = 0; a < shm->narm; a++) shm->g[shm->ncfg + a] = shm->g[bk];   // the arms cut where the split does
    }
    barrier();
    int bk = shm->best;
    // warm-up of the arms (untimed): the seam's thread and pool slots, the slices alone
    for (int k = shm->ncfg; k < shm->ncfg + shm->narm; k++) { set_cfg(x, k); barrier(); one_call(x, k, &v); }
    // warm-up before the block (untimed): the split at g for SPLIT_WARM_S seconds (default 2 with the
    // arms, 0 without), so the block does not start cool; process 0's call time sets the count for all
    const double warm_s = envd("SPLIT_WARM_S", shm->narm ? 2 : 0);
    if (warm_s > 0) {
        set_cfg(x, bk); barrier(); one_call(x, bk, &v);
        if (id == 0) { double n = ceil(warm_s / fmax(v.v[V_TOTAL], 1e-6)) - 1; shm->nwarm = n < 0 ? 0 : n > 10000 ? 10000 : (int)n; }
        barrier();
        for (int i = 0; i < shm->nwarm; i++) { barrier(); one_call(x, bk, &v); }
    }
    // ---- fixed block at the best share, shm->rounds rounds (fixed_plan has the order)
    int sreps = 0, sdiff = 0, greps = 0, gdiff = 0, areps = 0, adiff = 0; long smax = 0;
    int mreps = 0, mdiff = 0, s2reps = 0, s2diff = 0, hdone = 0, tdone = 0;
    shm->seam_vs_split[id] = shm->head_vs_split[id] = shm->tail_vs_split[id] = -1;
    const int M = c->M, N = c->N, cut = axis_len(c) - shm->g[bk];
    const double gap_s = envd("SPLIT_GAP_S", 1.0);
    Step plan[MAXSTEP];
    const int ncpu_steps = shm->narm ? c->fx_acc + c->fx_split : 0;   // fixed_plan's CPU-only calls come first
    LoadSnap s0 = {0}, s1 = {0};
    barrier();
    if (id == 0) load_init();
    for (int r = 0; r < shm->rounds; r++) {
        alarm(watchdog(c));
        barrier();
        if (id == 0) { load_snap(&s0); shm->rt0[r] = s0.t; }
        if (shm->narm) { sleep_s(gap_s); g_prev = PREV_IDLE; }   // the idle gap that starts a round
        int nst = fixed_plan(c, r, bk, plan);
        for (int st = 0; st < nst; st++) {
            int k = plan[st].k, rep = plan[st].rep;
            int kd = shm->kind[k]; set_cfg(x, k);
            barrier(); one_call(x, k, &v);
            if (id == 0 && st == ncpu_steps - 1) {   // the GPU-idle stretch ends with the last CPU-only call
                load_snap(&s1); shm->idle_t1[r] = s1.t;
                shm->fcpu_idle[r] = load_fcpu(&s0, &s1); shm->gpu_idle[r] = load_gpu_busy(&s0, &s1); load_free(&s1);
            }
            if (rep < 0) continue;   // untimed
            record(PH_FIXED, r, k, rep, &v);
            if (kd == KD_SPLIT2) { if (sreps) { s2diff += memcmp(Cs0, x->C, x->nC * es) != 0; s2reps++; } continue; }   // against the user split's first
            if (kd == KD_HEAD || kd == KD_TAIL) {   // the part of C the slice wrote, against the user split's first (N and M axes)
                int *done = kd == KD_HEAD ? &hdone : &tdone;
                if (*done || !sreps || c->axis == 'K') continue;
                bool head = kd == KD_HEAD, n_ax = c->axis == 'N';
                long d = cmp_block(x, Cs0, x->C, n_ax ? 0 : head ? 0 : cut, n_ax ? M : head ? cut : M, n_ax ? (head ? 0 : cut) : 0, n_ax ? (head ? cut : N) : N);
                if (head) shm->head_vs_split[id] = d; else shm->tail_vs_split[id] = d;
                *done = 1; continue;
            }
            void *ref = kd == KD_ACC ? Ca0 : kd == KD_GPU ? Cg0 : kd == KD_SEAM ? Cm0 : Cs0;
            int *nr = kd == KD_ACC ? &areps : kd == KD_GPU ? &greps : kd == KD_SEAM ? &mreps : &sreps;
            int *nd = kd == KD_ACC ? &adiff : kd == KD_GPU ? &gdiff : kd == KD_SEAM ? &mdiff : &sdiff;
            if (*nr == 0) memcpy(ref, x->C, x->nC * es);
            else if (memcmp(ref, x->C, x->nC * es)) {
                (*nd)++;
                if (k == bk) { long d = cmp_block(x, ref, x->C, 0, c->M, 0, c->N); if (d > smax) smax = d; }
            }
            (*nr)++;
        }
        barrier();
        if (id == 0) { load_snap(&s1); shm->rt1[r] = s1.t; shm->fcpu_round[r] = load_fcpu(&s0, &s1); load_free(&s1); load_free(&s0); }
    }
    alarm(watchdog(c));
    // ---- completion order: GPU slice starts after the CPU slice, then the reverse
    set_cfg(x, bk);
    for (int o = 1; o <= 2; o++) {
        g_delay = o; barrier(); one_call(x, bk, &v); record(PH_ORDER, 0, bk, o, &v);
        shm->order_diff[id][o - 1] = memcmp(Cs0, x->C, x->nC * es) ? cmp_block(x, Cs0, x->C, 0, c->M, 0, c->N) : 0;
    }
    g_delay = 0;
    // ---- sustained power (SPLIT_POWER_S = seconds per run): each config back to back, 3 interleaved
    // rounds, C not reset, so SMC's ~1 s lag can be trimmed; no bit checks here
    double pws = envd("SPLIT_POWER_S", 0);
    const int pcfg[3] = {CF_ACC, CF_GPU, bk};
    if (pws > 0) {
        barrier();
        if (id == 0) for (int i = 0; i < 3; i++) {
            int n = (int)ceil(pws / rstat(PH_FIXED, pcfg[i], -1, V_TOTAL, true, NULL));
            shm->pw_n[i] = n < 2 ? 2 : n;
        }
        barrier();
        g_noreset = true;
        for (int r = 0; r < 3; r++) {
            order_of(r, 3, ord);
            for (int i = 0; i < 3; i++) {
                int ci = ord[i], k = pcfg[ci]; set_cfg(x, k);
                barrier(); double t0 = now();
                for (int n = 0; n < shm->pw_n[ci]; n++) { barrier(); one_call(x, k, &v); }
                barrier();
                if (id == 0) { shm->pw_t0[r][ci] = t0; shm->pw_t1[r][ci] = now(); }
            }
        }
        g_noreset = false;
        set_cfg(x, bk);
    }
    shm->split_reps[id] = sreps; shm->split_diff_reps[id] = sdiff; shm->split_diff_max[id] = smax;
    shm->gpu_reps[id] = greps; shm->gpu_diff_reps[id] = gdiff; shm->acc_reps[id] = areps; shm->acc_diff_reps[id] = adiff;
    shm->hash_split[id] = h64(Cs0, x->nC * es); shm->hash_gpu[id] = h64(Cg0, x->nC * es); shm->hash_acc[id] = h64(Ca0, x->nC * es);
    shm->seam_reps[id] = mreps; shm->seam_diff_reps[id] = mdiff; shm->split2_reps[id] = s2reps; shm->split2_diff_reps[id] = s2diff;
    if (mreps) shm->seam_vs_split[id] = cmp_block(x, Cm0, Cs0, 0, M, 0, N);
    shm->ttotal[id] = now() - T0;
    barrier();
    if (id == 0) {  // which parts of the split equal which single-engine result; errors vs Dot2
        if (c->axis == 'N') {
            shm->cpu_vs_acc = cmp_block(x, Cs0, Ca0, 0, M, 0, cut); shm->part_n_cpu = (long)M * cut;
            shm->gpu_vs_gpu = cmp_block(x, Cs0, Cg0, 0, M, cut, N); shm->part_n_gpu = (long)M * (N - cut);
        } else if (c->axis == 'M') {
            shm->cpu_vs_acc = cmp_block(x, Cs0, Ca0, 0, cut, 0, N); shm->part_n_cpu = (long)cut * N;
            shm->gpu_vs_gpu = cmp_block(x, Cs0, Cg0, cut, M, 0, N); shm->part_n_gpu = (long)(M - cut) * N;
        }
        shm->split_vs_acc = cmp_block(x, Cs0, Ca0, 0, M, 0, N); shm->split_vs_gpu = cmp_block(x, Cs0, Cg0, 0, M, 0, N); shm->all_n = (long)M * N;
        int cols[32], ns = 0;
        if (c->axis == 'N') {
            for (int s = 0; s < 8; s++) cols[ns++] = (int)((long)s * (cut - 1) / 7);
            for (int s = 0; s < 8; s++) cols[ns++] = cut + (int)((long)s * (N - 1 - cut) / 7);
        } else for (int s = 0; s < 16; s++) cols[ns++] = (int)((long)s * (N - 1) / 15);
        ErrJob ej = {x, {Cs0, Cg0, Ca0}, {{0}}, ns, cols};
        ref_errors(&ej);
        shm->err_split_cpu = ej.err[0][0]; shm->err_split_gpu = ej.err[0][1];
        shm->err_gpu = fmax(ej.err[1][0], ej.err[1][1]); shm->err_acc = fmax(ej.err[2][0], ej.err[2][1]); shm->err_cols = ns;
    }
    barrier();
    return 0;
}

// ---------------------------------------------------------------- SMC power (parent, no root)
// SPLIT_SAMPLER=0: no SMC or IOReport sampling here (another sampler, such as powerlog, runs instead)
static bool sampler_on(void) { return envd("SPLIT_SAMPLER", 1) != 0; }
typedef struct { char major, minor, build, reserved; UInt16 release; } SMCVers;
typedef struct { UInt16 version, length; UInt32 cpuPLimit, gpuPLimit, memPLimit; } SMCPLimit;
typedef struct { UInt32 dataSize; UInt32 dataType; char dataAttributes; } SMCKeyInfo;
typedef struct { UInt32 key; SMCVers vers; SMCPLimit pLimitData; SMCKeyInfo keyInfo; char result, status, data8; UInt32 data32; unsigned char bytes[32]; } SMCKeyData;
static io_connect_t g_smc;
static UInt32 k4(const char *s) { return ((UInt32)s[0] << 24) | ((UInt32)s[1] << 16) | ((UInt32)s[2] << 8) | (UInt32)s[3]; }
static int smc_call(SMCKeyData *in, SMCKeyData *out) { size_t sz = sizeof(SMCKeyData); return IOConnectCallStructMethod(g_smc, 2, in, sizeof(SMCKeyData), out, &sz) == KERN_SUCCESS && !out->result ? 0 : -1; }
static int smc_info(UInt32 key, SMCKeyInfo *ki) { SMCKeyData in = {0}, out = {0}; in.key = key; in.data8 = 9; if (smc_call(&in, &out)) return -1; *ki = out.keyInfo; return 0; }
static int smc_flt(UInt32 key, SMCKeyInfo ki, float *f) { SMCKeyData in = {0}, out = {0}; in.key = key; in.keyInfo.dataSize = ki.dataSize; in.data8 = 5; if (smc_call(&in, &out)) return -1; memcpy(f, out.bytes, 4); return 0; }
// PDTR: DC input (the wall side on AC; includes battery charging). PPBR: battery.
// PC10+PC12+PC20+PC22: GPU rails. PSTR: SMC system total.
static const char *KEYS[] = {"PDTR", "PPBR", "PC10", "PC12", "PC20", "PC22", "PSTR"};
#define NK 7
#define MAXS 120000
static SMCKeyInfo g_ki[NK]; static bool g_kok[NK];
static double g_st[MAXS]; static float g_sv[MAXS][NK];
static atomic_int g_ns, g_stop;

// IOReport: GPU energy and the GPU's DVFS-state residency, no root
typedef CFDictionaryRef (*CopyChannelsInGroup_t)(CFStringRef, CFStringRef, uint64_t, uint64_t, uint64_t);
typedef void (*MergeChannels_t)(CFDictionaryRef, CFDictionaryRef, CFTypeRef);
typedef void *(*CreateSubscription_t)(void *, CFMutableDictionaryRef, CFMutableDictionaryRef *, uint64_t, CFTypeRef);
typedef CFDictionaryRef (*CreateSamples_t)(void *, CFMutableDictionaryRef, CFTypeRef);
typedef CFDictionaryRef (*CreateSamplesDelta_t)(CFDictionaryRef, CFDictionaryRef, CFTypeRef);
typedef int64_t (*SimpleGetIntegerValue_t)(CFDictionaryRef, int32_t);
typedef CFStringRef (*GetStr_t)(CFDictionaryRef);
typedef int32_t (*StateGetCount_t)(CFDictionaryRef);
typedef int64_t (*StateGetResidency_t)(CFDictionaryRef, int32_t);
static CreateSamples_t IOR_CreateSamples; static CreateSamplesDelta_t IOR_Delta;
static SimpleGetIntegerValue_t IOR_Int; static GetStr_t IOR_Name;
static StateGetCount_t IOR_SCount; static StateGetResidency_t IOR_SRes;
static void *g_sub; static CFMutableDictionaryRef g_subbed;
static bool g_ior;
// GPU DVFS states of this M2 Max (ioreg voltage-states9), MHz
static const double GPU_MHZ[] = {0, 444, 612, 808, 968, 1110, 1236, 1338, 1398};
#define MAXIS 20000
static double g_it[MAXIS]; static float g_iv[MAXIS][3];   // mid-time; GPU W, mean active MHz, busy fraction
static atomic_int g_nis;
static bool ior_init(bool energy) {   // energy: the "Energy Model" group too (GPU energy), else GPU states only
    void *h = dlopen("/usr/lib/libIOReport.dylib", RTLD_NOW);
    if (!h) return false;
    CopyChannelsInGroup_t CopyCh = (CopyChannelsInGroup_t)dlsym(h, "IOReportCopyChannelsInGroup");
    MergeChannels_t Merge = (MergeChannels_t)dlsym(h, "IOReportMergeChannels");
    CreateSubscription_t Sub = (CreateSubscription_t)dlsym(h, "IOReportCreateSubscription");
    IOR_CreateSamples = (CreateSamples_t)dlsym(h, "IOReportCreateSamples");
    IOR_Delta = (CreateSamplesDelta_t)dlsym(h, "IOReportCreateSamplesDelta");
    IOR_Int = (SimpleGetIntegerValue_t)dlsym(h, "IOReportSimpleGetIntegerValue");
    IOR_Name = (GetStr_t)dlsym(h, "IOReportChannelGetChannelName");
    IOR_SCount = (StateGetCount_t)dlsym(h, "IOReportStateGetCount");
    IOR_SRes = (StateGetResidency_t)dlsym(h, "IOReportStateGetResidency");
    if (!CopyCh || !Merge || !Sub || !IOR_CreateSamples || !IOR_Delta || !IOR_Int || !IOR_Name || !IOR_SCount || !IOR_SRes) return false;
    CFDictionaryRef en = energy ? CopyCh(CFSTR("Energy Model"), NULL, 0, 0, 0) : NULL;
    CFDictionaryRef gp = CopyCh(CFSTR("GPU Stats"), CFSTR("GPU Performance States"), 0, 0, 0);
    if (!(energy ? en : gp)) return false;
    CFMutableDictionaryRef ch = CFDictionaryCreateMutableCopy(kCFAllocatorDefault, 0, energy ? en : gp);
    if (energy && gp) Merge(ch, gp, NULL);
    g_sub = Sub(NULL, ch, &g_subbed, 0, NULL);
    return g_sub != NULL;
}
// GPU energy (J) and the mean active GPU MHz and busy fraction between two samples
static void ior_delta(CFDictionaryRef s0, CFDictionaryRef s1, double *gpuJ, double *mhz, double *busy) {
    *gpuJ = *mhz = *busy = NAN;
    CFDictionaryRef d = IOR_Delta(s0, s1, NULL);
    if (!d) return;
    CFArrayRef items = CFDictionaryGetValue(d, CFSTR("IOReportChannels"));
    for (CFIndex i = 0; items && i < CFArrayGetCount(items); i++) {
        CFDictionaryRef it = CFArrayGetValueAtIndex(items, i);
        CFStringRef n = IOR_Name(it);
        if (!n) continue;
        if (CFStringCompare(n, CFSTR("GPU Energy"), 0) == kCFCompareEqualTo) *gpuJ = IOR_Int(it, 0) * 1e-9;
        if (CFStringCompare(n, CFSTR("GPUPH"), 0) == kCFCompareEqualTo) {
            int cnt = IOR_SCount(it); double tot = 0, act = 0, wf = 0;
            for (int k = 0; k < cnt; k++) {
                double r = (double)IOR_SRes(it, k); tot += r;
                if (k >= 1 && k <= 8) { act += r; wf += r * GPU_MHZ[k]; }
            }
            if (act > 0) *mhz = wf / act;
            if (tot > 0) *busy = act / tot;
        }
    }
    CFRelease(d);
}
// SMC at 20 Hz; with g_ior also IOReport every 0.5 s
static void *sampler(void *arg) {
    (void)arg;
    CFDictionaryRef prev = NULL; double tprev = 0; int it = 0;
    while (!atomic_load(&g_stop)) {
        int n = atomic_load(&g_ns);
        if (n < MAXS) {
            g_st[n] = now();
            for (int k = 0; k < NK; k++) { float f = NAN; if (g_kok[k]) smc_flt(k4(KEYS[k]), g_ki[k], &f); g_sv[n][k] = f; }
            atomic_store(&g_ns, n + 1);
        }
        if (g_ior && it++ % 10 == 0) {
            CFDictionaryRef s = IOR_CreateSamples(g_sub, g_subbed, NULL); double t = now();
            int k = atomic_load(&g_nis);
            if (prev && s && k < MAXIS) {
                double j, mhz, busy; ior_delta(prev, s, &j, &mhz, &busy);
                g_it[k] = 0.5 * (t + tprev); g_iv[k][0] = (float)(j / (t - tprev)); g_iv[k][1] = (float)mhz; g_iv[k][2] = (float)busy;
                atomic_store(&g_nis, k + 1);
            }
            if (prev) CFRelease(prev);
            prev = s; tprev = t;
        }
        usleep(50000);
    }
    if (prev) CFRelease(prev);
    return NULL;
}
// mean of IOReport field f (0 GPU W, 1 MHz, 2 busy) over samples whose mid-time lies in any window
static double ior_mean(int f, const double *t0, const double *t1, int nw, int *count) {
    int n = atomic_load(&g_nis), cnt = 0; double s = 0;
    for (int i = 0; i < n; i++) {
        bool in = false; for (int w = 0; w < nw && !in; w++) in = g_it[i] >= t0[w] && g_it[i] <= t1[w];
        if (in && !isnan(g_iv[i][f])) { s += g_iv[i][f]; cnt++; }
    }
    if (count) *count = cnt;
    return cnt ? s / cnt : NAN;
}
// Outside load (run mode): process 0 snapshots the host's CPU ticks (busy and total, over all CPUs),
// this run's own CPU time (parent and children, proc_pidinfo) and, unless SPLIT_SAMPLER=0, an IOReport
// sample of the GPU's states, at the fixed block's round boundaries, between timed calls. Other
// processes' CPU over a stretch is the busy share times the CPUs minus our own. kernel_task's work
// counts as outside: root's processes cannot be read one by one without root. Over the GPU-idle
// stretch it has none of this run's GPU driver work; over the whole round it has.
static pid_t g_own[NPMAX + 1]; static int g_nown, g_ncpu;
static void load_init(void) {
    for (int p = 0; p < shm->np; p++) g_own[g_nown++] = shm->pid[p];
    g_own[g_nown++] = getppid();
    g_ior = sampler_on() && ior_init(false);
}
static void host_ticks(double *busy, double *total) {
    natural_t n; processor_info_array_t info; mach_msg_type_number_t cnt;
    *busy = *total = NAN;
    if (host_processor_info(mach_host_self(), PROCESSOR_CPU_LOAD_INFO, &n, &info, &cnt) != KERN_SUCCESS) return;
    const processor_cpu_load_info_t l = (processor_cpu_load_info_t)info;
    double b = 0, t = 0;
    for (natural_t i = 0; i < n; i++) {
        for (int k = 0; k < CPU_STATE_MAX; k++) t += l[i].cpu_ticks[k];
        b += (double)l[i].cpu_ticks[CPU_STATE_USER] + l[i].cpu_ticks[CPU_STATE_SYSTEM] + l[i].cpu_ticks[CPU_STATE_NICE];
    }
    vm_deallocate(mach_task_self(), (vm_address_t)info, cnt * sizeof(integer_t));
    *busy = b; *total = t; g_ncpu = (int)n;
}
static void load_snap(LoadSnap *s) {
    s->t = now();
    host_ticks(&s->busy, &s->total);
    s->own = 0;
    for (int i = 0; i < g_nown; i++) {
        struct proc_taskinfo ti;
        if (proc_pidinfo(g_own[i], PROC_PIDTASKINFO, 0, &ti, sizeof ti) == (int)sizeof ti) s->own += mach_s(ti.pti_total_user + ti.pti_total_system);
    }
    s->ior = g_ior ? IOR_CreateSamples(g_sub, g_subbed, NULL) : NULL;
}
static double load_fcpu(const LoadSnap *a, const LoadSnap *b) {   // cores
    if (!(b->total > a->total) || !(b->t > a->t)) return NAN;
    return (b->busy - a->busy) / (b->total - a->total) * g_ncpu - (b->own - a->own) / (b->t - a->t);
}
static double load_gpu_busy(const LoadSnap *a, const LoadSnap *b) {
    double j, mhz, busy = NAN;
    if (a->ior && b->ior) ior_delta(a->ior, b->ior, &j, &mhz, &busy);
    return busy;
}
static void load_free(LoadSnap *s) { if (s->ior) CFRelease(s->ior); s->ior = NULL; }

static bool smc_open(void) {
    io_service_t svc = IOServiceGetMatchingService(kIOMainPortDefault, IOServiceMatching("AppleSMC"));
    if (!svc || IOServiceOpen(svc, mach_task_self(), 0, &g_smc) != KERN_SUCCESS) return false;
    for (int k = 0; k < NK; k++) g_kok[k] = smc_info(k4(KEYS[k]), &g_ki[k]) == 0;
    return g_kok[0];
}
// mean of key k (k < 0: GPU rails summed) over samples inside any window [t0, t1] of the list
static double smc_mean(int k, const double *t0, const double *t1, int nw, int *count) {
    int n = atomic_load(&g_ns), cnt = 0; double s = 0;
    for (int i = 0; i < n; i++) {
        bool in = false; for (int w = 0; w < nw && !in; w++) in = g_st[i] >= t0[w] && g_st[i] <= t1[w];
        if (!in) continue;
        double val = k >= 0 ? g_sv[i][k] : (double)g_sv[i][2] + g_sv[i][3] + g_sv[i][4] + g_sv[i][5];
        if (!isnan(val)) { s += val; cnt++; }
    }
    if (count) *count = cnt;
    return cnt ? s / cnt : NAN;
}

// the samples as TSV: SMC keys over time, and IOReport's GPU power, clock and busy fraction
static void dump_smc(const char *path) {
    FILE *f = fopen(path, "w");
    if (!f) return;
    fprintf(f, "t"); for (int k = 0; k < NK; k++) fprintf(f, "\t%s", KEYS[k]); fprintf(f, "\n");
    int n = atomic_load(&g_ns);
    for (int i = 0; i < n; i++) { fprintf(f, "%.3f", g_st[i]); for (int k = 0; k < NK; k++) fprintf(f, "\t%.3f", g_sv[i][k]); fprintf(f, "\n"); }
    fclose(f);
}
static void dump_ior(const char *path) {
    FILE *f = fopen(path, "w");
    if (!f) return;
    fprintf(f, "t\tgpu_w\tgpu_mhz\tgpu_busy\n");
    int n = atomic_load(&g_nis);
    for (int i = 0; i < n; i++) fprintf(f, "%.3f\t%.2f\t%.1f\t%.4f\n", g_it[i], g_iv[i][0], g_iv[i][1], g_iv[i][2]);
    fclose(f);
}
// `pmset -g batt` on one line with the adapter's watts (`| adapter 140 W`), and the load averages
static void machine_state(char *pw, size_t npw, char *ld, size_t nld) {
    sh_line("pmset -g batt | tr '\\n' ' '; pmset -g adapter | sed -nE 's/^ *Wattage = ([0-9]+)W.*/| adapter \\1 W /p'", pw, npw);
    sh_line("sysctl -n vm.loadavg", ld, nld);
}
// on AC power from an adapter of at least SPLIT_MIN_W watts (default 90), by machine_state's line
static bool power_ok(const char *pw) {
    const char *a = strstr(pw, "| adapter "); double w;
    return strstr(pw, "'AC Power'") && a && sscanf(a, "| adapter %lf", &w) == 1 && w >= envd("SPLIT_MIN_W", 90);
}

// ---------------------------------------------------------------- parent: report
// config names, as raw.tsv's name column has them: ACC, GPU, S<g> (the user split), S<g>b (its twin),
// SEAM, HEAD, TAIL
static const char *cfg_name(int k, char *buf) {
    switch (shm->kind[k]) {
    case KD_ACC: return "ACC";
    case KD_GPU: return "GPU";
    case KD_SEAM: return "SEAM";
    case KD_HEAD: return "HEAD";
    case KD_TAIL: return "TAIL";
    default: snprintf(buf, 24, "S%d%s", shm->g[k], shm->kind[k] == KD_SPLIT2 ? "b" : ""); return buf;
    }
}
static const char *OUTCOME[] = {"KEPT", "KEPT_4M", "CPU", "RANGE", "GPU_FAILED", "DECLINED", "NOT_CUT", "INVALID"};
#define NOUT (int)(sizeof OUTCOME / sizeof OUTCOME[0])
static const char *outcome_name(int o) { return o >= 0 && o < NOUT ? OUTCOME[o] : "-"; }
// median over (round, rep) of the fixed block's lockstep time of config a over config b's
static double paired_median(int a, int b, int *count) {
    static double r[MAXREC]; int n = 0;
    for (int i = 0; i < shm->nrec[0]; i++) {
        const Rec *x = &shm->rec[0][i];
        if (x->phase != PH_FIXED || x->cfg != a) continue;
        for (int j = 0; j < shm->nrec[0]; j++) {
            const Rec *y = &shm->rec[0][j];
            if (y->phase == PH_FIXED && y->cfg == b && y->round == x->round && y->rep == x->rep) { r[n++] = lockv(i, V_TOTAL) / lockv(j, V_TOTAL); break; }
        }
    }
    if (count) *count = n;
    return median(r, n);
}
// median over every process's SEAM calls in the fixed block of a span of the seam's timeline (ms)
// or of a count; spans whose ends did not happen (a 0 stamp) are left out
enum { SM_INIT, SM_START, SM_CPU, SM_GPU, SM_WAIT, SM_AFTER, SM_PRE, SM_EXEC, SM_POST, SM_SUBMIT, SM_WAKE, SM_ENCODE,
       SM_CMDBUFS, SM_DISPATCHES, SM_GROUPS, SM_INPLACE };
static double seam_med(int what) {
    static double v[NPMAX * MAXREC]; int n = 0;
    for (int p = 0; p < shm->np; p++) for (int i = 0; i < shm->nrec[p]; i++) {
        const Rec *r = &shm->rec[p][i];
        if (r->phase != PH_FIXED || r->cfg != shm->k_seam) continue;
        const ab_split_times_t *m = &r->k.tm; const ab_prof_kernel_split_t *k = &m->gpu;
        uint64_t a = 0, b = 0; double cnt = NAN;
        switch (what) {
        case SM_INIT: a = m->cut_ns; b = m->create_ns; break;          // cut to the GPU thread's creation
        case SM_START: a = m->create_ns; b = m->gpu0_ns; break;        // creation to the GPU part's start
        case SM_CPU: a = m->cpu0_ns; b = m->cpu1_ns; break;
        case SM_GPU: a = m->gpu0_ns; b = m->gpu1_ns; break;
        case SM_WAIT: a = m->cpu1_ns; b = m->join_ns; break;           // the caller's wait for the GPU part
        case SM_AFTER: a = m->join_ns; b = m->end_ns; break;           // the K axis's sum, or a redo
        case SM_PRE: a = m->gpu0_ns; b = k->commit_ns; break;          // the GPU part's host work before its first commit
        case SM_EXEC: a = k->gpu_start_ns; b = k->gpu_end_ns; break;   // GPUStartTime to GPUEndTime
        case SM_POST: a = k->done_ns; b = m->gpu1_ns; break;           // after the last command buffer returned
        case SM_SUBMIT: cnt = k->submit_ns * 1e-6; break;
        case SM_WAKE: cnt = k->wake_ns * 1e-6; break;
        case SM_ENCODE: cnt = k->encode_ns * 1e-6; break;
        case SM_CMDBUFS: cnt = k->cmdbufs; break;
        case SM_DISPATCHES: cnt = k->dispatches; break;
        case SM_GROUPS: cnt = (double)k->groups; break;
        case SM_INPLACE: cnt = r->k.in_place; break;
        }
        if (what < SM_SUBMIT) { if (!a || !b) continue; cnt = ((double)b - (double)a) * 1e-6; }
        v[n++] = cnt;
    }
    return median(v, n);
}
static int windows(int phase, int cfg, double *t0, double *t1) {
    int n = 0;
    for (int i = 0; i < shm->nrec[0]; i++) {
        const Rec *r = &shm->rec[0][i]; if (r->phase != phase || r->cfg != cfg) continue;
        double a = 1e300, b = 0; for (int p = 0; p < shm->np; p++) { a = fmin(a, shm->rec[p][i].k.v[V_T0]); b = fmax(b, shm->rec[p][i].k.v[V_T1]); }
        t0[n] = a; t1[n] = b; n++;
    }
    return n;
}
static void dump_rounds(const char *path) {   // the outside load per round (Shm)
    FILE *f = fopen(path, "w");
    if (!f) return;
    fprintf(f, "round\tt0\tidle_t1\tt1\tfcpu_idle\tgpu_busy_idle\tfcpu_round\n");
    for (int r = 0; r < shm->rounds; r++)
        fprintf(f, "%d\t%.6f\t%.6f\t%.6f\t%.3f\t%.4f\t%.3f\n", r, shm->rt0[r], shm->idle_t1[r], shm->rt1[r], shm->fcpu_idle[r], shm->gpu_idle[r], shm->fcpu_round[r]);
    fclose(f);
}
static void report(const Case *c, int np, int data, bool smc, double idle_w) {
    char b1[24], b2[24];
    double F = flops(c) * np;
    printf("case %s: %cGEMM '%c','N' M=%d N=%d K=%d lda=%d ldb=%d ldc=%d beta=%d, split axis %c (align %d), %.2f GF per call x %d procs, data %s\n",
           c->name, c->prec == 'z' ? 'Z' : 'D', c->ta, c->M, c->N, c->K, c->lda, c->ldb, c->ldc, c->beta, c->axis, c->align, flops(c) / 1e9, np, data ? "QE-like (rows decay 9 decades)" : "uniform [-1,1)");
    printf("registration ok (of 4 arrays), per proc:"); for (int p = 0; p < np; p++) printf(" %d", shm->reg_ok[p]); printf("\n");
    if (shm->fixed) printf("-- sweep: none, the GPU part is fixed (SPLIT_G or SPLIT_F)\n");
    else printf("-- sweep: lockstep medians over 3 interleaved rounds x %d reps (ms); L = slowest proc, A = mean over procs\n", c->sweep_reps);
    double sacc = rstat(PH_SWEEP, CF_ACC, -1, V_TOTAL, true, NULL), sgpu = rstat(PH_SWEEP, CF_GPU, -1, V_TOTAL, true, NULL);
    if (!shm->fixed) printf("   %-6s %6s | %9s %9s | %7s %7s | %8s %8s %7s %6s\n", "config", "f", "L ms", "A ms", "/ACC", "/GPU", "cpu ms", "gpu ms", "join", "wake");
    for (int k = 0; k < (shm->fixed ? 0 : shm->ncfg); k++) {
        double L = rstat(PH_SWEEP, k, -1, V_TOTAL, true, NULL), A = rstat(PH_SWEEP, k, -1, V_TOTAL, false, NULL);
        double f = (double)cfg_g(c, k) / axis_len(c);
        printf("   %-6s %6.3f | %9.2f %9.2f | %7.3f %7.3f | %8.2f %8.2f %7.2f %6.2f%s\n", cfg_name(k, b1), f, L * 1e3, A * 1e3, L / sacc, L / sgpu,
               rstat(PH_SWEEP, k, -1, V_CPU, false, NULL) * 1e3, rstat(PH_SWEEP, k, -1, V_GPU, false, NULL) * 1e3,
               rstat(PH_SWEEP, k, -1, V_JOIN, false, NULL) * 1e3, rstat(PH_SWEEP, k, -1, V_WAKE, false, NULL) * 1e3, k == shm->best ? "  <- best" : "");
    }
    if (!shm->fixed && (shm->best == CF_SPLIT0 || shm->best == shm->ncfg - 1)) printf("   NOTE: best share is at the edge of the swept range\n");
    int bk = shm->best, na, ng, ns;
    double fb = (double)shm->g[bk] / axis_len(c);
    double La = rstat(PH_FIXED, CF_ACC, -1, V_TOTAL, true, &na), Lg = rstat(PH_FIXED, CF_GPU, -1, V_TOTAL, true, &ng), Ls = rstat(PH_FIXED, bk, -1, V_TOTAL, true, &ns);
    printf("-- fixed block at g=%d (f=%.3f, %s), %d interleaved rounds%s; medians of lockstep times (ms) [per-round medians]\n", shm->g[bk], fb,
           shm->fixed ? "fixed" : "the sweep's best", shm->rounds,
           shm->narm && shm->rounds % NW6 ? " (the trio's order is balanced over multiples of 6 rounds; this is not one)" : "");
    int cf[3] = {CF_ACC, CF_GPU, bk};
    int all[3 + NARM] = {CF_ACC, CF_GPU, bk}, nall = 3;
    if (shm->narm) { int arms[NARM] = {shm->k_seam, shm->k_split2, shm->k_head, shm->k_tail}; for (int a = 0; a < NARM; a++) all[nall++] = arms[a]; }
    double gb = (double)shm->g[bk] / axis_len(c);
    for (int i = 0; i < nall; i++) {   // GF/s of the part the config computes: HEAD the head's, TAIL the tail's
        int k = all[i], n; double L = rstat(PH_FIXED, k, -1, V_TOTAL, true, &n), A = rstat(PH_FIXED, k, -1, V_TOTAL, false, NULL);
        double part = shm->kind[k] == KD_HEAD ? 1 - gb : shm->kind[k] == KD_TAIL ? gb : 1;
        printf("   %-6s n=%2d  L %9.2f  A %9.2f  %7.1f GF/s  [", cfg_name(k, b1), n, L * 1e3, A * 1e3, F * part / L / 1e9);
        for (int r = 0; r < shm->rounds; r++) printf("%s%.2f", r ? " " : "", rstat(PH_FIXED, k, r, V_TOTAL, true, NULL) * 1e3);
        printf("]\n");
    }
    printf("   split parts (mean over procs, medians): cpu slice %.2f ms, gpu slice %.2f ms, join wait %.2f ms, gpu wake %.3f ms, K-add %.3f ms\n",
           rstat(PH_FIXED, bk, -1, V_CPU, false, NULL) * 1e3, rstat(PH_FIXED, bk, -1, V_GPU, false, NULL) * 1e3, rstat(PH_FIXED, bk, -1, V_JOIN, false, NULL) * 1e3,
           rstat(PH_FIXED, bk, -1, V_WAKE, false, NULL) * 1e3, rstat(PH_FIXED, bk, -1, V_KADD, false, NULL) * 1e3);
    double best1 = fmin(La, Lg);
    printf("   speedup of the split: vs Accelerate %.3fx, vs GPU %.3fx, vs the faster engine (%s) %.3fx\n", La / Ls, Lg / Ls, La <= Lg ? "ACC" : "GPU", best1 / Ls);
    // the arms
    double Lm = NAN, Ls2 = NAN, Lh = NAN, Lt = NAN, rseam = NAN, rnull = NAN; int nseam = 0, nnull = 0, outs[NOUT] = {0}, nout = 0;
    if (shm->narm) {
        Lm = rstat(PH_FIXED, shm->k_seam, -1, V_TOTAL, true, NULL); Ls2 = rstat(PH_FIXED, shm->k_split2, -1, V_TOTAL, true, NULL);
        Lh = rstat(PH_FIXED, shm->k_head, -1, V_TOTAL, true, NULL); Lt = rstat(PH_FIXED, shm->k_tail, -1, V_TOTAL, true, NULL);
        rseam = paired_median(shm->k_seam, bk, &nseam); rnull = paired_median(shm->k_split2, bk, &nnull);
        for (int p = 0; p < np; p++) for (int i = 0; i < shm->nrec[p]; i++) {
            const Rec *r = &shm->rec[p][i];
            if (r->phase == PH_FIXED && r->cfg == shm->k_seam && r->k.outcome >= 0 && r->k.outcome < NOUT) { outs[r->k.outcome]++; nout++; }
        }
        printf("-- library seam and alone arms at g=%d (lockstep medians; ratios: median over (round, rep) pairs)\n", shm->g[bk]);
        printf("   SEAM / %s %.3f (%d pairs), %s / %s %.3f (%d pairs, the user split against itself)\n", cfg_name(bk, b1), rseam, nseam,
               cfg_name(shm->k_split2, b2), cfg_name(bk, b1), rnull, nnull);
        printf("   seam outcomes (%d calls):", nout);
        for (int o = 0; o < NOUT; o++) if (outs[o]) printf(" %s %d", OUTCOME[o], outs[o]);
        printf("; operands in place (median) %.0f\n", seam_med(SM_INPLACE));
        printf("   seam timeline (medians over all procs' calls, ms): cut to thread %.3f, thread to GPU part %.3f | cpu part %.2f, gpu part %.2f, "
               "wait for the GPU %.2f, after the join %.3f | GPU part: host before first commit %.2f, submit %.3f, GPU exec %.2f, wake %.3f, "
               "after last done %.2f, encode %.3f | %.0f command buffers, %.0f dispatches, %.0f threadgroups\n",
               seam_med(SM_INIT), seam_med(SM_START), seam_med(SM_CPU), seam_med(SM_GPU), seam_med(SM_WAIT), seam_med(SM_AFTER), seam_med(SM_PRE),
               seam_med(SM_SUBMIT), seam_med(SM_EXEC), seam_med(SM_WAKE), seam_med(SM_POST), seam_med(SM_ENCODE),
               seam_med(SM_CMDBUFS), seam_med(SM_DISPATCHES), seam_med(SM_GROUPS));
        double hc = rstat(PH_FIXED, shm->k_head, -1, V_CPU, false, NULL), tg = rstat(PH_FIXED, shm->k_tail, -1, V_GPU, false, NULL);
        double sc = rstat(PH_FIXED, bk, -1, V_CPU, false, NULL), sg = rstat(PH_FIXED, bk, -1, V_GPU, false, NULL);
        printf("   alone: HEAD %.2f ms (cpu slice %.2f), TAIL %.2f ms (gpu slice %.2f); (HEAD + TAIL) / split %.3f; in the split the cpu slice takes "
               "%.3fx its time alone, the gpu slice %.3fx\n", Lh * 1e3, hc * 1e3, Lt * 1e3, tg * 1e3, (Lh + Lt) / Ls, sc / hc, sg / tg);
    }
    // bits
    int sr = 0, sd = 0, gr = 0, gd = 0, ar = 0, ad = 0; long smax = 0, od = 0; bool hs = true, hg = true, ha = true;
    for (int p = 0; p < np; p++) {
        sr += shm->split_reps[p]; sd += shm->split_diff_reps[p]; gr += shm->gpu_reps[p]; gd += shm->gpu_diff_reps[p]; ar += shm->acc_reps[p]; ad += shm->acc_diff_reps[p];
        if (shm->split_diff_max[p] > smax) smax = shm->split_diff_max[p];
        od += shm->order_diff[p][0] + shm->order_diff[p][1];
        hs &= shm->hash_split[p] == shm->hash_split[0]; hg &= shm->hash_gpu[p] == shm->hash_gpu[0]; ha &= shm->hash_acc[p] == shm->hash_acc[0];
    }
    printf("-- bits: split repeats differing from the first %d/%d (max %ld elements), forced completion orders differing %ld elements, cross-proc hashes equal: split %s gpu %s acc %s\n",
           sd, sr, smax, od, hs ? "yes" : "NO", hg ? "yes" : "NO", ha ? "yes" : "NO");
    printf("   single engines: GPU repeats differing %d/%d, Accelerate repeats differing %d/%d\n", gd, gr, ad, ar);
    if (c->axis != 'K') printf("   split CPU part vs Accelerate alone: %ld/%ld elements differ; GPU part vs GPU alone: %ld/%ld differ\n", shm->cpu_vs_acc, shm->part_n_cpu, shm->gpu_vs_gpu, shm->part_n_gpu);
    printf("   whole split C vs Accelerate alone %ld/%ld, vs GPU alone %ld/%ld elements differ\n", shm->split_vs_acc, shm->all_n, shm->split_vs_gpu, shm->all_n);
    int mr = 0, md = 0, s2r = 0, s2d = 0; long mvs = -1, hvs = -1, tvs = -1, part_h = 0, part_t = 0;
    if (shm->narm) {   // the arms' bits: summed over processes; element counts from the process that differs most
        for (int p = 0; p < np; p++) {
            mr += shm->seam_reps[p]; md += shm->seam_diff_reps[p]; s2r += shm->split2_reps[p]; s2d += shm->split2_diff_reps[p];
            mvs = shm->seam_vs_split[p] > mvs ? shm->seam_vs_split[p] : mvs;
            hvs = shm->head_vs_split[p] > hvs ? shm->head_vs_split[p] : hvs; tvs = shm->tail_vs_split[p] > tvs ? shm->tail_vs_split[p] : tvs;
        }
        int L = axis_len(c), g = shm->g[bk];
        part_h = c->axis == 'N' ? (long)c->M * (L - g) : (long)(L - g) * c->N; part_t = c->axis == 'N' ? (long)c->M * g : (long)g * c->N;
        printf("   seam: repeats differing from its first %d/%d; its C vs the user split's %ld/%ld elements differ | %s repeats differing from %s's first %d/%d\n",
               md, mr, mvs, shm->all_n, cfg_name(shm->k_split2, b2), cfg_name(bk, b1), s2d, s2r);
        if (c->axis != 'K') printf("   alone: HEAD's part vs the split's %ld/%ld elements differ, TAIL's %ld/%ld (first call per proc)\n", hvs, part_h, tvs, part_t);
    }
    int notgpu = 0, gcalls = 0; for (int p = 0; p < np; p++) { notgpu += shm->notgpu[p]; gcalls += shm->gpucalls[p]; }
    printf("   GPU-routed calls whose C was not computed on the GPU (ab_get_last_dispatch_path): %d of %d\n", notgpu, gcalls);
    double emax = c->axis == 'K' ? shm->err_split_cpu : fmax(shm->err_split_cpu, shm->err_split_gpu);
    printf("-- error vs Dot2 on %d sampled columns, max |C-R|/(|op(A)||op(B)|+|beta C0|) in u^2 = 2^-48 (u = 2^-24; bound 8 u^2): ", shm->err_cols);
    if (c->axis == 'K') printf("split %.3f", shm->err_split_cpu / U2);
    else printf("split CPU part %.4f, GPU part %.3f", shm->err_split_cpu / U2, shm->err_split_gpu / U2);
    printf(" | GPU alone %.3f | Accelerate alone %.4f\n", shm->err_gpu / U2, shm->err_acc / U2);
    // power
    double pw[3] = {NAN, NAN, NAN}, gw[3] = {NAN, NAN, NAN}; int pc[3] = {0, 0, 0};
    if (smc) {
        static double t0[MAXREC], t1[MAXREC];
        for (int i = 0; i < 3; i++) {
            int nw = windows(PH_FIXED, cf[i], t0, t1);
            pw[i] = smc_mean(0, t0, t1, nw, &pc[i]); gw[i] = smc_mean(-1, t0, t1, nw, NULL);
        }
        printf("-- power, fixed block, mean over the timed calls (20 Hz SMC): idle %.1f W | ACC %.1f W (%d samples), GPU %.1f W (%d), split %.1f W (%d); GPU rails ACC %.1f GPU %.1f split %.1f W\n",
               idle_w, pw[0], pc[0], pw[1], pc[1], pw[2], pc[2], gw[0], gw[1], gw[2]);
        printf("   energy per call above idle (J): ACC %.2f, GPU %.2f, split %.2f (np procs together)\n", (pw[0] - idle_w) * La, (pw[1] - idle_w) * Lg, (pw[2] - idle_w) * Ls);
    } else printf("-- power: not sampled (%s); power fields read nan\n", sampler_on() ? "SMC unavailable" : "SPLIT_SAMPLER=0");
    double sw[3] = {NAN, NAN, NAN}, sj[3] = {NAN, NAN, NAN};
    if (smc && shm->pw_n[0]) {  // sustained runs: trim SMC's ~1 s lag from each run's start
        printf("-- sustained power, %d/%d/%d calls per run x 3 interleaved rounds, mean over each run from 1 s after its start (wall = PDTR):\n",
               shm->pw_n[0], shm->pw_n[1], shm->pw_n[2]);
        for (int i = 0; i < 3; i++) {
            double a[3], b[3], tt = 0; int cnt;
            for (int r = 0; r < 3; r++) { a[r] = shm->pw_t0[r][i] + 1.0; b[r] = shm->pw_t1[r][i]; tt += shm->pw_t1[r][i] - shm->pw_t0[r][i]; }
            sw[i] = smc_mean(0, a, b, 3, &cnt); double rails = smc_mean(-1, a, b, 3, NULL), batt = smc_mean(1, a, b, 3, NULL), sys = smc_mean(6, a, b, 3, NULL);
            double mhz = ior_mean(1, a, b, 3, NULL), gpuw = ior_mean(0, a, b, 3, NULL);
            double per = tt / (3.0 * shm->pw_n[i]); sj[i] = sw[i] * per;
            printf("   %-6s %6.1f W wall, %6.1f W system (GPU rails %5.1f W, battery %+.1f W; %d samples) | GPU %4.0f MHz, %5.1f W (IOReport) | %.2f ms per call -> %.2f J per call at the wall\n",
                   cfg_name(cf[i], b1), sw[i], sys, rails, batt, cnt, mhz, gpuw, per * 1e3, sj[i]);
        }
    }
    for (int p = 0; p < np; p++) printf("   proc %d: setup %.1f s, total %.1f s\n", p, shm->tsetup[p], shm->ttotal[p]);
    double fcpu_max = NAN, fround_max = NAN, bmax = NAN;   // fmax() skips a NAN
    if (shm->narm) {
        for (int r = 0; r < shm->rounds; r++) { fcpu_max = fmax(fcpu_max, shm->fcpu_idle[r]); bmax = fmax(bmax, shm->gpu_idle[r]); }
        printf("   outside load per round, over the idle gap and CPU-only configs: other processes' CPU, cores [");
        for (int r = 0; r < shm->rounds; r++) printf("%s%.2f", r ? " " : "", shm->fcpu_idle[r]);
        printf("], GPU busy [");
        for (int r = 0; r < shm->rounds; r++) printf("%s%.3f", r ? " " : "", shm->gpu_idle[r]);
        printf("]\n");
    }
    for (int r = 0; r < shm->rounds; r++) fround_max = fmax(fround_max, shm->fcpu_round[r]);
    printf("   other processes' CPU over each whole round, cores (with this run's GPU driver work) [");
    for (int r = 0; r < shm->rounds; r++) printf("%s%.2f", r ? " " : "", shm->fcpu_round[r]);
    printf("]\n");
    {   // load1 after each of process 0's calls (raw.tsv has every one)
        static double la[MAXREC]; int n = 0; double lo = 1e300, hi = -1e300;
        for (int i = 0; i < shm->nrec[0]; i++) if (shm->rec[0][i].phase == PH_FIXED) { double l = shm->rec[0][i].load1; la[n++] = l; lo = fmin(lo, l); hi = fmax(hi, l); }
        printf("   load1 in the fixed block (after each call, proc 0): median %.2f [%.2f-%.2f]\n", median(la, n), n ? lo : NAN, n ? hi : NAN);
    }
    bool bits = sd == 0 && od == 0 && hs;
    printf("SUMMARY case=%s np=%d data=%d acc_ms=%.3f gpu_ms=%.3f split_ms=%.3f g=%d f=%.3f sp_acc=%.3f sp_gpu=%.3f sp_best=%.3f acc_gfs=%.1f gpu_gfs=%.1f split_gfs=%.1f "
           "bits_repeat=%s split_reps=%d gpu_rep_diff=%d acc_rep_diff=%d err_u2=%.3f err_gpu_u2=%.3f err_acc_u2=%.4f notgpu=%d pdtr_acc=%.1f pdtr_gpu=%.1f pdtr_split=%.1f idle=%.1f n=%d/%d/%d "
           "reg=%d sus_w=%.1f/%.1f/%.1f sus_j=%.3f/%.3f/%.3f h_split=%016llx h_gpu=%016llx h_acc=%016llx share=%s rounds=%d reps=%d/%d/%d arms=%d fcpu_max=%.2f gpu_idle_busy_max=%.3f fcpu_round_max=%.2f",
           c->name, np, data, La * 1e3, Lg * 1e3, Ls * 1e3, shm->g[bk], fb, La / Ls, Lg / Ls, best1 / Ls, F / La / 1e9, F / Lg / 1e9, F / Ls / 1e9,
           bits ? "yes" : "NO", sr, gd, ad, emax / U2, shm->err_gpu / U2, shm->err_acc / U2, notgpu, pw[0], pw[1], pw[2], idle_w, na, ng, ns,
           shm->reg_ok[0], sw[0], sw[1], sw[2], sj[0], sj[1], sj[2],
           (unsigned long long)shm->hash_split[0], (unsigned long long)shm->hash_gpu[0], (unsigned long long)shm->hash_acc[0],
           shm->fixed ? "fixed" : "sweep", shm->rounds, c->fx_acc, c->fx_gpu, c->fx_split, shm->narm, fcpu_max, bmax, fround_max);
    if (shm->narm) {
        printf(" seam_ms=%.3f split2_ms=%.3f head_ms=%.3f tail_ms=%.3f seam_ratio=%.4f null_ratio=%.4f pairs=%d seam_out=", Lm * 1e3, Ls2 * 1e3, Lh * 1e3, Lt * 1e3, rseam, rnull, nseam);
        bool first = true;
        for (int o = 0; o < NOUT; o++) if (outs[o]) { printf("%s%s:%d", first ? "" : ",", OUTCOME[o], outs[o]); first = false; }
        if (first) printf("-");
        printf(" seam_rep_diff=%d/%d seam_vs_split=%ld split2_rep_diff=%d/%d head_vs_split=%ld tail_vs_split=%ld", md, mr, mvs, s2d, s2r, hvs, tvs);
    }
    printf("\n");
}

// ---------------------------------------------------------------- sustained runs (one process)
// Blocks, each `secs` long after `idle` seconds of idle: Accelerate alone first (a control), then in a
// shuffled order (SPLIT_SEED, default from the clock; printed) the split at GPU part g, the GPU alone,
// the CPU slice alone (HEAD) and the GPU slice alone (TAIL), then the split at the share that balances
// the split block's steady slice rates, and Accelerate alone again (the control at the end). C is never
// reset between calls, so the calls run back to back: with beta = 0 every call writes the same C, and
// every 5th is compared with the block's first; with beta = 1 C accumulates and is not compared.
// Every call is kept (the records grow as needed).
// block ids (the `run` column of .sustain.calls.tsv): 0-3 as before this order was shuffled
enum { SB_SPLIT, SB_GPU, SB_ACC, SB_BAL, SB_HEAD, SB_TAIL, SB_ACC_END, NSB };
static const char *const SB_NAME[NSB] = {"split", "GPU alone", "Accelerate alone", "split rebalanced", "CPU slice alone (HEAD)",
                                         "GPU slice alone (TAIL)", "Accelerate alone, again"};
typedef struct { int blk, cfg, g, i; double t0, t1, total, cpu, gpu, join, wake, kadd; } SRec;
static SRec *srec;
static int nsrec, capsr;
static double sb_part[NSB];   // the fraction of the call's flops each block's calls compute
// median of field f (0 GF/s, 1 cpu slice, 2 gpu slice, 3 join, 4 ms per call) over block b's calls starting in [a, z) s into the block
static double sstat(const Case *c, int b, double bt0, double a, double z, int f, int *count, double *mn, double *mx) {
    double *v = malloc(sizeof(double) * (nsrec + 1)); int n = 0;
    for (int i = 0; i < nsrec; i++) {
        const SRec *r = &srec[i]; double s = r->t0 - bt0;
        if (r->blk != b || s < a || s >= z) continue;
        v[n++] = f == 0 ? flops(c) * sb_part[b] / r->total / 1e9 : f == 1 ? r->cpu : f == 2 ? r->gpu : f == 3 ? r->join : r->total * 1e3;
    }
    double m = median(v, n), lo = 1e300, hi = -1e300;
    for (int i = 0; i < n; i++) { lo = fmin(lo, v[i]); hi = fmax(hi, v[i]); }
    if (count) *count = n;
    if (mn) *mn = n ? lo : NAN;
    if (mx) *mx = n ? hi : NAN;
    free(v); return m;
}
static int sustain(const Case *c, const char *outdir, int g, double secs, double idle_s, double steady) {
    char pw[256], ld[256], path[1100];
    alarm((unsigned)(NSB * (secs + idle_s)) + 900);   // never hold the machine lock on a hang
    machine_state(pw, sizeof pw, ld, sizeof ld);
    printf("%s | VECLIB_MAXIMUM_THREADS=%s AB_MODE=%s AB_MIN_GPU_DIM=%s AB_STAGING_MB=%s | sustained runs, 1 proc\nstart: %s| load %s\n", ab_build_info(),
           getenv("VECLIB_MAXIMUM_THREADS"), getenv("AB_MODE"), getenv("AB_MIN_GPU_DIM"), getenv("AB_STAGING_MB"), pw, ld);
    fflush(stdout);
    if (!power_ok(pw)) { printf("not on AC power from an adapter of SPLIT_MIN_W watts or more: exit 75\n"); return 75; }
    shm = mmap(NULL, sizeof(Shm), PROT_READ | PROT_WRITE, MAP_SHARED | MAP_ANON, -1, 0);
    if (shm == MAP_FAILED) { perror("mmap"); return 1; }
    // configs: 0 ACC, 1 GPU, 2 the split at g, 3 the split rebalanced, 4 HEAD and 5 TAIL at g
    memset(shm, 0, sizeof(Shm));
    const int kd[6] = {KD_ACC, KD_GPU, KD_SPLIT, KD_SPLIT, KD_HEAD, KD_TAIL};
    for (int k = 0; k < 6; k++) { shm->kind[k] = kd[k]; shm->g[k] = g; }
    const int bcfg[NSB] = {CF_SPLIT0, CF_GPU, CF_ACC, CF_SPLIT0 + 1, CF_SPLIT0 + 2, CF_SPLIT0 + 3, CF_ACC};
    double gl = (double)g / axis_len(c);
    for (int b = 0; b < NSB; b++) sb_part[b] = b == SB_HEAD ? 1 - gl : b == SB_TAIL ? gl : 1;
    // block order: the control, the four shuffled, the rebalanced split (it needs the split's), the control
    const char *seed_env = getenv("SPLIT_SEED");
    uint64_t seed = seed_env ? strtoull(seed_env, NULL, 0) : mach_absolute_time(), st_ = seed;
    int order[NSB] = {SB_ACC, SB_SPLIT, SB_GPU, SB_HEAD, SB_TAIL, SB_BAL, SB_ACC_END};
    for (int i = 4; i > 1; i--) { int j = 1 + (int)(splitmix64(&st_) % (uint64_t)i); int t = order[i]; order[i] = order[j]; order[j] = t; }
    Ctx *x = &X;
    int ok = setup_operands(x, c, 0);
    if (ok < 0) return 1;
    size_t es_ = x->es;
    void *Cref = palloc(x->nC * es_);
    bool smc = sampler_on() && smc_open(); g_ior = sampler_on() && ior_init(true);
    sem_go = dispatch_semaphore_create(0); sem_done = dispatch_semaphore_create(0); sem_cpu_done = dispatch_semaphore_create(0);
    pthread_t th, st; pthread_create(&th, NULL, gpu_thread, NULL);
    if (smc || g_ior) pthread_create(&st, NULL, sampler, NULL);
    Call call; const double *v = call.v;
    g_noreset = true;
    for (int k = 0; k < 6; k++) { set_cfg(x, k); one_call(x, k, &call); }   // warm-up
    bool check = c->beta == 0;
    double bt0[NSB], bt1[NSB], it0[NSB], it1[NSB]; int bn[NSB] = {0}, bchk[NSB] = {0}, bdiff[NSB] = {0};
    for (int s = 0; s < NSB; s++) {
        int b = order[s];
        machine_state(pw, sizeof pw, ld, sizeof ld);
        if (!power_ok(pw)) { printf("left AC power, or the adapter is under SPLIT_MIN_W watts, before run %d: exit 75\n", s); fflush(stdout); return 75; }
        if (b == SB_BAL) {  // the share that balances the split block's steady-state slice rates
            int cut0 = axis_len(c) - g;
            double Tc = sstat(c, SB_SPLIT, bt0[SB_SPLIT], steady, 1e9, 1, NULL, NULL, NULL), Tg = sstat(c, SB_SPLIT, bt0[SB_SPLIT], steady, 1e9, 2, NULL, NULL, NULL);
            double rc = cut0 / Tc, rg = g / Tg;
            shm->g[bcfg[SB_BAL]] = gpu_part(c, rg / (rc + rg));
        }
        int k = bcfg[b]; set_cfg(x, k);
        it0[b] = now(); sleep_s(idle_s); it1[b] = now();
        bt0[b] = now(); int i = 0;
        while (now() - bt0[b] < secs) {
            one_call(x, k, &call);
            if (nsrec == capsr) {
                capsr = capsr ? 2 * capsr : 65536;
                if (!(srec = realloc(srec, sizeof(SRec) * (size_t)capsr))) { fprintf(stderr, "out of memory for records\n"); return 1; }
            }
            srec[nsrec++] = (SRec){b, k, cfg_g(c, k), i, v[V_T0], v[V_T1], v[V_TOTAL], v[V_CPU], v[V_GPU], v[V_JOIN], v[V_WAKE], v[V_KADD]};
            if (check && i == 0) memcpy(Cref, x->C, x->nC * es_);
            else if (check && i % 5 == 0) { bchk[b]++; bdiff[b] += memcmp(Cref, x->C, x->nC * es_) != 0; }
            i++;
        }
        bt1[b] = now(); bn[b] = i;
    }
    if (smc || g_ior) { atomic_store(&g_stop, 1); pthread_join(st, NULL); }
    // ---- report
    printf("case %s: %cGEMM M=%d N=%d K=%d, split axis %c; %.2f GF per call; registration %d/4 arrays%s; IOReport %s\n",
           c->name, c->prec == 'z' ? 'Z' : 'D', c->M, c->N, c->K, c->axis, flops(c) / 1e9, ok, noreg() ? " (SPLIT_NOREG=1)" : "",
           g_ior ? "on" : sampler_on() ? "unavailable" : "off (SPLIT_SAMPLER=0)");
    printf("runs of %.0f s back to back, each after %.0f s idle; burst = calls starting in the first 10 s, steady = calls starting %.0f s or more into the run\n", secs, idle_s, steady);
    printf("order (SPLIT_SEED=%llu):", (unsigned long long)seed);
    for (int s = 0; s < NSB; s++) printf("%s %s", s ? "," : "", SB_NAME[order[s]]);
    printf("\n");
    double sg[NSB], bg[NSB], sw[NSB], ss[NSB], sb[NSB], smhz[NSB], bmhz[NSB], sgw[NSB], iw[NSB], bw[NSB], sms[NSB];
    for (int s = 0; s < NSB; s++) {
        int b = order[s], ns, nb; double mn, mx;
        sg[b] = sstat(c, b, bt0[b], steady, 1e9, 0, &ns, &mn, &mx);
        bg[b] = sstat(c, b, bt0[b], 0, 10, 0, &nb, NULL, NULL);
        sms[b] = sstat(c, b, bt0[b], steady, 1e9, 4, NULL, NULL, NULL);
        double a = bt0[b] + steady, z = bt1[b], a1 = bt0[b] + 1, z1 = bt0[b] + 10, ia = it0[b] + 0.5 * idle_s, iz = it1[b];
        int nsm;
        sw[b] = smc_mean(0, &a, &z, 1, &nsm); ss[b] = smc_mean(6, &a, &z, 1, NULL); sb[b] = smc_mean(1, &a, &z, 1, NULL);
        bw[b] = smc_mean(0, &a1, &z1, 1, NULL); iw[b] = smc_mean(0, &ia, &iz, 1, NULL);
        smhz[b] = ior_mean(1, &a, &z, 1, NULL); bmhz[b] = ior_mean(1, &a1, &z1, 1, NULL); sgw[b] = ior_mean(0, &a, &z, 1, NULL);
        int k = bcfg[b];
        printf("-- run %d, %s: %d calls in %.1f s\n", s, SB_NAME[b], bn[b], bt1[b] - bt0[b]);
        if (shm->kind[k] != KD_ACC && shm->kind[k] != KD_GPU) printf("   GPU part g=%d (f=%.3f)%s\n", shm->g[k], (double)shm->g[k] / axis_len(c),
                                                                      sb_part[b] < 1 ? "; GF/s below count the slice's own flops" : "");
        printf("   burst: %6.1f GF/s (n=%d), wall %5.1f W, GPU %4.0f MHz | steady: %6.1f GF/s median (n=%d, min %.1f, max %.1f), %.3f ms per call, wall %5.1f W, system %5.1f W, battery %+.1f W (%d samples), GPU %4.0f MHz, %5.1f W | idle before: %5.1f W\n",
               bg[b], nb, bw[b], bmhz[b], sg[b], ns, mn, mx, sms[b], sw[b], ss[b], sb[b], nsm, smhz[b], sgw[b], iw[b]);
        if (shm->kind[k] == KD_SPLIT)
            printf("   steady split parts: cpu slice %.1f ms, gpu slice %.1f ms, join wait %.1f ms (medians)\n",
                   sstat(c, b, bt0[b], steady, 1e9, 1, NULL, NULL, NULL) * 1e3, sstat(c, b, bt0[b], steady, 1e9, 2, NULL, NULL, NULL) * 1e3,
                   sstat(c, b, bt0[b], steady, 1e9, 3, NULL, NULL, NULL) * 1e3);
        if (check) printf("   bits: %d of %d checked calls differ from the run's first result\n", bdiff[b], bchk[b]);
        else printf("   bits: not checked (beta = 1: C accumulates, as it is never reset)\n");
        printf("   every 5 s (s into the run: GF/s, wall W, GPU MHz):");
        for (double s0 = 0; s0 < bt1[b] - bt0[b]; s0 += 5) {
            double a5 = bt0[b] + s0, z5 = a5 + 5, gsum = 0; int gn = 0;
            for (int i = 0; i < nsrec; i++) { const SRec *r = &srec[i]; double mid = 0.5 * (r->t0 + r->t1); if (r->blk == b && mid >= a5 && mid < z5) { gsum += flops(c) * sb_part[b] / r->total / 1e9; gn++; } }
            printf(" [%.0f: %.0f, %.0f, %.0f]", s0, gn ? gsum / gn : NAN, smc_mean(0, &a5, &z5, 1, NULL), ior_mean(1, &a5, &z5, 1, NULL));
        }
        printf("\n");
    }
    double scpu = sstat(c, SB_SPLIT, bt0[SB_SPLIT], steady, 1e9, 1, NULL, NULL, NULL) * 1e3, sgpu = sstat(c, SB_SPLIT, bt0[SB_SPLIT], steady, 1e9, 2, NULL, NULL, NULL) * 1e3;
    printf("-- steady-state speedup of the split: vs GPU alone %.3fx, vs Accelerate alone %.3fx (rebalanced: %.3fx, %.3fx) | burst: %.3fx, %.3fx\n",
           sg[SB_SPLIT] / sg[SB_GPU], sg[SB_SPLIT] / sg[SB_ACC], sg[SB_BAL] / sg[SB_GPU], sg[SB_BAL] / sg[SB_ACC], bg[SB_SPLIT] / bg[SB_GPU], bg[SB_SPLIT] / bg[SB_ACC]);
    printf("-- alone and together (steady medians): HEAD %.3f ms, TAIL %.3f ms, split %.3f ms; (HEAD + TAIL) / split %.3f; in the split the cpu slice takes %.3fx "
           "its time alone (%.3f ms), the gpu slice %.3fx (%.3f ms); Accelerate at the end / at the start %.3f\n",
           sms[SB_HEAD], sms[SB_TAIL], sms[SB_SPLIT], (sms[SB_HEAD] + sms[SB_TAIL]) / sms[SB_SPLIT], scpu / sms[SB_HEAD], scpu, sgpu / sms[SB_TAIL], sgpu,
           sg[SB_ACC_END] / sg[SB_ACC]);
    machine_state(pw, sizeof pw, ld, sizeof ld);
    // the first fields keep their meaning: split, GPU alone, Accelerate alone (the control at the start), rebalanced
#define F4(a) a[SB_SPLIT], a[SB_GPU], a[SB_ACC], a[SB_BAL]
    printf("SUSTAIN case=%s g=%d g_bal=%d steady_gfs=%.1f/%.1f/%.1f/%.1f burst_gfs=%.1f/%.1f/%.1f/%.1f sp_gpu=%.3f sp_acc=%.3f sp_gpu_bal=%.3f sp_acc_bal=%.3f "
           "wall_w=%.1f/%.1f/%.1f/%.1f sys_w=%.1f/%.1f/%.1f/%.1f batt_w=%.1f/%.1f/%.1f/%.1f mhz=%.0f/%.0f/%.0f/%.0f burst_mhz=%.0f/%.0f/%.0f/%.0f bits_diff=%d/%d/%d/%d "
           "acc_end_gfs=%.1f acc_end_w=%.1f head_ms=%.3f tail_ms=%.3f split_steady_ms=%.3f split_cpu_ms=%.3f split_gpu_ms=%.3f overlap=%.3f cpu_x=%.3f gpu_x=%.3f "
           "head_w=%.1f tail_w=%.1f head_mhz=%.0f tail_mhz=%.0f bits_checked=%s noreg=%d seed=%llu order=",
           c->name, g, shm->g[bcfg[SB_BAL]], F4(sg), F4(bg), sg[SB_SPLIT] / sg[SB_GPU], sg[SB_SPLIT] / sg[SB_ACC], sg[SB_BAL] / sg[SB_GPU], sg[SB_BAL] / sg[SB_ACC],
           F4(sw), F4(ss), F4(sb), F4(smhz), F4(bmhz), F4(bdiff),
           sg[SB_ACC_END], sw[SB_ACC_END], sms[SB_HEAD], sms[SB_TAIL], sms[SB_SPLIT], scpu, sgpu, (sms[SB_HEAD] + sms[SB_TAIL]) / sms[SB_SPLIT],
           scpu / sms[SB_HEAD], sgpu / sms[SB_TAIL], sw[SB_HEAD], sw[SB_TAIL], smhz[SB_HEAD], smhz[SB_TAIL], check ? "yes" : "no", noreg(), (unsigned long long)seed);
#undef F4
    static const char *const SB_TAG[NSB] = {"split", "GPU", "ACC", "bal", "HEAD", "TAIL", "ACC2"};
    for (int s = 0; s < NSB; s++) printf("%s%s", s ? "," : "", SB_TAG[order[s]]);
    printf("\nend: %s| load %s\n", pw, ld);
    // ---- raw data
    snprintf(path, sizeof path, "%s/%s.sustain.calls.tsv", outdir, c->name);
    FILE *f = fopen(path, "w");
    if (f) {
        char nb[24];
        fprintf(f, "run\tname\tcfg\tg\ti\tt0\tt1\ttotal\tcpu\tgpu\tjoin\twake\tkadd\tgfs\n");
        for (int i = 0; i < nsrec; i++) {
            const SRec *r = &srec[i];
            fprintf(f, "%d\t%s\t%d\t%d\t%d\t%.6f\t%.6f\t%.6f\t%.6f\t%.6f\t%.6f\t%.6f\t%.6f\t%.2f\n", r->blk, cfg_name(r->cfg, nb), r->cfg, r->g, r->i,
                    r->t0, r->t1, r->total, r->cpu, r->gpu, r->join, r->wake, r->kadd, flops(c) * sb_part[r->blk] / r->total / 1e9);
        }
        fclose(f);
    }
    snprintf(path, sizeof path, "%s/%s.sustain.runs.tsv", outdir, c->name);
    if ((f = fopen(path, "w"))) {
        fprintf(f, "run\tposition\tname\tidle_t0\tidle_t1\tt0\tt1\tcalls\n");
        for (int s = 0; s < NSB; s++) { int b = order[s]; fprintf(f, "%d\t%d\t%s\t%.3f\t%.3f\t%.3f\t%.3f\t%d\n", b, s, SB_NAME[b], it0[b], it1[b], bt0[b], bt1[b], bn[b]); }
        fclose(f);
    }
    snprintf(path, sizeof path, "%s/%s.sustain.smc.tsv", outdir, c->name); dump_smc(path);
    snprintf(path, sizeof path, "%s/%s.sustain.ior.tsv", outdir, c->name); dump_ior(path);
    return 0;
}

int main(int argc, char **argv) {
    setvbuf(stdout, NULL, _IOLBF, 0);   // whole lines, so the library's exit-time AB_REPORT (stderr) lands after them
    if (argc >= 2 && !strcmp(argv[1], "list")) { for (int i = 0; i < NCASES; i++) printf("%s\n", CASES[i].name); return 0; }
    // The library's AB_SPLIT would cut this program's slices again.
    if (strcmp(ab_split_source(), "off")) { fprintf(stderr, "bench_split: AB_SPLIT is set (split table %s); unset it\n", ab_split_source()); return 2; }
    if (argc >= 5 && !strcmp(argv[1], "sustain")) {
        const Case *c = find_case(argv[2]);
        if (!c) { fprintf(stderr, "unknown case\n"); return 2; }
        return sustain(c, argv[3], atoi(argv[4]), argc > 5 ? atof(argv[5]) : 80, argc > 6 ? atof(argv[6]) : 30, argc > 7 ? atof(argv[7]) : 20);
    }
    if (argc >= 7 && !strcmp(argv[1], "child")) return child(with_reps(find_case(argv[2])), atoi(argv[4]), argv[5], atoi(argv[6]));
    if (argc < 5 || strcmp(argv[1], "run")) { fprintf(stderr, "usage: bench_split run <case> <np> <outdir> [data] | bench_split sustain <case> <outdir> <g> [secs] [idle] [steady_after] | bench_split list\n"); return 2; }
    const Case *c = with_reps(find_case(argv[2])); int np = atoi(argv[3]), data = argc > 5 ? atoi(argv[5]) : 0;
    if (!c || np < 1 || np > NPMAX) { fprintf(stderr, "bad case or np\n"); return 2; }
    // a fixed GPU part: SPLIT_G as given, or SPLIT_F cut by the library's rule with its AB_MIN_GPU_DIM floor
    int fixed_g = 0, L = axis_len(c), rounds = (int)envd("SPLIT_ROUNDS", 3);
    const char *eg = getenv("SPLIT_G"), *ef = getenv("SPLIT_F");
    if (eg && ef) { fprintf(stderr, "SPLIT_G and SPLIT_F are both set: give one\n"); return 2; }
    if (eg) fixed_g = atoi(eg);
    if (ef) {
        const char *md = getenv("AB_MIN_GPU_DIM");
        int min_dim = md && *md && atoi(md) >= 0 ? atoi(md) : AB_DISPATCH_DEFAULT_MIN_DIM;   // as dispatch.c's env_dim
        fixed_g = ab_split_cut(c->axis, L, atof(ef), min_dim);
    }
    if ((eg || ef) && (fixed_g <= 0 || fixed_g >= L)) { fprintf(stderr, "SPLIT_G or SPLIT_F gives GPU part %d of %d: not a split\n", fixed_g, L); return 2; }
    if (rounds < 1 || rounds > MAXROUND) { fprintf(stderr, "SPLIT_ROUNDS must be 1 to %d\n", MAXROUND); return 2; }
    if (getenv("SPLIT_REPS") && envd("SPLIT_REPS", 0) < 1) { fprintf(stderr, "SPLIT_REPS must be at least 1\n"); return 2; }
    char pw[256], ld[256], path[1024];
    machine_state(pw, sizeof pw, ld, sizeof ld);
    printf("%s | VECLIB_MAXIMUM_THREADS=%s AB_MODE=%s AB_MIN_GPU_DIM=%s AB_STAGING_MB=%s AB_TRACE=%s AB_REPORT=%s | %d procs\nstart: %s| load %s\n", ab_build_info(),
           getenv("VECLIB_MAXIMUM_THREADS"), getenv("AB_MODE"), getenv("AB_MIN_GPU_DIM"), getenv("AB_STAGING_MB"), on_off("AB_TRACE"), on_off("AB_REPORT"), np, pw, ld);
    fflush(stdout);
    snprintf(path, sizeof path, "%s/shm.%s.%d.%d", argv[4], c->name, np, getpid());
    int fd = open(path, O_RDWR | O_CREAT | O_TRUNC, 0644); if (fd < 0 || ftruncate(fd, sizeof(Shm))) { perror("shm"); return 1; }
    shm = mmap(NULL, sizeof(Shm), PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0); memset(shm, 0, sizeof(Shm));
    shm->np = np; shm->ncfg = CF_SPLIT0; shm->kind[CF_ACC] = KD_ACC; shm->kind[CF_GPU] = KD_GPU;
    if (fixed_g > 0) { shm->fixed = 1; shm->g[shm->ncfg++] = fixed_g; }
    else for (int i = 0; i < c->nf; i++) { int g = gpu_part(c, c->f[i]); if (shm->ncfg == CF_SPLIT0 || g != shm->g[shm->ncfg - 1]) shm->g[shm->ncfg++] = g; }
    for (int k = CF_SPLIT0; k < shm->ncfg; k++) shm->kind[k] = KD_SPLIT;
    shm->rounds = rounds;
    shm->narm = envd("SPLIT_ARMS", 1) != 0 ? NARM : 0;   // the arms take their g from the split's best after the sweep
    if (shm->narm) {
        shm->k_seam = shm->ncfg; shm->k_split2 = shm->ncfg + 1; shm->k_head = shm->ncfg + 2; shm->k_tail = shm->ncfg + 3;
        shm->kind[shm->k_seam] = KD_SEAM; shm->kind[shm->k_split2] = KD_SPLIT2; shm->kind[shm->k_head] = KD_HEAD; shm->kind[shm->k_tail] = KD_TAIL;
    }
    int need = (shm->fixed ? 0 : 3 * shm->ncfg * c->sweep_reps) + rounds * (c->fx_acc + c->fx_gpu + c->fx_split * (1 + shm->narm)) + 2;
    if (c->fx_acc + c->fx_gpu + c->fx_split * (1 + shm->narm) + 1 > MAXSTEP) { fprintf(stderr, "too many repeats per round (MAXSTEP %d)\n", MAXSTEP); unlink(path); return 2; }
    if (need > MAXREC) {   // records per process: the sweep, the fixed block, the two completion orders
        fprintf(stderr, "SPLIT_ROUNDS=%d needs %d records per process, more than MAXREC %d\n", rounds, need, MAXREC);
        unlink(path); return 2;
    }
    bool smc = sampler_on() && smc_open(); pthread_t st;
    g_ior = sampler_on() && envd("SPLIT_POWER_S", 0) > 0 && ior_init(true);   // GPU clock for sustained runs
    if (smc || g_ior) pthread_create(&st, NULL, sampler, NULL);
    double idle_s = envd("SPLIT_IDLE_S", 1.0);   // mean over its second half
    double ti0 = now(); sleep_s(idle_s); double ti1 = now(), tih = ti0 + 0.5 * idle_s;
    double idle_w = smc ? smc_mean(0, &tih, &ti1, 1, NULL) : NAN;
    double t0 = now();
    pid_t kids[NPMAX]; char ids[NPMAX][8], nps[8], ds[8];
    snprintf(nps, sizeof nps, "%d", np); snprintf(ds, sizeof ds, "%d", data);
    for (int i = 0; i < np; i++) {
        snprintf(ids[i], sizeof ids[i], "%d", i);
        char *av[] = {argv[0], "child", (char *)c->name, nps, ids[i], path, ds, NULL};
        if (posix_spawn(&kids[i], argv[0], NULL, NULL, av, environ)) { perror("spawn"); return 1; }
    }
    int fail = 0;
    for (int i = 0; i < np; i++) { int s; waitpid(kids[i], &s, 0); if (!WIFEXITED(s) || WEXITSTATUS(s)) { fprintf(stderr, "child %d failed (status %d)\n", i, s); fail = 1; } }
    double el = now() - t0;
    if (smc || g_ior) { atomic_store(&g_stop, 1); pthread_join(st, NULL); }
    if (!fail) {
        report(c, np, data, smc, idle_w);
        char rp[1100]; snprintf(rp, sizeof rp, "%s/%s.np%d.d%d.raw.tsv", argv[4], c->name, np, data);
        FILE *f = fopen(rp, "w");
        if (f) {
            // the SEAM calls' outcome and timeline (ns on the clock of t0 and t1); 0 (or - for the outcome)
            // on other configs' rows and for a step that did not happen
            fprintf(f, "proc\tphase\tround\tcfg\tname\tg\trep\tprev\ttotal\tcpu\tgpu\tjoin\twake\tkadd\tt0\tt1\toutcome\tretries\tin_place"
                       "\tat_cut_ns\tat_create_ns\tat_cpu0_ns\tat_cpu1_ns\tat_gpu0_ns\tat_gpu1_ns\tat_join_ns\tat_end_ns"
                       "\tgpu_commit_ns\tgpu_start_ns\tgpu_end_ns\tgpu_done_ns\tgpu_encode_ns\tgpu_submit_ns\tgpu_wake_ns\tgpu_temps_ns"
                       "\tgpu_groups\tgpu_groups_max\tgpu_dispatches\tgpu_cmdbufs\tgpu_kinds\tgpu_mtl_error\tload1\n");
            char nb[24];
            for (int p = 0; p < np; p++) for (int i = 0; i < shm->nrec[p]; i++) {
                const Rec *r = &shm->rec[p][i]; const ab_split_times_t *m = &r->k.tm; const ab_prof_kernel_split_t *q = &m->gpu;
                int pv = r->k.prev; char pb[24];
                fprintf(f, "%d\t%d\t%d\t%d\t%s\t%d\t%d\t%s", p, r->phase, r->round, r->cfg, cfg_name(r->cfg, nb), cfg_g(c, r->cfg), r->rep,
                        pv == PREV_IDLE ? "idle" : pv == PREV_NONE ? "-" : cfg_name(pv, pb));
                for (int k = 0; k < NV; k++) fprintf(f, "\t%.6f", r->k.v[k]);
                fprintf(f, "\t%s\t%d\t%d\t%llu\t%llu\t%llu\t%llu\t%llu\t%llu\t%llu\t%llu", outcome_name(r->k.outcome), r->k.retries, r->k.in_place,
                        (unsigned long long)m->cut_ns, (unsigned long long)m->create_ns, (unsigned long long)m->cpu0_ns, (unsigned long long)m->cpu1_ns,
                        (unsigned long long)m->gpu0_ns, (unsigned long long)m->gpu1_ns, (unsigned long long)m->join_ns, (unsigned long long)m->end_ns);
                fprintf(f, "\t%llu\t%llu\t%llu\t%llu\t%llu\t%llu\t%llu\t%llu\t%llu\t%u\t%u\t%u\t%u\t%d\t%.2f\n",
                        (unsigned long long)q->commit_ns, (unsigned long long)q->gpu_start_ns, (unsigned long long)q->gpu_end_ns, (unsigned long long)q->done_ns,
                        (unsigned long long)q->encode_ns, (unsigned long long)q->submit_ns, (unsigned long long)q->wake_ns, (unsigned long long)q->temps_ns,
                        (unsigned long long)q->groups, q->groups_max, q->dispatches, q->cmdbufs, q->kinds, q->mtl_error, r->load1);
            }
            fclose(f);
        }
        if (smc) { snprintf(rp, sizeof rp, "%s/%s.np%d.d%d.power.tsv", argv[4], c->name, np, data); dump_smc(rp); }
        if (g_ior) { snprintf(rp, sizeof rp, "%s/%s.np%d.d%d.ior.tsv", argv[4], c->name, np, data); dump_ior(rp); }
        snprintf(rp, sizeof rp, "%s/%s.np%d.d%d.rounds.tsv", argv[4], c->name, np, data); dump_rounds(rp);
    }
    machine_state(pw, sizeof pw, ld, sizeof ld);
    printf("end: %s| load %s | wall %.1f s\n", pw, ld, el);
    unlink(path);
    return fail;
}
