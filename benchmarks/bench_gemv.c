// =============================================================================
// bench_gemv.c — BLAS-2 on the GPU vs Accelerate and plain C, in GB/s
// =============================================================================
//
// DGEMV, ZGEMV ('N', 'T', 'C') and ZHEMV ('U', 'L'), y = A x, square shapes.
// Each shape is timed four ways:
//   resident  ab_dev_*gemv / ab_dev_zhemv on A registered once with
//             ab_dev_register (no copy); a call stages x, runs the kernels,
//             checks the range guard and writes y
//   staged    ab_*gemv_blas / ab_zhemv_blas with the GPU path forced
//             (AB_MODE=gpu, set by this program): a call also maps A with a
//             fresh no-copy wrapper, the cost of a one-shot dgemv_ call
//   accel     Accelerate cblas_*
//   cloop     plain C on 4, 8 and 12 threads (dispatch_apply), the fastest
// as the median wall time of --reps calls after --warmup calls (GPU modes:
// and at least 0.5 s of calls, so the GPU has raised its clocks). GB/s counts
// the bytes of A the routine references: m n 8 (DGEMV), m n 16 (ZGEMV),
// n (n + 1) / 2 16 (ZHEMV: one triangle, which the GPU reads once). err is
// the largest |y_i - ref_i| / (|A| |x|)_i over 64 sampled outputs against a
// compensated (Dot2) FP64 reference.
//
// --sweep times staged against accel over square sizes 256-16384 (3 warm-ups,
// median of 7, as `make calibrate` does for GEMM): the data behind the
// AB_MODE=auto rows of src/blas2_abi.c; --rect does the same on rectangular
// shapes (4096 x 1024 to 262144 x 1024 and their transposes).
//
// Usage: bench_gemv [--sizes 4096,8192,16384] [--routines dgemv,zgemv,zhemv]
//                   [--reps 9] [--warmup 2] [--pad 0] [--csv file] [--sweep | --rect]
// Run under `machlock.py bench` (the benchmark machine's lock).
// =============================================================================

#include "apple_bottom.h"
#include "apple_bottom_device.h"
#include "blas_trans.h"        // ab_blas_trans (src/)
#include "test_blas2_ref.h"    // the compensated reference (tests/)

#include <Accelerate/Accelerate.h>
#include <complex.h>
#include <dispatch/dispatch.h>
#include <mach/mach_time.h>
#include <math.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static const char *const kRoutineName[] = {"dgemv", "zgemv", "zhemv"};

// The ops benchmarked per routine.
static const char *ops_of(B2Routine r) { return r == B2_ZHEMV ? "UL" : r == B2_DGEMV ? "NT" : "NTC"; }

typedef struct {
    B2Routine r;
    char op;          // 'N', 'T', 'C' or 'U', 'L'
    int m, n, lda;
    const double *A;  // column-major, (re, im) pairs for Z
    const double *x;
    double *y;
} Shape;

static double now_s(void) {
    static mach_timebase_info_data_t tb;
    if (!tb.denom) mach_timebase_info(&tb);
    return (double)mach_absolute_time() * tb.numer / tb.denom * 1e-9;
}

static int cmp_double(const void *a, const void *b) {
    double x = *(const double *)a, y = *(const double *)b;
    return x < y ? -1 : x > y;
}

static int w_of(B2Routine r) { return r == B2_DGEMV ? 1 : 2; }
static bool rows_out(const Shape *s) { return s->r == B2_ZHEMV || s->op == 'N'; }
static int xlen(const Shape *s) { return rows_out(s) ? s->n : s->m; }
static int ylen(const Shape *s) { return rows_out(s) ? s->m : s->n; }

static double bytes_referenced(const Shape *s) {
    if (s->r == B2_ZHEMV) return (double)s->n * (s->n + 1) / 2 * 16;
    return (double)s->m * s->n * 8 * w_of(s->r);
}

// =============================================================================
// Data
// =============================================================================

// Fills v[0..count) with uniform values in [-1, 1), in parallel.
static void fill_random(double *v, size_t count, uint64_t seed) {
    const size_t chunks = 64, per = (count + chunks - 1) / chunks;
    dispatch_apply(chunks, dispatch_get_global_queue(QOS_CLASS_USER_INITIATED, 0), ^(size_t c) {
        uint64_t s = seed * 0x9E3779B97F4A7C15ULL + c * 0xD1B54A32D192ED03ULL + 1;
        size_t lo = c * per, hi = lo + per < count ? lo + per : count;
        for (size_t i = lo; i < hi; i++) {
            s ^= s << 13; s ^= s >> 7; s ^= s << 17;
            v[i] = (double)(s >> 11) * 0x1p-52 - 1.0;
        }
    });
}

// =============================================================================
// The four ways to run a shape (alpha = 1, beta = 0)
// =============================================================================

static const double kOne[2] = {1, 0}, kZero[2] = {0, 0};

static ABTranspose ab_trans(char op) { return op == 'N' ? AB_NO_TRANS : op == 'T' ? AB_TRANS : AB_CONJ_TRANS; }

static const char *run_resident(const Shape *s, ab_dev_buffer_t bA) {
    ABStatus st;
    if (s->r == B2_DGEMV)
        st = ab_dev_dgemv(ab_trans(s->op), s->m, s->n, 1.0, bA, 0, s->lda, s->x, 1, 0.0, s->y, 1);
    else if (s->r == B2_ZGEMV)
        st = ab_dev_zgemv(ab_trans(s->op), s->m, s->n, kOne, bA, 0, s->lda, s->x, 1, kZero, s->y, 1);
    else
        st = ab_dev_zhemv(s->op == 'U' ? AB_UPPER : AB_LOWER, s->n, kOne, bA, 0, s->lda, s->x, 1,
                          kZero, s->y, 1);
    return st == AB_OK ? ab_get_last_dispatch_path() : "error";
}

static const char *run_staged(const Shape *s) {
    if (s->r == B2_DGEMV)
        ab_dgemv_blas(s->op, s->m, s->n, 1.0, s->A, s->lda, s->x, 1, 0.0, s->y, 1);
    else if (s->r == B2_ZGEMV)
        ab_zgemv_blas(s->op, s->m, s->n, 1.0, (const double complex *)s->A, s->lda,
                      (const double complex *)s->x, 1, 0.0, (double complex *)s->y, 1);
    else
        ab_zhemv_blas(s->op, s->n, 1.0, (const double complex *)s->A, s->lda,
                      (const double complex *)s->x, 1, 0.0, (double complex *)s->y, 1);
    return ab_get_last_dispatch_path();
}

static const char *run_accel(const Shape *s) {
    if (s->r == B2_DGEMV)
        cblas_dgemv(CblasColMajor, ab_blas_trans(s->op, false), s->m, s->n, 1.0, s->A, s->lda,
                    s->x, 1, 0.0, s->y, 1);
    else if (s->r == B2_ZGEMV)
        cblas_zgemv(CblasColMajor, ab_blas_trans(s->op, true), s->m, s->n, (const double complex *)kOne,
                    (const double complex *)s->A, s->lda, (const double complex *)s->x, 1,
                    (const double complex *)kZero, (double complex *)s->y, 1);
    else
        cblas_zhemv(CblasColMajor, s->op == 'U' ? CblasUpper : CblasLower, s->n,
                    (const double complex *)kOne, (const double complex *)s->A, s->lda,
                    (const double complex *)s->x, 1, (const double complex *)kZero,
                    (double complex *)s->y, 1);
    return "cpu";
}

// Plain C: y = A x on nth threads, each thread a contiguous block of
// outputs ('N': rows, streaming every column over them; 'T'/'C': columns,
// one dot product each). ZHEMV: each thread a block of columns of equal
// triangle area, accumulating the row part into a private y, summed after.
static void cloop(const Shape *s, int nth) {
    const int m = s->m, n = s->n, lda = s->lda;
    const double *A = s->A, *x = s->x;
    double *y = s->y;
    dispatch_queue_t q = dispatch_get_global_queue(QOS_CLASS_USER_INITIATED, 0);
    if (s->r == B2_ZHEMV) {
        const bool up = s->op == 'U';
        double *priv = calloc((size_t)nth * n * 2, sizeof(double));
        int *cut = malloc(sizeof(int) * (nth + 1));
        for (int t = 0; t <= nth; t++) {   // equal-area column blocks
            double f = (double)t / nth;
            int j = (int)(n * (up ? sqrt(f) : 1.0 - sqrt(1.0 - f)));
            cut[t] = j < 0 ? 0 : j > n ? n : j;
        }
        cut[nth] = n;
        dispatch_apply(nth, q, ^(size_t t) {
            double *p = priv + t * n * 2;
            for (int j = cut[t]; j < cut[t + 1]; j++) {
                const double *col = A + (size_t)j * lda * 2;
                double xr = x[2 * j], xi = x[2 * j + 1], sr = 0, si = 0;
                int k0 = up ? 0 : j + 1, k1 = up ? j : n;
                for (int k = k0; k < k1; k++) {
                    double ar = col[2 * k], ai = col[2 * k + 1];
                    // row part: y_k += A(k,j) x_j; column part: y_j += conj(A(k,j)) x_k
                    p[2 * k] += ar * xr - ai * xi;
                    p[2 * k + 1] += ar * xi + ai * xr;
                    sr += ar * x[2 * k] + ai * x[2 * k + 1];
                    si += ar * x[2 * k + 1] - ai * x[2 * k];
                }
                double d = col[2 * j];
                p[2 * j] += sr + d * xr;
                p[2 * j + 1] += si + d * xi;
            }
        });
        dispatch_apply(nth, q, ^(size_t t) {
            int i0 = (int)((size_t)n * t / nth), i1 = (int)((size_t)n * (t + 1) / nth);
            for (int i = i0; i < i1; i++) {
                double r = 0, im = 0;
                for (int u = 0; u < nth; u++) { r += priv[((size_t)u * n + i) * 2]; im += priv[((size_t)u * n + i) * 2 + 1]; }
                y[2 * i] = r;
                y[2 * i + 1] = im;
            }
        });
        free(priv);
        free(cut);
        return;
    }
    const bool cplx = s->r == B2_ZGEMV, conj = s->op == 'C' && cplx;
    const int nout = ylen(s);
    dispatch_apply(nth, q, ^(size_t t) {
        int o0 = (int)((size_t)nout * t / nth), o1 = (int)((size_t)nout * (t + 1) / nth);
        if (s->op == 'N' && !cplx) {
            for (int i = o0; i < o1; i++) y[i] = 0;
            for (int j = 0; j < n; j++) {
                const double *c = A + (size_t)j * lda;
                double xj = x[j];
                for (int i = o0; i < o1; i++) y[i] += c[i] * xj;
            }
        } else if (s->op == 'N') {
            for (int i = o0; i < o1; i++) y[2 * i] = y[2 * i + 1] = 0;
            for (int j = 0; j < n; j++) {
                const double *c = A + (size_t)j * lda * 2;
                double xr = x[2 * j], xi = x[2 * j + 1];
                for (int i = o0; i < o1; i++) {
                    y[2 * i] += c[2 * i] * xr - c[2 * i + 1] * xi;
                    y[2 * i + 1] += c[2 * i] * xi + c[2 * i + 1] * xr;
                }
            }
        } else if (!cplx) {
            for (int j = o0; j < o1; j++) {
                const double *c = A + (size_t)j * lda;
                double s0 = 0, s1 = 0, s2 = 0, s3 = 0;
                int i = 0;
                for (; i + 4 <= m; i += 4) {
                    s0 += c[i] * x[i]; s1 += c[i + 1] * x[i + 1];
                    s2 += c[i + 2] * x[i + 2]; s3 += c[i + 3] * x[i + 3];
                }
                for (; i < m; i++) s0 += c[i] * x[i];
                y[j] = (s0 + s1) + (s2 + s3);
            }
        } else {
            const double sg = conj ? -1.0 : 1.0;
            for (int j = o0; j < o1; j++) {
                const double *c = A + (size_t)j * lda * 2;
                double r0 = 0, i0 = 0, r1 = 0, i1 = 0;
                int i = 0;
                for (; i + 2 <= m; i += 2) {
                    double ar = c[2 * i], ai = sg * c[2 * i + 1], br = c[2 * i + 2], bi = sg * c[2 * i + 3];
                    r0 += ar * x[2 * i] - ai * x[2 * i + 1]; i0 += ar * x[2 * i + 1] + ai * x[2 * i];
                    r1 += br * x[2 * i + 2] - bi * x[2 * i + 3]; i1 += br * x[2 * i + 3] + bi * x[2 * i + 2];
                }
                for (; i < m; i++) {
                    double ar = c[2 * i], ai = sg * c[2 * i + 1];
                    r0 += ar * x[2 * i] - ai * x[2 * i + 1]; i0 += ar * x[2 * i + 1] + ai * x[2 * i];
                }
                y[2 * j] = r0 + r1;
                y[2 * j + 1] = i0 + i1;
            }
        }
    });
}

// =============================================================================
// Accuracy: compensated reference at sampled outputs
// =============================================================================

// Max over 64 sampled outputs of |y_i - ref_i| / (|op(A)| |x|)_i against
// the compensated reference (tests/test_blas2_ref.h).
static double sampled_error(const Shape *s) {
    const int nout = ylen(s), w = w_of(s->r);
    int rows[64];
    double t[128], mag[64], worst = 0;
    for (int q = 0; q < 64; q++) rows[q] = (int)(((uint64_t)q * 2654435761u) % (uint64_t)nout);
    b2_reference(s->r, s->op, s->A, s->lda, s->x, xlen(s), 1, rows, 64, t, mag);
    for (int q = 0; q < 64; q++) {
        const double *y = s->y + (size_t)rows[q] * w;
        double d = w == 2 ? hypot(y[0] - t[2 * q], y[1] - t[2 * q + 1]) : fabs(y[0] - t[2 * q]);
        double e = mag[q] > 0 ? d / mag[q] : d;
        if (e > worst || isnan(e)) worst = e;
    }
    return worst;
}

// =============================================================================
// Timing
// =============================================================================

typedef struct { double median, best; const char *path; } Timing;

typedef enum { M_RESIDENT, M_STAGED, M_ACCEL, M_CLOOP } Mode;
static const char *const kModeName[] = {"resident", "staged", "accel", "cloop"};

// The GPU raises its clocks only after about 0.3 s of sustained work (on
// a cold M2 Max the first calls run at about half speed), so GPU modes warm
// up for at least kGpuWarmupS before the timed calls: the steady state of a
// solver loop.
static const double kGpuWarmupS = 0.5;

static Timing time_mode(const Shape *s, Mode mode, ab_dev_buffer_t bA, int nth, int warmup, int reps) {
    double *t = malloc(sizeof(double) * reps);
    const char *path = "cpu";
    bool gpu = mode == M_RESIDENT || mode == M_STAGED;
    double start = now_s();
    for (int r = -warmup; r < reps; r++) {
        if (r == 0 && gpu && now_s() - start < kGpuWarmupS) r = -1;   // keep warming up
        double t0 = now_s();
        switch (mode) {
        case M_RESIDENT: path = run_resident(s, bA); break;
        case M_STAGED:   path = run_staged(s); break;
        case M_ACCEL:    path = run_accel(s); break;
        case M_CLOOP:    cloop(s, nth); break;
        }
        if (r >= 0) t[r] = now_s() - t0;
    }
    qsort(t, reps, sizeof *t, cmp_double);
    Timing out = {t[reps / 2], t[0], path};
    free(t);
    return out;
}

static FILE *g_csv;

static void report(const Shape *s, const char *mode, int nth, Timing tm, double err) {
    double gbs = bytes_referenced(s) / tm.median / 1e9;
    printf("  %-8s %-4s%s %8.3f ms  %7.1f GB/s  err %.1e\n", mode, tm.path,
           nth ? (nth >= 10 ? "" : " ") : "  ", tm.median * 1e3, gbs, err);
    if (g_csv)
        fprintf(g_csv, "%s,%c,%d,%d,%d,%s,%d,%s,%.6f,%.6f,%.2f,%.3e\n", kRoutineName[s->r], s->op,
                s->m, s->n, s->lda, mode, nth, tm.path, tm.median * 1e3, tm.best * 1e3, gbs, err);
}

static void bench_shape(Shape *s, int warmup, int reps) {
    printf("\n%s '%c' %d x %d (lda %d), A referenced %.2f GB\n", kRoutineName[s->r], s->op, s->m,
           s->n, s->lda, bytes_referenced(s) / 1e9);
    size_t abytes = ((size_t)s->lda * (s->n - 1) + s->m) * 8 * w_of(s->r);
    ab_dev_buffer_t bA = ab_dev_register((void *)s->A, abytes);
    if (!bA) { fprintf(stderr, "ab_dev_register failed\n"); exit(1); }

    Timing tm = time_mode(s, M_RESIDENT, bA, 0, warmup, reps);
    report(s, kModeName[M_RESIDENT], 0, tm, sampled_error(s));
    tm = time_mode(s, M_STAGED, NULL, 0, warmup, reps);
    report(s, kModeName[M_STAGED], 0, tm, sampled_error(s));
    tm = time_mode(s, M_ACCEL, NULL, 0, warmup, reps);
    report(s, kModeName[M_ACCEL], 0, tm, sampled_error(s));
    Timing best = {1e30, 1e30, "cpu"};
    int best_nth = 0;
    for (int nth = 4; nth <= 12; nth += 4) {
        tm = time_mode(s, M_CLOOP, NULL, nth, warmup, reps);
        if (tm.median < best.median) { best = tm; best_nth = nth; }
    }
    cloop(s, best_nth);
    char name[16];
    snprintf(name, sizeof name, "cloop%d", best_nth);
    report(s, name, best_nth, best, sampled_error(s));
    ab_dev_free(bA);
}

// Staged GPU against Accelerate on an m x n shape for each op of routine
// r, with data filled once; returns Accelerate time / GPU time per op.
static void sweep_shape(B2Routine r, int m, int n, int pad, uint64_t seed, double *ratio) {
    int lda = m + pad, mx = m > n ? m : n;
    size_t na = (size_t)lda * n * w_of(r);
    double *A = malloc(na * 8), *x = malloc((size_t)mx * 16), *y = malloc((size_t)mx * 16);
    fill_random(A, na, seed);
    fill_random(x, (size_t)mx * 2, seed + 100);
    for (int o = 0; ops_of(r)[o]; o++) {
        Shape s = {r, ops_of(r)[o], m, n, lda, A, x, y};
        Timing g = time_mode(&s, M_STAGED, NULL, 0, 3, 7);
        Timing a = time_mode(&s, M_ACCEL, NULL, 0, 3, 7);
        ratio[o] = a.median / g.median;
        if (g_csv)
            fprintf(g_csv, "%s,%c,%d,%d,%d,%s,%.6f,%.6f,%.4f\n", kRoutineName[r], s.op, m, n, lda,
                    g.path, g.median * 1e3, a.median * 1e3, ratio[o]);
    }
    free(A); free(x); free(y);
}

// --sweep: staged GPU vs Accelerate (Accelerate time / GPU time) on square
// shapes; --rect: on rectangular ones, aspect ratios 4 to 256 both ways.
static void sweep(const char *routines, int pad, bool rect) {
    const int sizes[] = {256, 384, 512, 768, 1024, 1536, 2048, 3072, 4096, 6144, 8192, 12288, 16384};
    const int shapes[][2] = {{4096, 1024}, {1024, 4096}, {16384, 1024}, {1024, 16384},
                             {65536, 256}, {256, 65536}, {32768, 2048}, {2048, 32768},
                             {131072, 512}, {512, 131072}, {262144, 1024}, {1024, 262144}};
    const int nsq = sizeof sizes / sizeof *sizes, nre = sizeof shapes / sizeof *shapes;
    printf("\nstaged GPU vs Accelerate, median of 7 after 3 warm-ups (ratio > 1: GPU faster)\n");
    for (int r = 0; r < 3; r++) {
        if (!strstr(routines, kRoutineName[r]) || (rect && r == B2_ZHEMV)) continue;
        int count = rect ? nre : nsq;
        double ratio[16][3];
        for (int k = 0; k < count; k++)
            sweep_shape((B2Routine)r, rect ? shapes[k][0] : sizes[k], rect ? shapes[k][1] : sizes[k],
                        pad, 1 + k, ratio[k]);
        for (int o = 0; ops_of((B2Routine)r)[o]; o++) {
            printf("%s '%c':", kRoutineName[r], ops_of((B2Routine)r)[o]);
            for (int k = 0; k < count; k++) {
                if (rect) printf(" %dx%d:%.2f", shapes[k][0], shapes[k][1], ratio[k][o]);
                else printf(" %d:%.2f", sizes[k], ratio[k][o]);
            }
            printf("\n");
        }
    }
}

int main(int argc, char **argv) {
    const char *sizes_arg = "4096,8192,16384", *routines = "dgemv,zgemv,zhemv", *csv = NULL;
    int reps = 9, warmup = 2, pad = 0;
    bool do_sweep = false, do_rect = false;
    for (int i = 1; i < argc; i++) {
        if (!strcmp(argv[i], "--sizes") && i + 1 < argc) sizes_arg = argv[++i];
        else if (!strcmp(argv[i], "--routines") && i + 1 < argc) routines = argv[++i];
        else if (!strcmp(argv[i], "--reps") && i + 1 < argc) reps = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--warmup") && i + 1 < argc) warmup = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--pad") && i + 1 < argc) pad = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--csv") && i + 1 < argc) csv = argv[++i];
        else if (!strcmp(argv[i], "--sweep")) do_sweep = true;
        else if (!strcmp(argv[i], "--rect")) do_sweep = do_rect = true;
        else { fprintf(stderr, "usage: see the header of benchmarks/bench_gemv.c\n"); return 2; }
    }
    if (reps < 1) reps = 1;
    setenv("AB_MODE", "gpu", 1);   // the staged mode times the GPU path
    if (ab_init() != AB_OK) { fprintf(stderr, "no Metal device\n"); return 1; }
    printf("apple-bottom BLAS-2 benchmark: %s, %s\n", ab_build_info(), ab_device_name());
    if (csv) {
        g_csv = fopen(csv, "w");
        if (!g_csv) { perror(csv); return 1; }
        fprintf(g_csv, do_sweep ? "routine,op,m,n,lda,gpu_path,gpu_median_ms,accel_median_ms,speedup\n"
                                : "routine,op,m,n,lda,mode,threads,path,median_ms,best_ms,gbs,err\n");
    }
    if (do_sweep) {
        sweep(routines, pad, do_rect);
    } else {
        char buf[256];
        snprintf(buf, sizeof buf, "%s", sizes_arg);
        for (char *tok = strtok(buf, ","); tok; tok = strtok(NULL, ",")) {
            int n = atoi(tok), lda = n + pad;
            for (int r = 0; r < 3; r++) {
                if (!strstr(routines, kRoutineName[r])) continue;
                size_t na = (size_t)lda * n * w_of((B2Routine)r);
                double *A = malloc(na * 8), *x = malloc((size_t)n * 16), *y = malloc((size_t)n * 16);
                fill_random(A, na, 7 + r);
                fill_random(x, (size_t)n * 2, 77 + r);
                for (const char *op = ops_of((B2Routine)r); *op; op++) {
                    Shape s = {(B2Routine)r, *op, n, n, lda, A, x, y};
                    bench_shape(&s, warmup, reps);
                }
                free(A); free(x); free(y);
            }
        }
    }
    if (g_csv) fclose(g_csv);
    return 0;
}
