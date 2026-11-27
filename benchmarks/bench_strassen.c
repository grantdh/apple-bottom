// =============================================================================
// bench_strassen.c — speed and accuracy of the opt-in Strassen-Winograd GEMM
// =============================================================================
//   bench_strassen speed     [--complex] [--reps R] [--warm S] MxKxN ...
//   bench_strassen reference --ref-dir D [--rows R] MxKxN ...
//   bench_strassen accuracy  [--ref-dir D] [--rows R] MxKxN ...
//   bench_strassen abi       [--complex] [--reps R] [--warm S] MxKxN ...
// One CSV row per measurement on stdout, after a header; progress on stderr.
//
// speed: the native API, A, B and C resident in GPU memory. Per shape the
//   classical DD GEMM (ab_dgemm_scaled or ab_zgemm), Strassen
//   (ab_dgemm_strassen or ab_zgemm_strassen) and Accelerate FP64
//   (cblas_dgemm or cblas_zgemm, all cores) each warm up for --warm seconds
//   (default 0.5: the GPU raises its clock only after ~0.3 s of load), then
//   run --reps times (default 5): the two GPU methods back to back, in
//   alternating order, then Accelerate on its own, so that no GPU call
//   follows an idle GPU. A row gives the median and best wall time per call
//   and, for the GPU, the medians of the command-buffer time
//   (ab_internal_take_gpu_seconds), the host time spent on temporaries, and
//   the commit-to-start and end-to-return times.
//   GFLOP/s counts 2MNK (8MNK complex) per call at the median, so Strassen's
//   rate is the effective one. Strassen's first call for a shape, which
//   grows its workspace, is a row of its own.
// reference: Dot2 references (test_dot2.h) of A B for --rows evenly spaced
//   rows of C (default all) and the three kinds of data below, saved to
//   --ref-dir for accuracy; CPU only, so it can run beside GPU work.
// accuracy: the classical DD GEMM, Strassen and Accelerate against the
//   reference (loaded from --ref-dir if there, else computed), for
//     uniform  entries uniform in [-1, 1)
//     positive entries uniform in [0, 1): no cancellation, |C| = |A||B|
//     scaled   uniform in [-1, 1), the rows of A and the columns of B scaled
//              by 2^e, e an integer uniform in [-16, 16]
//   with the errors over the sampled rows
//     nw    max|Ĉ - C| / (max|A| max|B|)   the normwise measure of the bounds
//     fro   ||Ĉ - C||_F / ||C||_F          normwise relative error
//     cwab  max |Ĉ - C|_ij / (|A||B|)_ij   componentwise, scale of the GEMM bound
//     cw    max |Ĉ - C|_ij / |C_ij|        componentwise relative error
// abi: ab_dgemm_blas or ab_zgemm_blas end to end (column-major FP64 in and
//   out, alpha 1, beta 0), routed by the environment: AB_MODE=gpu with
//   AB_STRASSEN unset or 1. The method column is the path the call took,
//   with "+strassen" if AB_STRASSEN applies to its shape.
// =============================================================================

#include "apple_bottom.h"
#include "profiling/blas_profiler.h"
#include "strassen.h"
#include "test_blas_check.h"   // splitmix
#include "test_dot2.h"
#include <Accelerate/Accelerate.h>
#include <complex.h>
#include <float.h>
#include <math.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>

typedef struct { int M, K, N; } Shape;

static int g_reps = 5;
static double g_warm = 0.5;
static int g_rows = 0;            // 0: every row
static const char *g_ref_dir = NULL;

static double now_s(void) { return ab_prof_ticks_to_ns(ab_prof_now_ticks()) * 1e-9; }

static double uniform(uint64_t seed, size_t i, double lo, double hi) {
    return lo + (hi - lo) * (double)(splitmix(seed + i * 0x9E3779B97F4A7C15ULL) >> 11) * 0x1p-53;
}

static double *xmalloc(size_t n) {
    double *p = malloc(n * sizeof *p);
    if (!p) {
        fprintf(stderr, "out of memory (%zu doubles)\n", n);
        exit(1);
    }
    return p;
}

static ABMatrix upload(int rows, int cols, const double *x) {
    ABMatrix m = ab_matrix_create(rows, cols);
    if (!m) {
        fprintf(stderr, "ab_matrix_create(%d, %d) failed\n", rows, cols);
        exit(1);
    }
    if (x) ab_matrix_upload(m, x, true);
    return m;
}

static int cmp_double(const void *a, const void *b) {
    double x = *(const double *)a, y = *(const double *)b;
    return (x > y) - (x < y);
}

// Sorts x.
static double median(double *x, int n) {
    qsort(x, (size_t)n, sizeof *x, cmp_double);
    return n % 2 ? x[n / 2] : 0.5 * (x[n / 2 - 1] + x[n / 2]);
}

// =============================================================================
// speed and abi
// =============================================================================

enum { M_CLASSICAL, M_STRASSEN, M_ACCELERATE, M_COUNT };
static const char *const kMethod[M_COUNT] = {"classical", "strassen", "accelerate"};

typedef struct {
    bool cplx;
    Shape s;
    ABMatrix A[2], B[2], C[2];      // real and imaginary parts (DGEMM: [0])
    double *a, *b, *c;              // host FP64, row-major; interleaved if complex
} SpeedCase;

// One timed call: wall time and, from the GPU calls' profile split, the
// command-buffer time, host time on temporaries, commit-to-GPU-start and
// GPU-end-to-return (all 0 for Accelerate). Seconds.
typedef struct { double wall, gpu, temps, submit, wake; } Sample;

static Sample run_method(const SpeedCase *t, int m) {
    const int M = t->s.M, K = t->s.K, N = t->s.N;
    ABStatus st = AB_OK;
    (void)ab_internal_take_gpu_seconds(NULL);
    double t0 = now_s();
    if (m == M_ACCELERATE) {
        if (t->cplx) {
            const double complex one = 1.0, zero = 0.0;
            cblas_zgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, M, N, K, &one,
                        (const double complex *)t->a, K, (const double complex *)t->b, N, &zero,
                        (double complex *)t->c, N);
        } else {
            cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, M, N, K, 1.0, t->a, K, t->b, N,
                        0.0, t->c, N);
        }
    } else if (t->cplx) {
        st = m == M_STRASSEN
                 ? ab_zgemm_strassen(t->A[0], t->A[1], t->B[0], t->B[1], t->C[0], t->C[1])
                 : ab_zgemm(t->A[0], t->A[1], t->B[0], t->B[1], t->C[0], t->C[1]);
    } else {
        st = m == M_STRASSEN ? ab_dgemm_strassen(1.0, t->A[0], t->B[0], 0.0, t->C[0])
                             : ab_dgemm_scaled(1.0, t->A[0], t->B[0], 0.0, t->C[0]);
    }
    Sample x = {.wall = now_s() - t0};
    if (st != AB_OK) {
        fprintf(stderr, "%s failed: %s\n", kMethod[m], ab_status_string(st));
        exit(1);
    }
    ab_prof_kernel_split_t split;
    x.gpu = ab_internal_take_gpu_seconds(&split);
    x.temps = split.temps_ns * 1e-9;
    x.submit = split.submit_ns * 1e-9;
    x.wake = split.wake_ns * 1e-9;
    return x;
}

static void warm_up(const SpeedCase *t, int m) {
    for (double t0 = now_s(); now_s() - t0 < g_warm;) run_method(t, m);
}

static void speed_header(void) {
    printf("mode,routine,M,K,N,method,reps,wall_med_ms,wall_min_ms,gpu_med_ms,temps_med_ms,"
           "submit_med_ms,wake_med_ms,gflops\n");
}

// Medians (and the best wall time) of reps samples; GFLOP/s at the median.
static void speed_row(const char *mode, Shape s, bool cplx, const char *method, int reps,
                      const Sample *x) {
    double wall[64], gpu[64], temps[64], submit[64], wake[64];
    for (int r = 0; r < reps; r++) {
        wall[r] = x[r].wall;
        gpu[r] = x[r].gpu;
        temps[r] = x[r].temps;
        submit[r] = x[r].submit;
        wake[r] = x[r].wake;
    }
    double flops = (cplx ? 8.0 : 2.0) * s.M * s.N * s.K;
    double wmed = median(wall, reps), wmin = wall[0];
    printf("%s,%s,%d,%d,%d,%s,%d,%.3f,%.3f,%.3f,%.3f,%.3f,%.3f,%.1f\n", mode,
           cplx ? "zgemm" : "dgemm", s.M, s.K, s.N, method, reps, wmed * 1e3,
           wmin * 1e3, median(gpu, reps) * 1e3, median(temps, reps) * 1e3,
           median(submit, reps) * 1e3, median(wake, reps) * 1e3, flops / wmed * 1e-9);
    fflush(stdout);
}

static void speed_case(Shape s, bool cplx) {
    const size_t mk = (size_t)s.M * s.K, kn = (size_t)s.K * s.N, mn = (size_t)s.M * s.N;
    const int parts = cplx ? 2 : 1;
    SpeedCase t = {.cplx = cplx, .s = s};
    t.a = xmalloc(parts * mk);
    t.b = xmalloc(parts * kn);
    t.c = xmalloc(parts * mn);
    for (size_t i = 0; i < parts * mk; i++) t.a[i] = uniform(1, i, -1, 1);
    for (size_t i = 0; i < parts * kn; i++) t.b[i] = uniform(2, i, -1, 1);
    double *part = xmalloc(mk > kn ? mk : kn);
    for (int p = 0; p < parts; p++) {   // split interleaved (re, im) for the GPU
        for (size_t i = 0; i < mk; i++) part[i] = t.a[parts * i + p];
        t.A[p] = upload(s.M, s.K, part);
        for (size_t i = 0; i < kn; i++) part[i] = t.b[parts * i + p];
        t.B[p] = upload(s.K, s.N, part);
        t.C[p] = upload(s.M, s.N, NULL);
    }
    free(part);

    fprintf(stderr, "%s %dx%dx%d\n", cplx ? "zgemm" : "dgemm", s.M, s.K, s.N);
    Sample first = run_method(&t, M_STRASSEN);
    speed_row("speed", s, cplx, "strassen-first", 1, &first);
    Sample x[M_COUNT][64];
    warm_up(&t, M_CLASSICAL);
    warm_up(&t, M_STRASSEN);
    for (int r = 0; r < g_reps; r++)
        for (int i = 0; i < 2; i++) {
            int m = (r + i) % 2 ? M_STRASSEN : M_CLASSICAL;
            x[m][r] = run_method(&t, m);
        }
    warm_up(&t, M_ACCELERATE);
    for (int r = 0; r < g_reps; r++) x[M_ACCELERATE][r] = run_method(&t, M_ACCELERATE);
    for (int m = 0; m < M_COUNT; m++) speed_row("speed", s, cplx, kMethod[m], g_reps, x[m]);

    for (int p = 0; p < parts; p++) {
        ab_matrix_destroy(t.A[p]);
        ab_matrix_destroy(t.B[p]);
        ab_matrix_destroy(t.C[p]);
    }
    free(t.a); free(t.b); free(t.c);
}

// ab_*gemm_blas end to end; the environment decides the path.
static void abi_call(Shape s, bool cplx, const double *a, const double *b, double *c) {
    if (cplx)
        ab_zgemm_blas('N', 'N', s.M, s.N, s.K, 1.0, (const double complex *)a, s.M,
                      (const double complex *)b, s.K, 0.0, (double complex *)c, s.M);
    else
        ab_dgemm_blas('N', 'N', s.M, s.N, s.K, 1.0, a, s.M, b, s.K, 0.0, c, s.M);
}

static void abi_case(Shape s, bool cplx) {
    const int parts = cplx ? 2 : 1;
    const size_t mk = (size_t)s.M * s.K, kn = (size_t)s.K * s.N, mn = (size_t)s.M * s.N;
    double *a = xmalloc(parts * mk), *b = xmalloc(parts * kn), *c = xmalloc(parts * mn);
    for (size_t i = 0; i < parts * mk; i++) a[i] = uniform(1, i, -1, 1);
    for (size_t i = 0; i < parts * kn; i++) b[i] = uniform(2, i, -1, 1);
    for (double t0 = now_s(); now_s() - t0 < g_warm;) abi_call(s, cplx, a, b, c);
    Sample x[64] = {{0}};
    for (int r = 0; r < g_reps; r++) {
        double t0 = now_s();
        abi_call(s, cplx, a, b, c);
        x[r].wall = now_s() - t0;
    }
    char method[64];
    snprintf(method, sizeof method, "%s%s", ab_get_last_dispatch_path(),
             ab_internal_strassen_wanted(s.M, s.N, s.K) ? "+strassen" : "");
    speed_row("abi", s, cplx, method, g_reps, x);
    free(a); free(b); free(c);
}

// =============================================================================
// reference and accuracy
// =============================================================================

enum { K_UNIFORM, K_POSITIVE, K_SCALED, K_COUNT };
static const char *const kKind[K_COUNT] = {"uniform", "positive", "scaled"};

// A (M x K) and B (K x N) of kind k.
static void make_data(Shape s, int kind, double **pa, double **pb) {
    const size_t mk = (size_t)s.M * s.K, kn = (size_t)s.K * s.N;
    double lo = kind == K_POSITIVE ? 0.0 : -1.0;
    double *a = xmalloc(mk), *b = xmalloc(kn);
    for (size_t i = 0; i < mk; i++) a[i] = uniform(11 + kind, i, lo, 1);
    for (size_t i = 0; i < kn; i++) b[i] = uniform(21 + kind, i, lo, 1);
    if (kind == K_SCALED) {
        for (int i = 0; i < s.M; i++) {
            double f = ldexp(1.0, (int)(splitmix(31 + i) % 33) - 16);
            for (int k = 0; k < s.K; k++) a[(size_t)i * s.K + k] *= f;
        }
        for (int j = 0; j < s.N; j++) {
            double f = ldexp(1.0, (int)(splitmix(41 + j) % 33) - 16);
            for (int k = 0; k < s.K; k++) b[(size_t)k * s.N + j] *= f;
        }
    }
    *pa = a;
    *pb = b;
}

static int sample_rows(Shape s) { return g_rows > 0 && g_rows < s.M ? g_rows : s.M; }
static int sampled_row(Shape s, int r) { return (int)((long long)r * s.M / sample_rows(s)); }

// The reference of A B and |A||B| on the sampled rows (R x N each).
typedef struct { double *ref, *abs; } Reference;

static void ref_path(char *buf, size_t n, Shape s, int kind) {
    snprintf(buf, n, "%s/ref_%s_%dx%dx%d_r%d.bin", g_ref_dir, kKind[kind], s.M, s.K, s.N,
             sample_rows(s));
}

static Reference compute_reference(Shape s, int kind, const double *a, const double *b) {
    const int R = sample_rows(s);
    const size_t rk = (size_t)R * s.K, rn = (size_t)R * s.N, kn = (size_t)s.K * s.N;
    double *as = xmalloc(rk), *absa = xmalloc(rk), *absb = xmalloc(kn);
    for (int r = 0; r < R; r++)
        memcpy(as + (size_t)r * s.K, a + (size_t)sampled_row(s, r) * s.K, s.K * sizeof *as);
    for (size_t i = 0; i < rk; i++) absa[i] = fabs(as[i]);
    for (size_t i = 0; i < kn; i++) absb[i] = fabs(b[i]);
    Reference ref = {calloc(rn, sizeof(double)), xmalloc(rn)};
    double *E = calloc(rn, sizeof *E);
    double t0 = now_s();
    dot2_gemm_acc(R, s.N, s.K, 1.0, as, b, ref.ref, E);
    for (size_t i = 0; i < rn; i++) ref.ref[i] += E[i];
    cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, R, s.N, s.K, 1.0, absa, s.K, absb,
                s.N, 0.0, ref.abs, s.N);
    fprintf(stderr, "  reference %s %dx%dx%d (%d rows): %.1f s\n", kKind[kind], s.M, s.K, s.N,
            R, now_s() - t0);
    free(E); free(as); free(absa); free(absb);
    return ref;
}

static bool load_reference(Shape s, int kind, Reference *ref) {
    if (!g_ref_dir) return false;
    char path[1024];
    ref_path(path, sizeof path, s, kind);
    FILE *f = fopen(path, "rb");
    if (!f) return false;
    size_t rn = (size_t)sample_rows(s) * s.N;
    ref->ref = xmalloc(rn);
    ref->abs = xmalloc(rn);
    bool ok = fread(ref->ref, sizeof(double), rn, f) == rn &&
              fread(ref->abs, sizeof(double), rn, f) == rn;
    fclose(f);
    if (!ok) {
        free(ref->ref);
        free(ref->abs);
    }
    return ok;
}

static void save_reference(Shape s, int kind, Reference ref) {
    char path[1024];
    ref_path(path, sizeof path, s, kind);
    FILE *f = fopen(path, "wb");
    size_t rn = (size_t)sample_rows(s) * s.N;
    if (!f || fwrite(ref.ref, sizeof(double), rn, f) != rn ||
        fwrite(ref.abs, sizeof(double), rn, f) != rn) {
        fprintf(stderr, "cannot write %s\n", path);
        exit(1);
    }
    fclose(f);
}

static void reference_case(Shape s) {
    for (int kind = 0; kind < K_COUNT; kind++) {
        char path[1024];
        struct stat st;
        ref_path(path, sizeof path, s, kind);
        if (!stat(path, &st) && (size_t)st.st_size == 2 * sizeof(double) * sample_rows(s) * s.N) {
            fprintf(stderr, "  reference %s %dx%dx%d: already saved\n", kKind[kind], s.M, s.K, s.N);
            continue;
        }
        double *a, *b;
        make_data(s, kind, &a, &b);
        Reference ref = compute_reference(s, kind, a, b);
        save_reference(s, kind, ref);
        free(a); free(b); free(ref.ref); free(ref.abs);
    }
}

static double max_abs(const double *x, size_t n) {
    double m = 0;
    for (size_t i = 0; i < n; i++) m = fmax(m, fabs(x[i]));
    return m;
}

static void accuracy_header(void) { printf("mode,kind,M,K,N,rows,method,nw,fro,cwab,cw\n"); }

static void accuracy_case(Shape s) {
    const size_t mk = (size_t)s.M * s.K, kn = (size_t)s.K * s.N, mn = (size_t)s.M * s.N;
    const int R = sample_rows(s);
    for (int kind = 0; kind < K_COUNT; kind++) {
        double *a, *b;
        make_data(s, kind, &a, &b);
        Reference ref;
        if (!load_reference(s, kind, &ref)) {
            ref = compute_reference(s, kind, a, b);
            if (g_ref_dir) save_reference(s, kind, ref);
        }
        double scale = max_abs(a, mk) * max_abs(b, kn);

        ABMatrix A = upload(s.M, s.K, a), B = upload(s.K, s.N, b), C = upload(s.M, s.N, NULL);
        double *x = xmalloc(mn);
        for (int m = 0; m < M_COUNT; m++) {
            ABStatus st = AB_OK;
            if (m == M_ACCELERATE)
                cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, s.M, s.N, s.K, 1.0, a, s.K,
                            b, s.N, 0.0, x, s.N);
            else
                st = m == M_STRASSEN ? ab_dgemm_strassen(1.0, A, B, 0.0, C)
                                     : ab_dgemm_scaled(1.0, A, B, 0.0, C);
            if (st != AB_OK) {
                fprintf(stderr, "%s failed: %s\n", kMethod[m], ab_status_string(st));
                exit(1);
            }
            if (m != M_ACCELERATE) ab_matrix_download(C, x, true);

            double worst = 0, num = 0, den = 0, cwab = 0, cw = 0;
            for (int r = 0; r < R; r++) {
                const double *xr = x + (size_t)sampled_row(s, r) * s.N;
                const double *rr = ref.ref + (size_t)r * s.N, *ar = ref.abs + (size_t)r * s.N;
                for (int j = 0; j < s.N; j++) {
                    double d = fabs(xr[j] - rr[j]);
                    if (isnan(d)) d = INFINITY;
                    worst = fmax(worst, d);
                    num += d * d;
                    den += rr[j] * rr[j];
                    cwab = fmax(cwab, d / (ar[j] + DBL_MIN));
                    cw = fmax(cw, d / (fabs(rr[j]) + DBL_MIN));
                }
            }
            printf("accuracy,%s,%d,%d,%d,%d,%s,%.3e,%.3e,%.3e,%.3e\n", kKind[kind], s.M, s.K,
                   s.N, R, kMethod[m], worst / scale, sqrt(num / den), cwab, cw);
            fflush(stdout);
        }
        ab_matrix_destroy(A); ab_matrix_destroy(B); ab_matrix_destroy(C);
        free(x); free(a); free(b); free(ref.ref); free(ref.abs);
    }
}

// =============================================================================
// main
// =============================================================================

static void usage(void) {
    fprintf(stderr,
            "usage: bench_strassen speed|reference|accuracy|abi [--complex] [--reps R] "
            "[--warm S] [--rows R] [--ref-dir D] MxKxN ...\n");
    exit(2);
}

int main(int argc, char **argv) {
    if (argc < 3) usage();
    const char *mode = argv[1];
    bool cplx = false;
    Shape shapes[64];
    int nshapes = 0;
    for (int i = 2; i < argc; i++) {
        if (!strcmp(argv[i], "--complex")) cplx = true;
        else if (!strcmp(argv[i], "--reps") && i + 1 < argc) g_reps = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--warm") && i + 1 < argc) g_warm = atof(argv[++i]);
        else if (!strcmp(argv[i], "--rows") && i + 1 < argc) g_rows = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--ref-dir") && i + 1 < argc) g_ref_dir = argv[++i];
        else if (nshapes < 64 &&
                 sscanf(argv[i], "%dx%dx%d", &shapes[nshapes].M, &shapes[nshapes].K,
                        &shapes[nshapes].N) == 3)
            nshapes++;
        else usage();
    }
    if (!nshapes || g_reps < 1 || g_reps > 64) usage();

    if (!strcmp(mode, "reference")) {
        if (!g_ref_dir) usage();
        for (int i = 0; i < nshapes; i++) reference_case(shapes[i]);
        return 0;
    }
    if (ab_init() != AB_OK) {
        fprintf(stderr, "no Metal device\n");
        return 1;
    }
    fprintf(stderr, "%s on %s\n", ab_build_info(), ab_device_name());
    if (!strcmp(mode, "speed")) {
        speed_header();
        for (int i = 0; i < nshapes; i++) speed_case(shapes[i], cplx);
    } else if (!strcmp(mode, "accuracy")) {
        accuracy_header();
        for (int i = 0; i < nshapes; i++) accuracy_case(shapes[i]);
    } else if (!strcmp(mode, "abi")) {
        speed_header();
        for (int i = 0; i < nshapes; i++) abi_case(shapes[i], cplx);
    } else {
        usage();
    }
    ab_shutdown();
    return 0;
}
