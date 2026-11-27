// =============================================================================
// bench_fft.c — the double-float GPU FFT against FFTW and vDSP, FP64 in and out
// =============================================================================
//
// For each n^3 grid and batch size: time per forward 3D complex transform of
//   gpu-staged  ab_fft_execute on malloc'd arrays (copies in and out included)
//   gpu-zc      arrays from ab_fft_malloc (zero copy, Metal-allocated memory)
//   gpu-reg     the malloc'd arrays registered with ab_fft_register (zero copy)
//   fftw-1      FFTW, one thread, plan of one transform (FFTW_PATIENT with
//               --patient, else FFTW_MEASURE)
//   fftw-8      FFTW, one plan with 8 threads (fftw3_threads)
//   fftw-8x1    8 workers, each running single-thread plans on 1/8 of the
//               batch, as MPI ranks with QE_FFTW_THREADS=1 do
//   vdsp-1/-8   vDSP_DFT_Interleaved_ExecuteD line by line (1 or 8 threads),
//               for the lengths vDSP supports (f * 2^k, f in 2,3,5,9,15,25)
// Each method gets a time-based warm-up (--warm, default 3 s: the GPU raises
// its clock only after sustained load, and short calls are bursty), then the
// median of at least --reps calls (as many as fit in a second, up to 64). Its first output transform is then compared with
// FFTW's (relative L2 difference in the note), so a method that computes the
// wrong thing cannot post a time.
//
// --accuracy: relative L2 and max error of the GPU and of FFTW against a
// double-double reference (tests/fft_ref.h), on 53-bit inputs, and of the GPU
// against the reference of its double-float-rounded input (transform only).
//
// Build: make bench-fft (Homebrew FFTW) or make bench-fft FFTW_PREFIX=<dir>.
// Run under the machine lock: machlock.py bench -- build/bench_fft ...
// =============================================================================

#include <Accelerate/Accelerate.h>
#include <dispatch/dispatch.h>
#include <fftw3.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#include "apple_bottom_fft.h"
#include "../tests/fft_ref.h"
#include "../tests/test_blas_check.h"   // splitmix

#ifndef FFTW_LABEL
#define FFTW_LABEL "fftw"
#endif

static double now_s(void) { return clock_gettime_nsec_np(CLOCK_UPTIME_RAW) * 1e-9; }

static int cmp_d(const void* a, const void* b)
{
    double x = *(const double*)a, y = *(const double*)b;
    return x < y ? -1 : x > y;
}

static double median(double* v, int n)
{
    qsort(v, (size_t)n, sizeof *v, cmp_d);
    return n % 2 ? v[n / 2] : 0.5 * (v[n / 2 - 1] + v[n / 2]);
}

static uint64_t g_rng = 1;
static void fill(double* x, long n)   // n values in [-1, 1), 53 random bits
{
    for (long i = 0; i < n; ++i) {
        x[i] = (double)(splitmix(g_rng) >> 11) * (2.0 / 9007199254740992.0) - 1.0;
        g_rng += 0x9E3779B97F4A7C15ull;
    }
}

typedef struct {
    int grids[16], ngrids, batches[8], nbatches, reps, threads, accuracy, speed;
    unsigned plan_flags;
    double warm;
    const char* only;
    FILE* csv;
} opts;

// Run fn() until warm seconds have passed (at least twice), then timed calls:
// --reps, or as many as fit in a second, up to 64. Returns the median
// seconds per call.
typedef void (^job_t)(void);
static double time_job(job_t fn, const opts* o)
{
    const double t0 = now_s();
    int k = 0;
    while (k < 2 || now_s() - t0 < o->warm) { fn(); ++k; }
    double t[64];
    const double per = (now_s() - t0) / k;
    int r = per > 0 ? (int)(1.0 / per) : 64;
    if (r < o->reps) r = o->reps;
    if (r > 64) r = 64;
    for (int i = 0; i < r; ++i) {
        const double a = now_s();
        fn();
        t[i] = now_s() - a;
    }
    return median(t, r);
}

static int want(const opts* o, const char* m) { return !o->only || strstr(o->only, m); }

// Relative L2 difference of one transform's output from FFTW's.
static double vs_ref(const double* y, const double* ref, long N)
{
    double d = 0, r = 0;
    for (long i = 0; i < 2 * N; ++i) { d += (y[i] - ref[i]) * (y[i] - ref[i]); r += ref[i] * ref[i]; }
    return sqrt(d / r);
}

static void report(const opts* o, int n, int B, const char* method, double s, const double* out, const double* ref,
                   const char* note)
{
    const double N = (double)n * n * n, ms = s / B * 1e3;
    const double gf = 5.0 * N * log2(N) / (s / B) * 1e-9;   // FFTW's flop convention
    const double err = vs_ref(out, ref, (long)N);
    printf("  %4d^3 x%-4d %-14s %10.4f ms/transform %8.1f GF/s  vs FFTW %.1e  %s\n", n, B, method, ms, gf, err,
           note ? note : "");
    if (o->csv) fprintf(o->csv, "%d,%d,%s,%.6f,%.2f,%.2e,%s\n", n, B, method, ms, gf, err, note ? note : "");
    fflush(stdout);
}

// ---- vDSP: line by line on interleaved data, tiles of 16 lines for y and z --
static int vdsp_ok(int n)
{
    vDSP_DFT_Interleaved_SetupD s = vDSP_DFT_Interleaved_CreateSetupD(NULL, (vDSP_Length)n, vDSP_DFT_FORWARD,
                                                                       vDSP_DFT_Interleaved_ComplextoComplex);
    if (s) vDSP_DFT_Interleaved_DestroySetupD(s);
    return s != NULL;
}

// First element of line g of an axis with stride s, in packed n^3 cubes: the
// lines of a cube start at lo + hi*s*n, lo < s.
static long line_start(long g, int n, long s)
{
    const long N = (long)n * n * n, per_cube = N / n, r = g % per_cube;
    return g / per_cube * N + r % s + r / s * s * n;
}

// One axis of packed n^3 cubes: lines [l0, l1) of the axis.
static void vdsp_axis(const double* in, double* out, int n, int axis, long l0, long l1,
                      vDSP_DFT_Interleaved_SetupD su, DSPDoubleComplex* t1, DSPDoubleComplex* t2)
{
    const long s = axis == 0 ? 1 : (axis == 1 ? n : (long)n * n);
    const DSPDoubleComplex* ci = (const DSPDoubleComplex*)in;
    DSPDoubleComplex* co = (DSPDoubleComplex*)out;
    if (axis == 0) {
        for (long l = l0; l < l1; ++l) vDSP_DFT_Interleaved_ExecuteD(su, ci + l * n, co + l * n);
        return;
    }
    // 16 lines with consecutive lo (contiguous in memory) at a time, as a [16][n] tile.
    for (long l = l0; l < l1; l += 16) {
        const long cnt = (l1 - l) < 16 ? (l1 - l) : 16;
        for (long k = 0; k < cnt; ++k) {
            const DSPDoubleComplex* src = ci + line_start(l + k, n, s);
            for (long i = 0; i < n; ++i) t1[k * n + i] = src[i * s];
        }
        for (long k = 0; k < cnt; ++k) vDSP_DFT_Interleaved_ExecuteD(su, t1 + k * n, t2 + k * n);
        for (long k = 0; k < cnt; ++k) {
            DSPDoubleComplex* dst = co + line_start(l + k, n, s);
            for (long i = 0; i < n; ++i) dst[i * s] = t2[k * n + i];
        }
    }
}

// Forward 3D DFT of nb cubes with nthr threads; y and z run in place on out.
static void vdsp_3d(const double* in, double* out, int n, long nb, int nthr, vDSP_DFT_Interleaved_SetupD* su)
{
    const long nlines = nb * (long)n * n;
    for (int axis = 0; axis < 3; ++axis) {
        const double* src = axis == 0 ? in : out;
        const long per = ((nlines + nthr - 1) / nthr + 15) / 16 * 16;   // whole tiles per thread
        dispatch_apply((size_t)nthr, dispatch_get_global_queue(QOS_CLASS_USER_INITIATED, 0), ^(size_t t) {
            const long l0 = (long)t * per, l1 = l0 + per < nlines ? l0 + per : nlines;
            if (l0 >= l1) return;
            DSPDoubleComplex* t1 = malloc(sizeof(DSPDoubleComplex) * 16 * (size_t)n);
            DSPDoubleComplex* t2 = malloc(sizeof(DSPDoubleComplex) * 16 * (size_t)n);
            vdsp_axis(src, out, n, axis, l0, l1, su[t], t1, t2);
            free(t1);
            free(t2);
        });
    }
}

// ---- speed -------------------------------------------------------------------
static void bench_gpu(const opts* o, int n, int B, double* in, double* out, const double* ref)
{
    const long N = (long)n * n * n;
    const size_t bytes = (size_t)(N * B) * 16;
    ABStatus st;
    ab_fft_plan_t p = ab_fft_plan_3d(n, n, n, B, AB_FFT_FORWARD, AB_FFT_NORM_NONE, &st);
    if (!p) { printf("  plan failed: %s\n", ab_fft_last_error()); return; }
    char d[512], note[128];
    ab_fft_describe(p, d, sizeof d);
    printf("  [%s]\n", d);
    __block ab_fft_stats stt;
    __block double* src = in;
    __block double* dst = out;
    job_t run = ^{
        ab_fft_execute(p, src, dst);
        ab_fft_get_stats(p, &stt);
    };
    double s = time_job(run, o);
    snprintf(note, sizeof note, "gpu %.3f copy-in %.3f copy-out %.3f ms per call", stt.gpu_ms, stt.copy_in_ms,
             stt.copy_out_ms);
    report(o, n, B, "gpu-staged", s, out, ref, note);

    double* zi = ab_fft_malloc(bytes);
    double* zo = ab_fft_malloc(bytes);
    if (zi && zo) {
        memcpy(zi, in, bytes);
        src = zi; dst = zo;
        s = time_job(run, o);
        snprintf(note, sizeof note, "gpu %.3f ms per call", stt.gpu_ms);
        report(o, n, B, "gpu-zc", s, zo, ref, note);
    } else {
        printf("  ab_fft_malloc failed: %s\n", ab_fft_last_error());
    }
    ab_fft_free(zi);
    ab_fft_free(zo);

    if (ab_fft_register(in, bytes) == AB_OK && ab_fft_register(out, bytes) == AB_OK) {
        src = in; dst = out;
        s = time_job(run, o);
        snprintf(note, sizeof note, "gpu %.3f ms per call", stt.gpu_ms);
        report(o, n, B, "gpu-reg", s, out, ref, note);
    } else {
        printf("  register failed: %s\n", ab_fft_last_error());
    }
    ab_fft_unregister(in);
    ab_fft_unregister(out);
    ab_fft_destroy(p);
}

// Plan of one transform with nthreads threads, run over the batch by
// `workers` concurrent workers. Wisdom carries over within the process, so
// the second batch size reuses the plan search.
static void bench_fftw(const opts* o, int n, int B, int nthreads, int workers, const char* label, double* in,
                       double* out, const double* ref)
{
    const long N = (long)n * n * n;
    fftw_plan_with_nthreads(nthreads);
    double* scratch = malloc((size_t)N * 32);   // MEASURE overwrites its arrays
    fftw_plan fp = fftw_plan_dft_3d(n, n, n, (fftw_complex*)scratch, (fftw_complex*)(scratch + 2 * N), FFTW_FORWARD,
                                    o->plan_flags);
    free(scratch);
    const double s = time_job(^{
        dispatch_apply((size_t)workers, dispatch_get_global_queue(QOS_CLASS_USER_INITIATED, 0), ^(size_t t) {
            for (int b = (int)t; b < B; b += workers)
                fftw_execute_dft(fp, (fftw_complex*)(in + 2 * N * b), (fftw_complex*)(out + 2 * N * b));
        });
    }, o);
    report(o, n, B, label, s, out, ref, NULL);
    fftw_destroy_plan(fp);
}

static void bench_speed(const opts* o, int n, int B)
{
    const long N = (long)n * n * n;
    const size_t bytes = (size_t)(N * B) * 16;
    double* in = malloc(bytes);
    double* out = malloc(bytes);
    double* ref = malloc((size_t)N * 16);
    if (!in || !out || !ref) {
        printf("  %d^3 x%d: out of memory\n", n, B);
        free(in); free(out); free(ref);
        return;
    }
    fill(in, 2 * N * B);
    memset(out, 0, bytes);
    fftw_plan_with_nthreads(1);
    fftw_plan fr = fftw_plan_dft_3d(n, n, n, (fftw_complex*)in, (fftw_complex*)ref, FFTW_FORWARD, FFTW_ESTIMATE);
    fftw_execute(fr);   // the reference output of transform 0
    fftw_destroy_plan(fr);

    if (want(o, "gpu")) bench_gpu(o, n, B, in, out, ref);
    if (want(o, "fftw")) {
        char lab[32];
        bench_fftw(o, n, B, 1, 1, FFTW_LABEL "-1", in, out, ref);
        if (B >= o->threads && want(o, "8x1")) {
            snprintf(lab, sizeof lab, FFTW_LABEL "-%dx1", o->threads);
            bench_fftw(o, n, B, 1, o->threads, lab, in, out, ref);
        }
        snprintf(lab, sizeof lab, FFTW_LABEL "-%d", o->threads);
        bench_fftw(o, n, B, o->threads, 1, lab, in, out, ref);
    }
    if (want(o, "vdsp")) {
        if (!vdsp_ok(n)) {
            printf("  %4d^3 x%-4d vdsp           (length %d not supported by vDSP_DFT_Interleaved)\n", n, B, n);
        } else {
            const int nthr[2] = {1, o->threads};
            for (int i = 0; i < (o->threads > 1 ? 2 : 1); ++i) {
                const int nt = nthr[i];
                vDSP_DFT_Interleaved_SetupD* su = malloc(sizeof *su * (size_t)nt);
                for (int t = 0; t < nt; ++t)
                    su[t] = vDSP_DFT_Interleaved_CreateSetupD(NULL, (vDSP_Length)n, vDSP_DFT_FORWARD,
                                                              vDSP_DFT_Interleaved_ComplextoComplex);
                const double s = time_job(^{ vdsp_3d(in, out, n, B, nt, su); }, o);
                char lab[32];
                snprintf(lab, sizeof lab, "vdsp-%d", nt);
                report(o, n, B, lab, s, out, ref, NULL);
                for (int t = 0; t < nt; ++t) vDSP_DFT_Interleaved_DestroySetupD(su[t]);
                free(su);
            }
        }
    }
    free(in);
    free(out);
    free(ref);
}

// ---- accuracy ----------------------------------------------------------------
static void bench_accuracy(const opts* o, int n)
{
    const long N = (long)n * n * n, nn[3] = {n, n, n};
    double* x = malloc((size_t)N * 16);
    double* xr = malloc((size_t)N * 16);
    double* y = malloc((size_t)N * 16);
    double* yf = malloc((size_t)N * 16);
    fill(x, 2 * N);
    for (long i = 0; i < 2 * N; ++i) {
        const float h = (float)x[i];
        xr[i] = (double)h + (double)(float)(x[i] - (double)h);   // the GPU's input rounding
    }
    ABStatus st;
    ab_fft_plan_t p = ab_fft_plan_3d(n, n, n, 1, AB_FFT_FORWARD, AB_FFT_NORM_NONE, &st);
    ab_fft_execute(p, x, y);
    ab_fft_destroy(p);
    double* tmp = malloc((size_t)N * 16);
    fftw_plan_with_nthreads(1);
    fftw_plan fp = fftw_plan_dft_3d(n, n, n, (fftw_complex*)tmp, (fftw_complex*)yf, FFTW_FORWARD, FFTW_MEASURE);
    fftw_execute_dft(fp, (fftw_complex*)x, (fftw_complex*)yf);
    fftw_destroy_plan(fp);
    free(tmp);
    const double t0 = now_s();
    ref_cplx* r = ref_transform(x, nn, 1, -1, 0);
    ref_cplx* rr = ref_transform(xr, nn, 1, -1, 0);
    const ref_err eg = ref_error(y, r, N), ef = ref_error(yf, r, N), et = ref_error(y, rr, N);
    printf("  %4d^3  GPU L2 %.2e max %.2e | %s L2 %.2e max %.2e | GPU/FFTW L2 %.1fx | GPU transform only L2 %.2e "
           "max %.2e  (reference %.1f s)\n",
           n, eg.l2, eg.mx, FFTW_LABEL, ef.l2, ef.mx, eg.l2 / ef.l2, et.l2, et.mx, now_s() - t0);
    if (o->csv) fprintf(o->csv, "%d,%.3e,%.3e,%.3e,%.3e,%.3e,%.3e\n", n, eg.l2, eg.mx, ef.l2, ef.mx, et.l2, et.mx);
    free(r); free(rr); free(x); free(xr); free(y); free(yf);
}

static int parse_list(const char* s, int* v, int max)
{
    int k = 0;
    while (*s && k < max) {
        v[k++] = atoi(s);
        while (*s && *s != ',') ++s;
        if (*s == ',') ++s;
    }
    return k;
}

int main(int argc, char** argv)
{
    opts o = {.reps = 3, .threads = 8, .warm = 3.0, .plan_flags = FFTW_MEASURE};
    o.ngrids = parse_list("48,64,72,96,120,128", o.grids, 16);
    o.nbatches = parse_list("1,128", o.batches, 8);
    for (int i = 1; i < argc; ++i) {
        if (!strcmp(argv[i], "--grids") && i + 1 < argc) o.ngrids = parse_list(argv[++i], o.grids, 16);
        else if (!strcmp(argv[i], "--batch") && i + 1 < argc) o.nbatches = parse_list(argv[++i], o.batches, 8);
        else if (!strcmp(argv[i], "--reps") && i + 1 < argc) o.reps = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--threads") && i + 1 < argc) o.threads = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--warm") && i + 1 < argc) o.warm = atof(argv[++i]);
        else if (!strcmp(argv[i], "--only") && i + 1 < argc) o.only = argv[++i];
        else if (!strcmp(argv[i], "--accuracy")) o.accuracy = 1;
        else if (!strcmp(argv[i], "--speed")) o.speed = 1;
        else if (!strcmp(argv[i], "--patient")) o.plan_flags = FFTW_PATIENT;
        else if (!strcmp(argv[i], "--csv") && i + 1 < argc) o.csv = fopen(argv[++i], "w");
        else {
            fprintf(stderr, "usage: %s [--speed] [--accuracy] [--grids 48,72,...] [--batch 1,128] [--reps R] "
                            "[--threads T] [--warm S] [--patient] [--only gpu,fftw,8x1,vdsp] [--csv FILE]\n", argv[0]);
            return 2;
        }
    }
    if (!o.speed && !o.accuracy) o.speed = 1;
    if (o.threads < 1 || o.threads > 64) o.threads = 8;
    fftw_init_threads();
    printf("bench_fft: %s; FFTW %s (%s, %s plans), %d threads, reps %d, warm-up %.1f s\n", ab_build_info(),
           fftw_version, FFTW_LABEL, o.plan_flags == FFTW_PATIENT ? "PATIENT" : "MEASURE", o.threads, o.reps, o.warm);
    if (o.accuracy) {
        printf("accuracy (forward, batch 1, random 53-bit inputs in [-1,1)):\n");
        if (o.csv) fprintf(o.csv, "n,gpu_l2,gpu_max,fftw_l2,fftw_max,gpu_transform_l2,gpu_transform_max\n");
        for (int g = 0; g < o.ngrids; ++g) bench_accuracy(&o, o.grids[g]);
    }
    if (o.speed) {
        printf("speed (forward, complex FP64 in and out):\n");
        if (o.csv) fprintf(o.csv, "n,batch,method,ms_per_transform,gflops,vs_fftw,note\n");
        for (int g = 0; g < o.ngrids; ++g)
            for (int b = 0; b < o.nbatches; ++b) bench_speed(&o, o.grids[g], o.batches[b]);
    }
    if (o.csv) fclose(o.csv);
    return 0;
}
