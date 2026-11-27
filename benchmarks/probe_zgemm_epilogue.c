// =============================================================================
// probe_zgemm_epilogue.c: a ZGEMM's FP64 epilogue at the edge of FP64's range
// =============================================================================
//   AB_MODE=gpu AB_MIN_GPU_DIM=0 build/probe_zgemm_epilogue overflow
//   AB_MODE=gpu AB_MIN_GPU_DIM=0 build/probe_zgemm_epilogue bits
//   AB_MODE=gpu build/probe_zgemm_epilogue time TA TB M N K ar ai br bi W R
// (each again with AB_HOST_PACK=1 for the host path; `time` under machlock
// bench). The calls behind CHANGELOG 1.6.0's "A ZGEMM through the BLAS ABI
// whose FP64 epilogue could overflow runs on Accelerate" (results:
// benchmarks/results/2026-11-09-f0c325b-zgemm-epilogue/).
//   overflow  one line per call: ab_zgemm_blas or ab_dgemm_blas, then
//             Accelerate, on 32 x 32 x 32 data whose product, alpha times it
//             or beta times C reaches FP64's range, and ordinary controls;
//             then the reproduction through ab_dev_zgemm. The path, whether
//             the staged path kept the call, how far the GEMM counters
//             moved, how many parts of C are not finite, whether C is
//             Accelerate's bit for bit, and C[0] from both.
//   bits      100 ordinary ZGEMMs (5 shapes, 4 TRANS pairs, 5 scalar sets:
//             the GPU epilogue's and the host epilogue's): one FNV-1a hash of
//             C per call (fnv1a), to compare two libraries line by line.
//   time      W warm-up and R timed calls of one ab_zgemm_blas on uniform
//             data, C reset each time: every call's ms, the median and the
//             smallest, the path, and a hash of the last C.
// Correctness only, except `time`.
// =============================================================================

#include "apple_bottom.h"
#include "apple_bottom_device.h"
#include "test_blas_check.h"
#include <stdio.h>
#include <time.h>

// Exported for the tests (src/gpu_stage.h, declared here so that the probe
// also builds against an older library's include/ alone): whether this
// thread's last BLAS-ABI call kept the staged path's result.
bool ab_internal_last_staged(int *in_place);

static uint64_t rng_state = 1;
static double urand(void) {   // splitmix64, in [-1, 1)
    const uint64_t s = rng_state;
    rng_state += 0x9E3779B97F4A7C15ULL;
    return (double)(splitmix(s) >> 11) * 0x1p-52 - 1.0;
}

// Uniform op(A) (M x K), op(B) (K x N) and C (M x N), stored with no padding;
// each element's real part drawn before its imaginary part.
typedef struct {
    Layout la, lb;
    double complex *A, *B, *C;
} Operands;

static void fill_uniform(double complex *x, size_t n) {
    for (size_t i = 0; i < n; i++) {
        const double re = urand();
        x[i] = CMPLX(re, urand());
    }
}

static Operands make_operands(char ta, char tb, int M, int N, int K) {
    Operands o = {stored(ta, M, K, 0), stored(tb, K, N, 0), NULL, NULL, NULL};
    o.A = malloc(sizeof *o.A * nelem(o.la));
    o.B = malloc(sizeof *o.B * nelem(o.lb));
    o.C = malloc(sizeof *o.C * (size_t)M * N);
    fill_uniform(o.A, nelem(o.la));
    fill_uniform(o.B, nelem(o.lb));
    fill_uniform(o.C, (size_t)M * N);
    return o;
}

static void free_operands(Operands *o) { free(o->A); free(o->B); free(o->C); }

// -----------------------------------------------------------------------------
// overflow
// -----------------------------------------------------------------------------

typedef struct {
    const char *name;
    char routine;
    double a, b, c;                 // every entry of A, B and C (real parts)
    double complex alpha, beta;
    double im;                      // A's and B's imaginary parts, as a fraction of the real
} Probe;

static void probe(const Probe *p) {
    const int n = 32;
    const size_t nn = (size_t)n * n;
    const bool z = p->routine == 'z';
    const size_t es = z ? sizeof(double complex) : sizeof(double);
    void *A = malloc(es * nn), *B = malloc(es * nn), *C = malloc(es * nn), *R = malloc(es * nn);
    for (size_t i = 0; i < nn; i++) {
        if (z) {
            ((double complex *)A)[i] = CMPLX(p->a, p->im * p->a);
            ((double complex *)B)[i] = CMPLX(p->b, p->im * p->b);
            ((double complex *)C)[i] = ((double complex *)R)[i] = p->c;
        } else {
            ((double *)A)[i] = p->a;
            ((double *)B)[i] = p->b;
            ((double *)C)[i] = ((double *)R)[i] = p->c;
        }
    }
    const uint64_t g0 = gpu_gemm_count();
    if (z)
        ab_zgemm_blas('N', 'N', n, n, n, p->alpha, A, n, B, n, p->beta, C, n);
    else
        ab_dgemm_blas('N', 'N', n, n, n, creal(p->alpha), A, n, B, n, creal(p->beta), C, n);
    const uint64_t dg = gpu_gemm_count() - g0;
    const char *path = ab_get_last_dispatch_path();
    const bool staged = ab_internal_last_staged(NULL);
    if (z)
        cblas_zgemm(CblasColMajor, CblasNoTrans, CblasNoTrans, n, n, n, &p->alpha, A, n, B, n, &p->beta, R, n);
    else
        cblas_dgemm(CblasColMajor, CblasNoTrans, CblasNoTrans, n, n, n, creal(p->alpha), A, n, B, n,
                    creal(p->beta), R, n);
    int bad = 0;
    for (size_t i = 0; i < (z ? 2 : 1) * nn; i++) bad += !isfinite(((double *)C)[i]);
    printf("%-36s path=%-3s staged=%d gemms=%llu nonfinite_parts=%d same_as_accelerate=%d ", p->name, path,
           staged, (unsigned long long)dg, bad, !memcmp(C, R, es * nn));
    if (z)
        printf("C[0]=%a%+ai Accelerate=%a%+ai\n", ((double *)C)[0], ((double *)C)[1], ((double *)R)[0],
               ((double *)R)[1]);
    else
        printf("C[0]=%a Accelerate=%a\n", ((double *)C)[0], ((double *)R)[0]);
    free(A); free(B); free(C); free(R);
}

// The reproduction through the device API, which has no Accelerate fallback.
static void probe_dev(double alpha_re) {
    const int n = 32;
    const size_t bytes = sizeof(double complex) * n * n;
    double complex *A = malloc(bytes), *B = malloc(bytes), *C = malloc(bytes);
    for (int i = 0; i < n * n; i++) { A[i] = 0x1p600; B[i] = 0x1p500; C[i] = 0.0; }
    const double alpha[2] = {alpha_re, 0.0}, beta[2] = {0.0, 0.0};
    ab_dev_buffer_t dA = ab_dev_malloc(bytes), dB = ab_dev_malloc(bytes), dC = ab_dev_malloc(bytes);
    ABStatus st = dA && dB && dC ? AB_OK : AB_ERROR_ALLOC_FAILED;
    if (st == AB_OK) st = ab_dev_memcpy_h2d(dA, 0, A, bytes);
    if (st == AB_OK) st = ab_dev_memcpy_h2d(dB, 0, B, bytes);
    if (st == AB_OK) st = ab_dev_memcpy_h2d(dC, 0, C, bytes);
    if (st == AB_OK) st = ab_dev_zgemm(AB_NO_TRANS, AB_NO_TRANS, n, n, n, alpha, dA, n, dB, n, beta, dC, n);
    if (st == AB_OK) st = ab_dev_memcpy_d2h(C, dC, 0, bytes);
    int bad = 0;
    for (int i = 0; i < n * n; i++) bad += !isfinite(creal(C[i])) + !isfinite(cimag(C[i]));
    printf("ab_dev_zgemm A=2^600 B=2^500 alpha=%a  status=%s nonfinite_parts=%d C[0]=%a%+ai\n", alpha_re,
           ab_status_string(st), bad, creal(C[0]), cimag(C[0]));
    ab_dev_free(dA); ab_dev_free(dB); ab_dev_free(dC);
    free(A); free(B); free(C);
}

static void run_overflow(void) {
    const Probe P[] = {
        // The reproduction: C = 32 * 2^1100 * 2^-200 = 2^905.
        {"z repro alpha=2^-200 beta=0", 'z', 0x1p600, 0x1p500, 0.0, 0x1p-200, 0.0, 0.0},
        {"z alpha=1 beta=0 P=2^1105", 'z', 0x1p600, 0x1p500, 0.0, 1.0, 0.0, 0.0},
        {"z alpha=1 beta=1 P=2^1105", 'z', 0x1p600, 0x1p500, 1.0, 1.0, 1.0, 0.0},
        {"z alpha=1 beta=1 Cin=2^1023", 'z', 1.0, 1.0, 0x1p1023, 1.0, 1.0, 0.0},
        {"z alpha=1 beta=1 P=2^1023 Cin=2^1023", 'z', 0x1p509, 0x1p509, 0x1p1023, 1.0, 1.0, 0.0},
        {"z alpha=1 beta=2 Cin=2^1023", 'z', 1.0, 1.0, 0x1p1023, 1.0, 2.0, 0.0},
        {"z alpha=2^8 P=2^1020", 'z', 0x1p508, 0x1p507, 0.0, 0x1p8, 0.0, 0.0},
        {"z alpha=1+i complex P", 'z', 0x1p505, 0x1p505, 0.0, CMPLX(1.0, 1.0), 0.0, 1.0},
        {"z alpha=Inf ordinary", 'z', 1.0, 1.0, 0.0, INFINITY, 0.0, 0.0},
        {"z alpha=NaN ordinary", 'z', 1.0, 1.0, 0.0, NAN, 0.0, 0.0},
        {"z beta=Inf ordinary", 'z', 1.0, 1.0, 1.0, 1.0, INFINITY, 0.0},
        {"z ordinary alpha=0.5-0.25i beta=0.5", 'z', 1.0, 1.0, 1.0, CMPLX(0.5, -0.25), 0.5, 0.25},
        // DGEMM applies alpha and beta in DD before its one scale-back.
        {"d repro alpha=2^-200 beta=0", 'd', 0x1p600, 0x1p500, 0.0, 0x1p-200, 0.0, 0.0},
        {"d alpha=1 beta=0 P=2^1105", 'd', 0x1p600, 0x1p500, 0.0, 1.0, 0.0, 0.0},
        {"d alpha=1 beta=1 P=2^1023 Cin=2^1023", 'd', 0x1p509, 0x1p509, 0x1p1023, 1.0, 1.0, 0.0},
        {"d alpha=2^8 P=2^1020", 'd', 0x1p508, 0x1p507, 0.0, 0x1p8, 0.0, 0.0},
        {"d alpha=1 beta=2 Cin=2^1023", 'd', 1.0, 1.0, 0x1p1023, 1.0, 2.0, 0.0},
        {"d alpha=0.5 P=2^1023 (finite)", 'd', 0x1p509, 0x1p509, 0.0, 0.5, 0.0, 0.0},
    };
    for (size_t i = 0; i < sizeof P / sizeof *P; i++) probe(&P[i]);
    probe_dev(0x1p-200);
    probe_dev(1.0);
}

// -----------------------------------------------------------------------------
// bits
// -----------------------------------------------------------------------------

static void run_bits(void) {
    static const int S[][3] = {{96, 80, 72}, {257, 129, 300}, {512, 512, 512}, {160, 160, 1152}, {1000, 33, 500}};
    static const char T[][2] = {{'N', 'N'}, {'C', 'N'}, {'N', 'T'}, {'T', 'C'}};
    const double complex AL[] = {1.0, 1.0, CMPLX(0.75, -0.5), 2.0, CMPLX(-1.25, 0.25)};
    const double complex BE[] = {0.0, 1.0, 0.0, CMPLX(0.5, 0.5), -1.0};
    for (size_t s = 0; s < sizeof S / sizeof *S; s++)
        for (size_t t = 0; t < sizeof T / sizeof *T; t++)
            for (size_t a = 0; a < sizeof AL / sizeof *AL; a++) {
                const int M = S[s][0], N = S[s][1], K = S[s][2];
                const char ta = T[t][0], tb = T[t][1];
                Operands o = make_operands(ta, tb, M, N, K);
                ab_zgemm_blas(ta, tb, M, N, K, AL[a], o.A, o.la.ld, o.B, o.lb.ld, BE[a], o.C, M);
                printf("%dx%dx%d %c%c alpha=%g%+gi beta=%g%+gi path=%s hash=%016llx\n", M, N, K, ta, tb,
                       creal(AL[a]), cimag(AL[a]), creal(BE[a]), cimag(BE[a]), ab_get_last_dispatch_path(),
                       (unsigned long long)fnv1a(o.C, sizeof *o.C * M * N, FNV_OFFSET));
                free_operands(&o);
            }
}

// -----------------------------------------------------------------------------
// time
// -----------------------------------------------------------------------------

static double now_ms(void) {
    struct timespec t;
    clock_gettime(CLOCK_MONOTONIC_RAW, &t);
    return t.tv_sec * 1e3 + t.tv_nsec / 1e6;
}

static int cmp_double(const void *a, const void *b) {
    double x = *(const double *)a, y = *(const double *)b;
    return (x > y) - (x < y);
}

static int run_time(char **v) {
    const char ta = v[0][0], tb = v[1][0];
    const int M = atoi(v[2]), N = atoi(v[3]), K = atoi(v[4]);
    const double complex alpha = CMPLX(atof(v[5]), atof(v[6])), beta = CMPLX(atof(v[7]), atof(v[8]));
    const int W = atoi(v[9]), R = atoi(v[10]);
    if (M < 1 || N < 1 || K < 1 || W < 0 || R < 1) return 2;
    Operands o = make_operands(ta, tb, M, N, K);     // o.C holds C on entry
    double complex *C = malloc(sizeof *C * (size_t)M * N);
    double *t = malloc(sizeof *t * R);
    for (int i = 0; i < W + R; i++) {
        memcpy(C, o.C, sizeof *C * (size_t)M * N);
        const double t0 = now_ms();
        ab_zgemm_blas(ta, tb, M, N, K, alpha, o.A, o.la.ld, o.B, o.lb.ld, beta, C, M);
        if (i >= W) t[i - W] = now_ms() - t0;
    }
    printf("calls_ms");
    for (int i = 0; i < R; i++) printf(" %.4f", t[i]);
    qsort(t, R, sizeof *t, cmp_double);
    printf("\nmedian_ms %.4f min_ms %.4f path %s staged %d hash %016llx\n", t[R / 2], t[0],
           ab_get_last_dispatch_path(), ab_internal_last_staged(NULL),
           (unsigned long long)fnv1a(C, sizeof *C * (size_t)M * N, FNV_OFFSET));
    free_operands(&o);
    free(C);
    free(t);
    return 0;
}

int main(int argc, char **argv) {
    if (argc == 2 && !strcmp(argv[1], "overflow")) {
        run_overflow();
        return 0;
    }
    if (argc == 2 && !strcmp(argv[1], "bits")) {
        run_bits();
        return 0;
    }
    if (argc == 13 && !strcmp(argv[1], "time") && run_time(argv + 2) == 0) return 0;
    fprintf(stderr, "usage: %s overflow | bits | time TA TB M N K ar ai br bi W R\n", argv[0]);
    return 2;
}
