// bench_rect_zgemm — shape-list-driven ZGEMM benchmark (REVIEW_04 Tranche A).
// See bench_rect_dgemm.c for CLI and CSV schema (identical). This variant
// calls ab_zgemm_blas with interleaved complex arrays and scores Frobenius
// relative error against an Accelerate cblas_zgemm reference.

#include "apple_bottom.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <complex.h>
#include <time.h>
#include <stdbool.h>
#include <Accelerate/Accelerate.h>

typedef struct { int M, N, K; char tag[64]; } Shape;

static double now_sec(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec + ts.tv_nsec * 1e-9;
}

static void iso_utc(char *buf, size_t n) {
    time_t t = time(NULL);
    struct tm tm;
    gmtime_r(&t, &tm);
    strftime(buf, n, "%Y-%m-%dT%H:%M:%SZ", &tm);
}

static int load_shapes(const char *path, Shape **out_shapes, int *out_n) {
    FILE *f = fopen(path, "r");
    if (!f) { fprintf(stderr, "cannot open config: %s\n", path); return -1; }
    int cap = 32, n = 0;
    Shape *sh = malloc(cap * sizeof(Shape));
    char line[512];
    while (fgets(line, sizeof(line), f)) {
        char *p = line;
        while (*p == ' ' || *p == '\t') p++;
        if (*p == '#' || *p == '\n' || *p == '\0') continue;
        Shape s;
        if (sscanf(p, "%d %d %d %63s", &s.M, &s.N, &s.K, s.tag) != 4) continue;
        if (n == cap) { cap *= 2; sh = realloc(sh, cap * sizeof(Shape)); }
        sh[n++] = s;
    }
    fclose(f);
    *out_shapes = sh;
    *out_n = n;
    return 0;
}

int main(int argc, char **argv) {
    const char *config = NULL, *mode = "auto", *output = NULL;
    int runs = 5, warmup = 2;
    bool verify = false;
    for (int i = 1; i < argc; i++) {
        if (!strcmp(argv[i], "--config") && i+1 < argc) config = argv[++i];
        else if (!strcmp(argv[i], "--mode") && i+1 < argc) mode = argv[++i];
        else if (!strcmp(argv[i], "--runs") && i+1 < argc) runs = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--warmup") && i+1 < argc) warmup = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--output") && i+1 < argc) output = argv[++i];
        else if (!strcmp(argv[i], "--verify")) verify = true;
        else { fprintf(stderr, "unknown arg: %s\n", argv[i]); return 1; }
    }
    if (!config || !output) {
        fprintf(stderr, "usage: %s --config PATH --mode MODE --output CSV [--runs N] [--warmup N] [--verify]\n", argv[0]);
        return 1;
    }

    const char *env_mode = getenv("AB_MODE");
    if (!env_mode || (env_mode[0] != mode[0] && !(env_mode[0]=='\0' && !strcmp(mode,"auto")))) {
        fprintf(stderr, "warning: --mode=%s but AB_MODE=%s; the runner should set both\n",
                mode, env_mode ? env_mode : "(unset)");
    }

    Shape *shapes = NULL;
    int ns = 0;
    if (load_shapes(config, &shapes, &ns) != 0) return 1;

    if (ab_init() != AB_OK) { fprintf(stderr, "ab_init failed\n"); return 1; }

    bool need_header = true;
    {
        FILE *probe = fopen(output, "r");
        if (probe) { need_header = false; fclose(probe); }
    }
    FILE *csv = fopen(output, "a");
    if (!csv) { fprintf(stderr, "cannot open output: %s\n", output); return 1; }
    if (need_header)
        fprintf(csv, "timestamp,mode,tag,M,N,K,run_idx,gflops,wall_s,frob_rel_err,dispatch_hint\n");

    printf("bench_rect_zgemm: %d shapes, mode=%s, runs=%d, warmup=%d, verify=%d\n",
           ns, mode, runs, warmup, verify);
    printf("  tag                      M      N      K   dispatch   gflops (mean)   frob_rel_err\n");
    printf("─────────────────────────────────────────────────────────────────────────────────\n");

    for (int s = 0; s < ns; s++) {
        Shape sh = shapes[s];
        int M = sh.M, N = sh.N, K = sh.K;
        size_t szA = (size_t)M * K, szB = (size_t)K * N, szC = (size_t)M * N;
        double _Complex *A = malloc(szA * sizeof(double _Complex));
        double _Complex *B = malloc(szB * sizeof(double _Complex));
        double _Complex *C = malloc(szC * sizeof(double _Complex));
        double _Complex *C_ref = verify ? malloc(szC * sizeof(double _Complex)) : NULL;
        if (!A || !B || !C || (verify && !C_ref)) {
            fprintf(stderr, "alloc failed at shape %s\n", sh.tag); return 1;
        }
        srand48(42);
        for (size_t i = 0; i < szA; i++) A[i] = (drand48()*2-1) + I*(drand48()*2-1);
        for (size_t i = 0; i < szB; i++) B[i] = (drand48()*2-1) + I*(drand48()*2-1);

        double _Complex alpha = 1.0 + 0*I, beta = 0.0 + 0*I;
        if (verify) {
            memset(C_ref, 0, szC * sizeof(double _Complex));
            cblas_zgemm(CblasColMajor, CblasNoTrans, CblasNoTrans,
                        M, N, K, &alpha, A, M, B, K, &beta, C_ref, M);
        }

        for (int w = 0; w < warmup; w++) {
            memset(C, 0, szC * sizeof(double _Complex));
            ab_zgemm_blas('N','N', M,N,K, alpha, A, M, B, K, beta, C, M);
        }

        double flops = 8.0 * (double)M * (double)N * (double)K;
        double sum_g = 0; const char *last_path = "none";
        double frob = -1.0;

        for (int r = 0; r < runs; r++) {
            memset(C, 0, szC * sizeof(double _Complex));
            double t0 = now_sec();
            ab_zgemm_blas('N','N', M,N,K, alpha, A, M, B, K, beta, C, M);
            double wall = now_sec() - t0;
            double g = flops / (wall * 1e9);
            last_path = ab_get_last_dispatch_path();
            sum_g += g;

            if (verify) {
                double num = 0, den = 0;
                for (size_t i = 0; i < szC; i++) {
                    double dr = creal(C[i]) - creal(C_ref[i]);
                    double di = cimag(C[i]) - cimag(C_ref[i]);
                    num += dr*dr + di*di;
                    double rr = creal(C_ref[i]), ii = cimag(C_ref[i]);
                    den += rr*rr + ii*ii;
                }
                frob = (den > 0) ? sqrt(num / den) : 0.0;
            }
            char ts[32]; iso_utc(ts, sizeof(ts));
            fprintf(csv, "%s,%s,%s,%d,%d,%d,%d,%.4f,%.6f,%.3e,%s\n",
                    ts, mode, sh.tag, M, N, K, r, g, wall,
                    verify ? frob : -1.0, last_path);
            fflush(csv);
        }
        // Verify guard is dispatch-aware: cpu path is cblas vs cblas, so
        // frob = 0.0 is legitimate (bit-identical). gpu path is DD-FP32×2
        // vs cblas FP64 reference, so frob must land in (0, 1e-12].
        // frob < 1e-18 on a gpu row means the verify loop is silently
        // comparing identical arrays (bug), so we reject that too.
        const char *verdict = "-";
        if (verify) {
            bool is_gpu = (last_path && strcmp(last_path, "gpu") == 0);
            if (is_gpu) {
                verdict = (frob < 1e-18) ? "FAIL(too-close)"
                        : (frob < 1e-12) ? "ok"
                        : "FAIL(precision)";
            } else {
                verdict = (frob == 0.0) ? "ok(bit-id)"
                        : (frob < 1e-12) ? "ok"
                        : "FAIL(precision)";
            }
        }
        printf("  %-22s %6d %6d %6d   %-8s   %10.2f      %s\n",
               sh.tag, M, N, K, last_path, sum_g / runs, verdict);
        free(A); free(B); free(C); if (C_ref) free(C_ref);
    }
    fclose(csv);
    free(shapes);
    ab_shutdown();
    return 0;
}
