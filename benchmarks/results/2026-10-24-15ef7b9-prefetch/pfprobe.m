// pfprobe.m — apple-bottom's register-prefetch kernels (*_pf) against their plain twins, using the
// library's own MSL (extracted from src/apple_bottom.m, compiled with MTLMathModeSafe as the library does).
//
// build: clang -O2 -fobjc-arc -ffp-contract=off -framework Foundation -framework Metal pfprobe.m -o pfprobe
// usage: pfprobe <apple_bottom.m> bits
//        pfprobe <apple_bottom.m> time <shapes file> [--rounds R] [--np 1|3|13] [--only v,v,...] [--raw] [--min]
//               [--gap MS]   (idle the GPU MS before each timed run, as between a host-bound caller's calls)
// shapes file: "M N K" per line, '#' starts a comment. dd_dsyrk variants use N = M (A is M x K).
// Run every timing alone on the machine (the repo's runs held an exclusive machine lock).
//
// bits: every variant of a family on ragged shapes and three kinds of data (moderate, wide-range
//   exponents, specials: signed zeros, Inf, NaN, 2^+-100), once alone and once as three concurrent
//   products; each C is memcmp'd with the family's reference (sq, ab or sy) over all M*N elements.
//   Scaled kernels run with alpha = -1.7, beta = 0.3 on a C filled beforehand (the same for every
//   variant); dd_dsyrk's C starts as 0xff bytes so the untouched lower triangle is compared too.
// time: GPU time (GPUEndTime - GPUStartTime) of one command buffer per run. "1x" is one product,
//   "3x" three products with distinct operands in one concurrent encoder (Gauss-3M's middle encoder).
//   Variants run round-robin, the order rotating each round, after a 1.5 s warm-up. Each line gives
//   every variant's mean ms (--min: best) and, for each pair x/y, mean t(x) / mean t(y), then the
//   median over rounds of t(x)/t(y) within the round (above 1: y is faster). Small grids can time
//   bimodally (how the GPU spreads them over its cores), so the ratio of means is the routing figure.
//   t64 / t32: threadgroups of the 64x64 and 32x32 tilings. Command buffers are kept to <= 32 GF
//   (macOS's GPU watchdog can kill ~100 ms ones run back to back).
#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
// The MHz column: the mean active GPU clock over each line's rounds, from IOReport, when built with
// -DPFPROBE_CLOCK and an ior.h that provides ior_init/ior_sample/ior_mhz; NAN otherwise.
#ifdef PFPROBE_CLOCK
#include "ior.h"
#else
static int ior_init(void) { return -1; }
static CFDictionaryRef ior_sample(void) { return NULL; }
static double ior_mhz(CFDictionaryRef s0, CFDictionaryRef s1) { (void)s0; (void)s1; return NAN; }
#endif

typedef struct { float hi, lo; } DDF;
enum { PLAIN, SCALED, SYRK };
typedef struct { const char *name, *fn; int bm, bn, tgx, tgy, kind; } Var;
static const Var VARS[] = {
    {"sq",   "dd_dgemm",              64, 64, 16, 16, PLAIN},
    {"sqp",  "dd_dgemm_pf",           64, 64, 16, 16, PLAIN},
    {"nw",   "dd_dgemm_narrow",       32, 32, 16, 16, PLAIN},
    {"nwp",  "dd_dgemm_narrow_pf",    32, 32, 16, 16, PLAIN},
    {"ab",   "dd_dgemm_ab",           64, 64, 16, 16, SCALED},
    {"abp",  "dd_dgemm_ab_pf",        64, 64, 16, 16, SCALED},
    {"nws",  "dd_dgemm_ab_narrow",    32, 32, 16, 16, SCALED},
    {"nwsp", "dd_dgemm_ab_narrow_pf", 32, 32, 16, 16, SCALED},
    {"ts",   "dd_dgemm_ab_ts",       128, 16,  8, 32, SCALED},
    {"tsp",  "dd_dgemm_ab_ts_pf",    128, 16,  8, 32, SCALED},
    {"sy",   "dd_dsyrk",              64, 64, 16, 16, SYRK},
    {"syp",  "dd_dsyrk_pf",           64, 64, 16, 16, SYRK},
};
enum { NV = sizeof VARS / sizeof VARS[0], MAXR = 64 };
static const char *PAIRS[][2] = {
    {"sq", "sqp"}, {"nw", "nwp"}, {"ab", "abp"}, {"nws", "nwsp"}, {"ts", "tsp"}, {"sy", "syp"},
    {"nw", "sqp"}, {"nwp", "sqp"}, {"sq", "nwp"}, {"nws", "abp"}, {"nwsp", "abp"}, {"ts", "abp"}, {"tsp", "nwsp"},
};
enum { NPAIRS = sizeof PAIRS / sizeof PAIRS[0] };

static id<MTLDevice> dev;
static id<MTLCommandQueue> q;
static id<MTLComputePipelineState> pso[NV];
static bool enabled[NV];
static DDF g_alpha = {1, 0}, g_beta = {0, 0};
static bool g_raw, g_min;   // --raw: every round's time per variant; --min: min instead of median ms
static double g_gap_ms;     // --gap MS: idle the GPU this long before each timed run (a host-bound caller's duty cycle)

static int var_index(const char *name) {
    for (int v = 0; v < NV; v++) if (!strcmp(VARS[v].name, name)) return v;
    return -1;
}

static void enc1(id<MTLComputeCommandEncoder> en, int v, id<MTLBuffer> A, id<MTLBuffer> B, id<MTLBuffer> C,
                 uint32_t M, uint32_t N, uint32_t K) {
    const Var *x = &VARS[v];
    [en setComputePipelineState:pso[v]];
    if (x->kind == SYRK) {   // C = A A^T, A is M x K
        uint32_t g = (M + 63) / 64;
        [en setBuffer:A offset:0 atIndex:0]; [en setBuffer:C offset:0 atIndex:1];
        [en setBytes:&M length:4 atIndex:2]; [en setBytes:&K length:4 atIndex:3];
        [en setBytes:&g length:4 atIndex:4]; [en setBytes:&g length:4 atIndex:5];
        [en dispatchThreadgroups:MTLSizeMake(g, g, 1) threadsPerThreadgroup:MTLSizeMake(16, 16, 1)];
        return;
    }
    uint32_t gW = (N + x->bn - 1) / x->bn, gH = (M + x->bm - 1) / x->bm;
    [en setBuffer:A offset:0 atIndex:0]; [en setBuffer:B offset:0 atIndex:1]; [en setBuffer:C offset:0 atIndex:2];
    [en setBytes:&M length:4 atIndex:3]; [en setBytes:&N length:4 atIndex:4]; [en setBytes:&K length:4 atIndex:5];
    if (x->kind == SCALED) {
        [en setBytes:&g_alpha length:8 atIndex:6]; [en setBytes:&g_beta length:8 atIndex:7];
        [en setBytes:&gW length:4 atIndex:8]; [en setBytes:&gH length:4 atIndex:9];
    } else {
        [en setBytes:&gW length:4 atIndex:6]; [en setBytes:&gH length:4 atIndex:7];
    }
    [en dispatchThreadgroups:MTLSizeMake(gW, gH, 1) threadsPerThreadgroup:MTLSizeMake(x->tgx, x->tgy, 1)];
}

static double run(int v, int nprod, __strong id<MTLBuffer> *A, __strong id<MTLBuffer> *B, __strong id<MTLBuffer> *C,
                  uint32_t M, uint32_t N, uint32_t K) {
    id<MTLCommandBuffer> cb = [q commandBuffer];
    id<MTLComputeCommandEncoder> en = nprod > 1 ? [cb computeCommandEncoderWithDispatchType:MTLDispatchTypeConcurrent]
                                                : [cb computeCommandEncoder];
    for (int g = 0; g < nprod; g++) enc1(en, v, A[g], B[g], C[g], M, N, K);
    [en endEncoding]; [cb commit]; [cb waitUntilCompleted];
    if (cb.status == MTLCommandBufferStatusError) {
        printf("!! %s %ux%ux%u %dx: command buffer error: %s\n", VARS[v].name, M, N, K, nprod,
               cb.error.localizedDescription.UTF8String);
        return -1;
    }
    return cb.GPUEndTime - cb.GPUStartTime;
}

// data: 0 moderate (2^[-4,4]), 1 wide-range (2^[-20,20]), 2 wide-range with 1/32 specials
static void fill(DDF *d, size_t n, int kind) {
    static const double SPEC[] = {0.0, -0.0, INFINITY, -INFINITY, NAN, 0x1p100, -0x1p-100, 0x1p-126};
    for (size_t i = 0; i < n; i++) {
        double x = kind == 0 ? ldexp(drand48() * 2 - 1, (int)(drand48() * 9) - 4)
                             : ldexp(drand48() * 2 - 1, (int)(drand48() * 41) - 20);
        if (kind == 2 && (lrand48() & 31) == 0) x = SPEC[lrand48() & 7];
        float h = (float)x;
        d[i] = (DDF){h, isfinite(h) ? (float)(x - (double)h) : 0.0f};
    }
}

static id<MTLBuffer> newbuf(size_t n) { return [dev newBufferWithLength:(n ? n : 1) * 8 options:MTLResourceStorageModeShared]; }

static size_t ndiff(const DDF *a, const DDF *b, size_t n) {
    size_t d = 0;
    for (size_t i = 0; i < n; i++) if (memcmp(&a[i], &b[i], 8)) d++;
    return d;
}

// ---------------------------------------------------------------------------
static int bits_mode(void) {
    // ragged K, non-power-of-two M and N, every routing family's edge cases; ts-shaped ones last
    static const uint32_t S[][3] = {
        {1, 1, 1}, {7, 5, 3}, {33, 31, 17}, {64, 64, 16}, {65, 63, 129}, {100, 77, 513}, {191, 383, 1500},
        {256, 256, 1144}, {480, 480, 513}, {513, 257, 1500}, {640, 640, 1144}, {1000, 777, 1144},
        {704, 576, 1500}, {600, 48, 1500}, {2000, 150, 513}, {4000, 171, 64}, {300, 300, 18433},
    };
    int ns = sizeof S / sizeof S[0];
    const int ref[3] = {var_index("sq"), var_index("ab"), var_index("sy")};
    long total_bad = 0, compared = 0;
    for (int si = 0; si < ns; si++) {
        uint32_t M = S[si][0], N = S[si][1], K = S[si][2];
        for (int kind = 0; kind < 3; kind++) {
            srand48(1000 * si + kind + 7);
            size_t nA = (size_t)M * K, nB = (size_t)K * N, nC = (size_t)M * N, nS = (size_t)M * M;
            id<MTLBuffer> A[3], B[3], C0 = newbuf(nC);
            for (int g = 0; g < 3; g++) {
                A[g] = newbuf(nA); B[g] = newbuf(nB);
                fill(A[g].contents, nA, kind); fill(B[g].contents, nB, kind);
            }
            fill(C0.contents, nC, kind == 2 ? 1 : kind);
            id<MTLBuffer> C[NV][3], ref1x[3] = {nil, nil, nil};
            for (int np = 1; np <= 3; np += 2) {
                for (int v = 0; v < NV; v++) {
                    if (!enabled[v]) continue;
                    size_t nc = VARS[v].kind == SYRK ? nS : nC;
                    for (int g = 0; g < 3; g++) {
                        C[v][g] = newbuf(nc);
                        if (VARS[v].kind == SCALED) memcpy(C[v][g].contents, C0.contents, nC * 8);
                        else memset(C[v][g].contents, 0xff, nc * 8);
                    }
                    if (VARS[v].kind == SCALED) { g_alpha = (DDF){-1.7f, (float)(-1.7 - (double)-1.7f)}; g_beta = (DDF){0.3f, (float)(0.3 - (double)0.3f)}; }
                    if (run(v, np, A, B, C[v], M, N, K) < 0) { total_bad++; continue; }
                }
                g_alpha = (DDF){1, 0}; g_beta = (DDF){0, 0};
                char line[1024]; int len = snprintf(line, sizeof line, "%5u %5u %5u data%d %dx:", M, N, K, kind, np);
                bool all_ok = true;
                for (int v = 0; v < NV; v++) {
                    if (!enabled[v]) continue;
                    int r = ref[VARS[v].kind];
                    if (r == v || r < 0 || !enabled[r]) continue;
                    size_t nc = VARS[v].kind == SYRK ? nS : nC, d = 0;
                    for (int g = 0; g < np; g++) d += ndiff(C[r][g].contents, C[v][g].contents, nc);
                    compared++;
                    if (d) { all_ok = false; total_bad++; }
                    len += snprintf(line + len, sizeof line - len, " %s%s%zu", VARS[v].name, d ? "=DIFF:" : "=", d);
                }
                // concurrency: each reference's product 0 in the 3x run against its 1x result
                for (int k = 0; k < 3; k++) {
                    int r = ref[k];
                    if (r < 0 || !enabled[r]) continue;
                    size_t nc = k == SYRK ? nS : nC;
                    if (np == 1) { ref1x[k] = C[r][0]; continue; }
                    size_t d = ndiff(ref1x[k].contents, C[r][0].contents, nc);
                    compared++;
                    if (d) { all_ok = false; total_bad++; }
                    len += snprintf(line + len, sizeof line - len, " %s(1x)%s%zu", VARS[r].name, d ? "=DIFF:" : "=", d);
                }
                printf("%s%s\n", line, all_ok ? "  ok" : "  MISMATCH");
            }
        }
    }
    printf("bits: %ld comparisons, %ld with differing elements or errors\n", compared, total_bad);
    return total_bad ? 1 : 0;
}

// ---------------------------------------------------------------------------
static int cmpd(const void *a, const void *b) { double x = *(const double *)a, y = *(const double *)b; return x < y ? -1 : x > y; }
static double median(const double *x, int n) {
    double t[MAXR]; memcpy(t, x, n * sizeof(double)); qsort(t, n, sizeof(double), cmpd);
    return n & 1 ? t[n / 2] : 0.5 * (t[n / 2 - 1] + t[n / 2]);
}

static int time_mode(const char *shapes, int R, int npmask) {
    uint32_t shp[512][3]; int ns = 0;
    FILE *f = fopen(shapes, "r"); char ln[256];
    while (f && fgets(ln, sizeof ln, f) && ns < 512)
        if (ln[0] != '#' && sscanf(ln, "%u %u %u", &shp[ns][0], &shp[ns][1], &shp[ns][2]) == 3) ns++;
    if (f) fclose(f);
    if (!ns) { fprintf(stderr, "no shapes in %s\n", shapes); return 2; }
    int order[NV], ne = 0;
    for (int v = 0; v < NV; v++) if (enabled[v]) order[ne++] = v;
    {   // warm-up: >= 1.5 s of continuous GPU work so the clock is up before the first timing
        uint32_t M = 1024, N = 1024, K = 1024;
        id<MTLBuffer> A[3], B[3], C[3];
        for (int g = 0; g < 3; g++) { A[g] = newbuf(M * K); B[g] = newbuf(K * N); C[g] = newbuf(M * N); }
        NSDate *t0 = [NSDate date];
        while (-[t0 timeIntervalSinceNow] < 1.5) run(var_index("sq"), 3, A, B, C, M, N, K);
    }
    printf("%5s %5s %6s %3s %5s %5s", "M", "N", "K", "np", "t64", "t32");
    for (int x = 0; x < ne; x++) printf(" %8s", VARS[order[x]].name);
    printf("  |");
    for (int p = 0; p < NPAIRS; p++) {
        int a = var_index(PAIRS[p][0]), b = var_index(PAIRS[p][1]);
        if (enabled[a] && enabled[b]) printf(" %11s", [[NSString stringWithFormat:@"%s/%s", PAIRS[p][0], PAIRS[p][1]] UTF8String]);
    }
    printf("   MHz\n");
    for (int si = 0; si < ns; si++) {
        uint32_t M = shp[si][0], N = shp[si][1], K = shp[si][2];
        size_t nA = (size_t)M * K, nB = (size_t)K * N, nC = (size_t)M * (N > M ? N : M);
        id<MTLBuffer> A[3], B[3], C[NV][3];
        srand48(11 + si);
        for (int g = 0; g < 3; g++) { A[g] = newbuf(nA); B[g] = newbuf(nB); fill(A[g].contents, nA, 0); fill(B[g].contents, nB, 0); }
        for (int x = 0; x < ne; x++) for (int g = 0; g < 3; g++) C[order[x]][g] = newbuf(nC);
        for (int np = 1; np <= 3; np += 2) {
            if (!(npmask & (np == 1 ? 1 : 2))) continue;
            // macOS's GPU watchdog can kill command buffers of ~100 ms run back to back: stay near 50 ms
            if (np * 2.0 * M * N * K > 32e9) { printf("%5u %5u %6u %2dx skipped (over 32 GF per command buffer)\n", M, N, K, np); continue; }
            static double t[NV][MAXR];
            bool bad = false;
            for (int x = 0; x < ne; x++) if (run(order[x], np, A, B, C[order[x]], M, N, K) < 0) bad = true;
            if (bad) continue;
            CFDictionaryRef s0 = ior_sample();
            for (int r = 0; r < R; r++)
                for (int x = 0; x < ne; x++) {
                    int v = order[(x + r) % ne];
                    if (g_gap_ms > 0) usleep((useconds_t)(g_gap_ms * 1e3));
                    t[v][r] = run(v, np, A, B, C[v], M, N, K);
                }
            CFDictionaryRef s1 = ior_sample();
            double mhz = ior_mhz(s0, s1);
            if (s0) CFRelease(s0);
            if (s1) CFRelease(s1);
            printf("%5u %5u %6u %2dx %5u %5u", M, N, K, np, ((M + 63) / 64) * ((N + 63) / 64), ((M + 31) / 32) * ((N + 31) / 32));
            for (int x = 0; x < ne; x++) {   // mean ms (--min: min)
                int v = order[x]; double b = t[v][0], s = 0;
                for (int r = 0; r < R; r++) { if (t[v][r] < b) b = t[v][r]; s += t[v][r]; }
                printf(" %8.3f", (g_min ? b : s / R) * 1e3);
            }
            printf("  |");
            for (int p = 0; p < NPAIRS; p++) {
                int a = var_index(PAIRS[p][0]), b = var_index(PAIRS[p][1]);
                if (!enabled[a] || !enabled[b]) continue;
                double rat[MAXR], sa = 0, sb = 0;
                for (int r = 0; r < R; r++) { rat[r] = t[a][r] / t[b][r]; sa += t[a][r]; sb += t[b][r]; }
                // timings can be bimodal (how the GPU spreads a small grid over its cores): the mean
                // is the expected cost, so the ratio of means is the routing figure; median of paired ratios after it
                printf(" %5.3f/%5.3f", sa / sb, median(rat, R));
            }
            printf("  %5.0f\n", mhz);
            if (g_raw)
                for (int x = 0; x < ne; x++) {
                    int v = order[x];
                    printf("    raw %-5s", VARS[v].name);
                    for (int r = 0; r < R; r++) printf(" %.3f", t[v][r] * 1e3);
                    printf("\n");
                }
        }
    }
    return 0;
}

int main(int argc, char **argv) { @autoreleasepool {
    setvbuf(stdout, NULL, _IOLBF, 0);
    if (argc < 3) { fprintf(stderr, "usage: %s apple_bottom.m bits | time <shapes> [--rounds R] [--np 1|3|13] [--only v,...]\n", argv[0]); return 2; }
    const char *only = NULL; int R = 9, npmask = 1 | 2;
    for (int i = 3; i < argc; i++) {
        if (!strcmp(argv[i], "--only") && i + 1 < argc) only = argv[++i];
        else if (!strcmp(argv[i], "--rounds") && i + 1 < argc) R = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--raw")) g_raw = true;
        else if (!strcmp(argv[i], "--min")) g_min = true;
        else if (!strcmp(argv[i], "--gap") && i + 1 < argc) g_gap_ms = atof(argv[++i]);
        else if (!strcmp(argv[i], "--np") && i + 1 < argc) { int x = atoi(argv[++i]); npmask = x == 13 ? 3 : x == 3 ? 2 : 1; }
    }
    if (R < 1) R = 1;
    if (R > MAXR) R = MAXR;
    NSString *src = [NSString stringWithContentsOfFile:@(argv[1]) encoding:NSUTF8StringEncoding error:nil];
    NSRange a = [src rangeOfString:@"kShaderSource = @(R\"("];
    if (a.location == NSNotFound) { fprintf(stderr, "no shader source\n"); return 2; }
    NSRange b = [src rangeOfString:@")\");" options:0 range:NSMakeRange(a.location, src.length - a.location)];
    NSUInteger s0 = a.location + a.length;
    NSString *msl = [src substringWithRange:NSMakeRange(s0, b.location - s0)];
    dev = MTLCreateSystemDefaultDevice(); q = [dev newCommandQueue];
    ior_init();
    MTLCompileOptions *o = [MTLCompileOptions new];
    o.mathMode = MTLMathModeSafe;
    NSError *err = nil;
    NSDate *c0 = [NSDate date];
    id<MTLLibrary> lib = [dev newLibraryWithSource:msl options:o error:&err];
    if (!lib) { NSLog(@"%@", err); return 1; }
    printf("# %s, library compiled in %.0f ms\n", dev.name.UTF8String, -[c0 timeIntervalSinceNow] * 1e3);
    for (int v = 0; v < NV; v++) {
        bool want = true;
        if (only) {   // exact names, comma separated
            want = false;
            NSArray *names = [@(only) componentsSeparatedByString:@","];
            for (NSString *s in names) if ([s isEqualToString:@(VARS[v].name)]) want = true;
        }
        id<MTLFunction> fn = [lib newFunctionWithName:@(VARS[v].fn)];
        if (!fn) { printf("# %-22s not in the library\n", VARS[v].fn); continue; }
        NSDate *p0 = [NSDate date];
        pso[v] = [dev newComputePipelineStateWithFunction:fn error:&err];
        if (!pso[v]) { NSLog(@"%s: %@", VARS[v].fn, err); return 1; }
        enabled[v] = want;
        printf("# %-22s maxThreads %4lu  staticTG %5lu  pipeline %.1f ms%s\n", VARS[v].fn,
               (unsigned long)pso[v].maxTotalThreadsPerThreadgroup, (unsigned long)pso[v].staticThreadgroupMemoryLength,
               -[p0 timeIntervalSinceNow] * 1e3, want ? "" : "  (not run)");
    }
    if (!strcmp(argv[2], "bits")) return bits_mode();
    if (!strcmp(argv[2], "time") && argc > 3) return time_mode(argv[3], R, npmask);
    fprintf(stderr, "unknown mode\n");
    return 2;
}}
