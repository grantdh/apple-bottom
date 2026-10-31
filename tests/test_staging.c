// =============================================================================
// apple-bottom staged path: bit for bit the host path
// =============================================================================
//
// A GPU-routed dgemm_/zgemm_ takes the staged path (src/gpu_stage.h): its
// operands are copied into reused GPU buffers, converted to DD on the GPU,
// and C gets the GPU epilogue's FP64. This test makes every case of the
// table below in four child processes, all with AB_MODE=gpu and
// AB_MIN_GPU_DIM=0:
//   host    AB_HOST_PACK=1: the host path (host pack, conversion, epilogue
//           and the exponent-range guard)
//   staged  the default
//   nopool  AB_STAGING_MB=0: the staged path, keeping no buffers between calls
//   registered  A, B and C registered (ab_register_host_memory), so the GPU
//          reads and writes them in place; in every third case the complex
//          arrays start 8 bytes past a 16-byte boundary and must be staged
// For each case A and B must be unchanged, and the two calls a child makes
// in a row (the second on the buffers the first left) must give the same C.
// The staged result must stand (ab_internal_last_staged()) in the three
// staged children exactly where expected_staged() says (scaled lines and
// 4M included), with its operands in place in the registered child unless
// misaligned; every other call is made on the host path. The GPU must
// compute C in every child and the whole C array (padding included) hash the
// same in all four, except where the host path's guard sends the call to
// Accelerate: there every child must, and their bits are not compared. The
// registered child also checks ab_register_host_memory()'s refusals.
//
// The native ZGEMMs take their Gauss-3M or 4M temporaries from the same
// pool, so after the run it must hold buffers in the host child (whose
// ZGEMMs go through ab_internal_zgemm) and the staged one, and none with
// AB_STAGING_MB=0. A fifth child (--child-native) checks those ZGEMMs on the
// pool: ab_zgemm, ab_zgemm_async, ab_batch_zgemm and ab_zgemm_batched,
// repeated after the pool served larger and smaller shapes and from threads
// calling at once, must all give the bits of the first ab_zgemm, and on
// nearly real data (4M) those of an explicit 4M call, also in a batch that
// switches between 3M and 4M; the pool must keep the temporaries between
// calls. ab_shutdown() must empty it, and after shapes that grow it must
// hold what the largest alone leaves.
//
// Usage: test_staging                  the test
//        test_staging --child          one child: a line per case on stdout
//        test_staging --child-native   the native ZGEMMs: "native ok" or what failed
// =============================================================================

#include "apple_bottom.h"
#include "gpu_stage.h"
#include "test_blas_check.h"
#include "test_spawn.h"
#include <complex.h>
#include <inttypes.h>
#include <math.h>
#include <pthread.h>
#include <stdio.h>
#include <errno.h>
#include <stdlib.h>
#include <string.h>
#include <sys/wait.h>

// =============================================================================
// Cases
// =============================================================================

typedef enum { S_A1B0, S_A1B1, S_A1BX, S_AXB0, S_AXBX, S_A2B0, S_AM1B1, S_ABIG, S_COUNT } ScalarSet;

static const struct { const char *tag; double complex alpha, beta; } SCALARS[S_COUNT] = {
    [S_A1B0]  = {"a1b0",  CMPLX(1.0, 0.0),  CMPLX(0.0, 0.0)},
    [S_A1B1]  = {"a1b1",  CMPLX(1.0, 0.0),  CMPLX(1.0, 0.0)},
    [S_A1BX]  = {"a1bx",  CMPLX(1.0, 0.0),  CMPLX(0.3, -0.45)},
    [S_AXB0]  = {"axb0",  CMPLX(-1.7, 0.6), CMPLX(0.0, 0.0)},
    [S_AXBX]  = {"axbx",  CMPLX(-1.7, 0.6), CMPLX(0.3, -0.45)},
    [S_A2B0]  = {"a2b0",  CMPLX(2.0, 0.0),  CMPLX(0.0, 0.0)},
    [S_AM1B1] = {"am1b1", CMPLX(-1.0, 0.0), CMPLX(1.0, 0.0)},
    [S_ABIG]  = {"abig",  CMPLX(0x1p25, 0.0), CMPLX(1.0, 0.0)},    // a DGEMM alpha the guard scales
};

typedef enum {
    D_UNIFORM,      // A, B, C uniform in [-1, 1)
    D_MIXED,        // A, B, C uniform times 2^e, e in [-12, 6]
    D_WIDE,         // A times 2^e, e in [-140, 100] (FP32 subnormal low words); B, C uniform
    D_BITS,         // A random bits in the guard's window (a complex one's parts in one
                    // binade), B = 1, C random bits (K = 1 cases)
    D_SIGNED_ZERO,  // +-0 rows of op(A), columns of op(B), scattered entries; C all +-0
    D_TINY_C,       // A, B as D_SIGNED_ZERO; C tiny, down to FP64 subnormals
    D_CANCEL,       // C on entry = -op(A) op(B) as summed on the host: beta = 1 cancels
    D_NAN_A,        // A holds a NaN and an Inf
    D_NAN_C,        // C on entry cycles NaN, +Inf, -Inf
    D_HUGE,         // A, B near 2^100: the DD products overflow FP32
    D_HUGE_C,       // C on entry near 2^140: beyond FP32, so DGEMM's DD copy of C overflows
    D_STAGE2,       // rows of op(A) from 1 down to 2^-90: the guard certifies them in stage 2
    D_IMBAL,        // A, B with imaginary parts 2^-6 of the real: the host path runs 4M
    D_COUNT
} DataKind;

static const char *const DATA_TAG[D_COUNT] = {
    [D_UNIFORM] = "uniform", [D_MIXED] = "mixed", [D_WIDE] = "wide", [D_BITS] = "bits",
    [D_SIGNED_ZERO] = "signedzero", [D_TINY_C] = "tinyC", [D_CANCEL] = "cancel",
    [D_NAN_A] = "nanA", [D_NAN_C] = "nanC", [D_HUGE] = "huge", [D_HUGE_C] = "hugeC",
    [D_STAGE2] = "stage2", [D_IMBAL] = "imbal",
};

// What each kind of data is for: the staged result must stand in at least
// one of a routine's cases of that kind (scaled lines and 4M included), or
// (false) in none of them, which go to Accelerate. The test fails if the
// guard's rules ever leave a kind short of its purpose.
static const bool STAGES[D_COUNT] = {
    [D_UNIFORM] = true, [D_MIXED] = true, [D_WIDE] = true, [D_BITS] = true,
    [D_SIGNED_ZERO] = true, [D_TINY_C] = true, [D_CANCEL] = true, [D_NAN_C] = true,
    [D_HUGE] = true, [D_HUGE_C] = true, [D_STAGE2] = true, [D_IMBAL] = true,
};

typedef struct {
    char routine;            // 'd' dgemm_, 'z' zgemm_
    char ta, tb;
    int M, N, K, pad;        // pad: ld = stored rows + pad for A, B and C
    ScalarSet sc;
    DataKind data;
} Case;

#define MAX_CASES 256
static Case g_cases[MAX_CASES];
static int g_ncases;

static void add(char routine, char ta, char tb, int M, int N, int K, int pad, ScalarSet sc,
                DataKind data) {
    if (g_ncases < MAX_CASES)
        g_cases[g_ncases++] = (Case){routine, ta, tb, M, N, K, pad, sc, data};
}

static void make_cases(void) {
    static const char TRANS[] = "NTC";
    // Every TRANS pair with the GPU epilogue's scalars, and with others
    for (int a = 0; a < 3; a++)
        for (int b = 0; b < 3; b++) {
            add('z', TRANS[a], TRANS[b], 75, 53, 47, 3, S_A1B0, D_UNIFORM);
            add('z', TRANS[a], TRANS[b], 75, 53, 47, 3, S_A1B1, D_UNIFORM);
            add('z', TRANS[a], TRANS[b], 75, 53, 47, 3, (a + b) % 2 ? S_AXBX : S_A1BX, D_UNIFORM);
            add('d', TRANS[a], TRANS[b], 75, 53, 47, 3, S_A1B0, D_UNIFORM);
            add('d', TRANS[a], TRANS[b], 75, 53, 47, 3, (a + b) % 2 ? S_AXBX : S_A1B1, D_UNIFORM);
        }
    // Shapes: tall-skinny kernel, several tiles, M or N = 1, no padding,
    // padding wide enough that operands are copied column by column
    for (int r = 0; r < 2; r++) {
        char rt = r ? 'z' : 'd';
        add(rt, 'N', 'N', 600, 48, 40, 3, S_A1B1, D_UNIFORM);
        add(rt, 'N', 'T', 513, 257, 129, 2, S_A1B0, D_UNIFORM);
        add(rt, 'T', 'N', 1, 64, 64, 0, S_A1B1, D_UNIFORM);
        add(rt, 'N', 'N', 64, 1, 64, 0, S_A1B0, D_UNIFORM);
        add(rt, 'C', 'N', 33, 31, 200, 0, S_A1B0, D_UNIFORM);
        add(rt, 'N', 'C', 70, 90, 50, 200, S_A1B1, D_UNIFORM);
        add(rt, 'C', 'C', 70, 90, 50, 200, S_AXBX, D_UNIFORM);
    }
    // Data
    for (int r = 0; r < 2; r++) {
        char rt = r ? 'z' : 'd';
        for (int s = 0; s < 2; s++) {
            ScalarSet sc = s ? S_A1B1 : S_A1B0;
            add(rt, 'T', 'N', 96, 80, 72, 1, sc, D_MIXED);
            add(rt, 'N', 'N', 96, 80, 72, 3, sc, D_WIDE);
            add(rt, 'N', 'N', 300, 200, 1, 2, sc, D_BITS);
            add(rt, 'T', 'T', 200, 300, 1, 0, sc, D_BITS);
            add(rt, 'N', 'C', 75, 53, 47, 3, sc, D_SIGNED_ZERO);
            add(rt, 'C', 'N', 75, 53, 47, 3, sc, D_TINY_C);
            add(rt, 'N', 'N', 96, 80, 72, 3, sc, D_NAN_A);
            add(rt, 'C', 'N', 75, 53, 47, 3, sc, D_NAN_C);
            add(rt, 'N', 'T', 64, 64, 64, 0, sc, D_HUGE);
            add(rt, 'T', 'N', 64, 64, 64, 1, sc, D_HUGE_C);
            add(rt, 'N', 'N', 96, 80, 72, 3, sc, D_STAGE2);
        }
        add(rt, 'N', 'T', 96, 80, 72, 3, S_ABIG, D_UNIFORM);
        add(rt, 'C', 'N', 96, 80, 72, 3, S_A1B1, D_CANCEL);
        add(rt, 'N', 'N', 96, 80, 72, 3, S_A1B1, D_CANCEL);
        add(rt, 'T', 'N', 75, 53, 47, 3, S_AM1B1, D_SIGNED_ZERO);
        add(rt, 'N', 'N', 75, 53, 47, 3, S_AXBX, D_NAN_C);
        add(rt, 'N', 'N', 75, 53, 47, 3, S_AXB0, D_NAN_C);
        add(rt, 'N', 'T', 300, 200, 1, 0, S_AXBX, D_BITS);
    }
    // Complex parts imbalanced enough for 4M
    add('z', 'T', 'N', 96, 80, 72, 3, S_A1B0, D_IMBAL);
    add('z', 'N', 'C', 96, 80, 72, 3, S_A1B1, D_IMBAL);
    add('z', 'C', 'T', 96, 80, 72, 3, S_AXBX, D_IMBAL);
    // Quantum ESPRESSO shapes (as test_qe_golden.c), large enough that the
    // copies run in pieces
    add('z', 'C', 'N', 160, 160, 9140, 7, S_A1B0, D_UNIFORM);
    add('z', 'N', 'N', 9140, 160, 160, 7, S_A1B1, D_UNIFORM);
    add('z', 'C', 'N', 64, 64, 20000, 0, S_A1B0, D_UNIFORM);
    add('z', 'N', 'N', 20000, 64, 64, 9, S_A1B1, D_UNIFORM);
    add('d', 'T', 'N', 36, 64, 73062, 0, S_A2B0, D_UNIFORM);
    add('d', 'N', 'T', 4000, 171, 64, 0, S_A1B0, D_UNIFORM);
    add('d', 'N', 'N', 40000, 64, 64, 5, S_A1B1, D_UNIFORM);
}

// Whether the GPU reads C on the staged path: DGEMM beta != 0, ZGEMM
// alpha = 1 and beta = 1 (gpu_stage.h).
static bool case_c_in(const Case *c) {
    double complex al = SCALARS[c->sc].alpha, be = SCALARS[c->sc].beta;
    return c->routine == 'z' ? al == 1.0 && be == 1.0 : be != 0.0;
}

static void case_name(const Case *c, char *buf, size_t n) {
    snprintf(buf, n, "%cgemm_/%c%c/%dx%dx%d/ld+%d/%s/%s", c->routine, c->ta, c->tb, c->M, c->N,
             c->K, c->pad, SCALARS[c->sc].tag, DATA_TAG[c->data]);
}

// =============================================================================
// Data (deterministic: the generator is seeded with the case name)
// =============================================================================

static uint64_t sm_state;
static uint64_t next64(void) {
    uint64_t s = sm_state;
    sm_state += 0x9E3779B97F4A7C15ULL;
    return splitmix(s);
}

static double uniform(void) { return (double)(next64() >> 11) * 0x1.0p-52 - 1.0; }
static double signed_zero(void) { return (next64() & 1) ? -0.0 : 0.0; }

// A double with random sign and mantissa and an exponent in [lo, hi]
// (biased exponents 1 .. 2046, so never Inf or NaN; lo < -1022 gives
// subnormals).
static double random_bits(int lo, int hi) {
    uint64_t r = next64();
    int e = lo + (int)(next64() % (uint64_t)(hi - lo + 1));
    uint64_t bits = r & 0x800FFFFFFFFFFFFFULL;
    if (e >= -1022) {
        bits |= (uint64_t)(e + 1023) << 52;
    } else {
        bits &= 0x8000000000000000ULL | ((1ULL << (52 - (-1022 - e))) - 1);   // subnormal
    }
    double x;
    memcpy(&x, &bits, sizeof x);
    return x;
}

typedef enum { V_UNIFORM, V_MIXED, V_WIDE, V_BITS_A, V_BITS_C, V_ONE, V_SIGNED_ZERO, V_TINY,
               V_NAN_INF, V_HUGE, V_HUGE_C } ValueKind;

static double value(ValueKind v, size_t idx) {
    static const double NAN_INF[] = {NAN, INFINITY, -INFINITY};
    switch (v) {
    case V_MIXED: {
        double x = uniform();
        return ldexp(x, (int)(next64() % 19) - 12);
    }
    case V_WIDE: {
        double x = uniform();
        return ldexp(x, (int)(next64() % 241) - 140);
    }
    case V_BITS_A:      return random_bits(-20, 19);     // inside the guard's window
    case V_BITS_C:      return idx % 7 ? random_bits(-160, 120) : random_bits(-1074, -1000);
    case V_ONE:         return 1.0;
    case V_SIGNED_ZERO: return signed_zero();
    case V_TINY:        return idx % 3 ? random_bits(-1074, -1023) : random_bits(-1060, -900);
    case V_NAN_INF:     return NAN_INF[idx % 3];
    case V_HUGE:        return ldexp(1.0 + uniform() / 4, 100);
    case V_HUGE_C:      return ldexp(uniform(), 140);
    default:            return uniform();
    }
}

static size_t op_at(char t, Layout l, int w, int i, int j) { return op_index(t, l, i, j) * w; }

// NaN everywhere, then the rows x cols block from v, column by column.
static void fill(double *X, Layout l, int w, ValueKind v) {
    for (size_t i = 0; i < nelem(l) * w; i++) X[i] = NAN;
    size_t idx = 0;
    for (int j = 0; j < l.cols; j++)
        for (int i = 0; i < l.rows; i++)
            for (int p = 0; p < w; p++) X[((size_t)j * l.ld + i) * w + p] = value(v, idx++);
}

typedef struct {
    const Case *c;
    Layout la, lb, lc;
    size_t na, nb, nc;                  // doubles of A, B, C
    double *A, *B, *C0, *C, *A0, *B0;
    double *mem[3];                     // the allocations of A, B and C
} Problem;

// shift: A, B and C start that many doubles into their allocations.
static void make_problem(Problem *P, const Case *c, const char *name, int shift) {
    const int M = c->M, N = c->N, K = c->K, w = c->routine == 'z' ? 2 : 1;
    P->c = c;
    P->la = stored(c->ta, M, K, c->pad);
    P->lb = stored(c->tb, K, N, c->pad);
    P->lc = (Layout){M, N, M + c->pad};
    P->na = nelem(P->la) * w;
    P->nb = nelem(P->lb) * w;
    P->nc = nelem(P->lc) * w;
    const size_t n[3] = {P->na, P->nb, P->nc};
    for (int i = 0; i < 3; i++)
        if (posix_memalign((void **)&P->mem[i], 64, (n[i] + shift) * sizeof(double))) abort();
    P->A = P->mem[0] + shift;
    P->B = P->mem[1] + shift;
    P->C = P->mem[2] + shift;
    P->A0 = malloc(P->na * sizeof(double));
    P->B0 = malloc(P->nb * sizeof(double));
    P->C0 = malloc(P->nc * sizeof(double));

    sm_state = fnv1a(name, strlen(name), FNV_OFFSET);
    ValueKind va = V_UNIFORM, vb = V_UNIFORM, vc = V_UNIFORM;
    switch (c->data) {
    case D_MIXED:       va = vb = vc = V_MIXED; break;
    case D_WIDE:        va = V_WIDE; break;
    case D_BITS:        va = V_BITS_A; vb = V_ONE; vc = V_BITS_C; break;
    case D_SIGNED_ZERO: vc = V_SIGNED_ZERO; break;
    case D_TINY_C:      vc = V_TINY; break;
    case D_NAN_C:       vc = V_NAN_INF; break;
    case D_HUGE:        va = vb = V_HUGE; break;
    case D_HUGE_C:      vc = V_HUGE_C; break;
    default: break;
    }
    fill(P->A, P->la, w, va);
    fill(P->B, P->lb, w, vb);
    fill(P->C0, P->lc, w, vc);
    if (c->data == D_BITS && w == 2) {      // B = 1 + 0i
        for (int j = 0; j < P->lb.cols; j++)
            for (int i = 0; i < P->lb.rows; i++) P->B[((size_t)j * P->lb.ld + i) * 2 + 1] = 0.0;
        // |Im a| in |Re a|'s binade: rho = 1 + |Re a|/|Im a| < 3 keeps Gauss-3M
        for (size_t i = 0; i < P->na; i += 2) {
            int e;
            P->A[i + 1] = ldexp(frexp(P->A[i + 1], &e), ilogb(P->A[i]) + 1);
        }
    }

    if (c->data == D_SIGNED_ZERO || c->data == D_TINY_C) {
        for (int i = 0; i < M; i++)
            for (int k = 0; k < K; k++)
                if (i < 6 || (next64() & 7) == 0)
                    for (int p = 0; p < w; p++) P->A[op_at(c->ta, P->la, w, i, k) + p] = signed_zero();
        for (int k = 0; k < K; k++)
            for (int j = 0; j < N; j++)
                if (j < 4 || (next64() & 7) == 0)
                    for (int p = 0; p < w; p++) P->B[op_at(c->tb, P->lb, w, k, j) + p] = signed_zero();
    }
    if (c->data == D_STAGE2)                // a tiny element in every eighth row
        for (int i = 0; i < M; i += 8)
            for (int p = 0; p < w; p++) P->A[op_at(c->ta, P->la, w, i, K / 2) + p] = 0x1p-90;
    if (c->data == D_IMBAL) {               // |Im| ~ 2^-6 |Re|: Gauss-3M's rho ~ 32
        for (size_t i = 1; i < P->na; i += 2) P->A[i] *= 0x1p-6;
        for (size_t i = 1; i < P->nb; i += 2) P->B[i] *= 0x1p-6;
    }
    if (c->data == D_NAN_A) {
        P->A[op_at(c->ta, P->la, w, M / 2, K / 3)] = NAN;
        P->A[op_at(c->ta, P->la, w, M / 3, K / 2) + w - 1] = INFINITY;
    }
    if (c->data == D_CANCEL)               // C0 = -op(A) op(B), summed in order on the host
        for (int j = 0; j < N; j++)
            for (int i = 0; i < M; i++) {
                double complex s = 0.0;
                for (int k = 0; k < K; k++) {
                    const double *a = &P->A[op_at(c->ta, P->la, w, i, k)];
                    const double *b = &P->B[op_at(c->tb, P->lb, w, k, j)];
                    if (w == 1) {
                        s += a[0] * b[0];
                    } else {
                        double complex x = CMPLX(a[0], c->ta == 'C' ? -a[1] : a[1]);
                        double complex y = CMPLX(b[0], c->tb == 'C' ? -b[1] : b[1]);
                        s += x * y;
                    }
                }
                double *d = &P->C0[((size_t)j * P->lc.ld + i) * w];
                d[0] = -creal(s);
                if (w == 2) d[1] = -cimag(s);
            }
    memcpy(P->A0, P->A, P->na * sizeof(double));
    memcpy(P->B0, P->B, P->nb * sizeof(double));
}

static void free_problem(Problem *P) {
    for (int i = 0; i < 3; i++) free(P->mem[i]);
    free(P->A0); free(P->B0); free(P->C0);
}

static void call_gemm(Problem *P) {
    const Case *c = P->c;
    const int M = c->M, N = c->N, K = c->K;
    const int lda = P->la.ld, ldb = P->lb.ld, ldc = P->lc.ld;
    memcpy(P->C, P->C0, P->nc * sizeof(double));
    if (c->routine == 'd') {
        const double alpha = creal(SCALARS[c->sc].alpha), beta = creal(SCALARS[c->sc].beta);
        fortran_dgemm(&c->ta, &c->tb, &M, &N, &K, &alpha, P->A, &lda, P->B, &ldb, &beta,
                      P->C, &ldc);
    } else {
        const double complex alpha = SCALARS[c->sc].alpha, beta = SCALARS[c->sc].beta;
        fortran_zgemm(&c->ta, &c->tb, &M, &N, &K, &alpha, (double complex *)P->A, &lda,
                      (double complex *)P->B, &ldb, &beta, (double complex *)P->C, &ldc);
    }
}

// Whether the staged path keeps its result for case P: wherever the host
// path computes it on the GPU, scaled or not, a ZGEMM by Gauss-3M or 4M, but
// a DGEMM only with alpha and beta in their window. There the
// exponent-range guard (ab_internal.h) certifies the product of the lines
// as scaled (and a DGEMM's C, when read, as uploaded: scaled by its row's
// and column's scales). Then no Inf or NaN may reach the GPU epilogue: a
// ZGEMM's C with beta = 1 must be finite. The host path makes every other
// call: on the GPU with such scalars or C's Inf or NaN, or on Accelerate.
static bool expected_staged(const Problem *P) {
    const Case *c = P->c;
    bool z = c->routine == 'z';
    int w = z ? 2 : 1;
    double complex al = SCALARS[c->sc].alpha, be = SCALARS[c->sc].beta;
    if (!z && (ab_scalar_exp(creal(al), AB_SCALAR_WINDOW) || ab_scalar_exp(creal(be), AB_SCALAR_WINDOW)))
        return false;
    ab_op_view_t va = {P->A, P->la.ld, c->M, c->K, !is_notrans(c->ta), z};
    ab_op_view_t vb = {P->B, P->lb.ld, c->K, c->N, !is_notrans(c->tb), z};
    ab_line_stats_t sa = {0}, sb = {0};
    bool ok = ab_line_stats_init(&sa, c->M, z) && ab_line_stats_init(&sb, c->N, z);
    if (ok) {
        ab_line_stats_scan(&va, true, &sa);
        ab_line_stats_scan(&vb, false, &sb);
        ab_line_scales(&sa);
        ab_line_scales(&sb);
        ok = ab_range_guard(c->K, &sa, &sb, z ? 0 : ilogb(creal(al)), &va, &vb);
    }
    ab_range_t r = AB_RANGE_EMPTY;
    for (int j = 0; j < c->N; j++)
        for (int i = 0; i < c->M; i++)
            for (int p = 0; p < w; p++) {
                double v = P->C0[((size_t)j * P->lc.ld + i) * w + p];
                AB_RANGE_ADD(ok && !z ? ab_scale2(v, -(sa.e[i] + sb.e[j])) : v, r.max, r.min);
            }
    if (ok && case_c_in(c))
        ok = z ? !ab_range_nonfinite(r)
               : ab_range_guard_c(&sa, &sb, r, ilogb(creal(be)), P->C0, P->lc.ld, 0);
    ab_line_stats_free(&sa);
    ab_line_stats_free(&sb);
    return ok;
}

// =============================================================================
// Child: each case twice, one line per case
// =============================================================================

// ab_register_host_memory()'s refusals and limits; false after printing one
// that does not hold.
typedef struct { const char *what; ABStatus got, want; } ApiCheck;

static bool check_registration_api(void) {
    static double x[4096];
    ApiCheck r[8];
    int n = 0;
    r[n++] = (ApiCheck){"NULL", ab_register_host_memory(NULL, 8), AB_ERROR_INVALID_ARG};
    r[n++] = (ApiCheck){"0 bytes", ab_register_host_memory(x, 0), AB_ERROR_INVALID_ARG};
    r[n++] = (ApiCheck){"register", ab_register_host_memory(x + 1024, 8192), AB_OK};
    r[n++] = (ApiCheck){"overlap", ab_register_host_memory(x + 2000, 64), AB_ERROR_INVALID_ARG};
    r[n++] = (ApiCheck){"unknown ptr", ab_unregister_host_memory(x + 1025), AB_ERROR_INVALID_ARG};
    r[n++] = (ApiCheck){"unregister", ab_unregister_host_memory(x + 1024), AB_OK};
    int ok = 0;
    for (int i = 0; i < AB_MAX_HOST_REGISTRATIONS; i++)
        ok += ab_register_host_memory(x + 8 * i, 64) == AB_OK;
    r[n++] = (ApiCheck){"one too many", ab_register_host_memory(x + 4000, 64), AB_ERROR_TOO_LARGE};
    for (int i = 0; i < AB_MAX_HOST_REGISTRATIONS; i++)
        ok += ab_unregister_host_memory(x + 8 * i) == AB_OK;
    bool pass = ok == 2 * AB_MAX_HOST_REGISTRATIONS;
    if (!pass) printf("api: %d of %d registrations and unregistrations OK\n", ok,
                      2 * AB_MAX_HOST_REGISTRATIONS);
    for (int i = 0; i < n; i++)
        if (r[i].got != r[i].want) {
            printf("api: %s gave %s, expected %s\n", r[i].what, ab_status_string(r[i].got),
                   ab_status_string(r[i].want));
            pass = false;
        }
    return pass;
}

static int run_child(bool reg) {
    make_cases();
    if (reg && !check_registration_api()) return 1;
    for (int i = 0; i < g_ncases; i++) {
        const Case *c = &g_cases[i];
        char name[96];
        case_name(c, name, sizeof name);
        Problem P;
        int shift = reg && i % 3 == 2;                  // every third: 8 bytes off 16
        make_problem(&P, c, name, shift);
        const size_t n[3] = {P.na, P.nb, P.nc};
        for (int k = 0; reg && k < 3; k++)
            if (ab_register_host_memory(P.mem[k], (n[k] + shift) * sizeof(double)) != AB_OK) {
                printf("case %s: cannot register operand %d\n", name, k);
                return 1;
            }
        uint64_t hash[2], g0 = gpu_gemm_count();
        int gpu = 0, staged = 0, in_place = 0;
        for (int run = 0; run < 2; run++) {
            call_gemm(&P);
            hash[run] = fnv1a(P.C, P.nc * sizeof(double), FNV_OFFSET);
            gpu += !strcmp(ab_get_last_dispatch_path(), "gpu");
            int ip;
            if (ab_internal_last_staged(&ip)) {
                staged++;
                in_place += ip;
            }
        }
        for (int k = 0; reg && k < 3; k++) ab_unregister_host_memory(P.mem[k]);
        bool ab_same = !memcmp(P.A, P.A0, P.na * sizeof(double)) &&
                       !memcmp(P.B, P.B0, P.nb * sizeof(double));
        printf("case %s hash=%016" PRIx64 " repeat=%d staged=%d gpu=%d gemms=%llu ab=%d "
               "in_place=%d expect=%d\n", name, hash[0], hash[0] == hash[1], staged, gpu,
               (unsigned long long)(gpu_gemm_count() - g0), ab_same, in_place,
               expected_staged(&P));
        free_problem(&P);
    }
    printf("pool_bytes=%zu\n", ab_internal_stage_pool_bytes());
    return 0;
}

// =============================================================================
// Native ZGEMMs on the pool (--child-native)
// =============================================================================

typedef struct {
    int M, N, K;
    double *in[4];              // Ar, Ai (M x K), Br, Bi (K x N), row-major
    uint64_t hash;              // of Cr then Ci, by the reference call
} NativeShape;

// The entry points, then ab_internal_zgemm by a given method.
typedef enum { VIA_ZGEMM, VIA_ASYNC, VIA_BATCH, VIA_3M, VIA_4M } NativeVia;

// sh's inputs from seed, uniform in [-1, 1), the imaginary parts times im.
static void native_make(NativeShape *sh, int M, int N, int K, uint64_t seed, double im) {
    sh->M = M; sh->N = N; sh->K = K;
    const size_t n[4] = {(size_t)M * K, (size_t)M * K, (size_t)K * N, (size_t)K * N};
    sm_state = seed;
    for (int i = 0; i < 4; i++) {
        sh->in[i] = malloc(n[i] * sizeof(double));
        for (size_t j = 0; j < n[i]; j++) sh->in[i][j] = (i % 2 ? im : 1.0) * uniform();
    }
}

// Ar, Ai, Br, Bi holding sh's inputs, then Cr, Ci.
static void native_operands(const NativeShape *sh, ABMatrix m[6]) {
    const int rows[6] = {sh->M, sh->M, sh->K, sh->K, sh->M, sh->M};
    const int cols[6] = {sh->K, sh->K, sh->N, sh->N, sh->N, sh->N};
    for (int i = 0; i < 6; i++) {
        m[i] = ab_matrix_create(rows[i], cols[i]);
        if (m[i] && i < 4) ab_matrix_upload(m[i], sh->in[i], true);
    }
}

// The hash of Cr then Ci if ok, else 0; destroys m.
static uint64_t native_result(const NativeShape *sh, ABMatrix m[6], bool ok) {
    uint64_t h = 0;
    if (ok) {
        size_t mn = (size_t)sh->M * sh->N;
        double *c = malloc(2 * mn * sizeof *c);
        ab_matrix_download(m[4], c, true);
        ab_matrix_download(m[5], c + mn, true);
        h = fnv1a(c, 2 * mn * sizeof *c, FNV_OFFSET);
        free(c);
    }
    for (int i = 0; i < 6; i++) ab_matrix_destroy(m[i]);
    return h;
}

// The hash of Cr then Ci after sh's product through via; 0 if a call failed.
static uint64_t native_hash(const NativeShape *sh, NativeVia via) {
    ABMatrix m[6];
    native_operands(sh, m);
    ABStatus st;
    if (via == VIA_ZGEMM) {
        st = ab_zgemm(m[0], m[1], m[2], m[3], m[4], m[5]);
    } else if (via == VIA_ASYNC) {
        ABFuture f = ab_zgemm_async(m[0], m[1], m[2], m[3], m[4], m[5]);
        st = f ? ab_future_wait(f) : AB_ERROR_KERNEL_FAILED;
        ab_future_destroy(f);
    } else if (via == VIA_BATCH) {
        ABBatch b = ab_batch_create();
        st = ab_batch_zgemm(b, m[0], m[1], m[2], m[3], m[4], m[5]);
        if (st == AB_OK) st = ab_batch_commit(b);
        if (st == AB_OK) st = ab_batch_wait(b);
        ab_batch_destroy(b);
    } else {
        st = ab_internal_zgemm(AB_NO_TRANS, AB_NO_TRANS, m[0], m[1], m[2], m[3], m[4], m[5],
                               via == VIA_4M ? AB_ZGEMM_4M : AB_ZGEMM_3M);
    }
    return native_result(sh, m, st == AB_OK);
}

// Whether ab_zgemm_batched over shapes sh[0..n), n <= 8, gives each product
// its reference bits.
static bool native_batched(NativeShape *const *sh, int n) {
    ABMatrix m[8][6], role[6][8];       // by product, and the batched call's arrays
    for (int b = 0; b < n; b++) {
        native_operands(sh[b], m[b]);
        for (int i = 0; i < 6; i++) role[i][b] = m[b][i];
    }
    bool ok = ab_zgemm_batched(n, role[0], role[1], role[2], role[3], role[4], role[5]) == AB_OK;
    for (int b = 0; b < n; b++) ok = native_result(sh[b], m[b], ok) == sh[b]->hash && ok;
    return ok;
}

typedef struct { NativeShape *sh[2]; int bad; } NativeThread;

static void *native_thread(void *arg) {
    NativeThread *t = arg;
    for (int r = 0; r < 6; r++) {
        const NativeShape *s = t->sh[r % 2];
        t->bad += native_hash(s, VIA_ZGEMM) != s->hash;
    }
    return NULL;
}

static int run_native_child(void) {
    if (ab_init() != AB_OK) {
        printf("native: no Metal device\n");
        return 1;
    }
    NativeShape small, mid, big, tilt;
    native_make(&mid, 257, 193, 129, 11, 1.0);    // odd, several tiles
    native_make(&big, 640, 576, 512, 12, 1.0);
    native_make(&small, 40, 24, 33, 13, 1.0);
    native_make(&tilt, 200, 168, 136, 14, 1e-6);  // nearly real: 4M
    char problem[160] = "";
#define NATIVE_CHECK(cond, ...) \
    do { if (!problem[0] && !(cond)) snprintf(problem, sizeof problem, __VA_ARGS__); } while (0)

    mid.hash = native_hash(&mid, VIA_ZGEMM);
    size_t pool = ab_internal_stage_pool_bytes();
    NATIVE_CHECK(mid.hash != 0, "ab_zgemm failed");
    NATIVE_CHECK(pool > 0, "the pool kept no temporaries after ab_zgemm");
    NATIVE_CHECK(native_hash(&mid, VIA_ZGEMM) == mid.hash, "a repeated ab_zgemm gave other bits");
    NATIVE_CHECK(ab_internal_stage_pool_bytes() == pool,
                 "the pool grew from %zu to %zu bytes on a repeated ab_zgemm", pool,
                 ab_internal_stage_pool_bytes());
    big.hash = native_hash(&big, VIA_ZGEMM);
    small.hash = native_hash(&small, VIA_ZGEMM);
    NATIVE_CHECK(big.hash && small.hash, "ab_zgemm failed");
    NATIVE_CHECK(native_hash(&mid, VIA_ZGEMM) == mid.hash,
                 "ab_zgemm gave other bits after larger and smaller shapes");
    static const char *const via_name[] = {"ab_zgemm", "ab_zgemm_async", "ab_batch_zgemm"};
    for (int v = VIA_ASYNC; v <= VIA_BATCH; v++)
        NATIVE_CHECK(native_hash(&mid, (NativeVia)v) == mid.hash, "%s gave other bits than ab_zgemm",
                     via_name[v]);

    // Nearly real data runs as 4M, whose four M x N temporaries come from the
    // same pool: every entry point must give the bits of an explicit 4M call,
    // which differ from 3M's, also in a batch that changes shape and method.
    tilt.hash = native_hash(&tilt, VIA_4M);
    NATIVE_CHECK(tilt.hash && native_hash(&tilt, VIA_3M) != tilt.hash,
                 "3M gave 4M's bits on nearly real data");
    for (int v = VIA_ZGEMM; v <= VIA_BATCH; v++)
        NATIVE_CHECK(native_hash(&tilt, (NativeVia)v) == tilt.hash,
                     "%s on nearly real data gave other bits than 4M", via_name[v]);
    NativeShape *batch[5] = {&mid, &tilt, &tilt, &big, &mid};
    NATIVE_CHECK(native_batched(batch, 5),
                 "ab_zgemm_batched gave other bits across shapes and between 3M and 4M");

    NativeThread t[4];
    pthread_t th[4];
    for (int i = 0; i < 4; i++) {
        t[i] = (NativeThread){{i % 2 ? &big : &mid, &small}, 0};
        pthread_create(&th[i], NULL, native_thread, &t[i]);
    }
    int bad = 0;
    for (int i = 0; i < 4; i++) {
        pthread_join(th[i], NULL);
        bad += t[i].bad;
    }
    NATIVE_CHECK(!bad, "%d of 24 ab_zgemm calls from 4 threads gave other bits", bad);

    // ab_shutdown() empties the pool. Growing shapes remake a slot's buffers
    // rather than add to them: after small, mid and big it holds what big
    // alone leaves.
    ab_shutdown();
    NATIVE_CHECK(ab_internal_stage_pool_bytes() == 0, "ab_shutdown() left %zu bytes in the pool",
                 ab_internal_stage_pool_bytes());
    ab_init();
    bool same = native_hash(&big, VIA_ZGEMM) == big.hash;
    size_t alone = ab_internal_stage_pool_bytes();
    ab_shutdown();
    ab_init();
    uint64_t hs = native_hash(&small, VIA_ZGEMM), hm = native_hash(&mid, VIA_ZGEMM),
             hb = native_hash(&big, VIA_ZGEMM);
    NATIVE_CHECK(same && hs == small.hash && hm == mid.hash && hb == big.hash,
                 "ab_zgemm gave other bits after ab_shutdown() and ab_init()");
    NATIVE_CHECK(alone && ab_internal_stage_pool_bytes() == alone,
                 "the pool kept %zu bytes after small, mid and big, %zu after big alone",
                 ab_internal_stage_pool_bytes(), alone);
#undef NATIVE_CHECK

    for (int i = 0; i < 4; i++) {
        free(small.in[i]); free(mid.in[i]); free(big.in[i]); free(tilt.in[i]);
    }
    ab_shutdown();
    if (problem[0]) printf("native: %s\n", problem);
    else printf("native ok\n");
    return 0;
}

// =============================================================================
// Parent
// =============================================================================

typedef struct {
    const char *label;
    const char *env[4];
    bool staged;                // the staged path is on
    bool registered;            // --child-registered
} Child;

static const Child CHILDREN[] = {
    {"host",   {"AB_MODE=gpu", "AB_MIN_GPU_DIM=0", "AB_HOST_PACK=1", NULL}, false, false},
    {"staged", {"AB_MODE=gpu", "AB_MIN_GPU_DIM=0", NULL}, true, false},
    {"nopool", {"AB_MODE=gpu", "AB_MIN_GPU_DIM=0", "AB_STAGING_MB=0", NULL}, true, false},
    {"registered", {"AB_MODE=gpu", "AB_MIN_GPU_DIM=0", NULL}, true, true},
};
#define N_CHILDREN ((int)(sizeof CHILDREN / sizeof CHILDREN[0]))

typedef struct {
    uint64_t hash;
    int repeat, gpu, ab, staged, in_place, expect;
    unsigned long long gemms;
} Line;

static char *run(const Child *ch, const char *self, const char *arg, int *status) {
    size_t n;
    char **env = child_environ(4, &n);
    for (int i = 0; ch->env[i]; i++) env[n++] = (char *)ch->env[i];
    env[n] = NULL;
    char *argv[] = {(char *)self, (char *)arg, NULL};
    char why[256];
    pid_t pid;
    FILE *out = spawn_piped(argv, env, &pid, why, sizeof why);
    free(env);
    if (!out) {
        printf("✗ %s child: %s\n", ch->label, why);
        *status = -1;
        return NULL;
    }
    char *s = slurp(out);
    while (waitpid(pid, status, 0) < 0 && errno == EINTR) {}
    return s;
}

static bool find_line(const char *out, const char *name, Line *l) {
    char key[128];
    snprintf(key, sizeof key, "case %s hash=", name);
    const char *p = strstr(out, key);
    memset(l, 0, sizeof *l);
    return p && sscanf(p + strlen(key), "%16" SCNx64 " repeat=%d staged=%d gpu=%d gemms=%llu "
                       "ab=%d in_place=%d expect=%d", &l->hash, &l->repeat, &l->staged, &l->gpu,
                       &l->gemms, &l->ab, &l->in_place, &l->expect) == 8;
}

int main(int argc, char **argv) {
    if (argc == 2 && !strcmp(argv[1], "--child")) return run_child(false);
    if (argc == 2 && !strcmp(argv[1], "--child-registered")) return run_child(true);
    if (argc == 2 && !strcmp(argv[1], "--child-native")) return run_native_child();
    if (argc != 1) {
        fprintf(stderr, "usage: %s [--child | --child-registered | --child-native]\n", argv[0]);
        return 2;
    }
    make_cases();
    printf("╔══════════════════════════════════════════════════════════════════╗\n");
    printf("║  apple-bottom staged path vs the host path, bit for bit          ║\n");
    printf("╚══════════════════════════════════════════════════════════════════╝\n");

    char *out[N_CHILDREN];
    static const char *const PATH[] = {"staged", "host", "accelerate"};
    int status[N_CHILDREN], failed = 0, passed = 0, paths[3] = {0};
    int kind_cases[D_COUNT][2] = {{0}}, kind_staged[D_COUNT][2] = {{0}};   // [kind][zgemm]
    for (int k = 0; k < N_CHILDREN; k++) {
        out[k] = run(&CHILDREN[k], argv[0],
                     CHILDREN[k].registered ? "--child-registered" : "--child", &status[k]);
        if (!out[k] || !WIFEXITED(status[k]) || WEXITSTATUS(status[k]) != 0) {
            printf("✗ %s child failed (status %d):\n%s\n", CHILDREN[k].label, status[k],
                   out[k] ? out[k] : "");
            failed++;
        }
    }
    for (int i = 0; i < g_ncases && !failed; i++) {
        const Case *c = &g_cases[i];
        char name[96], problem[256] = "";
        case_name(c, name, sizeof name);
        // In place, per staged call: A, B and C, but C not when the GPU reads
        // it (DGEMM beta != 0, ZGEMM alpha = 1, beta = 1) or the host writes it
        // (ZGEMM with alpha != 1 or beta not 0 or 1); nothing for a complex
        // operand 8 bytes off 16.
        double complex al = SCALARS[c->sc].alpha, be = SCALARS[c->sc].beta;
        bool z = c->routine == 'z';
        bool host_epi = z && !(al == 1.0 && (be == 0.0 || be == 1.0));
        int in_place_each = z && i % 3 == 2 ? 0 : 2 + (!host_epi && !case_c_in(c));
        Line l[N_CHILDREN];
        bool accelerate = false;            // the guard sends the call there: no bits to compare
        for (int k = 0; k < N_CHILDREN && !problem[0]; k++) {
            const Child *ch = &CHILDREN[k];
            if (!find_line(out[k], name, &l[k])) {
                snprintf(problem, sizeof problem, "%s: no result line", ch->label);
                break;
            }
            if (k == 0) accelerate = l[0].gpu == 0;
            int want = ch->staged && l[k].expect ? 2 : 0;
            int want_in_place = ch->registered ? want * in_place_each : 0;
            if (l[k].gpu != (accelerate ? 0 : 2) || l[k].gemms != (accelerate ? 0u : 2u))
                snprintf(problem, sizeof problem, "%s: gpu=%d gemms=%llu, expected %s", ch->label,
                         l[k].gpu, l[k].gemms, accelerate ? "Accelerate" : "the GPU");
            else if (!l[k].ab)
                snprintf(problem, sizeof problem, "%s: A or B modified", ch->label);
            else if (!l[k].repeat)
                snprintf(problem, sizeof problem, "%s: the second call gave other bits", ch->label);
            else if (l[k].staged != want)
                snprintf(problem, sizeof problem, "%s: %d of 2 calls staged, expected %d",
                         ch->label, l[k].staged, want);
            else if (l[k].in_place != want_in_place)
                snprintf(problem, sizeof problem, "%s: %d operands in place, expected %d",
                         ch->label, l[k].in_place, want_in_place);
            else if (k > 0 && !accelerate && l[k].hash != l[0].hash)
                snprintf(problem, sizeof problem, "%s: bits %016" PRIx64 " differ from the host "
                         "path's %016" PRIx64, ch->label, l[k].hash, l[0].hash);
        }
        bool ok = !problem[0];
        int path = accelerate ? 2 : l[0].expect ? 0 : 1;
        passed += ok;
        paths[path]++;
        kind_cases[c->data][z]++;
        kind_staged[c->data][z] += path == 0;
        printf("  %-46s %016" PRIx64 " %-10s %s%s\n", name, l[0].hash, PATH[path], ok ? "✓" : "✗ ",
               problem);
    }
    for (int d = 0; d < D_COUNT && !failed; d++)
        for (int z = 0; z < 2; z++)
            if (kind_cases[d][z] && (STAGES[d] ? !kind_staged[d][z] : kind_staged[d][z])) {
                printf("✗ %cgemm_ %s data: %d of %d cases staged, expected %s\n", z ? 'z' : 'd',
                       DATA_TAG[d], kind_staged[d][z], kind_cases[d][z], STAGES[d] ? "some" : "none");
                failed++;
            }
    if (!failed) {
        printf("\npaths: %d staged, %d host, %d Accelerate\n", paths[0], paths[1], paths[2]);
        size_t pool[N_CHILDREN];
        for (int k = 0; k < N_CHILDREN; k++) {
            const char *p = strstr(out[k], "pool_bytes=");
            pool[k] = p ? strtoull(p + 11, NULL, 10) : (size_t)-1;
        }
        printf("pool after the run: host %zu bytes, staged %zu, nopool %zu, registered %zu\n",
               pool[0], pool[1], pool[2], pool[3]);
        // With AB_HOST_PACK=1 the pool keeps the native ZGEMM's temporaries.
        bool kept = pool[0] && pool[0] != (size_t)-1 && pool[1] && pool[1] != (size_t)-1;
        if (!kept || pool[2] != 0) {
            printf("✗ expected some by default and with AB_HOST_PACK=1, none with AB_STAGING_MB=0\n");
            failed++;
        }
        int st;
        char *nout = run(&(Child){.label = "native"}, argv[0], "--child-native", &st);
        if (nout && WIFEXITED(st) && !WEXITSTATUS(st) && strstr(nout, "native ok")) {
            printf("✓ native ZGEMMs on the pool\n");
        } else {
            printf("✗ native ZGEMMs on the pool: %s", nout ? nout : "no output\n");
            failed++;
        }
        free(nout);
    }
    for (int k = 0; k < N_CHILDREN; k++) free(out[k]);

    printf("\n═══════════════════════════════════════════════════════════════════\n");
    printf("Results: %d passed, %d failed\n", passed, g_ncases - passed + failed);
    return passed == g_ncases && !failed ? 0 : 1;
}
