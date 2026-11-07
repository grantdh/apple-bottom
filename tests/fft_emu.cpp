/*
 * fft_emu.cpp -- the GPU FFT's arithmetic replayed on the CPU, from the same
 * device code (src/fft/fft_device.h) and the same plan rules
 * (src/fft/fft_host.h). It must give the GPU's bits exactly: same FP64 ->
 * double-float conversion, same butterflies, twiddles and order of passes,
 * same scaling and FP64 rounding. Compile with -ffp-contract=off (no implicit
 * FMA) and -fno-strict-aliasing.
 *
 * Only packed batches (dist = N); tests gather strided data first. Inputs
 * must stay clear of FP32 subnormals: the GPU flushes them, the CPU does not.
 */
#include <stdint.h>
#include <string.h>

#include <vector>

#include "../src/fft/fft_device.h"
#include "../src/fft/fft_host.h"

namespace {

struct Line {
    int n = 1, np = 0, radix[ABF_MAX_PASSES];
    std::vector<abf_cplx> tw, tmp;

    bool init(int len)
    {
        n = len;
        np = abf_factor(len, radix);
        if (np < 0) return false;
        tw.resize((size_t)len);
        tmp.resize((size_t)len);
        abf_twiddles(len, len, reinterpret_cast<float*>(tw.data()));
        return true;
    }
};

template <int S, abf_u32 R>
void pass_r(Line& f, abf_cplx* a, abf_u32 ns)
{
    const abf_u32 n = (abf_u32)f.n;
    abf_cplx v[R];
    for (abf_u32 j = 0; j < n / R; ++j) {   // all loads, then all stores: the GPU's barrier
        abf_bfly_load<S, R>(a, n, ns, j, f.tw.data(), v);
        abf_bfly_store<R>(f.tmp.data(), ns, j, v);
    }
    memcpy(a, f.tmp.data(), (size_t)n * sizeof(abf_cplx));
}

template <int S>
void fft_line(Line& f, abf_cplx* a)
{
    abf_u32 ns = 1;
    for (int p = 0; p < f.np; ++p) {
        switch (f.radix[p]) {
        case 2: pass_r<S, 2>(f, a, ns); break;
        case 3: pass_r<S, 3>(f, a, ns); break;
        case 4: pass_r<S, 4>(f, a, ns); break;
        case 5: pass_r<S, 5>(f, a, ns); break;
        case 7: pass_r<S, 7>(f, a, ns); break;
        case 8: pass_r<S, 8>(f, a, ns); break;
        case 11: pass_r<S, 11>(f, a, ns); break;
        case 13: pass_r<S, 13>(f, a, ns); break;
        default: return;
        }
        ns *= (abf_u32)f.radix[p];
    }
}

void line_fft(Line& f, abf_cplx* a, int sign)
{
    if (sign > 0) fft_line<+1>(f, a);
    else fft_line<-1>(f, a);
}

abf_cplx load(const double* x)
{
    uint64_t b[2];
    memcpy(b, x, 16);
    return abf_mkc(abf_f64bits_to_dd(b[0]), abf_f64bits_to_dd(b[1]));
}

void store(double* y, abf_cplx c)
{
    const uint64_t b[2] = {abf_dd_to_f64bits(c.re), abf_dd_to_f64bits(c.im)};
    memcpy(y, b, 16);
}

}  // namespace

extern "C" int abf_emu_fft(int rank, const int* n_, int howmany, int sign, int norm, const double* in, double* out)
{
    long n[3] = {1, 1, 1};
    for (int d = 0; d < rank; ++d) n[d] = n_[d];
    const long N = n[0] * n[1] * n[2];
    float sh, sl;
    abf_hdd_split(abf_scale_hdd(norm, (double)N), &sh, &sl);
    const abf_dd scale = abf_mkdd(sh, sl);
    std::vector<abf_cplx> w((size_t)(N * howmany));
    for (long i = 0; i < N * howmany; ++i) w[(size_t)i] = load(in + 2 * i);
    auto finish = [&](abf_cplx c) { return norm ? abf_cmul_r(c, scale) : c; };

    int n1, n2;
    if (abf_line_split(rank, n[0], &n1, &n2) == ABF_FOUR_STEP) {
        Line fa, fb;
        if (!fa.init(n1) || !fb.init(n2)) return -1;
        std::vector<abf_cplx> fine((size_t)n1), line((size_t)(n1 > n2 ? n1 : n2)), y((size_t)N);
        abf_twiddles(N, n1, reinterpret_cast<float*>(fine.data()));
        for (long b = 0; b < howmany; ++b) {
            abf_cplx* x = &w[(size_t)(b * N)];
            for (long j2 = 0; j2 < n2; ++j2) {
                for (long j1 = 0; j1 < n1; ++j1) line[(size_t)j1] = x[n2 * j1 + j2];
                line_fft(fa, line.data(), sign);
                for (long k1 = 0; k1 < n1; ++k1) {
                    const long t = j2 * k1;
                    const abf_cplx tw = abf_ctw<+1>(fb.tw[(size_t)(t / n1)], fine[(size_t)(t % n1)]);
                    y[(size_t)(k1 * n2 + j2)] = sign > 0 ? abf_ctw<+1>(line[(size_t)k1], tw)
                                                         : abf_ctw<-1>(line[(size_t)k1], tw);
                }
            }
            for (long k1 = 0; k1 < n1; ++k1) {
                for (long j2 = 0; j2 < n2; ++j2) line[(size_t)j2] = y[(size_t)(k1 * n2 + j2)];
                line_fft(fb, line.data(), sign);
                for (long k2 = 0; k2 < n2; ++k2) store(out + 2 * (b * N + k1 + n1 * k2), finish(line[(size_t)k2]));
            }
        }
        return 0;
    }

    int axes[3], na = 0;
    for (int a = 0; a < rank; ++a)
        if (n[a] > 1) axes[na++] = a;
    for (int k = 0; k < na; ++k) {
        const int a = axes[k];
        Line f;
        if (!f.init((int)n[a])) return -1;
        long stride = 1;
        for (int d = 0; d < a; ++d) stride *= n[d];
        std::vector<abf_cplx> line((size_t)n[a]);
        const long nl = N / n[a];
        for (long b = 0; b < howmany; ++b)
            for (long l = 0; l < nl; ++l) {
                abf_cplx* base = &w[(size_t)(b * N + l % stride + (l / stride) * stride * n[a])];
                for (long i = 0; i < n[a]; ++i) line[(size_t)i] = base[i * stride];
                line_fft(f, line.data(), sign);
                for (long i = 0; i < n[a]; ++i) base[i * stride] = line[(size_t)i];
            }
    }
    for (long i = 0; i < N * howmany; ++i) store(out + 2 * i, finish(w[(size_t)i]));
    return 0;
}

/* The GPU's FP64 <-> double-float conversions, for test_conversions. */
extern "C" void abf_emu_to_dd(double x, float* hi, float* lo)
{
    uint64_t b;
    memcpy(&b, &x, 8);
    const abf_dd d = abf_f64bits_to_dd(b);
    *hi = d.hi;
    *lo = d.lo;
}

extern "C" double abf_emu_to_f64(float hi, float lo)
{
    const uint64_t b = abf_dd_to_f64bits(abf_mkdd(hi, lo));
    double x;
    memcpy(&x, &b, 8);
    return x;
}
