// =============================================================================
// apple_bottom_fft.h — Double-float complex FFTs on the GPU, FP64 in and out
// Copyright (c) 2026 Grant Heileman, Technology Residue. MIT License.
// =============================================================================
//
// Batched 1D, 2D and 3D complex discrete Fourier transforms on interleaved
// FP64 complex data (double[2] per element, as fftw_complex or Fortran
// COMPLEX(8)). The GPU computes in double-float (each value is an unevaluated
// sum of two FP32 numbers, ~48 significant bits), not IEEE FP64:
//
//   - the input is rounded to double-float, the transform runs in
//     double-float, and each output is the double-float result rounded to
//     FP64. The error is normwise, as an FFT's always is: a relative L2 error
//     of 3e-15 to 7e-15 on random data (N = 2 to 4M points), more than ten
//     times FFTW's FP64 error, and no FP64-identical results. Additions use
//     the faster double-float sum whose error bound is absolute, so a
//     component much smaller than the data's norm can have a large relative
//     error (it has with FFTW too, at FP64's level);
//   - the exponent range is FP32's: values below about 1e-31 in magnitude
//     lose low-order bits and below 1.2e-38 become zero; results above
//     3.4e38 overflow to Inf or NaN.
//
// Layout (column-major, Fortran order): n[0] is the contiguous, fastest
// varying dimension, as in FFTW's Fortran interface (dfftw_plan_many_dft) and
// Quantum ESPRESSO's psic(nr1x, nr2x, nr3x). FFTW's C interface lists the
// dimensions the other way round. Element (i0, i1, i2) of transform b is
//
//     data[b*dist + stride*(i0 + embed[0]*(i1 + embed[1]*i2))]
//
// counted in complex elements. With embed == NULL the array is packed
// (embed = n). embed[rank-1] is not used.
//
// Transform:  out[k] = scale * sum_j in[j] * exp(sign * 2*pi*i * j.k / n),
// with sign AB_FFT_FORWARD (-1) or AB_FFT_BACKWARD (+1), as FFTW, and the
// scale chosen at plan time: 1, 1/N or 1/sqrt(N), N = n[0]*...*n[rank-1].
//
// Lengths: each n[d] must factor into 2, 3, 5, 7, 11 and 13. Rank-2 and
// rank-3 dimensions are at most 2048; a rank-1 length up to 2048*2048 runs
// as two GPU passes (four-step).
//
// Memory: ab_fft_execute works on any caller arrays. By default it copies the
// input into staging buffers the plan reuses (never a new GPU buffer per call)
// and copies the output back, touching only the output's elements (padding
// between them is left as it was); a large batch moves in chunks of about
// 32 MB whose copies overlap the GPU work. Arrays inside memory from
// ab_fft_malloc or registered with ab_fft_register are read and written in
// place (zero copy). Registration is explicit: the library never wraps an
// arbitrary address on its own, because a wrapper kept by address goes stale
// once that memory is freed and reused.
//
// In-place transforms: in == out with the same layout. Any other overlap of
// the input and output arrays is rejected.
//
// Threads: one plan may be executed from several threads; its executions run
// one at a time. Different plans run concurrently. A plan created before
// fork() cannot be used in the child (ab_fft_execute returns
// AB_ERROR_NO_DEVICE there).
//
// The FFT does not need ab_init(); it creates its own Metal objects. Plans of
// more than one GPU pass (rank 2 and 3, and rank-1 lengths past 2048) keep a
// double-float work buffer of at most AB_FFT_WORK_MB megabytes (default 256,
// read when the plan is created): larger batches run in groups that fit it.
// =============================================================================
#ifndef APPLE_BOTTOM_FFT_H
#define APPLE_BOTTOM_FFT_H

#include <stddef.h>
#include "apple_bottom.h"   // ABStatus

#ifdef __cplusplus
extern "C" {
#endif

typedef struct ab_fft_plan_s* ab_fft_plan_t;

enum {
    AB_FFT_FORWARD  = -1,   // exp(-2 pi i j k / n), FFTW_FORWARD
    AB_FFT_BACKWARD = +1    // exp(+2 pi i j k / n), FFTW_BACKWARD
};

// Scale applied to every output. There is no default: a plan states it.
typedef enum {
    AB_FFT_NORM_NONE   = 0,   // 1 (FFTW's convention)
    AB_FFT_NORM_N      = 1,   // 1/N
    AB_FFT_NORM_SQRT_N = 2    // 1/sqrt(N) (unitary)
} ab_fft_norm;

// howmany transforms of rank 1..3 and size n[0] x ... x n[rank-1] (n[0]
// fastest), with FFTW's advanced-interface layout: stride and dist in complex
// elements, embed[] >= n[] (NULL: packed). Returns NULL on failure, with the
// reason in *status (if status != NULL) and ab_fft_last_error():
// AB_ERROR_INVALID_ARG for a bad argument or an unsupported length,
// AB_ERROR_NO_DEVICE without a Metal device, AB_ERROR_SHADER_COMPILE,
// AB_ERROR_ALLOC_FAILED.
ab_fft_plan_t ab_fft_plan_many(int rank, const int* n, int howmany,
                               const int* inembed, int istride, int idist,
                               const int* onembed, int ostride, int odist,
                               int sign, ab_fft_norm norm, ABStatus* status);

// Packed transforms, one after the other in memory (dist = N).
ab_fft_plan_t ab_fft_plan_1d(int n0, int howmany, int sign, ab_fft_norm norm, ABStatus* status);
ab_fft_plan_t ab_fft_plan_2d(int n0, int n1, int howmany, int sign, ab_fft_norm norm, ABStatus* status);
ab_fft_plan_t ab_fft_plan_3d(int n0, int n1, int n2, int howmany, int sign, ab_fft_norm norm, ABStatus* status);

void ab_fft_destroy(ab_fft_plan_t plan);

// Transform `in` into `out` (interleaved FP64 complex; in == out for in place).
// Synchronous: the result is in `out` when it returns.
ABStatus ab_fft_execute(ab_fft_plan_t plan, const double* in, double* out);

// Zero copy. ab_fft_malloc returns page-aligned memory that the GPU uses in
// place (free it with ab_fft_free). ab_fft_register makes [ptr, ptr+bytes) of
// caller-owned memory usable in place until ab_fft_unregister(ptr); the
// memory must stay allocated until then. Registered ranges must not overlap.
void*    ab_fft_malloc(size_t bytes);
void     ab_fft_free(void* ptr);
ABStatus ab_fft_register(void* ptr, size_t bytes);
ABStatus ab_fft_unregister(void* ptr);

// What the last ab_fft_execute on this plan did.
typedef struct {
    double total_ms;      // whole call
    double copy_in_ms;    // host copies into staging (0 with zero copy)
    double copy_out_ms;   // host copies out of staging, not overlapped with the GPU
    double gpu_ms;        // sum of GPUEndTime - GPUStartTime over command buffers
    int    zero_copy_in;  // 1 if the input was read in place
    int    zero_copy_out; // 1 if the output was written in place
    int    chunks;        // groups of transforms the GPU passes ran over
} ab_fft_stats;
void ab_fft_get_stats(ab_fft_plan_t plan, ab_fft_stats* stats);

// Plan summary: sizes, radix passes, threads per line and lines per
// threadgroup of each GPU pass, e.g. for benchmark logs.
void ab_fft_describe(ab_fft_plan_t plan, char* buf, size_t len);

// Reason for the last failure on this thread ("" if none).
const char* ab_fft_last_error(void);

#ifdef __cplusplus
}
#endif

#endif // APPLE_BOTTOM_FFT_H
