/*
 * fft_types.h -- the per-dispatch parameter block, shared verbatim by the host
 * (src/fft.m) and the Metal kernel.
 *
 * Every member is 4 bytes wide so the layout is the same for both compilers.
 */
#ifndef ABF_TYPES_H
#define ABF_TYPES_H

#if defined(__METAL_VERSION__)
typedef uint abf_param_u32;
#else
#include <stdint.h>
typedef uint32_t abf_param_u32;
#endif

/* One GPU pass transforms lines of ABF_N points along one axis. A threadgroup
 * takes ABF_L consecutive lines of the partner index; grid y and z run over
 * the two outer indices. Strides count complex elements, relative to the
 * buffer offset the host binds for the current group of transforms. */
typedef struct abf_params {
    abf_param_u32 np;                             /* lines along the partner index */
    abf_param_u32 in_ls, in_ps, in_q1s, in_q2s;   /* input: line, partner, outer 1, outer 2 */
    abf_param_u32 out_ls, out_ps, out_q1s, out_q2s;
    float scale_hi, scale_lo;                     /* double-float scale (ABF_SCALE passes) */
} abf_params;

#endif /* ABF_TYPES_H */
