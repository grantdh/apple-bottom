/*
 * fft_kernel.metal -- one GPU pass of batched line FFTs along one axis.
 *
 * src/fft.m compiles fft_types.h + fft_device.h + this file as one library
 * per pass configuration, behind a block of #defines:
 *   ABF_N ABF_T ABF_L        line length, lanes per line, lines per threadgroup
 *   ABF_PLAN                 radix passes, e.g. ABF_P(8) ABF_P(3) ABF_P(3)
 *   ABF_SIGN                 -1 forward, +1 backward
 *   ABF_IN_F64 ABF_OUT_F64   1: interleaved FP64 complex (raw bits), 0: the
 *                            plan's double-float work buffer (float4)
 *   ABF_IN_LFAST ABF_OUT_LFAST
 *                            tile load / store order: 1 = consecutive threads
 *                            take consecutive points of a line, 0 =
 *                            consecutive lines (whichever is contiguous)
 *   ABF_SCALE                multiply by P.scale before the store
 *   ABF_TWOUT                four-step: multiply point x of line p by
 *                            W_N^(sign p x) before the store, from the next
 *                            pass's line table (twc, W_N^(ABF_N j)) and the
 *                            first ABF_N entries of W_N (twf)
 *
 * A threadgroup has ABF_T * ABF_L threads and holds ABF_L lines (partner
 * indices p0 .. p0+L-1) in threadgroup memory: tile load, the Stockham
 * passes (each lane ROUNDS butterflies), tile store. Lines past P.np compute
 * on zeros so that every thread reaches every barrier.
 */
#if ABF_IN_F64
#define ABF_IN_T ulong
#else
#define ABF_IN_T float4
#endif
#if ABF_OUT_F64
#define ABF_OUT_T ulong
#else
#define ABF_OUT_T float4
#endif
#define ABF_P(R) abf_fft_pass<ABF_SIGN, ABF_N, ABF_T, R>(a, lt, ns, tw); ns *= (R);

kernel void abf_pass(device const ABF_IN_T* in  [[buffer(0)]],
                     device ABF_OUT_T*      out [[buffer(1)]],
                     constant abf_cplx*     tw  [[buffer(2)]],
                     constant abf_params&   P   [[buffer(3)]],
#if ABF_TWOUT
                     device const float4*   twc [[buffer(4)]],
                     device const float4*   twf [[buffer(5)]],
#endif
                     uint3 tg  [[threadgroup_position_in_grid]],
                     uint  tid [[thread_index_in_threadgroup]])
{
    threadgroup abf_cplx sh[ABF_L * ABF_N];
    const uint li = tid / ABF_T, lt = tid % ABF_T;
    const uint p0 = tg.x * ABF_L;
    const uint ib = tg.y * P.in_q1s + tg.z * P.in_q2s;
    const uint ob = tg.y * P.out_q1s + tg.z * P.out_q2s;

    for (uint e = tid; e < ABF_L * ABF_N; e += ABF_T * ABF_L) {
#if ABF_IN_LFAST
        const uint l = e / ABF_N, x = e % ABF_N;
#else
        const uint l = e % ABF_L, x = e / ABF_L;
#endif
        const uint p = p0 + l;
        sh[l * ABF_N + x] = p < P.np ? abf_ld(in, ib + p * P.in_ps + x * P.in_ls) : abf_czero();
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    {
        threadgroup abf_cplx* a = sh + li * ABF_N;
        uint ns = 1u;
        ABF_PLAN
        (void)a; (void)ns; (void)lt;
    }
    for (uint e = tid; e < ABF_L * ABF_N; e += ABF_T * ABF_L) {
#if ABF_OUT_LFAST
        const uint l = e / ABF_N, x = e % ABF_N;
#else
        const uint l = e % ABF_L, x = e / ABF_L;
#endif
        const uint p = p0 + l;
        if (p >= P.np) continue;
        abf_cplx c = sh[l * ABF_N + x];
#if ABF_TWOUT
        const uint t = p * x;
        c = abf_ctw<ABF_SIGN>(c, abf_ctw<+1>(abf_from4(twc[t / ABF_N]), abf_from4(twf[t % ABF_N])));
#endif
#if ABF_SCALE
        c = abf_cmul_r(c, abf_mkdd(P.scale_hi, P.scale_lo));
#endif
        abf_st(out, ob + p * P.out_ps + x * P.out_ls, c);
    }
}
