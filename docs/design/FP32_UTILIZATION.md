# FP32 utilization derivation

## Purpose

Standalone derivation of the "42% FP32 utilization of theoretical peak
at boost clock" claim. This document shows the arithmetic that connects
the DD-DGEMM benchmark CSVs to the utilization figure anchored in
`docs/vv/VV_REPORT.md §10`.

## Inputs

| Symbol    | Description                          | Value          | Source                                                      |
|-----------|--------------------------------------|----------------|-------------------------------------------------------------|
| G_DD      | DD-DGEMM throughput (DP GFLOP/s)     | 643–670        | `benchmarks/results/2026-04-22-b9b0641/dgemm.csv` (N≥2048)  |
| k         | FP32 FLOPs per dd_fma call           | 18             | `src/apple_bottom.m:209-218` (DWTimesDW2, default path)     |
| k'        | FP32 FLOPs per DP FLOP               | 9              | = k/2; each dd_fma delivers 1 DP mul + 1 DP add = 2 DP FLOPs |
| f_boost   | GPU boost clock                      | 1.398 GHz      | Apple M2 Max GPU specification                              |
| N_cores   | GPU cores                            | 38             | M2 Max 38-core SKU                                          |
| n_FMA     | FP32 FMA per cycle per core          | 128            | M2 Max GPU spec                                             |
| P_peak    | Theoretical FP32 peak                | 13.60 TFLOP/s  | f_boost × N_cores × n_FMA × 2 FLOP/FMA                      |

## Derivation

Each dd_fma call computes one double-float multiply-add. Expanding the
double-float arithmetic into FP32 operations gives k FP32 ops per call
(see `src/apple_bottom.m:209-218` for the explicit sequence).

Each dd_fma call produces 2 double-precision FLOPs (one multiply and
one add in double-float arithmetic). Therefore the FP32 cost per DP
FLOP is:

    k' = k / 2

Equivalent FP32 throughput from measured DD-DGEMM:

    G_FP32_effective = G_DD × k'

Utilization of theoretical FP32 peak:

    utilization = G_FP32_effective / P_peak

With k=18 and k'=9:

    G_FP32_effective ∈ [643 × 9, 670 × 9] = [5.79, 6.03] TFLOP/s
    utilization      ∈ [5.79 / 13.60, 6.03 / 13.60] = [42.6%, 44.3%]

Point estimate at G_DD = 643 GFLOP/s (N=2048): 42.6%.

The tighter 42.6–44.3% range here reflects benchmark size variation
only. The wider 35–45% band reported in VV_REPORT §10 includes
additional uncertainty from the denominator (boost clock reachability
— see §10 for the powermetrics-based discussion).

## Limitations

1. **Static FP32 op count.** The k value is a count from source code,
   not a measured hardware op count. Actual hardware utilization may
   differ due to pipelining, register pressure, texture sampler stalls,
   and memory latency.

2. **Compute-only.** This figure characterizes compute utilization only.
   Memory bandwidth utilization is a separate metric (not computed here).

3. **Peak assumes sustained boost.** P_peak assumes all 38 cores at
   sustained 1.398 GHz. `docs/vv/VV_REPORT.md §10` documents that
   bench_paper sustains this clock in only 7 of 240 powermetrics samples
   (~3% of wall time). Utilization of *sustained-boost compute* is
   correspondingly higher than this figure suggests.

4. **Size range.** G_DD is stable at N≥2048 (643–670 GFLOP/s in the
   committed CSV). Smaller N produces lower utilization due to
   per-kernel overhead amortization — see the dgemm.csv for the full
   size-dependent curve.

5. **Negation convention.** The `-p` sign flip in `fma(a, b, -p)`
   (line 211) is counted as free, per IEEE 754 convention. Counting it
   as 1 FLOP would adjust the total to 19 per call / 9.5 per DP FLOP,
   shifting the point estimate to 45.0%.

6. **Renormalization overhead excluded.** Periodic renormalization
   (`RENORM_INTERVAL`, `src/apple_bottom.m:262`) adds approximately 1%
   amortized FP32 overhead across full GEMM kernels. Not included in
   the 18-op-per-call count; would shift the point estimate by <0.5%.

7. **Variant kernel not covered.** The `DWTimesDW3` variant
   (`src/apple_bottom.m:184-196`) is opt-in and has a different FP32 op
   count. The default path analyzed here is `DWTimesDW2`. Any
   utilization figure for the alternate variant requires a separate
   derivation.

## References

- Benchmark data: `benchmarks/results/2026-04-22-b9b0641/dgemm.csv`
- Clock-pin methodology: `docs/vv/VV_REPORT.md §10`
- DD arithmetic precision envelope: `docs/vv/PRECISION_ENVELOPE.md`
- Reproducibility: `docs/REPRODUCIBILITY.md`
- dd_fma source: `src/apple_bottom.m:209-218`
