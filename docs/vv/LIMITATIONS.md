# Limitations and Epistemic Posture

This document captures the epistemic stance shared across the apple-bottom V&V package. It is referenced from `PRECISION_ENVELOPE.md` and `VV_REPORT.md` and is intended to be read alongside them.

## Framing

Throughout the V&V package we scope claims to what we measure rather than to what prior theorems might be extended to cover; we state conservative envelopes where tight bounds are open, and preempt cross-hardware generalization until revalidation is done.

Three specific instances of this stance, linked to the sections where each is developed:

- **Fused double-word FMA envelope.** The `dd_fma(a, b, c) = a·b + c` primitive elides the mul-side `fastTwoSum` renormalization of Joldes–Muller–Popescu 2017 Algorithm 11 (DWTimesDW2). The Muller–Rideau 2022 Theorem 2.7 bound on DWTimesDW2 therefore does not apply by inheritance to the fused form. We state a conservative `≲ 8u² · max(|a·b|, |c|)` envelope from triangle inequality over the non-fused composition and validate it empirically at `N ∈ {1024, 2048, 3072}` in the `N·u²` accumulation regime. The tight bound for the fused form is an open theoretical question. See `PRECISION_ENVELOPE.md §1.2`.

- **Hardware scoping to Apple M2 Max.** All measurements in the V&V package are taken on a single Apple M2 Max (Apple8 GPU family) under `mathMode = .safe`. The Metal Shading Language specification commits to per-operation ULP tolerances but does not document cross-GPU-family bit-identity across the Apple7/Apple8/Apple9 families. The reported numerical envelope is an M2 Max measurement, not a platform guarantee; cross-family revalidation on M1, M3, and M4 is future work. See `PRECISION_ENVELOPE.md §2` and the hardware generality discussion in the companion paper manuscript.

- **DWTimesDW3 opt-in variant retained as non-default.** The DWTimesDW3 primitive (JMP 2017 Algorithm 12, MR 2022 Theorem 2.8) is available as a compile-time opt-in via `DWTIMESDW3=1` but is not the default. The default retains DWTimesDW2 based on measured cost/benefit: the tighter theoretical bound does not produce a measurable accuracy improvement on the DFT workloads targeted by this library, while the extra operations add FP32 cost on the hot path. The opt-in path is validated and supported; the decision to retain DWTimesDW2 as default is empirical, not theoretical. See `DD_PRIMITIVES_CURRENT.md §1.2`.

## What this posture rules in and out

This posture is deliberately narrower than "the library is correct under all conditions." It is also deliberately stronger than "the library works in practice." Specifically:

- Claims carry a named scope (workload, matrix size, hardware, compile-time flag set) and an envelope defensible within that scope.
- Claims outside that scope are either flagged as open (fused FMA tight bound), flagged as future revalidation work (cross-family hardware generality), or flagged as empirical choices that may not generalize (DWTimesDW3 opt-in default).
- Claims that would require extending published theorems beyond their stated preconditions are not made.

Production evaluators should read each numerical claim in the V&V package as scoped by this framing.
