# DD Multiplication Primitives — Current State and v1.3.0 Investigation

**Purpose.** Document the current `dd_mul` / `dd_fma` implementations so future
work on the DD multiplication algorithm has a precise, version-stamped baseline
to measure against.

**Scope.** Descriptive only. No behavior change. If this document and
`src/apple_bottom.m` disagree, the source is authoritative and this file is
stale.

---

## 1. Current implementations

Both variants live in `src/apple_bottom.m` inside the Metal Shading Language
string `kShaderSource`, gated by the preprocessor macro
`APPLEBOTTOM_USE_DWTIMESDW3`. The gating is propagated from the Makefile's
`DWTIMESDW3=1` flag into `MTLCompileOptions.preprocessorMacros` so the MSL
compile step sees the same macro as the ObjC++ side. Default build is
DWTimesDW2.

### 1.1 Default: DWTimesDW2 (Alg 11 / Thm 2.7)

Source: `src/apple_bottom.m:198-216` (inside the `#else` branch at line 196).

```metal
inline DD dd_mul(DD a, DD b) {
    float p1, e1;
    twoProduct(a.hi, b.hi, p1, e1);
    e1 = fma(a.hi, b.lo, fma(a.lo, b.hi, e1));
    float s, e;
    fastTwoSum(p1, e1, s, e);
    return {s, e};
}

inline DD dd_fma(DD a, DD b, DD c) {
    float p1, e1;
    twoProduct(a.hi, b.hi, p1, e1);
    e1 = fma(a.hi, b.lo, fma(a.lo, b.hi, e1));
    float s2, e2;
    twoSum(p1, c.hi, s2, e2);
    e2 += e1 + c.lo;
    fastTwoSum(s2, e2, s2, e2);
    return {s2, e2};
}
```

Operation counts:

| Primitive | twoProduct | twoSum / fastTwoSum | FMA | Add | Rounding events |
|-----------|:----------:|:-------------------:|:---:|:---:|:---------------:|
| `dd_mul`  | 1 | 1 fastTwoSum | 2 (nested) | 0 | 2 |
| `dd_fma`  | 1 | 1 twoSum + 1 fastTwoSum | 2 (nested) | 2 (scalar `e2 += …`) | 2 |

The two cross-term FMAs are nested: `fma(a.hi, b.lo, fma(a.lo, b.hi, e1))`.
This form replaced the earlier additive-accumulation sequence in commit
`af722c3` ("perf: nested FMA cross-terms in dd_mul/dd_fma (2 roundings vs 4)"),
which is what drove the effective rounding-event count per cross-term from 4
down to 2.

**Error bound.** `< 5u² / (1+u)² < 5u²` where `u = 2⁻²⁴` is the FP32 unit
roundoff — Muller–Rideau 2022 Theorem 2.7, tightened from JMP 2017's original
`< 6u²` bound via a formal Coq proof.

### 1.2 Opt-in: DWTimesDW3 (Alg 12 / Thm 2.8)

Source: `src/apple_bottom.m:168-194` (inside the `#ifdef APPLEBOTTOM_USE_DWTIMESDW3`
branch at line 168). Landed as `feat: add DWTimesDW3 compile-time variant (opt-in)`
in commit `55bc79a`.

```metal
inline DD dd_mul(DD a, DD b) {
    float ch, cl1;
    twoProduct(a.hi, b.hi, ch, cl1);
    float tl0 = a.lo * b.lo;
    float tl1 = fma(a.hi, b.lo, tl0);
    float cl2 = fma(a.lo, b.hi, tl1);
    float cl3 = cl1 + cl2;
    float s, e;
    fastTwoSum(ch, cl3, s, e);
    return {s, e};
}

inline DD dd_fma(DD a, DD b, DD c) {
    float ch, cl1;
    twoProduct(a.hi, b.hi, ch, cl1);
    float tl0 = a.lo * b.lo;
    float tl1 = fma(a.hi, b.lo, tl0);
    float cl2 = fma(a.lo, b.hi, tl1);
    float cl3 = cl1 + cl2;
    float s2, e2;
    twoSum(ch, c.hi, s2, e2);
    e2 += cl3 + c.lo;
    fastTwoSum(s2, e2, s2, e2);
    return {s2, e2};
}
```

Operation counts (DWTimesDW3 vs DWTimesDW2 delta):

| Primitive | twoProduct | twoSum / fastTwoSum | FMA | Plain mul | Add |
|-----------|:----------:|:-------------------:|:---:|:---------:|:---:|
| `dd_mul`  | 1 (same) | 1 fastTwoSum (same) | 2 (same count, not nested) | **+1** (`a.lo*b.lo`) | **+1** (`cl1+cl2`) |
| `dd_fma`  | 1 (same) | 1 twoSum + 1 fastTwoSum (same) | 2 (same count, not nested) | **+1** | **+1** |

> Note: The source comment at line 164 reads "Adds one multiplication (al*bl),
> two FMAs, and one add vs Alg 11." The count on FMAs there is off — the delta
> is +1 mul and +1 add; the FMA count is the same (2) as DWTimesDW2. This is a
> documentation-only drift in the source comment; worth correcting but out of
> this doc's scope (no behavior change).

**Error bound.** `< 4u²` — Muller–Rideau 2022 Theorem 2.8. Compared to
DWTimesDW2's `< 5u²`, the gap is 20% at the per-operation level. At
accumulation length `K`, worst-case Higham bound improves from `O(5K·u²)` to
`O(4K·u²)`; probabilistic `√K·u²` bound improves by the same ratio.

---

## 2. What was already done (don't repeat)

The FMA-cross-term migration described in earlier planning notes ("replace
additive accumulation with nested `fma()` calls, moving from ~5-13u² down to
≤2u² for the cross-term step") has already landed:

- `af722c3` (April 2026) introduced nested FMA cross-terms in the default
  path. That is what made the default `< 5u²` instead of the pre-af722c3
  `< ~6-13u²` bound.
- `55bc79a` (April 2026) added DWTimesDW3 as a compile-time opt-in, pushing
  the bound to `< 4u²` at the cost of +1 mul +1 add per primitive.

There is no lower-hanging cross-term rewrite left in a DD representation;
`< 4u²` is the tightest bound known for DD × DD multiplication under the
Muller–Rideau framework (Thm 2.8). Going below `4u²` in the current
representation would require a sub-2u² bound that is not known to exist for
unevaluated-pair DW arithmetic.

---

## 3. What the v1.3.0 investigation actually needs

The open v1.3.0 question is not "how do we tighten the bound further" but
"which variant should ship as the default." The Makefile comment at
`Makefile:20-24` frames this as the "v1.3 A/C investigation":

1. **Precision measurement.** Run V-2 convergence study (`make
   test-verification`) under DWTimesDW3 and compare the Frobenius error
   distribution to the DWTimesDW2 baseline across `N ∈ {64, 128, …, 4096}`.
   Expected improvement is ~20% on worst-case matrices; the `√N`
   probabilistic regime is unlikely to show a visible change because the
   uncorrelated-rounding-error assumption already dominates.
2. **Performance measurement.** Run `make bench-paper` under both
   variants. Each `dd_mul` call costs one additional FP32 multiply and one
   additional FP32 add under DWTimesDW3. On an inner-loop kernel that
   retires one `dd_fma` per FP32 issue slot, the expected slowdown is
   bounded by the ratio of DWTimesDW3's total FP32 ops to DWTimesDW2's —
   roughly 9:7 or ≈ +29%. Actual impact depends on whether the kernel was
   FMA-throughput-bound or memory-bound to begin with.
3. **Decision rubric.**
   - If DWTimesDW3 shows visible Frobenius-error improvement in the V-2
     study (beyond Monte Carlo noise) AND DGEMM 4096² throughput drops by
     less than 5%: promote DWTimesDW3 to default, keep DWTimesDW2 as the
     opt-in-for-compat variant.
   - If the precision gain is within Monte Carlo noise OR throughput drops
     by more than 10%: keep DWTimesDW2 as default and leave DWTimesDW3 as
     opt-in documented for users who need the tighter guarantee.
   - Mid cases: defer to benchmark on real QE / Yambo workloads
     (`benchmarks/qe_yambo/`, eventual `benchmarks/si64_scf/`) before
     deciding.

---

## 4. References

- Joldes, Muller, Popescu (2017). *Tight and Rigorous Error Bounds for Basic
  Building Blocks of Double-Word Arithmetic.* ACM TOMS 44(2), Article 15.
  DOI: 10.1145/3121432
- Muller, Rideau (2022). *Formalization of Double-Word Arithmetic, and
  Comments on 'Tight and Rigorous Error Bounds for Basic Building Blocks of
  Double-Word Arithmetic.'* ACM TOMS 48(1), Article 9.
  DOI: 10.1145/3484514
- `docs/vv/PRECISION_ENVELOPE.md` §1.2 — per-operation DD bound derivation
  for the default (DWTimesDW2) path.
- `src/apple_bottom.m:157-218` — the gated implementations themselves.
- `Makefile:20-27` — the `DWTIMESDW3=1` build flag and `-DAPPLEBOTTOM_USE_DWTIMESDW3`
  propagation.
