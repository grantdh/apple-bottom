# DD Multiplication Primitives — Current State and v1.3.0 Investigation

> **Status (2026-10-30).** Written during 1.3.0 development and kept as
> written, with corrections marked. DWTimesDW2 stayed the default
> ([LIMITATIONS.md](LIMITATIONS.md)), so the question of §3 is settled.
> Correction (2026-10-30, second pass): the default product is not Algorithm 11
> of Joldes–Muller–Popescu 2017. f5dd2d3 folds the error of `a.hi*b.hi` into the
> first FMA, so Theorem 2.7 of Muller–Rideau covers neither `dd_mul` nor
> `dd_fma`. Bounds proven for the code as written are in
> `PRECISION_ENVELOPE.md` §1.2: `dd_mul` ≤ 6.000001u², `dd_fma` ≤ (13+43u)u²|ab|
> + (3+5u)u²|c|. They are checked by `make test-dd-bound`. The opt-in DWTimesDW3
> is Algorithm 12 verbatim, so Theorem 2.8 does hold for its `dd_mul`.

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

### 1.1 Default: DWTimesDW2 ~~(Alg 11 / Thm 2.7)~~ (a two-FMA variant of Alg 11; see the correction below)

Source: `dd_mul` / `dd_fma` in the `#else  // DWTimesDW2 (default)` branch of `src/apple_bottom.m`.

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
`f5dd2d3` ("perf: nested FMA cross-terms in dd_mul/dd_fma (2 roundings vs 4)"),
which is what drove the effective rounding-event count per cross-term from 4
down to 2.

**Error bound.** ~~`< 5u² / (1+u)² < 5u²` where `u = 2⁻²⁴` is the FP32 unit
roundoff — Muller–Rideau 2022 Theorem 2.7, tightened from JMP 2017's original
`< 6u²` bound via a formal Coq proof. This is `dd_mul`'s bound. The fused
`dd_fma` skips the product's own renormalization, so the theorem does not
cover it; `PRECISION_ENVELOPE.md` §1.2 gives it a conservative `≲ 8u²`
envelope (corrected 2026-10-30: this section first gave `dd_fma` the same
bound).~~

*Corrected 2026-10-30 (second pass).* Theorem 2.7 is for JMP Algorithm 11. That
algorithm computes `tl = RN(xh·yl)`, `cl2 = RN(tl + xl·yh)` and
`cl3 = RN(cl1 + cl2)`, which is three roundings. The code above makes two,
`fma(a.hi, b.lo, fma(a.lo, b.hi, e1))`. That is a different algorithm, and no
published theorem covers it. Proven for the code as written
(`PRECISION_ENVELOPE.md` §1.2):

- `dd_mul`: relative error ≤ (6u² + 2u³)/(1−u)² ≤ 6.000001u². The worst
  observed is 4.33u², against 4.20u² for Algorithm 11 on the same inputs.
- `dd_fma`: absolute error ≤ (13+43u)u²|ab| + (3+5u)u²|c|. Its final
  `fastTwoSum` is always exact.

### 1.2 Opt-in: DWTimesDW3 (Alg 12 / Thm 2.8)

Source: `dd_mul` / `dd_fma` in `src/apple_bottom.m` (inside the `#ifdef APPLEBOTTOM_USE_DWTIMESDW3`
branch). Landed as `feat: add DWTimesDW3 compile-time variant (opt-in)`
in commit `8c18ef3`.

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
>
> *Fixed in 65a4f17: the comment above the `#ifdef APPLEBOTTOM_USE_DWTIMESDW3`
> block of `src/apple_bottom.m` now gives one multiplication and one add.*

**Error bound.** `< 4u²` — Muller–Rideau 2022 Theorem 2.8, again for
`dd_mul` (this variant is Algorithm 12 verbatim); no bound is stated for its
fused `dd_fma`. Compared to ~~DWTimesDW2's `< 5u²`~~ the default's proven
6.000001u² (2026-10-30), the gap is ~~20%~~ 33% at the per-operation level,
although the observed worst cases differ by about 17% (4.33u² against 3.6u²). At
accumulation length `K`, ~~worst-case Higham bound improves from `O(5K·u²)` to
`O(4K·u²)`; probabilistic `√K·u²` bound improves by the same ratio~~ the
default's proven GEMM bound is `(3K+13)u²|A||B|` (`PRECISION_ENVELOPE.md`
§1.2); no GEMM bound is proven for this variant's `dd_fma`.

---

## 2. What was already done (don't repeat)

The FMA-cross-term migration described in earlier planning notes ("replace
additive accumulation with nested `fma()` calls, moving from ~5-13u² down to
≤2u² for the cross-term step") has already landed:

- `f5dd2d3` (April 2026) introduced nested FMA cross-terms in the default
  path. ~~That is what made the default `< 5u²` instead of the pre-f5dd2d3
  `< ~6-13u²` bound.~~ It also made the default differ from Algorithm 11, so
  its bound is the 6.000001u² proven in `PRECISION_ENVELOPE.md` §1.2
  (corrected 2026-10-30).
- `8c18ef3` (April 2026) added DWTimesDW3 as a compile-time opt-in, pushing
  the bound to `< 4u²` at the cost of +1 mul +1 add per primitive.

There is no lower-hanging cross-term rewrite left in a DD representation;
`< 4u²` is the tightest bound known for DD × DD multiplication under the
Muller–Rideau framework (Thm 2.8). Going below `4u²` in the current
representation would require a sub-2u² bound that is not known to exist for
unevaluated-pair DW arithmetic.

---

## 3. What the v1.3.0 investigation actually needs

*Outcome: DWTimesDW2 stayed the default ([LIMITATIONS.md](LIMITATIONS.md)).
The plan below is kept as written.*

The open v1.3.0 question is not "how do we tighten the bound further" but
"which variant should ship as the default." The Makefile comment
above the `DWTIMESDW3` block frames this as the "v1.3 A/C investigation":

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
     (~~`benchmarks/qe_yambo/`, eventual `benchmarks/si64_scf/`~~; neither
     was created, and the Quantum ESPRESSO campaign is in `benchmarks/qe/`)
     before deciding.

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
- `src/apple_bottom.m`, the `#ifdef APPLEBOTTOM_USE_DWTIMESDW3` … `#endif` block — the gated implementations themselves.
- The `DWTIMESDW3` block at the top of the `Makefile` — the `DWTIMESDW3=1` build flag and
  `-DAPPLEBOTTOM_USE_DWTIMESDW3` propagation.
