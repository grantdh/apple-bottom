# Skill edits provenance — 2026-04-23

## Purpose

The `apple-bottom-qe` Claude Desktop skill lives on disk at a runtime path
outside any git tree. This document captures the L458 (R2-5) and L464 (R3-4)
edits to that skill so the repository retains a reviewable record of what
was changed and why, and so the edits can be re-applied deterministically
if the skill is re-materialized from its cloud-backed source.

## Scope

- **Skill file:** `~/Library/Application Support/Claude/local-agent-mode-sessions/skills-plugin/10b74ea8-ba14-48d5-8080-0dfbed724ed1/3fbd1047-405e-4e89-a8f2-2ef726867034/skills/apple-bottom-qe/CLAUDE.md`
- **Plugin:** `anthropic-skills` v1.0.0 ("Anthropic-managed skills for Claude Desktop")
- **Skill ID:** `skill_01ApMhes6Mn1rDb3U8gSQ54g`, creatorType: `user`
- **Edit date:** 2026-04-23
- **Pre-edit md5:** `58a5e7dfe20d68e6588dc0023547d866`
- **Post-edit md5:** `3aa5607ae0d4ee7064094ae908c44e48`

## Rationale

Two targeted edits reconciling skill-level claims with the observed data
committed in the apple-bottom repo. Both edits follow the pattern
"strike bare assertion, link to repo artifact" established by REVIEW_02
and REVIEW_03.

- **L458 (R2-5).** The old text — `"42% FP32 utilization ceiling is intrinsic
  to error-free transforms on SIMD hardware — publishable result"` — was a
  bare claim with no CSV source. Replaced with a 43.3% floor framing that
  cross-references `docs/vv/VV_REPORT.md §10` (methodology) and
  `docs/design/FP32_UTILIZATION.md` (derivation), anchored to the 654 GFLOP/s
  measurement in `benchmarks/results/2026-04-22-b9b0641/dgemm.csv`. Follows
  the `974badf` correction (643 → 654 GFLOP/s CSV-provenance fix).
- **L464 (R3-4).** The old text — `"Gauss 3-multiply (3× DGEMM): +23-43% vs
  AMX at ≥1024. This is the production path."` — both overclaimed the upper
  bound (neither of the two 2026-04-22 runs reached +43%) and mischaracterized
  the dispatcher (AB_MODE AUTO is FLOP-based, not N-based). Replaced with the
  observed +14–33% range across two independent runs, crossover at N=1024,
  peak at N=2048–3072, with explicit ±10% boost-volatility caveat and a
  pointer to the code in `src/blas_wrapper.c` that actually implements
  routing. "Production path" dropped.

## Diff

```diff
--- CLAUDE.md.pre-commit4	2026-04-23 01:43:44
+++ CLAUDE.md	2026-04-23 01:44:10
@@ -455,13 +455,13 @@
 | comp-10/12 no renorm | — | 7.7 bits | ✗ c.lo overflow |
 | FP16 Ozaki via MPS | — | — | ✗ DEAD (no FP16 speedup) |

-42% FP32 utilization ceiling is intrinsic to error-free transforms on SIMD hardware — publishable result.
+43.3% FP32 utilization is a floor on sustained-boost compute, not a ceiling of theoretical peak — error-free transforms on SIMD hardware are intrinsically constrained by dd_fma's 18 FP32 FLOPs (9 per DP FLOP). See `docs/vv/VV_REPORT.md` §10 (methodology) and `docs/design/FP32_UTILIZATION.md` (derivation). Anchored to 654 GFLOP/s measurement in `benchmarks/results/2026-04-22-b9b0641/dgemm.csv`.

 ### ZGEMM strategy

 Native complex DD: ~320 GFLOP/s (AMX: ~620). **Failed.**

-Gauss 3-multiply (3× DGEMM): **+23-43% vs AMX at ≥1024**. This is the production path.
+Gauss 3-multiply (3× DGEMM): **+14–33% vs AMX** across two independent 2026-04-22 runs (`benchmarks/results/2026-04-22-b9b0641/zgemm.csv`, `benchmarks/results/2026-04-22-94a699d-run2/zgemm.csv`). Crossover at N=1024, peak at N=2048–3072, ±10% run-to-run from GPU boost-clock volatility. The older +23–43% range (see table ~L678) is historical; post-correction measurements show the narrower envelope. Dispatcher selects GPU by FLOP threshold (see `src/blas_wrapper.c`), not by raw N.
 ```
 K1 = Ar × Br,  K2 = Ai × Bi,  K3 = (Ar+Ai) × (Br+Bi)
 Cr = K1 - K2,  Ci = K3 - K1 - K2
```

## Out of scope / follow-up

- **L450 comp-13 row strike** (REVIEW_02 R2-3) → deferred to commit 5.
  `git log -S` confirmed no comp-13 kernel ever existed in source or
  history; the "✓ Best kernel" row will be struck cleanly then.
- **L687–L688 routing rule** claims `max(M,N,K) ≥ 1024` as the DGEMM/ZGEMM
  dispatch threshold. This disagrees with `DEFAULT_CROSSOVER_FLOPS_REAL=50M`
  in `src/blas_wrapper.c`, which puts square DGEMM crossover at N ≈ 292.
  Flagged for REVIEW_04 tranche — not fixed in commit 4.
- **L678–L680 historical ZGEMM data table** (the +23/+35/+43% table) kept
  as-is. The L464 edit now date-anchors both the old measurement regime
  (pre-correction, values in the L678 table) and the new one (two
  2026-04-22 runs referenced in the L464 text), so readers can see both.

## Re-application instructions

If the skill is re-materialized by Claude Desktop and the edits are lost,
reapply using the exact string matches below:

### Edit 1 — around line 458

Replace:
```
42% FP32 utilization ceiling is intrinsic to error-free transforms on SIMD hardware — publishable result.
```

With:
```
43.3% FP32 utilization is a floor on sustained-boost compute, not a ceiling of theoretical peak — error-free transforms on SIMD hardware are intrinsically constrained by dd_fma's 18 FP32 FLOPs (9 per DP FLOP). See `docs/vv/VV_REPORT.md` §10 (methodology) and `docs/design/FP32_UTILIZATION.md` (derivation). Anchored to 654 GFLOP/s measurement in `benchmarks/results/2026-04-22-b9b0641/dgemm.csv`.
```

### Edit 2 — around line 464

Replace:
```
Gauss 3-multiply (3× DGEMM): **+23-43% vs AMX at ≥1024**. This is the production path.
```

With:
```
Gauss 3-multiply (3× DGEMM): **+14–33% vs AMX** across two independent 2026-04-22 runs (`benchmarks/results/2026-04-22-b9b0641/zgemm.csv`, `benchmarks/results/2026-04-22-94a699d-run2/zgemm.csv`). Crossover at N=1024, peak at N=2048–3072, ±10% run-to-run from GPU boost-clock volatility. The older +23–43% range (see table ~L678) is historical; post-correction measurements show the narrower envelope. Dispatcher selects GPU by FLOP threshold (see `src/blas_wrapper.c`), not by raw N.
```

### Verification after re-application

```bash
SKILL='/Users/grantheileman/Library/Application Support/Claude/local-agent-mode-sessions/skills-plugin/10b74ea8-ba14-48d5-8080-0dfbed724ed1/3fbd1047-405e-4e89-a8f2-2ef726867034/skills/apple-bottom-qe/CLAUDE.md'
grep -c '43.3% FP32 utilization is a floor' "$SKILL"   # expect 1
grep -c '+14–33% vs AMX' "$SKILL"                       # expect 1
grep -c '42% FP32 utilization ceiling' "$SKILL"         # expect 0
grep -c '+23-43% vs AMX at ≥1024' "$SKILL"              # expect 0
```
