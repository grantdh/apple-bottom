# Skill strike provenance — 2026-04-23 — phantom comp-13 row

## Purpose

Provenance record for the strike of the `comp-13 (13 FLOPs/FMA) | 760 GFLOP/s |
34 bits | ✓ Best kernel` row from the apple-bottom-qe Claude Desktop skill.
The row claimed a production-grade kernel that has no existence in source, no
committed CSV provenance, and no git history. Captured here because the skill
file lives at a runtime path outside version control.

Addresses REVIEW_02 F3 / R2-3.

## Scope

- **Skill file:** `~/Library/Application Support/Claude/local-agent-mode-sessions/skills-plugin/10b74ea8-ba14-48d5-8080-0dfbed724ed1/3fbd1047-405e-4e89-a8f2-2ef726867034/skills/apple-bottom-qe/CLAUDE.md`
- **Edit date:** 2026-04-23
- **Pre-strike md5:** `3aa5607ae0d4ee7064094ae908c44e48` (post-commit-4 state)
- **Post-strike md5:** `ab69c1d154509b6ad4b7ab9cedbe0dfe` (verified on disk)

## Rationale

`git log -S "comp-13" --all` in the apple-bottom repo returns no hits. No
source file defines, implements, or benchmarks a comp-13 kernel. The
`760 GFLOP/s @ 34 bits` numbers have no CSV provenance. The row was
architectural speculation from early-stage kernel exploration that escaped
into the skill as a production-status claim. A skill should not list
kernels that don't ship.

## Pre-flight evidence

All five grep queries confirming the strike target was isolated (one row,
no dangling references elsewhere in the skill):

```
grep -n 'comp-13'       → 1 hit, only on the struck row
grep -n '13 FLOPs/FMA'  → 1 hit, only on the struck row
grep -n '760 GFLOP/s'   → 1 hit, only on the struck row
grep -n '34 bits'       → 1 hit, only on the struck row
grep -n 'Best kernel'   → 1 hit, only on the struck row
```

Post-strike verification: all five queries return zero hits in the current
skill file. No orphaned references to the phantom kernel remain.

## Diff

```diff
@@ What works, what doesn't @@
 | Approach | Speed | Precision | Verdict |
 |----------|-------|-----------|---------|
 | Register-blocked DD (4×4) | 640 GFLOP/s | ~10⁻¹⁵ | ✓ Production |
-| comp-13 (13 FLOPs/FMA) | 760 GFLOP/s | 34 bits | ✓ Best kernel |
 | Strassen d=1 @ 4096 | 715 GFLOP/s | ~10⁻¹⁵ | ✓ 1.12× |
 | simdgroup_matrix + DD | 2.7× faster | ~10⁻⁷ (FP32) | ✗ Blocked |
 | Ozaki scheme | 2× faster | ~10⁻⁷ (FP32) | ✗ Same problem |
 | Dual accumulator | -36% | — | ✗ Register pressure |
 | comp-10/12 no renorm | — | 7.7 bits | ✗ c.lo overflow |
 | FP16 Ozaki via MPS | — | — | ✗ DEAD (no FP16 speedup) |
```

## Table state after strike

7 data rows, narrative coherent without the phantom:

- **✓ rows (2):** Register-blocked DD (Production, 640 GFLOP/s) and
  Strassen d=1 @ 4096 (1.12×, 715 GFLOP/s)
- **✗ rows (5):** simdgroup_matrix + DD, Ozaki scheme, Dual accumulator,
  comp-10/12 no renorm, FP16 Ozaki via MPS

## Re-application instructions

If the skill is re-materialized by Claude Desktop and the phantom row
reappears, strike it deterministically:

```bash
SKILL='/Users/grantheileman/Library/Application Support/Claude/local-agent-mode-sessions/skills-plugin/10b74ea8-ba14-48d5-8080-0dfbed724ed1/3fbd1047-405e-4e89-a8f2-2ef726867034/skills/apple-bottom-qe/CLAUDE.md'
sed -i.bak-sed '/| comp-13 (13 FLOPs\/FMA) | 760 GFLOP\/s | 34 bits | ✓ Best kernel |/d' "$SKILL"
rm "$SKILL.bak-sed"
# Expected post-md5 (at current skill contents): ab69c1d154509b6ad4b7ab9cedbe0dfe
```

### Verification after re-application

```bash
grep -c 'comp-13'       "$SKILL"   # expect 0
grep -c '760 GFLOP/s'   "$SKILL"   # expect 0
grep -c '34 bits'       "$SKILL"   # expect 0
grep -c 'Best kernel'   "$SKILL"   # expect 0
```

## Out of scope / follow-up

- **L687–L688 routing rule** (`max(M,N,K) ≥ 1024`) disagrees with
  `DEFAULT_CROSSOVER_FLOPS_REAL=50M` in `src/blas_wrapper.c`, which puts
  square DGEMM crossover at N ≈ 292. Flagged for REVIEW_04 — not fixed
  in commit 5.
- **L671–L680 historical data tables** (DD-DGEMM GPU vs AMX, Gauss 3-multiply
  ZGEMM) kept intact. The commit-4 edit at L464 already date-anchors the
  old and new measurement regimes; these tables remain useful as the
  historical reference.
