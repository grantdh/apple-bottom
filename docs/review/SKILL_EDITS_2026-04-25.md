# REVIEW_04 commit #2 — skill regime-rectification

## Files touched

- **Runtime skill** (not git-tracked, in-place edit):
  `~/Library/Application Support/Claude/local-agent-mode-sessions/skills-plugin/10b74ea8-ba14-48d5-8080-0dfbed724ed1/3fbd1047-405e-4e89-a8f2-2ef726867034/skills/apple-bottom-qe/CLAUDE.md`
- **`include/apple_bottom.h`** (tracked; staged in this commit alongside the
  provenance doc).

## md5 transition

| File | Pre-edit | Post-edit |
|---|---|---|
| Runtime skill `CLAUDE.md` | `ab69c1d154509b6ad4b7ab9cedbe0dfe` | `2eefd99009cca50bd33a99e340edf0a4` |
| `include/apple_bottom.h` | `78efab23bcf39f2f1780db6205cb977c` | `b8d974e55d76efc933a33450efe21555` |

Pre-edit skill md5 matches the post-strike state from the 2026-04-23 sprint
(see `docs/review/SKILL_STRIKE_2026-04-23_comp13.md`). File was unchanged
between sprints.

## Empirical grounding

Anchored to the REVIEW_04 Tranche A commit chain:

- **`271a0f0`** — bench: add rect bench + dispatch-path hook for REVIEW_04
  (infrastructure: `bench_rect_{dgemm,zgemm}`, configs, sweep runner,
  `ab_get_last_dispatch_path()` thread-local hook).
- **`19130d5`** — docs: REVIEW_04 Tranche A — end-to-end rect sweep
  (12 CSVs under `benchmarks/results/2026-04-24-271a0f0-rect/`, 1890 verify
  rows / 0 failures, R4-2 split between threshold-tuning and unexplained
  overhead).
- **`be4c6fd`** — bench: AB_PROFILE env-gated per-stage timing in `ab_*_blas`
  (R4-2b instrumentation; per-stage breakdown in `ab_zgemm_blas`).
- **`1b70258`** — docs: REVIEW_04 Tranche A followup — R4-2b localized
  (sum-of-stages = wall to 0.04 ms; FP64↔DD conversion at ~140 MB/s
  identified as dominant cost).

**Headline finding** the rectified text encodes: the BLAS-ABI path
(`ab_*_blas`) runs end-to-end at 30–49× below steady-state across all
measured shapes through N=2048 ZGEMM, dominated by per-call FP64↔DD
conversion overhead at ~140 MB/s. Not threshold tuning territory; the
fix is either persistent ABMatrix handles or a vectorized conversion
kernel.

## Embedded diff — `include/apple_bottom.h`

```diff
--- a/include/apple_bottom.h
+++ b/include/apple_bottom.h
@@ -13,11 +13,25 @@
 //   - Matrix operations: NOT thread-safe — Metal command queue serializes
 //   - Use separate contexts for concurrent workloads (future feature)
 //
-// Performance Notes:
-//   - DGEMM: GPU wins for N >= 2048
-//   - ZGEMM: GPU wins for N >= 1024
-//   - DSYRK: GPU wins for N >= 3072
-//   - ZHERK: GPU wins for N >= 1024 (v1.4: GPU-native transpose, no CPU roundtrip)
+// Performance notes — TWO regimes (see docs/review/REVIEW_04_TRANCHE_A_2026-04-24.md
+// and the apple-bottom-qe skill §0 for the full distinction).
+//
+// Steady-state regime — native ABMatrix API, ABMatrix handles persisted across calls,
+// upload and FP64<->DD conversion amortized:
+//   DGEMM: GPU wins for N >= 2048   (anchor: benchmarks/results/2026-04-22-b9b0641/dgemm.csv)
+//   ZGEMM: GPU wins for N >= 1024   (anchor: benchmarks/results/2026-04-22-b9b0641/zgemm.csv)
+//   DSYRK: GPU wins for N >= 3072   (steady-state; prior measurement, not re-measured for end-to-end regime)
+//   ZHERK: GPU wins for N >= 1024   (steady-state; v1.4 introduced GPU-native transpose / no CPU roundtrip; not re-measured for end-to-end regime)
+//
+// End-to-end regime — BLAS-ABI (ab_*_blas, raw pointers, no handle persistence; this is
+// the path Fortran host codes hit through dgemm_/zgemm_):
+//   No crossover in the measured grid through N=2048. AMX wins by 15-170x at all
+//   tested shapes, dominated by FP64<->DD conversion at ~140 MB/s effective bandwidth
+//   (root cause localized in commit be4c6fd via AB_PROFILE=1).
+//   For Fortran host codes calling dgemm_/zgemm_ without handle caching: AB_MODE=cpu
+//   is the right default. Anchor: benchmarks/results/2026-04-24-271a0f0-rect/.
+//   DSYRK/ZHERK end-to-end behavior expected to be similar to GEMM (same conversion
+//   path); not directly measured.
 //
 // API Limits:
 //   - AB_MAX_DIMENSION = 46340 (max matrix dimension, overflow protection)
```

## Embedded diff — runtime skill `apple-bottom-qe/CLAUDE.md`

(Not git-tracked; recorded here for re-application if the runtime is
re-materialized. Diff is against md5 `ab69c1d154509b6ad4b7ab9cedbe0dfe`.)

```diff
--- a/CLAUDE.md
+++ b/CLAUDE.md
@@ -7,8 +7,41 @@

 ---

+## 0. Two Regimes — Read This First
+
+apple-bottom is effectively two libraries depending on call pattern. Mistaking which regime a measurement belongs to is the most consequential reading error a reviewer can make with this codebase. Several performance numbers in this document have historically been ambiguous on this point; the present revision (commit referenced in `docs/review/SKILL_EDITS_2026-04-25.md`) tags each number with its regime.
+
+### Steady-state regime — native ABMatrix API, persistent handles
+
+Callers retain `ABMatrix` handles across calls; matrices are uploaded once per shape and reused across iterations. Dispatcher FLOP thresholds (50 MFLOPs DGEMM real path, 100 MFLOPs complex/skinny path) are calibrated to this regime. Crossover is real and falls in the documented range: N ≥ 1024 (ZGEMM), N ≥ 2048 (DGEMM). Headline kernel throughput at N=2048: DGEMM 654 GFLOP/s, ZGEMM 813 GFLOP/s.
+
+Anchors: `benchmarks/results/2026-04-22-b9b0641/{dgemm,zgemm}.csv` (steady-state benches `bench_dgemm.c`, `bench_zgemm.c`).
+
+### End-to-end regime — BLAS-ABI path, ab_*_blas, no handle persistence
+
+Callers reach the library through Fortran `dgemm_`/`zgemm_` symbols mapped to `ab_dgemm_blas`/`ab_zgemm_blas`. Each call allocates ABMatrix handles, col→row repacks, runs FP64↔DD conversion (upload), dispatches the compute kernel, downloads, and destroys — every call. **There is no crossover in the measured grid through N=2048**; AMX wins at all tested shapes by 15–170×. Headline at N=2048: DGEMM 20 GFLOP/s, ZGEMM 40 GFLOP/s.
+
+Anchors: `benchmarks/results/2026-04-24-271a0f0-rect/` (12 CSVs, 1890 verify rows / 0 failures), provenance in `docs/review/REVIEW_04_TRANCHE_A_2026-04-24.md`.
+
+Root cause is **neither** the GPU kernel **nor** dispatcher threshold calibration. AB_PROFILE=1 stage breakdown on a single N=2048 ZGEMM call (commit `be4c6fd`) localized the dominant cost to FP64↔DD conversion in upload/download, running at ~140 MB/s effective bandwidth — ~280× below naive memcpy on Apple Silicon UMA. Detail in `docs/review/REVIEW_04_TRANCHE_A_2026-04-24.md`, "Update 2026-04-24 — R4-2b localized."
+
+### Recommendation matrix
+
+| Caller pattern | Recommendation |
+|---|---|
+| Fortran host codes (QE, Yambo, EPW) calling `dgemm_`/`zgemm_` with no handle caching | `AB_MODE=cpu`. The BLAS-ABI GPU path is unprofitable at all measured shapes through N=2048. |
+| Native C/Obj-C caller using `ab_dgemm`/`ab_zgemm` with `ABMatrix` handles persisted across iterations | Defaults are correct. GPU wins at N ≥ 1024 (ZGEMM) / N ≥ 2048 (DGEMM). |
+| Future Fortran integration with handle caching introduced at the call site (rewrite required) | Migrate to native API and expect speedups consistent with the steady-state benches. |
+
+### Reading the rest of this document
+
+Every GFLOP/s figure, speedup percentage, and routing claim below applies to **one** of the two regimes. Where prior corrections (commits `10af966`, `a98e91f`, and the present commit) have added regime labels, trust them. Where a number is unlabeled and predates 2026-04-25, assume steady-state by lineage but verify against the cited CSV before relying on it for an end-to-end conclusion.
+
+---
+
 ## Table of Contents

+0. [Two Regimes — Read This First](#0-two-regimes--read-this-first)
 1. [Repository Layout](#1-repository-layout)
 2. [Symbol Chain — How Fortran Finds Metal](#2-symbol-chain)
 3. [⛔ STOP-AND-CHECK — The #1 Recurring Bug](#3-stop-and-check)
@@ -444,6 +477,8 @@

 ### What works, what doesn't

+**Regime: steady-state (native ABMatrix API, persistent handles).** Throughput in this table is kernel-only, with allocation and FP64↔DD conversion amortized across calls. For end-to-end BLAS-ABI behavior see §0.
+
 | Approach | Speed | Precision | Verdict |
 |----------|-------|-----------|---------|
 | Register-blocked DD (4×4) | 640 GFLOP/s | ~10⁻¹⁵ | ✓ Production |
@@ -454,13 +489,17 @@
 | comp-10/12 no renorm | — | 7.7 bits | ✗ c.lo overflow |
 | FP16 Ozaki via MPS | — | — | ✗ DEAD (no FP16 speedup) |

-43.3% FP32 utilization is a floor on sustained-boost compute, not a ceiling of theoretical peak — error-free transforms on SIMD hardware are intrinsically constrained by dd_fma's 18 FP32 FLOPs (9 per DP FLOP). See `docs/vv/VV_REPORT.md` §10 (methodology) and `docs/design/FP32_UTILIZATION.md` (derivation). Anchored to 654 GFLOP/s measurement in `benchmarks/results/2026-04-22-b9b0641/dgemm.csv`.
+In the steady-state regime [§0], 43.3% FP32 utilization is a floor on sustained-boost compute, not a ceiling of theoretical peak — error-free transforms on SIMD hardware are intrinsically constrained by dd_fma's 18 FP32 FLOPs (9 per DP FLOP). See `docs/vv/VV_REPORT.md` §10 (methodology) and `docs/design/FP32_UTILIZATION.md` (derivation). Anchored to 654 GFLOP/s measurement in `benchmarks/results/2026-04-22-b9b0641/dgemm.csv`. The end-to-end BLAS-ABI path achieves ~1.3% effective FP32 utilization at the same shape (N=2048 DGEMM, 20 GFLOP/s wall) — dominated by FP64↔DD conversion overhead at ~140 MB/s, not by kernel work. See §0 and `docs/review/REVIEW_04_TRANCHE_A_2026-04-24.md`.

 ### ZGEMM strategy

 Native complex DD: ~320 GFLOP/s (AMX: ~620). **Failed.**

-Gauss 3-multiply (3× DGEMM): **+14–33% vs AMX** across two independent 2026-04-22 runs (`benchmarks/results/2026-04-22-b9b0641/zgemm.csv`, `benchmarks/results/2026-04-22-94a699d-run2/zgemm.csv`). Crossover at N=1024, peak at N=2048–3072, ±10% run-to-run from GPU boost-clock volatility. The older +23–43% range (see table ~L678) is historical; post-correction measurements show the narrower envelope. Dispatcher selects GPU by FLOP threshold (see `src/blas_wrapper.c`), not by raw N.
+The Gauss 3-multiply ZGEMM kernel is **+14–33% faster than AMX in the steady-state regime** across two independent 2026-04-22 runs (`benchmarks/results/2026-04-22-b9b0641/zgemm.csv`, `benchmarks/results/2026-04-22-94a699d-run2/zgemm.csv`). Crossover at N=1024, peak at N=2048–3072, ±10% run-to-run from GPU boost-clock volatility. The older +23–43% range (see §16 historical table) is a kernel-evolution snapshot from earlier 2025–2026; the current envelope is the +14–33% figure.
+
+In the end-to-end regime (Fortran BLAS-ABI through `ab_zgemm_blas`), **AMX wins at all measured shapes through N=2048** by 15–40× (`benchmarks/results/2026-04-24-271a0f0-rect/`). The +14–33% figure characterizes kernel-vs-kernel throughput; it does not predict end-to-end behavior for callers without `ABMatrix` handle persistence. See §0.
+
+Dispatcher routing uses **two** gates: `min(M, N, K) ≥ AB_MIN_GPU_DIM` (default 32) AND a FLOP-class threshold (50 MFLOPs DGEMM real path, 100 MFLOPs complex/skinny path). Both gates are calibrated to the steady-state crossover and do not predict end-to-end crossover. See §16 for the full routing rule, `src/blas_wrapper.c::ab_use_gpu*()` for the authoritative source.
@@ -663,6 +702,8 @@

 ### DD-DGEMM (GPU vs AMX)

+**Regime: steady-state (kernel-only, native ABMatrix path).** End-to-end equivalents are 25–40× lower; see `docs/review/REVIEW_04_TRANCHE_A_2026-04-24.md`.
+
 | Size | DD GPU | AMX CPU | Winner |
 |------|--------|---------|--------|
 | 512 | 183 GFLOP/s | 444 GFLOP/s | AMX |
@@ -672,6 +713,9 @@

 ### Gauss 3-multiply ZGEMM

+**Regime: steady-state — kernel-only historical deltas, 2025–early-2026 kernel evolution.** Current ZGEMM steady-state envelope is +14–33% (see §8 around L462 and `benchmarks/results/2026-04-22-*/`); rows below are retained as a kernel-evolution snapshot, not a current performance claim.
+End-to-end via `ab_zgemm_blas` (BLAS-ABI path): all rows below lose to AMX at all measured shapes through N=2048 regardless of the kernel-vs-kernel delta they record.
+
 | Size | GPU | AMX | Winner |
 |------|-----|-----|--------|
 | 1024 | 12.3ms | 15.1ms | **GPU +23%** |
@@ -680,13 +724,20 @@

 ### Routing decision

-| Operation | Route | When |
-|-----------|-------|------|
-| DGEMM | GPU | max(M,N,K) ≥ 1024 |
-| ZGEMM | GPU (Gauss 3-multiply) | max(M,N,K) ≥ 1024 |
-| DSYRK/ZHERK | AMX always | Conversion overhead |
-| Everything else | AMX | — |
+Dispatch is gated by **two independent conditions**, both of which must hold for GPU dispatch:

+| Gate | Condition | Env override |
+|---|---|---|
+| Min-dim floor | `min(M, N, K) ≥ AB_MIN_GPU_DIM` (default 32) | `AB_MIN_GPU_DIM=<n>` — set to `0` for diagnostic forced-GPU only |
+| FLOP threshold — DGEMM, all dims ≥ 64 | `2·M·N·K ≥ DEFAULT_CROSSOVER_FLOPS_REAL` (default 50 MFLOPs) | `AB_CROSSOVER_FLOPS_REAL=<flops>` |
+| FLOP threshold — ZGEMM, OR any dim < 64 (skinny path) | `8·M·N·K ≥ DEFAULT_CROSSOVER_FLOPS` (default 100 MFLOPs) | `AB_CROSSOVER_FLOPS=<flops>` |
+
+**`AB_MODE=gpu` does not bypass the min-dim floor.** To force GPU on sub-32 shapes for diagnostic measurement, set `AB_MIN_GPU_DIM=0` in addition. Production callers should not override the floor — sub-32 shapes are always slower on GPU than on CPU, regardless of regime, due to launch and conversion overhead.
+
+**Regime caveat.** These thresholds are calibrated to the steady-state crossover (native ABMatrix API with persistent handles). They do **not** predict the end-to-end crossover for callers using the BLAS-ABI path (`ab_*_blas`) without handle caching. In that regime there is no crossover in the measured grid through N=2048, and `AB_MODE=cpu` is the right default. See §0 for the full regime distinction and `docs/review/REVIEW_04_TRANCHE_A_2026-04-24.md` for empirical grounding.
+
+**Authoritative source.** `src/blas_wrapper.c::ab_use_gpu()` and `ab_use_gpu_complex()`. If this table disagrees with the code, **the code is right** — file a correction to the skill.
+
 ### Tricks results (ULP Fiction paper)
```

## Per-edit rationale

### Edit 1 — Skill `§0 Two Regimes — Read This First` (insert at L9)

**What:** New top-level subsection inserted between the file header and the
ToC, before any performance number appears in the document. Establishes the
steady-state vs end-to-end regime distinction explicitly, with a
recommendation matrix and pointers to anchor commits/CSVs.

**Why:** REVIEW_04 Tranche A established that the same library produces
two operationally different performance regimes depending on whether
callers retain `ABMatrix` handles. Every prior performance number in the
skill was implicitly steady-state by lineage but not labeled as such.
A future reader skimming the skill for "what GFLOP/s should I expect
in QE / Yambo" would arrive at the wrong number without this subsection.

**Cross-refs:** Empirical grounding from commits `19130d5` (sweep CSVs)
and `1b70258` (R4-2b localization). The commit reference inside the
new subsection (`docs/review/SKILL_EDITS_2026-04-25.md`) is this file.

### Edit 2 — Skill ToC (L12)

**What:** Inserted `0. [Two Regimes — Read This First](#0-two-regimes--read-this-first)`
as the first numbered ToC entry.

**Why:** Skimming readers should see §0 before scanning past the ToC.
GitHub-flavored markdown anchor slug is `0-two-regimes--read-this-first`
(em-dash collapses to two hyphens because the surrounding spaces also
become hyphens). Existing entries 1-18 are unchanged.

### Edit 3 — Skill §8 "What works, what doesn't" table label (L444)

**What:** Inserted a single bold-prefix line above the existing table:
`**Regime: steady-state (native ABMatrix API, persistent handles).** …
For end-to-end BLAS-ABI behavior see §0.`

**Why:** The 7 data rows in the table characterize kernel-only steady-state
throughput (DD vs AMX, Strassen, etc.). Without a regime label, readers
infer "640 GFLOP/s production" applies to QE-via-Fortran-`dgemm_`. It
doesn't.

**Cross-ref:** Steady-state numbers anchored to
`benchmarks/results/2026-04-22-b9b0641/dgemm.csv` (committed in earlier
sprints).

### Edit 4 — Skill §8 43.3% utilization paragraph (L456 pre-edit, L489 post-edit)

**What:** (a) prepend `In the steady-state regime [§0], ` to the existing
paragraph, (b) append a sentence quantifying end-to-end utilization at the
same shape: ~1.3% effective FP32 utilization (20 GFLOP/s DP wall × k'=9 /
13.6 TFLOP/s peak FP32 = 1.32%).

**Why:** The 43.3% figure is the FP32_UTILIZATION.md derivation, which is
explicit about being a floor on sustained-boost steady-state compute. The
end-to-end equivalent is two orders of magnitude lower, dominated by
conversion overhead — not a property of the kernel. Reader needs both
numbers, one phrase apart, to see the regime gap.

**Cross-refs:** `docs/design/FP32_UTILIZATION.md` for the steady-state
derivation; `docs/review/REVIEW_04_TRANCHE_A_2026-04-24.md` for the
end-to-end measurement (DGEMM 2048³ at 20 GFLOP/s in
`dgemm_auto_sweep1.csv` / `dgemm_auto_sweep2.csv`).

### Edit 5 — Skill §8 Gauss ZGEMM paragraph rewrite (L462 pre-edit, L495 post-edit)

**What:** Full three-paragraph rewrite of the +14–33% paragraph:
(1) restated +14–33% as steady-state-regime kernel-vs-kernel delta with
historical-table cross-ref to §16, (2) added an end-to-end paragraph
naming the AMX win at all measured shapes through N=2048, (3) made the
two-gate (min_dim AND FLOP threshold) routing rule explicit with pointer
to §16 for the full table.

**Why:** Pre-edit text said "Dispatcher selects GPU by FLOP threshold,
not by raw N" — directionally right but undertells. Actual dispatch
involves min_dim floor that outranks `AB_MODE=gpu` for sub-32 shapes
(established by REVIEW_04 Phase 1 reading of `src/blas_wrapper.c`),
and the FLOP threshold itself differs by op class. Plus the regime
conflation is the load-bearing fix.

**Cross-refs:** Steady-state +14–33% from `benchmarks/results/2026-04-22-*`;
end-to-end 15–40× loss from `benchmarks/results/2026-04-24-271a0f0-rect/`;
two-gate rule from `src/blas_wrapper.c::ab_use_gpu*()` (commit `271a0f0`
introduced the dispatch-path hook used by `bench_rect_*` to validate this
on every call).

### Edit 6 — Skill §16 DD-DGEMM table label (L666 pre-edit, L702 post-edit)

**What:** Single-line steady-state regime label above the existing
table, with end-to-end-equivalents-25-40×-lower forward pointer.

**Why:** Same as Edit 3 but for the §16 reference table. Numbers in the
table (551, 637, 648 GFLOP/s) are steady-state lineage from
2026-04-22 benchmarks.

### Edit 7 — Skill §16 Gauss-ZGEMM historical-table label (L674 pre-edit, L713 post-edit)

**What:** Two-line label above the table identifying the +23/+35/+43%
rows as a 2025–early-2026 kernel-evolution snapshot (steady-state),
with explicit pointer to the current +14–33% envelope and an end-to-end
caveat.

**Why:** This is the table the L462 text already referenced as "see table
~L678 is historical." The label makes the historical-vs-current
distinction self-contained at the table site, and adds the regime
caveat for the +23/+35/+43% rows themselves.

### Edit 8 — Skill §16 Routing decision rewrite (L683-689 pre-edit, L725-743 post-edit)

**What:** Full replacement. Old 4-row Operation/Route/When table
(claiming `max(M,N,K) ≥ 1024` for both DGEMM and ZGEMM, "AMX always"
for DSYRK/ZHERK) replaced by a 3-row Gate table that states the
actual two-condition dispatch rule, plus an `AB_MODE=gpu` clarifier,
a regime caveat, and an "authoritative source" pointer to the code.

**Why:** Old rule was wrong on three counts: (1) routing is FLOP-based not
N-based, (2) DGEMM and ZGEMM have different FLOP thresholds (50M vs
100M), (3) the implicit min-dim floor that outranks `AB_MODE=gpu` for
sub-32 shapes was undocumented. The dropped DSYRK/ZHERK row was
inconsistent with `apple_bottom.h` even pre-this-commit (header showed
GPU crossovers at N≥3072 / N≥1024 for those ops; routing table claimed
"AMX always") — drop is a correction, not just a refactor.

**Cross-ref:** `src/blas_wrapper.c::ab_use_gpu()` and `ab_use_gpu_complex()`
(authoritative). `bench_rect_zgemm` smoke test at N=24 (qe_si8_ovlp,
sub-32) under `AB_MODE=gpu` confirmed the floor outranks the mode
(routed cpu) — that was the key empirical demonstration in the
2026-04-24 sweep.

### Edit 9 — `include/apple_bottom.h` L16-21

**What:** Replace 4-line "GPU wins for N >= K" Performance Notes block
with a two-regime block: 4 steady-state lines (DGEMM/ZGEMM/DSYRK/ZHERK
with anchor CSV references for the directly-measured ones and "prior
measurement, not re-measured" / v1.4 historical notes for DSYRK/ZHERK)
+ 5-line end-to-end paragraph naming the no-crossover finding,
~140 MB/s root cause, AB_MODE=cpu recommendation, and CSV anchor.
Style preserved as `//` line comments to match the surrounding block
(L1-26 is one continuous `//`-comment header).

**Why:** Same conflation as the skill, in the public API header. Anyone
reading `apple_bottom.h` to decide how to integrate apple-bottom into
a host code arrives at the wrong conclusion without this. DSYRK and
ZHERK numbers preserved at 3072 / 1024 (not silently downgraded to
"lineage from DGEMM/ZGEMM") with an explicit "not re-measured for
end-to-end regime" qualifier.

**Cross-ref:** Same anchor commit chain. v1.4 historical note retained
on the ZHERK line.

## Re-application steps

If the skill is re-materialized by Claude Desktop and the §0 + edits
are lost, reapply in this order. All commands assume `bash`/`zsh` on
macOS with the runtime path resolvable.

```bash
SKILL='/Users/grantheileman/Library/Application Support/Claude/local-agent-mode-sessions/skills-plugin/10b74ea8-ba14-48d5-8080-0dfbed724ed1/3fbd1047-405e-4e89-a8f2-2ef726867034/skills/apple-bottom-qe/CLAUDE.md'

# 0. Verify pre-edit md5 — abort if file has drifted
[ "$(md5 -q "$SKILL")" = "ab69c1d154509b6ad4b7ab9cedbe0dfe" ] || \
  { echo "skill drifted; reconcile manually"; exit 1; }

# 1. Backup
cp "$SKILL" "$SKILL.pre-commit2"

# 2. Apply the embedded skill diff above. The diff is in standard unified
#    format with hunks in original-file coordinates against md5
#    ab69c1d154509b6ad4b7ab9cedbe0dfe; standard `patch -p1` will work
#    against the diff above with -p0 if line numbers and context are
#    preserved. The diff blocks above can be saved to /tmp/skill.patch and
#    applied via:
#       patch "$SKILL" < /tmp/skill.patch
#    No subdirectory wrappers; -p0 form.

# 3. Verify post-edit md5
[ "$(md5 -q "$SKILL")" = "2eefd99009cca50bd33a99e340edf0a4" ] || \
  { echo "post-edit md5 mismatch; verify manually"; exit 1; }

# 4. Verify §0 + ToC entry present (sanity)
grep -q "^## 0. Two Regimes — Read This First" "$SKILL"
grep -q "^0. \[Two Regimes" "$SKILL"
grep -c "Regime: steady-state" "$SKILL"   # expect 3 occurrences

# 5. apple_bottom.h is git-tracked; re-apply via git checkout if needed:
#       git checkout <this-commit-sha> -- include/apple_bottom.h
```

## Verification

- **`make clean && make test`** post-edit: 113/113 tests pass
  (test_precision + test_correctness + test_device_api).
- **AB_PROFILE smoke** (zero-cost when off): probe binary built against
  post-edit `apple_bottom.h` runs without `AB_PROFILE` env var produces
  no profile output (env-gated path remains off, single cached env read
  + branch in `ab_zgemm_blas`).
- **Skill md5 transition** verified: `ab69c1d…` → `2eefd99…`.
- **`apple_bottom.h` md5 transition** verified: `78efab2…` → `b8d974e…`.

## Out of scope this commit

Deferred to REVIEW_05 (separate concern, separate commit):

- **Section 1 Repository Layout paths.** Skill §1 references stale paths
  (`~/Dev/arm/metal-algos`, etc.) that no longer reflect the current
  ecosystem layout. Not touched in this commit per explicit scope
  instruction.
