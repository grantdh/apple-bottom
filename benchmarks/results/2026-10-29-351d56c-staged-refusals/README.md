# Why GPU calls left the staged path, and the fix, on QE si64, 2026-10-29

The 1.5.0 release check
([`2026-10-28-415cb5d-qe-si64-release-check`](../2026-10-28-415cb5d-qe-si64-release-check/))
found that about half of the GPU-routed BLAS calls in Quantum ESPRESSO skipped
the staged path (PR #9) and paid the host path's packing, conversion and
epilogue. This folder gives the reasons, counted by the new `staged` line of
`AB_REPORT`, and measures the fix on the same QE build: the staged path now keeps
calls whose lines the exponent-range guard scales and ZGEMMs that run as 4M, and a
call the guard does not certify goes straight to Accelerate.

| | |
|---|---|
| Machine | MacBook Pro (Mac14,6), M2 Max, 38-core GPU, 64 GB, macOS 27.0.1 (26A434). AC power from the block's first run to its last: 152 samples of `pmset -g batt`, every 20 s ([`power.txt`](power.txt)). |
| QE | The release check's build: develop 86c214974, every Fortran object `-O3 -fopenmp`, Open MPI 5.0.9, `-lfftw3_omp`, one OpenMP thread per rank. Every `pw.x` here is a copy of the release check's 1.5.0 binary with its first `LC_RPATH` rewritten to the library's frozen directory (`tools/freeze.sh`): the same machine code, only the library it loads differs. |
| Libraries | **1.5.0** = 415cb5d, the release check's frozen copy (`apple-bottom 1.4.0 (415cb5d4d888)`; `<1.5.0>` in `results.csv`). **counter** = 2da6a89, 1.5.0 plus the commit that adds the `staged` counts (`<counter>`). **fix** = 351d56c, `release/v1.5.0` 882d2d8 plus the counter and the fix (`<fix>`, `apple-bottom 1.4.0 (351d56ced974)`); its later rebases onto 6985543 and 2ea4077 change no library source file. **serial3m** = an experiment-only build of 415cb5d whose Gauss-3M products run in a serial encoder (`<serial3m>`, `415cb5d4d888-dirty`), never for release. **refactor** = 8ebe73c, the branch's last code commit, made after these measurements and checked in [the last section](#after-the-measurements-the-refactor-8ebe73c) (`<refactor>`); rebased onto 2ea4077 as 935e74d, whose library sources and tests are 8ebe73c's. |
| Inputs, layouts | `B_si64_160b_k222` and `A_si64_500b_k222` from [`benchmarks/qe/`](../../qe/), at 1 × 1 and at 8 × 1 with `-nk 4`, as in the release check. |
| Plan | [`tools/step2.sh`](tools/step2.sh), one exclusive machine lock taken on AC power (`acgate.sh`, `inlock.sh`), 2026-10-29 00:18–01:10 EDT ([`step2-log.txt`](step2-log.txt)): the counter's tests and one traced `gpu` run of B 1 × 1; the fix's `make test`; one traced `gpu` run of B 1 × 1 with the fix (and with 1.5.0); then per input and layout, 1.5.0 and the fix alternating in `gpu` mode twice each and one `cpu` run; then four `gpu` runs of A 8 × 1 for the cause of its kernel time. `run_campaign.sh` sets `AB_REPORT=1 AB_PROFILE=1` in every run. |

## Why calls left the staged path

The counter's traced run of B 1 × 1 ([`breakdown.txt`](breakdown.txt); `staged=` of
each `AB_TRACE` line). Of the 1,110 GPU-routed calls, 598 kept the staged result.
The others, by the first reason the staged path met:

| Reason | Calls | Then | Shapes (M × N × K; npw = 17,845–18,016, one per k-point; nkb = 1,152) |
|---|---:|---|---|
| `four_m`: a ZGEMM whose real and imaginary parts call for 4M (`ab_lines_method`) | 490 | host path, on the GPU | `C`,`N`: 106 nkb × N × npw (96 with N = 160), 50 overlaps N × N × npw (N = 160, 256). `N`,`N`: 334 npw × N × K (84 with N = 160 and K = 320, 64 with N = K = 160, 45 with K = nkb) |
| `four_m` | 2 | Accelerate (`range`) | nkb × 256 × npw (`C`,`N`) |
| `scaled`: the guard scales a line of op(A) or op(B) | 10 | host path, on the GPU | DGEMM 287,330 × 171 × 64 (`N`,`T`) |
| `scaled` | 10 | Accelerate (`range`) | DGEMM 171 × 64 × 287,330 (`C`,`N`), a PAW augmentation product |

No other reason occurred (floating-point control, scalars, buffers, failures, Inf
or NaN). `four_m` and `scaled` are the counter build's reasons on 1.5.0; with the
fix the staged path keeps those calls (`kept_4m` counts the ZGEMMs it keeps by
4M), so neither is a reason any more. Each refused call had already run the
staged path's whole command buffer, conversion and Gauss-3M product included,
before the host path ran it again: in the release check's per-call lines the
refused ZGEMMs took 1.8–2.2× 1.4.0's GPU kernel time for the same shapes at
B 1 × 1, and 2.5–3.2× at A 8 × 1, where the kept ones took 0.84–0.92× (B 1 × 1).

## The fix

- **Scaled lines.** After the pre-passes have gathered the lines' statistics, a
  GPU pass gives each line the host path's power-of-two scale, and fix-up passes,
  dispatched from indirect arguments only for an operand with a scaled line,
  convert its elements again at x · 2⁻ˢᶜᵃˡᵉ, rounded as the host rounds them (a
  DGEMM's C at its row's and column's scales). The epilogue multiplies C back.
  The host checks that the GPU's scales equal its own.
- **4M.** The staged run computes Gauss-3M as before. Where the parts call for
  4M, a second command buffer computes Ai·Br and Ar·Bi into Gauss-3M's two
  product buffers and forms Ci from them as the host path's 4M does (4M's real
  part is Gauss-3M's), then runs the epilogue again.
- **Range.** A call the guard does not certify goes to Accelerate at once: the
  host path's guard decides from the same statistics.

## Results

Per configuration ([`summary.md`](summary.md), from the runs'
[`reports/`](reports/)); BLAS, kernel and host stages summed over ranks and
averaged over the repetitions. Host stages: pack, upload, download and epilogue of
the GPU calls. Staged: calls that kept the staged result by Gauss-3M / by 4M / that
went to Accelerate.

| Input × layout | Library | `gpu` wall, s | `cpu` wall, s | `gpu` / `cpu` | BLAS, s | GPU kernel, s | host stages, s | staged |
|---|---|---|---:|---:|---:|---:|---:|---|
| B 1 × 1 | 1.5.0 | 217.42, 217.37 | 182.49 | 1.19 | 64.2 | 31.8 | 23.9 | counter run: 598 / – / 12, and 500 on the host path |
| B 1 × 1 | fix | **191.33, 191.26** | | **1.05** | 38.3 | 26.6 | 5.5 | 607 / 491 / 12 |
| A 8 × 1 | 1.5.0 | 114.71, 112.19 | 96.64 | 1.17 | 377.7 | 311.9 | 31.0 | |
| A 8 × 1 | fix | **102.69, 102.18** | | **1.06** | 327.1 | 289.6 | 8.6 | 1,294 / 592 / 42 |
| B 8 × 1 | 1.5.0 | 62.41, 60.45 | 54.28 | 1.13 | 214.0 | 143.4 | 36.3 | |
| B 8 × 1 | fix | **54.88, 54.39** | | **1.01** | 187.7 | 146.7 | 9.9 | 1,336 / 922 / 83 |

- **The fix is 1.14× faster at B 1 × 1, 1.11× at A 8 × 1 and 1.12× at B 8 × 1** than
  1.5.0, in alternating runs in one lock. At B 1 × 1 it saves 26.1 s of wall time,
  and the BLAS calls 25.9 s. At 8 ranks it gets better, not worse: the GPU no
  longer runs the refused calls twice.
- **The staged path keeps every call the guard certifies.** At B 1 × 1 that is
  1,098 of the 1,110 GPU-routed calls; the other 12 are the guard's refusals,
  which ran on Accelerate before too. 491 of the ZGEMMs ran as 4M.
- **`gpu` against `cpu` (Accelerate):** 1.05 at B 1 × 1, 1.06 at A 8 × 1 and 1.01 at
  B 8 × 1 in this block, against 1.13–1.19 for 1.5.0. The release check's 1.4.0
  ran A 8 × 1 in 111.9 and 117.0 s on AC; the fix takes 102.2–102.7 s.
- **Energies.** Every run of an input gives the same total energy and SCF
  iterations: B −2990.44093535 Ry (9) at 1 × 1 and −2990.44093534 Ry (9) at 8 × 1,
  A −2990.44276150 Ry (7).

## A 8 × 1: what raised the GPU kernel time per call

The release check found the GPU kernel time of A 8 × 1's ZGEMMs at 1.15× of
Accelerate's for the same calls, against 0.77 for 1.4.0. Kernel time of the same
1,790 ZGEMMs, summed over ranks (one run each for the cause runs; the 1.5.0 and fix
pairs differ by 3.3% and 0.9% between their two runs):

| Run | Path | GPU kernel, s | wall, s |
|---|---|---:|---:|
| 1.4.0, release check part 4 (AC) | host | 193.4, 181.6 | 116.95, 111.86 |
| 1.5.0, `AB_HOST_PACK=1` | host | 208.5 | 102.77 |
| 1.5.0, `AB_HOST_PACK=1 AB_PREFETCH=0` | host | 225.1 | 103.15 |
| serial3m, `AB_HOST_PACK=1` | host | 207.3 | 102.14 |
| serial3m, `AB_HOST_PACK=1 AB_PREFETCH=0` | host | 215.2 | 103.06 |
| 1.5.0 | staged, refusals redone | 316.1, 305.8 | 114.71, 112.19 |
| fix | staged | 290.0, 287.5 | 102.69, 102.18 |

- **Neither the concurrent Gauss-3M encoder nor the prefetch routing causes it.**
  Running Gauss-3M's three products serially changes the kernel time by −0.6%
  (207.3 against 208.5 s). Turning prefetch off raises it by 4–8%, so prefetch
  lowers it. Neither changes the wall time by more than 1%.
- **The refusals ran on the GPU twice**: 1.5.0's 306–316 s against the fix's
  288–290 s, and 112–115 s of wall time against 102 s.
- **The rest is the staged path itself at 8 ranks**: the conversion and epilogue
  run on the GPU, and the ranks spend less host time between GPU calls, so eight
  ranks' command buffers overlap more and each one's span (`kernel`) includes more
  of the others' work. The fix's 290 s of kernel time give the same wall time as
  the host path's 208 s (102.4 against 102.8 s).
- **Ratios against Accelerate depend on the block's `cpu` run.** Here A 8 × 1 `cpu`
  took 96.6 s (ZGEMM 232 s over ranks), in the release check 104–110 s (ZGEMM
  287–293 s), so the table compares kernel seconds. Against this block's
  Accelerate time for the same calls, kernel ÷ Accelerate is 0.94 for the 1.5.0
  host path, 1.30–1.31 for the fix and 1.38–1.44 for 1.5.0 ([`kernel_ratio.txt`](kernel_ratio.txt)).

## Bit identity

- **`make test` on 351d56c passed:** correctness 79/79, precision 16/16,
  device API, dispatch 9/9, BLAS ABI 1069/1069 (with the exponent-range and
  ZGEMM-imbalance suites), trace 5/5, golden bits 32/32 equal to 1.4.0, and
  `test_staging` 127/127. The staged path gives the host path's bits in every
  case, and now keeps 117 of the 127 (2 take the host path, 8 go to Accelerate),
  against 106 in 1.5.0.
- **QE's own operands are not reproducible run to run,** so the traced runs'
  operand hashes cannot compare libraries: the first BLAS call of B 1 × 1 (the
  augmentation DGEMM above) already reads a different op(B) in two runs of the same
  numerics, 1.5.0 and the counter: its op(A) hashes the same, its op(B) does not
  ([`compare-traces-counter-1.5.0.txt`](compare-traces-counter-1.5.0.txt),
  [`compare-traces-counter-fix.txt`](compare-traces-counter-fix.txt)).
- **On QE's data, call by call, the staged path gives the host path's bits.**
  An experiment-only build of the fix (never for release) with
  `AB_VERIFY_STAGED=1` redid every call the staged path kept on the host path,
  from the same C, and compared C bit for bit ([`verify.txt`](verify.txt),
  [`tools/step3v.sh`](tools/step3v.sh); 2026-10-29 10:46–10:55, AC power): B 1 × 1,
  A 8 × 1 and B 8 × 1 in `gpu` mode, 5,241 calls (5,055 ZGEMMs, 2,005 of them by
  4M, and 186 DGEMMs), no element differs. Their energies equal the timing runs'.
- **`make ci-local`** passed on the committed tree (91accc0): 177 tests, BLAS ABI
  1069/1069.

## After the measurements: the refactor 8ebe73c

The branch's last code commit, 8ebe73c, came after these measurements. It records
one staged outcome per call and encodes the staging passes by operand, and changes
no result by design. After the rebase onto 2ea4077 it is 935e74d, which differs
from it only by the split benchmark's Makefile rule. It was checked again on
2026-10-29 between 20:33 and 21:20, on AC power: [`step4-log.txt`](step4-log.txt),
[`step5-log.txt`](step5-log.txt), [`verify-8ebe73c.txt`](verify-8ebe73c.txt).

- **`make test` and `make ci-local` pass on 8ebe73c.** Golden bits are 32/32
  equal to 1.4.0, `test_staging` is 127/127, and `ci-local` passes all 177 tests,
  with the BLAS ABI cases at 1069/1069.
- **On QE's data the staged path still gives the host path's bits** wherever the
  check can see them. With `AB_VERIFY_STAGED=1`, every kept call whose host-path
  redo ran on the GPU matched bit for bit:
  - 9,373 calls of 8ebe73c: B 1 × 1, B 8 × 1 twice, and the two instrumented
    A 8 × 1 runs;
  - 6,011 calls of 351d56c, run alternately with it.
- **16 kept ZGEMMs differed in (nearly) every element, all at A 8 × 1:** 6 of 8ebe73c
  and 10 of 351d56c, 13 of them the `C`,`N` overlap 500 × 1000 × npw.
  - **The instrumented builds recorded 14 of them** ([`tools/verify_more.py`](tools/verify_more.py)).
    In each, macOS had stopped the GPU work of the host-path redo
    (`kIOGPUCommandBufferCallbackErrorImpactingInteractivity`: GPU work that macOS
    judges to hold up the display). The redo then ran on Accelerate, so the check
    compared FP64 against double-float. The earlier runs, at night and in the
    morning, met no such stop.
  - **The staged results of those calls are accurate:** their normwise error against
    the FP64 result is 0.7–1.4 × 10⁻¹³, as expected of double-float products of
    length K ≈ 9,100.
  - **The other 2 are the uninstrumented first A 8 × 1 run of 8ebe73c,** with the
    same signature: same shape, every element, and a stopped staged command buffer
    in the same run.
- **The staged path met the same stops on its own command buffers:** 1–6 per
  A 8 × 1 run, none at B. It counts each as `gpu_failed` and gives the call to the
  host path, which made it on the GPU. No call ended on Accelerate, and every energy
  equals the timing runs'. No run of the timing block (steps 2 and 3) counted a
  `gpu_failed`.
- **8ebe73c times as the fix did.** Both libraries ran in one lock on AC power,
  alternating, twice each (2026-10-29 22:39–23:03, [`step6-log.txt`](step6-log.txt),
  runs `t6-*` in [`summary.md`](summary.md)):

  | Input × layout | 8ebe73c wall, s | 351d56c wall, s |
  |---|---|---|
  | B 1 × 1 | 192.57, 194.41 | 194.88, 194.11 |
  | A 8 × 1 | 108.84, 107.62 | 106.86, 106.66 |
  | B 8 × 1 | 58.46, 61.32 | 59.67, 63.57 |

  - **At one rank the GPU work is the same.** Kernel and host stages are 26.5 and
    5.6 s against 26.3 and 5.7 s. Each shape's per-call kernel time lies within the
    spread of the fix's own runs: the median of the 3M `vkb^H psi` is 38.3–44.1 ms
    over the fix's four runs (night and evening) and 38.8–44.2 ms over 8ebe73c's two.
  - **At eight ranks the differences point both ways:** A +1.4%, B −2.8%.
  - **That evening every run took 2–13% longer than the night block's,** the fix's
    included. None counted a `gpu_failed`.
  - **The library's build string says `-dirty`** only because this README was
    modified when it was built. No source file was.

## Files

| File | Contents |
|---|---|
| `results.csv` | Every run, as `run_campaign.sh` writes it; frozen libraries as `<1.5.0>`, `<counter>`, `<fix>`, `<serial3m>`, `<refactor>` (8ebe73c). The timing runs and the `t6-*` runs are copied with `tools/collect.py`. |
| `reports/` | Each run's header (run, library, power, energy) and `AB_REPORT` lines. |
| `summary.md` | The per-configuration table above, from `tools/summary.py`. |
| `breakdown.txt` | The counter run's staged outcomes by reason and shape, and the stages the calls paid (`tools/breakdown.py`). |
| `kernel_ratio.txt` | GPU kernel time against Accelerate's for the same calls, per run (`tools/kernel_ratio.py`). |
| `compare-traces-*.txt` | The traced runs compared call by call (`tools/compare_traces.py`). |
| `verify.txt` | The `AB_VERIFY_STAGED=1` runs: calls kept and compared, per routine. |
| `verify-8ebe73c.txt` | The same after the refactor, with the redo's path, each differing call and the failed command buffers (`tools/verify_summary.py`). |
| `power.txt`, `step2-log.txt` | Power during the block; the block's log. |
| `step4-log.txt`, `step5-log.txt`, `step6-log.txt` | The logs of the refactor's checks and timings (`tools/step4.sh`, `tools/step5.sh`, `tools/step6.sh` behind `tools/step-driver.sh`). |
| `plans/`, `tools/` | The plans and scripts (paths under `~`), and the experiment-only patches on top of the measured commits, never for release: `serial-3m.patch` (415cb5d), `verify-staged.patch` (351d56c), `verify-staged-8ebe73c.patch` (8ebe73c) and `verify_more.py` (applied to both). |
