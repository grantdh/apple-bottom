# VAL-1: Quantum ESPRESSO Si64 SCF

| | |
|---|---|
| **Test ID** | VAL-1, revision 2 |
| **Type** | Code-to-code validation inside one Quantum ESPRESSO build: DD GEMMs on the GPU against FP64 GEMMs on the CPU |
| **Date** | 2026-10-14 |
| **Library** | apple-bottom 1dbdb52 (the 1.4.0 candidate; `ab_build_info()` reports `1dbdb52abeec`), and eaad87e (main before 1.4.0) for comparison |
| **Status** | Accuracy: **PASS**. Routing: **PASS**. Performance: the GPU path is slower than Accelerate (reported, not a criterion) |
| **Data** | [`benchmarks/results/2026-10-14-1dbdb52-qe-si64/`](../../benchmarks/results/2026-10-14-1dbdb52-qe-si64/) |
| **Kit** | [`benchmarks/qe/`](../../benchmarks/qe/) (inputs, `run_campaign.sh`, `analyze.py`, `compare.py`, `tables.py`) |

## 1. Questions

1. **Accuracy.** With the SCF's ZGEMMs and DGEMMs computed in double-float
   arithmetic on the GPU, does Quantum ESPRESSO produce the same result as
   with FP64 on the CPU?
2. **Routing.** Does the default, `AB_MODE=auto`, send QE's calls where the
   calibration says it should (for these shapes on an M2 Max: Accelerate)?
3. **Cost.** What does the GPU path cost end to end, and where does its time
   go?

## 2. Revision 1 is withdrawn

Revision 1 (2026-03-31) reported energy agreement "to 11 decimal places"
and a 1.22× speed-up over 6-thread OpenBLAS. Neither measured the GPU.
Versions 1.0–1.3 created the Metal context only in `ab_init()`, which no
QE build called, so every BLAS-ABI call ran on the CPU; agreement between
two CPU runs says nothing about DD arithmetic. The 1.22× compared a serial
QE build with an MPI build. Since 1.4.0 the BLAS ABI initializes the GPU on
first use, `AB_REPORT` records where each call ran and how long it took,
and this revision checks that the calls did run where the mode says.

## 3. Setup

| | |
|---|---|
| Machine | MacBook Pro (Mac14,6), Apple M2 Max: 8 performance + 4 efficiency cores, 38-core GPU, 64 GB; on AC power, `caffeinate -is`, no other GPU work |
| OS, toolchain | macOS 27.0.1 (26A434); Apple clang 17; GNU Fortran 15.2.0; Open MPI 5.0.9 |
| Quantum ESPRESSO | develop at 5b7f8e1bc (2026-04-03, `qe-7.5CN-26`; reports 7.5), built with gfortran 15.2 and Open MPI. **QE's own Fortran was compiled `-O0 -g -fbounds-check`** (all objects in PW, FFTXlib, Modules, upflib, XClib, LAXlib and UtilXlib, built 2026-04-04 before `make.inc` was changed to `-O3`), except the two patched Davidson files, `cegterg` and `regterg` (`-O3 -fopenmp`). The libraries it calls (Accelerate, OpenBLAS 0.3.31, FFTW 3 without threads, ScaLAPACK) are optimized. In this build `cegterg`/`regterg` call `ab_zgemm`/`ab_dgemm` and all other code calls `ZGEMM`/`DGEMM`; both names forward to the same `ab_*gemm_blas`, so an unpatched QE linked as in the integration guide routes every GEMM the same way |
| Linking | The link is flat-namespace (Open MPI's wrapper adds `-flat_namespace`). At run time `dgemm_`, `zgemm_`, `ab_dgemm_` and `ab_zgemm_` bind to libapplebottom and the other 66 BLAS/LAPACK symbols pw.x imports bind to OpenBLAS (checked with `DYLD_PRINT_BINDINGS`). apple-bottom's CPU route is Accelerate (`cblas_zgemm`/`cblas_dgemm`). |
| Library under test | Each library was built once from a clean worktree of its commit and copied aside (libapplebottom.dylib SHA-256 `b8cd26bdaee1abb3…` for 1dbdb52, `7bbb86408ee571fb…` for eaad87e); pw.x loads it through its rpath, and every run's stderr records the dylib that was loaded |
| Parallelism | `mpirun -np 1`, `OMP_NUM_THREADS=8` |
| Modes | `AB_MODE=cpu` (every GEMM on Accelerate), `auto` (calibrated table), `gpu` (every GEMM with all dimensions > 32 on the GPU) |
| Repetitions | Input A: 3 per mode; B, C: 2 per mode; eaad87e: input A, `auto`, 2. The mode order rotates with each repetition; every run starts from an empty `outdir`. |
| Reports | `AB_REPORT=1` and `AB_PROFILE=1` in every mode (per-call accounting in all three, so its small cost is common to all) |

### Inputs

64-atom silicon (2 × 2 × 2 conventional cells), PBE, PAW (pslibrary 1.0.0),
`ecutwfc = 25` Ry, `conv_thr = 1e-6` Ry, Davidson, `verbosity = 'high'`.

| Input | Bands | k-points | Plane waves | Davidson GEMMs | Why |
|---|---|---|---|---|---|
| A `A_si64_500b_k222` | 500 | 2 × 2 × 2, 4 irreducible | ~18,270 per k | ZGEMM (`cegterg`) | BLAS-heavy: the largest GEMMs and BLAS share of the three |
| B `B_si64_160b_k222` | 160 | 2 × 2 × 2 | 17,845–18,016 per k | ZGEMM (`cegterg`) | smaller Davidson GEMMs (320-wide subspace), more PAW work (6 projectors) |
| C `C_si64_160b_gamma` | 160 | Γ | 8,923 (Γ: half the sphere) | DGEMM (`regterg`) | the real (gamma-point) code path |

## 4. Acceptance criteria

| | Criterion |
|---|---|
| Accuracy | In `gpu` mode, the final total energy is within `conv_thr` (10⁻⁶ Ry) of the `cpu` run, the SCF takes the same number of iterations, and every printed band energy is within 10⁻³ eV |
| Routing | In `auto` mode no GEMM of these inputs runs on the GPU (the built-in table has no row for their shapes); in `gpu` mode the large GEMMs do; no fallbacks |
| Stability | Every run exits with status 0 |

## 5. Results

### 5.1 Accuracy

| Input | Total energy, `cpu` (Ry) | Largest \|ΔE\|, `gpu` vs `cpu`: final / any iteration (Ry) | SCF iterations, `cpu` / `gpu` | Band energies per run | Largest \|Δε\|, any run vs `cpu` (eV) |
|---|---|---|---|---|---|
| A | -2990.44276149 | 0 / 0 | 7 / 7 | 2000 | 0 |
| B | -2990.44093535 | 0 / 0 | 9 / 9 | 640 | 0 |
| C | -2989.94060060 | 0 / 0 | 8 / 8 | 160 | 0 |

Every run of every mode ended with the same total energy as the `cpu`
run, after the same number of SCF iterations, with the same total energy
at every iteration and the same value for every printed band energy. QE
prints total energies to 10⁻⁸ Ry and band energies to 10⁻⁴ eV, so the
agreement is to the last printed digit; the differences in DD arithmetic
are below what the SCF output resolves. **PASS.**

### 5.2 Routing

| Input | Mode | Routine | Calls | On the GPU | GPU share of the routine's time | Fallbacks |
|---|---|---|---:|---:|---:|---:|
| A | `auto` | ZGEMM | 28425 | 0 | 0.0% | 0 |
| A | `auto` | DGEMM | 2064 | 0 | 0.0% | 0 |
| A | `gpu` | ZGEMM | 28425 | 951 | 97.8% | 0 |
| A | `gpu` | DGEMM | 2064 | 16 | 99.9% | 0 |
| B | `auto` | ZGEMM | 29048 | 0 | 0.0% | 0 |
| B | `auto` | DGEMM | 2580 | 0 | 0.0% | 0 |
| B | `gpu` | ZGEMM | 29048 | 1090 | 96.8% | 0 |
| B | `gpu` | DGEMM | 2580 | 20 | 99.8% | 0 |
| C | `auto` | DGEMM | 5624 | 0 | 0.0% | 0 |
| C | `gpu` | DGEMM | 5624 | 263 | 98.8% | 0 |

`auto` ran every GEMM of these inputs on Accelerate, as the calibrated
table says it should; `gpu` sent the large calls to the GPU: fewer than 5%
of the calls, but 89–99% of each routine's time on Accelerate (`cpu` mode). No call fell back. **PASS.**

### 5.3 Run time

| Input | `cpu` | `auto` | `gpu` | `gpu` / `cpu` | ZGEMM+DGEMM time, `cpu` | same, `gpu` | `auto`, eaad87e |
|---|---|---|---|---:|---:|---:|---|
| A | 509.9 [506.5–510.1] | 509.3 [508.0–509.3] | 598.0 [597.7–598.1] | 1.173 | 58.1 | 147.8 | 594.3 [593.3–595.2] |
| B | 1120.9 [1120.1–1121.7] | 1121.3 [1121.2–1121.3] | 1197.7 [1196.4–1199.1] | 1.069 | 30.5 | 107.0 | – |
| C | 748.6 [748.6–748.6] | 748.6 [748.5–748.7] | 758.4 [758.4–758.5] | 1.013 | 2.5 | 12.1 | – |

`auto` costs nothing: its median is within 0.2% of `cpu` for every input.
Forcing the GPU costs 17.3%, 6.9% and 1.3% of the run: the GEMMs take
2.5×, 3.5× and 4.9× as long as on Accelerate, and they are 11%, 2.7% and
0.3% of the `cpu` run (input A is the BLAS-heavy one; in B and C the PAW
routines dominate in this unoptimized build). The spread over repetitions is at most 0.7%. macOS
background services (Spotlight, media analysis) were not stopped.

**These run times belong to this build.** QE's own code was compiled
without optimization and with bounds checking (§3), while the GEMMs run in
optimized libraries. With an optimized QE the rest of the run shrinks, so
the GEMMs take a larger share and forcing the GPU costs a larger fraction
of the run than the 1.3–17.3% above. The accuracy results (§5.1), the
routing (§5.2) and the per-call comparisons (§5.4), which involve only
library code, do not depend on how QE was compiled.

### 5.4 Where the GPU path spends its time

| Input | Routine | GPU calls | init | alloc | pack | upload | kernel | download | epilogue | other | total | Accelerate, same calls (`cpu`) | kernel / Accelerate |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| A | ZGEMM | 951 | 0.00 | 0.17 | 48.12 | 13.80 | 55.01 | 1.77 | 10.62 | 11.62 | 141.16 | 54.80 | 1.00 |
| A | DGEMM | 16 | 0.03 | 0.00 | 1.02 | 0.19 | 1.89 | 0.02 | 0.27 | 0.11 | 3.54 | 0.19 | 10.20 |
| B | ZGEMM | 1090 | 0.00 | 0.18 | 38.08 | 12.69 | 32.13 | 0.83 | 4.65 | 9.01 | 97.58 | 26.44 | 1.22 |
| B | DGEMM | 20 | 0.03 | 0.00 | 1.66 | 0.41 | 2.39 | 0.14 | 1.36 | 0.26 | 6.26 | 0.89 | 2.68 |
| C | DGEMM | 263 | 0.03 | 0.03 | 4.67 | 1.76 | 3.84 | 0.13 | 0.73 | 0.77 | 11.95 | 2.30 | 1.67 |

Stages, as `AB_REPORT` defines them: `pack` builds op(A) and op(B) in
the layout the GPU expects and converts them to DD on the CPU, `upload`
copies them, `kernel` is the command buffers' GPU execution time,
`download` and `epilogue` bring C back and apply alpha and beta, and
`other` is the rest of the kernel call (encoding, waiting, and for ZGEMM
the per-call allocation of four temporary matrices). "Accelerate, same
calls" is the routine's `cpu`-mode time minus the time of the calls that
`gpu` mode also left on the CPU, i.e. what the GPU-routed calls cost on
Accelerate. Input C's DGEMM row holds both its Davidson (`regterg`) calls
and its PAW (`newq`, `addusdens`) calls; A's and B's DGEMM rows are PAW
only.

**The GPU kernels alone take as long as Accelerate needs for the same calls,
or longer:** 1.00× and 1.22× for the ZGEMMs of A and B, 1.67× for C's
DGEMMs, and 2.7–10× for the PAW DGEMMs of A and B. Removing every other
stage would at best bring the GPU path level with Accelerate on these
shapes, on this GPU. By the rule
in the [roadmap](../../docs/ROADMAP.md), there is therefore no performance
work on the BLAS-ABI path for M2-class GPUs.

### 5.5 What the calibrated routing changed

With eaad87e (unreleased main after the correctness fixes that made the BLAS
ABI initialize the GPU on first use, before calibrated routing), `AB_MODE=auto` used fixed FLOP
thresholds and sent the same 951 large ZGEMMs of input A to the GPU as
`gpu` mode does (its profile: 28,425 ZGEMM calls, 3.3% on the GPU, 144.6 s
of ZGEMM time). Input A took 594.3 s [593.3–595.2] with it, against 509.3 s
with the calibrated table: the default was 16.7% slower than it needed to
be. eaad87e predates `AB_REPORT`, so its figures come from its `AB_PROFILE`
table, which has no time-by-route columns.

## 6. Conclusions

1. **DD arithmetic is accurate enough for these SCF runs.** With the calls
   that make up 89–99% of the GEMM work on the GPU, the energies and band energies agree with the
   FP64 CPU run in every printed digit, for the complex (k-point) and real
   (Γ-point) Davidson paths and the PAW DGEMMs.
2. **The calibrated default is right for QE on an M2 Max.** It keeps every
   GEMM on Accelerate and costs nothing; the previous default cost 17% on
   the BLAS-heavy input.
3. **The GPU path is slower for QE's shapes, and not only because of
   overheads.** Its kernels alone are as slow as Accelerate or slower;
   packing, transfers and per-call allocation come on top.
4. **A side finding, fixed after these runs.** In every run that reached
   the GPU, gfortran reported `IEEE_OVERFLOW_FLAG` signalling at exit.
   The flag comes from `MTLCreateSystemDefaultDevice()`, which raises
   `FE_OVERFLOW` on its first call in a process; apple-bottom's own
   arithmetic raises none (a probe that converts and multiplies unit-scale
   data leaves every flag clear). Since 0cf1852, `ab_init()` restores the
   caller's floating-point environment around device creation, and
   `test_init_keeps_host_fp_flags` checks it. The fix changes no
   computed value.

## 7. Reproducing

Build pw.x against apple-bottom as in the
[integration guide](../../docs/INTEGRATION.md#quantum-espresso-integration),
put the two pslibrary files in `benchmarks/qe/pseudo/` and run, on AC power
with nothing else on the GPU:

```bash
cat > plan.txt <<'EOF'
1dbdb52 A_si64_500b_k222  1 8 3 cpu,auto,gpu
1dbdb52 B_si64_160b_k222  1 8 2 cpu,auto,gpu
1dbdb52 C_si64_160b_gamma 1 8 2 cpu,auto,gpu
EOF
PW_X=/path/to/pw.x caffeinate -is benchmarks/qe/run_campaign.sh plan.txt out
python3 benchmarks/qe/analyze.py out
python3 benchmarks/qe/compare.py out
python3 benchmarks/qe/tables.py out 1dbdb52
```

The baseline column of §5.3 needs a second campaign into the same `out`,
with a pw.x that loads the eaad87e build and the plan line
`eaad87e A_si64_500b_k222 1 8 2 auto`; then run `tables.py out 1dbdb52 eaad87e`.

The script sets `AB_REPORT=1` and `AB_PROFILE=1` and records the loaded
library, as the 2026-10-14 runs did; those ran an earlier version of it
with the same environment and output (later changes add CSV columns, make
the time parser work with the Python 3.9 that ships with macOS, and fix the
analysis scripts).

## 8. Limitations

- One machine (M2 Max), one QE build, one system (Si64 PAW) at a modest
  cutoff. Other chips, codes and systems are not covered.
- `np = 1`. With more MPI ranks each rank's GEMMs are smaller, which favours
  Accelerate further.
- QE's own code was compiled at `-O0` with bounds checking (§3), so the
  whole-run times and shares are those of an unoptimized QE; a repetition
  with an optimized build is planned.
- Printed precision bounds the comparison: QE prints total energies to
  10⁻⁸ Ry and band energies to 10⁻⁴ eV.

## 9. Revision history

| Revision | Date | |
|---|---|---|
| 1 | 2026-03-31 | Withdrawn: the GPU path was not exercised, and the timing compared different builds (§2) |
| 2 | 2026-10-14 | Repeated with the GPU path exercised and time-weighted reports; three inputs, three modes, repetitions |
