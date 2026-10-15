# Quantum ESPRESSO validation runs

The inputs and scripts behind [VAL-1](../../tests/validation/VAL001_QE_Si64.md):
the same pw.x SCF run with `AB_MODE=cpu`, `auto` and `gpu`, repeated,
with the energies, the QE timers and apple-bottom's `AB_REPORT` of every run.

| File | |
|---|---|
| `inputs/*.in` | the three si64 inputs (placeholders `OUTDIR` and `PSEUDO_DIR` are filled in per run) |
| `run_campaign.sh` | runs a plan of inputs × modes × repetitions and writes `results.csv` |
| `analyze.py` | per-run and per-configuration summary: wall time, BLAS time by route (summed over MPI ranks), GPU stage costs, and the ratio of GPU kernel time to Accelerate's time for the same calls |
| `compare.py` | compares every run with the `cpu` run of the same input: final and per-iteration energies, every printed band energy |
| `tables.py` | the VAL-1 tables (accuracy, routing, run time, GPU stages) in Markdown |

## Inputs

64-atom silicon (2 × 2 × 2 conventional cells), PBE, PAW, `ecutwfc = 25` Ry,
`conv_thr = 1e-6` Ry, `verbosity = 'high'` (so band energies are printed).

| Input | Bands | k-points | Pseudopotential | Davidson GEMMs |
|---|---|---|---|---|
| `A_si64_500b_k222` | 500 | 2 × 2 × 2 (4 irreducible) | `Si.pbe-nl-kjpaw_psl.1.0.0.UPF` | complex (`cegterg`) |
| `B_si64_160b_k222` | 160 | 2 × 2 × 2 | `Si.pbe-n-kjpaw_psl.1.0.0.UPF` | complex (`cegterg`) |
| `C_si64_160b_gamma` | 160 | Γ only | `Si.pbe-n-kjpaw_psl.1.0.0.UPF` | real (`regterg`) |

The pseudopotentials are from pslibrary 1.0.0 (A. Dal Corso), as
distributed with Quantum ESPRESSO's pseudopotential library; they are not
in this repository. Put them in `benchmarks/qe/pseudo/` or point
`PSEUDO_DIR` at them. The files used on 2026-10-14 (SHA-256):

```
4fb30411d7d95fe5c171720cb049d8b20752a72f46380ecfb25021e1d696bad0  Si.pbe-nl-kjpaw_psl.1.0.0.UPF
8f630aadb55097e3b456a4a7c49b0aabafb5c4b8ef586ff15fe247f98361d39e  Si.pbe-n-kjpaw_psl.1.0.0.UPF
```

## Running

pw.x must load libapplebottom; see the
[Quantum ESPRESSO section](../../docs/INTEGRATION.md#quantum-espresso-integration)
of the integration guide. Then, on AC power with nothing else on the GPU:

```bash
cat > plan.txt <<'EOF'
# label  input              np omp reps modes
1.4.0    A_si64_500b_k222    1  8   3   cpu,auto,gpu
EOF
PW_X=/path/to/pw.x caffeinate -is benchmarks/qe/run_campaign.sh plan.txt /path/to/out
python3 benchmarks/qe/analyze.py /path/to/out
python3 benchmarks/qe/compare.py /path/to/out
python3 benchmarks/qe/tables.py /path/to/out 1.4.0
```

A plan line may end with extra pw.x arguments, e.g. `-nk 4` for k-point
pools, and `VAR=value` environment settings (see the header of
`run_campaign.sh`). With more than one rank, the runner sets each rank's
`OPENBLAS_NUM_THREADS` and `VECLIB_MAXIMUM_THREADS` to its OpenMP threads
unless the plan line sets them; a single rank keeps the libraries'
defaults. Without that, eight ranks each start their own Accelerate
threads: input A at 8 ranks × 1 thread with `-nk 4` (the QE build of the
2026-10-15 results) took 110 s uncapped and 101 s capped (one run each). `LOCK_CMD`, if set, prefixes each `mpirun`, e.g. with a machine-lock
wrapper.

The scripts need only the Python 3 that ships with macOS (3.9 or later).

Input A takes about 8.5 minutes per run in `cpu` mode on an M2 Max with
VAL-1's `-O0` QE build (`np = 1`, 8 OpenMP threads). With QE compiled `-O3`
it takes about 5 minutes at 1 rank × 1 thread and 1.6 minutes at 8 ranks
× 1 thread with `-nk 4`. Keep one OpenMP thread per rank with that build;
see the 2026-10-15 results.

## Results

- 2026-10-14, library 1dbdb52 (1.4.0 candidate) and eaad87e (main before
  1.4.0): [`../results/2026-10-14-1dbdb52-qe-si64/`](../results/2026-10-14-1dbdb52-qe-si64/)
- 2026-10-15, library c5a1fa6 (1.4.0), QE compiled `-O3`, 1 rank and
  8 ranks: [`../results/2026-10-15-c5a1fa6-qe-si64-opt/`](../results/2026-10-15-c5a1fa6-qe-si64-opt/)
