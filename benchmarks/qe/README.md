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

## QE builds

The campaigns here (VAL-1 and the results below) used QE develop
5b7f8e1bc with `cegterg.f90` and `regterg.f90` calling `ab_zgemm` and
`ab_dgemm` in place of `ZGEMM` and `DGEMM` (12 calls each). Those patches
are not needed: linking apple-bottom already sends the calls to it
([integration guide](../../docs/INTEGRATION.md#quantum-espresso-integration)).
A control on 2026-10-18 built one fresh QE develop checkout (4a626b7ce)
three ways: upstream with OpenBLAS, the same objects relinked with
apple-bottom 1.4.0 (`c5a1fa6`), and that plus the patches. It ran input B's
geometry with `conv_thr` 1e-9 at one rank. Three runs matched in every SCF
iteration, eigenvalue, force and stress: `cpu` mode with and without the
patches, and `gpu` mode with them. The library received the same calls with
and without the patches. The total energy was the same in every run; single
printed energy terms moved by at most 2e-8 Ry.

## Replay

`benchmarks/replay.c` replays the BLAS calls of one traced pw.x run, rank
by rank, with QE's CPU time between them, so the GPU path can be timed on
QE's exact call mix without QE, and also without its host costs.

**Capture.** Add `AB_TRACE=trace` to a plan line's arguments. Every rank
then writes `trace.<pid>` into its run directory, one line per BLAS call
(see "Per-call trace" in [`docs/INTEGRATION.md`](../../docs/INTEGRATION.md)):

```
# label        input              np omp reps modes    args
v15tr-np8nk4   A_si64_500b_k222    8  1   1    cpu,gpu  -nk 4 AB_TRACE=trace
```

**Replay and summarise**, on a machine with nothing else running:

```bash
make bench-replay
R=/path/to/out/runs/v15tr-np8nk4/A_si64_500b_k222
for m in cpu gpu nohost; do ./build/replay $R/cpu_r1 --mode $m --pools 4; done
python3 benchmarks/replay_summary.py $R/cpu_r1 $R/gpu_r1 $R/cpu_r1/replay_*
```

```
replay <trace-dir> --mode cpu|gpu|nohost [--pools P] [--gap-scale X] [--no-gaps]
       [--timeout S] [--out DIR]
```

The header line `# mpi_rank=R mpi_size=S` of each trace file gives its
rank; a directory with a single file without one is a run without MPI.
Before the workers start, `replay` prints each rank's calls, pool
barriers, reductions left without one (`skipped`) and the traced pool waits
taken out of its gaps (`removed_s`).

- **Processes.** One worker per rank, each a `posix_spawn()`ed copy of
  `replay`: a `fork()`ed child cannot reach Metal's shader compiler. The
  parent uses neither Metal nor Accelerate; it hands the workers the calls
  and the barriers in POSIX shared memory. A worker runs with
  `AB_MODE=cpu` (`--mode cpu`) or `gpu` (`gpu`, `nohost`), the
  `AB_MIN_GPU_DIM` of the trace's header (the routing's default for a
  trace without it) and `AB_TRACE=<out>/trace`; with more than one rank,
  also `VECLIB_MAXIMUM_THREADS=1` and `OPENBLAS_NUM_THREADS=1` unless they
  are set, as `run_campaign.sh` gives each rank at one OpenMP thread. A
  single rank keeps Accelerate's default threads. In `gpu` and `nohost`
  modes a worker sets up Metal (`ab_init()`) before the start. The workers
  start together.
- **Modes.**
  - `cpu`, `gpu`: every call through `dgemm_`/`zgemm_`, with the traced
    TRANS, M, N, K, leading dimensions, alpha and beta, in that
    `AB_MODE`. The library routes them as it routed QE's: calls with a
    dimension ≤ `AB_MIN_GPU_DIM` or alpha = 0 stay on Accelerate.
  - `nohost`: the calls `AB_MODE=gpu` sends to the GPU, as the library's
    own routing decides it in the worker, run as the native
    `ab_dgemm_scaled()`/`ab_zgemm()` on operands resident on the GPU. Each
    call skips the allocation, pack, upload, download and epilogue.
    `ab_zgemm()` takes its four GPU temporaries from the buffer pool
    (`temps` in "other"). The other calls go through `dgemm_`/`zgemm_`,
    which keep them on Accelerate.
- **Operands.** Contents do not change the timing. Each worker fills one
  buffer each for A, B and C, sized for its largest call, and every call
  uses them with its own leading dimensions. Buffers per distinct shape
  would take 13–17 GB per rank for input A at 8 ranks. `nohost` uploads
  one GPU matrix per operand part once and gives it each call's shape
  through the memory-pool API.
- **CPU time between calls.** Before each call a worker spins for the
  traced gap since its previous call. Before its first call it spins for
  its traced offset from the earliest rank. `--gap-scale X` scales the
  gaps; `--no-gaps` drops them.
- **Pools.** `--pools` is QE's `-nk`: ranks r / (N/P) form a pool. QE sums
  the result of a call with transA `C` or `T` and transB `N` over the pool
  (`mp_sum`). The pool's workers meet at a barrier after such a call where
  the trace shows the whole pool waited there.
  - **Which calls.** Every rank of the pool made the same call (routine,
    TRANS classes, M and N; K is the rank's share), and no rank started its
    next call before the last of them ended. Then the barriers come in the
    same order on every rank. Ranks may make different numbers of such
    calls. In the `cpu` trace of input C at 8 ranks, even ranks make 979
    and odd ranks 897, as the even ranks also run the subspace
    diagonalization; its reductions get no barrier.
  - **PAW's per-atom DGEMMs** (`'C','N'` or `'T','N'`, nh × nh: 8 × 8 for
    input A, 18 × 18 for C) are summed only after the loop over atoms.
    Ranks going through their atoms side by side still often fit the rule:
    1872 of the 2048 in the `cpu` trace of input A at 8 × 1 `-nk 4` get a
    barrier.
    Each rank reaches such a barrier when the trace has it there and less
    of the next gap is waited, so the barrier only matters where the
    replay's ranks drift apart from the trace's timing.
  - **Gaps after a barrier.** In the trace, the gap after a barrier call
    includes the rank's wait for the last rank of its pool. The barrier
    now does that wait, so it is taken out of the gap.
  - **Failures.** A worker waits at most `--timeout` seconds (default 300)
    for the others, at the start and at each barrier, so a hung worker
    fails its partners. When a worker fails or dies, `replay` names it,
    kills the others and writes no summary.
- **Output** goes to `<trace-dir>/replay_<mode>/`, or `--out`. Each worker
  writes `trace.<pid>`. `replay.txt` is `key=value` tokens: a line of
  settings (`source`, `mode`, `ranks`, `pools`, `gap_scale`,
  `min_gpu_dim`, `source_mode`), then a line per rank that maps it to its
  worker's `pid` and holds the replay's wall time (first call start to last
  call end), call time, gaps, offset and barrier waits, the barrier counts
  and the traced wall time. `nohost` workers trace their native calls
  themselves as `path=gpu`, with every host stage zero and the kernel stage
  as the BLAS-ABI layer stages a GPU call's.
- **`replay_summary.py`** groups its arguments by input. For each group it
  prints Markdown tables of the traced runs and the replays:
  - wall time (longest rank) and the sums over ranks of call, Accelerate,
    GPU-call and barrier time;
  - GPU-busy time (the union of every rank's GPU intervals) and the
    kernel time per GPU call;
  - the GPU stages with "other" split.

  It then compares each replay with the traced run of its mode (`nohost`
  with `gpu`): the kernel and Accelerate sums (target ±15%), the replay's
  `gpu` − `cpu` wall difference against the traced runs' (±30%), and the
  `nohost` replay's wall against the `cpu` replay's. pw.x's own wall time
  is printed when a `results.csv` of `run_campaign.sh` lists the run.

**Limits.**
- Spinning keeps a core busy but moves no memory, while QE's own work
  between calls (FFTs, mostly) loads memory as the other ranks' calls run;
  that is the likely reason for what one `cpu` replay of the `cpu` trace
  of input A at 8 × 1 `-nk 4` measured. Accelerate's large `'C','N'`
  ZGEMMs took 0.74× their time in QE and the large `'N','N'` ones 0.90×.
  The calls on Accelerate summed to 0.83× QE's (208.0 against 249.7 s
  over the ranks), and the replay's wall was 93.2 s against the trace's
  95.6 s (pw.x: 96.3 s).
- A gap is fixed time. It holds QE's own work but also waits that follow
  no BLAS call (reductions of other data, I/O). A replay cannot move those
  waits when the GPU path changes the ranks' timing, so confirm an 8-rank
  result with QE itself.
- Reused, pre-touched operands skip the page faults a call makes when QE
  writes a fresh array. Cache state differs too.
- Calls of one rank that overlapped in the trace (several threads) are
  replayed one after another, with a warning.

`make test-replay` (not part of `make test`; it starts GPU work in several
processes) runs `tests/test_replay.py`. The test writes a synthetic 2-rank
trace, replays it in every mode, and checks:
- call counts, arguments and paths;
- wall = calls + gaps + barrier waits;
- the gaps, gap scaling and barriers;
- that the routing follows the trace header's `AB_MIN_GPU_DIM`;
- that a killed or hung worker stops the replay.

It also runs `replay_summary.py`.

## Results

- 2026-10-14, library 1dbdb52 (1.4.0 candidate) and eaad87e (main before
  1.4.0): [`../results/2026-10-14-1dbdb52-qe-si64/`](../results/2026-10-14-1dbdb52-qe-si64/)
- 2026-10-15, library c5a1fa6 (1.4.0), QE compiled `-O3`, 1 rank and
  8 ranks: [`../results/2026-10-15-c5a1fa6-qe-si64-opt/`](../results/2026-10-15-c5a1fa6-qe-si64-opt/)
- 2026-10-26 to 10-28, library 415cb5d (the 1.5.0 release check) and
  c5a1fa6 (1.4.0) on one fresh QE develop build (86c214974), inputs A and
  B, 1 rank and 8 ranks: [`../results/2026-10-28-415cb5d-qe-si64-release-check/`](../results/2026-10-28-415cb5d-qe-si64-release-check/)
