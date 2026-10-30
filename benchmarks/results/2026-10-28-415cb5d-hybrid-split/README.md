# One GEMM on Accelerate and the GPU at once, 2026-10-28/29

Does one GEMM split between Accelerate (the CPU's matrix units) and apple-bottom's GPU path finish sooner
than on either engine alone? With well-scaled operands, it did in all 20 cases measured here:
- **Short runs:** 1.5–1.9× faster than the faster engine alone, with bit-for-bit repeatable results and
  errors within 8 u².
- **80 s of back-to-back work:** the power limit slows the GPU, and DGEMM 8192² keeps 1.63× and ZGEMM
  8192² 1.47× over the GPU alone.
- **QE-like operands:** where the library's exponent-range guard sends the GPU slice to the host path,
  the gain shrinks to 1.23–1.28× or disappears.

Program and how to run it: [`benchmarks/split/`](../../split/README.md) (`bench_split.c`).

| | |
|---|---|
| Machine | MacBook Pro (Mac14,6), Apple M2 Max, 38-core GPU, 64 GB, macOS 27.0.1 (26A434) |
| Library | 415cb5d (release/v1.5.0 at the time), built with `make clean && make lib` from a clean checkout. `ab_build_info()`: `1.4.0 (415cb5d4d888)`, the version string not yet bumped. |
| Settings | `AB_MODE=gpu AB_MIN_GPU_DIM=0 AB_STAGING_MB=8192 AB_REPORT=1`, an `AB_TRACE` file per process, `VECLIB_MAXIMUM_THREADS` 8 (1 process) or 2 (4 processes). Operands registered with `ab_register_host_memory` unless marked "copied". |
| Power | Every case started and ended on AC power (`pmset` in each report and in `power.log`). 2026-10-28: battery 100%, charged. 2026-10-29: battery held at 80%, "AC attached; not charging". The battery rail read +1.5 to +3.1 W throughout. |
| Machine state | Each batch held an exclusive lock that kept every other benchmark job off the machine. Load averages before and after each case are in `power.log`. |
| Data | [`raw/`](raw), one folder per `run.sh` suite; [`traces.tar.xz`](traces.tar.xz) (93 `AB_TRACE` files, 5.0 MB unpacked); [`traces_summary.txt`](traces_summary.txt); [`tables.md`](tables.md), every table below with every case, from `table.py raw`; [`verification/`](verification), the committed program's check (the last section); [`cpu-use/`](cpu-use/README.md), CPU cores in use per mode. |

## Method in brief

- **The split:** the CPU slice (`cblas_[dz]gemm`) runs on the calling thread. The GPU slice
  (`ab_[dz]gemm_blas`, the staged path) runs on a second thread at the same time.
  - Squares and U-N split the columns of C, U-M splits its rows.
  - R-K splits the sum: the GPU's partial sum lands in a separate array and is added after both finish.
- **Layouts:**
  - 1 process × 8 Accelerate threads.
  - 4 processes × 2 threads in lockstep: every repeat starts behind a barrier, and the time is the
    slowest process's. This models four synchronized ranks calling the same GEMM.
- **One case:**
  1. A sweep over the GPU share f, 3 rounds, interleaved.
  2. A fixed block at the sweep's best f: 3 interleaved rounds of Accelerate alone, GPU alone and the
     split, with 9, 9 and 12 repeats per process (6, 6, 12 at 8192²). The tables give the block's
     median [min–max]. Speedups are ratios of medians.
- **Bits:** every split repeat is compared with the first using `memcmp` on all of C, as are both forced
  completion orders and the result hashes of all processes.
- **Error:** max |C − R| / (|op(A)||op(B)| + |βC₀|) on 16 sampled columns, with R from a Dot2 reference,
  in u² = 2⁻⁴⁸ (u = 2⁻²⁴). The library's tests use 8 u² as the long-K bound.
- **Shapes:**
  - U-N, U-M: Quantum ESPRESSO's Davidson updates, `'N','N'` with M = 18,277 plane waves, β = 1.
  - R-K: overlaps, `'C','N'` with K = 18,277 or 18,432, β = 0.
  - The data is synthetic.

## Results: well-scaled operands (uniform in [−1, 1))

Times in ms.

| shape | layout | Accelerate alone | GPU alone | split (f) | vs Accelerate | vs GPU | split ÷ (Acc + GPU) throughput | bits repeat | max error (u²) | power |
|---|---|---|---|---|---|---|---|---|---|---|
| DGEMM 2048² | 1×8 | 28.44 [28.21–29.66] | 26.45 [26.39–34.61] | 15.67 [15.62–18.66] (0.59) | 1.82× | 1.69× | 0.87 | yes, 12/12 | 2.04 | AC |
| DGEMM 4096² | 1×8 | 222.62 [221.75–223.59] | 195.63 [194.60–199.78] | 110.58 [109.19–116.81] (0.55) | 2.01× | 1.77× | 0.94 | yes, 12/12 | 2.56 | AC |
| DGEMM 8192² | 1×8 | 1806.8 [1790.0–1815.4] | 1591.4 [1565.7–1651.8] | 916.7 [867.3–945.3] (0.55) | 1.97× | 1.74× | 0.92 | yes, 12/12 | 2.67 | AC |
| ZGEMM 2048² | 1×8 | 110.38 [109.80–113.20] | 75.23 [74.52–82.23] | 49.56 [48.91–52.06] (0.59) | 2.23× | 1.52× | 0.90 | yes, 12/12 | 3.09 | AC |
| ZGEMM 4096² | 1×8 | 878.86 [876.90–880.21] | 581.69 [576.96–607.21] | 370.93 [367.66–388.78] (0.60) | 2.37× | 1.57× | 0.94 | yes, 12/12 | 2.22 | AC |
| ZGEMM 8192² | 1×8 | 7123 [7062–7195] | 4646 [4586–4806] | 3093 [2943–3265] (0.60) | 2.30× | 1.50× | 0.91 | yes, 12/12 | 4.02 | AC |
| U-N 18277×500×1000 | 4×2 | 470.56 [461.52–497.67] | 423.83 [400.48–444.70] | 222.39 [209.23–239.94] (0.51) | 2.12× | 1.91× | 1.00 | yes, 48/48 | 2.42 | AC |
| U-N 18277×500×512 | 4×2 | 231.11 [225.82–255.94] | 212.81 [208.71–222.42] | 111.01 [106.70–119.85] (0.51) | 2.08× | 1.92× | 1.00 | yes, 48/48 | 3.13 | AC |
| U-M 18277×160×160 | 4×2 | 26.97 [24.79–29.26] | 26.81 [24.89–34.34] | 13.91 [13.33–19.62] (0.50) | 1.94× | 1.93× | 0.97 | yes, 48/48 | 2.80 | AC |
| U-M 18277×160×320 | 4×2 | 45.12 [43.12–47.42] | 45.75 [44.76–53.33] | 24.22 [22.33–30.44] (0.50) | 1.86× | 1.89× | 0.94 | yes, 48/48 | 2.93 | AC |
| U-M 18277×160×1000 | 4×2 | 139.02 [133.74–144.50] | 132.69 [131.34–140.61] | 69.52 [67.44–73.01] (0.50) | 2.00× | 1.91× | 0.98 | yes, 48/48 | 2.61 | AC |
| R-K 500×500×18277 | 4×2 | 241.96 [232.25–249.14] | 210.93 [200.01–223.68] | 115.04 [110.95–122.51] (0.55) | 2.10× | 1.83× | 0.98 | yes, 48/48 | 1.01 | AC |
| R-K 160×160×18432 | 4×2 | 33.19 [31.72–37.80] | 31.44 [30.14–40.38] | 17.57 [17.23–21.84] (0.55) | 1.89× | 1.79× | 0.92 | yes, 48/48 | 1.22 | AC |
| U-N 18277×500×1000 | 1×8 | 122.66 [117.50–129.09] | 101.61 [100.26–107.38] | 59.81 [58.33–64.26] (0.58) | 2.05× | 1.70× | 0.93 | yes, 12/12 | 3.01 | AC |
| U-N 18277×500×512 | 1×8 | 60.13 [58.66–62.31] | 55.76 [54.69–62.25] | 31.76 [30.38–34.05] (0.51) | 1.89× | 1.76× | 0.91 | yes, 12/12 | 3.13 | AC |
| U-M 18277×160×160 | 1×8 | 8.41 [8.22–9.30] | 11.46 [9.14–14.63] | 5.57 [5.10–5.98] (0.40) | 1.51× | 2.06× | 0.87 | yes, 12/12 | 2.69 | AC |
| U-M 18277×160×320 | 1×8 | 13.81 [13.54–13.96] | 13.45 [12.99–20.40] | 7.43 [7.29–13.79] (0.50) | 1.86× | 1.81× | 0.92 | yes, 12/12 | 2.93 | AC |
| U-M 18277×160×1000 | 1×8 | 42.76 [41.64–45.27] | 34.88 [34.38–42.36] | 19.83 [19.70–27.81] (0.55) | 2.16× | 1.76× | 0.97 | yes, 12/12 | 2.61 | AC |
| R-K 500×500×18277 | 1×8 | 62.37 [62.21–63.01] | 53.30 [53.13–60.51] | 30.93 [30.45–34.50] (0.55) | 2.02× | 1.72× | 0.93 | yes, 12/12 | 1.01 | AC |
| R-K 160×160×18432 | 1×8 | 8.76 [8.70–8.82] | 8.43 [8.40–11.66] | 4.85 [4.80–5.33] (0.55) | 1.81× | 1.74× | 0.89 | yes, 12/12 | 1.22 | AC |

- **Throughput:** DGEMM reaches 1.10–1.24 TF/s and ZGEMM 1.39–1.48 TF/s (2MNK and 8MNK flops). The split
  delivers 87–100% of the two engines' throughputs summed.
- **Paths:** all 4,171 GPU-routed calls in these runs took the staged path, as `traces_summary.txt` shows.
- **Bits:**
  - Column splits: the CPU part equals Accelerate alone and the GPU part equals the GPU alone, bit for bit.
  - Row splits: the GPU rows equal the GPU alone. The CPU rows differ from one full Accelerate call in
    31–50% of elements, because Accelerate's blocking depends on M, but they repeat bit for bit.
  - K splits: a different sum, which also repeats.
- **Error:** the CPU parts are at FP64 level (0.06–0.12 u²). The GPU parts are at the GPU path's own
  level (2.0–4.0 u²). The K splits are 1.0–1.2 u².
- **Best share:** at 0.50–0.60 everywhere except U-M K=160 at 1×8 (0.40). One sweep step (±0.05) away
  costs 0–18%.

## Sustained load: 80 s runs, 1 × 8, 8192²

Each run is preceded by 30 s idle (wall 3–5 W, display off). The order is: split at the best share,
GPU alone, Accelerate alone, then the split at the share that balances the first run's steady state
("rebalanced"). "Burst" covers calls starting in the first 10 s; "steady" covers calls starting 20 s
or later; GF/s are medians.

| case | run | GPU part (f) | burst GF/s | steady GF/s | steady wall W | steady GPU MHz |
|---|---|---|---|---|---|---|
| DGEMM 8192² | split | 4512 (0.551) | 1291 | 1135 [1105–1185] | 105.8 | 1233 |
| | GPU alone | – | 720 | 695 [690–721] | 93.1 | 1354 |
| | Accelerate alone | – | 615 | 615 [613–618] | 38.2 | – |
| | split, rebalanced | 4256 (0.520) | 1244 | 1137 [1112–1196] | 100.3 | 1170 |
| ZGEMM 8192² | split | 4928 (0.602) | 1540 | 1320 [1299–1417] | 98.9 | 1161 |
| | GPU alone | – | 962 | 897 [888–910] | 86.4 | 1303 |
| | Accelerate alone | – | 626 | 627 [627–628] | 39.2 | – |
| | split, rebalanced | 4704 (0.574) | 1431 | 1334 [1302–1370] | 96.4 | 1128 |

- **Steady-state speedups:**
  - DGEMM: 1.63× over the GPU alone and 1.85× over Accelerate (rebalanced: 1.64× and 1.85×).
  - ZGEMM: 1.47× and 2.11× (rebalanced: 1.49× and 2.13×).
  - In the first 10 s the ratios were 1.79× and 1.60× over the GPU alone.
- **What happens (every 5 s in each report):**
  - In the first 10 s the split averages 126–127 W at the wall (5 s means up to 131 W), with the GPU
    at 1,387–1,392 MHz.
  - The GPU clock then falls, and from 20 s on averages 1,161–1,233 MHz at 99–106 W. Throughput settles
    12–14% below the burst.
  - Accelerate's throughput does not change. The GPU alone also eases after about 20 s (to
    1,300–1,354 MHz at 86–93 W, 3–7% below its burst).
- **Against the short runs:** the fixed-block ratios at 8192² overstate the sustained ones.
  - Over the GPU alone: 1.74× DGEMM and 1.50× ZGEMM, by about 6% and 2%.
  - Over Accelerate: 1.97× and 2.30×, against 1.85× and 2.11× sustained.
- **Bits:** every checked call in every run (every 5th) equalled the run's first result.

## Wall energy

Steady-state wall power (SMC PDTR, the DC input) times seconds per GEMM:

| case | Accelerate alone | GPU alone | split | split vs Accelerate | split vs GPU |
|---|---|---|---|---|---|
| DGEMM 8192², 1×8, sustained | 68.3 J | 147.3 J | 102.5 J | +50% | −30% |
| ZGEMM 8192², 1×8, sustained | 274.9 J | 423.6 J | 329.4 J | +20% | −22% |
| U-N 18277×500×1000, 4×2, 6 s runs ×3 | 23.2 J | 36.3 J | 26.3 J | +13% | −28% |
| R-K 500×500×18277, 4×2, 6 s runs ×3 | 11.8 J | 18.2 J | 13.8 J | +17% | −24% |

- **Comparison:** the split costs 13–50% more wall energy per GEMM than Accelerate alone and 22–30% less
  than the GPU alone. It draws 2.2–2.8× Accelerate's wall power.
- **What the 4×2 rows include:** the four processes together, and the memory traffic of resetting C
  between calls.
- **Mean wall power at 4×2:**
  - U-N: 53.6 W Accelerate, 87.5 W GPU, 115.5 W split.
  - R-K: 50.1, 85.3 and 116.1 W.
- **Not usable:** the per-call `pdtr_*` fields in the `SUMMARY` lines, and the "power" lines of the
  fixed-block reports. Those calls are mostly shorter than SMC's ~1 s lag; see the notes in
  `benchmarks/split/README.md`.

## Where the split does not win: QE-like operands

The data is synthetic: rows of the plane-wave arrays (A, and C₀ or B) scaled by 10^(−9·i/npwx). It
stresses the exponent-range guard and is not data from Quantum ESPRESSO. All runs are 4×2.

| shape | Accelerate alone | GPU alone | split (f) | vs Accelerate | vs GPU | bits repeat | max error (u²) | GPU calls: staged / host path |
|---|---|---|---|---|---|---|---|---|
| U-N 18277×500×1000 | 446.60 [427.36–462.16] | 819.42 [814.62–900.78] | 347.89 [330.11–363.71] (0.32) | 1.28× | 2.36× | yes, 48/48 | 3.11 | 0 / 344 |
| U-N, shares 0.06–0.32 | 481.08 [462.96–785.55] | 881.55 [846.80–927.41] | 390.64 [347.99–811.36] (0.32) | 1.23× | 2.26× | yes, 48/48 | 3.11 | 0 / 260 |
| U-N 18277×500×512 | 240.66 [233.17–280.27] | 515.53 [444.13–551.31] | 247.52 [221.09–348.32] (0.26) | **0.97×** | 2.08× | yes, 48/48 | 2.40 | 0 / 344 |
| U-N K=512, shares 0.06–0.32 | 231.92 [215.72–250.44] | 479.05 [445.57–511.10] | 230.75 [223.76–272.59] (0.19) | **1.01×** | 2.08× | yes, 48/48 | 3.79 | 0 / 260 |
| R-K 500×500×18277 | 235.52 [232.83–237.64] | 207.65 [198.72–242.80] | 114.82 [114.33–121.66] (0.55) | 2.05× | 1.81× | yes, 48/48 | 0.24 | 315 / 57 |

- **Updates:** the guard scales the rows of op(A) whose largest entry lies below 2⁻²⁰. The staged result
  then does not stand, and every GPU call (whole or slice) redoes the call on the host path.
  - The GPU alone becomes 1.9–2.4× slower than on uniform data.
  - The host path packs all of A whatever the slice, 49–134 ms per call at 4×2. That puts a floor under
    the GPU slice: for U-N K=512 the split only ties Accelerate at its best share (0.19) and loses up to
    10% at any other.
- **Overlaps:** the K split gives the GPU the tail of the sum, the high-G rows with the smallest values.
  - At f ≤ 0.30 all of that tail lies below 2⁻²⁰, and those 56 slices took the host path. At the best
    share (0.55) every slice was staged.
  - One of 64 identical GPU-alone calls also took the host path. Its staged attempt ran no kernel, so
    the cause is something other than the data. Bits are the same either way.
- **Accuracy on this data:** the GPU alone's R-K error is 10.1 u², above the tests' 8 u² long-K bound (1.8 u²
  on uniform data). Accelerate's grows in the same way, 0.06 → 0.46 u². The split's is the lowest,
  0.24 u², because Accelerate sums the dominant low-G head in FP64.
  - A likely mechanism: once the partial sum saturates on the large rows, each of the ~18,000 tail
    additions still rounds the full partial sum. The tests' bound argument assumes partial sums that
    grow like √k.

## Copied instead of registered operands (4×2)

With `SPLIT_NOREG=1` the staged path copies the operands instead of reading them in place.

| shape | split, registered (speedup over the faster engine) | split, copied (speedup over the faster engine) | GPU alone, registered → copied |
|---|---|---|---|
| U-N 18277×500×1000 | 222.39 ms (1.91×, GPU) | 222.87 ms (1.91×, GPU) | 423.83 → 424.83 ms |
| U-M 18277×160×160 | 13.91 ms (1.93×, GPU) | 14.47 ms (1.79×, Accelerate) | 26.81 → 28.60 ms |
| R-K 500×500×18277 | 114.65 ms (1.81×, GPU), same session | 122.69 ms (1.79×, GPU) | 207.35 → 219.75 ms |

- **Cost:** copying costs the split 0–7% and the GPU alone 0–7%.
- **Bits:** the R-K results (split, GPU alone and Accelerate alone) are the same with and without
  registration.
- **Ranges:** in `tables.md`.

## CPU cores in use

A separate run on 2026-10-29 added `getrusage` around each timed call ([`cpu-use/`](cpu-use/README.md)).
Per process, Accelerate alone kept 1.87–1.98 cores busy, although `VECLIB_MAXIMUM_THREADS=8` allowed
eight. The GPU alone kept at most 0.01 busy, and the split 1.69–1.85. The split adds the GPU to the CPU
that Accelerate alone already uses; it does not use more CPU cores.

## What this does not show

- **Data:** the operands are synthetic.
  - The QE-like pattern is a stress case for the exponent-range guard.
  - How often real QE data takes the host path was not measured here. Where it does, the split gains
    less, as above.
- **The 4×2 lockstep:** all four processes call the same GEMM at the same moment, and the GPU runs
  nothing else. In an application the calls are not synchronized, and the GPU may be busy with other
  work, such as FFTs.
- **The GPU alone is its best case here:** registered operands and well-scaled data, with every call
  staged.
- **Duration:** the sustained runs last 80 s each. Longer runs and other ambient conditions were not
  tested, and temperatures and fan speeds were not logged.
- **Hardware:** one M2 Max. The GPU clock comes from an M2 Max DVFS table, and the power figures from
  undocumented SMC and IOReport interfaces.
- **Disturbances:**
  - One round of the QE-like U-N K=1000 small-share run (2026-10-28 08:59) was disturbed: Accelerate
    took 547 ms against 467/481 ms in the other rounds. That was a minute before an unplanned system
    restart. The medians absorb it.
  - The copied-operand R-K run was cut by that restart and rerun the next day. The cut attempt is not
    included.
- **Report format:** in the two `power4x2` reports, the parent process's exit-time `AB_REPORT` line was
  written into the middle of the `SUMMARY` line, because stdout was block-buffered. The fields after
  the break are on the following line. The committed program line-buffers stdout.

## Program revisions

The data comes from three revisions of `bench_split.c`. All three have the same timed loops, checks
and error reference.

| when | suites | revision |
|---|---|---|
| 2026-10-28 04:23–04:36 | smoke, sq1x8, qe4x2, qe1x8, qelike4x2 | first; SMC keys PDTR, PPBR and the GPU rails |
| 2026-10-28 08:58–09:00 | qelike4x2-low, noreg4x2 (U-N, U-M) | + copied operands, 6 s power runs, the small-share cases |
| 2026-10-29 01:10–01:29 | noreg4x2 (R-K), power4x2, sustain | + sustained mode, PSTR, IOReport GPU clock and energy, result hashes in `SUMMARY` |

The committed program is the third revision with names, comments and helper functions reorganised and
stdout line-buffered. Verification below.

## Verification of the committed program

Run on 2026-10-29, 20:24–20:44, on AC power (battery 87–94%, charging).
- **Committed build:** `make lib build/bench_split` on this branch built the program as committed against
  the branch's own library, release/v1.5.0 882d2d8 plus this branch's two commits. `ab_build_info()` reported
  `1.4.0 (c30a298c1795)`, a clean tree: this branch before the section you are reading was added.
  It ran `run.sh`'s smoke suite and R-K 500×500×18277 at 4×2.
- **Measurement build:** the third revision against the 415cb5d library. It ran D2048 and Z2048 in the
  same session.
  - Each run's sweep picks its own share, and these two runs picked other shares than the committed
    build's.
  - So the measurement build also ran both at exactly the committed build's shares.

Result hashes (`h_split`, `h_gpu`, `h_acc` from the `SUMMARY` lines):

| case | GPU part g | committed build (branch library) | measurement build (415cb5d library) |
|---|---|---|---|
| R-K 500×500×18277, 4×2 | 10048 | `3ca65134c9846fe3` `2e61ed92c40717e9` `621d89f4095b2bad` | the same, in `raw/noreg4x2` and `raw/power4x2` |
| DGEMM 2048², 1×8 | 1120 | `40b0c4c54e06c205` `765ccbddf4874614` `38930d3d2059eafc` | the same, in `verification/measurement/same-share` |
| ZGEMM 2048², 1×8 | 1216 | `1c239643341eb6fc` `532175c3887a160e` `abfbfa37588196b3` | the same, in `verification/measurement/same-share` |

- **Bits:** all nine hashes are equal. The committed program, on this branch's library, computes the
  same bits as the program and library that produced the data above. Every GPU-routed call in these runs
  took the staged path (`verification/traces_summary.txt`).
- **Timings:** these runs are not used in any table. Their speedups are close to the published ones.

  | committed build | speedup | published |
  |---|---|---|
  | D2048, over the GPU alone | 1.70× | 1.69× |
  | Z2048, over the GPU alone | 1.47× | 1.52× |
  | U-M K=160 at 4×2, over the faster engine | 1.79× | 1.93× |
  | R-K 160×160 at 4×2, over the faster engine | 1.85× | 1.79× |
  | R-K 500×500 at 4×2, over the faster engine | 1.79× | 1.83× |

  The first session (20:31) ran 4–12% slower on the CPU side than the second (20:44) in both builds,
  for example Accelerate alone on Z2048 at 113.6 and 123.2 ms against 109.6 ms.

## Regenerating the tables

```bash
python3 benchmarks/split/table.py benchmarks/results/2026-10-28-415cb5d-hybrid-split/raw
```

```bash
mkdir -p /tmp/split-traces && tar -xJf benchmarks/results/2026-10-28-415cb5d-hybrid-split/traces.tar.xz -C /tmp/split-traces
```

```bash
python3 benchmarks/split/classify_trace.py /tmp/split-traces/qe4x2
```
