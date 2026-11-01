# Double-float FFT against FFTW, five runs, M2 Max, 2026-10-29

`build/bench_fft` (`benchmarks/bench_fft.c`) against the library built from a
clean checkout of 9fc4547, the tip of `feat/dd-fft` (`ab_build_info()`:
`1.4.0 (9fc4547d8bf6)`; its FFT code is that of c138543), and Homebrew's
FFTW 3.3.10 (`3.3.10_3`, linked statically), all in one acquisition of
`machlock.py bench` (exclusive use of the machine), 10:09:22–10:46:18. This
repeats the batch-128 speed rows and the accuracy of
`../2026-10-23-c138543-fft/`, which were one run each.

Machine: MacBook Pro (Mac14,6), Apple M2 Max, 38-core GPU, 8P+4E CPU, 64 GB,
macOS 27.0.1 (26A434), macOS SDK 26.0, Apple clang 17.0.0. AC power
throughout: checked inside the lock before and after every run, and AC in
all 221 samples taken every 10 s (battery charging, 67% to 77%). `pmset -g
therm` recorded no thermal or performance warning at the start or the end.
1-minute load 3.6 at the start, 1.9–4.4 over the batch.

| File | |
|---|---|
| `speed_run1.*` … `speed_run5.*` | the speed runs in order, 7 min 20 s each; the text files also have each GPU plan (`ab_fft_describe`) and per-call GPU and copy times |
| `accuracy_p1.*`, `accuracy_p2.*`, `accuracy_p3.*` | the accuracy runs, 11 s in all, before the speed runs |
| `machine.txt` | machine, power and load at the start and the end, the library's SHA-256, the commands |
| `batch.log` | each run's start and end with power and load; `pmset -g batt` and `pmset -g therm` at the start and the end |
| `power.log` | power source, battery and load every 10 s |

**Speed.** `bench_fft --speed --patient --grids 48,64,72,96,100,108,120,128
--batch 128 --only gpu,fftw,8x1` (with `--reps 3 --warm 3 --threads 8`, the
defaults) five times, each run a new process: forward 3D complex transforms
of n³ grids, FP64 in and out, in batches of 128, time per transform. Each
method is timed as `../2026-10-23-c138543-fft/README.md` describes: a 3 s
warm-up, then the median of 3 to 64 calls; GPU staged, zero copy and
registered; FFTW on one thread, as 8 single-thread workers and as one
8-thread plan, all with `FFTW_PATIENT` plans. Each method's first transform
is compared with FFTW's: 5.86–7.55 × 10⁻¹⁵ for the GPU (the same in every
run), 1.6–4.3 × 10⁻¹⁶ for FFTW's own plans. Unlike 2026-10-23, the GPU and
FFTW ran in the same process, each process made its own `FFTW_PATIENT`
plans, and batch 1 and vDSP were not run.

Median of the five runs, the range in brackets, ms per transform. "8 cores"
is, in each run, the faster of the 8 workers and the 8-thread plan (the
workers, except at 128³ in two runs). Ratios are FFTW's time over the GPU's,
taken within each run:

| n³ | GPU staged | GPU zero copy | FFTW 1 thread | FFTW 8 cores | vs 8 cores, staged | vs 8 cores, zero copy | vs 1 thread, staged | vs 1 thread, zero copy |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 48 | 0.057 (0.057–0.057) | 0.043 (0.043–0.043) | 0.596 (0.594–0.597) | 0.106 (0.105–0.106) | 1.86 (1.84–1.88) | 2.45 (2.45–2.47) | 10.5 (10.4–10.5) | 13.8 (13.8–13.9) |
| 64 | 0.130 (0.130–0.131) | 0.084 (0.084–0.084) | 1.227 (1.219–1.244) | 0.190 (0.188–0.198) | 1.45 (1.44–1.51) | 2.26 (2.23–2.36) | 9.4 (9.4–9.5) | 14.6 (14.5–14.8) |
| 72 | 0.195 (0.194–0.195) | 0.171 (0.171–0.171) | 1.997 (1.993–2.197) | 0.335 (0.330–0.346) | 1.72 (1.69–1.77) | 1.96 (1.93–2.02) | 10.3 (10.2–11.3) | 11.7 (11.6–12.9) |
| 96 | 0.445 (0.443–0.447) | 0.363 (0.362–0.363) | 5.009 (4.944–5.018) | 0.755 (0.751–0.801) | 1.70 (1.68–1.80) | 2.08 (2.07–2.21) | 11.3 (11.1–11.3) | 13.8 (13.6–13.8) |
| 100 | 0.487 (0.485–0.490) | 0.371 (0.369–0.372) | 6.297 (6.265–6.311) | 0.921 (0.920–0.946) | 1.90 (1.88–1.94) | 2.48 (2.48–2.56) | 12.9 (12.9–13.0) | 17.0 (16.9–17.1) |
| 108 | 0.629 (0.626–0.643) | 0.509 (0.509–0.510) | 7.920 (7.801–8.225) | 1.140 (1.131–1.156) | 1.82 (1.77–1.83) | 2.24 (2.22–2.27) | 12.5 (12.4–13.0) | 15.5 (15.3–16.1) |
| 120 | 0.925 (0.919–0.943) | 0.809 (0.808–0.809) | 10.543 (10.533–10.561) | 1.511 (1.509–1.575) | 1.63 (1.60–1.71) | 1.87 (1.87–1.95) | 11.4 (11.2–11.5) | 13.0 (13.0–13.1) |
| 128 | 1.017 (1.012–1.022) | 0.759 (0.758–0.759) | 15.054 (14.248–15.526) | 2.451 (2.320–2.605) | 2.42 (2.28–2.56) | 3.23 (3.06–3.43) | 14.8 (14.0–15.2) | 19.8 (18.8–20.5) |

- **Against eight cores** the GPU is faster on every grid: per-grid medians
  1.45–2.42× staged and 1.87–3.23× with zero copy; every single run lies in
  1.44–2.56× and 1.87–3.43×. The GPU's times agree within 3% from run to
  run, FFTW's eight-core times within 1–7% (12% at 128³), and that sets the
  ratios' ranges.
- **Against one thread:** 9.4–14.8× staged and 11.7–19.8× with zero copy
  (medians).
- **Registered memory** (`gpu-reg` lines) ran within 1% of `ab_fft_malloc`
  memory in the median, except at 108³ (1.4% slower); single runs were up to
  3% slower (100³, 108³).
- **Against the run of 2026-10-23**, whose ratios against eight cores were
  1.34–2.30× staged and 1.79–3.18× with zero copy: they lie below these
  runs' ranges at six grids staged and at five with zero copy, and within
  them elsewhere. The GPU's staged times are
  up to 8% shorter now from 48³ to 96³ and at 128³ (64³: 0.138 ms then,
  0.130–0.131 now; 96³: 0.482, now 0.443–0.447) and within 1% elsewhere,
  and FFTW's eight-core times are 3–8% longer at 64³, 72³, 100³ and 108³
  (72³: 0.310 ms then, 0.330–0.346 now) and 3% shorter at 96³ and 120³.
  Two sessions differ by more than the runs within one.

**Accuracy.** `bench_fft --accuracy`: forward transform, batch 1, random
53-bit inputs in [-1, 1), against a double-double reference DFT
(`tests/fft_ref.h`); FFTW here is Homebrew's on one thread with an
`FFTW_MEASURE` plan (on 2026-10-23, the NEON build). `bench_fft` draws each
grid's input from one random stream (seed 1) in the order the grids are
listed, so the order picks the inputs. `accuracy_p1` lists the eight grids
as on 2026-10-23 and gets the same inputs; `accuracy_p2` and `accuracy_p3`
list 16 grids each (`100,48,64,72,96,128,108,120,48,64,72,96,100,108,120,128`
and `128,120,48,64,72,96,100,108,100,48,64,72,96,128,108,120`), which gives
every grid four more inputs from stretches of the stream that do not
overlap: five inputs per grid. Relative L2 and max error (max |error| /
max |reference|), in units of 10⁻¹⁵ for the GPU and 10⁻¹⁶ for FFTW:

| n³ | GPU L2, 2026-10-23 input: then / now | GPU L2, five inputs | GPU max, five inputs | FFTW L2, five inputs | GPU / FFTW (L2) |
|---:|---:|---:|---:|---:|---:|
| 48 | 6.209 / 6.209 | 6.19–6.23 | 6.51–8.16 | 2.54–2.55 | 24.3–24.5× |
| 64 | 5.866 / 5.866 | 5.86–5.87 | 5.72–8.02 | 2.50 | 23.4× |
| 72 | 6.886 / 6.886 | 6.89–6.91 | 7.64–8.87 | 3.05–3.07 | 22.4–22.6× |
| 96 | 6.592 / 6.592 | 6.59–6.60 | 6.91–7.94 | 2.77 | 23.8× |
| 100 | 7.031 / 7.031 | 7.03–7.04 | 8.09–9.29 | 2.91–2.92 | 24.1× |
| 108 | 7.474 / 7.474 | 7.46–7.47 | 8.28–9.95 | 3.68–3.70 | 20.2–20.3× |
| 120 | 7.225 / 7.225 | 7.22–7.23 | 8.19–9.49 | 3.01–3.02 | 24.0× |
| 128 | 6.542 / 6.542 | 6.54 | 6.89–8.49 | 2.94–3.00 | 21.8–22.3× |

The 2026-10-23 inputs give that day's GPU error to every printed digit, L2
and max. Over all 40 transforms the GPU's relative L2 error is
5.86–7.47 × 10⁻¹⁵, 20.2–24.5× Homebrew FFTW's (2.50–3.70 × 10⁻¹⁶); the max
error depends more on the input (5.72–9.95 × 10⁻¹⁵). Against the reference
of the GPU's double-float-rounded input, the L2 error is 0.996–0.998 of
that: nearly all of it is the transform's own.
