# Double-float FFT against FFTW and vDSP, 2026-10-23

`build/bench_fft_neon --accuracy`, `--speed --patient` (FFTW 3.3.10 built
locally with NEON) and `build/bench_fft --speed --patient
--only fftw,8x1` (Homebrew's FFTW 3.3.10, which has no NEON double-precision
code), one run each under the machine lock and `caffeinate -is`, library built
at c138543 (`ab_build_info()`: `1.4.0 (c13854306d56)`). The batch-128 times
and the accuracy were measured again at 9fc4547, five runs, GPU and FFTW in
the same process: `../2026-10-29-9fc4547-fft/`.

Machine: MacBook Pro (Mac14,6), Apple M2 Max, 38-core GPU, 8P+4E CPU, 64 GB,
macOS 27.0.1 (26A434), AC power; 1-minute load 4.95 at the start, 4.6 at the
end (`machine.txt`).

**What is timed.** Forward 3D complex transforms of n³ grids, FP64 in and out,
at batch 1 and batch 128 (a band set), time per transform:

| Method | |
|---|---|
| GPU staged | `ab_fft_execute` on malloc'd arrays: copies into and out of the plan's staging buffers included |
| GPU zero copy | arrays from `ab_fft_malloc`, read and written in place |
| GPU registered | the malloc'd arrays after `ab_fft_register` (in the text files as `gpu-reg`) |
| FFTW 1 thread | a one-transform `FFTW_PATIENT` plan, run over the batch |
| FFTW 8 cores | the faster of one 8-thread plan (`fftw3_threads`) and 8 workers each running the 1-thread plan on 1/8 of the batch, as MPI ranks with `QE_FFTW_THREADS=1` do (batch 128 only) |
| vDSP | `vDSP_DFT_Interleaved_ExecuteD` line by line, 1 or 8 threads, for the lengths it supports (not 100 or 108) |

Each method gets a 3 s warm-up (single-transform GPU calls are bursty, and the
GPU clock needs that long to rise: at 0.5 s, 72³ took 0.65 ms of GPU time
instead of 0.22 ms), then the median of 3 to 64 calls. Each method's first
output transform is compared with FFTW's (column `vs_fftw`): 1e-16 to 1.1e-15
for vDSP and the other FFTW, 5.8e-15 to 7.6e-15 for the GPU.

The tables below take, for each grid, the faster of the two FFTW builds. That
was Homebrew's everywhere: on 1 thread, and on 8 cores (its 8 workers at batch
128, its 8-thread plan at batch 1). With `FFTW_PATIENT` plans the NEON build
was 1.2–4.6× slower on 1 thread, and its 8-thread plans were pathological (144
ms per 48³ transform, 926 ms per 96³; `speed_fftw_neon.txt`). In an earlier
run with `FFTW_MEASURE` plans, on battery, the NEON build was within ~20% of
Homebrew's, and both were slower than here. Time per transform in ms:

**Batch 1**

| n³ | GPU staged | GPU zero copy | FFTW 1 thread | FFTW 8 cores | vDSP (best) | vs FFTW 1 thread, staged / zero copy | vs FFTW 8 cores, staged / zero copy |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 48 | 0.310 | 0.223 | 0.527 | 0.164 | 0.215 | 1.7× / 2.4× | 0.53× / 0.73× |
| 64 | 0.484 | 0.269 | 1.197 | 0.267 | 0.757 | 2.5× / 4.4× | 0.55× / 0.99× |
| 72 | 0.668 | 0.387 | 1.923 | 0.327 | 0.568 | 2.9× / 5.0× | 0.49× / 0.85× |
| 96 | 0.959 | 0.551 | 4.899 | 0.825 | 1.621 | 5.1× / 8.9× | 0.86× / 1.50× |
| 100 | 1.018 | 0.577 | 6.153 | 0.936 | – | 6.0× / 10.7× | 0.92× / 1.62× |
| 108 | 1.244 | 0.742 | 7.770 | 1.181 | – | 6.2× / 10.5× | 0.95× / 1.59× |
| 120 | 1.637 | 1.024 | 10.513 | 2.442 | 1.834 | 6.4× / 10.3× | 1.49× / 2.39× |
| 128 | 2.232 | 0.975 | 14.186 | 2.278 | 6.457 | 6.4× / 14.6× | 1.02× / 2.34× |

**Batch 128**

| n³ | GPU staged | GPU zero copy | FFTW 1 thread | FFTW 8 cores | vDSP (best) | vs FFTW 1 thread, staged / zero copy | vs FFTW 8 cores, staged / zero copy |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 48 | 0.061 | 0.044 | 0.600 | 0.106 | 0.147 | 9.9× / 13.8× | 1.75× / 2.43× |
| 64 | 0.138 | 0.086 | 1.235 | 0.185 | 0.461 | 8.9× / 14.3× | 1.34× / 2.15× |
| 72 | 0.207 | 0.173 | 1.996 | 0.310 | 0.409 | 9.6× / 11.5× | 1.50× / 1.79× |
| 96 | 0.482 | 0.370 | 4.936 | 0.781 | 1.595 | 10.2× / 13.4× | 1.62× / 2.11× |
| 100 | 0.490 | 0.372 | 6.339 | 0.892 | – | 12.9× / 17.0× | 1.82× / 2.40× |
| 108 | 0.626 | 0.510 | 7.864 | 1.104 | – | 12.6× / 15.4× | 1.76× / 2.16× |
| 120 | 0.932 | 0.810 | 10.551 | 1.554 | 1.915 | 11.3× / 13.0× | 1.67× / 1.92× |
| 128 | 1.050 | 0.758 | 14.982 | 2.414 | 8.239 | 14.3× / 19.8× | 2.30× / 3.18× |

Registered memory (`gpu-reg` lines) ran as fast as `ab_fft_malloc` memory at
batch 128 (within 1%) and within 0.96–1.09× of it at batch 1. An earlier smoke
test with a 0.5 s warm-up had shown registered memory up to 2.6× slower at
batch 1; with the 3 s warm-up that difference is gone, which points to the GPU
clock rather than to the memory.

**Accuracy** (`accuracy.txt`): forward transform, batch 1, random 53-bit
inputs in [-1, 1), against a double-double reference DFT (`tests/fft_ref.h`):

| n³ | GPU relative L2 | GPU max | FFTW relative L2 | FFTW max | GPU / FFTW (L2) |
|---:|---:|---:|---:|---:|---:|
| 48 | 6.2e-15 | 7.8e-15 | 2.6e-16 | 3.1e-16 | 24× |
| 64 | 5.9e-15 | 6.2e-15 | 2.5e-16 | 3.0e-16 | 23× |
| 72 | 6.9e-15 | 8.6e-15 | 3.1e-16 | 4.1e-16 | 22× |
| 96 | 6.6e-15 | 7.9e-15 | 2.9e-16 | 3.5e-16 | 23× |
| 100 | 7.0e-15 | 8.1e-15 | 3.0e-16 | 3.5e-16 | 23× |
| 108 | 7.5e-15 | 8.3e-15 | 3.7e-16 | 4.5e-16 | 20× |
| 120 | 7.2e-15 | 8.2e-15 | 3.0e-16 | 3.5e-16 | 24× |
| 128 | 6.5e-15 | 7.0e-15 | 2.8e-16 | 3.4e-16 | 23× |

"max" is max |error| / max |reference|. Against the reference of the GPU's
double-float-rounded input (the transform alone) the GPU's error is the same
to two digits: rounding the input to ~48 bits costs nothing measurable.

An energy comparison made in the same session was withdrawn on 2026-10-29:
it measured FFTW as the NEON build with `FFTW_MEASURE` plans, not the
Homebrew build with `FFTW_PATIENT` plans that the times above compare
against.

| File | |
|---|---|
| `machine.txt` | machine, power, load, build |
| `accuracy.txt`, `accuracy.csv` | the accuracy run |
| `speed_fftw_neon.txt`, `.csv` | GPU, NEON FFTW and vDSP; the text file also has each GPU plan (`ab_fft_describe`) and per-call GPU and copy times |
| `speed_fftw_homebrew.txt`, `.csv` | Homebrew FFTW |
