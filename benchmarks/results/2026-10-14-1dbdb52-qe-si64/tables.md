### Accuracy

| Input | Total energy, `cpu` (Ry) | Largest \|ΔE\|, `gpu` vs `cpu`: final / any iteration (Ry) | SCF iterations, `cpu` / `gpu` | Band energies per run | Largest \|Δε\|, any run vs `cpu` (eV) |
|---|---|---|---|---|---|
| A_si64_500b_k222 | -2990.44276149 | 0 / 0 | 7 / 7 | 2000 | 0 |
| B_si64_160b_k222 | -2990.44093535 | 0 / 0 | 9 / 9 | 640 | 0 |
| C_si64_160b_gamma | -2989.94060060 | 0 / 0 | 8 / 8 | 160 | 0 |

### Routing (AB_REPORT, first repetition)

| Input | Mode | Routine | Calls | On the GPU | GPU share of the routine's time | Fallbacks |
|---|---|---|---:|---:|---:|---:|
| A_si64_500b_k222 | `auto` | ZGEMM | 28425 | 0 | 0.0% | 0 |
| A_si64_500b_k222 | `auto` | DGEMM | 2064 | 0 | 0.0% | 0 |
| A_si64_500b_k222 | `gpu` | ZGEMM | 28425 | 951 | 97.8% | 0 |
| A_si64_500b_k222 | `gpu` | DGEMM | 2064 | 16 | 99.9% | 0 |
| B_si64_160b_k222 | `auto` | ZGEMM | 29048 | 0 | 0.0% | 0 |
| B_si64_160b_k222 | `auto` | DGEMM | 2580 | 0 | 0.0% | 0 |
| B_si64_160b_k222 | `gpu` | ZGEMM | 29048 | 1090 | 96.8% | 0 |
| B_si64_160b_k222 | `gpu` | DGEMM | 2580 | 20 | 99.8% | 0 |
| C_si64_160b_gamma | `auto` | DGEMM | 5624 | 0 | 0.0% | 0 |
| C_si64_160b_gamma | `gpu` | DGEMM | 5624 | 263 | 98.8% | 0 |

### Run time (s, median [min–max])

| Input | `cpu` | `auto` | `gpu` | `gpu` / `cpu` | ZGEMM+DGEMM time, `cpu` | same, `gpu` | `auto`, eaad87e |
|---|---|---|---|---:|---:|---:|---|
| A_si64_500b_k222 | 509.9 [506.5–510.1] | 509.3 [508.0–509.3] | 598.0 [597.7–598.1] | 1.173 | 58.1 | 147.8 | 594.3 [593.3–595.2] |
| B_si64_160b_k222 | 1120.9 [1120.1–1121.7] | 1121.3 [1121.2–1121.3] | 1197.7 [1196.4–1199.1] | 1.069 | 30.5 | 107.0 | – |
| C_si64_160b_gamma | 748.6 [748.6–748.6] | 748.6 [748.5–748.7] | 758.4 [758.4–758.5] | 1.013 | 2.5 | 12.1 | – |

### GPU path stages, `gpu` mode (s, median over repetitions)

| Input | Routine | GPU calls | init | alloc | pack | upload | kernel | download | epilogue | other | total | Accelerate, same routine (`cpu`) | kernel / Accelerate |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| A_si64_500b_k222 | ZGEMM | 951 | 0.00 | 0.17 | 48.12 | 13.80 | 55.01 | 1.77 | 10.62 | 11.62 | 141.16 | 57.92 | 0.95 |
| A_si64_500b_k222 | DGEMM | 16 | 0.03 | 0.00 | 1.02 | 0.19 | 1.89 | 0.02 | 0.27 | 0.11 | 3.54 | 0.19 | 9.94 |
| B_si64_160b_k222 | ZGEMM | 1090 | 0.00 | 0.18 | 38.08 | 12.69 | 32.13 | 0.83 | 4.65 | 9.01 | 97.58 | 29.63 | 1.08 |
| B_si64_160b_k222 | DGEMM | 20 | 0.03 | 0.00 | 1.66 | 0.41 | 2.39 | 0.14 | 1.36 | 0.26 | 6.26 | 0.90 | 2.65 |
| C_si64_160b_gamma | DGEMM | 263 | 0.03 | 0.03 | 4.67 | 1.76 | 3.84 | 0.13 | 0.73 | 0.77 | 11.95 | 2.45 | 1.57 |
