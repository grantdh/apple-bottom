## 1.4.0-np1

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
| A_si64_500b_k222 | `gpu` | ZGEMM | 28425 | 951 | 97.6% | 0 |
| A_si64_500b_k222 | `gpu` | DGEMM | 2064 | 16 | 99.9% | 0 |
| B_si64_160b_k222 | `auto` | ZGEMM | 29048 | 0 | 0.0% | 0 |
| B_si64_160b_k222 | `auto` | DGEMM | 2580 | 0 | 0.0% | 0 |
| B_si64_160b_k222 | `gpu` | ZGEMM | 29048 | 1090 | 96.6% | 0 |
| B_si64_160b_k222 | `gpu` | DGEMM | 2580 | 20 | 99.8% | 0 |
| C_si64_160b_gamma | `auto` | DGEMM | 5624 | 0 | 0.0% | 0 |
| C_si64_160b_gamma | `gpu` | DGEMM | 5624 | 263 | 98.7% | 0 |

### Run time (s, median [min–max])

| Input | `cpu` | `auto` | `gpu` | `gpu` / `cpu` | ZGEMM+DGEMM time, `cpu` | same, `gpu` |
|---|---|---|---|---:|---:|---:|
| A_si64_500b_k222 | 306.4 [305.5–307.2] | 302.6 | 396.9 [395.7–398.1] | 1.295 | 58.8 | 150.2 |
| B_si64_160b_k222 | 188.3 [186.0–190.5] | 187.2 | 264.0 [262.9–265.2] | 1.402 | 32.3 | 109.0 |
| C_si64_160b_gamma | 68.1 [67.5–68.7] | 68.7 | 77.7 [77.7–77.7] | 1.141 | 2.3 | 11.4 |

### GPU path stages, `gpu` mode (s, median over repetitions)

| Input | Routine | GPU calls | init | alloc | pack | upload | kernel | download | epilogue | other | total | Accelerate, same calls (`cpu`) | kernel / Accelerate |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| A_si64_500b_k222 | ZGEMM | 951 | 0.00 | 0.18 | 49.30 | 14.39 | 55.26 | 1.94 | 10.87 | 11.29 | 143.25 | 55.31 | 1.00 |
| A_si64_500b_k222 | DGEMM | 16 | 0.03 | 0.00 | 1.04 | 0.20 | 1.89 | 0.02 | 0.30 | 0.11 | 3.60 | 0.15 | 12.58 |
| B_si64_160b_k222 | ZGEMM | 1090 | 0.00 | 0.21 | 40.63 | 13.00 | 31.96 | 0.93 | 5.06 | 7.70 | 99.49 | 28.18 | 1.13 |
| B_si64_160b_k222 | DGEMM | 20 | 0.03 | 0.00 | 1.45 | 0.42 | 2.38 | 0.15 | 1.47 | 0.28 | 6.20 | 0.75 | 3.19 |
| C_si64_160b_gamma | DGEMM | 263 | 0.04 | 0.03 | 4.45 | 1.62 | 3.76 | 0.11 | 0.72 | 0.52 | 11.24 | 2.18 | 1.72 |

## 1.4.0-np8nk4

### Accuracy

| Input | Total energy, `cpu` (Ry) | Largest \|ΔE\|, `gpu` vs `cpu`: final / any iteration (Ry) | SCF iterations, `cpu` / `gpu` | Band energies per run | Largest \|Δε\|, any run vs `cpu` (eV) |
|---|---|---|---|---|---|
| A_si64_500b_k222 | -2990.44276150 | 0 / 0 | 7 / 7 | 2000 | 0 |
| B_si64_160b_k222 | -2990.44093534 | 0 / 0 | 9 / 9 | 640 | 0 |

### Routing (AB_REPORT, first repetition)

| Input | Mode | Routine | Calls | On the GPU | GPU share of the routine's time | Fallbacks |
|---|---|---|---:|---:|---:|---:|
| A_si64_500b_k222 | `auto` | ZGEMM | 54162 | 0 | 0.0% | 0 |
| A_si64_500b_k222 | `auto` | DGEMM | 4224 | 0 | 0.0% | 0 |
| A_si64_500b_k222 | `gpu` | ZGEMM | 54162 | 1800 | 97.2% | 0 |
| A_si64_500b_k222 | `gpu` | DGEMM | 4224 | 128 | 99.9% | 0 |
| B_si64_160b_k222 | `auto` | ZGEMM | 59970 | 0 | 0.0% | 0 |
| B_si64_160b_k222 | `auto` | DGEMM | 5280 | 0 | 0.0% | 0 |
| B_si64_160b_k222 | `gpu` | ZGEMM | 59970 | 2180 | 95.0% | 0 |
| B_si64_160b_k222 | `gpu` | DGEMM | 5280 | 160 | 99.8% | 0 |

### Run time (s, median [min–max])

| Input | `cpu` | `auto` | `gpu` | `gpu` / `cpu` | ZGEMM+DGEMM time, `cpu` | same, `gpu` |
|---|---|---|---|---:|---:|---:|
| A_si64_500b_k222 | 97.1 [96.4–97.9] | 96.7 [95.8–105.6] | 102.4 [102.3–102.4] | 1.055 | 244.7 | 329.1 |
| B_si64_160b_k222 | 53.6 [53.4–53.8] | 54.4 | 65.1 [64.8–65.4] | 1.214 | 137.7 | 255.9 |

### GPU path stages, `gpu` mode (s, median over repetitions)

| Input | Routine | GPU calls | init | alloc | pack | upload | kernel | download | epilogue | other | total | Accelerate, same calls (`cpu`) | kernel / Accelerate |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| A_si64_500b_k222 | ZGEMM | 1800 | 0.01 | 0.35 | 64.42 | 21.13 | 182.84 | 3.55 | 13.57 | 25.96 | 311.82 | 235.08 | 0.78 |
| A_si64_500b_k222 | DGEMM | 128 | 1.17 | 0.05 | 3.25 | 0.42 | 2.53 | 0.06 | 0.38 | 0.33 | 8.18 | 0.59 | 4.25 |
| B_si64_160b_k222 | ZGEMM | 2180 | 0.01 | 0.53 | 95.97 | 22.70 | 80.60 | 2.37 | 7.99 | 20.40 | 230.56 | 123.28 | 0.65 |
| B_si64_160b_k222 | DGEMM | 160 | 1.07 | 0.02 | 4.19 | 1.00 | 3.49 | 0.34 | 2.19 | 0.95 | 13.25 | 2.35 | 1.49 |

## 1.4.0-np8

### Accuracy

| Input | Total energy, `cpu` (Ry) | Largest \|ΔE\|, `gpu` vs `cpu`: final / any iteration (Ry) | SCF iterations, `cpu` / `gpu` | Band energies per run | Largest \|Δε\|, any run vs `cpu` (eV) |
|---|---|---|---|---|---|
| C_si64_160b_gamma | -2989.94060129 | 0 / 0 | 9 / 9 | 160 | 0 |

### Routing (AB_REPORT, first repetition)

| Input | Mode | Routine | Calls | On the GPU | GPU share of the routine's time | Fallbacks |
|---|---|---|---:|---:|---:|---:|
| C_si64_160b_gamma | `auto` | DGEMM | 56000 | 0 | 0.0% | 0 |
| C_si64_160b_gamma | `gpu` | DGEMM | 56000 | 5416 | 93.9% | 0 |

### Run time (s, median [min–max])

| Input | `cpu` | `auto` | `gpu` | `gpu` / `cpu` | ZGEMM+DGEMM time, `cpu` | same, `gpu` |
|---|---|---|---|---:|---:|---:|
| C_si64_160b_gamma | 13.9 [13.7–14.2] | 13.7 | 16.2 [16.2–16.2] | 1.163 | 17.5 | 35.2 |

### GPU path stages, `gpu` mode (s, median over repetitions)

| Input | Routine | GPU calls | init | alloc | pack | upload | kernel | download | epilogue | other | total | Accelerate, same calls (`cpu`) | kernel / Accelerate |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| C_si64_160b_gamma | DGEMM | 5416 | 1.08 | 0.47 | 6.70 | 4.59 | 10.67 | 0.54 | 1.22 | 7.74 | 33.00 | 15.37 | 0.69 |

