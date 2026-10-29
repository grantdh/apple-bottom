## 1.5.0-np1

### Accuracy

| Input | Total energy, `cpu` (Ry) | Largest \|ΔE\|, `gpu` vs `cpu`: final / any iteration (Ry) | SCF iterations, `cpu` / `gpu` | Band energies per run | Largest \|Δε\|, any run vs `cpu` (eV) |
|---|---|---|---|---|---|
| A_si64_500b_k222 | -2990.44276149 | 0 / 0 | 7 / 7 | 2000 | 0 |
| B_si64_160b_k222 | -2990.44093535 | 0 / 0 | 9 / 9 | 640 | 0 |

### Routing (AB_REPORT, first repetition)

| Input | Mode | Routine | Calls | On the GPU | GPU share of the routine's time | Fallbacks |
|---|---|---|---:|---:|---:|---:|
| A_si64_500b_k222 | `auto` | ZGEMM | 28425 | 0 | 0.0% | 0 |
| A_si64_500b_k222 | `auto` | DGEMM | 2064 | 0 | 0.0% | 0 |
| A_si64_500b_k222 | `gpu` | ZGEMM | 28425 | 944 | 95.8% | 7 |
| A_si64_500b_k222 | `gpu` | DGEMM | 2064 | 8 | 43.2% | 8 |
| B_si64_160b_k222 | `auto` | ZGEMM | 29048 | 0 | 0.0% | 0 |
| B_si64_160b_k222 | `auto` | DGEMM | 2580 | 0 | 0.0% | 0 |
| B_si64_160b_k222 | `gpu` | ZGEMM | 29048 | 1088 | 93.7% | 2 |
| B_si64_160b_k222 | `gpu` | DGEMM | 2580 | 10 | 60.2% | 10 |

### Run time (s, median [min–max])

| Input | `cpu` | `auto` | `gpu` | `gpu` / `cpu` | ZGEMM+DGEMM time, `cpu` | same, `gpu` |
|---|---|---|---|---:|---:|---:|
| A_si64_500b_k222 | 306.4 [305.1–306.9] | 304.4 | 359.9 [358.8–360.4] | 1.175 | 57.1 | 111.7 |
| B_si64_160b_k222 | 184.6 [184.5–184.8] | 184.0 | 221.6 [221.5–221.8] | 1.200 | 29.5 | 66.2 |

### GPU path stages, `gpu` mode (s, median over repetitions)

| Input | Routine | GPU calls | init | alloc | pack | upload | kernel | download | epilogue | other | total | Accelerate, same calls (`cpu`) | kernel / Accelerate |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| A_si64_500b_k222 | ZGEMM | 944 | 0.00 | 0.07 | 13.04 | 8.54 | 72.86 | 1.29 | 4.69 | 4.27 | 104.85 | 52.31 | 1.39 |
| A_si64_500b_k222 | DGEMM | 8 | 0.00 | 0.00 | 0.43 | 0.09 | 0.09 | 0.02 | 0.29 | 0.06 | 0.99 | -1.13 | -0.08 |
| B_si64_160b_k222 | ZGEMM | 1088 | 0.00 | 0.11 | 10.89 | 7.97 | 31.70 | 0.65 | 2.42 | 2.86 | 56.53 | 25.01 | 1.27 |
| B_si64_160b_k222 | DGEMM | 10 | 0.00 | 0.00 | 1.03 | 0.12 | 0.32 | 0.14 | 1.69 | 0.22 | 3.53 | -1.71 | -0.19 |

## 1.4.0-np1

### Accuracy

| Input | Total energy, `cpu` (Ry) | Largest \|ΔE\|, `gpu` vs `cpu`: final / any iteration (Ry) | SCF iterations, `cpu` / `gpu` | Band energies per run | Largest \|Δε\|, any run vs `cpu` (eV) |
|---|---|---|---|---|---|
| A_si64_500b_k222 | -2990.44276149 | 0 / 0 | 7 / 7 | 2000 | 0 |
| B_si64_160b_k222 | -2990.44093535 | 0 / 0 | 9 / 9 | 640 | 0 |

### Routing (AB_REPORT, first repetition)

| Input | Mode | Routine | Calls | On the GPU | GPU share of the routine's time | Fallbacks |
|---|---|---|---:|---:|---:|---:|
| A_si64_500b_k222 | `gpu` | ZGEMM | 28425 | 951 | 97.8% | 0 |
| A_si64_500b_k222 | `gpu` | DGEMM | 2064 | 16 | 99.8% | 0 |
| B_si64_160b_k222 | `gpu` | ZGEMM | 29048 | 1090 | 96.9% | 0 |
| B_si64_160b_k222 | `gpu` | DGEMM | 2580 | 20 | 99.8% | 0 |

### Run time (s, median [min–max])

| Input | `cpu` | `auto` | `gpu` | `gpu` / `cpu` | ZGEMM+DGEMM time, `cpu` | same, `gpu` |
|---|---|---|---|---:|---:|---:|
| A_si64_500b_k222 | 304.9 [304.7–305.0] | – | 397.9 [397.6–398.3] | 1.305 | 57.1 | 149.8 |
| B_si64_160b_k222 | 184.7 [184.4–184.9] | – | 264.6 [264.4–264.8] | 1.433 | 29.6 | 109.3 |

### GPU path stages, `gpu` mode (s, median over repetitions)

| Input | Routine | GPU calls | init | alloc | pack | upload | kernel | download | epilogue | other | total | Accelerate, same calls (`cpu`) | kernel / Accelerate |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| A_si64_500b_k222 | ZGEMM | 951 | 0.00 | 0.17 | 48.74 | 14.52 | 55.06 | 1.81 | 10.71 | 11.99 | 142.99 | 53.70 | 1.03 |
| A_si64_500b_k222 | DGEMM | 16 | 0.04 | 0.00 | 1.00 | 0.20 | 1.90 | 0.02 | 0.27 | 0.13 | 3.55 | 0.14 | 13.90 |
| B_si64_160b_k222 | ZGEMM | 1090 | 0.00 | 0.21 | 38.48 | 13.58 | 31.98 | 0.85 | 4.93 | 10.00 | 100.03 | 25.67 | 1.25 |
| B_si64_160b_k222 | DGEMM | 20 | 0.03 | 0.01 | 1.36 | 0.43 | 2.39 | 0.14 | 1.35 | 0.39 | 6.09 | 0.69 | 3.47 |

## 1.5.0-np8nk4

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
| A_si64_500b_k222 | `gpu` | ZGEMM | 54162 | 1790 | 96.8% | 10 |
| A_si64_500b_k222 | `gpu` | DGEMM | 4224 | 96 | 72.8% | 32 |
| B_si64_160b_k222 | `auto` | ZGEMM | 59970 | 0 | 0.0% | 0 |
| B_si64_160b_k222 | `auto` | DGEMM | 5280 | 0 | 0.0% | 0 |
| B_si64_160b_k222 | `gpu` | ZGEMM | 59970 | 2177 | 93.4% | 3 |
| B_si64_160b_k222 | `gpu` | DGEMM | 5280 | 80 | 49.4% | 80 |

### Run time (s, median [min–max])

| Input | `cpu` | `auto` | `gpu` | `gpu` / `cpu` | ZGEMM+DGEMM time, `cpu` | same, `gpu` |
|---|---|---|---|---:|---:|---:|
| A_si64_500b_k222 | 104.7 [104.3–105.3] | 103.5 | 124.9 [123.3–126.9] | 1.193 | 300.4 | 404.1 |
| B_si64_160b_k222 | 52.6 [52.6–53.8] | 52.9 | 61.8 [61.6–62.8] | 1.174 | 136.0 | 215.9 |

### GPU path stages, `gpu` mode (s, median over repetitions)

| Input | Routine | GPU calls | init | alloc | pack | upload | kernel | download | epilogue | other | total | Accelerate, same calls (`cpu`) | kernel / Accelerate |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| A_si64_500b_k222 | ZGEMM | 1790 | 0.01 | 0.14 | 13.83 | 14.17 | 329.74 | 2.23 | 4.50 | 20.97 | 385.82 | 287.53 | 1.15 |
| A_si64_500b_k222 | DGEMM | 96 | 0.73 | 0.09 | 0.92 | 0.59 | 0.86 | 0.06 | 0.39 | 0.61 | 4.28 | -1.01 | -0.85 |
| B_si64_160b_k222 | ZGEMM | 2177 | 0.01 | 0.22 | 14.77 | 13.79 | 139.61 | 1.27 | 2.68 | 15.28 | 188.43 | 120.42 | 1.16 |
| B_si64_160b_k222 | DGEMM | 80 | 0.00 | 0.01 | 1.28 | 0.32 | 1.17 | 0.38 | 3.27 | 0.88 | 7.31 | -4.83 | -0.24 |

## 1.4.0-np8nk4

### Accuracy

| Input | Total energy, `cpu` (Ry) | Largest \|ΔE\|, `gpu` vs `cpu`: final / any iteration (Ry) | SCF iterations, `cpu` / `gpu` | Band energies per run | Largest \|Δε\|, any run vs `cpu` (eV) |
|---|---|---|---|---|---|
| A_si64_500b_k222 | -2990.44276150 | 0 / 0 | 7 / 7 | 2000 | 0 |
| B_si64_160b_k222 | -2990.44093534 | 0 / 0 | 9 / 9 | 640 | 0 |

### Routing (AB_REPORT, first repetition)

| Input | Mode | Routine | Calls | On the GPU | GPU share of the routine's time | Fallbacks |
|---|---|---|---:|---:|---:|---:|
| A_si64_500b_k222 | `gpu` | ZGEMM | 54162 | 1800 | 97.1% | 0 |
| A_si64_500b_k222 | `gpu` | DGEMM | 4224 | 128 | 99.8% | 0 |
| B_si64_160b_k222 | `gpu` | ZGEMM | 59970 | 2180 | 95.0% | 0 |
| B_si64_160b_k222 | `gpu` | DGEMM | 5280 | 160 | 99.8% | 0 |

### Run time (s, median [min–max])

| Input | `cpu` | `auto` | `gpu` | `gpu` / `cpu` | ZGEMM+DGEMM time, `cpu` | same, `gpu` |
|---|---|---|---|---:|---:|---:|
| A_si64_500b_k222 | 107.9 [105.9–109.9] | – | 114.4 [111.9–117.0] | 1.061 | 291.9 | 347.1 |
| B_si64_160b_k222 | 54.0 [53.7–54.2] | – | 67.6 [66.7–68.5] | 1.253 | 150.5 | 263.7 |

### GPU path stages, `gpu` mode (s, median over repetitions)

| Input | Routine | GPU calls | init | alloc | pack | upload | kernel | download | epilogue | other | total | Accelerate, same calls (`cpu`) | kernel / Accelerate |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| A_si64_500b_k222 | ZGEMM | 1800 | 0.01 | 0.45 | 70.45 | 23.70 | 187.49 | 4.34 | 16.25 | 26.36 | 329.06 | 281.57 | 0.67 |
| A_si64_500b_k222 | DGEMM | 128 | 1.02 | 0.12 | 3.31 | 0.48 | 2.45 | 0.07 | 0.39 | 0.41 | 8.27 | 0.55 | 4.47 |
| B_si64_160b_k222 | ZGEMM | 2180 | 0.01 | 0.62 | 97.47 | 24.77 | 78.21 | 2.65 | 9.19 | 24.00 | 236.91 | 135.40 | 0.58 |
| B_si64_160b_k222 | DGEMM | 160 | 0.86 | 0.05 | 4.28 | 1.11 | 3.43 | 0.43 | 2.34 | 1.51 | 14.01 | 2.26 | 1.52 |

