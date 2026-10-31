---
name: Performance report
about: Share performance results on your hardware
title: '[PERF] '
labels: performance
assignees: ''
---

**Hardware**
- Chip: [e.g., M3 Max]
- GPU cores: [e.g., 40]
- macOS version: [e.g., macOS 14.3]

**Library build**
The `ab_build_info()` string, as `AB_REPORT` and the benchmarks print it: [e.g., apple-bottom 1.5.0 (0123456789ab)]

**What you ran**
[e.g., `make bench-paper`, `make calibrate`, or your application with `AB_REPORT=1`]

**Results**
| Matrix Size | DD-GPU GFLOP/s | Accelerate GFLOP/s | Winner |
|-------------|----------------|--------------------|--------|
| 512×512     |                |                    |        |
| 1024×1024   |                |                    |        |
| 2048×2048   |                |                    |        |
| 4096×4096   |                |                    |        |

`make bench-paper` prints these rates. From `make calibrate`, attach the
`benchmarks/results/<date>-<sha>-calibrate/` directory it writes.

**Precision validation**
- Relative error vs Accelerate DGEMM: [e.g., < 10⁻¹⁴]

**Additional notes**
AC or battery power, other load on the machine, thermal throttling, power consumption data, etc.
