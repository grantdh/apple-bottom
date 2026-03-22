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

**Exercise/Kernel**
Which exercise did you test? [e.g., ex10_production_dd_dgemm]

**Results**
| Matrix Size | DD-GPU GFLOP/s | AMX GFLOP/s | Winner |
|-------------|----------------|-------------|--------|
| 512×512     |                |             |        |
| 1024×1024   |                |             |        |
| 2048×2048   |                |             |        |
| 4096×4096   |                |             |        |

**Precision validation**
- Relative error vs Accelerate DGEMM: [e.g., < 10⁻¹⁴]

**Build command**
```bash
swiftc -O -framework Metal -framework Foundation -framework Accelerate \
    exercises/10_production_dd_dgemm/ex10_production_dd_dgemm.swift -o test
```

**Additional notes**
Any observations, thermal throttling, power consumption data, etc.
