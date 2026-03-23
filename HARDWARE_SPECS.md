# Hardware Specifications (Verified)

## Test Machine

**Verified**: 2026-03-23 via `system_profiler`

```
Model: MacBook Pro (MNXA3LL/A)
Chip: Apple M2 Max
GPU Cores: 38
Memory: 64 GB unified memory
Memory Bandwidth: 400 GB/s (spec sheet)
Metal Support: Metal 4
```

## Performance Characteristics

### Peak Theoretical Performance
- **FP32**: ~13.6 TFLOPS (38-core GPU)
- **FP64 (AMX)**: ~580 GFLOP/s (measured)
- **Memory Bandwidth**: 400 GB/s

### Measured apple-bottom Performance
- **DGEMM N=2048**: 585-618 GFLOP/s (DD emulation on GPU)
- **DGEMM N=4096**: 611 GFLOP/s
- **ZGEMM N=2048**: 726 GFLOP/s
- **Speedup vs AMX**: 1.1-1.3× for large matrices (N ≥ 2048)

## Documentation Audit

### Files Checked ✓
- [x] README.md - No hardcoded specs, uses "M2 Max" generically
- [x] CHANGELOG.md - States "618 GFLOP/s on M2 Max" (correct ballpark)
- [x] CONTRIBUTING.md - States "38 GPU cores" (✓ correct)
- [x] .github/ISSUE_TEMPLATE/bug_report.md - Updated to "38-core GPU"
- [x] QUICKSTART.md - Shows measured performance, no hardcoded specs

### Spec Accuracy
- ✓ No incorrect "96 GB" found in any docs
- ✓ No incorrect "30-core" found (except bug template, now fixed)
- ✓ GFLOP/s numbers are measured values, not theoretical claims

## Notes

1. **TFLOPS vs GFLOPS**: Don't confuse FP32 TFLOPS (13.6) with FP64 GFLOPS (580-720). The latter is what matters for scientific computing.

2. **Variability**: Measured GFLOP/s varies based on:
   - Thermal state
   - Background processes
   - Matrix size
   - Power mode (battery vs plugged in)

3. **Comparison baseline**: All benchmarks compare against `cblas_dgemm` (Accelerate framework, using AMX).

## For Future Updates

When adding benchmark results, always include:
```
Hardware: M2 Max, 38-core GPU, 64 GB
Software: macOS 14.x, Metal 4
Conditions: Plugged in, no background tasks
```

## Sources
- System profiler: `system_profiler SPHardwareDataType SPDisplaysDataType`
- Apple spec sheet: https://www.apple.com/macbook-pro/specs/
- Measured benchmarks: `make bench`
