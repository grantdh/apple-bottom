## Description

Brief description of changes.

## Type of Change

- [ ] Bug fix
- [ ] New feature
- [ ] Performance improvement
- [ ] Documentation update

## Testing Checklist

⚠️ **CI cannot run GPU tests. You MUST test locally on Apple Silicon.**

- [ ] Ran `make clean && make` — builds without warnings
- [ ] Ran `make test` — all tests pass
- [ ] Kernels, conversion or routing changed: ran `make ci-local`
- [ ] Performance-related: ran `make bench-baseline` on `main`, then `make bench-regression` on this branch (see CONTRIBUTING.md)

### Test Output

```
(Paste output of ./build/test_correctness here)
```

### Benchmark Results (if performance-related)

```
(Paste output of ./build/bench_dgemm here)
```

## Performance Baseline (M2 Max)

DGEMM on matrices resident on the GPU (`make bench-paper`), median of three
runs at 3253a22 (`benchmarks/results/2026-10-14-3253a22-bench-paper/`):

| Size | GFLOP/s |
|------|---------|
| 2048 | 664 |
| 3072 | 683 |
| 4096 | 699 |

## Additional Notes

Any additional context about the changes.
