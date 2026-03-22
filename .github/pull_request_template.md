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
- [ ] Ran `make bench` — performance within 5% of baseline

### Test Output

```
(Paste output of ./build/test_correctness here)
```

### Benchmark Results (if performance-related)

```
(Paste output of ./build/bench_dgemm here)
```

## Performance Baseline (M2 Max)

| Size | Expected GFLOP/s |
|------|------------------|
| 2048 | 620-640 |
| 3072 | 630-650 |
| 4096 | 640-660 |

## Additional Notes

Any additional context about the changes.
