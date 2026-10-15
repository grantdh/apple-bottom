# Contributing

## Setup

```bash
git clone https://github.com/<your-account>/apple-bottom.git   # your fork
cd apple-bottom
make && make test
```

## Requirements

- macOS 14+ (Sonoma)
- Apple Silicon (M1/M2/M3/M4)
- Xcode Command Line Tools
- gfortran (optional, for `make test-fortran`)

## Code Style

### C/Objective-C

- C11 standard
- 4-space indentation
- `snake_case` for functions
- `PascalCase` for types
- Braces on same line

### Metal Shaders

- Match C style
- Document numerical algorithms
- Explain tile sizes and magic numbers

## Testing

All changes must:

1. Pass existing tests: `make test` (and `make ci-local` for changes to
   kernels, conversion or routing)
2. Include new tests for new functionality
3. Not regress performance: `make bench-baseline` on `main`, then
   `make bench-regression` on your branch

If you change the BLAS-ABI GPU path or the kernels, rerun `make calibrate`
on an idle machine on AC power and commit the regenerated
`src/dispatch_table.h` with its `benchmarks/results/` directory, so that
`AB_MODE=auto` keeps following measurement. Calibrations from other chips
are welcome as their own pull requests: run
`make calibrate CALIBRATE_SOURCE=/dev/null` so the built-in M2 Max table
stays as it is, and submit the `benchmarks/results/<date>-<sha>-calibrate/`
directory (its `dispatch_table.txt` can be used at once with
`AB_DISPATCH_TABLE`).
## Pull Requests

1. Create a feature branch
2. Write [Conventional Commits](https://www.conventionalcommits.org/)
   messages (`fix: …`, `feat: …`, `docs: …`), one change per commit
3. Run tests and benchmarks
4. Open a PR describing the change and, for performance claims, the
   machine, build (`ab_build_info()`) and data behind them

### Performance Changes

Include before/after benchmarks:

```
M2 Max, 38 GPU cores:
DGEMM 2048×2048:
  Before: 580 GFLOP/s
  After:  650 GFLOP/s (+12%)
```

## License

By contributing, you agree that your contributions will be licensed under the MIT License.
