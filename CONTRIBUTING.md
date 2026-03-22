# Contributing to apple-bottom

Thank you for your interest in contributing! This document provides guidelines for contributing to apple-bottom.

## Code of Conduct

Be respectful, inclusive, and constructive. We're all here to make FP64 computing on Apple Silicon better.

## Getting Started

1. Fork the repository
2. Clone your fork: `git clone git@github.com:YOUR_USERNAME/apple-bottom.git`
3. Create a branch: `git checkout -b feature/your-feature`
4. Make your changes
5. Test locally: `make && make test`
6. Commit with clear messages
7. Push and open a Pull Request

## Development Setup

### Requirements
- macOS 12+ (Monterey or later)
- Apple Silicon (M1/M2/M3/M4)
- Xcode Command Line Tools: `xcode-select --install`

### Build
```bash
make          # Build library and examples
make test     # Run tests
make bench    # Run benchmarks
make clean    # Clean build artifacts
```

## What We're Looking For

### High Priority
- [ ] Additional BLAS operations (DTRSM, DPOTRF, DGETRF)
- [ ] Python bindings (NumPy-compatible)
- [ ] Performance tuning for M3/M4 chips
- [ ] Async/non-blocking compute variants

### Medium Priority
- [ ] More comprehensive test coverage
- [ ] Integration examples (Quantum ESPRESSO, PySCF)
- [ ] Documentation improvements
- [ ] Fortran bindings

### Always Welcome
- Bug fixes with test cases
- Performance improvements with benchmarks
- Documentation clarifications
- Typo fixes

## Code Style

### C/Objective-C
- C11 standard
- 4-space indentation
- `snake_case` for functions: `ab_matrix_create`
- `PascalCase` for types: `ABMatrix`, `ABStatus`
- Braces on same line
- Clear, descriptive names

### Metal Shaders
- Match C style
- Document numerical algorithms
- Explain magic numbers (tile sizes, etc.)

### Example
```c
ABStatus ab_matrix_create(int rows, int cols) {
    if (rows <= 0 || cols <= 0) {
        return AB_ERROR_INVALID_ARG;
    }
    // ... implementation
}
```

## Testing

All changes must:
1. Pass existing tests: `./build/test_correctness && ./build/test_precision`
2. Include new tests for new functionality
3. Not regress performance (run `./build/bench_dgemm`)

### Adding Tests

Add tests to `tests/test_correctness.c`:
```c
static void test_your_feature(void) {
    TEST("your feature description");
    // ... test code
    if (success) PASS(); else FAIL("reason");
}
```

## Pull Request Process

1. **Title**: Clear, descriptive (e.g., "Add DTRSM implementation")
2. **Description**: What, why, and how
3. **Tests**: Include test results
4. **Benchmarks**: For performance changes, include before/after

### PR Checklist
- [ ] Code compiles without warnings (`make`)
- [ ] All tests pass (`make test`)
- [ ] New tests added for new features
- [ ] Documentation updated if needed
- [ ] Commit messages are clear

## Performance Contributions

For performance improvements:
1. Run benchmarks on your hardware
2. Include before/after numbers
3. Specify your chip (M1/M2/M3, Pro/Max/Ultra)
4. Test multiple matrix sizes

Example:
```
M2 Max, 38 GPU cores:
DGEMM 2048x2048:
  Before: 580 GFLOP/s
  After:  650 GFLOP/s (+12%)
```

## Questions?

- Open an issue for bugs or feature requests
- Tag with appropriate labels
- Be patient - this is a volunteer project

## License

By contributing, you agree that your contributions will be licensed under the MIT License.
