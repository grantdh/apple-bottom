# Contributing

## Setup

```bash
git clone https://github.com/YOUR_USERNAME/apple-bottom.git
cd apple-bottom
make && make test
```

## Requirements

- macOS 14+ (Sonoma)
- Apple Silicon (M1/M2/M3/M4)
- Xcode Command Line Tools

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

1. Pass existing tests: `make test`
2. Include new tests for new functionality
3. Not regress performance: `make bench`

## Pull Requests

1. Create a feature branch
2. Make changes with clear commit messages
3. Run tests and benchmarks
4. Open PR with description of changes

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
