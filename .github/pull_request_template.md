# Description

Please include a summary of the change and which issue is fixed.

Fixes # (issue)

## Type of change

- [ ] Bug fix (non-breaking change which fixes an issue)
- [ ] New feature (non-breaking change which adds functionality)
- [ ] Breaking change (fix or feature that would cause existing functionality to not work as expected)
- [ ] Documentation update
- [ ] Performance improvement

## Testing

- [ ] All exercises build successfully
- [ ] Correctness tests pass (relative error < 10⁻¹⁴ vs Accelerate DGEMM)
- [ ] Performance validated (GFLOP/s measurements included below)

### Performance Results (if applicable)

| Matrix Size | Before | After | Change |
|-------------|--------|-------|--------|
| 1024×1024   |        |       |        |
| 2048×2048   |        |       |        |
| 4096×4096   |        |       |        |

**Hardware:** [e.g., M2 Max, 30-core GPU]

## Checklist

- [ ] My code follows the style of this project
- [ ] I have commented my code, particularly in hard-to-understand areas
- [ ] I have updated the documentation accordingly
- [ ] I have added tests that prove my fix is effective or that my feature works
- [ ] New and existing tests pass locally
- [ ] Any dependent changes have been merged and published

## For DD (Double-Float) Code

- [ ] Uses `mathMode = .safe` in Metal compilation options
- [ ] Validated precision against Accelerate DGEMM
- [ ] Documented register usage estimates
- [ ] Included performance comparison data

## Additional context

Add any other context about the PR here.
