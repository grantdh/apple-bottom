# apple-bottom Roadmap

## In Progress

- **General alpha/beta** — GPU-accelerated `C = alpha*A*B + beta*C` (routes more QE calls to GPU)

## Planned

- DTRSM (triangular solve via iterative refinement)
- Batched DGEMM / ZGEMM
- Transpose support (transA, transB != 'N')
- Homebrew formula
- Multi-chip validation matrix (M1 through M4 family)
- Performance regression CI

## Future

- LU factorization (DGETRF/DGETRS)
- True async ZGEMM with overlapping command buffers
- Thread-safe contexts for concurrent workloads
- Adaptive GPU/CPU crossover tuning

See [CHANGELOG.md](../CHANGELOG.md) for what's already shipped.
