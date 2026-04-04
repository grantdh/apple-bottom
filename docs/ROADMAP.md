# apple-bottom Roadmap

## In Progress

- **Rectangular matrix support** — full correctness for non-square M, N, K dimensions
- **General alpha/beta** — GPU-accelerated `C = alpha*A*B + beta*C`

## Planned

- DTRSM (triangular solve)
- Batched DGEMM / ZGEMM
- Homebrew formula
- Multi-chip validation matrix (M1 through M4 family)
- Performance regression CI

## Future

- LU factorization (DGETRF/DGETRS)
- True async ZGEMM with overlapping command buffers
- Thread-safe contexts for concurrent workloads
- Adaptive GPU/CPU crossover tuning

See [CHANGELOG.md](../CHANGELOG.md) for what's already shipped.
