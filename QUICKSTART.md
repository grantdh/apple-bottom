# Quick Start

Get apple-bottom running in 60 seconds.

## 1. Clone and Build

```bash
git clone https://github.com/grantdh/apple-bottom.git
cd apple-bottom
make
```

Expected output:
```
✓ Build complete!

  Library:  build/libapplebottom.a
  Examples: build/examples/

Quick start:
  ./build/examples/basic_dgemm
```

## 2. Run the Benchmark

```bash
./build/examples/basic_dgemm 2048
```

Expected output:
```
═══════════════════════════════════════════════════════════════════
apple-bottom Basic DGEMM Example
═══════════════════════════════════════════════════════════════════

Device: Apple M2 Max

Initializing 2048 × 2048 matrices...
Uploading to GPU...
  Upload time: 23.4 ms

Computing C = A × B on GPU...
  Kernel time: 27.1 ms
  Performance: 634 GFLOP/s

Downloading from GPU...
  Download time: 18.2 ms

Computing reference with Accelerate (AMX)...
  AMX time: 29.8 ms (577 GFLOP/s)

═══════════════════════════════════════════════════════════════════
SUMMARY (2048 × 2048)
═══════════════════════════════════════════════════════════════════

  apple-bottom: 27.1 ms (634 GFLOP/s)
  AMX:          29.8 ms (577 GFLOP/s)
  Speedup:      1.10x ✓ GPU wins

  Precision:    3.12e-16 (vs FP64 reference)
```

## 3. Run an Iterative Example

This is where apple-bottom really shines:

```bash
./build/examples/eigenvalue_solver 2048
```

You'll see that for iterative algorithms (power iteration, CG, SCF loops), the GPU beats AMX because the matrix stays on GPU while we iterate.

## 4. Use in Your Code

```c
#include "apple_bottom.h"

int main() {
    ab_init();
    
    // Create matrices
    ABMatrix A = ab_matrix_create(N, N);
    ABMatrix B = ab_matrix_create(N, N);
    ABMatrix C = ab_matrix_create(N, N);
    
    // Upload ONCE
    ab_matrix_upload(A, A_data, true);
    ab_matrix_upload(B, B_data, true);
    
    // Compute (can call many times with no overhead)
    ab_dgemm(A, B, C);
    
    // Download ONCE
    ab_matrix_download(C, C_data, true);
    
    ab_shutdown();
}
```

Compile with:
```bash
clang -O3 your_code.c -I/path/to/apple-bottom/include \
      -L/path/to/apple-bottom/build -lapplebottom \
      -framework Metal -framework Foundation -framework Accelerate
```

## 5. Next Steps

- Read [INTEGRATION.md](docs/INTEGRATION.md) for patterns like CG, SCF, batch processing
- Check the [examples/](examples/) directory for more use cases
- Run `make bench` for comprehensive performance data

## Common Issues

**"No Metal GPU available"**
- Must be Apple Silicon (M1/M2/M3)
- Check: `system_profiler SPDisplaysDataType`

**GPU slower than AMX**
- Check matrix size (need N ≥ 2048)
- For single operations, AMX wins — apple-bottom is for iterative workloads

**Build fails**
- Need Xcode Command Line Tools: `xcode-select --install`
- Need macOS 12+ (Monterey)
