# Future Repository Split Strategy

This document outlines how to split apple-bottom into separate repositories when integrations with multiple applications (QE, MEEP, etc.) justify it.

---

## Current Structure (Option 1: Single Repo)

```
apple-bottom/                          # General BLAS library
├── src/                              # Core library (application-agnostic)
├── include/                          # Public API
├── tests/                            # Unit tests
├── benchmarks/                       # Synthetic benchmarks
├── docs/
│   ├── INTEGRATION.md               # C API integration
│   ├── fortran-integration.md       # General Fortran guide
│   └── qe-integration.md            # QE-specific guide (extractable)
└── README.md                         # Showcases QE as validation

~/qe-test/                            # OUTSIDE repo (not version-controlled)
├── q-e-qe-7.4.1/                    # QE source with patches
├── q-e-native/                      # QE sandbox
└── benchmark/                        # QE benchmarks
```

**Status:** ✓ Currently implemented

---

## Future Structure (Option 2: Split Repos)

### When to Split

Trigger conditions (any one):
- ✓ Integration with 2+ applications (QE + MEEP)
- ✓ Want to distribute QE patches as separate product
- ✓ Different release cycles for library vs integrations
- ✓ apple-bottom README becoming cluttered with app-specific details

### Target Structure

```
apple-bottom/                          # Core library repo
├── src/, include/, tests/            # Library code
├── docs/
│   ├── INTEGRATION.md               # C API
│   └── fortran-integration.md       # General Fortran
└── README.md                         # "BLAS library for Apple Silicon"
                                      # Links to integration repos

apple-bottom-qe/                      # QE integration repo (NEW)
├── patches/
│   └── cegterg-v7.4.1.patch         # Generated from QE source
├── scripts/
│   ├── install_qe.sh                # Downloads QE, applies patches
│   ├── build_qe.sh                  # Builds QE with apple-bottom
│   └── run_benchmark.sh             # Runs Si64 benchmark
├── benchmarks/
│   ├── si64.in                      # Input files
│   ├── si128.in
│   └── results/                     # Performance logs
├── docs/
│   └── TROUBLESHOOTING.md           # QE-specific issues
└── README.md                         # "apple-bottom for QE"

apple-bottom-meep/                    # MEEP integration repo (FUTURE)
├── patches/
├── scripts/
└── README.md
```

---

## Migration Path

### Phase 1: Extract QE Integration (Ready Now)

Everything needed is already in place:

**From apple-bottom repo:**
1. Copy `docs/qe-integration.md` → `apple-bottom-qe/README.md`
2. Extract QE recovery steps from `LESSONS_LEARNED.md` → `apple-bottom-qe/docs/TROUBLESHOOTING.md`
3. Keep `tests/test_qe_integration.sh` in apple-bottom (or copy to both)

**From ~/qe-test/:**
4. Generate patches: `cd ~/qe-test/q-e-qe-7.4.1 && git diff > patches/cegterg-v7.4.1.patch`
5. Copy benchmarks: `cp -r ~/qe-test/benchmark apple-bottom-qe/benchmarks/`
6. Create build scripts (see below)

**Update apple-bottom README:**
```markdown
## Validated Integrations

- **Quantum ESPRESSO** — 2.7× speedup on Si64 benchmark
  → [apple-bottom-qe](https://github.com/grantdh/apple-bottom-qe)
- **MEEP** (coming soon)
```

### Phase 2: Create Build Scripts

**apple-bottom-qe/scripts/install_qe.sh:**
```bash
#!/bin/bash
# Downloads QE 7.4.1 and applies apple-bottom patches

QE_VERSION=7.4.1
QE_DIR=q-e-qe-${QE_VERSION}

# Download QE
wget https://github.com/QEF/q-e/archive/qe-${QE_VERSION}.tar.gz
tar xzf qe-${QE_VERSION}.tar.gz

# Apply patches
cd ${QE_DIR}
patch -p1 < ../patches/cegterg-v7.4.1.patch

# Configure
./configure

# Update make.inc
echo "Updating make.inc with apple-bottom flags..."
sed -i.bak 's/DFLAGS\s*=.*/& -D__APPLE_BOTTOM__/' make.inc
sed -i.bak 's|BLAS_LIBS\s*=.*|BLAS_LIBS = -L$(HOME)/apple-bottom/build -lapplebottom -framework Accelerate -framework Metal -framework Foundation|' make.inc

echo "QE installed. Run ./scripts/build_qe.sh to compile."
```

**apple-bottom-qe/scripts/build_qe.sh:**
```bash
#!/bin/bash
# Builds QE with apple-bottom

QE_DIR=q-e-qe-7.4.1

# Verify apple-bottom is built
if [ ! -f ~/apple-bottom/build/libapplebottom.a ]; then
    echo "Error: apple-bottom library not found"
    echo "Build it first: cd ~/apple-bottom && make"
    exit 1
fi

# Build QE
cd ${QE_DIR}
make clean
make pw -j$(sysctl -n hw.ncpu)

echo "Build complete. Binary: ${QE_DIR}/bin/pw.x"
```

**apple-bottom-qe/scripts/run_benchmark.sh:**
```bash
#!/bin/bash
# Runs Si64 benchmark and compares to baseline

QE_DIR=q-e-qe-7.4.1
BENCH_DIR=benchmarks

cd ${BENCH_DIR}
rm -rf tmp && mkdir -p tmp

echo "Running Si64 benchmark..."
time ../${QE_DIR}/bin/pw.x < si64.in > si64_apple.out 2>&1

# Extract energy
ENERGY=$(grep '!' si64_apple.out | grep 'total energy' | awk '{print $5}')
echo "Total energy: ${ENERGY} Ry"

# Verify correctness
EXPECTED="-2990.44276157"
if [ "$ENERGY" = "$EXPECTED" ]; then
    echo "✓ Energy matches reference"
else
    echo "✗ Energy mismatch! Expected ${EXPECTED}, got ${ENERGY}"
    exit 1
fi
```

### Phase 3: Update apple-bottom README

**Remove QE-specific sections:**
- QE Benchmark table → Link to apple-bottom-qe repo
- Fortran Integration example → Link to docs/fortran-integration.md
- Keep one-line mention: "Validated with Quantum ESPRESSO (2.7× speedup)"

**Add Integration section:**
```markdown
## Validated Integrations

apple-bottom has been integrated with production scientific codes:

- **[Quantum ESPRESSO](https://github.com/grantdh/apple-bottom-qe)** — Density functional theory
  - 2.7× speedup on Si64 benchmark vs single-threaded OpenBLAS
  - 14% faster than 6-thread OpenBLAS
  - Energy validation: exact match to 11 decimal places

- **MEEP** (coming soon) — Electromagnetic simulations

See [Fortran Integration Guide](docs/fortran-integration.md) for integrating with your own code.
```

---

## Extraction Checklist

### Files Ready to Move

**From apple-bottom/docs/:**
- [ ] `qe-integration.md` → `apple-bottom-qe/README.md`

**From apple-bottom/LESSONS_LEARNED.md:**
- [ ] QE SUCCESS section → `apple-bottom-qe/docs/RESULTS.md`
- [ ] Critical Recovery Steps → `apple-bottom-qe/docs/TROUBLESHOOTING.md`
- [ ] What Failed sections → `apple-bottom-qe/docs/PITFALLS.md`

**From ~/qe-test/ (generate/copy):**
- [ ] Patches: `git diff cegterg.f90 > patches/cegterg-v7.4.1.patch`
- [ ] Benchmarks: `cp -r benchmark/ apple-bottom-qe/benchmarks/`
- [ ] Scripts: Create install/build/run scripts (templates above)

**New files to create:**
- [ ] `apple-bottom-qe/README.md` — Main integration guide
- [ ] `apple-bottom-qe/scripts/install_qe.sh` — Automated installer
- [ ] `apple-bottom-qe/scripts/build_qe.sh` — Build script
- [ ] `apple-bottom-qe/scripts/run_benchmark.sh` — Benchmark runner
- [ ] `apple-bottom-qe/docs/TROUBLESHOOTING.md` — Recovery procedures
- [ ] `apple-bottom-qe/LICENSE` — MIT (same as apple-bottom)

### Files to Update in apple-bottom

**README.md:**
- [ ] Remove QE Benchmark section (replace with link)
- [ ] Remove Fortran Integration code example (link to docs)
- [ ] Add "Validated Integrations" section
- [ ] Keep one-sentence QE mention in intro

**LESSONS_LEARNED.md:**
- [ ] Keep general insights (EXTERNAL declaration, Fortran ABI)
- [ ] Remove QE-specific recovery steps (link to apple-bottom-qe)
- [ ] Keep architecture lessons (routing, beta handling)

**docs/fortran-integration.md:**
- [ ] Already generic ✓
- [ ] Add link to apple-bottom-qe at bottom

---

## READMEs Comparison

### apple-bottom/README.md (after split)

```markdown
# apple-bottom

High-performance BLAS library for Apple Silicon GPU using Metal compute shaders.

## Overview

Implements FP64-class operations through double-float emulation with ~10⁻¹⁵ precision.
Validated in production with Quantum ESPRESSO (2.7× speedup).

## Quick Start

[installation, basic usage, API reference]

## Validated Integrations

- [Quantum ESPRESSO](https://github.com/grantdh/apple-bottom-qe) — 2.7× speedup
- MEEP (coming soon)

See [Fortran Integration Guide](docs/fortran-integration.md) to integrate with your code.
```

### apple-bottom-qe/README.md

```markdown
# apple-bottom for Quantum ESPRESSO

GPU-accelerated Quantum ESPRESSO using apple-bottom BLAS library.

## Performance

2.7× speedup over single-threaded OpenBLAS, 14% faster than 6-thread OpenBLAS.

[performance table, benchmark details]

## Quick Start

```bash
# 1. Install apple-bottom
git clone https://github.com/grantdh/apple-bottom
cd apple-bottom && make && make test

# 2. Install QE with patches
cd ~/
git clone https://github.com/grantdh/apple-bottom-qe
cd apple-bottom-qe
./scripts/install_qe.sh

# 3. Build QE
./scripts/build_qe.sh

# 4. Run benchmark
./scripts/run_benchmark.sh
```

## Integration Details

[cegterg.f90 patches, build configuration, troubleshooting]

## Repository

Requires [apple-bottom](https://github.com/grantdh/apple-bottom) library.
```

---

## Decision Points

### Stay Single Repo If:
- ✓ Only QE integration exists
- ✓ Documentation isn't cluttering apple-bottom README
- ✓ Easy to maintain both together
- ✓ Users primarily want the library, not QE specifically

### Split to Multiple Repos If:
- ✓ Adding MEEP or 2+ integrations
- ✓ Want to distribute QE patches independently
- ✓ Different user bases (library users vs QE users)
- ✓ apple-bottom README becoming QE-focused instead of library-focused

---

## Current Status

**Structure:** ✓ Ready to split anytime
- docs/qe-integration.md is self-contained
- docs/fortran-integration.md is generic
- Tests are in apple-bottom repo
- QE source is outside repo

**Recommendation:** Stay single repo until:
1. MEEP integration starts, OR
2. apple-bottom README becomes cluttered, OR
3. You want QE users to clone apple-bottom-qe without apple-bottom source

---

## Next Steps

**When ready to split:**

1. Create `apple-bottom-qe` repo on GitHub
2. Run extraction checklist above
3. Update apple-bottom README with links
4. Test installation from scratch:
   ```bash
   # Clean machine test
   git clone https://github.com/grantdh/apple-bottom
   git clone https://github.com/grantdh/apple-bottom-qe
   cd apple-bottom && make && make test
   cd ../apple-bottom-qe
   ./scripts/install_qe.sh
   ./scripts/build_qe.sh
   ./scripts/run_benchmark.sh
   ```

The structure is ready — split whenever complexity justifies it!
