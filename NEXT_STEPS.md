# NEXT STEPS: Start P1 Implementation

## Current Status
✅ v1.0.1: ZHERK deprecation committed (9b15dfd)
✅ **P1 COMPLETE**: Transpose Variants for ZGEMM (c82b33e)
🔧 **NOW**: Optional - Create QE calbec benchmark or release v1.1.0

---

## Quick Start (Copy-Paste Friendly)

### Step 1A: Add ABTranspose enum (5 min)

Open `include/apple_bottom.h` and add after line 48 (after `ABStatus` enum):

```c
typedef enum {
    AB_NO_TRANS = 0,      // Use matrix as-is
    AB_TRANS = 1,         // Transpose (swap rows/cols)
    AB_CONJ_TRANS = 2     // Conjugate transpose A^H (for complex)
} ABTranspose;
```

### Step 1B: Declare ab_zgemm_ex (5 min)

Add after line 95 (`ab_zgemm` declaration):

```c
// Extended ZGEMM with transpose support (for QE compatibility)
ABStatus ab_zgemm_ex(
    ABTranspose transA, ABTranspose transB,
    ABMatrix Ar, ABMatrix Ai,
    ABMatrix Br, ABMatrix Bi,
    ABMatrix Cr, ABMatrix Ci
);
```

### Step 1C: Build and verify (2 min)

```bash
make clean && make
# Should compile cleanly

git add include/apple_bottom.h
git commit -m "Add ABTranspose enum for ZGEMM variants"
```

**After Step 1**: Continue to Step 2 in `QE_IMPLEMENTATION_PLAN.md`

---

## Full Implementation Checklist

Use this to track your progress:

- [x] **P0**: ZHERK deprecation
  - [x] Add deprecation attribute
  - [x] Test warning appears
  - [x] Commit v1.0.1

- [x] **P1 Step 1**: ABTranspose enum (15 min)
  - [x] Add enum to header
  - [x] Declare ab_zgemm_ex
  - [x] Build & commit (0f7d532)

- [x] **P1 Step 2**: GPU transpose kernels (2 hours)
  - [x] Add dd_conj_transpose kernel
  - [x] Add dd_transpose kernel
  - [x] Add to kShaderSource
  - [x] Build & test syntax

- [x] **P1 Step 3**: Pipeline compilation (30 min)
  - [x] Add transposePipeline property
  - [x] Add conjTransposePipeline property
  - [x] Compile in init method
  - [x] Test pipelines created (f2c875a)

- [x] **P1 Step 4**: Implement ab_zgemm_ex (1.5 hours)
  - [x] Handle transA parameter
  - [x] Create temporary buffers
  - [x] Dispatch transpose kernel
  - [x] Call existing ab_zgemm
  - [x] Cleanup temporaries
  - [x] Make ab_zgemm a wrapper (cfe688c)

- [x] **P1 Step 5**: Tests (1.5 hours)
  - [x] Add test_zgemm_conj_transpose
  - [x] Run tests (37/37 pass) (c82b33e)
  - [ ] Optional: Create bench_qe_calbec
  - [ ] Optional: Run benchmark (verify speedup)

- [ ] **Release**: v1.1.0
  - [ ] Update CHANGELOG.md
  - [ ] Update README.md (add QE support)
  - [ ] Git tag v1.1.0
  - [ ] Push to GitHub

---

## Time Budget

| Task | Estimate | Running Total |
|------|----------|---------------|
| Step 1 (enum) | 15 min | 15 min |
| Step 2 (kernels) | 2 hours | 2h 15min |
| Step 3 (pipelines) | 30 min | 2h 45min |
| Step 4 (ab_zgemm_ex) | 1.5 hours | 4h 15min |
| Step 5 (tests) | 1 hour | 5h 15min |
| **Total** | **~5 hours** | |

**Realistic timeline**: 1-2 days of focused work

---

## Testing Strategy

After each step, verify:

```bash
# Build cleanly
make clean && make

# Tests pass
make test

# No warnings
clang -Wall -Wextra -Iinclude -c tests/test_correctness.c 2>&1 | grep warning
```

---

## When You Get Stuck

**Reference documents**:
1. `QE_IMPLEMENTATION_PLAN.md` - Detailed implementation with full code
2. `transpose_design.md` - Design rationale
3. `v1.1_roadmap.md` - High-level plan

**Debug commands**:
```bash
# Check shader compile errors
make 2>&1 | grep -A 5 "error"

# Test specific function
clang -Iinclude -DTEST_ONLY tests/test_correctness.c -o /tmp/test -Lbuild -lapplebottom -framework Metal -framework Foundation -framework Accelerate
/tmp/test
```

---

## Expected Output (After P1 Complete)

```
$ make test

Running Tests
═══════════════════════════════════════════════════════════════════
./build/test_correctness

Transpose Variants:
  ab_zgemm_ex with conjugate transpose     ✓ PASS

Results: 37 passed, 0 failed
✓ All 37 tests PASSED
```

```
$ ./build/bench_qe_calbec

╔══════════════════════════════════════════════════════════════════╗
║  QE calbec Pattern Benchmark                                     ║
╚══════════════════════════════════════════════════════════════════╝

Simulating: ZGEMM('C', 'N', 256, 32, 4096, ...)
           (Conjugate-transpose × No-transpose)

  AMX Time:    18.5 ms  (580 GFLOP/s)
  GPU Time:    15.2 ms  (710 GFLOP/s)
  Speedup:     1.22x ✓
  Precision:   3.2e-11 ✓
```

---

## After P1: What Next?

1. **Update README** with QE-specific usage
2. **Post to QE mailing list** announcing Apple Silicon support
3. **Optional**: Implement P2 (DD-native workflow) for bigger wins
4. **Optional**: Profile actual QE Si64 benchmark with patch

---

## Questions?

Refer to:
- `QE_IMPLEMENTATION_PLAN.md` - Step-by-step guide
- `SUMMARY.md` - Project overview
- Your existing design docs in repo

**Ready to start?**

👉 **Begin with Step 1A above** (add ABTranspose enum)
