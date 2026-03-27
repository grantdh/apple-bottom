# Parallel Workflow Setup

## Overview

You now have **two isolated workflows** set up to work in parallel without interfering with each other.

## Current State

### Main Branch (Protected Baseline)
- **Branch:** `main`
- **Commit:** `2728faf` - test: add QE integration validation script and update LESSONS_LEARNED
- **QE Directory:** `~/qe-test/q-e-qe-7.4.1` (WORKING, DO NOT MODIFY)
- **Library:** `~/apple-bottom/build/libapplebottom.a` (WORKING)
- **Performance:** 2:05 wall time, -2990.44276157 Ry ✓
- **Status:** ✅ LOCKED - Only merge to this after testing

## Workstream 1: Polish & Publish

### Purpose
Clean up documentation and project presentation for CERN application.

### Branch Info
- **Branch:** `publish`
- **Base:** `main` (commit `2728faf`)
- **QE Directory:** NONE (docs only, no code changes)
- **Risk:** ZERO

### Files to Work On
```
README.md              - Add QE benchmark section, update status
LESSONS_LEARNED.md     - Add QE integration section (already started)
CONTRIBUTING.md        - Update if needed
docs/                  - Any additional documentation
```

### Workflow
```bash
# Switch to publish branch
cd ~/Dev/arm/metal-algos
git checkout publish

# Make doc changes
# (edit README.md, LESSONS_LEARNED.md, etc.)

# Commit changes
git add README.md LESSONS_LEARNED.md
git commit -m "docs: polish for publication and CERN application"

# Review before merging
git diff main

# When happy, merge to main
git checkout main
git merge publish
```

### Safety
- No code changes → no risk of breaking working integration
- No QE directory → no build conflicts
- Can iterate freely on documentation

## Workstream 2: Native API Development

### Purpose
Implement GPU-resident matrix API to eliminate per-call upload/download overhead.

### Branch Info
- **Branch:** `native-api`
- **Base:** `main` (commit `2728faf`)
- **QE Directory:** `~/qe-test/q-e-native` (SANDBOX, experiment freely)
- **Risk:** ZERO (isolated sandbox)

### Files to Work On
```
src/native/
├── apple_bottom_native.h         - Public C API
├── metal_context_native.m        - ObjC implementation
├── apple_bottom_native_mod.f90   - Fortran module
└── test_native_api.c             - Standalone test

Makefile.native                    - Build system
NATIVE_API_PLAN.md                 - Development roadmap
```

### Workflow
```bash
# Switch to native-api branch
cd ~/Dev/arm/metal-algos
git checkout native-api

# Create scaffolding
mkdir -p src/native
# (add files: apple_bottom_native.h, metal_context_native.m, etc.)

# Commit scaffolding
git add src/native/ Makefile.native NATIVE_API_PLAN.md
git commit -m "feat: native API scaffolding for GPU-resident matrices"

# Build and test
make -f Makefile.native native
make -f Makefile.native test_native

# Patch QE in SANDBOX (not original!)
cd ~/qe-test/q-e-native/KS_Solvers/Davidson
# (edit cegterg.f90 to use native API)

# Build QE in SANDBOX
cd ~/qe-test/q-e-native
make pw -j8

# Test in SANDBOX
cd ~/qe-test/benchmark
~/qe-test/q-e-native/bin/pw.x < si64.in > si64_native.out 2>&1
grep '!' si64_native.out  # Must match -2990.44276157 Ry
```

### Safety
- Separate QE copy (`q-e-native`) → original QE untouched
- Separate branch (`native-api`) → main branch protected
- Can experiment freely without breaking working integration

## Branch Visualization

```
main (PROTECTED)
├── commit 2728faf
├── Working QE: ~/qe-test/q-e-qe-7.4.1 ✓
└── Performance: 2:05, correct energy ✓
    │
    ├─── publish (DOCS)
    │    ├── README.md updates
    │    ├── LESSONS_LEARNED.md polish
    │    └── No QE directory, no code changes
    │
    └─── native-api (EXPERIMENTS)
         ├── src/native/ (new API code)
         ├── Sandbox QE: ~/qe-test/q-e-native
         └── No changes to working QE
```

## Quick Commands

### Check Current Branch
```bash
git branch -v
```

### Switch Between Workflows
```bash
# Work on docs
git checkout publish

# Work on native API
git checkout native-api

# Return to baseline
git checkout main
```

### Verify Working State is Intact
```bash
# Should always pass (main branch)
git checkout main
./tests/test_qe_integration.sh

# Run quick benchmark
cd ~/qe-test/benchmark
time ~/qe-test/q-e-qe-7.4.1/bin/pw.x < si64.in > si64_check.out 2>&1
grep '!' si64_check.out  # Should be -2990.44276157 Ry
```

## Merge Strategy

### When publish branch is ready
```bash
git checkout main
git merge publish --no-ff -m "docs: merge polished documentation"
```

### When native-api branch is ready
```bash
# First, validate it works!
cd ~/qe-test/benchmark
~/qe-test/q-e-native/bin/pw.x < si64.in > si64_native_final.out 2>&1
grep '!' si64_native_final.out  # MUST match -2990.44276157 Ry

# Check performance improvement
# MUST be faster than 2:05 to justify the complexity

# Then merge
git checkout main
git merge native-api --no-ff -m "feat: native API for GPU-resident matrices"
```

## Safety Checklist

Before merging either branch to main:
- [ ] Run `./tests/test_qe_integration.sh` on main → must pass
- [ ] Run full si64 benchmark → energy must match
- [ ] Review diff: `git diff main..BRANCH_NAME`
- [ ] No conflicts with working integration

## Directory Isolation Summary

| Directory/File | Main | Publish | Native-API |
|----------------|------|---------|------------|
| `~/qe-test/q-e-qe-7.4.1/` | ✓ WORKING | Unused | Unused |
| `~/qe-test/q-e-native/` | N/A | N/A | ✓ SANDBOX |
| `~/apple-bottom/build/` | ✓ WORKING | Unused | Modified |
| `README.md` | Current | Polished | Current |
| `LESSONS_LEARNED.md` | Updated | Polished | Updated |
| `src/native/` | N/A | N/A | NEW |

## Notes

- **NEVER** modify `~/qe-test/q-e-qe-7.4.1` while on `native-api` branch
- **ALWAYS** use `~/qe-test/q-e-native` for native API experiments
- **ALWAYS** test on main branch before considering a branch "done"
- Keep `main` branch clean and deployable at all times
