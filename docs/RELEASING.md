# Release Process

This document describes the process for creating a new release of apple-bottom.

---

## Pre-Release Checklist

Before tagging a new release, complete all items in this checklist:

### 1. Code Changes Complete

- [ ] All features/fixes for this release are merged to `main`
- [ ] No pending PRs that should be included
- [ ] `CHANGELOG.md` updated with release notes

### 2. Local Validation

Run the full CI validation suite locally:

```bash
make ci-local
```

This will:
- Run every test (precision, correctness, chaos, device-API, routing, BLAS-ABI, trace, staged path; `tests/README.md`, Coverage status, gives the counts and the counting rule) plus the golden-bits, ghost-detection, symbol and Fortran checks
- Run convergence study (V-2)
- Generate `build/CI_REPORT.txt` with test results and convergence data hash

**Expected output:**
```
✓ test_precision: PASS
✓ test_correctness: PASS
✓ test_chaos: PASS
✓ test_device_api: PASS
✓ test_dispatch: PASS
✓ test_blas_abi: PASS
✓ test_trace: PASS
✓ test_qe_golden: PASS
✓ test_staging: PASS
✓ test_convergence: PASS

GIT_SHA: <current-commit-sha>
AB_BUILD_SHA: <12-digit commit; a release build must not end in -dirty>
TEST_QE_GOLDEN: PASS (<n> cases bitwise equal to <the golden file's library>)
TEST_STAGING: PASS (<n> cases, the staged path against the host path)
TEST_TOTAL: <the total in tests/README.md>
TEST_PASSED: <the same>
TEST_FAILED: 0
DEVICE_API_CHECKS: <the CHECK count in tests/README.md>
BLAS_ABI_CASES: <the case count in tests/README.md> passed, 0 xfailed, 0 failed, 0 xpassed, 0 stale XFAIL entries
CONVERGENCE_POINTS: 7
STATUS: PASS
```

- [ ] `make ci-local` completed successfully
- [ ] All tests passed: `TEST_PASSED` = `TEST_TOTAL`, and that total, `DEVICE_API_CHECKS`
      and `BLAS_ABI_CASES` are the counts in `tests/README.md` (a lower count means tests
      stopped running; a change that adds or removes tests updates `tests/README.md`)
- [ ] `TEST_QE_GOLDEN: PASS`, not `SKIP`: the GPU-routed results are bitwise
      those of the library named in `tests/golden/qe_golden.txt`. A `SKIP`
      means the file comes from another device: re-baseline it on this
      machine from that library (`tests/README.md`, Golden bits) and rerun
      `make ci-local`
- [ ] Convergence study generated 7 data points (N ∈ {64, 128, 256, 512, 1024, 2048, 4096})
- [ ] Review `build/convergence_data.csv` for expected error ranges (Frobenius ~10⁻¹⁴ to 5×10⁻¹⁴)

### 3. GitHub Actions Validation

Check that CI passed on the latest commit:

- [ ] GitHub Actions: the `CI` workflow passed on macos-15 and macos-26

**Check at:** `https://github.com/grantdh/apple-bottom/actions`

**Note:** GitHub-hosted runners don't give jobs a usable Metal GPU, so CI runs only the checks that need none: `make version-check`, `make build-check`, `make lib`, `make test-symbols` and the routing unit tests (`build/test_dispatch`). The GPU suites and the convergence study run locally via `make ci-local` (step 2 above).

### 4. Documentation Review

- [ ] `README.md` reflects current version features
- [ ] `docs/INTEGRATION.md` examples are up-to-date
- [ ] `docs/vv/VV_REPORT.md` references current validated baseline (if V&V changed)
- [ ] `docs/vv/PRECISION_ENVELOPE.md` validated range is accurate
- [ ] If the BLAS-ABI GPU path or the kernels changed since the last
      calibration: `make calibrate` on an idle machine, and commit the
      regenerated `src/dispatch_table.h` with its
      `benchmarks/results/<date>-<sha>-calibrate/` directory

### 5. Version Update

The version lives in `include/apple_bottom.h` (`APPLE_BOTTOM_VERSION_MAJOR`,
`_MINOR`, `_PATCH` and `_STRING`); the Makefile reads it from there for the
dylib and the pkg-config file. Set it, then bring the other copies in line:

- [ ] `python/apple_bottom/__init__.py` (`__version__`) and `python/setup.py`
- [ ] `CITATION.cff` (`version` and `date-released: YYYY-MM-DD`)
- [ ] The citation block at the end of `README.md`
- [ ] `CHANGELOG.md` — rename the top `## [Unreleased]` section to `## [X.Y.Z] - YYYY-MM-DD`
- [ ] `make version-check` passes (it is also the first step of `make ci-local` and of CI)

Example CHANGELOG update:
```markdown
## [X.Y.Z] - YYYY-MM-DD

### Added
- New feature X

### Fixed
- Bug Y
```

### 6. Commit CI Report (Optional but Recommended)

Commit the CI report as proof of validation:

```bash
git add build/CI_REPORT.txt
git commit -m "ci: local validation for vX.Y.Z"
git push origin main
```

This creates an audit trail showing when tests were last run and what the convergence data hash was.

---

## Release Process

Once the pre-release checklist is complete:

### 1. Create Git Tag

```bash
# Tag the release
git tag -a vX.Y.Z -m "Release X.Y.Z

- Summary of changes
- Key fixes or features
- Performance improvements"

# Push tag to GitHub
git push origin vX.Y.Z
```

### 2. Create GitHub Release

1. Go to https://github.com/grantdh/apple-bottom/releases/new
2. Select the tag you just created (`vX.Y.Z`)
3. Release title: `vX.Y.Z`
4. Description: Copy relevant section from `CHANGELOG.md`
5. Attach artifacts (optional):
   - `build/libapplebottom.a` — Pre-built library
   - `build/CI_REPORT.txt` — Validation report
   - `build/convergence_data.csv` — Convergence data
6. Click "Publish release"

### 3. Post-Release

- [ ] Verify release appears on GitHub: https://github.com/grantdh/apple-bottom/releases
- [ ] Update `CHANGELOG.md` with `[Unreleased]` section for next version
- [ ] Announce release (if applicable)

---

## Hotfix Process

For critical bug fixes that need immediate release:

1. Create a hotfix branch from the release tag:
   ```bash
   git checkout -b hotfix-X.Y.W vX.Y.Z    # W = Z + 1
   ```

2. Make the fix, commit, and run validation:
   ```bash
   # Fix the bug
   git commit -m "fix: critical issue X"

   # Validate
   make ci-local
   ```

3. Merge hotfix to main:
   ```bash
   git checkout main
   git merge hotfix-X.Y.W
   git push origin main
   ```

4. Follow normal release process above to tag `vX.Y.W`

---

## Validation Failures

If `make ci-local` fails:

**Test failures:**
1. Investigate which test failed (check `build/*.log` files)
2. Fix the issue
3. Re-run `make ci-local`
4. Do NOT proceed with release until all tests pass

**Convergence study failures:**
1. Check `build/convergence_data.csv` exists
2. Verify all 7 data points generated
3. Compare Frobenius errors to expected ranges in `docs/vv/PRECISION_ENVELOPE.md`
4. If errors exceed validated envelope, investigate before releasing

**CI Report issues:**
1. If `CI_REPORT.txt` shows `TEST_FAILED: N` where N > 0, do NOT release
2. Re-run tests to confirm failure is reproducible
3. Fix failing tests before proceeding

---

## Version Numbering

apple-bottom uses semantic versioning: `MAJOR.MINOR.PATCH`

- **MAJOR**: Breaking API changes, major architecture changes
- **MINOR**: New features, backwards-compatible API additions
- **PATCH**: Bug fixes, performance improvements, documentation updates

Examples:
- `1.0.0` → `1.0.1`: Bug fix (PATCH)
- `1.0.1` → `1.1.0`: New feature like async API (MINOR)
- `1.1.0` → `2.0.0`: Breaking change like native API (MAJOR)

---

## Rollback Procedure

If a release has critical issues:

1. **Immediate:** Delete the GitHub release (does not delete the tag)
2. **Revert tag** (if needed):
   ```bash
   git tag -d vX.Y.Z              # Delete local tag
   git push origin :refs/tags/vX.Y.Z  # Delete remote tag
   ```
3. **Fix the issue** on main
4. **Create new release** with incremented version (e.g., `vX.Y.W`)

---

## Questions?

- Review previous releases: https://github.com/grantdh/apple-bottom/releases
- Check GitHub Actions runs: https://github.com/grantdh/apple-bottom/actions
- See V&V documentation: `docs/vv/VV_REPORT.md`
