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
- Run all 42 tests (test_precision + test_correctness)
- Run convergence study (V-2)
- Generate `build/CI_REPORT.txt` with test results and convergence data hash

**Expected output:**
```
✓ test_precision: PASS
✓ test_correctness: PASS
✓ test_convergence: PASS

GIT_SHA: <current-commit-sha>
TEST_TOTAL: 42
TEST_PASSED: 42
TEST_FAILED: 0
CONVERGENCE_POINTS: 7
STATUS: PASS
```

- [ ] `make ci-local` completed successfully
- [ ] All 42 tests passed
- [ ] Convergence study generated 7 data points (N ∈ {64, 128, 256, 512, 1024, 2048, 4096})
- [ ] Review `build/convergence_data.csv` for expected error ranges (Frobenius ~10⁻¹⁴ to 5×10⁻¹⁴)

### 3. GitHub Actions Validation

Check that CI passed on the latest commit:

- [ ] GitHub Actions: `build-check` job passed (compile-time verification)

**Check at:** `https://github.com/grantdh/apple-bottom/actions`

**Note:** GitHub Actions runners don't expose Metal GPU access, so CI only verifies compile-time correctness. Full GPU validation (48 tests + convergence study) must be run locally via `make ci-local` (see step 2 above).

### 4. Documentation Review

- [ ] `README.md` reflects current version features
- [ ] `docs/INTEGRATION.md` examples are up-to-date
- [ ] `docs/vv/VV_REPORT.md` references current validated baseline (if V&V changed)
- [ ] `docs/vv/PRECISION_ENVELOPE.md` validated range is accurate

### 5. Version Update

Update version numbers in:

- [ ] `CHANGELOG.md` — Add release date to `[Unreleased]` section
- [ ] (Optional) `include/apple_bottom.h` — Update version comment if present

Example CHANGELOG update:
```markdown
## [1.0.3] - 2026-04-01

### Added
- New feature X

### Fixed
- Bug Y
```

### 6. Commit CI Report (Optional but Recommended)

Commit the CI report as proof of validation:

```bash
git add build/CI_REPORT.txt
git commit -m "ci: local validation for v1.0.3"
git push origin main
```

This creates an audit trail showing when tests were last run and what the convergence data hash was.

---

## Release Process

Once the pre-release checklist is complete:

### 1. Create Git Tag

```bash
# Tag the release
git tag -a v1.0.3 -m "Release 1.0.3

- Summary of changes
- Key fixes or features
- Performance improvements"

# Push tag to GitHub
git push origin v1.0.3
```

### 2. Create GitHub Release

1. Go to https://github.com/grantdh/apple-bottom/releases/new
2. Select the tag you just created (`v1.0.3`)
3. Release title: `v1.0.3`
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
   git checkout -b hotfix-1.0.4 v1.0.3
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
   git merge hotfix-1.0.4
   git push origin main
   ```

4. Follow normal release process above to tag `v1.0.4`

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
   git tag -d v1.0.3              # Delete local tag
   git push origin :refs/tags/v1.0.3  # Delete remote tag
   ```
3. **Fix the issue** on main
4. **Create new release** with incremented version (e.g., `v1.0.4`)

---

## Questions?

- Review previous releases: https://github.com/grantdh/apple-bottom/releases
- Check GitHub Actions runs: https://github.com/grantdh/apple-bottom/actions
- See V&V documentation: `docs/vv/VV_REPORT.md`
