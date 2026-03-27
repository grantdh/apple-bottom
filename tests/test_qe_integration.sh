#!/bin/bash
# =============================================================================
# test_qe_integration.sh - Validates apple-bottom QE integration
# =============================================================================
# Usage: ./tests/test_qe_integration.sh
#
# Prerequisites:
#   - QE source at ~/qe-test/q-e-qe-7.4.1
#   - Benchmark files at ~/qe-test/benchmark
#   - conda environment: qe-src
# =============================================================================

set -e
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

PASS=0
FAIL=0

check() {
    local desc="$1"
    local cmd="$2"
    local expected="$3"

    result=$(eval "$cmd" 2>/dev/null || echo "COMMAND_FAILED")

    if [[ "$result" == *"$expected"* ]]; then
        echo -e "${GREEN}✓${NC} $desc"
        ((PASS++))
    else
        echo -e "${RED}✗${NC} $desc"
        echo "  Expected: $expected"
        echo "  Got: $result"
        ((FAIL++))
    fi
}

check_count() {
    local desc="$1"
    local cmd="$2"
    local expected="$3"

    result=$(eval "$cmd" 2>/dev/null || echo "0")

    if [[ "$result" -eq "$expected" ]]; then
        echo -e "${GREEN}✓${NC} $desc (count: $result)"
        ((PASS++))
    else
        echo -e "${RED}✗${NC} $desc"
        echo "  Expected count: $expected"
        echo "  Got count: $result"
        ((FAIL++))
    fi
}

echo "=============================================="
echo "apple-bottom QE Integration Test Suite"
echo "=============================================="
echo ""

# ---------------------------------------------
# Step 0: Symlink verification
# ---------------------------------------------
echo "--- Step 0: Symlink ---"
SYMLINK_TARGET=$(readlink ~/apple-bottom 2>/dev/null || echo "NOT_A_SYMLINK")
check "~/apple-bottom symlink exists" "readlink ~/apple-bottom" "metal-algos"

# ---------------------------------------------
# Step 1: Library symbols
# ---------------------------------------------
echo ""
echo "--- Step 1: Library Symbols ---"
LIB=~/apple-bottom/build/libapplebottom.a

check "Library exists" "test -f $LIB && echo exists" "exists"
check "_ab_dgemm_ symbol (Fortran)" "nm $LIB | grep 'T _ab_dgemm_'" "_ab_dgemm_"
check "_ab_zgemm_ symbol (Fortran)" "nm $LIB | grep 'T _ab_zgemm_'" "_ab_zgemm_"
check "_ab_dgemm_blas symbol (C API)" "nm $LIB | grep 'T _ab_dgemm_blas'" "_ab_dgemm_blas"
check "_ab_zgemm_blas symbol (C API)" "nm $LIB | grep 'T _ab_zgemm_blas'" "_ab_zgemm_blas"

# Check for duplicate symbols
DUPE_COUNT=$(nm $LIB 2>/dev/null | grep "T _ab_print_stats" | wc -l | tr -d ' ')
if [[ "$DUPE_COUNT" -eq 1 ]]; then
    echo -e "${GREEN}✓${NC} No duplicate _ab_print_stats"
    ((PASS++))
else
    echo -e "${RED}✗${NC} Duplicate _ab_print_stats symbols: $DUPE_COUNT"
    ((FAIL++))
fi

# ---------------------------------------------
# Step 2: QE Modules/make.depend
# ---------------------------------------------
echo ""
echo "--- Step 2: QE make.depend ---"
QE=~/qe-test/q-e-qe-7.4.1

STALE=$(grep -c "apple_bottom_mod" $QE/Modules/make.depend 2>/dev/null || echo "0")
if [[ "$STALE" -eq 0 ]]; then
    echo -e "${GREEN}✓${NC} No stale apple_bottom_mod in make.depend"
    ((PASS++))
else
    echo -e "${RED}✗${NC} Stale apple_bottom_mod references in make.depend: $STALE"
    ((FAIL++))
fi

# ---------------------------------------------
# Step 3: QE source files are clean
# ---------------------------------------------
echo ""
echo "--- Step 3: QE Source Files ---"

check_count "becmod.f90 is clean" \
    "grep -c 'APPLE_BOTTOM\|ab_dgemm\|ab_zgemm' $QE/Modules/becmod.f90" 0

check_count "device_helper.f90 is clean" \
    "grep -c 'APPLE_BOTTOM\|apple_bottom' $QE/UtilXlib/device_helper.f90" 0

# ---------------------------------------------
# Step 4: cegterg.f90 patches
# ---------------------------------------------
echo ""
echo "--- Step 4: cegterg.f90 Patches ---"

check "EXTERNAL :: ab_zgemm declaration" \
    "grep 'EXTERNAL :: ab_zgemm' $QE/KS_Solvers/Davidson/cegterg.f90" "EXTERNAL :: ab_zgemm"

AB_CALLS=$(grep -c "CALL ab_zgemm" $QE/KS_Solvers/Davidson/cegterg.f90 2>/dev/null || echo "0")
if [[ "$AB_CALLS" -ge 12 ]]; then
    echo -e "${GREEN}✓${NC} cegterg.f90 has $AB_CALLS ab_zgemm calls (expected ≥12)"
    ((PASS++))
else
    echo -e "${RED}✗${NC} cegterg.f90 has only $AB_CALLS ab_zgemm calls (expected ≥12)"
    ((FAIL++))
fi

# ---------------------------------------------
# Step 5: make.inc configuration
# ---------------------------------------------
echo ""
echo "--- Step 5: make.inc ---"

check "DFLAGS has __APPLE_BOTTOM__" \
    "grep '__APPLE_BOTTOM__' $QE/make.inc" "__APPLE_BOTTOM__"

check "BLAS_LIBS has -lapplebottom" \
    "grep 'lapplebottom' $QE/make.inc" "lapplebottom"

# ---------------------------------------------
# Step 6: pw.x binary
# ---------------------------------------------
echo ""
echo "--- Step 6: pw.x Binary ---"

check "pw.x exists" "test -x $QE/bin/pw.x && echo executable" "executable"

# ---------------------------------------------
# Summary
# ---------------------------------------------
echo ""
echo "=============================================="
echo "Results: $PASS passed, $FAIL failed"
echo "=============================================="

if [[ $FAIL -eq 0 ]]; then
    echo -e "${GREEN}All checks passed!${NC}"
    echo ""
    echo "To run correctness validation:"
    echo "  cd ~/qe-test/benchmark"
    echo "  rm -rf tmp && mkdir -p tmp"
    echo "  time ~/qe-test/q-e-qe-7.4.1/bin/pw.x < si64.in > si64_test.out 2>&1"
    echo "  grep '!' si64_test.out"
    echo ""
    echo "Expected energy: -2990.44276157 Ry"
    exit 0
else
    echo -e "${RED}Some checks failed. Review output above.${NC}"
    exit 1
fi
