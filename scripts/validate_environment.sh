#!/bin/bash
# =============================================================================
# validate_environment.sh - Check that the build environment is suitable
# =============================================================================

set -e

echo "═══════════════════════════════════════════════════════════════════"
echo "apple-bottom Environment Validation"
echo "═══════════════════════════════════════════════════════════════════"
echo ""

ERRORS=0
WARNINGS=0

# Check macOS
echo -n "Checking macOS version... "
MACOS_VERSION=$(sw_vers -productVersion)
MAJOR_VERSION=$(echo "$MACOS_VERSION" | cut -d. -f1)
if [ "$MAJOR_VERSION" -ge 12 ]; then
    echo "✓ $MACOS_VERSION"
else
    echo "✗ $MACOS_VERSION (need 12+)"
    ERRORS=$((ERRORS + 1))
fi

# Check architecture
echo -n "Checking architecture... "
ARCH=$(uname -m)
if [ "$ARCH" = "arm64" ]; then
    echo "✓ Apple Silicon ($ARCH)"
else
    echo "✗ $ARCH (need arm64)"
    ERRORS=$((ERRORS + 1))
fi

# Check Xcode Command Line Tools
echo -n "Checking Xcode CLI tools... "
if xcode-select -p &>/dev/null; then
    echo "✓ $(xcode-select -p)"
else
    echo "✗ Not installed"
    echo "  Run: xcode-select --install"
    ERRORS=$((ERRORS + 1))
fi

# Check clang
echo -n "Checking clang... "
if command -v clang &>/dev/null; then
    CLANG_VERSION=$(clang --version | head -1)
    echo "✓ $CLANG_VERSION"
else
    echo "✗ Not found"
    ERRORS=$((ERRORS + 1))
fi

# Check Metal framework
echo -n "Checking Metal framework... "
if [ -d "/System/Library/Frameworks/Metal.framework" ]; then
    echo "✓ Found"
else
    echo "✗ Not found"
    ERRORS=$((ERRORS + 1))
fi

# Check Accelerate framework
echo -n "Checking Accelerate framework... "
if [ -d "/System/Library/Frameworks/Accelerate.framework" ]; then
    echo "✓ Found"
else
    echo "✗ Not found"
    ERRORS=$((ERRORS + 1))
fi

# Check for Metal device (may fail in CI without GPU)
echo -n "Checking Metal GPU... "
if system_profiler SPDisplaysDataType 2>/dev/null | grep -q "Metal"; then
    GPU_NAME=$(system_profiler SPDisplaysDataType 2>/dev/null | grep "Chipset Model" | head -1 | cut -d: -f2 | xargs)
    echo "✓ $GPU_NAME"
else
    echo "⚠ Cannot verify (may be CI environment)"
    WARNINGS=$((WARNINGS + 1))
fi

# Check make
echo -n "Checking make... "
if command -v make &>/dev/null; then
    echo "✓ $(make --version | head -1)"
else
    echo "✗ Not found"
    ERRORS=$((ERRORS + 1))
fi

# Summary
echo ""
echo "═══════════════════════════════════════════════════════════════════"
if [ $ERRORS -eq 0 ]; then
    echo "✓ Environment OK ($WARNINGS warnings)"
    exit 0
else
    echo "✗ $ERRORS errors, $WARNINGS warnings"
    echo "  Please fix the issues above before building."
    exit 1
fi
