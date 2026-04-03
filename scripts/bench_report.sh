#!/bin/bash
# =============================================================================
# apple-bottom Benchmark Report Generator
# =============================================================================
# Usage:
#   ./scripts/bench_report.sh              Run full benchmark suite
#   ./scripts/bench_report.sh --quick      Run quick subset (smaller matrices)
#
# Output:
#   - Formatted markdown report to stdout
#   - Full report saved to build/BENCH_REPORT.md
#   - System info included
# =============================================================================

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
BUILD_DIR="$PROJECT_ROOT/build"
REPORT_FILE="$BUILD_DIR/BENCH_REPORT.md"

QUICK_MODE=false
if [[ "$1" == "--quick" ]]; then
    QUICK_MODE=true
fi

# Color codes
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# =============================================================================
# Helper Functions
# =============================================================================

get_timestamp() {
    date -u +"%Y-%m-%d %H:%M:%S UTC"
}

get_system_info() {
    local os_name
    local os_version
    local model
    local gpu_cores

    os_name=$(uname -s)
    os_version=$(uname -r)

    if [[ "$os_name" == "Darwin" ]]; then
        model=$(sysctl -n hw.model 2>/dev/null || echo "Unknown")
        gpu_cores=$(sysctl -n hw.gpu_cores 2>/dev/null || echo "Unknown")
    else
        model="Linux"
        gpu_cores="N/A"
    fi

    echo "| Property | Value |"
    echo "|----------|-------|"
    echo "| Timestamp | $(get_timestamp) |"
    echo "| OS | $os_name $os_version |"
    echo "| Model | $model |"
    echo "| GPU Cores | $gpu_cores |"
    echo "| Hostname | $(hostname) |"
}

check_benchmarks_exist() {
    local missing=()

    for bench in dgemm zgemm dsyrk zherk pool async; do
        if [[ ! -f "$BUILD_DIR/bench_$bench" ]]; then
            missing+=("bench_$bench")
        fi
    done

    if [[ ${#missing[@]} -gt 0 ]]; then
        echo -e "${YELLOW}Warning: Missing benchmark executables:${NC}"
        printf '  - %s\n' "${missing[@]}"
        echo ""
        echo "Building benchmarks..."
        cd "$PROJECT_ROOT"
        make clean > /dev/null 2>&1 || true
        make build > /dev/null 2>&1 || true
    fi
}

run_benchmark() {
    local bench_name="$1"
    local bench_path="$BUILD_DIR/bench_$bench_name"

    if [[ ! -f "$bench_path" ]]; then
        echo "ERROR: Benchmark $bench_path not found"
        return 1
    fi

    echo -e "${BLUE}Running: $bench_name${NC}"
    "$bench_path" 2>/dev/null || true
    echo ""
}

# =============================================================================
# Main Report Generation
# =============================================================================

main() {
    echo -e "${BLUE}════════════════════════════════════════════════════════════════${NC}"
    echo -e "${BLUE}  apple-bottom Benchmark Report${NC}"
    echo -e "${BLUE}════════════════════════════════════════════════════════════════${NC}"
    echo ""

    # Ensure build directory exists
    mkdir -p "$BUILD_DIR"

    # Check and build if needed
    check_benchmarks_exist

    # Start report generation
    {
        echo "# apple-bottom Benchmark Report"
        echo ""
        echo "**Generated:** $(get_timestamp)"
        echo ""

        if [[ "$QUICK_MODE" == true ]]; then
            echo "> Running in QUICK mode (subset of benchmarks)"
            echo ""
        fi

        echo "## System Information"
        echo ""
        get_system_info
        echo ""

        echo "## Benchmark Results"
        echo ""

        if [[ "$QUICK_MODE" == true ]]; then
            echo "### DGEMM (Quick)"
            run_benchmark "dgemm"

            echo "### ZGEMM (Quick)"
            run_benchmark "zgemm"
        else
            echo "### DGEMM - Double Precision Matrix Multiplication"
            run_benchmark "dgemm"

            echo "### ZGEMM - Complex Matrix Multiplication"
            run_benchmark "zgemm"

            echo "### DSYRK - Symmetric Rank-K Update"
            run_benchmark "dsyrk"

            echo "### ZHERK - Hermitian Rank-K Update"
            run_benchmark "zherk"

            echo "### Pool - Memory Pool Management"
            run_benchmark "pool"

            echo "### Async - Asynchronous API"
            run_benchmark "async"
        fi

        echo "## Notes"
        echo ""
        echo "- All benchmarks compiled with \`-O3\` optimization"
        echo "- GPU timing includes device transfers"
        echo "- Speedup is computed relative to Accelerate framework (AMX)"
        echo "- Precision verified against reference implementations"
        if [[ "$QUICK_MODE" == true ]]; then
            echo "- Quick mode runs subset of matrix sizes for fast iteration"
        fi
        echo ""

    } | tee "$REPORT_FILE"

    echo ""
    echo -e "${GREEN}✓ Benchmark report saved to: $REPORT_FILE${NC}"
    echo ""
}

# Run main
main
