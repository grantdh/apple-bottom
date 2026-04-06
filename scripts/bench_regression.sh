#!/bin/bash
# =============================================================================
# apple-bottom Performance Regression Detection
# =============================================================================
# Compares current benchmark results against a saved baseline.
# Exits non-zero if any metric regresses beyond the configured threshold.
#
# Usage:
#   ./scripts/bench_regression.sh               Run and compare vs baseline
#   ./scripts/bench_regression.sh --save        Save current results as baseline
#   ./scripts/bench_regression.sh --threshold 15 Set regression threshold to 15%
#
# Baseline file: build/perf_baseline.json
# Results file:  build/perf_current.json
#
# Measured metrics per operation:
#   - GFLOP/s at each matrix size
#   - Speedup vs AMX
#   - Max element-wise error
# =============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
BUILD_DIR="$PROJECT_ROOT/build"
BASELINE_FILE="$BUILD_DIR/perf_baseline.json"
CURRENT_FILE="$BUILD_DIR/perf_current.json"
THRESHOLD=10  # Default: 10% regression threshold
SAVE_MODE=false
SIZES="256 512 1024 2048"
ITERS=3

# Parse args
while [[ $# -gt 0 ]]; do
    case "$1" in
        --save)      SAVE_MODE=true; shift ;;
        --threshold) THRESHOLD="$2"; shift 2 ;;
        --sizes)     SIZES="$2"; shift 2 ;;
        --iters)     ITERS="$2"; shift 2 ;;
        *)           echo "Unknown option: $1"; exit 1 ;;
    esac
done

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

# =============================================================================
# Benchmark Runner (outputs JSON)
# =============================================================================

run_benchmarks() {
    echo -e "${BLUE}Running performance benchmarks...${NC}"
    echo -e "  Sizes: $SIZES"
    echo -e "  Iterations: $ITERS"
    echo ""

    # Build if needed
    cd "$PROJECT_ROOT"
    make bench > /dev/null 2>&1 || { echo "Build failed"; exit 1; }

    local json="{"
    local first=true

    # Run each benchmark and capture structured output
    for bench in dgemm zgemm dsyrk zherk; do
        local bench_path="$BUILD_DIR/bench_$bench"
        if [[ ! -f "$bench_path" ]]; then
            echo -e "${YELLOW}Warning: $bench_path not found, skipping${NC}"
            continue
        fi

        echo -e "${BLUE}  Benchmarking $bench...${NC}"
        local output
        output=$("$bench_path" 2>&1) || true

        # Extract GFLOP/s values from the formatted output
        # Format expected: "  NNNN    │    GGGGG    │    GGGGG    │  S.SSx"
        local bench_json=""
        while IFS= read -r line; do
            # Match lines with matrix size and GFLOP/s data
            if echo "$line" | grep -qE '^\s+[0-9]+\s+│'; then
                local size amx_gflops gpu_gflops
                size=$(echo "$line" | awk -F'│' '{gsub(/[^0-9]/,"",$1); print $1}')
                amx_gflops=$(echo "$line" | awk -F'│' '{gsub(/[^0-9.]/,"",$2); print $2}')
                gpu_gflops=$(echo "$line" | awk -F'│' '{gsub(/[^0-9.]/,"",$3); print $3}')

                if [[ -n "$size" && -n "$gpu_gflops" ]]; then
                    if [[ -n "$bench_json" ]]; then bench_json+=","; fi
                    bench_json+="\"$size\":{\"gpu_gflops\":$gpu_gflops,\"amx_gflops\":$amx_gflops}"
                fi
            fi
        done <<< "$output"

        if [[ -n "$bench_json" ]]; then
            if [[ "$first" != true ]]; then json+=","; fi
            json+="\"$bench\":{"
            json+="$bench_json"
            json+="}"
            first=false
        fi
    done

    json+="}"

    # Add metadata
    local timestamp
    timestamp=$(date -u +"%Y-%m-%dT%H:%M:%SZ")
    local commit_hash
    commit_hash=$(git -C "$PROJECT_ROOT" rev-parse --short HEAD 2>/dev/null || echo "unknown")
    local device
    device=$(sysctl -n machdep.cpu.brand_string 2>/dev/null || echo "unknown")

    # Wrap in metadata
    local full_json="{\"timestamp\":\"$timestamp\",\"commit\":\"$commit_hash\",\"device\":\"$device\",\"threshold_pct\":$THRESHOLD,\"results\":$json}"

    echo "$full_json"
}

# =============================================================================
# Comparison Logic
# =============================================================================

compare_results() {
    local baseline="$1"
    local current="$2"

    echo ""
    echo -e "${BLUE}════════════════════════════════════════════════════════════════${NC}"
    echo -e "${BLUE}  Performance Regression Report${NC}"
    echo -e "${BLUE}════════════════════════════════════════════════════════════════${NC}"
    echo ""

    local baseline_commit current_commit
    baseline_commit=$(python3 -c "import json; print(json.load(open('$baseline'))['commit'])" 2>/dev/null || echo "?")
    current_commit=$(python3 -c "import json; print(json.load(open('$current'))['commit'])" 2>/dev/null || echo "?")

    echo "  Baseline: $baseline_commit"
    echo "  Current:  $current_commit"
    echo "  Threshold: ${THRESHOLD}% regression"
    echo ""

    local regressions=0

    # Use Python for JSON comparison (available on macOS)
    regressions=$(python3 << 'PYEOF'
import json, sys

threshold = float("$THRESHOLD")

with open("$baseline") as f:
    base = json.load(f)
with open("$current") as f:
    curr = json.load(f)

regressions = 0
base_r = base.get("results", {})
curr_r = curr.get("results", {})

for op in sorted(set(list(base_r.keys()) + list(curr_r.keys()))):
    print(f"  {op.upper()}:")
    base_op = base_r.get(op, {})
    curr_op = curr_r.get(op, {})

    for size in sorted(set(list(base_op.keys()) + list(curr_op.keys())), key=int):
        base_gf = base_op.get(size, {}).get("gpu_gflops", 0)
        curr_gf = curr_op.get(size, {}).get("gpu_gflops", 0)

        if base_gf > 0:
            pct_change = ((curr_gf - base_gf) / base_gf) * 100
            if pct_change < -threshold:
                status = f"\033[0;31m✗ REGRESSION ({pct_change:+.1f}%)\033[0m"
                regressions += 1
            elif pct_change > threshold:
                status = f"\033[0;32m↑ IMPROVEMENT ({pct_change:+.1f}%)\033[0m"
            else:
                status = f"  ({pct_change:+.1f}%)"
            print(f"    N={size:>5}: {base_gf:>7.0f} → {curr_gf:>7.0f} GFLOP/s {status}")
        else:
            print(f"    N={size:>5}: (new) {curr_gf:.0f} GFLOP/s")

    print()

print(regressions)
PYEOF
    ) 2>&1

    # Extract regression count (last line of output)
    local reg_count
    reg_count=$(echo "$regressions" | tail -1)
    # Print comparison output (everything except last line)
    echo "$regressions" | head -n -1

    if [[ "$reg_count" -gt 0 ]]; then
        echo -e "${RED}✗ $reg_count regression(s) detected (>${THRESHOLD}% slower)${NC}"
        return 1
    else
        echo -e "${GREEN}✓ No performance regressions detected${NC}"
        return 0
    fi
}

# =============================================================================
# Main
# =============================================================================

main() {
    mkdir -p "$BUILD_DIR"

    # Run benchmarks
    local results
    results=$(run_benchmarks)

    # Save current results
    echo "$results" > "$CURRENT_FILE"
    echo ""
    echo -e "${GREEN}Results saved to: $CURRENT_FILE${NC}"

    if [[ "$SAVE_MODE" == true ]]; then
        cp "$CURRENT_FILE" "$BASELINE_FILE"
        echo -e "${GREEN}✓ Baseline saved to: $BASELINE_FILE${NC}"
        echo ""
        echo "Future runs will compare against this baseline."
        return 0
    fi

    # Compare against baseline
    if [[ ! -f "$BASELINE_FILE" ]]; then
        echo ""
        echo -e "${YELLOW}No baseline found at $BASELINE_FILE${NC}"
        echo "Run with --save first to establish a baseline:"
        echo "  ./scripts/bench_regression.sh --save"
        echo ""
        echo "Saving current results as initial baseline..."
        cp "$CURRENT_FILE" "$BASELINE_FILE"
        echo -e "${GREEN}✓ Baseline established${NC}"
        return 0
    fi

    compare_results "$BASELINE_FILE" "$CURRENT_FILE"
}

main
