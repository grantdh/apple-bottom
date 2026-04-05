#!/bin/bash
# =============================================================================
# apple-bottom Performance Regression Detector
# =============================================================================
# Runs a suite of benchmarks, compares against a saved baseline, and reports
# any regressions beyond the noise threshold.
#
# Usage:
#   ./scripts/perf_regression.sh                Run benchmarks & compare to baseline
#   ./scripts/perf_regression.sh --save         Run benchmarks & save as new baseline
#   ./scripts/perf_regression.sh --threshold 10 Set regression threshold to 10%
#   ./scripts/perf_regression.sh --ci           Exit non-zero on regression (for CI)
#
# The baseline is stored in build/perf_baseline.json.
# This script must be run locally (requires Metal GPU access).
# =============================================================================

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
BUILD_DIR="$PROJECT_ROOT/build"
BASELINE_FILE="$BUILD_DIR/perf_baseline.json"
RESULTS_FILE="$BUILD_DIR/perf_results.json"
REPORT_FILE="$BUILD_DIR/PERF_REGRESSION.md"

SAVE_BASELINE=false
CI_MODE=false
THRESHOLD=15  # % regression threshold (default 15% to allow for thermal noise)
WARMUP_ITERS=2
BENCH_ITERS=5

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --save) SAVE_BASELINE=true; shift ;;
        --ci) CI_MODE=true; shift ;;
        --threshold) THRESHOLD=$2; shift 2 ;;
        *) echo "Unknown option: $1"; exit 1 ;;
    esac
done

# Color codes
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

# =============================================================================
# Build benchmark binary
# =============================================================================

BENCH_SRC="$BUILD_DIR/perf_bench.m"
BENCH_BIN="$BUILD_DIR/perf_bench"

cat > "$BENCH_SRC" << 'BENCH_EOF'
#import <Foundation/Foundation.h>
#import <mach/mach_time.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "apple_bottom.h"

static double get_time_ms(void) {
    static mach_timebase_info_data_t info;
    static dispatch_once_t once;
    dispatch_once(&once, ^{ mach_timebase_info(&info); });
    return (double)mach_absolute_time() * info.numer / info.denom / 1e6;
}

typedef struct {
    const char* name;
    int M, N, K;
    double gflops;
    double time_ms;
} BenchResult;

static BenchResult bench_dgemm(int M, int N, int K, int warmup, int iters) {
    ABMatrix A = ab_matrix_create(M, K);
    ABMatrix B = ab_matrix_create(K, N);
    ABMatrix C = ab_matrix_create(M, N);

    size_t a_count = (size_t)M * K;
    size_t b_count = (size_t)K * N;
    double* a_data = (double*)malloc(a_count * sizeof(double));
    double* b_data = (double*)malloc(b_count * sizeof(double));
    for (size_t i = 0; i < a_count; i++) a_data[i] = (double)(i % 97) / 97.0;
    for (size_t i = 0; i < b_count; i++) b_data[i] = (double)(i % 89) / 89.0;
    ab_matrix_upload(A, a_data, true);
    ab_matrix_upload(B, b_data, true);
    ab_matrix_zero(C);

    // Warmup
    for (int i = 0; i < warmup; i++) ab_dgemm(A, B, C);

    // Timed iterations
    double t0 = get_time_ms();
    for (int i = 0; i < iters; i++) ab_dgemm(A, B, C);
    double elapsed = (get_time_ms() - t0) / iters;

    double flops = 2.0 * M * N * K;
    double gflops = flops / (elapsed * 1e6);

    free(a_data); free(b_data);
    ab_matrix_destroy(A); ab_matrix_destroy(B); ab_matrix_destroy(C);

    BenchResult r = {0};
    r.M = M; r.N = N; r.K = K;
    r.gflops = gflops;
    r.time_ms = elapsed;
    return r;
}

static BenchResult bench_zgemm(int M, int N, int K, int warmup, int iters) {
    ABMatrix Ar = ab_matrix_create(M, K), Ai = ab_matrix_create(M, K);
    ABMatrix Br = ab_matrix_create(K, N), Bi = ab_matrix_create(K, N);
    ABMatrix Cr = ab_matrix_create(M, N), Ci = ab_matrix_create(M, N);

    size_t a_count = (size_t)M * K;
    size_t b_count = (size_t)K * N;
    double* data = (double*)malloc((a_count > b_count ? a_count : b_count) * sizeof(double));
    for (size_t i = 0; i < a_count; i++) data[i] = (double)(i % 97) / 97.0;
    ab_matrix_upload(Ar, data, true); ab_matrix_upload(Ai, data, true);
    for (size_t i = 0; i < b_count; i++) data[i] = (double)(i % 89) / 89.0;
    ab_matrix_upload(Br, data, true); ab_matrix_upload(Bi, data, true);
    ab_matrix_zero(Cr); ab_matrix_zero(Ci);
    free(data);

    for (int i = 0; i < warmup; i++) ab_zgemm(Ar, Ai, Br, Bi, Cr, Ci);

    double t0 = get_time_ms();
    for (int i = 0; i < iters; i++) ab_zgemm(Ar, Ai, Br, Bi, Cr, Ci);
    double elapsed = (get_time_ms() - t0) / iters;

    double flops = 8.0 * M * N * K;  // 3 real GEMMs + element-wise
    double gflops = flops / (elapsed * 1e6);

    ab_matrix_destroy(Ar); ab_matrix_destroy(Ai);
    ab_matrix_destroy(Br); ab_matrix_destroy(Bi);
    ab_matrix_destroy(Cr); ab_matrix_destroy(Ci);

    BenchResult r = {0};
    r.M = M; r.N = N; r.K = K;
    r.gflops = gflops;
    r.time_ms = elapsed;
    return r;
}

static BenchResult bench_batch_dgemm(int M, int N, int K, int batch_count, int warmup, int iters) {
    // Create batch_count sets of matrices
    ABMatrix* As = (ABMatrix*)malloc(batch_count * sizeof(ABMatrix));
    ABMatrix* Bs = (ABMatrix*)malloc(batch_count * sizeof(ABMatrix));
    ABMatrix* Cs = (ABMatrix*)malloc(batch_count * sizeof(ABMatrix));

    size_t a_count = (size_t)M * K;
    size_t b_count = (size_t)K * N;
    double* a_data = (double*)malloc(a_count * sizeof(double));
    double* b_data = (double*)malloc(b_count * sizeof(double));
    for (size_t i = 0; i < a_count; i++) a_data[i] = (double)(i % 97) / 97.0;
    for (size_t i = 0; i < b_count; i++) b_data[i] = (double)(i % 89) / 89.0;

    for (int j = 0; j < batch_count; j++) {
        As[j] = ab_matrix_create(M, K);
        Bs[j] = ab_matrix_create(K, N);
        Cs[j] = ab_matrix_create(M, N);
        ab_matrix_upload(As[j], a_data, true);
        ab_matrix_upload(Bs[j], b_data, true);
        ab_matrix_zero(Cs[j]);
    }

    // Warmup with batch API
    for (int w = 0; w < warmup; w++) {
        ABBatch batch = ab_batch_create();
        for (int j = 0; j < batch_count; j++)
            ab_batch_dgemm(batch, As[j], Bs[j], Cs[j]);
        ab_batch_commit(batch);
        ab_batch_wait(batch);
        ab_batch_destroy(batch);
    }

    // Timed
    double t0 = get_time_ms();
    for (int i = 0; i < iters; i++) {
        ABBatch batch = ab_batch_create();
        for (int j = 0; j < batch_count; j++)
            ab_batch_dgemm(batch, As[j], Bs[j], Cs[j]);
        ab_batch_commit(batch);
        ab_batch_wait(batch);
        ab_batch_destroy(batch);
    }
    double elapsed = (get_time_ms() - t0) / iters;

    double flops = 2.0 * M * N * K * batch_count;
    double gflops = flops / (elapsed * 1e6);

    free(a_data); free(b_data);
    for (int j = 0; j < batch_count; j++) {
        ab_matrix_destroy(As[j]); ab_matrix_destroy(Bs[j]); ab_matrix_destroy(Cs[j]);
    }
    free(As); free(Bs); free(Cs);

    BenchResult r = {0};
    r.M = M; r.N = N; r.K = K;
    r.gflops = gflops;
    r.time_ms = elapsed;
    return r;
}

int main(int argc, char* argv[]) {
    int warmup = 2, iters = 5;
    if (argc > 1) warmup = atoi(argv[1]);
    if (argc > 2) iters = atoi(argv[2]);

    ABStatus s = ab_init();
    if (s != AB_OK) { fprintf(stderr, "ab_init failed: %s\n", ab_status_string(s)); return 1; }

    printf("{\n");
    printf("  \"device\": \"%s\",\n", ab_device_name());
    printf("  \"warmup\": %d,\n", warmup);
    printf("  \"iters\": %d,\n", iters);
    printf("  \"benchmarks\": [\n");

    // DGEMM benchmarks: representative sizes
    int dgemm_sizes[][3] = {
        {256, 256, 256},
        {1024, 1024, 1024},
        {2048, 2048, 2048},
        {4096, 4096, 4096},
        // Tall-skinny (QE-like)
        {4096, 64, 4096},
        {8192, 128, 8192},
    };
    int n_dgemm = sizeof(dgemm_sizes) / sizeof(dgemm_sizes[0]);

    for (int i = 0; i < n_dgemm; i++) {
        BenchResult r = bench_dgemm(dgemm_sizes[i][0], dgemm_sizes[i][1], dgemm_sizes[i][2], warmup, iters);
        printf("    {\"op\": \"dgemm\", \"M\": %d, \"N\": %d, \"K\": %d, \"gflops\": %.2f, \"time_ms\": %.3f}%s\n",
               r.M, r.N, r.K, r.gflops, r.time_ms, ",");
    }

    // ZGEMM benchmarks
    int zgemm_sizes[][3] = {
        {256, 256, 256},
        {1024, 1024, 1024},
        {2048, 2048, 2048},
    };
    int n_zgemm = sizeof(zgemm_sizes) / sizeof(zgemm_sizes[0]);

    for (int i = 0; i < n_zgemm; i++) {
        BenchResult r = bench_zgemm(zgemm_sizes[i][0], zgemm_sizes[i][1], zgemm_sizes[i][2], warmup, iters);
        printf("    {\"op\": \"zgemm\", \"M\": %d, \"N\": %d, \"K\": %d, \"gflops\": %.2f, \"time_ms\": %.3f}%s\n",
               r.M, r.N, r.K, r.gflops, r.time_ms, ",");
    }

    // Batch DGEMM: 100 small GEMMs
    {
        BenchResult r = bench_batch_dgemm(128, 128, 128, 100, warmup, iters);
        printf("    {\"op\": \"batch_dgemm_100x128\", \"M\": %d, \"N\": %d, \"K\": %d, \"gflops\": %.2f, \"time_ms\": %.3f}\n",
               r.M, r.N, r.K, r.gflops, r.time_ms);
    }

    printf("  ]\n");
    printf("}\n");

    ab_shutdown();
    return 0;
}
BENCH_EOF

echo -e "${GREEN}Building benchmark binary...${NC}"
mkdir -p "$BUILD_DIR"
clang -O2 -framework Foundation -framework Metal -framework Accelerate \
    -I"$PROJECT_ROOT/include" \
    "$PROJECT_ROOT/src/apple_bottom.m" \
    "$BENCH_SRC" \
    -o "$BENCH_BIN" 2>&1

if [[ $? -ne 0 ]]; then
    echo -e "${RED}Build failed!${NC}"
    exit 1
fi

# =============================================================================
# Run benchmarks
# =============================================================================

echo -e "${GREEN}Running benchmarks (warmup=$WARMUP_ITERS, iters=$BENCH_ITERS)...${NC}"
"$BENCH_BIN" "$WARMUP_ITERS" "$BENCH_ITERS" > "$RESULTS_FILE"

if [[ $? -ne 0 ]]; then
    echo -e "${RED}Benchmark run failed!${NC}"
    exit 1
fi

echo -e "${GREEN}Results saved to $RESULTS_FILE${NC}"

# =============================================================================
# Save baseline or compare
# =============================================================================

if $SAVE_BASELINE; then
    cp "$RESULTS_FILE" "$BASELINE_FILE"
    echo -e "${GREEN}Baseline saved to $BASELINE_FILE${NC}"
    echo ""
    echo "Benchmark results:"
    cat "$RESULTS_FILE"
    exit 0
fi

# Compare against baseline
if [[ ! -f "$BASELINE_FILE" ]]; then
    echo -e "${YELLOW}No baseline found at $BASELINE_FILE${NC}"
    echo -e "${YELLOW}Run with --save first to create a baseline.${NC}"
    echo ""
    echo "Current results:"
    cat "$RESULTS_FILE"
    exit 0
fi

# =============================================================================
# Compare results using Python
# =============================================================================

REGRESSIONS=$(python3 << PYEOF
import json, sys

with open("$BASELINE_FILE") as f:
    baseline = json.load(f)
with open("$RESULTS_FILE") as f:
    current = json.load(f)

threshold = $THRESHOLD
regressions = 0
report_lines = []

report_lines.append("# Performance Regression Report")
report_lines.append("")
report_lines.append(f"**Device**: {current.get('device', 'unknown')}")
report_lines.append(f"**Baseline device**: {baseline.get('device', 'unknown')}")
report_lines.append(f"**Threshold**: {threshold}%")
report_lines.append("")
report_lines.append("| Operation | Size | Baseline GFLOP/s | Current GFLOP/s | Delta | Status |")
report_lines.append("|-----------|------|-----------------|-----------------|-------|--------|")

# Index baseline by a key
base_map = {}
for b in baseline["benchmarks"]:
    key = f"{b['op']}_{b['M']}x{b['N']}x{b['K']}"
    base_map[key] = b

for c in current["benchmarks"]:
    key = f"{c['op']}_{c['M']}x{c['N']}x{c['K']}"
    if key not in base_map:
        report_lines.append(f"| {c['op']} | {c['M']}x{c['N']}x{c['K']} | — | {c['gflops']:.1f} | NEW | — |")
        continue
    b = base_map[key]
    delta_pct = ((c["gflops"] - b["gflops"]) / b["gflops"]) * 100.0
    if delta_pct < -threshold:
        status = "REGRESSION"
        regressions += 1
    elif delta_pct > threshold:
        status = "IMPROVED"
    else:
        status = "OK"
    report_lines.append(f"| {c['op']} | {c['M']}x{c['N']}x{c['K']} | {b['gflops']:.1f} | {c['gflops']:.1f} | {delta_pct:+.1f}% | {status} |")

report_lines.append("")
if regressions > 0:
    report_lines.append(f"**{regressions} REGRESSION(S) DETECTED** (>{threshold}% slower than baseline)")
else:
    report_lines.append("No regressions detected.")

report = "\n".join(report_lines)
with open("$REPORT_FILE", "w") as f:
    f.write(report)

print(report)
sys.exit(1 if regressions > 0 else 0)
PYEOF
)
COMPARE_EXIT=$?

echo ""
echo "$REGRESSIONS"

if $CI_MODE && [[ $COMPARE_EXIT -ne 0 ]]; then
    echo -e "${RED}Performance regression detected! Failing CI.${NC}"
    exit 1
fi

exit 0
