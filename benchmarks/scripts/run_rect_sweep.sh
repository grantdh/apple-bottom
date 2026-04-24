#!/usr/bin/env bash
# run_rect_sweep.sh — REVIEW_04 Tranche A sweep runner.
# Runs bench_rect_{dgemm,zgemm} twice across {cpu,gpu,auto} AB_MODE values,
# emits CSVs under benchmarks/results/<date>-<sha>-rect/.
set -euo pipefail

cd "$(git rev-parse --show-toplevel)"
SHA=$(git rev-parse --short HEAD)
DATE=$(date +%Y-%m-%d)
OUT="benchmarks/results/${DATE}-${SHA}-rect"
mkdir -p "$OUT"

DGEMM_BIN=build/bench_rect_dgemm
ZGEMM_BIN=build/bench_rect_zgemm
[[ -x "$DGEMM_BIN" ]] || { echo "missing $DGEMM_BIN — run 'make bench-rect'"; exit 1; }
[[ -x "$ZGEMM_BIN" ]] || { echo "missing $ZGEMM_BIN — run 'make bench-rect'"; exit 1; }

for SWEEP in 1 2; do
  for MODE in cpu gpu auto; do
    for BLAS in dgemm zgemm; do
      BIN=$([ "$BLAS" = dgemm ] && echo "$DGEMM_BIN" || echo "$ZGEMM_BIN")
      CFG="benchmarks/configs/rect_${BLAS}.txt"
      CSV="$OUT/${BLAS}_${MODE}_sweep${SWEEP}.csv"
      echo "=== ${BLAS} mode=${MODE} sweep=${SWEEP} → ${CSV}"
      # AB_MODE=gpu stanza: override the min-dim floor so sub-32 shapes are
      # force-GPU'd. This isolates "what does the kernel achieve on this
      # shape in isolation" from "what would auto do", which is the whole
      # point of the gpu sweep. cpu and auto modes leave the floor intact.
      MIN_DIM_ARG=""
      [ "$MODE" = "gpu" ] && MIN_DIM_ARG="AB_MIN_GPU_DIM=0"
      env $MIN_DIM_ARG AB_MODE=$MODE "$BIN" \
        --config "$CFG" \
        --mode   "$MODE" \
        --runs   5 \
        --warmup 2 \
        --verify \
        --output "$CSV"
    done
  done
done

echo "Done. Results under $OUT"
