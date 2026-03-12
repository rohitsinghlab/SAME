#!/bin/bash
# ==============================================================================
# SAME ISS Heart — kNN Parameter Sweep
# ==============================================================================
# Reproduces: Fig S4 (kNN analysis, dp=5 fixed, MS=1)
#
# Usage:
#   bash run_parameter_sweep.sh
# ==============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo "Running kNN sweep (dp=5, MS=1)..."
for KNN in 1 2 4 6 8 10; do
    echo "  knn=${KNN}"
    bash "${SCRIPT_DIR}/run_same.sh" --dp 5 --knn ${KNN} --ms 1
done

echo ""
echo "Running MS/dp sweep (knn=8)..."
for MS in 1 3 7; do
    for DP in 0 1 5 10 25 50; do
        echo "  MS=${MS}, dp=${DP}"
        bash "${SCRIPT_DIR}/run_same.sh" --dp ${DP} --knn 8 --ms ${MS}
    done
done

echo "All sweeps complete."
