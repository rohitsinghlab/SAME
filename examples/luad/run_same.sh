#!/bin/bash
# Run SAME on LUAD33 dataset (PCF protein → Xenium RNA)
# Paper: Fig 5, S15–S18
#
# Default: dp=10 (main result used for cross-modal integration)
# For full dp sweep: run with --dp 0, --dp 1, --dp 5, --dp 10, --dp 25, --dp 50
#
# Usage:
#   bash run_same.sh              # dp=10 (default)
#   bash run_same.sh --dp 50      # specific dp value

set -euo pipefail

# --- Parse arguments ---
DP=10
KNN=8
MS=3

while [[ $# -gt 0 ]]; do
    case $1 in
        --dp)  DP="$2"; shift 2 ;;
        --knn) KNN="$2"; shift 2 ;;
        --ms)  MS="$2"; shift 2 ;;
        *)     echo "Unknown arg: $1"; exit 1 ;;
    esac
done

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
DATA_DIR="$SCRIPT_DIR/data"
OUT_DIR="$SCRIPT_DIR/results/dp${DP}_knn${KNN}_MS${MS}"

echo "=== LUAD33 SAME Run ==="
echo "  dp=$DP, knn=$KNN, MS=$MS"
echo "  Data: $DATA_DIR"
echo "  Output: $OUT_DIR"

mkdir -p "$OUT_DIR"

python - <<'PYTHON_SCRIPT'
import sys, argparse, time, pickle
import pandas as pd
import numpy as np

sys.path.insert(0, "$PROJECT_ROOT")
from src.metacell_utils import greedy_triangle_collapse
from src.same import init_gurobi_params, init_optim_params, sliding_window_matching

# --- Parameters from shell ---
DP = $DP
KNN = $KNN
MS = $MS
DATA_DIR = "$DATA_DIR"
OUT_DIR = "$OUT_DIR"

print(f"Running LUAD33: dp={DP}, knn={KNN}, MS={MS}")

# --- Load data ---
pcfDF = pd.read_csv(f'{DATA_DIR}/align_pcf.csv', index_col=0)
xenDF = pd.read_csv(f'{DATA_DIR}/ref_xen.csv', index_col=0)

commonCT = ['B cell', 'Epithelial', 'Mesenchymal', 'Myeloid', 'T cell']

pcfDF['Cell_Num_Old'] = pcfDF.index.values
xenDF['Cell_Num_Old'] = xenDF.index.values
pcfDF.index = pcfDF.Cell_Num_Old.values
xenDF.index = xenDF.Cell_Num_Old.values

pcfDF['cell_type'] = pcfDF[commonCT].idxmax(axis=1)
xenDF['cell_type'] = xenDF[commonCT].idxmax(axis=1)
pcfDF[commonCT] = pcfDF[commonCT] * 100
xenDF[commonCT] = xenDF[commonCT] * 100

print(f"Loaded: PCF={pcfDF.shape}, Xenium={xenDF.shape}")

# --- Metacells ---
mc_align = greedy_triangle_collapse(
    pcfDF, cell_type_col='cell_type', original_idx_col='Cell_Num_Old',
    x_col='X', y_col='Y', max_metacell_size=MS,
    r_max=250, min_angle_deg=15, use_alpha_shape=False, return_object=True)

mc_ref = greedy_triangle_collapse(
    xenDF, cell_type_col='cell_type', original_idx_col='Cell_Num_Old',
    x_col='X', y_col='Y', max_metacell_size=MS,
    r_max=250, min_angle_deg=15, use_alpha_shape=False, return_object=True)

# --- Gurobi + optim params ---
gurobi_params = init_gurobi_params()
gurobi_params['mip_gap'] = 0.05
gurobi_params['lazy_allowed_flip_fraction'] = 0.05

optim_params = init_optim_params()
optim_params['window_size'] = 13000
optim_params['overlap'] = 250
optim_params['min_cells_per_window'] = 30
optim_params['max_matches'] = 1
optim_params['radius'] = 250
optim_params['knn'] = KNN
optim_params['no_match_penalty'] = 10000
optim_params['dist_ct_coeff'] = 1
optim_params['penalty_coeff'] = 100
optim_params['delaunay_penalty'] = DP
optim_params['cell_id_col'] = 'metacell_id'
optim_params['ref_metacell_match_multiplier'] = MS

# --- Run ---
start = time.time()
outputDF = sliding_window_matching(
    mc_ref, mc_align, outprefix=OUT_DIR,
    optim_params=optim_params, gurobi_params=gurobi_params,
    ignore_precomputed_triangulation=False)
elapsed = time.time() - start

with open(f'{OUT_DIR}/../mc_align.pkl', 'wb') as f:
    pickle.dump(mc_align, f)
with open(f'{OUT_DIR}/../mc_ref.pkl', 'wb') as f:
    pickle.dump(mc_ref, f)

print(f"Done in {elapsed/3600:.2f}h — {outputDF.shape[0]} matches")
PYTHON_SCRIPT
