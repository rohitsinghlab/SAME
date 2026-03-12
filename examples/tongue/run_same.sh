#!/bin/bash
# ==============================================================================
# SAME Tongue — Protein + RNA Cross-Modal Alignment
# ==============================================================================
# Reproduces: Fig 4 (MERSCOPE RNA ↔ PCF Protein alignment)
# Dataset: Human tongue, 3608 RNA cells (MERSCOPE) + 4671 protein cells (PCF)
#          5 cell types
#
# Usage:
#   bash run_same.sh                     # default dp=10 (paper config)
#   bash run_same.sh --dp 5 --ms 3       # alternative
# ==============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
DATA_DIR="${SCRIPT_DIR}/data"
RESULTS_DIR="${SCRIPT_DIR}/results"

# --- Parse arguments ---
DP=10
KNN=8
MS=1

while [[ $# -gt 0 ]]; do
    case $1 in
        --dp) DP="$2"; shift 2;;
        --knn) KNN="$2"; shift 2;;
        --ms) MS="$2"; shift 2;;
        *) echo "Unknown: $1"; exit 1;;
    esac
done

# --- Fixed parameters ---
WINDOW_SIZE=4000
OVERLAP=300
MIN_CELLS=30
MAX_MATCHES=1
RADIUS=300
NO_MATCH_PENALTY=10000
PENALTY_COEFF=100
DIST_CT_COEFF=1
MIN_ANGLE_DEG=15
R_MAX=300
MIP_GAP=0.05
LAZY_FLIP_FRACTION=0.05

OUTDIR="${RESULTS_DIR}/dp${DP}_knn${KNN}_MS${MS}"

echo "=============================================="
echo "SAME Tongue (Protein + RNA)"
echo "  dp=${DP}, knn=${KNN}, MS=${MS}"
echo "  Output: ${OUTDIR}"
echo "=============================================="

mkdir -p "${OUTDIR}"

python3 -u - <<PYEOF
import sys
sys.path.insert(0, "${PROJECT_ROOT}")

import numpy as np
import pandas as pd
import pickle
import time
from src.metacell_utils import greedy_triangle_collapse
from src.same import init_gurobi_params, init_optim_params, sliding_window_matching

CELL_TYPES = ['Endothelial cells', 'Epithelial cells', 'Fibroblasts',
              'Lymphoid cells', 'Myeloid cells']

# Load data: protein = aligned (query), RNA = ref (template)
alignDF = pd.read_csv("${DATA_DIR}/prot_df.csv", index_col=0)
refDF = pd.read_csv("${DATA_DIR}/mer_df.csv", index_col=0)

# Coordinates
alignDF['X'] = alignDF['transformed_x']
alignDF['Y'] = alignDF['transformed_y']
refDF['X'] = refDF['transformed_x']
refDF['Y'] = refDF['transformed_y']

# Cell type from proportions (scale to 100 as in original)
alignDF[CELL_TYPES] = alignDF[CELL_TYPES] * 100
refDF[CELL_TYPES] = refDF[CELL_TYPES] * 100
alignDF['cell_type'] = alignDF[CELL_TYPES].idxmax(axis=1)
refDF['cell_type'] = refDF[CELL_TYPES].idxmax(axis=1)

print(f"Loaded: {len(refDF)} RNA (ref), {len(alignDF)} protein (query)")

# Metacells
mc_align = greedy_triangle_collapse(
    alignDF, cell_type_col='cell_type', original_idx_col='Cell_Num',
    x_col='X', y_col='Y', max_metacell_size=${MS}, r_max=${R_MAX},
    min_angle_deg=${MIN_ANGLE_DEG}, use_alpha_shape=False, return_object=True)
mc_ref = greedy_triangle_collapse(
    refDF, cell_type_col='cell_type', original_idx_col='Cell_Num',
    x_col='X', y_col='Y', max_metacell_size=${MS}, r_max=${R_MAX},
    min_angle_deg=${MIN_ANGLE_DEG}, use_alpha_shape=False, return_object=True)

# Parameters
gurobi_params = init_gurobi_params()
gurobi_params['mip_gap'] = ${MIP_GAP}
gurobi_params['lazy_allowed_flip_fraction'] = ${LAZY_FLIP_FRACTION}

optim_params = init_optim_params()
optim_params.update({
    'window_size': ${WINDOW_SIZE}, 'overlap': ${OVERLAP},
    'min_cells_per_window': ${MIN_CELLS}, 'max_matches': ${MAX_MATCHES},
    'radius': ${RADIUS}, 'knn': ${KNN},
    'no_match_penalty': ${NO_MATCH_PENALTY}, 'penalty_coeff': ${PENALTY_COEFF},
    'dist_ct_coeff': ${DIST_CT_COEFF}, 'delaunay_penalty': ${DP},
    'cell_id_col': 'metacell_id', 'ref_metacell_match_multiplier': ${MS},
    'lazy_constraints': True, 'min_angle_deg': ${MIN_ANGLE_DEG},
})

start = time.time()
matches_df = sliding_window_matching(
    mc_ref, mc_align, outprefix="${OUTDIR}",
    optim_params=optim_params, gurobi_params=gurobi_params,
    ignore_precomputed_triangulation=False)
elapsed = time.time() - start

pickle.dump(mc_align, open("${OUTDIR}/mc_align.pkl", 'wb'))
pickle.dump(mc_ref, open("${OUTDIR}/mc_ref.pkl", 'wb'))
print(f"\nSAME completed in {elapsed:.1f}s ({elapsed/60:.1f}min)")
print(f"Matches: {len(matches_df)}")
PYEOF

echo "Done."
