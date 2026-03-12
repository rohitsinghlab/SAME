#!/bin/bash
# ==============================================================================
# SAME ISS Heart — Noise Robustness Sweep
# ==============================================================================
# Reproduces: Fig S5 (robustness to phenotype labeling noise)
# Injects Dirichlet mixture noise into cell type compositions.
# noise=0.0: original, noise=1.0: fully random
#
# Usage:
#   bash run_robustness.sh
# ==============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
DATA_DIR="${SCRIPT_DIR}/data"
RESULTS_DIR="${SCRIPT_DIR}/results"

# Fixed params for robustness test
DP=10
KNN=8
MS=1
SEED=2026

for NOISE in 0.0 0.2 0.4 0.6 0.8 1.0; do
    RN_INT=$(python3 -c "print(int(round(${NOISE}*100)))")
    OUTDIR="${RESULTS_DIR}/noise_rn${RN_INT}"
    echo "=============================================="
    echo "Noise=${NOISE} (rn${RN_INT}), seed=${SEED}"
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
from src.robustness_utils import add_dirichlet_mixture_noise
from src.same import init_gurobi_params, init_optim_params, sliding_window_matching

CELL_TYPES = ['Smooth muscle cells', 'Fibroblast', 'Atrial cardiomyocytes',
              'Cardiomyocytes', 'Endothelium', 'Epicardium',
              'Schwan progenitors', 'Ventricular cardiomyocytes']

alignDF = pd.read_csv("${DATA_DIR}/queryAD_valis.csv")
refDF = pd.read_csv("${DATA_DIR}/refAD_valis.csv")
alignDF['X'] = alignDF['spot_x'] + 75
alignDF['Y'] = alignDF['spot_y'] + 75
refDF['X'] = refDF['spot_x'] + 75
refDF['Y'] = refDF['spot_y'] + 75
alignDF['cell_type'] = alignDF[CELL_TYPES].idxmax(axis=1)
refDF['cell_type'] = refDF[CELL_TYPES].idxmax(axis=1)

# Inject noise into query cell type compositions
rng = np.random.default_rng(${SEED})
alignDF = add_dirichlet_mixture_noise(
    alignDF, CELL_TYPES, ${NOISE}, target_sum=100.0, rng=rng, inplace=False)
alignDF['cell_type_noise'] = alignDF[CELL_TYPES].idxmax(axis=1)

mc_align = greedy_triangle_collapse(
    alignDF, cell_type_col='cell_type', original_idx_col='Cell_Num',
    x_col='X', y_col='Y', max_metacell_size=${MS}, r_max=50, min_angle_deg=15,
    use_alpha_shape=False, return_object=True)
mc_ref = greedy_triangle_collapse(
    refDF, cell_type_col='cell_type', original_idx_col='Cell_Num',
    x_col='X', y_col='Y', max_metacell_size=${MS}, r_max=50, min_angle_deg=15,
    use_alpha_shape=False, return_object=True)

gurobi_params = init_gurobi_params()
gurobi_params['mip_gap'] = 0.05
gurobi_params['lazy_allowed_flip_fraction'] = 0.05

optim_params = init_optim_params()
optim_params.update({
    'window_size': 4000, 'overlap': 100, 'min_cells_per_window': 30,
    'max_matches': 1, 'radius': 50, 'knn': ${KNN},
    'no_match_penalty': 10000, 'penalty_coeff': 100, 'dist_ct_coeff': 1,
    'delaunay_penalty': ${DP}, 'cell_id_col': 'metacell_id',
    'ref_metacell_match_multiplier': ${MS},
    'ignore_same_type_triangles': True, 'lazy_constraints': True,
    'min_angle_deg': 15,
})

start = time.time()
matches_df = sliding_window_matching(
    mc_ref, mc_align, outprefix="${OUTDIR}",
    optim_params=optim_params, gurobi_params=gurobi_params,
    ignore_precomputed_triangulation=False)
elapsed = time.time() - start

pickle.dump(mc_align, open("${OUTDIR}/mc_align.pkl", 'wb'))
pickle.dump(mc_ref, open("${OUTDIR}/mc_ref.pkl", 'wb'))
print(f"Noise={${NOISE}}: {len(matches_df)} matches in {elapsed:.1f}s")
PYEOF
done

echo "Robustness sweep complete."
