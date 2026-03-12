#!/bin/bash
# ==============================================================================
# SAME Synthetic 4-Quadrant Benchmark — Reproducibility Script
# ==============================================================================
# Reproduces: Fig 2, Fig S1
# Dataset: 411 template / 372 query cells, 3 cell types, 4 quadrant challenges
# Seed: 8899
#
# This script runs SAME on the synthetic benchmark dataset.
# For visualization/figure generation, use reproduce_figures.ipynb.
#
# Prerequisites:
#   - Python 3.10+ with packages: numpy, pandas, scipy, gurobipy, tqdm, networkx
#   - Valid Gurobi license (academic or commercial)
#   - SAME package (pip install -e . from SAME/ root)
#
# Usage:
#   bash run_same.sh [DELAUNAY_PENALTY]
#   bash run_same.sh          # default dp=10
#   bash run_same.sh 5        # sweep with dp=5
# ==============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
DATA_DIR="${SCRIPT_DIR}/data"
RESULTS_DIR="${SCRIPT_DIR}/results"

# --- Parameters ---
DELAUNAY_PENALTY=${1:-10}

# Metacell parameters
METACELL_SIZE=1          # No aggregation (1 cell = 1 metacell)
R_MAX=5                  # Max edge length for Delaunay filtering
MIN_ANGLE_DEG=5          # Min angle for degenerate triangle removal

# SAME optimization parameters
WINDOW_SIZE=100
OVERLAP=0
MIN_CELLS_PER_WINDOW=30
MAX_MATCHES=2
RADIUS=5
KNN=8
NO_MATCH_PENALTY=10000
DIST_CT_COEFF=1
PENALTY_COEFF=100
LAZY_CONSTRAINTS=true

# Gurobi parameters
TIME_LIMIT=7200
MIP_GAP=0.025
MIP_FOCUS=2
LAZY_FLIP_FRACTION=0.0

echo "=============================================="
echo "SAME Synthetic Benchmark"
echo "=============================================="
echo "Delaunay penalty: ${DELAUNAY_PENALTY}"
echo "Output: ${RESULTS_DIR}/dp${DELAUNAY_PENALTY}/"
echo "=============================================="

mkdir -p "${RESULTS_DIR}"

python3 -u - <<PYEOF
import sys
sys.path.insert(0, "${PROJECT_ROOT}")

import numpy as np
import pandas as pd
import pickle
import time
from pathlib import Path

from src.metacell_utils import greedy_triangle_collapse
from src.same import init_gurobi_params, init_optim_params, sliding_window_matching

# --- Load data ---
data_dir = Path("${DATA_DIR}")
ref_df = pd.read_csv(data_dir / 'ref.csv', index_col=0)
query_df = pd.read_csv(data_dir / 'query.csv', index_col=0)
print(f"Loaded: {len(ref_df)} template, {len(query_df)} query cells")

# --- Create metacells ---
mc_align = greedy_triangle_collapse(
    query_df, cell_type_col='cell_type', original_idx_col='cell_idx',
    x_col='X', y_col='Y',
    max_metacell_size=${METACELL_SIZE}, r_max=${R_MAX}, min_angle_deg=${MIN_ANGLE_DEG},
    use_alpha_shape=False, alpha=None, return_object=True)

mc_ref = greedy_triangle_collapse(
    ref_df, cell_type_col='cell_type', original_idx_col='cell_idx',
    x_col='X', y_col='Y',
    max_metacell_size=${METACELL_SIZE}, r_max=${R_MAX}, min_angle_deg=${MIN_ANGLE_DEG},
    use_alpha_shape=False, alpha=None, return_object=True)

print(f"Metacells: {len(mc_align.metacell_df)} aligned, {len(mc_ref.metacell_df)} ref")

# --- SAME parameters ---
gurobi_params = init_gurobi_params()
gurobi_params['mip_gap'] = ${MIP_GAP}
gurobi_params['lazy_allowed_flip_fraction'] = ${LAZY_FLIP_FRACTION}
gurobi_params['time_limit'] = ${TIME_LIMIT}
gurobi_params['mip_focus'] = ${MIP_FOCUS}

optim_params = init_optim_params()
optim_params.update({
    'window_size': ${WINDOW_SIZE},
    'overlap': ${OVERLAP},
    'min_cells_per_window': ${MIN_CELLS_PER_WINDOW},
    'max_matches': ${MAX_MATCHES},
    'radius': ${RADIUS},
    'knn': ${KNN},
    'no_match_penalty': ${NO_MATCH_PENALTY},
    'dist_ct_coeff': ${DIST_CT_COEFF},
    'min_angle_deg': ${MIN_ANGLE_DEG},
    'penalty_coeff': ${PENALTY_COEFF},
    'delaunay_penalty': ${DELAUNAY_PENALTY},
    'cell_id_col': 'metacell_id',
    'ref_metacell_match_multiplier': ${METACELL_SIZE},
    'ignore_same_type_triangles': False,
    'lazy_constraints': ${LAZY_CONSTRAINTS},
})

# --- Run SAME ---
results_dir = Path("${RESULTS_DIR}")
outprefix = str(results_dir / f"dp${DELAUNAY_PENALTY}")

start = time.time()
matches_df = sliding_window_matching(
    mc_ref, mc_align, outprefix=outprefix,
    optim_params=optim_params, gurobi_params=gurobi_params,
    ignore_precomputed_triangulation=False)
elapsed = time.time() - start

# --- Save results ---
pickle.dump(mc_align, open(results_dir / 'mc_align.pkl', 'wb'))
pickle.dump(mc_ref, open(results_dir / 'mc_ref.pkl', 'wb'))

# --- Report ---
print(f"\n{'='*50}")
print(f"SAME completed in {elapsed:.1f}s")
print(f"Matches: {len(matches_df)}")
ct_match = (matches_df['align_cell_type'] == matches_df['ref_cell_type']).mean() \
    if 'align_cell_type' in matches_df.columns else None
if ct_match is not None:
    print(f"Cell type accuracy: {ct_match:.1%}")
print(f"Results saved to: {outprefix}/")
PYEOF

echo "Done."
