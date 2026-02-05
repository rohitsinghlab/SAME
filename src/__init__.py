"""
SAME - Spatial Alignment for Multi-modal Experiments
=====================================================

SAME is a computational framework for aligning and integrating spatial omics
data across serial tissue sections and modalities. It uses Mixed Integer
Programming (MIP) to find optimal cell-to-cell matches while preserving
spatial relationships through Delaunay triangulation constraints.

Key Features
------------
- **Space-tearing transforms**: Enables controlled, localized topological
  disruptions during cross-sectional alignment
- **Metacell support**: Graph simplification for handling large datasets
- **Lazy constraints**: Memory-efficient optimization for large problems
- **Sliding window**: Process large spatial regions in overlapping windows

Quick Start
-----------
Basic matching:

>>> from src import run_same, init_optim_params
>>> matches, var_out = run_same(ref_df, aligned_df, commonCT=['TypeA', 'TypeB'])

With metacells for large datasets:

>>> from src import greedy_triangle_collapse, run_same, unpack_metacell_matches
>>> mc_aligned, tri = greedy_triangle_collapse(aligned_df, max_metacell_size=10)
>>> matches, _ = run_same(ref_df, mc_aligned, commonCT=commonCT)
>>> individual_matches = unpack_metacell_matches(matches, mc_aligned, ref_df)

Main Functions
--------------
run_same : Core optimization function
sliding_window_matching : Process large datasets in windows
greedy_triangle_collapse : Create metacells for graph simplification
unpack_metacell_matches : Convert metacell matches to individual cells
init_optim_params : Create optimization parameter dictionary
init_gurobi_params : Create Gurobi solver parameter dictionary

Requirements
------------
- gurobipy (requires Gurobi license - free for academics)
- numpy, pandas, scipy
- networkx (for window merging)
- Optional: alphashape, shapely (for alpha shape filtering)

Copyright (c) 2024-2025
"""

from .same import init_gurobi_params, init_optim_params, sliding_window_matching, run_same
from .helpers import merge_window_matches_unique_ref
from .metacell_utils import MetaCell, greedy_triangle_collapse, unpack_metacell_matches

__version__ = "0.1.0"
__all__ = [
    "init_gurobi_params",
    "init_optim_params",
    "sliding_window_matching",
    "run_same",
    "merge_window_matches_unique_ref",
    "MetaCell",
    "greedy_triangle_collapse",
    "unpack_metacell_matches"
]
