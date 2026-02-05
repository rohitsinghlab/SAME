# Quick Start

This guide will get you running SAME in under 5 minutes.

## Input Data Format

SAME requires two DataFrames with the following columns:

| Column | Required | Description |
|--------|----------|-------------|
| `X`, `Y` | Yes | Spatial coordinates |
| `cell_type` | Yes | Cell type annotation |
| `Cell_Num_Old` | Yes | Unique cell identifier |
| Cell type columns | Yes | Probability or one-hot columns (e.g., `TypeA`, `TypeB`) |

Example data structure:

```python
import pandas as pd

# Reference data (fixed)
ref_df = pd.DataFrame({
    'Cell_Num_Old': [0, 1, 2, 3],
    'X': [100, 150, 200, 250],
    'Y': [100, 120, 100, 130],
    'cell_type': ['TypeA', 'TypeB', 'TypeA', 'TypeB'],
    'TypeA': [0.9, 0.1, 0.85, 0.15],  # Probability columns
    'TypeB': [0.1, 0.9, 0.15, 0.85],
})

# Aligned/moving data (to be matched)
aligned_df = pd.DataFrame({
    'Cell_Num_Old': [0, 1, 2, 3, 4],
    'X': [105, 155, 205, 260, 180],
    'Y': [95, 125, 105, 125, 110],
    'cell_type': ['TypeA', 'TypeB', 'TypeA', 'TypeB', 'TypeA'],
    'TypeA': [0.88, 0.12, 0.9, 0.2, 0.82],
    'TypeB': [0.12, 0.88, 0.1, 0.8, 0.18],
})
```

## Basic Usage

### Simple Matching

```python
from src import run_same

# Run SAME optimization
matches, var_out = run_same(
    ref_df=ref_df,
    aligned_df=aligned_df,
    commonCT=['TypeA', 'TypeB'],  # Cell type columns
    outprefix='results/'
)

# View results
print(f"Found {len(matches)} matches")
print(matches[['aligned_idx', 'ref_idx', 'X', 'Y', 'ref_X', 'ref_Y']])
```

### With Custom Parameters

```python
from src import run_same, init_optim_params, init_gurobi_params

# Customize optimization parameters
optim = init_optim_params(
    radius=300,           # Search radius for KNN
    knn=10,               # Number of nearest neighbors
    max_matches=1,        # Each ref can be matched once
    lazy_constraints=True # Memory-efficient mode
)

# Customize Gurobi parameters
gurobi = init_gurobi_params(
    time_limit=3600,  # 1 hour timeout
    mip_gap=0.01      # 1% optimality gap
)

matches, var_out = run_same(
    ref_df=ref_df,
    aligned_df=aligned_df,
    commonCT=['TypeA', 'TypeB'],
    optim_params=optim,
    gurobi_params=gurobi
)
```

## Large Datasets (1000+ cells)

For larger datasets, use sliding windows:

```python
from src import sliding_window_matching, init_optim_params

optim = init_optim_params(
    window_size=1000,
    overlap=250,
    radius=250,
    lazy_constraints=True
)

matches = sliding_window_matching(
    ref=ref_df,
    moving=aligned_df,
    commonCT=['TypeA', 'TypeB'],
    optim_params=optim,
    outprefix='results/'
)
```

## Very Large Datasets (10,000+ cells)

For very large datasets, use metacells to reduce problem size:

```python
from src import greedy_triangle_collapse, run_same, unpack_metacell_matches

# Step 1: Create metacells
mc_aligned, tri = greedy_triangle_collapse(
    aligned_df,
    max_metacell_size=3,  # Max cells per metacell
    r_max=500,             # Remove long edges
)

print(f"Reduced from {len(aligned_df)} to {len(mc_aligned)} metacells")

# Step 2: Run SAME on metacells
matches, var_out = run_same(
    ref_df=ref_df,
    aligned_df=mc_aligned,
    commonCT=['TypeA', 'TypeB'],
    cell_id_col='Cell_Num'  # Metacells use Cell_Num
)

# Step 3: Unpack to individual cells
individual_matches = unpack_metacell_matches(
    matches,
    mc_aligned,
    ref_df
)
```

## Output Format

The `matches` DataFrame contains:

| Column | Description |
|--------|-------------|
| `aligned_idx` | Row index in aligned_df |
| `ref_idx` | Row index in ref_df |
| `X`, `Y` | Coordinates of aligned point |
| `ref_X`, `ref_Y` | Coordinates of matched ref point |
| `Aligned_Cell_Num_Old` | Original aligned cell ID |
| `Ref_Cell_Num_Old` | Original ref cell ID |
| `triangle_violation` | Whether point is in a flipped triangle |
| `run_time` | Optimization time (seconds) |

The `var_out` dictionary contains optimization diagnostics.

## Next Steps

- [Algorithm Overview](concepts/algorithm.md) - Understand how SAME works
- [Metacells](concepts/metacells.md) - Learn about graph simplification
- [Parameters](concepts/parameters.md) - Full parameter reference
- [Tutorials](tutorials/basic_usage.md) - Detailed examples
