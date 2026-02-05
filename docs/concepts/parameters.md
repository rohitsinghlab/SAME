# Parameter Reference

Complete reference for all SAME parameters.

## Optimization Parameters (`init_optim_params`)

These parameters control the matching problem formulation.

### Sliding Window

| Parameter | Default | Description |
|-----------|---------|-------------|
| `window_size` | 1000 | Size of each window in coordinate units |
| `overlap` | 250 | Overlap between adjacent windows |
| `min_cells_per_window` | 10 | Minimum cells required to process a window |

### Matching Problem

| Parameter | Default | Description |
|-----------|---------|-------------|
| `max_matches` | 1 | Maximum times a reference point can be matched |
| `ref_metacell_match_multiplier` | None | For ref metacells, multiplier for max_matches. None = use max metacell size |
| `radius` | 250 | Search radius for finding KNN candidates |
| `knn` | 8 | Number of nearest neighbors to consider |

### Objective Coefficients

| Parameter | Default | Description |
|-----------|---------|-------------|
| `penalty_coeff` | 100 | Penalty for reference matched more than once |
| `no_match_penalty` | 100 | Penalty for unmatched aligned point (per cell) |
| `delaunay_penalty` | 5 | Penalty for triangle orientation flip |
| `dist_ct_coeff` | 1 | Weight for cell type distance in objective |

!!! tip "Tuning penalties"
    - Increase `delaunay_penalty` to discourage space-tearing
    - Decrease `delaunay_penalty` to allow more flexible matching
    - Increase `no_match_penalty` to encourage more matches

### Constraint Flags

| Parameter | Default | Description |
|-----------|---------|-------------|
| `hard_spatial_constraints` | False | If True, strictly forbid triangle flips |
| `ignore_same_type_triangles` | True | Skip constraints for homogeneous triangles |
| `ignore_knn_if_matched` | False | Use cell-type priority in KNN search |
| `lazy_constraints` | True | Add spatial constraints lazily (memory efficient) |

### Triangle Quality

| Parameter | Default | Description |
|-----------|---------|-------------|
| `min_angle_deg` | 15 | Filter triangles with minimum angle below threshold |

### Output

| Parameter | Default | Description |
|-----------|---------|-------------|
| `cell_id_col` | 'Cell_Num_Old' | Column name for cell identifiers |

## Gurobi Parameters (`init_gurobi_params`)

These parameters control the Gurobi solver.

### Core Controls

| Parameter | Default | Description |
|-----------|---------|-------------|
| `time_limit` | 7200 | Maximum solve time in seconds (2 hours) |
| `mip_gap` | 0.05 | MIP optimality gap tolerance (5%) |

### Solver Tuning

| Parameter | Default | Description |
|-----------|---------|-------------|
| `mip_focus` | 2 | MIPFocus: 1=find solutions, 2=prove optimality |
| `cuts` | 2 | Cut generation (0=none to 3=aggressive) |
| `heuristics` | 0.1 | Fraction of time on heuristics (0.0-1.0) |

### MIP Start

| Parameter | Default | Description |
|-----------|---------|-------------|
| `init_method` | None | Warm-start method: None, 'greedy', or 'hungarian' |
| `init_big_m` | 1e9 | Large cost for forbidden pairs in Hungarian init |
| `init_hungarian_max_n` | 5000 | Skip Hungarian if n_aligned + n_ref exceeds this |

### Lazy Constraints

| Parameter | Default | Description |
|-----------|---------|-------------|
| `lazy_max_cuts` | None | Global cap on lazy cuts (None = unlimited) |
| `lazy_allowed_flip_fraction` | 0.05 | Allowed fraction of flipped triangles |
| `lazy_max_cuts_per_incumbent` | 1000 | Max cuts per incumbent solution |

## Metacell Parameters (`greedy_triangle_collapse`)

| Parameter | Default | Description |
|-----------|---------|-------------|
| `max_metacell_size` | 3 | Maximum cells per metacell |
| `max_iterations` | 1000 | Maximum collapse iterations |
| `r_max` | None | Maximum edge length (removes long edges) |
| `min_angle_deg` | 10 | Minimum triangle angle in degrees |
| `use_alpha_shape` | False | Filter triangles to alpha complex |
| `alpha` | 0.05 | Alpha parameter (smaller = tighter boundary) |

## Common Configurations

### Small Dataset (<1000 cells)

```python
optim = init_optim_params(
    radius=500,
    knn=10,
    lazy_constraints=False,  # Eager mode OK for small data
)
gurobi = init_gurobi_params(
    time_limit=1800,  # 30 minutes
    mip_gap=0.01,     # 1% gap
)
```

### Medium Dataset (1000-10000 cells)

```python
optim = init_optim_params(
    window_size=1000,
    overlap=250,
    radius=250,
    knn=8,
    lazy_constraints=True,
)
gurobi = init_gurobi_params(
    time_limit=3600,  # 1 hour per window
    mip_gap=0.05,
)
```

### Large Dataset (>10000 cells)

```python
# Use metacells
mc_aligned, _ = greedy_triangle_collapse(
    aligned_df,
    max_metacell_size=3,
    r_max=500,
)

optim = init_optim_params(
    window_size=1000,
    overlap=250,
    radius=250,
    lazy_constraints=True,
    cell_id_col='Cell_Num',
)
gurobi = init_gurobi_params(
    time_limit=7200,
    mip_gap=0.05,
    init_method='greedy',  # Warm start
)
```

### High Accuracy (more time)

```python
optim = init_optim_params(
    radius=300,
    knn=15,
    delaunay_penalty=50,  # Strong spatial preservation
    lazy_constraints=True,
)
gurobi = init_gurobi_params(
    time_limit=14400,  # 4 hours
    mip_gap=0.01,      # 1% gap
    mip_focus=2,       # Focus on optimality
)
```

### Fast Approximate (less time)

```python
optim = init_optim_params(
    radius=200,
    knn=5,
    delaunay_penalty=1,  # Allow more space-tearing
    lazy_constraints=True,
)
gurobi = init_gurobi_params(
    time_limit=600,    # 10 minutes
    mip_gap=0.10,      # 10% gap OK
    mip_focus=1,       # Find solutions quickly
    heuristics=0.3,    # More heuristics
)
```
