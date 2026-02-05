# Large Datasets Tutorial

This tutorial covers techniques for handling datasets with 10,000+ cells.

## Approach Overview

For large datasets, we combine two strategies:

1. **Metacells**: Reduce cell count by merging same-type triangles
2. **Sliding Windows**: Process regions independently and merge

## Option 1: Metacells Only

Best for: 5,000 - 50,000 cells with spatial homogeneity.

```python
from src import greedy_triangle_collapse, run_same, unpack_metacell_matches
from src import init_optim_params, init_gurobi_params

# Step 1: Create metacells
print("Creating metacells...")
mc_aligned, tri = greedy_triangle_collapse(
    aligned_df,
    max_metacell_size=3,
    r_max=500,           # Remove edges > 500 units
    min_angle_deg=15,    # Remove thin triangles
)
print(f"Reduced: {len(aligned_df)} -> {len(mc_aligned)} metacells")

# Step 2: Configure and run SAME
optim = init_optim_params(
    radius=300,
    knn=10,
    lazy_constraints=True,
    cell_id_col='Cell_Num',  # Metacells use Cell_Num
)

gurobi = init_gurobi_params(
    time_limit=7200,
    mip_gap=0.05,
    init_method='greedy',
)

matches, var_out = run_same(
    ref_df=ref_df,
    aligned_df=mc_aligned,
    commonCT=commonCT,
    optim_params=optim,
    gurobi_params=gurobi,
)

# Step 3: Unpack to individual cells
individual_matches = unpack_metacell_matches(
    matches, mc_aligned, ref_df
)
print(f"Individual matches: {len(individual_matches)}")
```

## Option 2: Sliding Windows Only

Best for: Wide spatial regions where metacells aren't effective.

```python
from src import sliding_window_matching, init_optim_params

optim = init_optim_params(
    window_size=1000,
    overlap=250,
    radius=250,
    knn=8,
    lazy_constraints=True,
)

matches = sliding_window_matching(
    ref=ref_df,
    moving=aligned_df,
    commonCT=commonCT,
    optim_params=optim,
    outprefix='results/'  # Enables resumption
)
```

## Option 3: Metacells + Sliding Windows (Best for Very Large Data)

Best for: 50,000+ cells.

```python
from src import greedy_triangle_collapse, sliding_window_matching
from src import unpack_metacell_matches, init_optim_params

# Create metacells for both datasets
mc_aligned, _ = greedy_triangle_collapse(
    aligned_df, max_metacell_size=3, r_max=500
)
mc_ref, _ = greedy_triangle_collapse(
    ref_df, max_metacell_size=3, r_max=500
)

print(f"Aligned: {len(aligned_df)} -> {len(mc_aligned)}")
print(f"Ref: {len(ref_df)} -> {len(mc_ref)}")

# Run sliding windows on metacells
optim = init_optim_params(
    window_size=1000,
    overlap=250,
    cell_id_col='Cell_Num',
)

metacell_matches = sliding_window_matching(
    ref=mc_ref,
    moving=mc_aligned,
    commonCT=commonCT,
    optim_params=optim,
    outprefix='results/'
)

# Unpack to individual cells
individual_matches = unpack_metacell_matches(
    metacell_matches, mc_aligned, mc_ref,
    aligned_df=aligned_df, ref_df=ref_df,
    strategy='nearest'
)
```

## Memory Management

### Monitor Memory Usage

```python
import psutil

def print_memory():
    process = psutil.Process()
    mem = process.memory_info().rss / 1024 / 1024 / 1024
    print(f"Memory: {mem:.2f} GB")

print_memory()  # Check before
matches = run_same(...)
print_memory()  # Check after
```

### Reduce Memory with Lazy Constraints

```python
# Lazy mode: O(n) memory instead of O(n*k^3)
optim = init_optim_params(lazy_constraints=True)
```

### Reduce KNN

```python
# Fewer neighbors = fewer constraints
optim = init_optim_params(knn=5)  # Default is 8
```

## Speed Optimization

### Use Warm Start

```python
gurobi = init_gurobi_params(init_method='greedy')
```

### Allow Earlier Termination

```python
gurobi = init_gurobi_params(
    mip_gap=0.10,      # Accept 10% gap
    time_limit=1800,   # 30 min per window
)
```

### Reduce Triangle Constraints

```python
optim = init_optim_params(
    ignore_same_type_triangles=True,  # Skip homogeneous triangles
    min_angle_deg=20,                  # Filter more thin triangles
)
```

## Resumption

Sliding window matching supports resumption:

```python
# First run (interrupted)
matches = sliding_window_matching(..., outprefix='results/')
# ^C (interrupted at window 5/10)

# Resume (automatically continues from window 6)
matches = sliding_window_matching(..., outprefix='results/')
# Continues from saved progress
```

## Parallel Processing

For very large datasets, run windows in parallel:

```python
# Manual parallelization (advanced)
from concurrent.futures import ProcessPoolExecutor

def process_window(window_bounds):
    x_min, x_max, y_min, y_max = window_bounds
    ref_subset = ref_df[
        (ref_df['X'] >= x_min) & (ref_df['X'] < x_max) &
        (ref_df['Y'] >= y_min) & (ref_df['Y'] < y_max)
    ]
    aligned_subset = aligned_df[
        (aligned_df['X'] >= x_min) & (aligned_df['X'] < x_max) &
        (aligned_df['Y'] >= y_min) & (aligned_df['Y'] < y_max)
    ]
    return run_same(ref_subset, aligned_subset, commonCT)

# Create window list
windows = [(x, x+1000, y, y+1000)
           for x in range(0, 5000, 750)
           for y in range(0, 5000, 750)]

# Process in parallel
with ProcessPoolExecutor(max_workers=4) as executor:
    results = list(executor.map(process_window, windows))
```

## Benchmarking Results

| Cells | Metacells | Windows | Time | Memory |
|-------|-----------|---------|------|--------|
| 5,000 | None | None | 5 min | 2 GB |
| 10,000 | 1,000 | None | 8 min | 3 GB |
| 50,000 | 5,000 | 10×10 | 30 min | 8 GB |
| 100,000 | 10,000 | 10×10 | 60 min | 12 GB |

*Results on 32-core server with 128 GB RAM*
