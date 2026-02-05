# Basic Usage Tutorial

This tutorial walks through a complete SAME analysis on a small dataset.

## Setup

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

# Add SAME to path
import sys
sys.path.insert(0, str(Path.cwd().parent))

from src import run_same, init_optim_params, init_gurobi_params
```

## Load Example Data

```python
# Download example data (if not already present)
import pooch

DATA_URL = "https://zenodo.org/record/XXXXXX/files/"
data_cache = pooch.create(
    path=pooch.os_cache("same_data"),
    base_url=DATA_URL,
    registry={
        "example_ref.csv": "sha256:...",
        "example_aligned.csv": "sha256:...",
    }
)

ref_df = pd.read_csv(data_cache.fetch("example_ref.csv"))
aligned_df = pd.read_csv(data_cache.fetch("example_aligned.csv"))

print(f"Reference: {len(ref_df)} cells")
print(f"Aligned: {len(aligned_df)} cells")
```

## Visualize Input Data

```python
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# Reference
ax = axes[0]
for ct in ref_df['cell_type'].unique():
    subset = ref_df[ref_df['cell_type'] == ct]
    ax.scatter(subset['X'], subset['Y'], label=ct, s=10, alpha=0.7)
ax.set_title('Reference')
ax.legend()

# Aligned
ax = axes[1]
for ct in aligned_df['cell_type'].unique():
    subset = aligned_df[aligned_df['cell_type'] == ct]
    ax.scatter(subset['X'], subset['Y'], label=ct, s=10, alpha=0.7)
ax.set_title('Aligned (to be matched)')
ax.legend()

plt.tight_layout()
plt.show()
```

## Run SAME

```python
# Define cell type columns
commonCT = ['TypeA', 'TypeB', 'TypeC']

# Configure parameters
optim = init_optim_params(
    radius=300,           # Search radius
    knn=10,               # Nearest neighbors
    lazy_constraints=True # Memory efficient
)

gurobi = init_gurobi_params(
    time_limit=1800,  # 30 minutes max
    mip_gap=0.05      # 5% gap tolerance
)

# Run optimization
matches, var_out = run_same(
    ref_df=ref_df,
    aligned_df=aligned_df,
    commonCT=commonCT,
    optim_params=optim,
    gurobi_params=gurobi,
    outprefix='results/'
)

print(f"Found {len(matches)} matches")
```

## Analyze Results

```python
# Match rate
print(f"Match rate: {len(matches) / len(aligned_df) * 100:.1f}%")

# Cell type accuracy
matches['type_match'] = (
    matches['aligned_idx'].map(aligned_df.set_index(aligned_df.index)['cell_type']) ==
    matches['ref_idx'].map(ref_df.set_index(ref_df.index)['cell_type'])
)
print(f"Cell type accuracy: {matches['type_match'].mean() * 100:.1f}%")

# Spatial violations
n_violations = matches['triangle_violation'].sum()
print(f"Triangle violations: {n_violations} ({n_violations / len(matches) * 100:.1f}%)")
```

## Visualize Matches

```python
fig, ax = plt.subplots(figsize=(10, 10))

# Plot reference
ax.scatter(ref_df['X'], ref_df['Y'], c='blue', s=20, alpha=0.3, label='Reference')

# Plot aligned
ax.scatter(aligned_df['X'], aligned_df['Y'], c='red', s=20, alpha=0.3, label='Aligned')

# Draw match lines
for _, row in matches.iterrows():
    ax.plot([row['X'], row['ref_X']], [row['Y'], row['ref_Y']],
            'g-', alpha=0.3, linewidth=0.5)

ax.set_title(f'SAME Matches (n={len(matches)})')
ax.legend()
plt.show()
```

## Save Results

```python
# Save matches
matches.to_csv('results/matches.csv', index=False)

# Create aligned coordinates (SAME_X, SAME_Y)
aligned_with_same = aligned_df.copy()
aligned_with_same = aligned_with_same.merge(
    matches[['aligned_idx', 'ref_X', 'ref_Y']],
    left_index=True, right_on='aligned_idx', how='left'
)
aligned_with_same.rename(columns={'ref_X': 'SAME_X', 'ref_Y': 'SAME_Y'}, inplace=True)

# For unmatched cells, keep original coordinates
aligned_with_same['SAME_X'] = aligned_with_same['SAME_X'].fillna(aligned_with_same['X'])
aligned_with_same['SAME_Y'] = aligned_with_same['SAME_Y'].fillna(aligned_with_same['Y'])

aligned_with_same.to_csv('results/aligned_with_same_coords.csv', index=False)
```

## Next Steps

- [Large Datasets Tutorial](large_datasets.md) - Handle 10,000+ cells
- [Synthetic Benchmarks](synthetic.md) - Validate with ground truth
