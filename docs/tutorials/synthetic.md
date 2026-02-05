# Synthetic Benchmarks Tutorial

This tutorial shows how to use SAME's synthetic data generation for benchmarking and validation.

## Overview

The synthetic benchmark creates a 4-quadrant grid with different challenges:

- **Top-Left**: Missing class (some cells removed in query)
- **Top-Right**: Noisy labels (uncertain cell type probabilities)
- **Bottom-Right**: Space tearing (point swaps + shear)
- **Bottom-Left**: Topological split (one region splits into two)

## Generate Benchmark Data

```python
from src.synthetic_datagen import (
    create_full_benchmark,
    visualize_benchmark,
    print_statistics,
    CLASS_NAMES,
    CLASS_COLORS
)
import numpy as np

# Set seed for reproducibility
np.random.seed(42)

# Create benchmark
ref_df, query_df, quadrants, ground_truth_df, extra_df = create_full_benchmark()

# Print statistics
print_statistics(ref_df, query_df, quadrants)
print(f"\nGround truth pairs: {len(ground_truth_df)}")
```

## Visualize Benchmark

```python
import matplotlib.pyplot as plt

fig, axes = plt.subplots(1, 2, figsize=(14, 7))

# Reference
ax = axes[0]
for ct in CLASS_NAMES:
    subset = ref_df[ref_df['cell_type'] == ct]
    ax.scatter(subset['X'], subset['Y'], c=CLASS_COLORS[ct],
               label=ct, s=20, alpha=0.7)
ax.set_title('Reference')
ax.legend()
ax.set_aspect('equal')

# Query
ax = axes[1]
for ct in CLASS_NAMES:
    subset = query_df[query_df['cell_type'] == ct]
    ax.scatter(subset['X'], subset['Y'], c=CLASS_COLORS[ct],
               label=ct, s=20, alpha=0.7)
ax.set_title('Query (to be matched)')
ax.legend()
ax.set_aspect('equal')

plt.tight_layout()
plt.show()
```

## Run SAME on Benchmark

```python
from src import run_same, init_optim_params

# Extract cell type columns
commonCT = CLASS_NAMES

# Configure SAME
optim = init_optim_params(
    radius=50,            # Small radius for grid data
    knn=5,
    lazy_constraints=True,
    delaunay_penalty=10,  # Penalize space-tearing
)

# Run SAME
matches, var_out = run_same(
    ref_df=ref_df,
    aligned_df=query_df,
    commonCT=commonCT,
    optim_params=optim,
)

print(f"Found {len(matches)} matches")
```

## Evaluate Against Ground Truth

```python
def evaluate_matches(matches, ground_truth, query_df, ref_df):
    """Compute matching accuracy metrics."""

    # Create lookup for ground truth
    gt_map = {}
    for _, row in ground_truth.iterrows():
        gt_map[row['query_idx']] = row['ref_idx']

    # Evaluate
    correct = 0
    total = 0

    for _, row in matches.iterrows():
        query_idx = row['aligned_idx']
        pred_ref = row['ref_idx']

        if query_idx in gt_map:
            total += 1
            if gt_map[query_idx] == pred_ref:
                correct += 1

    # Metrics
    accuracy = correct / total if total > 0 else 0
    recall = correct / len(ground_truth) if len(ground_truth) > 0 else 0
    precision = correct / len(matches) if len(matches) > 0 else 0

    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0,
        'correct': correct,
        'total_gt': len(ground_truth),
        'total_pred': len(matches),
    }

metrics = evaluate_matches(matches, ground_truth_df, query_df, ref_df)

print("\nMatching Metrics:")
print(f"  Accuracy: {metrics['accuracy']:.1%}")
print(f"  Precision: {metrics['precision']:.1%}")
print(f"  Recall: {metrics['recall']:.1%}")
print(f"  F1 Score: {metrics['f1']:.3f}")
```

## Per-Quadrant Analysis

```python
def evaluate_quadrant(matches, ground_truth, quadrant_bounds):
    """Evaluate matches within a specific quadrant."""
    x_min, x_max, y_min, y_max = quadrant_bounds

    # Filter to quadrant
    quad_matches = matches[
        (matches['X'] >= x_min) & (matches['X'] < x_max) &
        (matches['Y'] >= y_min) & (matches['Y'] < y_max)
    ]

    quad_gt = ground_truth[
        (ground_truth['X'] >= x_min) & (ground_truth['X'] < x_max) &
        (ground_truth['Y'] >= y_min) & (ground_truth['Y'] < y_max)
    ]

    return evaluate_matches(quad_matches, quad_gt, None, None)

# Define quadrant bounds
quadrant_names = {
    'Top-Left (Missing Class)': (-110, 0, 0, 110),
    'Top-Right (Noisy Labels)': (0, 110, 0, 110),
    'Bottom-Right (Space Tearing)': (0, 110, -110, 0),
    'Bottom-Left (Topological Split)': (-110, 0, -110, 0),
}

print("\nPer-Quadrant Results:")
for name, bounds in quadrant_names.items():
    metrics = evaluate_quadrant(matches, ground_truth_df, bounds)
    print(f"\n{name}:")
    print(f"  Accuracy: {metrics['accuracy']:.1%}")
    print(f"  F1: {metrics['f1']:.3f}")
```

## Visualize Space-Tearing Detection

```python
# Highlight matches with triangle violations
fig, ax = plt.subplots(figsize=(10, 10))

# Good matches
good_matches = matches[~matches['triangle_violation']]
ax.scatter(good_matches['X'], good_matches['Y'],
           c='green', s=30, alpha=0.5, label='Valid matches')

# Violated matches (space-tearing)
violated_matches = matches[matches['triangle_violation']]
ax.scatter(violated_matches['X'], violated_matches['Y'],
           c='red', s=50, marker='x', label='Space-tearing')

ax.set_title(f'Space-Tearing Detection ({len(violated_matches)} violations)')
ax.legend()
ax.set_aspect('equal')
plt.show()
```

## Compare Different Parameters

```python
def run_with_params(ref_df, query_df, commonCT, delaunay_penalty):
    """Run SAME with specific delaunay_penalty."""
    optim = init_optim_params(
        radius=50,
        knn=5,
        delaunay_penalty=delaunay_penalty,
        lazy_constraints=True,
    )
    matches, _ = run_same(ref_df, query_df, commonCT, optim_params=optim)
    return evaluate_matches(matches, ground_truth_df, query_df, ref_df)

# Compare different penalty values
penalties = [1, 5, 10, 50, 100]
results = []

for p in penalties:
    metrics = run_with_params(ref_df, query_df, commonCT, p)
    results.append({
        'penalty': p,
        'accuracy': metrics['accuracy'],
        'f1': metrics['f1'],
        'recall': metrics['recall'],
    })

# Plot results
import pandas as pd
results_df = pd.DataFrame(results)

fig, ax = plt.subplots(figsize=(8, 5))
ax.plot(results_df['penalty'], results_df['accuracy'], 'o-', label='Accuracy')
ax.plot(results_df['penalty'], results_df['f1'], 's-', label='F1')
ax.plot(results_df['penalty'], results_df['recall'], '^-', label='Recall')
ax.set_xlabel('Delaunay Penalty')
ax.set_ylabel('Score')
ax.set_title('Effect of Delaunay Penalty on Matching Quality')
ax.legend()
ax.set_xscale('log')
plt.show()
```

## Create Custom Benchmarks

```python
from src.synthetic_datagen import (
    create_diffeomorphic_grid,
    create_space_tearing_grid,
    create_topological_split,
)

# Create a custom grid with space tearing
ref_grid, query_grid, gt = create_space_tearing_grid(
    n_x=20,           # 20x20 grid
    n_y=20,
    spacing=10,
    n_classes=3,
    n_swaps=5,        # Number of point swaps
    shear_amount=0.2, # Shear deformation
)

print(f"Custom grid: {len(ref_grid)} ref, {len(query_grid)} query cells")
```
