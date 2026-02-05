# Metacells

Metacells are a graph simplification technique that reduces problem size while preserving spatial constraints.

## Motivation

In spatial omics data, regions with homogeneous cell types have many same-type triangles in the Delaunay triangulation. These triangles:

1. Provide redundant spatial constraints
2. Increase memory and computation time
3. Can be safely collapsed without losing important structure

## How Metacells Work

### Step 1: Build Delaunay Triangulation

Compute Delaunay triangulation on the input cells.

### Step 2: Identify Same-Type Triangles

Find triangles where all three vertices have the same cell type:

```
   A (TypeA)
   /\
  /  \
 /    \
B------C
(TypeA) (TypeA)
```

These triangles represent "homogeneous" regions.

### Step 3: Collapse Triangles

Iteratively merge same-type triangles:

1. Select a same-type triangle
2. Check if merging would exceed `max_metacell_size`
3. If valid, merge the three cells into one metacell
4. Update coordinates to centroid
5. Average cell type probabilities
6. Track original cell members
7. Repeat until no more valid merges

### Step 4: Update Triangulation

Recompute Delaunay on metacells. The new triangulation:
- Has fewer vertices (metacells instead of cells)
- Preserves boundary structure (mixed-type triangles kept)
- Reduces constraint count significantly

## Usage

### Basic Metacell Creation

```python
from src import greedy_triangle_collapse

metacell_df, metacell_delaunay = greedy_triangle_collapse(
    aligned_df,
    max_metacell_size=3,  # Max cells per metacell
    r_max=500,             # Remove edges > 500 units
    min_angle_deg=15,      # Remove thin triangles
)

print(f"Reduced from {len(aligned_df)} to {len(metacell_df)} metacells")
```

### With Alpha Shape Filtering

```python
metacell_df, metacell_delaunay = greedy_triangle_collapse(
    aligned_df,
    max_metacell_size=3,
    r_max=500,
    use_alpha_shape=True,  # Filter to alpha complex
    alpha=0.05,            # Smaller = tighter boundary
)
```

### Using MetaCell Object

For more control, use `return_object=True`:

```python
mc = greedy_triangle_collapse(aligned_df, max_metacell_size=3, return_object=True)

# Access results
print(f"Original: {len(mc.original_df)} cells")
print(f"Metacells: {len(mc.metacell_df)} metacells")
print(f"Original triangles: {len(mc.original_delaunay)}")
print(f"Metacell triangles: {len(mc.metacell_delaunay)}")

# Get members of a metacell
members = mc.metacell_members(0)
print(f"Metacell 0 contains cells: {members}")
```

## Matching with Metacells

### Case 1: Metacells on Aligned Only (Recommended)

```python
from src import greedy_triangle_collapse, run_same, unpack_metacell_matches

# Create metacells for aligned data
mc_aligned, _ = greedy_triangle_collapse(aligned_df, max_metacell_size=3)

# Run SAME: metacells vs individual ref cells
matches, var_out = run_same(
    ref_df=ref_df,              # Individual cells
    aligned_df=mc_aligned,      # Metacells
    commonCT=commonCT,
    cell_id_col='Cell_Num',     # Metacells use Cell_Num
)

# Unpack to individual cells
individual_matches = unpack_metacell_matches(
    matches, mc_aligned, ref_df
)
```

### Case 2: Metacells on Both Sides (Fastest)

```python
# Create metacells for both datasets
mc_aligned, _ = greedy_triangle_collapse(aligned_df, max_metacell_size=3)
mc_ref, _ = greedy_triangle_collapse(ref_df, max_metacell_size=3)

# Run SAME on metacells
matches, var_out = run_same(
    ref_df=mc_ref,
    aligned_df=mc_aligned,
    commonCT=commonCT,
    cell_id_col='Cell_Num',
)

# Unpack using nearest neighbor strategy
individual_matches = unpack_metacell_matches(
    matches, mc_aligned, mc_ref,
    aligned_df=aligned_df,
    ref_df=ref_df,
    strategy='nearest'  # Required for both-metacells case
)
```

## Unpacking Strategies

### `strategy='distribute'` (default)

All cells in an aligned metacell are matched to the same reference cell/metacell.

- Simple and fast
- Works when reference has individual cells
- May create many-to-one matches

### `strategy='nearest'`

Each aligned cell is matched to its nearest reference cell within the matched metacells.

- Required when both sides have metacells
- Creates more natural one-to-one matches
- Slightly slower (requires distance computation)

## Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `max_metacell_size` | 3 | Maximum cells per metacell |
| `max_iterations` | 1000 | Maximum collapse iterations |
| `r_max` | None | Maximum edge length |
| `min_angle_deg` | 10 | Minimum triangle angle (degrees) |
| `use_alpha_shape` | False | Filter to alpha complex |
| `alpha` | 0.05 | Alpha parameter (smaller = tighter) |

## Tips

1. **Choose `max_metacell_size` based on cell density**: Higher density → larger metacells OK
2. **Set `r_max` to typical cell spacing**: Removes long edges across gaps
3. **Use `min_angle_deg=15`**: Filters degenerate thin triangles
4. **Enable `use_alpha_shape`**: Helps with complex boundaries
5. **Monitor reduction ratio**: Aim for 3-10x reduction
