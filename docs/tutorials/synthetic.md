# Synthetic Benchmark (Fig 2, S1)

Reproduces **Fig 2** and **Fig S1** from the paper.

- **Source**: [`examples/synthetic/reproduce_figures.ipynb`](https://github.com/rohitsinghlab/SAME/tree/main/examples/synthetic/reproduce_figures.ipynb)

## Dataset

411 template / 372 query cells, 3 cell types (c1, c2, c3), arranged in 4 quadrants with distinct alignment challenges:

| Quadrant | Challenge | Template | Query |
|----------|-----------|----------|-------|
| Top-Left | Missing cell type (c3 absent in query) | 100 | 62 |
| Top-Right | Noisy class probabilities (~uniform) | 100 | 100 |
| Bottom-Right | Space tearing (point swaps + shear) | 100 | 100 |
| Bottom-Left | Topological split (1 ellipse → 2 rings) | 111 | 110 |

Data is pre-generated with `np.random.seed(8899)` and stored in `examples/synthetic/data/`.

## SAME Parameters

| Parameter | Value | Notes |
|-----------|-------|-------|
| `delaunay_penalty` | **10** | Paper figure value |
| `window_size` | 100 | Single window covers full dataset |
| `max_matches` | 2 | |
| `radius` | 5 | |
| `knn` | 8 | |
| `metacell_size` | 1 | No aggregation |
| `lazy_constraints` | True | |

## Reproducing from Scratch

```bash
# From SAME/examples/synthetic/
bash run_same.sh        # default dp=10 (paper value)
bash run_same.sh 25     # alternative penalty
```

## Paper Results (dp=10)

- **Cell type matching accuracy**: 100%
- **Triangle violations**: 46 / 372 (in space-tearing and topological split quadrants, as expected)

## Figures Produced

| Output File | Paper Figure |
|------------|-------------|
| `Fig2ac_benchmark_overview.svg` | Fig 2a,c |
| `Fig2e_SAME_alignment.svg` | Fig 2e |
| `Fig2e_zoom_bottom_left.svg` | Fig 2e (inset) |
| `Fig2_triangle_violations.svg` | Fig 2 (violations) |
| `FigS1a_missing_class.svg` | Fig S1a |
| `FigS1b_topological_split.svg` | Fig S1b |
| `FigS1c_space_tearing.svg` | Fig S1c |
| `FigS1d_noisy_probabilities.svg` | Fig S1d |
| `FigS_delaunay_triangulation.svg` | Supplementary |

## Interactive Notebook

<iframe src="synthetic_benchmark.html" width="100%" height="800px" style="border:none;"></iframe>
