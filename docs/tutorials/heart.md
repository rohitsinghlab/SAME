# ISS Heart — Serial Section Alignment

Reproduces **Fig 3** (a–c), **Fig S4**, **Fig S5**, **Fig S6**, **Fig S7** from the paper.

## Notebook

- **[Interactive HTML](heart_benchmark.html)** — rendered notebook with all outputs
- **Source**: [`examples/heart/reproduce_figures.ipynb`](https://github.com/rohitsinghlab/SAME/tree/main/examples/heart/reproduce_figures.ipynb)

## Dataset

Human embryonic heart, smFISH-based ISS profiling (VALIS-aligned).

- Template: ~3160 cells, Query: ~3160 cells
- 8 cell types: Atrial/Ventricular cardiomyocytes, Endothelium, Epicardium, Fibroblast, Smooth muscle cells, Schwan progenitors

## SAME Parameters (Main Result)

| Parameter | Value |
|-----------|-------|
| `delaunay_penalty` | **10** (paper Fig 3d) |
| `window_size` | 4000 |
| `overlap` | 100 |
| `max_matches` | 1 |
| `radius` | 50 |
| `knn` | 8 |
| `metacell_size` | 1 (also 3, 7 in sweep) |
| `lazy_constraints` | True |
| `ignore_same_type_triangles` | True |

## Reproducing from Scratch

```bash
# Single run (paper's main config)
bash run_same.sh --dp 10 --knn 8 --ms 1

# Full parameter sweep (Fig 3c, S6)
bash run_parameter_sweep.sh

# Noise robustness (Fig S5)
bash run_robustness.sh
```

## Paper Results

| Configuration | Cell Type Match | Triangle Violations |
|--------------|----------------|-------------------|
| dp=10, MS=1 (main) | 71.6% | 5.0% |
| dp=0, MS=1 | 73.0% | 12.1% |
| dp=0, MS=3 | 79.9% | 20.6% |
| dp=0, MS=7 | 85.2% | 25.5% |
| Initial (no SAME) | 57.6% | 0% |

## Figures Produced

| Output File | Paper Figure |
|------------|-------------|
| `Fig3ab_cell_types.svg` | Fig 3a,b |
| `Fig3c_accuracy_vs_violations.svg` | Fig 3c |
| `Fig3_spatial_alignment.svg` | Fig 3f (SAME panel) |
| `FigS4_knn_comparison.svg` | Fig S4 |
| `FigS5_noise_robustness.svg` | Fig S5 |
| `FigS6_heatmap_ms_dp.svg` | Fig S6a,b |
| `FigS6_time_heatmap.svg` | Fig S6c |
| `FigS7_nodes_violating.svg` | Fig S7 |

## Deferred

Fig 3d-f (method comparisons vs GPSA, PASTE2, STAlign, etc.) require benchmarking baselines from a separate machine.
