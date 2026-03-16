# SAME — Data and Pre-computed Results

[![DOI](https://zenodo.org/badge/DOI/PLACEHOLDER.svg)](https://doi.org/PLACEHOLDER)

This archive contains input data and pre-computed SAME results for reproducing
all figures in:

> Pratapa et al. "SAME: Topology-flexible transforms enable robust integration
> of multimodal spatial omics." bioRxiv (2025).
> https://doi.org/10.1101/2025.07.12.664419

**Code**: https://github.com/rohitsinghlab/SAME

## Usage

1. Clone the repository:
   ```bash
   git clone https://github.com/rohitsinghlab/SAME.git
   cd SAME
   ```

2. Download and extract this archive into `examples/`:
   ```bash
   # Extract so that synthetic/, heart/, tongue/, luad/ folders
   # merge with the existing examples/ directory
   unzip SAME_data.zip -d examples/
   ```

3. Open any `reproduce_figures.ipynb` notebook to regenerate paper figures.

## Contents

### Synthetic Benchmark — Fig 2, S1
- `synthetic/data/ref.csv` — 411 reference cells, 3 cell types
- `synthetic/data/query.csv` — 372 query cells (transformed)
- `synthetic/data/ref.h5ad`, `query.h5ad` — AnnData versions
- `synthetic/data/ground_truth.csv` — Known cell correspondences
- `synthetic/data/quadrants.pkl` — Quadrant definitions for benchmarking
- `synthetic/results/` — matchedDF.csv, mc_align.pkl, mc_ref.pkl

### ISS Heart — Fig 3, S4–S7
- `heart/data/refAD_valis.csv` — ~1600 reference cells (ISS), 8 cell types
- `heart/data/queryAD_valis.csv` — ~1600 query cells (VALIS-aligned)
- `heart/results/dp10_knn8_MS1/` — Main result (dp=10, MS=1)
- `heart/results/dp{0,1,5,10,25,50}_knn8_MS3/` — Penalty sweep with metacell size 3
- `heart/results/dp{0,1,5,10,25,50}_knn8_MS7/` — Penalty sweep with metacell size 7
- `heart/results/mc_align_MS3.pkl`, `mc_ref_MS3.pkl` — Metacell objects for MS=3
- `heart/results/mc_align_MS7.pkl`, `mc_ref_MS7.pkl` — Metacell objects for MS=7
- `heart/results/noise_rn{0,20,40,60,80,100}/` — Noise robustness sweep
- `heart/results/results.csv` — Summary across parameter settings
- `heart/results/knn_results.csv` — kNN sweep summary

### MERSCOPE Tongue (Protein + RNA) — Fig 4, S9–S14
- `tongue/data/prot_df.csv` — 4671 protein cells (PCF query)
- `tongue/data/mer_df.csv` — 3608 RNA cells (MERSCOPE template)
- `tongue/data/pcfSubAD.h5ad` — PCF AnnData (30 protein markers)
- `tongue/data/merSubAD.h5ad` — MERSCOPE AnnData (300 genes)
- `tongue/results/dp10_knn8_MS1/` — matchedDF.csv, mc_align.pkl, mc_ref.pkl

### LUAD33 (Protein + Xenium) — Fig 5, S15–S19
- `luad/data/align_pcf.csv` — ~94K protein cells (PCF query)
- `luad/data/ref_xen.csv` — ~100K RNA cells (Xenium template)
- `luad/data/0325_pcf_annotated.h5ad` — PCF AnnData (33 protein markers)
- `luad/data/0325_tsu33_annotated.h5ad` — Xenium AnnData (300+ genes)
- `luad/results/mc_align.pkl`, `mc_ref.pkl` — Metacell objects (shared across dp)
- `luad/results/dp{0,1,5,10,25,50}_knn8_MS3/matchedDF.csv` — Penalty sweep results

## File Formats

| Extension | Format | Description |
|-----------|--------|-------------|
| `.csv` | CSV | Cell coordinates, type probabilities, match results |
| `.h5ad` | AnnData (HDF5) | Single-cell data with expression matrices and annotations |
| `.pkl` | Python pickle | MetaCell objects (Delaunay triangulation + cell groupings) |

## Parameters

Each dataset was run with `sliding_window_matching()` using lazy constraints.
See `run_same.sh` in each example folder for exact parameters.

| Dataset | Cells | Cell Types | Key Params |
|---------|-------|------------|------------|
| Synthetic | 411 + 372 | 3 | dp=100, knn=5, MS=1 |
| ISS Heart | ~1600 + ~1600 | 8 | dp=0–50, knn=8, MS=1/3/7 |
| Tongue | 3608 + 4671 | 5 | dp=10, knn=8, MS=1 |
| LUAD33 | ~100K + ~94K | 5 | dp=0–50, knn=8, MS=3 |

## License

MIT License. See https://github.com/rohitsinghlab/SAME/blob/main/LICENSE
