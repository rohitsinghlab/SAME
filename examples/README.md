# Examples

Each folder contains a `reproduce_figures.ipynb` notebook and a `run_same.sh` script to reproduce paper figures.

## Data Download

Input data and pre-computed results are hosted on Zenodo:

**https://zenodo.org/records/19056915**

Download and extract into this directory:

```bash
cd examples/
wget https://zenodo.org/records/19056915/files/SAME_data.zip
unzip SAME_data.zip
```

This will populate the `data/` and `results/` folders for each dataset.

## Datasets

| Folder | Dataset | Paper Figures |
|--------|---------|---------------|
| `synthetic/` | Synthetic benchmark | Fig 2, S1 |
| `heart/` | ISS Heart (8 cell types) | Fig 3, S4–S7 |
| `tongue/` | Tongue protein + RNA | Fig 4, S9–S14 |
| `luad/` | LUAD33 protein + Xenium | Fig 5, S15–S19 |
