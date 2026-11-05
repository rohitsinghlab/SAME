# Paper Figures

Interactive notebooks reproducing the main figures from the SAME paper.

## Synthetic Benchmark (Fig 2, S1)

411 template / 372 query cells, 3 cell types, 4 quadrants with distinct alignment challenges.

- [**Interactive Notebook**](https://rohitsinghlab.github.io/SAME/tutorials/synthetic_benchmark.html){:target="_blank"}
- [Source notebook](https://github.com/rohitsinghlab/SAME/tree/main/examples/synthetic/reproduce_figures.ipynb)

## ISS Heart (Fig 3, S4–S7)

Human embryonic heart, ~3160 cells, 8 cell types. Includes parameter sweeps, noise robustness, and kNN analysis.

- [**Interactive Notebook**](https://rohitsinghlab.github.io/SAME/tutorials/heart_benchmark.html){:target="_blank"}
- [Source notebook](https://github.com/rohitsinghlab/SAME/tree/main/examples/heart/reproduce_figures.ipynb)

## ISS Heart — Comparison with Other Methods (Fig 3d–f, S8)

SAME against CAST, GPSA, PASTE2, SPACEL, ST-GEARS, STAlign, STAligner and STIM, all starting from the same image-based alignment. Cell type and cell neighborhood matches per method. Aligned outputs: `SAME_benchmarks_data.zip` on [Zenodo](https://zenodo.org/records/19056915).

- [**Interactive Notebook**](https://rohitsinghlab.github.io/SAME/tutorials/heart_compare_methods.html){:target="_blank"}
- [Source notebook](https://github.com/rohitsinghlab/SAME/tree/main/benchmarks/heart/compare_methods.ipynb)
- [Method run notebooks](https://github.com/rohitsinghlab/SAME/tree/main/benchmarks/heart/methods)

## MERSCOPE Tongue — Protein + RNA (Fig 4, S9–S14)

Human tongue, 3608 RNA cells (MERSCOPE) + 4671 protein cells (PCF), 5 cell types. Cross-modal alignment with matrixplot and differential expression.

- [**Interactive Notebook**](https://rohitsinghlab.github.io/SAME/tutorials/tongue_benchmark.html){:target="_blank"}
- [Source notebook](https://github.com/rohitsinghlab/SAME/tree/main/examples/tongue/reproduce_figures.ipynb)

## MERSCOPE Tongue — Comparison with Other Methods (Fig 4g–i, S12, S13)

SAME against CAST, GPSA, PASTE2, SPACEL, ST-GEARS, STAlign and STAligner on protein–RNA alignment, all starting from the same image-based alignment. Cell type and cell neighborhood matches, and predicted RNA expression per PCF cell type. Aligned outputs: `SAME_benchmarks_data.zip` on [Zenodo](https://zenodo.org/records/19056915).

- [**Interactive Notebook**](https://rohitsinghlab.github.io/SAME/tutorials/tongue_compare_methods.html){:target="_blank"}
- [Source notebook](https://github.com/rohitsinghlab/SAME/tree/main/benchmarks/tongue/compare_methods.ipynb)
- [Method run notebooks](https://github.com/rohitsinghlab/SAME/tree/main/benchmarks/tongue/methods)

## MERSCOPE Tongue — Ligand–Receptor Map Agreement (Fig 4j,k, S15)

LIANA+ local ligand–receptor scores on the protein section after RNA transfer by SAME, compared with a nearest same-celltype baseline. Each is scored against the maps measured on the RNA section, over 25–100 µm windows.

- [**Interactive Notebook**](https://rohitsinghlab.github.io/SAME/tutorials/tongue_liana_lr.html){:target="_blank"}
- [Source notebook](https://github.com/rohitsinghlab/SAME/tree/main/examples/tongue/liana_lr_map_agreement.ipynb)

## LUAD33 — Protein + Xenium (Fig 5, S15–S19)

Lung adenocarcinoma, ~94K PCF protein + ~100K Xenium RNA cells, 5 cell types. Penalty sweep, cross-modal matrixplot, and T cell exhaustion analysis.

- [**Interactive Notebook**](https://rohitsinghlab.github.io/SAME/tutorials/luad_benchmark.html){:target="_blank"}
- [Source notebook](https://github.com/rohitsinghlab/SAME/tree/main/examples/luad/reproduce_figures.ipynb)

## LUAD33 — Ligand–Receptor Map Agreement (Fig 5d,e)

The same ligand–receptor analysis on lung adenocarcinoma metacells: SAME vs a nearest same-celltype baseline, each scored against the maps measured on the Xenium RNA section.

- [**Interactive Notebook**](https://rohitsinghlab.github.io/SAME/tutorials/luad_liana_lr.html){:target="_blank"}
- [Source notebook](https://github.com/rohitsinghlab/SAME/tree/main/examples/luad/liana_lr_map_agreement.ipynb)
