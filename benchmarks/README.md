# Benchmarks Against Existing Alignment Methods

Code for the method comparisons in the SAME paper: SAME versus eight published spatial alignment methods on

* **ISS heart** (RNA–RNA, near-serial sections of a human embryonic heart): Fig 3d–f, Fig S8;
* **MERSCOPE tongue** (protein–RNA, PhenoCycler-Fusion vs MERSCOPE sections of human dorsal tongue): Fig 4g–i,
  Fig S12, Fig S13.

```
benchmarks/
├── heart/
│   ├── compare_methods.ipynb   # scores every method and draws Fig 3d–f, S8
│   ├── methods/                # notebooks that ran each method (as run, with outputs)
│   ├── figures/                # output of compare_methods.ipynb
│   └── data/                   # from SAME_benchmarks_data.zip (Zenodo)
└── tongue/
    ├── compare_methods.ipynb   # scores every method and draws Fig 4g–i, S12, S13
    ├── methods/
    ├── figures/
    └── data/
```

## Reproducing the figures

1. Download `SAME_benchmarks_data.zip` from the SAME Zenodo record, <https://zenodo.org/records/19056915>
   (latest version), and unzip it here, in `benchmarks/`. It contains every method's aligned output
   (`heart/data/`, `tongue/data/`).
2. For the tongue notebook, also download `SAME_data.zip` from the same record and unzip it in `examples/`
   (see [`examples/README.md`](../examples/README.md)); Fig S13 reads `examples/tongue/data/merSubAD.h5ad` and
   `pcfSubAD.h5ad`.
3. Run `compare_methods.ipynb` from its own folder. Requirements (SAME itself is not needed):

   ```bash
   pip install "scanpy==1.11.4" "anndata==0.10.9" "numpy==1.26.4" "scipy==1.16.2" "pandas==2.3.2" \
       "scikit-learn==1.9.1" "matplotlib==3.10.6" "jinja2==3.1.6"
   ```

Each notebook ends by checking its scores against the values reported in the paper.

## Evaluation

All methods start from the same image-based initial alignment, which is also the baseline. Aligned query cells are
assigned one-to-one to template cells (Hungarian algorithm on Euclidean distance, with a maximum distance), and the
score is the percentage of query cells whose assigned template cell has the same **cell type** (CT matches) or the same
**cell neighborhood** (CN matches). Neighborhoods are the cell-type composition of each cell's five nearest neighbours
(scimap `spatial_count`, kNN = 6), clustered by k-means fitted jointly on both sections (6 neighborhoods for the heart,
4 for the tongue). Details are in each notebook.

## Method runs

Each method was run in its own environment, following its documentation, on the inputs in `data/`. The notebooks in
`methods/` are copied as they were run, with their outputs, as a record of the settings used; file paths refer to
the original working folders. Each notebook's metadata records its Python version.

| Method | Version | Heart (RNA–RNA) | Tongue (protein–RNA) |
|---|---|---|---|
| CAST | 0.4 | [`CAST.ipynb`](heart/methods/CAST.ipynb) | [`CAST.ipynb`](tongue/methods/CAST.ipynb) |
| GPSA | 0.8 | not included (aligned output in the data archive) | [`GPSA.ipynb`](tongue/methods/GPSA.ipynb) |
| PASTE2 | 1.0.1 | not included (aligned output in the data archive) | [`PASTE2.ipynb`](tongue/methods/PASTE2.ipynb) |
| SPACEL | 1.1.8 | [`SPACEL.ipynb`](heart/methods/SPACEL.ipynb) | [`SPACEL.ipynb`](tongue/methods/SPACEL.ipynb) |
| ST-GEARS | 1.0.0 | [`ST-GEARS.ipynb`](heart/methods/ST-GEARS.ipynb) | [`ST-GEARS.ipynb`](tongue/methods/ST-GEARS.ipynb) |
| STAlign (STalign) | 1.0 | [`STAlign.ipynb`](heart/methods/STAlign.ipynb) | [`STAlign.ipynb`](tongue/methods/STAlign.ipynb) |
| STAligner | 1.0.0 | [`STAligner.ipynb`](heart/methods/STAligner.ipynb) | [`STAligner.ipynb`](tongue/methods/STAligner.ipynb) |
| STIM | 0.3.0 | [`STIM.ipynb`](heart/methods/STIM.ipynb) | not included: failed in the protein–RNA setting |

* Tongue inputs: [`STAlign_preprocess.ipynb`](tongue/methods/STAlign_preprocess.ipynb) builds the shared
  initial-registration coordinates (`New_X`, `New_Y`) and the cell neighborhoods, and exports the expression tables
  read by `GPSA.ipynb` (it also contains a first STAlign run; the scored STAlign result is from `STAlign.ipynb`);
  [`PASTE2.ipynb`](tongue/methods/PASTE2.ipynb) writes `query_input_Prot_tongue.h5ad` and `ref_input_RNA_tongue.h5ad`.
* In the protein–RNA setting, expression-based methods were given corresponding protein–RNA marker pairs; STAlign was
  given cell density only, as expression input did not improve on the baseline.
* SLAT was not benchmarked: it produces many-to-one correspondences, which the one-to-one evaluation does not support.
* The SAME runs are in [`examples/heart`](../examples/heart) and [`examples/tongue`](../examples/tongue).
