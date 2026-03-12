# ISS Heart (Fig 3, S4–S7)

Reproduces **Fig 3** (a–c), **Fig S4**, **Fig S5**, **Fig S6**, **Fig S7** from the paper.

**[View Interactive Notebook](../heart_benchmark.html){:target="_blank"}** — full rendered notebook with all outputs and figures.

- **Source**: [`examples/heart/reproduce_figures.ipynb`](https://github.com/rohitsinghlab/SAME/tree/main/examples/heart/reproduce_figures.ipynb)

## Reproducing from Scratch

```bash
# Single run (paper's main config)
bash run_same.sh --dp 10 --knn 8 --ms 1

# Full parameter sweep (Fig 3c, S6)
bash run_parameter_sweep.sh

# Noise robustness (Fig S5)
bash run_robustness.sh
```
