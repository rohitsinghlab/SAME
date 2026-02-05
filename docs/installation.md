# Installation

## Requirements

- Python >= 3.9
- Gurobi optimizer (requires license - free for academics)

## Basic Installation

Clone the repository and install dependencies:

```bash
git clone https://github.com/rohitsinghlab/SAME.git
cd SAME
pip install -e .
```

Or install dependencies directly:

```bash
pip install numpy pandas scipy gurobipy tqdm networkx scikit-learn alphashape shapely
```

## Gurobi License Setup

SAME requires Gurobi for optimization. Academic users can obtain a free license.

### Option 1: Web License (Recommended)

1. Create an account at [gurobi.com](https://www.gurobi.com/)
2. Request an academic license
3. Get your Web License Service (WLS) credentials
4. Create `src/.gurobienv` file:

```
WLSACCESSID=your_access_id
WLSSECRET=your_secret
LICENSEID=your_license_id
```

### Option 2: Environment Variables

Alternatively, set environment variables:

```bash
export GUROBI_WLSACCESSID=your_access_id
export GUROBI_WLSSECRET=your_secret
export GUROBI_LICENSEID=your_license_id
```

### Option 3: Local License

For local or cluster installations, follow Gurobi's documentation to install and activate a local license.

## Optional Dependencies

For alpha shape filtering (recommended for better boundary detection):

```bash
pip install alphashape shapely
```

For Jupyter notebook support:

```bash
pip install jupyter ipywidgets
```

## Verify Installation

```python
from src import run_same, init_optim_params

# Check version
import src
print(f"SAME version: {src.__version__}")

# Verify Gurobi
import gurobipy as gp
env = gp.Env()
print("Gurobi initialized successfully")
```

## Troubleshooting

### Gurobi License Issues

If you see "No Gurobi license found":

1. Verify your `.gurobienv` file exists in `src/`
2. Check credentials are correct
3. Ensure you have network access (for WLS licenses)
4. Try setting environment variables directly

### Memory Issues

For large datasets (>10,000 cells), use metacells and lazy constraints:

```python
from src import greedy_triangle_collapse, init_optim_params

# Create metacells to reduce problem size
mc_aligned, _ = greedy_triangle_collapse(aligned_df, max_metacell_size=3)

# Use lazy constraints for memory efficiency
optim = init_optim_params(lazy_constraints=True)
```

### Import Errors

If imports fail, ensure you're in the SAME directory or have it in your Python path:

```python
import sys
sys.path.insert(0, '/path/to/SAME')
```
