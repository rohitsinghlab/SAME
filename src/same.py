from gurobipy import Model, GRB, quicksum
import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist
import gurobipy as gp
from typing import Any, Dict, Optional
try:
    get_ipython()
    from tqdm.notebook import tqdm
except:
    from tqdm import tqdm
import os
from scipy.spatial import Delaunay
from itertools import permutations
from itertools import combinations
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
import multiprocessing

# Local imports
from .utils import preprocess_data, find_knn_within_radius
from .triangle_utils import compute_filtered_delaunay
from .violationhelper import verify_spatial_preservation, print_violation_report
from .knn_utils import find_knn_with_cell_type_priority
from .init_helpers import apply_mip_start
from .helpers import (
    get_unprocessed_windows,
    add_basic_constraints_optimized,
    precompute_coordinate_maps,
    precompute_triangle_info,
    filter_triangles_by_radius,
    add_spatial_constraints_triangle_based,
    calculate_signed_area,
    signed_area_terms,
    calc_ref_area,
    load_matching_results,
    merge_window_matches_unique_ref
)

def init_gurobi_params(**overrides) -> Dict[str, Any]:
    """
    Create default Gurobi solver parameters for SAME optimization.

    Returns a dictionary of Gurobi-related parameters that control solver
    behavior, time limits, and lazy constraint generation. Use keyword
    arguments to override specific defaults.

    Parameters
    ----------
    **overrides : dict
        Keyword arguments to override default values.

    Returns
    -------
    Dict[str, Any]
        Dictionary of Gurobi parameters with the following keys:

        **Core solve controls:**

        - ``time_limit`` : int, default=7200
            Maximum solve time in seconds (2 hours).
        - ``mip_gap`` : float, default=0.05
            MIP optimality gap tolerance (5%).

        **Gurobi tuning knobs:**

        - ``mip_focus`` : int, default=2
            MIPFocus parameter (2 = focus on proving optimality).
        - ``cuts`` : int, default=2
            Cut generation aggressiveness (0=none to 3=aggressive).
        - ``heuristics`` : float, default=0.1
            Fraction of time spent on heuristics (0.0-1.0).

        **MIP start / initialization:**

        - ``init_method`` : str or None, default=None
            Method for warm-starting the solver:
            - None: no initialization
            - 'greedy': fast greedy assignment
            - 'hungarian': optimal assignment (requires max_matches=1)
        - ``init_big_m`` : float, default=1e9
            Large cost for forbidden pairs in Hungarian init.
        - ``init_hungarian_max_n`` : int, default=5000
            Skip Hungarian init if n_aligned + n_ref exceeds this.

        **Lazy constraints:**

        - ``lazy_max_cuts`` : int or None, default=None
            Global cap on lazy cuts added (None = unlimited).
        - ``lazy_allowed_flip_fraction`` : float, default=0.05
            Allowed fraction of flipped triangles before adding cuts.
        - ``lazy_max_cuts_per_incumbent`` : int, default=1000
            Max cuts added per incumbent solution.

    Examples
    --------
    >>> params = init_gurobi_params(time_limit=3600, mip_gap=0.01)
    >>> params['time_limit']
    3600

    See Also
    --------
    init_optim_params : Create optimization parameters.
    run_same : Main optimization function using these parameters.
    """
    params = {
        # Core solve controls
        "time_limit": 7200,
        "mip_gap": 0.05,

        # Gurobi knobs (optional)
        "mip_focus": 2,
        "cuts": 2, #
        "heuristics": 0.1, #0.1-0.3


        # MIP start / initialization knobs
        "init_method": None,
        "init_big_m": 1e9,
        "init_hungarian_max_n": 5000,

        # Lazy constraints knobs
        "lazy_max_cuts": None,
        "lazy_allowed_flip_fraction": 0.05,
        "lazy_max_cuts_per_incumbent": 1000,


    }
    params.update(overrides)
    return params


def init_optim_params(**overrides) -> Dict[str, Any]:
    """
    Create default optimization parameters for SAME matching.

    Returns a dictionary of parameters controlling the matching problem
    formulation, spatial constraints, and sliding window behavior.
    Use keyword arguments to override specific defaults.

    Parameters
    ----------
    **overrides : dict
        Keyword arguments to override default values.

    Returns
    -------
    Dict[str, Any]
        Dictionary of optimization parameters with the following keys:

        **Sliding window parameters:**

        - ``window_size`` : int, default=1000
            Size of each window in coordinate units.
        - ``overlap`` : int, default=250
            Overlap between adjacent windows.
        - ``min_cells_per_window`` : int, default=10
            Minimum cells required to process a window.

        **Matching problem parameters:**

        - ``max_matches`` : int, default=1
            Maximum times a reference point can be matched.
        - ``ref_metacell_match_multiplier`` : int or None, default=None
            For metacells, multiplier for max_matches. None = use max size.
        - ``radius`` : float, default=250
            Search radius for finding KNN candidates.
        - ``knn`` : int, default=8
            Number of nearest neighbors to consider.

        **Objective function coefficients:**

        - ``penalty_coeff`` : float, default=100
            Penalty for reference point matched more than once.
        - ``no_match_penalty`` : float, default=100
            Penalty for unmatched aligned point (per cell).
        - ``delaunay_penalty`` : float, default=5
            Penalty for triangle orientation flip (space-tearing).
        - ``dist_ct_coeff`` : float, default=1
            Weight for cell type distance in objective.

        **Output labeling:**

        - ``cell_id_col`` : str, default='Cell_Num_Old'
            Column name for cell identifiers.

        **Constraint/behavior flags:**

        - ``hard_spatial_constraints`` : bool, default=False
            If True, strictly forbid triangle flips (no soft penalty).
        - ``ignore_same_type_triangles`` : bool, default=True
            If True, skip constraints for triangles with homogeneous cell types.
        - ``ignore_knn_if_matched`` : bool, default=False
            If True, use cell-type priority in KNN search.
        - ``lazy_constraints`` : bool, default=True
            If True, add spatial constraints lazily (memory efficient).

        **Triangle quality filtering:**

        - ``min_angle_deg`` : float, default=15
            Filter out thin triangles with minimum angle below this threshold.

    Examples
    --------
    >>> params = init_optim_params(radius=500, knn=10, lazy_constraints=True)
    >>> params['radius']
    500

    See Also
    --------
    init_gurobi_params : Create Gurobi solver parameters.
    run_same : Main optimization function using these parameters.
    sliding_window_matching : Sliding window approach for large datasets.
    """
    params = {
        # Sliding window params
        "window_size": 1000,
        "overlap": 250,
        "min_cells_per_window": 10,

        # Matching problem params
        "max_matches": 1,
        "ref_metacell_match_multiplier": None,  # If ref has metacells, each metacell accepts max_matches * this many aligned cells
                                                 # None = use max metacell size in ref_df
        "radius": 250,
        "penalty_coeff": 100,
        "no_match_penalty": 100,
        "delaunay_penalty": 5,
        "dist_ct_coeff": 1,
        "knn":8,
        # Output labeling
        "cell_id_col": "Cell_Num_Old",
        # Constraint/behavior flags
        "hard_spatial_constraints": False,
        "ignore_same_type_triangles": True,
        "ignore_knn_if_matched": False,
        "lazy_constraints": True,
        # Triangle quality filtering
        "min_angle_deg": 15,  # Filter out thin triangles with angles < 15 degrees
    }
    params.update(overrides)
    return params


def _as_triangle_array(delaunay_like):
    """Normalize a triangulation-like object into an int ndarray of shape (n, 3)."""
    if delaunay_like is None:
        return None
    if isinstance(delaunay_like, np.ndarray):
        tri = delaunay_like
    elif isinstance(delaunay_like, pd.DataFrame):
        tri = delaunay_like.iloc[:, :3].to_numpy()
    else:
        tri = np.asarray(delaunay_like)
    if tri.size == 0:
        return np.array([], dtype=int).reshape(0, 3)
    if tri.ndim != 2 or tri.shape[1] != 3:
        raise ValueError(f"aligned_delaunay must have shape (n, 3); got {tri.shape}")
    return tri.astype(int, copy=False)


def _remap_triangles_by_vertex_ids(triangles, vertex_ids):
    """
    Remap triangles expressed in a vertex-ID space onto 0..n-1 row indices.

    Parameters
    ----------
    triangles : array-like, shape (n, 3)
        Vertex IDs for each triangle (not necessarily 0..n-1).
    vertex_ids : array-like, shape (n_vertices,)
        For the current aligned_df (post-window + post-KNN), the stable vertex ID
        for each row, in row order.

    Returns
    -------
    np.ndarray
        Triangles in 0..n-1 row-index space, dropping any triangle with a missing vertex.
    """
    tri = _as_triangle_array(triangles)
    if tri is None:
        return None
    if tri.size == 0:
        return tri

    id_to_row = {v: i for i, v in enumerate(vertex_ids)}
    flat = tri.reshape(-1)
    remapped_flat = np.fromiter((id_to_row.get(int(v), -1) for v in flat), dtype=int, count=flat.size)
    remapped = remapped_flat.reshape(tri.shape)
    remapped = remapped[(remapped >= 0).all(axis=1)]
    return remapped


def subset_data(df, x_min, x_max, y_min, y_max):
    return df[(df['X'] >= x_min) & (df['X'] < x_max) & 
                (df['Y'] >= y_min) & (df['Y'] < y_max)]

def sliding_window_matching(
    ref,
    moving,
    commonCT=None,
    outprefix=None,
    moving_delaunay=None,
    moving_delaunay_vertex_col=None,
    optim_params: Optional[Dict[str, Any]] = None,
    gurobi_params: Optional[Dict[str, Any]] = None,
    ignore_precomputed_triangulation: bool = False,
):
    """
    Match cells between reference and moving datasets using a sliding window approach.

    This function divides the spatial domain into overlapping windows and runs
    the SAME optimization on each window. Results from overlapping regions are
    resolved using bipartite matching to ensure unique assignments.

    Supports resumption from partial results when ``outprefix`` is provided.

    Parameters
    ----------
    ref : pd.DataFrame or MetaCell
        Reference dataset with columns ['X', 'Y', 'cell_type'] plus cell type
        probability columns. Can also be a MetaCell object with ``metacell_df``
        attribute.
    moving : pd.DataFrame or MetaCell
        Moving/aligned dataset to be matched to reference. Same column
        requirements as ``ref``. Can be a MetaCell object with ``metacell_df``
        and ``metacell_delaunay`` attributes.
    commonCT : list of str, optional
        List of cell type column names (probability/one-hot columns).
        If None, inferred from unique values of 'cell_type' column.
    outprefix : str, optional
        Output directory path. If provided, intermediate results are saved
        and the function can resume from partial runs.
    moving_delaunay : array-like, optional
        Precomputed Delaunay triangulation for moving data as (n, 3) array
        of vertex indices. If moving is a MetaCell, this is extracted
        automatically.
    moving_delaunay_vertex_col : str, optional
        Column name in moving DataFrame containing vertex IDs that correspond
        to indices in ``moving_delaunay``. Required if triangulation uses
        non-sequential IDs.
    optim_params : dict, optional
        Optimization parameters. See ``init_optim_params()`` for defaults.
        Key parameters include:

        - ``window_size``: Size of each window (default: 1000)
        - ``overlap``: Overlap between windows (default: 250)
        - ``radius``: KNN search radius (default: 250)
        - ``knn``: Number of nearest neighbors (default: 8)
        - ``lazy_constraints``: Use memory-efficient constraints (default: True)

    gurobi_params : dict, optional
        Gurobi solver parameters. See ``init_gurobi_params()`` for defaults.
    ignore_precomputed_triangulation : bool, default=False
        If True, compute fresh Delaunay triangulation even if precomputed
        one is available.

    Returns
    -------
    pd.DataFrame
        Matched pairs with columns:

        - ``aligned_idx``, ``ref_idx``: Integer indices into input DataFrames
        - ``X``, ``Y``: Coordinates of aligned points
        - ``ref_X``, ``ref_Y``: Coordinates of matched reference points
        - Cell type probability columns
        - ``size``, ``ref_size``: Metacell sizes (if applicable)
        - ``window_id``: Which window the match came from
        - ``time_limit_reached``: Whether solver hit time limit
        - ``triangle_violation``: Whether point is in a flipped triangle
        - ``run_time``: Optimization time for the window

    Examples
    --------
    Basic usage:

    >>> from src import sliding_window_matching, init_optim_params
    >>> matches = sliding_window_matching(
    ...     ref=ref_df,
    ...     moving=aligned_df,
    ...     commonCT=['TypeA', 'TypeB', 'TypeC'],
    ...     outprefix='results/'
    ... )

    With custom parameters:

    >>> optim = init_optim_params(window_size=500, overlap=100, radius=300)
    >>> matches = sliding_window_matching(
    ...     ref=ref_df,
    ...     moving=aligned_df,
    ...     optim_params=optim
    ... )

    With metacells:

    >>> from src import MetaCell
    >>> mc_aligned = MetaCell(aligned_df, max_metacell_size=10)
    >>> matches = sliding_window_matching(ref=ref_df, moving=mc_aligned)

    See Also
    --------
    run_same : Core optimization for a single region.
    init_optim_params : Create optimization parameter dictionary.
    init_gurobi_params : Create Gurobi parameter dictionary.
    merge_window_matches_unique_ref : Merge overlapping window results.
    """
    


    # Allow passing a MetaCell object directly as `ref` / `moving`
    # (duck-typing to avoid importing metacell_utils here).
    ref_cell_type_col = "cell_type"
    moving_cell_type_col = "cell_type"
    if optim_params is None:
        optim_params = {}
    if gurobi_params is None:
        gurobi_params = {}

    if hasattr(ref, "metacell_df"):
        mc_ref = ref
        ref = mc_ref.metacell_df
        if hasattr(mc_ref, "cell_type_col"):
            ref_cell_type_col = mc_ref.cell_type_col
        if (optim_params.get("cell_id_col") is None) and hasattr(mc_ref, "metacell_idx_col"):
            optim_params["cell_id_col"] = mc_ref.metacell_idx_col
    if hasattr(moving, "metacell_df") and hasattr(moving, "metacell_delaunay"):
        mc = moving
        moving = mc.metacell_df
        if moving_delaunay is None and not ignore_precomputed_triangulation:
            moving_delaunay = mc.metacell_delaunay
        if moving_delaunay_vertex_col is None and hasattr(mc, "metacell_idx_col"):
            moving_delaunay_vertex_col = mc.metacell_idx_col
        if hasattr(mc, "cell_type_col"):
            moving_cell_type_col = mc.cell_type_col
        if (optim_params.get("cell_id_col") is None) and hasattr(mc, "metacell_idx_col"):
            optim_params["cell_id_col"] = mc.metacell_idx_col

    optim_params = init_optim_params(**(optim_params or {}))
    gurobi_params = init_gurobi_params(**(gurobi_params or {}))

    window_size = optim_params["window_size"]
    overlap = optim_params["overlap"]
    min_cells_per_window = optim_params["min_cells_per_window"]
    cell_id_col = optim_params["cell_id_col"]

    # Strict check: ensure the cell_type categories match across datasets
    ref_types = None
    mov_types = None
    if ref_cell_type_col in ref.columns and moving_cell_type_col in moving.columns:
        ref_types = set(pd.Series(ref[ref_cell_type_col]).dropna().unique().tolist())
        mov_types = set(pd.Series(moving[moving_cell_type_col]).dropna().unique().tolist())
        if ref_types != mov_types:
            raise ValueError(
                f"Cell type categories differ between ref and moving.\n"
                f"ref ({ref_cell_type_col}) has {len(ref_types)} types, moving ({moving_cell_type_col}) has {len(mov_types)} types.\n"
                f"Only-in-ref: {sorted(ref_types - mov_types)[:20]}\n"
                f"Only-in-moving: {sorted(mov_types - ref_types)[:20]}"
            )

    # If commonCT is not provided, define it as the unique values of the cell_type column.
    # This assumes your one-hot/probability columns are named by those cell type strings.
    if commonCT is None:
        if ref_types is None:
            raise ValueError(
                "commonCT is None, but cell_type columns were not found to infer it. "
                "Pass commonCT explicitly (list of probability/one-hot columns), or ensure "
                f"both dataframes have '{ref_cell_type_col}'/'{moving_cell_type_col}'."
            )
        commonCT = sorted(ref_types)
        missing_ref = [c for c in commonCT if c not in ref.columns]
        missing_mov = [c for c in commonCT if c not in moving.columns]
        if missing_ref or missing_mov:
            raise ValueError(
                "commonCT is None so it was inferred as the unique values of the cell_type column, "
                "but those names are not present as probability/one-hot columns.\n"
                f"Missing in ref columns (first 20): {missing_ref[:20]}\n"
                f"Missing in moving columns (first 20): {missing_mov[:20]}\n"
                "Either rename your probability columns to match cell type names, or pass commonCT explicitly."
            )

    # Initialize variables
    x_min, x_max = min(ref['X'].min(), moving['X'].min()), max(ref['X'].max(), moving['X'].max())
    y_min, y_max = min(ref['Y'].min(), moving['Y'].min()), max(ref['Y'].max(), moving['Y'].max())
    step = window_size - overlap
    all_matches = []
    
    # Create a grid of windows
    x_windows = list(range(int(x_min), int(x_max), step))
    y_windows = list(range(int(y_min), int(y_max), step))
    
    # Create output directory if outprefix is provided
    output_file = None
    if outprefix:
        os.makedirs(outprefix, exist_ok=True)
        output_file = os.path.join(outprefix, 'matchedDF.csv')
    
    # Check for existing results if outprefix is provided
    windows_to_process = None
    if outprefix:
        windows_to_process, existing_matches = get_unprocessed_windows(moving, output_file, x_windows, y_windows, window_size, overlap, cell_id_col=cell_id_col)
        if existing_matches is not None:
            all_matches.append(existing_matches)
    
    # Calculate total number of iterations
    total_iterations = len(windows_to_process) if windows_to_process is not None else len(x_windows) * len(y_windows)

    # Create progress bar
    with tqdm(total=total_iterations, desc="Processing windows") as pbar:
        i = 0
        while i < len(x_windows):
            j = 0
            while j < len(y_windows):
                # Skip if this window is already processed
                if windows_to_process is not None and (i,j) not in windows_to_process:
                    j += 1
                    continue
                
                x = x_windows[i]
                y = y_windows[j]
                x_window_min, x_window_max = x, x + window_size
                y_window_min, y_window_max = y, y + window_size
                
                # Get initial window data
                ref_subset = subset_data(ref, x_window_min, x_window_max, y_window_min, y_window_max)
                moving_subset = subset_data(moving, x_window_min, x_window_max, y_window_min, y_window_max)
                
                # Check if window needs merging
                if len(ref_subset) < min_cells_per_window or len(moving_subset) < min_cells_per_window:
                    # Try merging with right window if available
                    if i + 1 < len(x_windows):
                        x_window_max = x_windows[i + 1] + window_size
                        ref_subset = subset_data(ref, x_window_min, x_window_max, y_window_min, y_window_max)
                        moving_subset = subset_data(moving, x_window_min, x_window_max, y_window_min, y_window_max)
                        if len(ref_subset) >= min_cells_per_window and len(moving_subset) >= min_cells_per_window:
                            i += 1  # Skip next x window since we merged it
                    
                    # If still too few cells and bottom window available, merge vertically
                    if (len(ref_subset) < min_cells_per_window or len(moving_subset) < min_cells_per_window) and j + 1 < len(y_windows):
                        y_window_max = y_windows[j + 1] + window_size
                        ref_subset = subset_data(ref, x_window_min, x_window_max, y_window_min, y_window_max)
                        moving_subset = subset_data(moving, x_window_min, x_window_max, y_window_min, y_window_max)
                        if len(ref_subset) >= min_cells_per_window and len(moving_subset) >= min_cells_per_window:
                            j += 1  # Skip next y window since we merged it
                
                print(f"Window at ({x}, {y}) - Ref cells: {len(ref_subset)}, Moving cells: {len(moving_subset)}")
                
                if len(ref_subset) >= min_cells_per_window and len(moving_subset) >= min_cells_per_window:
                    # Create window-specific output directory if outprefix is provided
                    window_outprefix = None
                    if outprefix:
                        window_id = len(x_windows) * j + i
                        window_outprefix = os.path.join(outprefix, f'window_{window_id}')
                    
                    window_matches, var_out = run_same(
                        aligned_df=moving_subset,
                        ref_df=ref_subset,
                        commonCT=commonCT,
                        optim_params=optim_params,
                        gurobi_params=gurobi_params,
                        outprefix=window_outprefix,
                        aligned_delaunay=moving_delaunay,
                        aligned_delaunay_vertex_col=moving_delaunay_vertex_col,
                        ignore_precomputed_triangulation=ignore_precomputed_triangulation)
                    
                    if window_matches.shape[0] > 0:
                        is_left_edge = x == int(x_min)
                        is_right_edge = x_window_max >= int(x_max)
                        is_top_edge = y == int(y_min)
                        is_bottom_edge = y_window_max >= int(y_max)

                        # Adjust the filtering conditions based on edge cases
                        x_min_filter = x_window_min if is_left_edge else x_window_min + overlap/2
                        x_max_filter = x_window_max if is_right_edge else x_window_max - overlap/2
                        y_min_filter = y_window_min if is_top_edge else y_window_min + overlap/2
                        y_max_filter = y_window_max if is_bottom_edge else y_window_max - overlap/2

                        # Apply the adjusted filters
                        central_matches = window_matches[
                            (window_matches['X'] >= x_min_filter) & 
                            (window_matches['X'] < x_max_filter) & 
                            (window_matches['Y'] >= y_min_filter) & 
                            (window_matches['Y'] < y_max_filter)
                        ]
                        central_matches['window_id'] = len(x_windows) * j + i
                        print('Central matches: ', central_matches.shape)
                        if len(central_matches) > 0:
                            all_matches.append(central_matches)
                            
                            # Save intermediate results if outprefix is provided
                            if outprefix:
                                pd.concat(all_matches, ignore_index=True).to_csv(output_file, index=False)                
                j += 1
                pbar.update(1)  # Update progress bar
            i += 1
    
    return pd.concat(all_matches, ignore_index=True) if all_matches else pd.DataFrame()


def load_gurobi_config():
    """Load Gurobi configuration from .gurobienv file in the src directory."""
    config = {}
    # Get the directory where this file is located (src/)
    src_dir = os.path.dirname(os.path.abspath(__file__))
    config_file = os.path.join(src_dir, '.gurobienv')
    
    try:
        with open(config_file, 'r') as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith('#') and '=' in line:
                    key, value = line.split('=', 1)
                    config[key.strip()] = value.strip()
        print(f"Loaded Gurobi config from {config_file}")
    except FileNotFoundError:
        print(f"No .gurobienv file found at {config_file}, using environment variables")
    except Exception as e:
        print(f"Error reading .gurobienv file: {e}, using environment variables")
    
    return config


def _lazy_orientation_callback(model, where):
    """Callback for lazy constraint generation - adds triangle orientation cuts on demand."""
    if where != GRB.Callback.MIPSOL:
        return

    # Optional global cap on number of lazy cuts
    if getattr(model, "_lazy_max_cuts", None) is not None:
        if model._cuts_added >= model._lazy_max_cuts:
            return

    x_vals = model.cbGetSolution(model._x)

    # Build current matching from solution
    matching = {}
    match_pair_idx = {}
    for idx, (ip, jp) in enumerate(model._valid_pairs):
        if x_vals[idx] > 0.5:
            matching[ip] = jp
            match_pair_idx[ip] = idx

    # Check each triangle for orientation violations
    # We may choose to add cuts only if flip rate exceeds a user threshold.
    violating_tris = []
    checked = 0
    for tri_idx, tri in enumerate(model._aligned_delaunay):
        a, b, c = tri

        # Skip if not all vertices are matched
        if a not in matching or b not in matching or c not in matching:
            continue

        ja, jb, jc = matching[a], matching[b], matching[c]

        # Compute reference triangle orientation
        ax, ay = model._ref_coords[ja]
        bx, by = model._ref_coords[jb]
        cx, cy = model._ref_coords[jc]
        ref_sign = np.sign((bx - ax) * (cy - ay) - (by - ay) * (cx - ax))

        source_sign = model._source_signs[tri_idx]

        # Skip degenerate triangles (zero area = collinear points, not a flip)
        if source_sign == 0 or ref_sign == 0:
            continue

        checked += 1
        # Check for orientation flip (actual space folding/tearing)
        if source_sign != ref_sign:
            violating_tris.append((tri_idx, a, b, c))

    if checked == 0 or not violating_tris:
        return

    allowed_frac = getattr(model, "_lazy_allowed_flip_fraction", None)
    if allowed_frac is not None:
        flip_frac = len(violating_tris) / float(checked)
        if flip_frac <= allowed_frac:
            # Within allowed flip budget: do not add more cuts for this incumbent
            return

    # Add cuts, optionally limiting how many we add per incumbent and overall.
    per_inc_limit = getattr(model, "_lazy_max_cuts_per_incumbent", None)
    remaining_global = None
    if getattr(model, "_lazy_max_cuts", None) is not None:
        remaining_global = max(0, model._lazy_max_cuts - model._cuts_added)

    added_here = 0
    for tri_idx, a, b, c in violating_tris:
        if per_inc_limit is not None and added_here >= per_inc_limit:
            break
        if remaining_global is not None and remaining_global <= 0:
            break

        # Add lazy constraint: this combination must pay penalty
        model.cbLazy(
            model._x[match_pair_idx[a]] +
            model._x[match_pair_idx[b]] +
            model._x[match_pair_idx[c]] <= 2 + model._q_tri[tri_idx]
        )
        model._cuts_added += 1
        added_here += 1
        if remaining_global is not None:
            remaining_global -= 1


def run_same(
    ref_df,
    aligned_df,
    commonCT,
    outprefix=None,
    aligned_delaunay=None,
    aligned_delaunay_vertex_col=None,
    optim_params: Optional[Dict[str, Any]] = None,
    gurobi_params: Optional[Dict[str, Any]] = None,
    ignore_precomputed_triangulation: bool = False,
):
    """
    Find optimal spatial matches between aligned and reference cells using MIP.

    This is the core SAME optimization function. It formulates cell matching as
    a Mixed Integer Program (MIP) that minimizes cell type distance and coordinate
    distance while preserving spatial relationships through Delaunay triangle
    orientation constraints.

    The function supports both "eager" mode (all constraints upfront) and "lazy"
    mode (constraints added on-demand via callbacks), with lazy mode being more
    memory-efficient for large problems.

    Parameters
    ----------
    ref_df : pd.DataFrame
        Reference dataset with required columns:

        - ``X``, ``Y``: Spatial coordinates
        - ``cell_type``: Cell type annotation
        - Cell type probability columns (names in ``commonCT``)
        - Cell ID column (name specified in optim_params['cell_id_col'])

    aligned_df : pd.DataFrame or MetaCell
        Aligned/moving dataset to match against reference. Same column
        requirements as ``ref_df``. Can be a MetaCell object.
    commonCT : list of str
        Names of columns containing cell type probabilities or one-hot
        encodings. These are used to compute cell type distance.
    outprefix : str, optional
        Output directory. If provided, saves:

        - ``matches_df.csv``: Final matched pairs
        - ``aligned_df.csv``, ``ref_df.csv``: Filtered input data
        - ``var_out.npy``: Optimization variables and diagnostics
        - ``matching_model.lp``: Gurobi model file

    aligned_delaunay : array-like, shape (n_triangles, 3), optional
        Precomputed Delaunay triangulation. If None, computed automatically.
        Useful when using metacells with pre-filtered triangulation.
    aligned_delaunay_vertex_col : str, optional
        Column name containing vertex IDs that correspond to indices in
        ``aligned_delaunay``. Required if triangulation uses non-default IDs.
    optim_params : dict, optional
        Optimization parameters from ``init_optim_params()``. Key parameters:

        - ``radius``: KNN search radius (default: 250)
        - ``knn``: Number of nearest neighbors (default: 8)
        - ``max_matches``: Max times ref can be matched (default: 1)
        - ``lazy_constraints``: Use lazy constraint generation (default: True)
        - ``delaunay_penalty``: Penalty for triangle flips (default: 5)
        - ``no_match_penalty``: Penalty for unmatched cells (default: 100)

    gurobi_params : dict, optional
        Gurobi solver parameters from ``init_gurobi_params()``. Key parameters:

        - ``time_limit``: Max solve time in seconds (default: 7200)
        - ``mip_gap``: Optimality gap tolerance (default: 0.05)
        - ``init_method``: MIP start method ('greedy', 'hungarian', or None)

    ignore_precomputed_triangulation : bool, default=False
        If True, compute fresh Delaunay even if precomputed one provided.

    Returns
    -------
    matches_df : pd.DataFrame
        Matched pairs with columns:

        - ``aligned_idx``, ``ref_idx``: Row indices into filtered DataFrames
        - ``X``, ``Y``: Coordinates of aligned points
        - ``ref_X``, ``ref_Y``: Coordinates of matched reference points
        - ``size``, ``ref_size``: Metacell sizes (1 for individual cells)
        - ``Ref_{cell_id_col}``, ``Aligned_{cell_id_col}``: Original cell IDs
        - ``time_limit_reached``: Whether solver hit time limit
        - ``triangle_violation``: Whether point is in a flipped triangle
        - ``filtered_violation``: Violation detected by both methods
        - ``run_time``: Optimization time in seconds

    var_out : dict
        Optimization diagnostics including:

        - ``x``: Match variable values (list of 0/1)
        - ``no_match_vars``: Unmatched penalty values
        - ``penalty_vars``: Multi-match penalty values
        - ``area_penalty_vars``: Triangle flip penalty values
        - ``violations``: Spatial violation report
        - ``triangle_data``: Triangle analysis (areas, flips)
        - ``lazy_cuts_added``: Number of lazy constraints added

    Notes
    -----
    **Objective Function:**

    The solver minimizes:

    .. math::

        \\sum_{(i,j)} c_{ij} x_{ij} + \\alpha \\sum_j p_j + \\beta \\sum_i s_i n_i + \\gamma \\sum_t w_t q_t

    where:
    - :math:`c_{ij}` = cell type distance + coordinate distance
    - :math:`p_j` = penalty for ref j matched more than once
    - :math:`n_i` = penalty for aligned i not matched, weighted by size :math:`s_i`
    - :math:`q_t` = penalty for triangle t orientation flip, weighted by :math:`w_t`

    **Lazy Constraints:**

    When ``lazy_constraints=True``, triangle orientation constraints are added
    on-demand during optimization. This reduces memory from O(n×k³) to O(n)
    and is recommended for large problems (>1000 cells).

    **Space-Tearing:**

    Triangle orientation flips indicate space-tearing events where the spatial
    relationship between cells changes (e.g., due to tissue deformation or
    missing cells). The ``delaunay_penalty`` controls how strongly these are
    penalized.

    Examples
    --------
    Basic usage:

    >>> from src import run_same
    >>> matches, var_out = run_same(
    ...     ref_df=ref_df,
    ...     aligned_df=aligned_df,
    ...     commonCT=['TypeA', 'TypeB', 'TypeC'],
    ...     outprefix='results/'
    ... )
    >>> print(f"Found {len(matches)} matches")

    With custom parameters:

    >>> from src import run_same, init_optim_params, init_gurobi_params
    >>> optim = init_optim_params(radius=500, lazy_constraints=True)
    >>> gurobi = init_gurobi_params(time_limit=3600, mip_gap=0.01)
    >>> matches, var_out = run_same(
    ...     ref_df=ref_df,
    ...     aligned_df=aligned_df,
    ...     commonCT=commonCT,
    ...     optim_params=optim,
    ...     gurobi_params=gurobi
    ... )

    See Also
    --------
    sliding_window_matching : Process large datasets in windows.
    init_optim_params : Create optimization parameters.
    init_gurobi_params : Create Gurobi parameters.
    greedy_triangle_collapse : Create metacells for faster optimization.
    """
    # Create a more robust log directory
    log_dir = os.path.join(os.getcwd(), 'gurobi_logs')
    os.makedirs(log_dir, exist_ok=True)
    log_file = os.path.join(log_dir, f'gurobi_{os.getpid()}.log')
    
    # Load Gurobi configuration from .gurobienv file or environment variables
    gurobi_config = load_gurobi_config()
    
    options = {
        'WLSACCESSID': os.environ.get('GUROBI_WLSACCESSID', '') or gurobi_config.get('WLSACCESSID', ''),
        'WLSSECRET': os.environ.get('GUROBI_WLSSECRET', '') or gurobi_config.get('WLSSECRET', ''),
        'LICENSEID': int(os.environ.get('GUROBI_LICENSEID', 0)) or int(gurobi_config.get('LICENSEID', 0)),
        'OutputFlag': 1,
        'LogFile': log_file
    }
    try:
        env = gp.Env(params=options)

        if gurobi_params is None:
            gurobi_params = {}
        if optim_params is None:
            optim_params = {}

        # Allow passing a MetaCell object directly as aligned_df
        if hasattr(aligned_df, "metacell_df") and hasattr(aligned_df, "metacell_delaunay"):
            mc = aligned_df
            aligned_df = mc.metacell_df
            if aligned_delaunay is None and not ignore_precomputed_triangulation:
                aligned_delaunay = mc.metacell_delaunay
            if aligned_delaunay_vertex_col is None and hasattr(mc, "metacell_idx_col"):
                aligned_delaunay_vertex_col = mc.metacell_idx_col
            if (optim_params.get("cell_id_col") is None) and hasattr(mc, "metacell_idx_col"):
                optim_params["cell_id_col"] = mc.metacell_idx_col

        # Build/fill param dicts
        optim_params = init_optim_params(**(optim_params or {}))
        gurobi_params = init_gurobi_params(**gurobi_params)

        # Unpack dicts into local vars
        max_matches = optim_params["max_matches"]
        ref_metacell_match_multiplier = optim_params["ref_metacell_match_multiplier"]
        radius = optim_params["radius"]
        penalty_coeff = optim_params["penalty_coeff"]
        no_match_penalty = optim_params["no_match_penalty"]
        delaunay_penalty = optim_params["delaunay_penalty"]
        dist_ct_coeff = optim_params["dist_ct_coeff"]
        knn = optim_params["knn"]
        hard_spatial_constraints = optim_params["hard_spatial_constraints"]
        ignore_same_type_triangles = optim_params["ignore_same_type_triangles"]
        ignore_knn_if_matched = optim_params["ignore_knn_if_matched"]
        lazy_constraints = optim_params["lazy_constraints"]
        min_angle_deg = optim_params.get("min_angle_deg", 15)

        time_limit = gurobi_params["time_limit"]
        mip_gap = gurobi_params["mip_gap"]
        mip_focus = gurobi_params["mip_focus"]
        cuts = gurobi_params["cuts"]
        heuristics = gurobi_params["heuristics"]
        init_method = gurobi_params["init_method"]
        init_big_m = gurobi_params["init_big_m"]
        init_hungarian_max_n = gurobi_params["init_hungarian_max_n"]
        lazy_max_cuts = gurobi_params["lazy_max_cuts"]
        lazy_allowed_flip_fraction = gurobi_params["lazy_allowed_flip_fraction"]
        lazy_max_cuts_per_incumbent = gurobi_params["lazy_max_cuts_per_incumbent"]
        cell_id_col = optim_params["cell_id_col"]

        # Auto-detect metacells: add 'size' column if missing (default to 1)
        if 'size' not in aligned_df.columns:
            aligned_df = aligned_df.copy()
            aligned_df['size'] = 1
        if 'size' not in ref_df.columns:
            ref_df = ref_df.copy()
            ref_df['size'] = 1

        # NOTE: n_aligned/n_ref will be recomputed after KNN filtering.
        n_aligned = len(aligned_df)
        n_ref = len(ref_df)
        total_aligned_cells = aligned_df['size'].sum()
        total_ref_cells = ref_df['size'].sum()

        print(f"Number of aligned points: {n_aligned} (total cells: {total_aligned_cells:.0f})")
        print(f"Number of reference points: {n_ref} (total cells: {total_ref_cells:.0f})")
        if aligned_df['size'].max() > 1:
            print(f"  Metacells detected - avg size: {aligned_df['size'].mean():.1f}, max size: {aligned_df['size'].max():.0f}")

        # Save original for comparison
        # Keep stable IDs to track what KNN filtering removed. (KNN resets indices.)
        aligned_df = aligned_df.copy()
        ref_df = ref_df.copy()
        if "__orig_idx" not in aligned_df.columns:
            aligned_df["__orig_idx"] = aligned_df.index.to_numpy()
        if "__orig_idx" not in ref_df.columns:
            ref_df["__orig_idx"] = ref_df.index.to_numpy()
        aligned_df_original = aligned_df.copy()

        # Store stable vertex IDs for triangulation remapping.
        # - If aligned_delaunay_vertex_col is None: assume triangles use aligned_df.index values.
        # - Else: assume triangles use aligned_df[aligned_delaunay_vertex_col] values.
        if aligned_delaunay_vertex_col is None:
            aligned_df["__tri_vid"] = aligned_df.index.to_numpy()
        else:
            if aligned_delaunay_vertex_col not in aligned_df.columns:
                raise ValueError(f"aligned_delaunay_vertex_col='{aligned_delaunay_vertex_col}' not in aligned_df")
            aligned_df["__tri_vid"] = aligned_df[aligned_delaunay_vertex_col].to_numpy()

        # Get valid pairs using KNN
        n_before_knn = len(aligned_df)
        if ignore_knn_if_matched:
            print("Using KNN with cell type priority")
            aligned_df, ref_df, valid_pairs = find_knn_with_cell_type_priority(aligned_df, ref_df, radius, knn=knn)
        else:
            print("Using all KNN")
            aligned_df, ref_df, valid_pairs = find_knn_within_radius(aligned_df, ref_df, radius, knn=knn)

        # Check how many metacells were filtered by KNN
        n_after_knn = len(aligned_df)
        if n_after_knn < n_before_knn:
            kept_orig = set(aligned_df["__orig_idx"].to_numpy())
            cells_lost = aligned_df_original.loc[~aligned_df_original["__orig_idx"].isin(kept_orig)]
            if 'size' in cells_lost.columns:
                print(f"\n⚠️  KNN filtered out {n_before_knn - n_after_knn} metacells (lost {cells_lost['size'].sum():.0f} cells)")
                print(f"  Size distribution of filtered metacells: {cells_lost['size'].describe()}")

        # Recompute sizes AFTER KNN filtering (aligned_df/ref_df may be smaller now).
        n_aligned = len(aligned_df)
        n_ref = len(ref_df)
        total_aligned_cells = aligned_df["size"].sum()
        total_ref_cells = ref_df["size"].sum()

        if n_after_knn != n_before_knn:
            print(
                f"\nPost-KNN sizes: aligned={n_aligned} (cells={total_aligned_cells:.0f}), "
                f"ref={n_ref} (cells={total_ref_cells:.0f})"
            )

        if len(valid_pairs) == 0:
            raise ValueError("No valid_pairs after KNN filtering. Increase radius and/or knn.")

        # Debug print for first pair
        first_pair = valid_pairs[0]
        
        # Precompute coordinate maps and valid pairs mapping
        print("\nPrecomputing coordinate maps and valid pairs...")
        aligned_coords, ref_coords, valid_pairs_map = precompute_coordinate_maps(aligned_df, ref_df, valid_pairs)
        
        # Debug print for valid pairs mapping of first point
        print(f"\nValid pairs mapping for first aligned point {first_pair[0]}:")
        print(f"All possible reference matches: {valid_pairs_map[first_pair[0]]}")
        
        # Compute / reuse Delaunay triangulation for the moving/aligned points
        print("\nComputing Delaunay triangulation...")
        aligned_coords_array = aligned_df[['X', 'Y']].values
        using_precomputed = False
        if aligned_delaunay is None or ignore_precomputed_triangulation:
            if ignore_precomputed_triangulation and aligned_delaunay is not None:
                print("  (ignoring precomputed triangulation, computing fresh Delaunay)")
            aligned_delaunay = Delaunay(aligned_coords_array).simplices
        else:
            # Remap user-provided triangles onto the post-KNN row indices using __tri_vid
            using_precomputed = True
            print("  (using precomputed triangulation, remapping to window)")
            aligned_delaunay = _remap_triangles_by_vertex_ids(
                aligned_delaunay,
                vertex_ids=aligned_df["__tri_vid"].to_numpy(),
            )
        
        # Filter triangles; if using precomputed, also identify unconstrained nodes to remove
        unconstrained_nodes = set()
        if using_precomputed:
            aligned_delaunay, unconstrained_nodes = filter_triangles_by_radius(
                aligned_coords_array, 
                aligned_delaunay, 
                radius,
                aligned_df=aligned_df,
                ignore_same_type_triangles=ignore_same_type_triangles,
                remove_unconstrained_nodes=True,
                min_angle_deg=min_angle_deg
            )
        else:
            aligned_delaunay = filter_triangles_by_radius(
                aligned_coords_array, 
                aligned_delaunay, 
                radius,
                aligned_df=aligned_df,
                ignore_same_type_triangles=ignore_same_type_triangles,
                min_angle_deg=min_angle_deg
            )
        
        # If using precomputed triangulation, remove unconstrained nodes from the optimization
        if unconstrained_nodes:
            print(f"\nRemoving {len(unconstrained_nodes)} unconstrained nodes from optimization...")
            # Filter valid_pairs to exclude pairs involving unconstrained aligned nodes
            valid_pairs = [(i, j) for i, j in valid_pairs if i not in unconstrained_nodes]
            print(f"Valid pairs after removing unconstrained: {len(valid_pairs)}")
            
            # Rebuild aligned_df without unconstrained nodes and remap indices
            constrained_nodes = sorted(set(range(len(aligned_df))) - unconstrained_nodes)
            old_to_new = {old: new for new, old in enumerate(constrained_nodes)}
            
            # Remap valid_pairs to new indices
            valid_pairs = [(old_to_new[i], j) for i, j in valid_pairs]
            
            # Remap triangles to new indices
            aligned_delaunay = [
                [old_to_new[v] for v in tri] 
                for tri in aligned_delaunay 
                if all(v in old_to_new for v in tri)
            ]
            aligned_delaunay = np.array(aligned_delaunay) if aligned_delaunay else np.array([]).reshape(0, 3)
            
            # Subset aligned_df
            aligned_df = aligned_df.iloc[constrained_nodes].reset_index(drop=True)
            aligned_coords_array = aligned_df[['X', 'Y']].values
            n_aligned = len(aligned_df)
            
            # Rebuild coordinate maps and valid_pairs_map
            aligned_coords, ref_coords, valid_pairs_map = precompute_coordinate_maps(aligned_df, ref_df, valid_pairs)
            
            print(f"After removal: {n_aligned} aligned points, {len(aligned_delaunay)} triangles")
        
        # Debug print for triangles containing first point
        if len(valid_pairs) > 0:
            first_pair = valid_pairs[0]
            print(f"\nTriangles containing first aligned point {first_pair[0]}:")
            triangles_with_first_point = [tri for tri in aligned_delaunay if first_pair[0] in tri]
            print(f"Number of triangles: {len(triangles_with_first_point)}")
            print(f"Triangle indices: {triangles_with_first_point}")
        
        # Create simplex mapping
        aligned_simplex_map = {i: set() for i in range(len(aligned_df))}
        for idx, simplex in enumerate(aligned_delaunay):
            for i in simplex:
                aligned_simplex_map[i].add(idx)
        
        # Debug print for simplex mapping of first point
        if len(valid_pairs) > 0:
            print(f"\nSimplex mapping for first aligned point {first_pair[0]}:")
            print(f"Triangle indices: {aligned_simplex_map[first_pair[0]]}")
        
        # Precompute triangle information
        print("\nPrecomputing triangle information...")
        triangle_info = precompute_triangle_info(aligned_df, aligned_delaunay, aligned_simplex_map)
        

        print("\nCreating Gurobi model...")
        model = Model("optimal_matches", env=env)
        
        # Add variables
        print("Adding variables...")
        x = model.addVars(len(valid_pairs), vtype=GRB.BINARY, lb=0, ub=1, name="x")
        penalty_vars = model.addVars(n_ref, vtype=GRB.CONTINUOUS, lb=0, ub=1000, name="penalty")
        no_match_vars = model.addVars(n_aligned, vtype=GRB.CONTINUOUS, lb=0, ub=1, name="no_match")
        model.update()
        
        # Add basic constraints
        print("\nAdding basic constraints...")
        model = add_basic_constraints_optimized(model, valid_pairs, n_ref, n_aligned,max_matches, x, penalty_vars, no_match_vars,aligned_df=aligned_df, ref_df=ref_df,ref_metacell_match_multiplier=ref_metacell_match_multiplier)
        
        # Add spatial constraints
        print("\nAdding spatial constraints...")

        # Precompute triangle weights (sum of vertex sizes)
        triangle_weights = []
        for tri in aligned_delaunay:
            a, b, c = tri
            tri_weight = (aligned_df.iloc[a]['size'] +
                         aligned_df.iloc[b]['size'] +
                         aligned_df.iloc[c]['size'])
            triangle_weights.append(tri_weight)

        if lazy_constraints:
            print("Using LAZY constraint generation (O(n) memory)")
            # Precompute source triangle orientations
            source_signs = []
            for tri in aligned_delaunay:
                a, b, c = tri
                ax, ay = aligned_df.iloc[a]['X'], aligned_df.iloc[a]['Y']
                bx, by = aligned_df.iloc[b]['X'], aligned_df.iloc[b]['Y']
                cx, cy = aligned_df.iloc[c]['X'], aligned_df.iloc[c]['Y']
                source_signs.append(np.sign((bx - ax) * (cy - ay) - (by - ay) * (cx - ax)))

            # Add penalty variables for triangles (one per triangle, not per combination)
            q_tri = model.addVars(len(aligned_delaunay), vtype=GRB.CONTINUOUS, lb=0, name="q_tri")
            model.update()

            # Store data on model for callback access
            model._x = x
            model._q_tri = q_tri
            model._valid_pairs = valid_pairs
            model._aligned_delaunay = aligned_delaunay
            model._source_signs = source_signs
            model._ref_coords = {j: (ref_df.iloc[j]['X'], ref_df.iloc[j]['Y']) for j in range(len(ref_df))}
            model._cuts_added = 0
            model._lazy_max_cuts = lazy_max_cuts
            model._lazy_allowed_flip_fraction = lazy_allowed_flip_fraction
            model._lazy_max_cuts_per_incumbent = lazy_max_cuts_per_incumbent

            # Enable lazy constraints
            model.Params.LazyConstraints = 1
            # Stop if the solver doesn't find a better solution for 600 seconds
            #model.Params.ImproveStartGap = 0.1 # Only start checking this once gap is < 2%
            #model.Params.ImproveStartNodes = 100000
            model.Params.Method = gp.GRB.METHOD_PDHG  # PDHG in Gurobi 13
            model.Params.PDHGGPU = 1   # enable GPU for PDHG
            # For objective, use q_tri instead of area_penalty_vars
            area_penalty_vars = [q_tri[i] for i in range(len(aligned_delaunay))]
            z_penalty_vars = []  # Not used in lazy mode
        else:
            print("Using EAGER constraint generation (O(n*k^3) memory)")
            area_penalty_vars, z_penalty_vars = add_spatial_constraints_triangle_based(
                model, valid_pairs_map, x, aligned_df, ref_df, valid_pairs, aligned_simplex_map, aligned_delaunay, hard_constraints=hard_spatial_constraints)
        model.update()

        # Set objective
        print("\nSetting objective function...")
        c = []
        dist_coeff = dist_ct_coeff * 0.001
        for idx, (i, j) in enumerate(valid_pairs):
            dist_ct = np.sum(np.abs(aligned_df.iloc[i][commonCT] - ref_df.iloc[j][commonCT]))
            dist_coords = np.abs(aligned_df.iloc[i]['X'] - ref_df.iloc[j]['X']) + \
                         np.abs(aligned_df.iloc[i]['Y'] - ref_df.iloc[j]['Y'])
            total_dist = (dist_ct_coeff * dist_ct + dist_coeff * dist_coords)
            c.append(total_dist)

        model.setObjective(
            quicksum(c[idx] * x[idx] for idx in range(len(valid_pairs))) +
            penalty_coeff * quicksum(penalty_vars[j] for j in range(n_ref)) +
            no_match_penalty * quicksum(aligned_df.iloc[i]['size'] * no_match_vars[i] for i in range(n_aligned)) +
            delaunay_penalty * quicksum(triangle_weights[idx] * area_penalty_var
                                       for idx, area_penalty_var in enumerate(area_penalty_vars)),
            GRB.MINIMIZE)

        # Optional MIP start initialization (x and no_match_vars)
        # This can significantly improve time-to-good-incumbent when max_matches == 1.
        apply_mip_start(
            x_vars=x,
            no_match_vars=no_match_vars,
            valid_pairs=valid_pairs,
            costs=c,
            n_aligned=n_aligned,
            n_ref=n_ref,
            aligned_sizes=aligned_df["size"].to_numpy(dtype=float),
            no_match_penalty=no_match_penalty,
            max_matches=max_matches,
            init_method=init_method,
            init_big_m=init_big_m,
            init_hungarian_max_n=init_hungarian_max_n,
            verbose=True,
        )
        
        # Save model to file for later inspection
        if outprefix:
            os.makedirs(outprefix, exist_ok=True)
            model_file = os.path.join(outprefix, 'matching_model.lp')
        else:
            model_file = 'matching_model.lp'
        print(f"\nSaving model to {model_file}...")
        model.write(model_file)
        
        # Set optimization parameters
        print("\nSetting optimization parameters...")
        model.Params.timeLimit = float(time_limit) if time_limit is not None else float("inf")
        model.Params.MIPGap = float(mip_gap)
        if mip_focus is not None:
            model.Params.MIPFocus = int(mip_focus)
        if cuts is not None:
            model.Params.Cuts = int(cuts)
        if heuristics is not None:
            model.Params.Heuristics = float(heuristics)
        # Let Gurobi use its defaults (typically uses all available threads).
        
        # Optimize
        print("\nStarting optimization...")
        if lazy_constraints:
            model.optimize(_lazy_orientation_callback)
            print(f"Lazy cuts added: {model._cuts_added}")
        else:
            model.optimize()
        time_limit_reached = model.status == GRB.TIME_LIMIT
        solve_time = model.Runtime
        print(f"Solve time: {solve_time:.2f} seconds")
        
        # Process results
        print("\nProcessing results...")
        if model.status == GRB.OPTIMAL or model.status == GRB.TIME_LIMIT:
            # Debug print for first pair result
            print(f"\nFirst pair ({valid_pairs[0]}) result:")
            print(f"x[0] value: {x[0].x}")
            print(f"no_match_vars[{valid_pairs[0][0]}] value: {no_match_vars[valid_pairs[0][0]].x}")
            if valid_pairs[0][1] < len(penalty_vars):
                print(f"penalty_vars[{valid_pairs[0][1]}] value: {penalty_vars[valid_pairs[0][1]].x}")
                    
            valid_matches = []
            for idx, (i, j) in enumerate(valid_pairs):
                if x[idx].x > 0.5:
                    valid_matches.append((i, j))
            
            out_df = pd.DataFrame(valid_matches, columns=['aligned_idx', 'ref_idx'])
            for ct in list(commonCT)+['X','Y']:
                out_df[ct] = out_df['aligned_idx'].map(aligned_df[ct])
            for ct in ['X','Y']:
                out_df[f'ref_{ct}'] = out_df['ref_idx'].map(ref_df[ct])

            # Copy size columns if they exist
            if 'size' in aligned_df.columns:
                out_df['size'] = out_df['aligned_idx'].map(aligned_df['size'])
            if 'size' in ref_df.columns:
                out_df['ref_size'] = out_df['ref_idx'].map(ref_df['size'])

            out_df[f'Ref_{cell_id_col}'] = out_df['ref_idx'].map(ref_df[cell_id_col])
            out_df[f'Aligned_{cell_id_col}'] = out_df['aligned_idx'].map(aligned_df[cell_id_col])
            out_df['time_limit_reached'] = time_limit_reached

            # Check for unmatched metacells
            if 'size' in aligned_df.columns:
                matched_indices = set(out_df['aligned_idx'].unique())
                all_indices = set(range(len(aligned_df)))
                unmatched_indices = all_indices - matched_indices

                if unmatched_indices:
                    unmatched_df = aligned_df.iloc[list(unmatched_indices)]
                    total_unmatched_cells = unmatched_df['size'].sum()
                    total_matched_cells = out_df['size'].sum()
                    total_cells_after_knn = aligned_df['size'].sum()

                    print(f"\n📊 Match Statistics:")
                    print(f"  Matched: {len(matched_indices)} metacells ({total_matched_cells:.0f} cells)")
                    print(f"  Unmatched: {len(unmatched_indices)} metacells ({total_unmatched_cells:.0f} cells)")
                    print(f"  Match rate: {100 * total_matched_cells / total_cells_after_knn:.1f}% of cells after KNN filtering")
                    print(f"  Unmatched size distribution: min={unmatched_df['size'].min()}, max={unmatched_df['size'].max()}, mean={unmatched_df['size'].mean():.2f}")

                    # Check no_match_vars for unmatched points
                    unmatched_no_match_vals = [no_match_vars[i].x for i in unmatched_indices]
                    print(f"  no_match_vars for unmatched: {sum(unmatched_no_match_vals)} / {len(unmatched_indices)} are set to 1")

            # Add spatial violation checking
            print("\nChecking spatial preservation...")
            violations = verify_spatial_preservation(
                aligned_df=aligned_df,
                ref_df=ref_df,
                matches_df=out_df,
                triangle_info=triangle_info
            )
            print_violation_report(violations)
            
            # Count non-zero penalty variables
            nonzero_area_penalties = sum(1 for var in area_penalty_vars if var.x > 1e-6)
            nonzero_z_penalties = sum(1 for var in z_penalty_vars if var.x > 1e-6)
            total_area_penalties = len(area_penalty_vars)
            total_z_penalties = len(z_penalty_vars)
            if total_area_penalties > 0:
                print(f"\nPenalty Variable Analysis:")
                print(f"Area penalties > 0: {nonzero_area_penalties}/{total_area_penalties} ({nonzero_area_penalties/total_area_penalties*100:.2f}%)")
            if total_z_penalties > 0:
                print(f"Z penalties > 0: {nonzero_z_penalties}/{total_z_penalties} ({nonzero_z_penalties/total_z_penalties*100:.2f}%)")
            
            # Compare violations with penalty variables
            print("\nComparing violations with penalty variables:")
            violation_points = set(violations['points_with_violations'])
            penalty_points = set()
            for idx, var in enumerate(area_penalty_vars):
                if var.x > 1e-6:
                    # Get triangle vertices - works for both lazy and eager modes
                    if lazy_constraints:
                        # Lazy mode: q_tri[idx] maps directly to triangle index
                        tri_idx = idx
                    else:
                        # Eager mode: parse from variable name area_penalty_tri{tri_idx}_{p1}_{p2}_{p3}
                        var_parts = var.VarName.split('_')
                        tri_idx = int(var_parts[2][3:])  # Remove 'tri' prefix

                    # Look up vertices from triangle
                    p1, p2, p3 = aligned_delaunay[tri_idx]

                    # Add all vertices of violated triangle
                    penalty_points.add(p1)
                    penalty_points.add(p2)
                    penalty_points.add(p3)
            
            points_both = violation_points.intersection(penalty_points)
            points_only_violations = violation_points - penalty_points
            points_only_penalties = penalty_points - violation_points
            
            print(f"Points with both violations and penalties: {len(points_both)}")
            print(f"Points with only violations: {len(points_only_violations)}")
            print(f"Points with only penalties: {len(points_only_penalties)}")
            print(f"Total matches: {sum(x[idx].x for idx in range(len(valid_pairs)))}")
            
            # Calculate triangle areas before and after matching
            triangle_areas_before = {}
            triangle_areas_after = {}
            triangle_flipped = []
            triangle_matched_vertices = {}

            # Calculate areas before matching
            for tri_idx, triangle in enumerate(aligned_delaunay):
                p1, p2, p3 = triangle
                coords = [
                    (aligned_df.iloc[p1]['X'], aligned_df.iloc[p1]['Y']),
                    (aligned_df.iloc[p2]['X'], aligned_df.iloc[p2]['Y']),
                    (aligned_df.iloc[p3]['X'], aligned_df.iloc[p3]['Y'])
                ]
                area = calculate_signed_area(*coords)
                triangle_areas_before[tri_idx] = area

            # Create mapping from aligned index to reference index
            aligned_to_ref = {}
            for idx, (i, j) in enumerate(valid_pairs):
                if x[idx].x > 0.5:
                    aligned_to_ref[i] = j

            # Calculate areas after matching
            for tri_idx, triangle in enumerate(aligned_delaunay):
                p1, p2, p3 = triangle
                
                # Track which vertices are matched
                matched = [p in aligned_to_ref for p in triangle]
                triangle_matched_vertices[tri_idx] = matched
                
                # Skip triangles where not all vertices are matched
                if not all(matched):
                    triangle_areas_after[tri_idx] = None
                    continue
                
                # Get reference coordinates for matched vertices
                ref_coords = [
                    (ref_df.iloc[aligned_to_ref[p]]['X'], ref_df.iloc[aligned_to_ref[p]]['Y'])
                    for p in triangle
                ]
                
                area = calculate_signed_area(ref_coords[0], ref_coords[1], ref_coords[2])
                triangle_areas_after[tri_idx] = area
                
                # Check if triangle flipped orientation
                if triangle_areas_before[tri_idx] * area < 0:
                    triangle_flipped.append(tri_idx)

            # Print summary
            print(f"\nTriangle area analysis:")
            print(f"Total triangles: {len(aligned_delaunay)}")
            print(f"Triangles with all vertices matched: {sum(1 for a in triangle_areas_after.values() if a is not None)}")
            print(f"Triangles with flipped orientation: {len(triangle_flipped)}")

            var_out = {
                'x': [x[idx].x for idx in range(len(valid_pairs))],
                'no_match_vars': [no_match_vars[idx].x for idx in range(n_aligned)],
                'penalty_vars': [penalty_vars[idx].x for idx in range(n_ref)],
                'area_penalty_vars': [area_penalty_vars[idx].x for idx in range(len(area_penalty_vars))],
                'violations': violations,
                'violation_penalty_comparison': {
                    'points_both': list(points_both),
                    'points_only_violations': list(points_only_violations),
                    'points_only_penalties': list(points_only_penalties)
                },
                'triangle_data': {
                    'triangles': aligned_delaunay,
                    'triangle_info': triangle_info,
                    'aligned_simplex_map': aligned_simplex_map,
                    'areas_before': triangle_areas_before,
                    'areas_after': triangle_areas_after,
                    'flipped_triangles': triangle_flipped,
                    'matched_vertices': triangle_matched_vertices
                },
                'lazy_constraints': lazy_constraints,
                'lazy_cuts_added': model._cuts_added if lazy_constraints else 0
            }

            # After optimization, add these prints:
            print("\nTriangle matching analysis:")
            for tri_idx, triangle in enumerate(aligned_delaunay):
                p1, p2, p3 = triangle
                
                # Check if points are matched
                matched_points = []
                for p in [p1, p2, p3]:
                    matches = [idx for idx, (i,j) in enumerate(valid_pairs) if i == p and x[idx].x > 0.5]
                    if matches:
                        matched_points.append((p, valid_pairs[matches[0]][1]))
                
                if len(matched_points) == 3:
                    # Calculate areas
                    aligned_area = signed_area_terms(aligned_df.iloc[list(triangle)])
                    ref_points = [ref_df.iloc[m[1]] for m in matched_points]
                    ref_area = signed_area_terms(pd.DataFrame(ref_points))
                    if aligned_area * ref_area < 0:
                        s=1

            # After computing Delaunay triangulation
            if outprefix:
                print(f"Saving data to {outprefix}...")
                # Save all data in var_out using numpy save
                np.save(os.path.join(outprefix, 'var_out.npy'), var_out, allow_pickle=True)

                # Save DataFrames
                aligned_df.to_csv(os.path.join(outprefix, 'aligned_df.csv'), index=False)
                ref_df.to_csv(os.path.join(outprefix, 'ref_df.csv'), index=False)

            # Add triangle_violation column to out_df based on actual signed area flips
            # (not optimizer penalty_points, which only captures a subset due to lazy constraints)
            flipped_nodes = set()
            for tri_idx in triangle_flipped:
                for v in aligned_delaunay[tri_idx]:
                    flipped_nodes.add(v)
            out_df['triangle_violation'] = out_df['aligned_idx'].isin(flipped_nodes)
            out_df['filtered_violation'] = out_df['aligned_idx'].isin(points_both)
            out_df['run_time'] = solve_time
            
        else:
            print("No optimal solution found")
            out_df = pd.DataFrame()
            var_out = {}
        
        if outprefix:
            out_df.to_csv(os.path.join(outprefix, 'matches_df.csv'), index=False)
        return out_df, var_out
    
    finally:
        # Try to remove log directory if empty
        try:
            if os.path.exists(log_dir) and not os.listdir(log_dir):
                os.rmdir(log_dir)
        except:
            pass
