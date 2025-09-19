from gurobipy import Model, GRB, quicksum
import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist
import gurobipy as gp
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
import networkx as nx


def get_unprocessed_windows(moving_df, output_name, x_windows, y_windows, window_size, overlap):
    """
    First check if windows exist in output, then only check cells for windows that don't exist.
    Returns a set of (i,j) tuples for windows that need processing.
    """
    # First create all window boundaries
    all_windows = {}
    for i, x in enumerate(x_windows):
        for j, y in enumerate(y_windows):
            x_window_min, x_window_max = x, x + window_size
            y_window_min, y_window_max = y, y + window_size
            
            # Store window boundaries and cells
            window_cells = moving_df[
                (moving_df['X'] >= x_window_min) & 
                (moving_df['X'] < x_window_max) & 
                (moving_df['Y'] >= y_window_min) & 
                (moving_df['Y'] < y_window_max)
            ]['Cell_Num_Old'].unique()
            
            if len(window_cells) > 0:  # Only store windows with cells
                all_windows[(i,j)] = {
                    'bounds': (x_window_min, x_window_max, y_window_min, y_window_max),
                    'cells': set(window_cells)
                }
    
    print(f"Created {len(all_windows)} windows with cells")
    
    try:
        existing_matches = pd.read_csv(output_name)
        # First check which windows exist in the output
        processed_windows = set()
        if 'window_id' in existing_matches.columns:
            processed_windows = set(existing_matches['window_id'].unique())
        
        # Convert window_id back to (i,j)
        processed_window_coords = {(window_id % len(x_windows), window_id // len(x_windows)) 
                                 for window_id in processed_windows}
        
        print(f"Found {len(processed_window_coords)} processed windows in existing file.")
        
        # Windows to process are those that don't exist in output
        windows_to_process = set(all_windows.keys()) - processed_window_coords
        print(f"Found {len(windows_to_process)} windows that need processing out of {len(all_windows)} total windows.")
        
        return windows_to_process, existing_matches
        
    except FileNotFoundError:
        print(f"No existing file found at {output_name}. Will process all {len(all_windows)} windows.")
        return set(all_windows.keys()), None


def calculate_signed_area(p1, p2, p3):
    x1, y1 = p1
    x2, y2 = p2
    x3, y3 = p3
    return 0.5 * (x1 * (y2 - y3) + x2 * (y3 - y1) + x3 * (y1 - y2))


def order_vertices_for_positive_area(points):
    # Extract the points and their names
    point_names = list(points.keys())
    point_values = list(points.values())
    
    # Calculate the signed area
    def signed_area(p1, p2, p3):
        return (p1[0] * (p2[1] - p3[1]) +
                p2[0] * (p3[1] - p1[1]) +
                p3[0] * (p1[1] - p2[1]))
    
    # Try all permutations of the points
    for perm in permutations(zip(point_names, point_values)):
        _, (p1, p2, p3) = zip(*perm)
        area = signed_area(p1, p2, p3)
        if area > 0:
            return [name for name, _ in perm]
    
    # If no positive area is found, return an empty list (should not happen with valid triangles)
    return []


def add_basic_constraints_optimized(model, valid_pairs, n_ref, n_aligned, max_matches, x, penalty_vars, no_match_vars):
    print("Building constraint mappings...")
    # Pre-build mappings of indices
    ref_to_pairs = defaultdict(list)    # j -> list of valid_pairs indices where jp == j
    aligned_to_pairs = defaultdict(list) # i -> list of valid_pairs indices where ip == i
    
    for idx, (ip, jp) in enumerate(valid_pairs):
        ref_to_pairs[jp].append(idx)
        aligned_to_pairs[ip].append(idx)
    
    # Track constraints added for each type
    n_max_match = 0
    n_one_match = 0
    n_penalty = 0
    n_no_match = 0

    print("Adding max matches constraints...")
    for j, pair_indices in tqdm(ref_to_pairs.items(), total=len(ref_to_pairs)):
        model.addConstr(quicksum(x[idx] for idx in pair_indices) <= max_matches, 
                       name=f'max_matches_{j}')
        n_max_match += 1
    model.update()

    print("Adding one match constraints...")
    for i, pair_indices in tqdm(aligned_to_pairs.items(), total=len(aligned_to_pairs)):
        model.addConstr(quicksum(x[idx] for idx in pair_indices) <= 1,
                       name=f'one_match_{i}')
        n_one_match += 1
    model.update()

    print("Adding penalty constraints...")
    for j, pair_indices in tqdm(ref_to_pairs.items(), total=len(ref_to_pairs)):
        model.addConstr(quicksum(x[idx] for idx in pair_indices) - penalty_vars[j] <= 1,
                       name=f'penalty_{j}')
        n_penalty += 1
    model.update()

    print("Adding no match constraints...")
    for i, pair_indices in tqdm(aligned_to_pairs.items(), total=len(aligned_to_pairs)):
        model.addConstr(quicksum(x[idx] for idx in pair_indices) + no_match_vars[i] == 1, name=f'no_match_{i}')
        n_no_match += 1
    model.update()
    
    return model


def precompute_coordinate_maps(aligned_df, ref_df, valid_pairs):
    """Precompute coordinate lookups and valid pairs mapping for faster access"""
    # Create coordinate lookup dictionaries
    aligned_coords = {
        i: {'X': aligned_df.iloc[i]['X'], 'Y': aligned_df.iloc[i]['Y']}
        for i in range(len(aligned_df))
    }
    ref_coords = {
        i: {'X': ref_df.iloc[i]['X'], 'Y': ref_df.iloc[i]['Y']}
        for i in range(len(ref_df))
    }
    
    # Create valid pairs mapping
    valid_pairs_map = defaultdict(list)
    for idx, (ip, jp) in enumerate(valid_pairs):
        valid_pairs_map[ip].append((idx, jp))
    
    return aligned_coords, ref_coords, valid_pairs_map


def precompute_triangle_info(aligned_df, aligned_delaunay, aligned_simplex_map):
    """Precompute triangle information including bounds and vertex positions"""
    triangle_info = {}
    for ip in range(len(aligned_df)):
        for simplex_idx in aligned_simplex_map[ip]:
            if simplex_idx not in triangle_info:
                simplex = aligned_delaunay[simplex_idx]
                coords = [(i, aligned_df.iloc[i]['X'], aligned_df.iloc[i]['Y']) 
                         for i in simplex]
                
                # Find min/max coordinates
                min_x = min(c[1] for c in coords)
                max_x = max(c[1] for c in coords)
                min_y = min(c[2] for c in coords)
                max_y = max(c[2] for c in coords)
                
                # Store vertex indices with their positions
                triangle_info[simplex_idx] = {
                    'vertices': simplex,
                    'bounds': {'min_x': min_x, 'max_x': max_x, 
                             'min_y': min_y, 'max_y': max_y},
                    'max_x_vertex': next(c[0] for c in coords if c[1] == max_x),
                    'min_x_vertex': next(c[0] for c in coords if c[1] == min_x),
                    'max_y_vertex': next(c[0] for c in coords if c[2] == max_y),
                    'min_y_vertex': next(c[0] for c in coords if c[2] == min_y)
                }
    return triangle_info


def add_position_constraint(model, x, vertex_pairs, ref_pairs, penalty_var, ref_coords, coord, constraint_type, name):
    """Add a single position constraint with penalty variable"""
    if constraint_type == 'max':
        ref_val = max(ref_coords[jp][coord] for _, jp in ref_pairs)
        model.addConstr(
            quicksum(ref_coords[jp][coord] * x[idx] for idx, jp in vertex_pairs) 
            <= ref_val + penalty_var,
            name=name
        )
        model.update()
    elif constraint_type == 'min':
        ref_val = min(ref_coords[jp][coord] for _, jp in ref_pairs)
        model.addConstr(
            quicksum(ref_coords[jp][coord] * x[idx] for idx, jp in vertex_pairs) 
            >= ref_val - penalty_var,
            name=name
        )
        model.update()


def filter_triangles_by_radius(points, triangles, radius, aligned_df=None, ignore_same_type_triangles=False):
    """
    Filter triangles to only keep those with all sides less than radius and optionally filter out same-type triangles.
    
    Parameters:
    -----------
    points : np.ndarray
        Array of point coordinates (n_points, 2)
    triangles : np.ndarray
        Array of triangle indices (n_triangles, 3)
    radius : float
        Maximum allowed side length
    aligned_df : pd.DataFrame, optional
        DataFrame containing cell type information
    ignore_same_type_triangles : bool, optional (default=False)
        If True, filter out triangles where all vertices have the same cell type
        
    Returns:
    --------
    filtered_triangles : list
        List of triangle indices that satisfy the radius and cell type constraints
    """
    filtered_triangles = []
    triangles_skipped_radius = 0
    triangles_skipped_type = 0
    
    for triangle in triangles:
        # Get the three vertices
        p1, p2, p3 = points[triangle]
        
        # Calculate side lengths
        side1 = np.linalg.norm(p2 - p1)
        side2 = np.linalg.norm(p3 - p2)
        side3 = np.linalg.norm(p1 - p3)
        
        # Check if all sides are less than radius
        if max(side1, side2, side3) >= radius:
            triangles_skipped_radius += 1
            continue
            
        # Check cell type constraint if requested
        if ignore_same_type_triangles and aligned_df is not None:
            vertex_types = set(aligned_df.iloc[v]['cell_type'] for v in triangle)
            if len(vertex_types) == 1:  # All vertices have same type
                triangles_skipped_type += 1
                continue
                
        filtered_triangles.append(triangle)
    
    print(f"\nTriangle filtering summary:")
    print(f"Total triangles: {len(triangles)}")
    print(f"Triangles skipped (radius): {triangles_skipped_radius}")
    if ignore_same_type_triangles:
        print(f"Triangles skipped (same type): {triangles_skipped_type}")
    print(f"Triangles kept: {len(filtered_triangles)}")
    
    return filtered_triangles


def signed_area_terms(points, x_col='X', y_col='Y'):
    """Return 1 for CCW orientation, -1 for CW orientation"""
    x1, y1 = points.iloc[0][x_col], points.iloc[0][y_col]
    x2, y2 = points.iloc[1][x_col], points.iloc[1][y_col]
    x3, y3 = points.iloc[2][x_col], points.iloc[2][y_col]
    
    area = np.round((x2-x1)*(y3-y1) - (y2-y1)*(x3-x1), 3)

    if area > 0:
        return 1
    elif area < 0:
        return -1
    else:
        return 0


def order_triangle_ccw(points):
    """Order triangle vertices counterclockwise"""
    # Center point
    center_x = points['X'].mean()
    center_y = points['Y'].mean()
    # Sort points by angle from center
    angles = np.arctan2(points['Y'] - center_y, points['X'] - center_x)
    return points.iloc[np.argsort(angles)]


# Define this function at the module level, not inside another function
def calc_ref_area(ref_indices_and_df):
    """Calculate reference area sign for a triangle"""
    ref_indices, ref_df_dict = ref_indices_and_df
    ref1, ref2, ref3 = ref_indices
    
    # Extract coordinates from the dataframe dictionary
    x1, y1 = ref_df_dict[ref1]['X'], ref_df_dict[ref1]['Y']
    x2, y2 = ref_df_dict[ref2]['X'], ref_df_dict[ref2]['Y']
    x3, y3 = ref_df_dict[ref3]['X'], ref_df_dict[ref3]['Y']
    
    area = np.round((x2-x1)*(y3-y1) - (y2-y1)*(x3-x1), 3)
    if area > 0:
        return 1
    elif area < 0:
        return -1
    else:
        return 0


def add_spatial_constraints_triangle_based(model, valid_pairs_map, x, aligned_df, ref_df, valid_pairs, aligned_simplex_map, aligned_delaunay, hard_constraints=False):
    """Add spatial constraints ensuring triangle orientation is preserved"""
    print("Adding triangle orientation constraints")
    area_penalty_vars = []
    z_penalty_vars = []

    # Create mapping from points to valid pair indices
    valid_pairs_imap = {ip:[] for ip in range(len(aligned_df))}
    for idx, (ip, jp) in enumerate(valid_pairs):
        valid_pairs_imap[ip].append(idx)
    
    # Pre-calculate all possible reference triangles and their areas
    print("Pre-calculating reference areas...")
    
    # Collect all unique reference triangle combinations
    ref_triangle_combos = []
    for triangle in aligned_delaunay:
        p1, p2, p3 = triangle
        pairs_1 = valid_pairs_imap[p1]
        pairs_2 = valid_pairs_imap[p2]
        pairs_3 = valid_pairs_imap[p3]
        
        for idx1 in pairs_1:
            ref1 = valid_pairs[idx1][1]
            for idx2 in pairs_2:
                ref2 = valid_pairs[idx2][1]
                for idx3 in pairs_3:
                    ref3 = valid_pairs[idx3][1]
                    ref_triangle_combos.append((ref1, ref2, ref3))
    
    # Remove duplicates
    ref_triangle_combos = list(set(ref_triangle_combos))
    print(f"Found {len(ref_triangle_combos)} unique reference triangle combinations")
    
    # Convert ref_df to a dictionary for easier pickling
    ref_df_dict = {i: {'X': row['X'], 'Y': row['Y']} for i, row in ref_df.iterrows()}
    
    # Prepare input for parallel processing
    ref_inputs = [(combo, ref_df_dict) for combo in ref_triangle_combos]
    
    # Calculate areas in parallel
    num_cores = max(1, multiprocessing.cpu_count() - 1)
    num_cores = 32
    print(f"Using {num_cores} cores for parallel processing")
    
    # Use a simpler approach without multiprocessing if needed
    ref_areas_dict = {}
    
    try:
        with ProcessPoolExecutor(max_workers=num_cores) as executor:
            results = list(executor.map(calc_ref_area, ref_inputs))
            ref_areas_dict = dict(results)
    except Exception as e:
        print(f"Parallel processing failed: {e}")
        print("Falling back to sequential processing...")
        # Sequential fallback
        for combo in tqdm(ref_triangle_combos, desc="Calculating reference areas"):
            ref1, ref2, ref3 = combo
            x1, y1 = ref_df.iloc[ref1]['X'], ref_df.iloc[ref1]['Y']
            x2, y2 = ref_df.iloc[ref2]['X'], ref_df.iloc[ref2]['Y']
            x3, y3 = ref_df.iloc[ref3]['X'], ref_df.iloc[ref3]['Y']
            
            area = np.round((x2-x1)*(y3-y1) - (y2-y1)*(x3-x1), 3)
            if area == 0:
                ref_areas_dict[combo] = 0
            else:
                ref_areas_dict[combo] = 1 if area > 0 else -1
    
    # Now process each triangle with the pre-calculated areas
    print("Adding constraints...")
    all_constraints = []
    
    for tri_idx, triangle in tqdm(enumerate(aligned_delaunay), total=len(aligned_delaunay)):
        # Get aligned points in original order
        aligned_points = aligned_df.iloc[list(triangle)]
        p1, p2, p3 = triangle  # Use original vertex order
        # Calculate aligned triangle area
        aligned_area = signed_area_terms(aligned_points)
        
        # Add penalty variable
        area_penalty_var = model.addVar(vtype=GRB.CONTINUOUS, lb=0, 
                                      name=f"area_penalty_tri{tri_idx}_{p1}_{p2}_{p3}")
        area_penalty_vars.append(area_penalty_var)

        # Get valid pairs for each vertex
        pairs_1 = valid_pairs_imap[p1]
        pairs_2 = valid_pairs_imap[p2]
        pairs_3 = valid_pairs_imap[p3]

        # For each possible combination of reference points
        for idx1 in pairs_1:
            ref1 = valid_pairs[idx1][1]
            for idx2 in pairs_2:
                ref2 = valid_pairs[idx2][1]
                for idx3 in pairs_3:
                    ref3 = valid_pairs[idx3][1]
                    
                    # Get pre-calculated reference area
                    ref_key = (ref1, ref2, ref3)
                    ref_area = ref_areas_dict.get(ref_key)
                    
                    # If not found (shouldn't happen), calculate it
                    if ref_area is None:
                        ref_points = pd.DataFrame([
                            ref_df.iloc[ref1],
                            ref_df.iloc[ref2],
                            ref_df.iloc[ref3]
                        ])
                        ref_area = signed_area_terms(ref_points)
                    
                    # Add constraint that areas must have same sign
                    z = model.addVar(vtype=GRB.CONTINUOUS, lb=0, ub=1,
                                   name=f"z_tri{tri_idx}_{idx1}_{idx2}_{idx3}")
                    z_penalty_vars.append(z)
                    
                    ## z = x1 * x2 * x3 linearization
                    all_constraints.append(z <= x[idx1])
                    all_constraints.append(z <= x[idx2])
                    all_constraints.append(z <= x[idx3])
                    all_constraints.append(z >= x[idx1] + x[idx2] + x[idx3] - 2)
                    
                    # Areas must have same sign (or pay penalty)
                    all_constraints.append(aligned_area * ref_area * z >= -area_penalty_var)
    
    # Add all constraints at once
    print(f"Adding {len(all_constraints)} constraints to model...")
    model.addConstrs((constraint for constraint in all_constraints))
    model.update()
    return area_penalty_vars, z_penalty_vars


def compute_filtered_violations(out_df, aligned_df, aligned_delaunay, var_out, triangle_info):
    """
    Compute filtered violations based on triangle orientation changes.
    
    Args:
        out_df: DataFrame with matching results
        aligned_df: DataFrame with aligned points
        aligned_delaunay: List of triangles
        var_out: Dictionary with optimization results
        triangle_info: Dictionary with triangle information
    
    Returns:
        set: Points with filtered violations
    """
    # Create sets for points with x and y penalties
    x_penalty_points = set()
    y_penalty_points = set()
    
    # Process x penalties
    for var_name, value in var_out['x_penalty_vars'].items():
        if value > 1e-6:
            var_parts = var_name.split('_')

            tri_idx = int(var_parts[2][3:])  # Remove 'tri' prefix
            vertex_pos = int(var_parts[3][1:])  # Remove 'v' prefix
            point_idx = triangle_info[tri_idx]['vertices'][vertex_pos]
            x_penalty_points.add(point_idx)
    
    # Process y penalties
    for var_name, value in var_out['y_penalty_vars'].items():
        if value > 1e-6:
            var_parts = var_name.split('_')
            tri_idx = int(var_parts[2][3:])  # Remove 'tri' prefix
            vertex_pos = int(var_parts[3][1:])  # Remove 'v' prefix
            point_idx = triangle_info[tri_idx]['vertices'][vertex_pos]
            y_penalty_points.add(point_idx)
    
    # Points with either x or y penalties
    violated_points = x_penalty_points.union(y_penalty_points)
    
    # Create position mappings for matched points
    pos_mapping = {}
    for idx, row in out_df.iterrows():
        aligned_idx = row['aligned_idx']
        pos_mapping[aligned_idx] = (row['ref_X'], row['ref_Y'])
    
    # Get aligned points coordinates
    aligned_points = [(row['X'], row['Y']) for _, row in aligned_df.iterrows()]
    
    # Filter violated points based on triangle area sign changes
    filtered_violated_points = set()
    
    # Get initial triangle areas using aligned points
    initial_areas = []
    for tri in aligned_delaunay:
        p1 = aligned_points[tri[0]]
        p2 = aligned_points[tri[1]]
        p3 = aligned_points[tri[2]]
        area = 0.5 * (p1[0]*(p2[1]-p3[1]) + p2[0]*(p3[1]-p1[1]) + p3[0]*(p1[1]-p2[1]))
        initial_areas.append(area)
    
    # Get final triangle areas using mapped points where possible
    final_areas = []
    for tri in aligned_delaunay:
        # Skip triangles where not all points are matched
        if not all(idx in pos_mapping for idx in tri):
            final_areas.append(None)
            continue
            
        p1 = pos_mapping[tri[0]]
        p2 = pos_mapping[tri[1]]
        p3 = pos_mapping[tri[2]]
        area = 0.5 * (p1[0]*(p2[1]-p3[1]) + p2[0]*(p3[1]-p1[1]) + p3[0]*(p1[1]-p2[1]))
        final_areas.append(area)
    
    # Check each violated point
    for point_idx in violated_points:
        # Find triangles containing this point
        for tri_idx, tri in enumerate(aligned_delaunay):
            if point_idx in tri:
                # Skip if triangle area couldn't be computed
                if final_areas[tri_idx] is None:
                    continue
                # If triangle changes orientation (area sign changes), keep the point
                if initial_areas[tri_idx] * final_areas[tri_idx] < 0:
                    filtered_violated_points.add(point_idx)
                    break
    
    return filtered_violated_points


def load_matching_results(outprefix):
    """
    Load matching results from the specified output directory.
    
    Args:
        outprefix: Path to directory containing saved results
        
    Returns:
        tuple: (var_out, aligned_df, ref_df, triangle_info)
            - var_out: Dictionary containing optimization results
            - aligned_df: DataFrame of aligned points
            - ref_df: DataFrame of reference points
            - triangle_info: Dictionary containing triangle information
    """
    # Load the main var_out data
    var_out = np.load(os.path.join(outprefix, 'var_out.npy'), allow_pickle=True).item()
    

    # Load DataFrames
    aligned_df = pd.read_csv(os.path.join(outprefix, 'aligned_df.csv'))
    ref_df = pd.read_csv(os.path.join(outprefix, 'ref_df.csv'))
    matches_df = pd.read_csv(os.path.join(outprefix, 'matches_df.csv'))
    return var_out, aligned_df, ref_df, matches_df


def merge_window_matches_unique_ref(matches_list):
    """Merge per-window matches and enforce one-to-one matching maximizing aligned count.

    This function concatenates a list of per-window match DataFrames and returns
    a single DataFrame with no duplicates on either endpoint: each
    `Aligned_Cell_Num_Old` and each `Ref_Cell_Num_Old` appears at most once.

    It maximizes the number of unique `Aligned_Cell_Num_Old` kept by computing a
    maximum-cardinality bipartite matching between `Aligned_Cell_Num_Old` and
    `Ref_Cell_Num_Old`. For duplicate occurrences of the same (aligned, ref)
    pair across windows, it prefers rows with `filtered_violation == False` and
    then smaller `window_id`.

    Required columns in each DataFrame:
    - 'window_id'
    - 'Aligned_Cell_Num_Old'
    - 'Ref_Cell_Num_Old'
    - 'X'
    - 'Y'
    - 'filtered_violation' (if missing, treated as True)

    Parameters
    ----------
    matches_list : list[pd.DataFrame]
        List of per-window matches DataFrames.

    Returns
    -------
    pd.DataFrame
        Maximum-cardinality one-to-one merged matches.
    """
    if not matches_list:
        return pd.DataFrame()

    merged_df = pd.concat(matches_list, ignore_index=True)

    required_cols = [
        'window_id',
        'Aligned_Cell_Num_Old',
        'Ref_Cell_Num_Old',
        'X',
        'Y',
        'filtered_violation'
    ]

    missing = [c for c in required_cols if c not in merged_df.columns]
    if missing:
        raise ValueError(f"Missing required columns in matches: {missing}")

    # Normalize filtered_violation: missing/NaN -> True (worst), ensure boolean
    merged_df['filtered_violation'] = (
        merged_df['filtered_violation']
        .fillna(True)
        .astype(bool)
    )

    # For identical (Aligned, Ref) pairs coming from multiple windows, keep a single
    # representative with preferred attributes: non-violating first, then smaller window_id.
    merged_df = merged_df.sort_values(
        by=['filtered_violation', 'window_id'], ascending=[True, True], kind='mergesort'
    )
    merged_df = merged_df.drop_duplicates(
        subset=['Aligned_Cell_Num_Old', 'Ref_Cell_Num_Old'], keep='first'
    )

    # Build bipartite graph (Aligned -> list of Ref)

    aligned_vals = merged_df['Aligned_Cell_Num_Old'].values
    ref_vals = merged_df['Ref_Cell_Num_Old'].values

    # Map node ids to 0..n-1 indices for arrays
    unique_aligned = sorted(pd.unique(aligned_vals))
    unique_ref = sorted(pd.unique(ref_vals))
    a_to_idx = {a: i for i, a in enumerate(unique_aligned)}
    b_to_idx = {b: i for i, b in enumerate(unique_ref)}

    # Adjacency from A-index to list of B-index
    adj = [[] for _ in range(len(unique_aligned))]
    # Keep a mapping from (ai, bi) to row index for reconstruction
    edge_row_index = {}

    for row_idx, (a, b) in enumerate(zip(aligned_vals, ref_vals)):
        ai = a_to_idx[a]
        bi = b_to_idx[b]
        adj[ai].append(bi)
        # Only store one representative row index per (ai, bi) (we already de-duped pairs)
        edge_row_index[(ai, bi)] = row_idx
    # Use networkx's hopcroft_karp_matching to find the maximum matching between Aligned and Ref nodes.

    align_nodes = [f"align_{a}" for a in unique_aligned]
    ref_nodes = [f"ref_{b}" for b in unique_ref]
    align_label_to_id = {f"align_{a}": a for a in unique_aligned}
    ref_label_to_id = {f"ref_{b}": b for b in unique_ref}

    G = nx.Graph()
    G.add_nodes_from(align_nodes, bipartite=0)
    G.add_nodes_from(ref_nodes, bipartite=1)
    for ai, a in enumerate(unique_aligned):
        for bi in adj[ai]:
            b = unique_ref[bi]
            G.add_edge(f"align_{a}", f"ref_{b}")

    # Compute maximum matching using networkx's Hopcroft-Karp implementation

    matching = nx.bipartite.hopcroft_karp_matching(G, top_nodes=set(align_nodes))

    selected_rows = []
    for a_label in align_nodes:
        b_label = matching.get(a_label)
        if b_label:
            a = align_label_to_id[a_label]
            b = ref_label_to_id[b_label]
            ai, bi = a_to_idx[a], b_to_idx[b]
            row_idx = edge_row_index.get((ai, bi))
            if row_idx is not None:
                selected_rows.append(row_idx)

    result_df = merged_df.iloc[selected_rows].copy().reset_index(drop=True)
    return result_df
