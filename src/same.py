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

# Local imports
from .utils import preprocess_data, find_knn_within_radius
from .triangle_utils import compute_filtered_delaunay
from .violationhelper import verify_spatial_preservation, print_violation_report
from .knn_utils import find_knn_with_cell_type_priority
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

def subset_data(df, x_min, x_max, y_min, y_max):
    return df[(df['X'] >= x_min) & (df['X'] < x_max) & 
                (df['Y'] >= y_min) & (df['Y'] < y_max)]

def sliding_window_matching(ref, moving, commonCT, window_size=1000, overlap=250, max_matches=5, radius=500, penalty_coeff=50, no_match_penalty=100, delaunay_penalty=100, dist_ct_coeff=1, knn=5, min_cells_per_window=10, hard_spatial_constraints=False, ignore_same_type_triangles=False, ignore_knn_if_matched=False, outprefix=None):
    """
    Sliding window matching function.

    Parameters:
    -----------
    ref : pd.DataFrame
        Reference data
    moving : pd.DataFrame
        Moving data
    commonCT : list
        List of common cell types
    window_size : int
        Window size
    overlap : int
        Overlap size
    max_matches : int
        Maximum matches per window
    radius : int
        Radius for KNN
    penalty_coeff : int
        Penalty coefficient
    no_match_penalty : int
        No match penalty
    delaunay_penalty : int
        Delaunay penalty
    dist_ct_coeff : int
        Distance coefficient
    knn : int
        KNN
    min_cells_per_window : int
        Minimum cells per window
    hard_spatial_constraints : bool
        Hard spatial constraints
    ignore_same_type_triangles : bool
        Ignore same type triangles
    ignore_knn_if_matched : bool
        Ignore KNN if matched
    outprefix : str
        Output prefix
    """
    


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
        windows_to_process, existing_matches = get_unprocessed_windows(moving, output_file, x_windows, y_windows, window_size, overlap)
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
                    try:
                        # Create window-specific output directory if outprefix is provided
                        window_outprefix = None
                        if outprefix:
                            window_id = len(x_windows) * j + i
                            window_outprefix = os.path.join(outprefix, f'window_{window_id}')
                        
                        window_matches, var_out = run_same(
                            aligned_df=moving_subset, 
                            ref_df=ref_subset, 
                            commonCT=commonCT, 
                            max_matches=max_matches, 
                            radius=radius, 
                            penalty_coeff=penalty_coeff, 
                            no_match_penalty=no_match_penalty, 
                            delaunay_penalty=delaunay_penalty, 
                            knn=knn,
                            dist_ct_coeff=dist_ct_coeff,
                            hard_spatial_constraints=hard_spatial_constraints,
                            ignore_same_type_triangles=ignore_same_type_triangles,
                            ignore_knn_if_matched=ignore_knn_if_matched,
                            outprefix=window_outprefix)
                        
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

                    except Exception as e:
                        print(f'Error in window ({x}, {y}): {e}')
                
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


def run_same(ref_df, aligned_df, commonCT, max_matches=3, 
             radius=10, penalty_coeff=10, no_match_penalty=100,
             dist_ct_coeff=10, delaunay_penalty=100, knn=5,
             hard_spatial_constraints=False,
             ignore_same_type_triangles=False,
             ignore_knn_if_matched=False,
             outprefix=None):
    """Main function for finding optimal matches using Delaunay triangulation (renamed from find_optimal_matches_gurobi_delaunayv2)"""
    # Create a more robust log directory
    log_dir = os.path.join(os.getcwd(), 'gurobi_logs')
    os.makedirs(log_dir, exist_ok=True)
    log_file = os.path.join(log_dir, f'gurobi_{os.getpid()}.log')
    
    # Load Gurobi configuration from .gurobienv file or environment variables
    gurobi_config = load_gurobi_config()
    
    options = {
        'WLSACCESSID': gurobi_config.get('WLSACCESSID') or os.environ.get('GUROBI_WLSACCESSID', ''),
        'WLSSECRET': gurobi_config.get('WLSSECRET') or os.environ.get('GUROBI_WLSSECRET', ''),
        'LICENSEID': int(gurobi_config.get('LICENSEID', 0)) or int(os.environ.get('GUROBI_LICENSEID', 0)),
        'OutputFlag': 1,
        'LogFile': log_file
    }
    try:
        env = gp.Env(params=options)
        n_aligned = len(aligned_df)
        n_ref = len(ref_df)
        print(f"Number of aligned points: {n_aligned}")
        print(f"Number of reference points: {n_ref}")

        # Get valid pairs using KNN
        if ignore_knn_if_matched:
            print("Using KNN with cell type priority")
            aligned_df, ref_df, valid_pairs = find_knn_with_cell_type_priority(aligned_df, ref_df, radius, knn=knn)
        else:
            print("Using all KNN")
            aligned_df, ref_df, valid_pairs = find_knn_within_radius(aligned_df, ref_df, radius, knn=knn)
        
        # Debug print for first pair
        first_pair = valid_pairs[0]
        
        # Precompute coordinate maps and valid pairs mapping
        print("\nPrecomputing coordinate maps and valid pairs...")
        aligned_coords, ref_coords, valid_pairs_map = precompute_coordinate_maps(aligned_df, ref_df, valid_pairs)
        
        # Debug print for valid pairs mapping of first point
        print(f"\nValid pairs mapping for first aligned point {first_pair[0]}:")
        print(f"All possible reference matches: {valid_pairs_map[first_pair[0]]}")
        
        # Compute Delaunay triangulation
        print("\nComputing Delaunay triangulation...")
        aligned_coords_array = aligned_df[['X', 'Y']].values
        aligned_delaunay = Delaunay(aligned_coords_array).simplices
        aligned_delaunay = filter_triangles_by_radius(
            aligned_coords_array, 
            aligned_delaunay, 
            radius,
            aligned_df=aligned_df,
            ignore_same_type_triangles=ignore_same_type_triangles
        )
        
        # Debug print for triangles containing first point
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
        model = add_basic_constraints_optimized(model, valid_pairs, n_ref, n_aligned, 
                                              max_matches, x, penalty_vars, no_match_vars)
        
        # Add spatial constraints
        print("\nAdding spatial constraints...")
        area_penalty_vars, z_penalty_vars = add_spatial_constraints_triangle_based(
            model, valid_pairs_map, x, aligned_df, ref_df, valid_pairs, aligned_simplex_map, aligned_delaunay, hard_constraints=hard_spatial_constraints)
        model.update()
        
        # Set objective
        print("\nSetting objective function...")
        c = []
        dist_coeff = dist_ct_coeff*0.01
        for idx, (i, j) in enumerate(valid_pairs):
            dist_ct = np.sum(np.abs(aligned_df.iloc[i][commonCT] - ref_df.iloc[j][commonCT]))
            dist_coords = np.abs(aligned_df.iloc[i]['X'] - ref_df.iloc[j]['X']) + \
                         np.abs(aligned_df.iloc[i]['Y'] - ref_df.iloc[j]['Y'])
            total_dist = dist_ct_coeff * dist_ct + dist_coeff * dist_coords
            c.append(total_dist)
            

        model.setObjective(
            quicksum(c[idx] * x[idx] for idx in range(len(valid_pairs))) +
            penalty_coeff * quicksum(penalty_vars[j] for j in range(n_ref)) +
            no_match_penalty * quicksum(no_match_vars[i] for i in range(n_aligned)) +
            delaunay_penalty * (quicksum(area_penalty_var for area_penalty_var in area_penalty_vars)),
            GRB.MINIMIZE)
        
        # Save model to file for later inspection
        os.makedirs(outprefix, exist_ok=True)
        model_file = os.path.join(outprefix, 'matching_model.lp') if outprefix else 'matching_model.lp'
        print(f"\nSaving model to {model_file}...")
        model.write(model_file)
        
        # Set optimization parameters
        print("\nSetting optimization parameters...")
        model.Params.timeLimit = 7200
        model.Params.MIPGap = 0.005
        
        # Optimize
        print("\nStarting optimization...")
        model.optimize()
        time_limit_reached = False
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
            out_df['Ref_Cell_Num_Old'] = out_df['ref_idx'].map(ref_df['Cell_Num_Old'])
            out_df['Aligned_Cell_Num_Old'] = out_df['aligned_idx'].map(aligned_df['Cell_Num_Old'])
            out_df['time_limit_reached'] = time_limit_reached
            
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
            for var in area_penalty_vars:
                if var.x > 1e-6:
                    # Extract point index from variable name
                    var_parts = var.VarName.split('_')  
                    tri_idx = int(var_parts[2][3:])  # Remove 'tri' prefix
                    p1 = int(var_parts[3])  # First vertex
                    p2 = int(var_parts[4])  # Second vertex 
                    p3 = int(var_parts[5])  # Third vertex
                    
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
                }
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

            # Add triangle_violation column to out_df
            out_df['triangle_violation'] = out_df['aligned_idx'].isin(penalty_points)
            out_df['filtered_violation'] = out_df['aligned_idx'].isin(points_both)
            out_df['run_time'] = solve_time
            
        else:
            print("No optimal solution found")
            out_df = pd.DataFrame()
            var_out = {}
        
        out_df.to_csv(os.path.join(outprefix, 'matches_df.csv'), index=False)
        return out_df, var_out
    
    finally:
        # Try to remove log directory if empty
        try:
            if os.path.exists(log_dir) and not os.listdir(log_dir):
                os.rmdir(log_dir)
        except:
            pass
