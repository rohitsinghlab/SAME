import numpy as np
import matplotlib.pyplot as plt
import scanpy as sc
from scipy.spatial import cKDTree
from scipy.spatial.distance import euclidean
from collections import Counter
import pandas as pd


from scipy.optimize import minimize

    


def create_spots(inAD, spot_diameter):
    # Get min and max coordinates from the AnnData object
    x_min, x_max = inAD.obs['Centroid X µm'].min(), inAD.obs['Centroid X µm'].max()
    y_min, y_max = inAD.obs['Centroid Y µm'].min(), inAD.obs['Centroid Y µm'].max()

    # Calculate center-to-center distance for equal horizontal, vertical, and diagonal spacing
    center_to_center = spot_diameter * 2  # µm

    # Calculate the radius of each spot
    radius = spot_diameter / 2

    # Calculate the number of spots in each dimension
    n_x = int((x_max - x_min) // (center_to_center * np.cos(np.pi/4))) + 1
    n_y = int((y_max - y_min) // (center_to_center * np.sin(np.pi/4))) + 1

    # Create a grid of spot centers
    x_centers = np.arange(x_min + center_to_center/2, x_max, center_to_center * np.cos(np.pi/4))
    y_centers = np.arange(y_min + center_to_center/2, y_max, center_to_center * np.sin(np.pi/4))
    xx, yy = np.meshgrid(x_centers, y_centers)

    # Offset every other row to create hexagonal-like pattern
    xx[1::2] += center_to_center * np.cos(np.pi/4) / 2

    # Create new columns in inAD.obs
    inAD.obs['CircleID'] = np.nan
    inAD.obs['circ_x'] = np.nan
    inAD.obs['circ_y'] = np.nan

    circID = 0
    for x, y in zip(xx.flatten(), yy.flatten()):
        # Calculate squared distances from circle center to all cells
        distances_squared = (inAD.obs['Centroid X µm'] - x)**2 + (inAD.obs['Centroid Y µm'] - y)**2
        # Find cells within the circle (distance <= radius)
        cells_in_spot = inAD.obs[distances_squared <= radius**2]
        if cells_in_spot.shape[0] > 0:
            inAD.obs.loc[cells_in_spot.index, 'CircleID'] = circID
            inAD.obs.loc[cells_in_spot.index, 'circ_x'] = x
            inAD.obs.loc[cells_in_spot.index, 'circ_y'] = y
            circID += 1

    return inAD

def plot_spots(inAD, spot_diameter):
    # Check if spots have been created, if not, create them
    if 'CircleID' not in inAD.obs.columns:
        inAD = create_spots(inAD, spot_diameter)

    # Get min and max coordinates from the AnnData object
    x_min, x_max = inAD.obs['Centroid X µm'].min(), inAD.obs['Centroid X µm'].max()
    y_min, y_max = inAD.obs['Centroid Y µm'].min(), inAD.obs['Centroid Y µm'].max()

    # Calculate the radius of each spot
    radius = spot_diameter / 2

    # Create a figure and axis
    fig, ax = plt.subplots(figsize=(2, 6))
    sc.pl.scatter(inAD, basis='spatial', color='Annotations', ax=ax, show=False)

    # Plot the circles representing spots
    for _, group in inAD.obs.groupby('CircleID'):
        if group.shape[0] > 0:
            x, y = group['circ_x'].iloc[0], group['circ_y'].iloc[0]
            circle = plt.Circle((x, y), radius, fill=True, facecolor='r', edgecolor='r', alpha=0.2)
            ax.add_artist(circle)

    # Set the aspect ratio to equal for circular appearance
    ax.set_aspect('equal')

    # Set the limits of the plot
    ax.set_xlim(x_min-10, x_max+10)
    ax.set_ylim(y_min-10, y_max+10)

    # Add labels and title
    ax.set_xlabel('Centroid X µm')
    ax.set_ylabel('Centroid Y µm')
    ax.set_title(f'Grid of {spot_diameter/2}µm radius circles')
    ax.invert_yaxis()

    # Show the plot
    plt.show()
def shuffle_points(points, seed):
    rng = np.random.default_rng(seed)
    shuffled_indices = rng.permutation(len(points))
    return points[shuffled_indices]

def find_closest_points(ref_df, aligned_df, commonCT, radius, max_matches=3, seed=2024):
    # Create KD-Tree for efficient nearest neighbor search
    ref_points = np.column_stack((ref_df['array_row'], ref_df['array_col']))
    moving_points = np.column_stack((aligned_df['scaled_circ_x'], aligned_df['scaled_circ_y']))
    # Shuffle the moving points with a random seed

    # Apply the shuffling to moving_points
    moving_points = shuffle_points(moving_points, seed)

    shuffled_indices = np.arange(len(moving_points))
    shuffled_indices = shuffle_points(shuffled_indices, seed)
    tree = cKDTree(ref_points)

    matches = []
    match_counts = Counter()

    for i, idx in enumerate(shuffled_indices):
        point = moving_points[i]
        # Find all neighbors within the specified radius
        neighbors = tree.query_ball_point(point, r=radius)
        
        if neighbors:
            # Calculate Euclidean distances for cell type compositions
            distances = []
            valid_neighbors = []
            for neighbor in neighbors:
                if neighbor in ref_df.index and match_counts[neighbor] < max_matches:
                    distances.append(euclidean(aligned_df.loc[idx, commonCT], ref_df.loc[neighbor, commonCT]))
                    valid_neighbors.append(neighbor)
            
            if distances:
                # Find the index of the minimum distance
                min_distance_idx = np.argmin(distances)
                # Find the index of the minimum distance
                min_distance_idx = np.argmin(distances)
                min_distance = distances[min_distance_idx]
                best_match = valid_neighbors[min_distance_idx]
                
                matches.append((idx, best_match))
                match_counts[best_match] += 1
            else:
                if 0:
                    print(f"Warning: No valid neighbors found for point {idx}")
                else:
                    pass
    # Create a new DataFrame with the matched points
    matched_df = pd.DataFrame(matches, columns=['aligned_idx', 'ref_idx'])
    # Drop rows with NaN values
    matched_df = matched_df.dropna()
    for ct in list(commonCT)+['scaled_circ_x','scaled_circ_y']:
        matched_df[ct] = matched_df['aligned_idx'].map(aligned_df[ct])
    for ct in ['array_row','array_col']:
        matched_df[ct] = matched_df['ref_idx'].map(ref_df[ct])

    return matched_df

# Normalize ref_df[commonCT] to sum to 1 (or 0 if sum is 0)
def normalize_row(row):
    row_sum = row.sum()
    if row_sum == 0:
        return row  # Return the original row if sum is 0
    return row / row_sum

def compute_objective(ref_df, aligned_df, commonCT, max_matches, x):
    total_distance = 0
    match_counts = Counter()
    for i in range(len(aligned_df)):
        ref_idx = int(x[i])
        if ref_idx >= 0 and ref_idx < len(ref_df):
            total_distance += euclidean(aligned_df.iloc[i][commonCT], ref_df.iloc[ref_idx][commonCT])
            match_counts[ref_idx] += 1
    penalty = sum(max(0, count - max_matches) for count in match_counts.values())
    return total_distance + penalty * 100  # Large penalty for exceeding max_matches



def find_optimal_matches(ref_df, aligned_df, commonCT, max_matches=3, radius=10):

    def objective(x):
        total_distance = 0
        match_counts = Counter()
        for i in range(len(aligned_df)):
            ref_idx = int(x[i])
            if ref_idx >= 0 and ref_idx < len(ref_df):
                total_distance += euclidean(aligned_df.iloc[i][commonCT], ref_df.iloc[ref_idx][commonCT])
                match_counts[ref_idx] += 1
        penalty = sum(max(0, count - max_matches) for count in match_counts.values())
        return total_distance + penalty * 100  # Large penalty for exceeding max_matches
    def constraint(x):
        return [radius - euclidean((aligned_df.iloc[i]['scaled_circ_x'], aligned_df.iloc[i]['scaled_circ_y']), (ref_df.iloc[int(x[i])]['array_row'], ref_df.iloc[int(x[i])]['array_col'])) if x[i] >= 0 and x[i] < len(ref_df) else 0 for i in range(len(aligned_df))]

    x0 = np.full(len(aligned_df), -1)  # Initial guess: no matches
    bounds = [(0, len(ref_df) - 1) for _ in range(len(aligned_df))]
    constraints = {'type': 'ineq', 'fun': constraint}
    result = minimize(objective, x0, bounds=bounds, constraints=constraints, method='SLSQP')

    valid_matches = [(i, int(result.x[i])) for i in range(len(aligned_df)) if 0 <= result.x[i] < len(ref_df)]

    return valid_matches

from scipy.optimize import linprog
from scipy.sparse import lil_matrix

from scipy.optimize import linprog
import numpy as np

def find_optimal_matches_lp(ref_df, aligned_df, commonCT, max_matches=3, radius=10):
    n_aligned = len(aligned_df)
    n_ref = len(ref_df)

    # Objective function: minimize the sum of absolute differences (Manhattan distance)
    c = []
    for i in range(n_aligned):
        for j in range(n_ref):
            dist = np.sum(np.abs(aligned_df.iloc[i][commonCT] - ref_df.iloc[j][commonCT]))
            c.append(dist)
    
    # Penalty for ref points matching to more than one aligned point
    for j in range(n_ref):
        penalty_col = np.zeros(n_aligned * n_ref)
        penalty_col[j::n_ref] = 100
        c += penalty_col.tolist()

    # Constraints: distance must be within the square of side 'radius'
    A_dist = []
    b_dist = []
    for i in range(n_aligned):
        for j in range(n_ref):
            dx = np.abs(aligned_df.iloc[i]['scaled_circ_x'] - ref_df.iloc[j]['array_row'])
            dy = np.abs(aligned_df.iloc[i]['scaled_circ_y'] - ref_df.iloc[j]['array_col'])
            if dx <= radius and dy <= radius:
                constraint = np.zeros(n_aligned * n_ref)
                constraint[i * n_ref + j] = 1
                A_dist.append(constraint)
                b_dist.append(1)

    A_dist = np.array(A_dist)
    b_dist = np.array(b_dist)

    # Combine all constraints
    A = A_dist
    b = b_dist

    # Bounds for the decision variables (0 or 1)
    bounds = [(0, 1) for _ in range(n_aligned * n_ref)]

    # Solve the linear programming problem
    result = linprog(c, A_ub=A, b_ub=b, bounds=bounds, method='highs')

    # Extract the valid matches
    valid_matches = []
    if result.success:
        x = result.x.reshape((n_aligned, n_ref))
        for i in range(n_aligned):
            for j in range(n_ref):
                if x[i, j] > 0.5:  # Consider as matched if the value is close to 1
                    valid_matches.append((i, j))

    return valid_matches

def find_optimal_matches_lpv2(ref_df, aligned_df, commonCT, max_matches=3, radius=10):
    n_aligned = len(aligned_df)
    n_ref = len(ref_df)

    # Vectorized computation of distances
    aligned_coords = aligned_df[['scaled_circ_x', 'scaled_circ_y']].values
    ref_coords = ref_df[['array_row', 'array_col']].values
    distances = np.sqrt(((aligned_coords[:, np.newaxis, :] - ref_coords[np.newaxis, :, :]) ** 2).sum(axis=2))

    # Find pairs within the radius
    valid_pairs = np.argwhere(distances <= radius)
    print(len(valid_pairs))
    # Check if each i from n_aligned is in valid_pairs
    i_in_valid_pairs = np.unique(valid_pairs[:, 0])
    missing_i = set(range(n_aligned)) - set(i_in_valid_pairs)
    
    if missing_i:
        print(f"Warning: The following i values from n_aligned are not in valid_pairs: {missing_i}")
    else:
        print("All i values from n_aligned are present in valid_pairs")
    # Objective function: minimize the sum of absolute differences (Manhattan distance)
    c = []

    for i, j in valid_pairs:
        dist = np.sum(np.abs(aligned_df.iloc[i][commonCT] - ref_df.iloc[j][commonCT]))
        c.append(dist)

    # Update c, valid_pairs, and other relevant variables
    c = np.array(c)
    valid_pairs = extended_valid_pairs
    # Constraint: each j in valid pairs must be used at most max_matches times
    A_ub = []
    b_ub = []
    for j in range(n_ref):
        constraint = np.zeros(len(valid_pairs))
        for idx, (_, jp) in enumerate(valid_pairs):
            if j == jp:
                constraint[idx] = 1
        if np.any(constraint):
            A_ub.append(constraint)
            b_ub.append(max_matches)
    
    A_eq = []
    b_eq = []
    for i in range(n_aligned):
        constraint = np.zeros(len(valid_pairs))
        for idx, (ip, _) in enumerate(valid_pairs):
            if i == ip:
                constraint[idx] = 1
        if np.any(constraint):
            A_eq.append(constraint)
            b_eq.append(1)

    A_ub = np.array(A_ub)
    b_ub = np.array(b_ub)


    bounds = [(0, 1) for _ in range(len(valid_pairs))]

    # Solve the linear programming problem
    result = linprog(c, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq, bounds=bounds, method='highs')
    print(result)
    # Extract the valid matches
    valid_matches = []
    if result.success:
        x = result.x
        for idx, (i, j) in enumerate(valid_pairs):
            if x[idx] > 0.5:  # Consider as matched if the value is close to 1
                valid_matches.append((i, j))
        out_df = pd.DataFrame(valid_matches, columns=['aligned_idx', 'ref_idx'])
        for ct in list(commonCT)+['scaled_circ_x','scaled_circ_y']:
            out_df[ct] = out_df['aligned_idx'].map(aligned_df[ct])
        for ct in ['array_row','array_col']:
            out_df[ct] = out_df['ref_idx'].map(ref_df[ct])
    else:
        print("No valid matches found")

    return out_df

from scipy.optimize import linprog
import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist

# Precompute nearest neighbors for all points
def find_nearest_neighbors(coords, k=9):
    tree = cKDTree(coords)
    _, indices = tree.query(coords, k=k)  # k because it includes the point itself
    return indices

def find_optimal_matches_lpv3(ref_df, aligned_df, commonCT, max_matches=3, radius=10, penalty_coeff=10):
    n_aligned = len(aligned_df)
    n_ref = len(ref_df)

    # Vectorized computation of distances
    aligned_coords = aligned_df[['X', 'Y']].values
    ref_coords = ref_df[['X', 'Y']].values
    distances = np.sqrt(((aligned_coords[:, np.newaxis, :] - ref_coords[np.newaxis, :, :]) ** 2).sum(axis=2))

    # Find pairs within the radius
    valid_pairs = np.argwhere(distances <= radius)
    print(len(valid_pairs))
    # Check if each i from n_aligned is in valid_pairs
    i_in_valid_pairs = np.unique(valid_pairs[:, 0])
    missing_i = set(range(n_aligned)) - set(i_in_valid_pairs)
    
    if missing_i:
        print(f"Warning: The following i values from n_aligned are not in valid_pairs: {missing_i}")
    else:
        print("All i values from n_aligned are present in valid_pairs")
    # Objective function: minimize the sum of absolute differences (Manhattan distance)
    c = []
    dist_coeff = 0.1
    for i, j in valid_pairs:
        dist_ct = np.sum(np.abs(aligned_df.iloc[i][commonCT] - ref_df.iloc[j][commonCT]))
        dist_coords = np.abs(aligned_df.iloc[i]['X'] - ref_df.iloc[j]['X']) + \
                      np.abs(aligned_df.iloc[i]['Y'] - ref_df.iloc[j]['Y'])
        total_dist = dist_ct + dist_coeff*dist_coords
        c.append(total_dist)

    # Add penalty terms to the objective function
    c.extend([penalty_coeff] * n_ref)


    n_vars = len(valid_pairs) + n_ref 
    # Constraint: each j in valid pairs must be used at most max_matches times (hard constraint)
    A_ub = []
    b_ub = []
    for j in range(n_ref):
        constraint = np.zeros(n_vars)
        for idx, (_, jp) in enumerate(valid_pairs):
            if j == jp:
                constraint[idx] = 1
        A_ub.append(constraint)
        b_ub.append(max_matches)
    # Constraint: each j in valid pairs incurs a penalty if used more than once
    for j in range(n_ref):
        constraint = np.zeros(n_vars)
        for idx, (_, jp) in enumerate(valid_pairs):
            if j == jp:
                constraint[idx] = 1
        constraint[len(valid_pairs) + j] = -1  # Penalty term
        A_ub.append(constraint)
        b_ub.append(1)

    # Constraint: if xij is 1, then at least k of neighboring x variables should be 1
    k=0
    print(k)
    aligned_neighbors = find_nearest_neighbors(aligned_coords)
    ref_neighbors = find_nearest_neighbors(ref_coords)

    # Create a dictionary to quickly look up indices in valid_pairs
    valid_pairs_dict = {(i, j): idx for idx, (i, j) in enumerate(valid_pairs)}
    print('Skipping the constraint')
    for idx, (i, j) in enumerate(valid_pairs):
        constraint = np.zeros(n_vars)
        
        # Get the indices of the 8 nearest neighbors
        i_neighbors = aligned_neighbors[i][1:]  # Exclude the point itself
        j_neighbors = ref_neighbors[j][1:]  # Exclude the point itself
        
        # Find valid neighbor pairs
        neighbor_count = 0
        for ni in i_neighbors:
            for nj in j_neighbors:
                if (ni, nj) in valid_pairs_dict:
                    neighbor_idx = valid_pairs_dict[(ni, nj)]
                    constraint[neighbor_idx] = -1
                    neighbor_count += 1
        constraint[idx] = min(k,neighbor_count)  # xij

        #A_ub.append(constraint)
        #b_ub.append(0)  # xij +(sum of neighboring x) >=k
    # Constraint: each i in valid pairs must be used exactly once
    A_eq = []
    b_eq = []
    for i in range(n_aligned):
        constraint = np.zeros(n_vars)
        for idx, (ip, _) in enumerate(valid_pairs):
            if i == ip:
                constraint[idx] = 1
        A_ub.append(constraint)
        b_ub.append(1)


    A_ub = np.array(A_ub)
    b_ub = np.array(b_ub)
    A_eq = np.array(A_eq)
    b_eq = np.array(b_eq)

    # Update the number of variables
    bounds = [(0, 1) for _ in range(len(valid_pairs))] + [(0, None) for _ in range(n_ref)]


    # Solve the linear programming problem
    result = linprog(c, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq, bounds=bounds, method='highs')
    print(result)
    # Extract the valid matches
    valid_matches = []
    if result.success:
        x = result.x[:len(valid_pairs)]
        print(x)
        nnz = np.count_nonzero(x)
        print(f"Number of non-zero elements in x: {nnz}")
        for idx, (i, j) in enumerate(valid_pairs):
            if x[idx] > 0.:  # Consider as matched if the value is >0
                valid_matches.append((i, j))
        out_df = pd.DataFrame(valid_matches, columns=['aligned_idx', 'ref_idx'])
        for ct in list(commonCT)+['X','Y']:
            out_df[ct] = out_df['aligned_idx'].map(aligned_df[ct])
        for ct in ['X','Y']:
            out_df[f'ref_{ct}'] = out_df['ref_idx'].map(ref_df[ct])
    else:
        print("No valid matches found")

    return out_df

from scipy.optimize import milp
import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

from scipy.optimize import milp, LinearConstraint, Bounds
import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

def find_optimal_matches_milp(ref_df, aligned_df, commonCT, max_matches=3, radius=10, penalty_coeff=10, k=2):
    n_aligned = len(aligned_df)
    n_ref = len(ref_df)

    # Vectorized computation of distances
    aligned_coords = aligned_df[['X', 'Y']].values
    ref_coords = ref_df[['X', 'Y']].values
    distances = np.sqrt(((aligned_coords[:, np.newaxis, :] - ref_coords[np.newaxis, :, :]) ** 2).sum(axis=2))

    # Find pairs within the radius
    valid_pairs = np.argwhere(distances <= radius)
    print(f"Number of valid pairs: {len(valid_pairs)}")

    # Objective function: minimize the sum of absolute differences (Manhattan distance)
    c = []
    dist_coeff = 0.1
    for i, j in valid_pairs:
        dist_ct = np.sum(np.abs(aligned_df.iloc[i][commonCT] - ref_df.iloc[j][commonCT]))
        dist_coords = np.abs(aligned_df.iloc[i]['X'] - ref_df.iloc[j]['X']) + \
                      np.abs(aligned_df.iloc[i]['Y'] - ref_df.iloc[j]['Y'])
        total_dist = dist_ct + dist_coeff * dist_coords
        c.append(total_dist)

    # Add penalty terms to the objective function
    c.extend([penalty_coeff] * n_ref)

    n_vars = len(valid_pairs) + n_ref

    constraints = []

    # Constraint: each j in valid pairs must be used at most max_matches times
    for j in range(n_ref):
        constraint = np.zeros(n_vars)
        for idx, (_, jp) in enumerate(valid_pairs):
            if j == jp:
                constraint[idx] = 1
        constraints.append(LinearConstraint(constraint, -np.inf, max_matches))

    # Constraint: each j in valid pairs incurs a penalty if used more than once
    for j in range(n_ref):
        constraint = np.zeros(n_vars)
        for idx, (_, jp) in enumerate(valid_pairs):
            if j == jp:
                constraint[idx] = 1
        constraint[len(valid_pairs) + j] = -1  # Penalty term
        constraints.append(LinearConstraint(constraint, -np.inf, 1))


    aligned_neighbors = find_nearest_neighbors(aligned_coords, )
    ref_neighbors = find_nearest_neighbors(ref_coords)

    # Create a dictionary to quickly look up indices in valid_pairs
    valid_pairs_dict = {(i, j): idx for idx, (i, j) in enumerate(valid_pairs)}
    k =4
    # Constraint: if xij is 1, then at least k of neighboring x variables should be 1
    for idx, (i, j) in enumerate(valid_pairs):
        constraint = np.zeros(n_vars)
        constraint[idx] = k  # k * xij
        
        # Get the indices of the 8 nearest neighbors
        i_neighbors = aligned_neighbors[i][1:]  # Exclude the point itself
        j_neighbors = ref_neighbors[j][1:]  # Exclude the point itself
        
        # Find valid neighbor pairs
        for ni in i_neighbors:
            for nj in j_neighbors:
                if (ni, nj) in valid_pairs_dict:
                    neighbor_idx = valid_pairs_dict[(ni, nj)]
                    constraint[neighbor_idx] = -1  # -1 * xij_neighbor
        
        constraints.append(LinearConstraint(constraint, -np.inf, 0))

    # Constraint: each i in valid pairs must be used exactly once
    for i in range(n_aligned):
        constraint = np.zeros(n_vars)
        for idx, (ip, _) in enumerate(valid_pairs):
            if i == ip:
                constraint[idx] = 1
        constraints.append(LinearConstraint(constraint, 1, 1))

    # Bounds and integrality
    lb = np.zeros(n_vars)
    ub = np.ones(n_vars)
    ub[len(valid_pairs):] = np.inf  # Upper bound for penalty variables
    bounds = Bounds(lb, ub)
    
    integrality = np.zeros(n_vars)
    integrality[:len(valid_pairs)] = 1  # Set to 1 for integer variables (our x_ij)

    # Solve the mixed integer linear programming problem
    result = milp(c=c, constraints=constraints, integrality=integrality, bounds=bounds)

    print(result)

    # Extract the valid matches
    valid_matches = []
    if result.success:
        x = result.x[:len(valid_pairs)]
        print(f"Number of non-zero elements in x: {np.count_nonzero(x)}")
        for idx, (i, j) in enumerate(valid_pairs):
            if x[idx] > 0.5:  # Consider as matched if the value is close to 1
                valid_matches.append((i, j))
        out_df = pd.DataFrame(valid_matches, columns=['aligned_idx', 'ref_idx'])
        # Keep a copy of old X and Y
        for ct in list(commonCT)+['X','Y']:
            out_df[ct] = out_df['aligned_idx'].map(aligned_df[ct])
        for ct in ['X','Y']:
            out_df[f'ref_{ct}'] = out_df['ref_idx'].map(ref_df[ct])

    else:
        print("No valid matches found")
        out_df = pd.DataFrame()

    return out_df

def find_optimal_matches_lpv4(ref_df, aligned_df, commonCT, max_matches=3, radius=10, penalty_coeff=10, no_match_penalty=100):
    n_aligned = len(aligned_df)
    n_ref = len(ref_df)

    # Vectorized computation of distances
    aligned_coords = aligned_df[['X', 'Y']].values
    ref_coords = ref_df[['X', 'Y']].values
    distances = np.sqrt(((aligned_coords[:, np.newaxis, :] - ref_coords[np.newaxis, :, :]) ** 2).sum(axis=2))

    # Find pairs within the radius
    valid_pairs = np.argwhere(distances <= radius)
    print(len(valid_pairs))
    # Check if each i from n_aligned is in valid_pairs
    i_in_valid_pairs = np.unique(valid_pairs[:, 0])
    missing_i = set(range(n_aligned)) - set(i_in_valid_pairs)
    
    if missing_i:
        print(f"Warning: The following i values from n_aligned are not in valid_pairs: {missing_i}")
    else:
        print("All i values from n_aligned are present in valid_pairs")
    # Objective function: minimize the sum of absolute differences (Manhattan distance)
    c = []
    dist_coeff = 0.005
    neighbor_coeff = -1
    for i, j in valid_pairs:
        dist_ct = np.sum(np.abs(aligned_df.iloc[i][commonCT] - ref_df.iloc[j][commonCT]))
        dist_coords = np.abs(aligned_df.iloc[i]['X'] - ref_df.iloc[j]['X']) + \
                      np.abs(aligned_df.iloc[i]['Y'] - ref_df.iloc[j]['Y'])
        total_dist = dist_ct + dist_coeff*dist_coords
        c.append(total_dist)

    # Add penalty terms to the objective function
    c.extend([penalty_coeff] * n_ref)
    # Add new penalty terms for each i not having a match
    c.extend([no_match_penalty] * n_aligned)


    n_vars = len(valid_pairs) + n_ref + n_aligned 
    # Constraint: each j in valid pairs must be used at most max_matches times (hard constraint)
    A_ub = []
    b_ub = []
    for j in range(n_ref):
        constraint = np.zeros(n_vars)
        for idx, (_, jp) in enumerate(valid_pairs):
            if j == jp:
                constraint[idx] = 1
        A_ub.append(constraint)
        b_ub.append(max_matches)
    # Constraint: each j in valid pairs incurs a penalty if used more than once
    for j in range(n_ref):
        constraint = np.zeros(n_vars)
        for idx, (_, jp) in enumerate(valid_pairs):
            if j == jp:
                constraint[idx] = 1
        constraint[len(valid_pairs) + j] = -1  # Penalty term
        A_ub.append(constraint)
        b_ub.append(1)

    # Constraint: each i in valid pairs must be used at most once and incurs a penalty if not used
    for i in range(n_aligned):
        constraint = np.zeros(n_vars)
        for idx, (ip, _) in enumerate(valid_pairs):
            if i == ip:
                constraint[idx] = 1
        constraint[len(valid_pairs) + n_ref + i] = 1  # Changed from -1 to 1
        A_ub.append(-constraint)  # Note the negative sign
        b_ub.append(-1)  # Changed from 1 to -1


    A_ub = np.array(A_ub)
    b_ub = np.array(b_ub)
    # Update the number of variables
    bounds = [(0, 1) for _ in range(len(valid_pairs))] + [(0, None) for _ in range(n_ref + n_aligned )]

    # Solve the linear programming problem
    result = linprog(c, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method='highs')
    print(result)
    # Extract the valid matches
    valid_matches = []
    if result.success:
        x = result.x[:len(valid_pairs)]
        print(x)
        nnz = np.count_nonzero(x)
        print(f"Number of non-zero elements in x: {nnz}")
        for idx, (i, j) in enumerate(valid_pairs):
            if x[idx] > 0.:  # Consider as matched if the value is >0
                valid_matches.append((i, j))
        out_df = pd.DataFrame(valid_matches, columns=['aligned_idx', 'ref_idx'])
        for ct in list(commonCT)+['X','Y']:
            out_df[ct] = out_df['aligned_idx'].map(aligned_df[ct])
        for ct in ['X','Y']:
            out_df[f'ref_{ct}'] = out_df['ref_idx'].map(ref_df[ct])
    else:
        print("No valid matches found")

    return out_df
from scipy.optimize import linprog
from scipy.spatial import cKDTree
import numpy as np
import pandas as pd

from scipy.spatial import cKDTree
import numpy as np
import pandas as pd
import numpy as np
from scipy.spatial import cKDTree

def find_knn_within_radius(aligned_df, ref_df, radius=25, knn=5):
    aligned_coords = aligned_df[['X', 'Y']].values
    ref_coords = ref_df[['X', 'Y']].values

    # Build a KD-Tree for the reference coordinates
    tree = cKDTree(ref_coords)

    # Initialize an empty array for k-nearest neighbors
    knn_pairs = np.empty((0, 2), int)

    # Find k-nearest neighbors within the radius for each aligned point
    for i, coord in enumerate(aligned_coords):
        # Query the tree for neighbors within the radius
        indices = tree.query_ball_point(coord, r=radius)
        
        # If there are more than k neighbors, sort by distance and select the top k
        if len(indices) > 0:
            distances = np.linalg.norm(ref_coords[indices] - coord, axis=1)
            sorted_indices = np.argsort(distances)
            selected_indices = np.array(indices)[sorted_indices[:min(knn, len(indices))]]  # Convert indices to a NumPy array
            
            # Add the pairs to the knn_pairs array
            knn_pairs = np.vstack((knn_pairs, np.column_stack((np.full(len(selected_indices), i), selected_indices))))

    print(f"Number of valid pairs after knn: {len(knn_pairs)}")
    unique_aligned_indices = np.unique(knn_pairs[:, 0])
    unique_ref_indices = np.unique(knn_pairs[:, 1])
    new_aligned_df = aligned_df.iloc[unique_aligned_indices].reset_index(drop=True)
    new_ref_df = ref_df.iloc[unique_ref_indices].reset_index(drop=True)
    # Update valid_pairs to reflect new indices
    index_map_aligned = {old: new for new, old in enumerate(unique_aligned_indices)}
    index_map_ref = {old: new for new, old in enumerate(unique_ref_indices)}
    new_valid_pairs = np.array([(index_map_aligned[i], index_map_ref[j]) for i, j in knn_pairs])
    return new_aligned_df, new_ref_df, new_valid_pairs

def preprocess_data(aligned_df, ref_df, radius):
    aligned_coords = aligned_df[['X', 'Y']].values
    ref_coords = ref_df[['X', 'Y']].values
    
    #distances = np.sqrt(((aligned_coords[:, np.newaxis, :] - ref_coords[np.newaxis, :, :]) ** 2).sum(axis=2))
    # Find pairs within the radius
    tree = cKDTree(ref_coords)
    indices  = tree.query_ball_point(aligned_coords, r=radius)
    valid_pairs = [(i, j) for i, neighbors in enumerate(indices) for j in neighbors]
    # Convert valid_pairs to a NumPy array
    valid_pairs = np.array(valid_pairs)
    #valid_pairs = np.argwhere(distances <= radius)
    print(f"Number of valid pairs before knn: {len(valid_pairs)}")

    # Remove values from aligned_df and ref_df if not in valid pairs
    unique_aligned_indices = np.unique(valid_pairs[:, 0])
    unique_ref_indices = np.unique(valid_pairs[:, 1])
    new_aligned_df = aligned_df.iloc[unique_aligned_indices].reset_index(drop=True)
    new_ref_df = ref_df.iloc[unique_ref_indices].reset_index(drop=True)
    
    # Update valid_pairs to reflect new indices
    index_map_aligned = {old: new for new, old in enumerate(unique_aligned_indices)}
    index_map_ref = {old: new for new, old in enumerate(unique_ref_indices)}
    new_valid_pairs = np.array([(index_map_aligned[i], index_map_ref[j]) for i, j in valid_pairs])
    
    print(f"Updated number of aligned points: {len(new_aligned_df)}")
    print(f"Updated number of reference points: {len(new_ref_df)}")
    print(f"Updated number of valid pairs: {len(new_valid_pairs)}")
    return new_aligned_df, new_ref_df, new_valid_pairs


def find_optimal_matches_lpv5(ref_df, aligned_df, commonCT, max_matches=3, radius=10, penalty_coeff=10, no_match_penalty=100, neighbor_coeff=-1):
    n_aligned = len(aligned_df)
    n_ref = len(ref_df)
    print(f"Number of aligned points: {n_aligned}")
    print(f"Number of reference points: {n_ref}")

    # Call the function
    aligned_df, ref_df, valid_pairs = preprocess_data(aligned_df, ref_df, radius)
    aligned_coords = aligned_df[['X', 'Y']].values
    ref_coords = ref_df[['X', 'Y']].values
    n_aligned, n_ref = len(aligned_df), len(ref_df)

    knnK = 8    
    print(f"Finding kNN for {knnK}...")

    aligned_knn = np.argsort(cdist(aligned_coords, aligned_coords), axis=1)[:, 1:knnK+1]
    ref_knn = np.argsort(cdist(ref_coords, ref_coords), axis=1)[:, 1:knnK+1]
    print(aligned_knn)
    print(ref_knn)
    neighbor_dict = {(i,j):[] for i,j in valid_pairs}
    # Convert valid_pairs to a set for faster lookup
    valid_pairs_set = set(map(tuple, valid_pairs))
    
    # Pre-compute sets of neighbors for each point
    aligned_neighbor_sets = {i: set(aligned_knn[i]) for i in range(len(aligned_knn))}
    ref_neighbor_sets = {j: set(ref_knn[j]) for j in range(len(ref_knn))}
    
    for i, j in valid_pairs:
        aligned_neighbors = aligned_neighbor_sets[i]
        ref_neighbors = ref_neighbor_sets[j]
        
        # Use set intersection for efficient filtering
        neighbor_pairs = valid_pairs_set.intersection(
            (a, r) for a in aligned_neighbors for r in ref_neighbors
        )
        neighbor_dict[(i, j)] = list(neighbor_pairs)
    # Objective function
    c = []
    dist_coeff = 0.005
    neighbor_pairs = set()
    neighbor_coeff = -1  # Negative coefficient for neighbor consistency, adjust as needed
    neighbor_k = 2  # Number of neighbors required to be 1 for the new variable to be 1

    # Initialize the new variable for neighbor consistency
    neighbor_consistency = np.zeros(len(valid_pairs))

    for idx, (i, j) in enumerate(valid_pairs):
        dist_ct = np.sum(np.abs(aligned_df.iloc[i][commonCT] - ref_df.iloc[j][commonCT]))
        dist_coords = np.abs(aligned_df.iloc[i]['X'] - ref_df.iloc[j]['X']) + \
                      np.abs(aligned_df.iloc[i]['Y'] - ref_df.iloc[j]['Y'])
        total_dist = dist_ct + dist_coeff * dist_coords


    # Add penalty terms to the objective function
    c.extend([penalty_coeff] * n_ref)
    c.extend([no_match_penalty] * n_aligned)
    # Add the new variable for neighbor consistency to the objective function
    c.extend([neighbor_coeff] * len(valid_pairs))

    n_vars = len(valid_pairs) + n_ref + n_aligned + len(valid_pairs)

    # Constraints
    A_ub = []
    b_ub = []

    # Constraint: each j in valid pairs must be used at most max_matches times
    for j in range(n_ref):
        constraint = np.zeros(n_vars)
        for idx, (_, jp) in enumerate(valid_pairs):
            if j == jp:
                constraint[idx] = 1
        A_ub.append(constraint)
        b_ub.append(max_matches)

    # Constraint: each j in valid pairs incurs a penalty if used more than once
    for j in range(n_ref):
        constraint = np.zeros(n_vars)
        for idx, (_, jp) in enumerate(valid_pairs):
            if j == jp:
                constraint[idx] = 1
        constraint[len(valid_pairs) + j] = -1  # Penalty term
        A_ub.append(constraint)
        b_ub.append(1)

    # Constraint: each i in valid pairs must be used at least once or incur a penalty
    for i in range(n_aligned):
        constraint = np.zeros(n_vars)
        for idx, (ip, _) in enumerate(valid_pairs):
            if i == ip:
                constraint[idx] = 1
        constraint[len(valid_pairs) + n_ref + i] = 1
        A_ub.append(-constraint)  # Note the negative sign
        b_ub.append(-1)

    # Constraint: neighbor consistency variable is 1 if at least k neighbors are 1
    for idx, (i, j) in enumerate(valid_pairs):
        constraint = np.zeros(n_vars)
        constraint[idx] = -1  # The pair itself
        for neighbor_idx, _ in enumerate(neighbor_dict[(i, j)]):
            constraint[neighbor_idx] = -1  # The neighbors
        constraint[len(valid_pairs) + n_ref + n_aligned + idx] = neighbor_k  # The new variable
        A_ub.append(constraint)
        b_ub.append(0)

    # The constraint "neighbor consistency variable is at most 1" is redundant
    # because it's already handled by the bounds definition:
    # [(0, 1) for _ in range(len(valid_pairs))]  # Bounds for the new variable
    # Therefore, we can remove this constraint.

    A_ub = np.array(A_ub)
    b_ub = np.array(b_ub)

    # Bounds
    bounds = [(0, 1) for _ in range(len(valid_pairs))] + \
             [(0, None) for _ in range(n_ref + n_aligned)] + \
             [(0, 1) for _ in range(len(valid_pairs))]  # Bounds for the new variable

    # Solve the linear programming problem
    result = linprog(c, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method='highs')

    # Extract the valid matches
    valid_matches = []
    if result.success:
        x = result.x[:len(valid_pairs)]
        print(f"Number of non-zero elements in x: {np.count_nonzero(x > 0.5)}")
        for idx, (i, j) in enumerate(valid_pairs):
            if x[idx] > 0.5:  # Consider as matched if the value is >0.5
                valid_matches.append((i, j))
        out_df = pd.DataFrame(valid_matches, columns=['aligned_idx', 'ref_idx'])
        for ct in list(commonCT)+['X','Y']:
            out_df[ct] = out_df['aligned_idx'].map(aligned_df[ct])
        for ct in ['X','Y']:
            out_df[f'ref_{ct}'] = out_df['ref_idx'].map(ref_df[ct])
    else:
        print("No valid matches found")
        out_df = pd.DataFrame()

    return out_df