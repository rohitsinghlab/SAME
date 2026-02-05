"""
Synthetic 4-Quadrant Benchmark for SAME
========================================
Addresses reviewer comment about:
- Diffeomorphic vs space-tearing distortions
- Missing cells between consecutive tissues
- Different cell numbers/entities

Quadrants (all 10x10 = 100 points, 3 classes total but each quadrant uses 2):
- Top-Left: c1 + c3, with c3 missing in query + jitter on c1
- Top-Right: c1 + c2, with ~30% class label flips
- Bottom-Right: c1 + c2, with space tearing distortion
- Bottom-Left: c1 (background) + c2 in 2 rings → 1 ellipse
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.spatial import Delaunay
from sklearn.gaussian_process.kernels import RBF
from scipy.stats import multivariate_normal as mvn

# Set seeds for reproducibility
np.random.seed(2024)

# Class names and colors for visualization (3 classes)
CLASS_NAMES = ['c1', 'c2', 'c3']
CLASS_COLORS = {'c1': '#FF692A', 'c2': '#9810FA', 'c3': '#31C950'}  # blue, green, red


def filter_triangles_by_min_angle(points, simplices, min_angle_deg=15):
    """
    Filter Delaunay triangles to remove thin/degenerate ones.
    
    Parameters
    ----------
    points : ndarray, shape (n, 2)
        Point coordinates
    simplices : ndarray, shape (m, 3)
        Triangle vertex indices from Delaunay
    min_angle_deg : float
        Minimum angle in degrees. Triangles with any angle < this are removed.
    
    Returns
    -------
    filtered_simplices : ndarray
        Triangles that pass the angle filter
    """
    min_angle_rad = np.deg2rad(min_angle_deg)
    
    def triangle_angles(p1, p2, p3):
        """Compute all three angles of a triangle."""
        # Vectors
        v1 = p2 - p1
        v2 = p3 - p1
        v3 = p3 - p2
        
        # Edge lengths
        a = np.linalg.norm(v3)  # opposite to p1
        b = np.linalg.norm(p3 - p1)  # opposite to p2
        c = np.linalg.norm(p2 - p1)  # opposite to p3
        
        # Avoid division by zero
        if a < 1e-10 or b < 1e-10 or c < 1e-10:
            return [0, 0, 0]
        
        # Angles using law of cosines
        angle1 = np.arccos(np.clip((b**2 + c**2 - a**2) / (2*b*c), -1, 1))
        angle2 = np.arccos(np.clip((a**2 + c**2 - b**2) / (2*a*c), -1, 1))
        angle3 = np.pi - angle1 - angle2
        
        return [angle1, angle2, angle3]
    
    keep_mask = []
    for tri in simplices:
        p1, p2, p3 = points[tri[0]], points[tri[1]], points[tri[2]]
        angles = triangle_angles(p1, p2, p3)
        min_angle = min(angles)
        keep_mask.append(min_angle >= min_angle_rad)
    
    return simplices[np.array(keep_mask)]


def compute_filtered_delaunay(points, min_angle_deg=15):
    """
    Compute Delaunay triangulation and filter out thin triangles.
    
    Returns filtered simplices array.
    """
    if len(points) < 3:
        return np.array([]).reshape(0, 3)
    
    try:
        tri = Delaunay(points)
        return filter_triangles_by_min_angle(points, tri.simplices, min_angle_deg)
    except:
        return np.array([]).reshape(0, 3)


def create_grid_in_region(x_range, y_range, n_per_side=10, jitter=0.1):
    """Create a jittered grid of points in a rectangular region."""
    x = np.linspace(x_range[0], x_range[1], n_per_side)
    y = np.linspace(y_range[0], y_range[1], n_per_side)
    X, Y = np.meshgrid(x, y)
    points = np.column_stack([X.ravel(), Y.ravel()])
    points += np.random.normal(0, jitter, points.shape)
    return points


def create_ring(center, radius, n_points, width=0.3):
    """Create points arranged in a ring/annulus."""
    angles = np.linspace(0, 2*np.pi, n_points, endpoint=False)
    r = radius + np.random.uniform(-width/2, width/2, n_points)
    x = center[0] + r * np.cos(angles)
    y = center[1] + r * np.sin(angles)
    return np.column_stack([x, y])


def assign_classes_checkerboard(points, n_classes=2, classes_to_use=None):
    """Assign classes in a checkerboard-like pattern (alternates in both x and y).
    
    Parameters:
    -----------
    points : array-like
        (N, 2) array of point coordinates
    n_classes : int
        Number of classes (used if classes_to_use is None)
    classes_to_use : list, optional
        Specific class indices to use (e.g., [0, 2] for c1 and c3)
    """
    grid_scale = 0.6  # Adjust based on grid spacing
    hash_vals = (np.floor(points[:, 0] / grid_scale) + 
                 np.floor(points[:, 1] / grid_scale)).astype(int)
    
    if classes_to_use is not None:
        n_classes = len(classes_to_use)
        class_indices = hash_vals % n_classes
        return np.array([classes_to_use[i] for i in class_indices])
    else:
        return hash_vals % n_classes


def assign_classes_alternating(n_points, classes_to_use):
    """Assign classes in strict alternating pattern (linear, not checkerboard)."""
    n_classes = len(classes_to_use)
    return np.array([classes_to_use[i % n_classes] for i in range(n_points)])


def add_gp_noise(points, length_scale=2.0, variance=0.05):
    """Add Gaussian Process noise for smooth deformation."""
    n = len(points)
    K = variance * RBF(length_scale=length_scale)(points)
    K += 1e-6 * np.eye(n)  # Numerical stability
    noise = mvn.rvs(mean=np.zeros(n), cov=K, size=2).T
    return noise


def create_one_hot(classes, n_classes=3):
    """Create one-hot encoding with some noise for soft labels."""
    n = len(classes)
    one_hot = np.zeros((n, n_classes))
    for i, c in enumerate(classes):
        one_hot[i, c] = 0.85 + np.random.uniform(0, 0.1)
        other_classes = [j for j in range(n_classes) if j != c]
        remaining = 1 - one_hot[i, c]
        for j in other_classes:
            one_hot[i, j] = remaining / len(other_classes) + np.random.uniform(-0.02, 0.02)
        one_hot[i] = np.clip(one_hot[i], 0, 1)
        one_hot[i] = one_hot[i] / one_hot[i].sum()
    return one_hot * 100


def create_noisy_one_hot(classes, n_classes=3):
    """Create noisy/uncertain one-hot encoding (~0.33 + small noise for each class)."""
    n = len(classes)
    one_hot = np.zeros((n, n_classes))
    for i, c in enumerate(classes):
        one_hot[i, c] = 0.33 + np.random.uniform(0.05, 0.15)
        other_classes = [j for j in range(n_classes) if j != c]
        remaining = 1 - one_hot[i, c]
        for j in other_classes:
            one_hot[i, j] = remaining / len(other_classes) + np.random.uniform(-0.02, 0.02)
        one_hot[i] = np.clip(one_hot[i], 0, 1)
        one_hot[i] = one_hot[i] / one_hot[i].sum()
    return one_hot * 100


# =============================================================================
# QUADRANT 1: TOP-LEFT - Missing Class Scenario
# =============================================================================
def create_quadrant_topleft():
    """
    Top-Left: Missing class scenario
    - Ref has c1 and c3 alternating (10x10 grid)
    - Query is missing ALL c3 cells
    - c1 cells in query have extra jitter/noise
    """
    print("Creating Top-Left quadrant (c1+c3, c3 missing in query)...")
    
    x_range = (1, 6)
    y_range = (7.25, 12.25)
    
    # Reference: 10x10 grid with checkerboard c1/c3 pattern (alternates in both x and y)
    ref_points = create_grid_in_region(x_range, y_range, n_per_side=10)
    ref_classes = assign_classes_checkerboard(ref_points, 3, classes_to_use=[0, 1, 2])  # c1 and c3
    
    # Query: apply GP deformation first
    query_points = ref_points.copy()
    query_points += add_gp_noise(query_points, length_scale=2.5, variance=0.05)
    
    # Remove c3 cells from query (keep only c1)
    mask_keep = ref_classes != 2  # Keep only c1
    query_points = query_points[mask_keep]
    query_classes = ref_classes[mask_keep]
    
    # Add extra jitter/noise to c1 cells in query
    extra_jitter = np.random.normal(0, 0.1, query_points.shape)
    query_points += extra_jitter
    
    # Ground truth: maps query idx to ref idx
    ref_idx_for_query = np.where(mask_keep)[0]
    
    return {
        'ref_points': ref_points,
        'ref_classes': ref_classes,
        'query_points': query_points,
        'query_classes': query_classes,
        'ground_truth_ref_idx': ref_idx_for_query,
        'description': 'Missing class (c3 removed) + jitter on c1'
    }


# =============================================================================
# QUADRANT 2: TOP-RIGHT - Class Label Flips
# =============================================================================
def create_quadrant_topright():
    """
    Top-Right: Noisy annotations / soft uncertainty
    - 10x10 grid with 3 classes (c1, c2, c3)
    - GP noise only (smooth deformation, no extra jitter)
    - Query has noisy/uncertain probabilities (~0.33 + noise for each class)
    """
    print("Creating Top-Right quadrant (3 classes, GP only, noisy annotations/probabilities)...")
    
    x_range = (7.25, 12.25)
    y_range = (7.25, 12.25)
    
    # Reference: 10x10 grid with 3-class checkerboard pattern
    ref_points = create_grid_in_region(x_range, y_range, n_per_side=10)
    ref_classes = assign_classes_checkerboard(ref_points, n_classes=3)
    
    # Query: GP noise only (smooth deformation)
    query_points = ref_points.copy()
    query_points += add_gp_noise(query_points, length_scale=2.5, variance=0.05)

    # Labels preserved (no flipping)
    query_classes = ref_classes.copy()
    
    return {
        'ref_points': ref_points,
        'ref_classes': ref_classes,
        'query_points': query_points,
        'query_classes': query_classes,
        'ground_truth_ref_idx': np.arange(len(ref_points)),
        'description': 'GP only + noisy probabilities',
        'use_noisy_probs': True  # Flag for special probability handling
    }


# =============================================================================
# QUADRANT 3: BOTTOM-RIGHT - Space Tearing
# =============================================================================
def create_quadrant_bottomright():
    """
    Bottom-Right: Space fold (space-tearing) distortion
    - 10x10 grid with 3 classes (c1, c2, c3)
    - GP noise + extra jitter
    - Local point swaps that cause triangle flips
    - Non-diffeomorphic transformation
    """
    print("Creating Bottom-Right quadrant (3 classes, GP+jitter, space fold/tearing)...")
    
    from scipy.spatial.distance import cdist
    
    x_range = (7.25, 12.25)
    y_range = (1, 6)
    
    # Reference: 10x10 grid with 3-class checkerboard
    ref_points = create_grid_in_region(x_range, y_range, n_per_side=10)
    ref_classes = assign_classes_checkerboard(ref_points, n_classes=3)
    
    # Query: GP noise + extra jitter
    query_points = ref_points.copy()
    query_points += add_gp_noise(query_points, length_scale=2.0, variance=0.05)
    query_points += np.random.normal(0, 0.05, query_points.shape)  # Extra jitter
    
    # Apply SHEAR to part of the region
    center = np.array([8.5, 2.5])
    mask_shear = np.linalg.norm(ref_points - center, axis=1) < 2.5
    shear_matrix = np.array([[1, 0.35], [0, 1]])
    query_points[mask_shear] = ((query_points[mask_shear] - center) @ shear_matrix.T) + center
    
    # Swap 3 pairs of adjacent points (guaranteed triangle flips!)
    dists = cdist(ref_points, ref_points)
    np.fill_diagonal(dists, np.inf)
    
    swapped_pairs = []
    used_indices = set()
    ground_truth = np.arange(len(ref_points))  # Start with identity
    
    for _ in range(3):  # Swap 3 pairs
        min_dist = np.inf
        best_pair = None
        for i in range(len(ref_points)):
            if i in used_indices:
                continue
            for j in range(i+1, len(ref_points)):
                if j in used_indices:
                    continue
                if dists[i, j] < min_dist:
                    min_dist = dists[i, j]
                    best_pair = (i, j)
        
        if best_pair:
            i, j = best_pair
            # Swap positions in query
            query_points[i], query_points[j] = query_points[j].copy(), query_points[i].copy()
            # Update ground truth
            ground_truth[i], ground_truth[j] = j, i
            swapped_pairs.append(best_pair)
            used_indices.add(i)
            used_indices.add(j)
            dists[i, :] = np.inf
            dists[:, i] = np.inf
            dists[j, :] = np.inf
            dists[:, j] = np.inf
    
    query_classes = ref_classes.copy()
    
    return {
        'ref_points': ref_points,
        'ref_classes': ref_classes,
        'query_points': query_points,
        'query_classes': query_classes,
        'ground_truth_ref_idx': ground_truth,
        'description': 'Space fold (point swaps + shear)',
        'swapped_pairs': swapped_pairs
    }


# =============================================================================
# QUADRANT 4: BOTTOM-LEFT - Topological Merger
# =============================================================================
def create_quadrant_bottomleft():
    """
    Bottom-Left: Topological split (opposite of merger)
    - REF: background (c1+c3 alternating) + ONE ellipse (c2)
    - QUERY: background (c1+c3 alternating) + TWO separate rings (c2)
    - Same total number of points
    """
    print("Creating Bottom-Left quadrant (split: 1 ellipse → 2 rings)...")
    
    x_range = (1, 6)
    y_range = (1, 6)
    
    # Create background grid with checkerboard c1/c3 pattern (alternates in both x and y)
    grid_points = create_grid_in_region(x_range, y_range, n_per_side=10, jitter=0.05)
    grid_classes = assign_classes_checkerboard(grid_points, classes_to_use=[0, 2])  # c1 and c3
    
    # Ellipse and ring parameters
    ellipse_center = np.array([3.5, 3.5])
    ellipse_a = 1.5  # semi-major axis (horizontal)
    ellipse_b = 0.8  # semi-minor axis (vertical)
    
    # Rings slightly closer together
    ring1_center = np.array([2.1, 3.5])
    ring2_center = np.array([4.1, 3.5])
    ring_radius = 0.6
    n_per_ring = 10
    n_ellipse = 2 * n_per_ring  # Same number of c2 points
    
    # Compute distances for filtering
    dist_to_ellipse = np.sqrt(((grid_points[:, 0] - ellipse_center[0]) / ellipse_a)**2 + 
                               ((grid_points[:, 1] - ellipse_center[1]) / ellipse_b)**2)
    dist_to_ring1 = np.linalg.norm(grid_points - ring1_center, axis=1)
    dist_to_ring2 = np.linalg.norm(grid_points - ring2_center, axis=1)
    
    # REF background: exclude points inside ellipse
    ref_bg_mask = dist_to_ellipse > 1.0
    ref_background = grid_points[ref_bg_mask]
    ref_bg_classes = grid_classes[ref_bg_mask]  # Keep alternating c1/c3 pattern
    
    # QUERY background: exclude points inside rings (but keep points BETWEEN rings)
    query_bg_mask = (dist_to_ring1 > ring_radius + 0.1) & (dist_to_ring2 > ring_radius + 0.1)
    query_background = grid_points[query_bg_mask]
    query_bg_classes = grid_classes[query_bg_mask]  # Keep alternating c1/c3 pattern
    
    # === REF: ref_background + ONE ellipse ===
    n_ref_bg = len(ref_background)
    
    ellipse_angles = np.linspace(0, 2*np.pi, n_ellipse, endpoint=False)
    ellipse_points = np.column_stack([
        ellipse_center[0] + ellipse_a * np.cos(ellipse_angles),
        ellipse_center[1] + ellipse_b * np.sin(ellipse_angles)
    ])
    ellipse_points += np.random.normal(0, 0.03, ellipse_points.shape)
    
    ref_points = np.vstack([ref_background, ellipse_points])
    
    # Assign classes: background=c1/c3 alternating, ellipse=c2(1)
    ref_classes = np.concatenate([ref_bg_classes, np.ones(n_ellipse, dtype=int)])
    
    # === QUERY: query_background + TWO rings ===
    n_query_bg = len(query_background)
    
    # Apply GP noise to query background
    query_bg_noisy = query_background.copy()
    if len(query_bg_noisy) > 0:
        query_bg_noisy += add_gp_noise(query_bg_noisy, length_scale=2, variance=0.05)
    
    # Create two rings
    ring1_angles = np.linspace(0, 2*np.pi, n_per_ring, endpoint=False)
    ring1_points = np.column_stack([
        ring1_center[0] + ring_radius * np.cos(ring1_angles),
        ring1_center[1] + ring_radius * np.sin(ring1_angles)
    ])
    ring1_points += np.random.normal(0, 0.03, ring1_points.shape)
    
    ring2_angles = np.linspace(0, 2*np.pi, n_per_ring, endpoint=False)
    ring2_points = np.column_stack([
        ring2_center[0] + ring_radius * np.cos(ring2_angles),
        ring2_center[1] + ring_radius * np.sin(ring2_angles)
    ])
    ring2_points += np.random.normal(0, 0.03, ring2_points.shape)
    
    query_points = np.vstack([query_bg_noisy, ring1_points, ring2_points])
    
    # Assign classes for query: background=c1/c3 alternating, rings=c2(1)
    query_classes = np.concatenate([query_bg_classes, np.ones(2 * n_per_ring, dtype=int)])
    
    # Ground truth: only background points that exist in BOTH ref and query can be matched
    # For c2 points (ellipse/rings), there's no 1:1 correspondence (topology change)
    # Create ground truth for query points based on nearest ref point of same class
    ground_truth = np.full(len(query_points), -1, dtype=int)  # -1 means no match
    
    # Background points: find matching ref background points by original grid index
    # (This is approximate since masks differ)
    for i in range(n_query_bg):
        ground_truth[i] = i if i < n_ref_bg else -1
    
    return {
        'ref_points': ref_points,
        'ref_classes': ref_classes,
        'query_points': query_points,
        'query_classes': query_classes,
        'ground_truth_ref_idx': ground_truth,
        'description': 'Topological split (1 ellipse → 2 rings)',
        'n_ref_background': n_ref_bg,
        'n_query_background': n_query_bg,
        'ellipse_center': ellipse_center,
        'ellipse_a': ellipse_a,
        'ellipse_b': ellipse_b,
        'ring1_center': ring1_center,
        'ring2_center': ring2_center,
        'ring_radius': ring_radius
    }


# =============================================================================
# SIMULATED GENE EXPRESSION
# =============================================================================
def generate_expression(classes, n_genes=100):
    """
    Generate simulated gene expression data based on class labels.
    Each class has a distinct mean expression profile.
    Values are positive (like real count data).
    
    Parameters:
    -----------
    classes : array-like
        Class labels (0, 1, 2 for c1, c2, c3)
    n_genes : int
        Number of genes to simulate
    
    Returns:
    --------
    expression : np.ndarray
        Shape (n_samples, n_genes) expression matrix (all positive values)
    gene_names : list
        List of gene names
    """
    n_samples = len(classes)
    
    # Base expression level (all genes have some baseline expression)
    baseline = 2.0
    
    # Create distinct mean profiles for each class
    # c1 (class 0): high expression in genes 0-33
    # c2 (class 1): high expression in genes 33-66
    # c3 (class 2): high expression in genes 66-100
    cluster_means = {
        0: np.concatenate([np.random.uniform(8, 12, 34),   # high
                          np.random.uniform(1, 3, 33),     # low
                          np.random.uniform(1, 3, 33)]),   # low
        1: np.concatenate([np.random.uniform(1, 3, 34),    # low
                          np.random.uniform(8, 12, 33),    # high
                          np.random.uniform(1, 3, 33)]),   # low
        2: np.concatenate([np.random.uniform(1, 3, 34),    # low
                          np.random.uniform(1, 3, 33),     # low
                          np.random.uniform(8, 12, 33)])   # high
    }
    
    # Generate expression for each sample (log-normal-like, always positive)
    expression = np.zeros((n_samples, n_genes))
    for i in range(n_samples):
        c = classes[i]
        # Add noise proportional to mean (coefficient of variation ~0.2)
        noise = np.random.normal(0, 0.2 * cluster_means[c])
        expression[i] = np.maximum(cluster_means[c] + noise, 0.1)  # Ensure positive
    
    # Gene names
    gene_names = [f'gene_{i}' for i in range(n_genes)]
    
    return expression, gene_names


# =============================================================================
# COMBINE ALL QUADRANTS
# =============================================================================
def create_full_benchmark():
    """Create the complete 4-quadrant benchmark dataset."""
    
    quadrants = {
        'top_left': create_quadrant_topleft(),
        'top_right': create_quadrant_topright(),
        'bottom_right': create_quadrant_bottomright(),
        'bottom_left': create_quadrant_bottomleft()
    }
    
    # Track offsets for global indices
    ref_offset = 0
    query_offset = 0
    
    ref_points_list = []
    ref_classes_list = []
    query_points_list = []
    query_classes_list = []
    quadrant_labels_ref = []
    quadrant_labels_query = []
    ground_truth_pairs = []
    
    for qname in ['top_left', 'top_right', 'bottom_right', 'bottom_left']:
        q = quadrants[qname]
        
        # Store ground truth with global indices
        for query_local_idx, ref_local_idx in enumerate(q['ground_truth_ref_idx']):
            ground_truth_pairs.append((
                query_offset + query_local_idx,
                ref_offset + ref_local_idx
            ))
        
        ref_points_list.append(q['ref_points'])
        ref_classes_list.append(q['ref_classes'])
        query_points_list.append(q['query_points'])
        query_classes_list.append(q['query_classes'])
        quadrant_labels_ref.extend([qname] * len(q['ref_points']))
        quadrant_labels_query.extend([qname] * len(q['query_points']))
        
        ref_offset += len(q['ref_points'])
        query_offset += len(q['query_points'])
    
    ref_points = np.vstack(ref_points_list)
    ref_classes = np.concatenate(ref_classes_list)
    query_points = np.vstack(query_points_list)
    query_classes = np.concatenate(query_classes_list)
    
    # Create one-hot encodings (3 classes: c1, c2, c3)
    ref_onehot = create_one_hot(ref_classes, n_classes=3)
    
    # For query: use noisy probabilities for top_right quadrant
    query_onehot = np.zeros((len(query_classes), 3))
    query_idx = 0
    for qname in ['top_left', 'top_right', 'bottom_right', 'bottom_left']:
        q = quadrants[qname]
        n_query = len(q['query_points'])
        if q.get('use_noisy_probs', False):
            # Use noisy/uncertain probabilities (~0.33 each)
            query_onehot[query_idx:query_idx + n_query] = create_noisy_one_hot(
                q['query_classes'], n_classes=3
            )
        else:
            # Use normal confident probabilities
            query_onehot[query_idx:query_idx + n_query] = create_one_hot(
                q['query_classes'], n_classes=3
            )
        query_idx += n_query
    
    # Create DataFrames
    ref_df = pd.DataFrame({
        'X': ref_points[:, 0],
        'Y': ref_points[:, 1],
        'cell_type': [CLASS_NAMES[c] for c in ref_classes],
        'c1': ref_onehot[:, 0],
        'c2': ref_onehot[:, 1],
        'c3': ref_onehot[:, 2],
        'quadrant': quadrant_labels_ref,
        'cell_idx': np.arange(len(ref_points))
    })
    
    query_df = pd.DataFrame({
        'X': query_points[:, 0],
        'Y': query_points[:, 1],
        'cell_type': [CLASS_NAMES[c] for c in query_classes],
        'c1': query_onehot[:, 0],
        'c2': query_onehot[:, 1],
        'c3': query_onehot[:, 2],
        'quadrant': quadrant_labels_query,
        'cell_idx': np.arange(len(query_points))
    })
    
    # Ground truth DataFrame
    ground_truth_df = pd.DataFrame(ground_truth_pairs, columns=['query_idx', 'ref_idx'])
    
    # Generate simulated gene expression for ref and query
    ref_expr, gene_names = generate_expression(ref_classes, n_genes=100)
    query_expr, _ = generate_expression(query_classes, n_genes=100)
    
    # Create expression DataFrames with cell_idx as index
    ref_expression_df = pd.DataFrame(ref_expr, columns=gene_names)
    ref_expression_df['cell_idx'] = ref_df['cell_idx'].values
    ref_expression_df = ref_expression_df.set_index('cell_idx')
    
    query_expression_df = pd.DataFrame(query_expr, columns=gene_names)
    query_expression_df['cell_idx'] = query_df['cell_idx'].values
    query_expression_df = query_expression_df.set_index('cell_idx')
    
    # Return as dictionary
    expression_dict = {
        'ref': ref_expression_df,
        'query': query_expression_df
    }
    
    return ref_df, query_df, quadrants, ground_truth_df, expression_dict


def visualize_benchmark(ref_df, query_df, quadrants):
    """Create 4-panel visualization: ref, query, overlay, nearest neighbor class match."""
    from scipy.spatial import cKDTree
    from matplotlib.colors import to_rgb
    
    fig, axes = plt.subplots(1, 4, figsize=(18, 4.5))
    
    # Plot 1: Reference (full)
    ax = axes[0]
    for ct, color in CLASS_COLORS.items():
        mask = ref_df['cell_type'] == ct
        if mask.sum() > 0:
            ax.scatter(ref_df.loc[mask, 'X'], ref_df.loc[mask, 'Y'], 
                       c=color, label=ct, s=40, alpha=0.8, edgecolors='black', linewidth=0.3)
    ax.set_title(r'$\mathbf{b.}$ Template', fontsize=14, loc='left')
    ax.legend()
    ax.set_aspect('equal')
    # ax.axhline(6.625, color='gray', linestyle='--', alpha=0.5)
    # ax.axvline(6.625, color='gray', linestyle='--', alpha=0.5)
    
    # Plot 2: Query (full)
    ax = axes[1]
    # Probability color-mixing: each point's RGB = p(c1)*orange + p(c2)*purple + p(c3)*green.
    prob_cols = [ct for ct in CLASS_COLORS.keys() if ct in query_df.columns]
    if set(prob_cols) >= {"c1", "c2", "c3"}:
        probs = query_df[["c1", "c2", "c3"]].astype(float).to_numpy(copy=True)
        row_sums = probs.sum(axis=1, keepdims=True)
        row_sums = np.where(row_sums <= 0, 1.0, row_sums)
        probs = probs / row_sums

        base_rgbs = np.array([to_rgb(CLASS_COLORS["c1"]),
                              to_rgb(CLASS_COLORS["c2"]),
                              to_rgb(CLASS_COLORS["c3"])], dtype=float)
        mixed_rgb = probs @ base_rgbs  # (N,3)

        ax.scatter(
            query_df["X"].values,
            query_df["Y"].values,
            c=mixed_rgb,
            s=60,
            marker="P",
            edgecolors="black",
            linewidth=0.3,
        )
        ax.set_title(r'$\mathbf{b.}$ Query (probability color-mix)', fontsize=14, loc='left')
        # Legend: show the base colors (mixture components)
        for ct in ["c1", "c2", "c3"]:
            ax.scatter([], [], c=CLASS_COLORS[ct], s=60, marker="P",
                       edgecolors="black", linewidth=0.3, label=ct)
        ax.legend()
    else:
        # Fallback: discrete colors by hard cell_type
        for ct, color in CLASS_COLORS.items():
            mask = query_df['cell_type'] == ct
            if mask.sum() > 0:
                ax.scatter(query_df.loc[mask, 'X'], query_df.loc[mask, 'Y'], 
                           c=color, label=ct, s=60, alpha=0.8, marker='P',
                           edgecolors='black', linewidth=0.3)
        ax.set_title(r'$\mathbf{b.}$ Query', fontsize=14, loc='left')
        ax.legend()
    ax.set_aspect('equal')
    # ax.axhline(6.625, color='gray', linestyle='--', alpha=0.5)
    # ax.axvline(6.625, color='gray', linestyle='--', alpha=0.5)
    
    # Plot 3: Overlay
    ax = axes[2]
    ax.scatter(ref_df['X'], ref_df['Y'], c='blue', s=40, alpha=0.5, label='Ref')
    ax.scatter(query_df['X'], query_df['Y'], c='red', s=60, alpha=0.5, marker='P', label='Query')
    ax.set_title(r'$\mathbf{c.}$ Overlay', fontsize=14, loc='left')
    ax.legend(labels=['Template', 'Query'])
    ax.set_aspect('equal')
    # ax.axhline(6.625, color='gray', linestyle='--', alpha=0.5)
    # ax.axvline(6.625, color='gray', linestyle='--', alpha=0.5)
    
    # Plot 4: Nearest neighbor class match
    # For each query point, find nearest ref point and check if classes match
    ax = axes[3]
    
    ref_coords = ref_df[['X', 'Y']].values
    query_coords = query_df[['X', 'Y']].values
    
    # Build KD-tree on reference points
    tree = cKDTree(ref_coords)
    
    # Find nearest neighbor for each query point
    distances, nearest_idx = tree.query(query_coords, k=1)
    
    # Check if classes match
    query_classes = query_df['cell_type'].values
    nearest_ref_classes = ref_df['cell_type'].values[nearest_idx]
    class_match = query_classes == nearest_ref_classes
    
    # Calculate match rate
    match_rate = 100 * class_match.sum() / len(class_match)
    
    # Plot: green for match, red for mismatch
    match_mask = class_match
    mismatch_mask = ~class_match
    
    ax.scatter(query_df.loc[match_mask, 'X'], query_df.loc[match_mask, 'Y'],
               c='green', s=40, alpha=0.7, label=f'Match ({match_mask.sum()})', 
               edgecolors='darkgreen', linewidth=0.5)
    ax.scatter(query_df.loc[mismatch_mask, 'X'], query_df.loc[mismatch_mask, 'Y'],
               c='red', s=40, alpha=0.7, marker=f'x', label=f'Mismatch ({mismatch_mask.sum()})',
               linewidths=1.5)
    ax.set_title(r'$\mathbf{d.}$ NN Class Match (' + f'{match_rate:.1f}%)', fontsize=14, loc='left')

    ax.legend()
    ax.set_aspect('equal')
    # Remove top and right spines and set tick labels for all subplots
    for ax in axes.flatten():
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        # Set tick labels to only show 0, 5, 10
        ax.set_xticks([0, 5, 10])
        ax.set_yticks([0, 5, 10])
        #ax.axhline(6.75, color='gray', linestyle='--', alpha=0.5)
        #ax.axvline(6.75, color='gray', linestyle='--', alpha=0.5)
    plt.tight_layout()
    return fig


def visualize_benchmark_v2(
    ref_df,
    query_df,
    quadrants,
    *,
    alpha_min=0.5,
    arrow_stride=3,
):
    """
    Create a 4-panel visualization:

    - a. Template: reference points colored by class
    - b. Query: query points colored by predicted class, with transparency indicating
      per-cell confidence (probability of the assigned class from the c1/c2/c3 columns)
    - c. Template→Query deformation: arrows from matched template points to query points
      using per-quadrant ground-truth mappings
    - d. NN class match: nearest-neighbor class agreement (query vs. nearest template)

    Notes
    -----
    - This function expects `query_df` to contain probability columns named after the
      class labels (e.g., "c1", "c2", "c3") in the same scale as generated by
      `create_one_hot` / `create_noisy_one_hot` (typically 0–100).
    - For quadrants with missing / non-1:1 correspondences, arrows are drawn only for
      query points with valid `ground_truth_ref_idx` entries (>= 0).
    """
    from matplotlib.colors import to_rgba
    from scipy.spatial import cKDTree

    def _annotate_quadrants(ax, df):
        """Annotate scenario labels at each quadrant centroid."""
        label_map = {
            "top_left": "Missing class",
            "top_right": "Low confidence",
            "bottom_left": "Split",
            "bottom_right": "Space fold",
        }
        if "quadrant" not in df.columns:
            return
        for qname, label in label_map.items():
            sub = df[df["quadrant"] == qname]
            if len(sub) == 0:
                continue
            cx = float(sub["X"].mean())
            cy = float(sub["Y"].mean())
            ax.text(
                cx,
                cy,
                label,
                ha="center",
                va="center",
                fontsize=10,
                weight="bold",
                bbox=dict(boxstyle="round,pad=0.25", facecolor="white", alpha=0.75, edgecolor="none"),
                zorder=10,
            )

    fig, axes = plt.subplots(1, 4, figsize=(18, 4.5))

    # -------------------------
    # Panel a: Template (ref)
    # -------------------------
    ax = axes[0]
    for ct, color in CLASS_COLORS.items():
        mask = ref_df["cell_type"] == ct
        if mask.sum() > 0:
            ax.scatter(
                ref_df.loc[mask, "X"],
                ref_df.loc[mask, "Y"],
                c=color,
                label=ct,
                s=40,
                alpha=0.85,
                edgecolors="black",
                linewidth=0.3,
            )
    ax.set_title(r"$\mathbf{a.}$ Template", fontsize=14, loc="left")
    ax.legend()
    ax.set_aspect("equal")
    _annotate_quadrants(ax, ref_df)

    # -------------------------
    # Panel b: Query (confidence→alpha)
    # -------------------------
    ax = axes[1]
    n_query = len(query_df)
    # confidence is probability of assigned class (c1/c2/c3 columns), normalized to 0..1
    conf = np.full(n_query, np.nan, dtype=float)
    for ct in CLASS_COLORS.keys():
        if ct not in query_df.columns:
            continue
        m = query_df["cell_type"] == ct
        if m.sum() > 0:
            conf[m.values] = query_df.loc[m, ct].astype(float).values
    # fall back to fully opaque if probabilities missing for a row
    conf = np.where(np.isfinite(conf), conf, 100.0)
    alpha = np.clip(conf / 100.0, alpha_min, 1.0)

    # Build per-point RGBA colors
    colors = np.zeros((n_query, 4), dtype=float)
    for ct, base_hex in CLASS_COLORS.items():
        m = (query_df["cell_type"] == ct).values
        if not m.any():
            continue
        base_rgba = np.array(to_rgba(base_hex))
        colors[m, 0:3] = base_rgba[0:3]
        colors[m, 3] = alpha[m]

    ax.scatter(
        query_df["X"].values,
        query_df["Y"].values,
        c=colors,
        s=60,
        marker="P",
        edgecolors="black",
        linewidth=0.3,
    )
    ax.set_title(r"$\mathbf{b.}$ Query (alpha = class probability)", fontsize=14, loc="left")
    # Legend (fixed alpha)
    for ct, color in CLASS_COLORS.items():
        if (query_df["cell_type"] == ct).sum() > 0:
            ax.scatter([], [], c=color, s=60, marker="P", edgecolors="black", linewidth=0.3, label=ct)
    ax.legend()
    ax.set_aspect("equal")
    _annotate_quadrants(ax, query_df)

    # -------------------------
    # Panel c: Template→Query arrows
    # -------------------------
    ax = axes[2]
    # light background points for context
    ax.scatter(ref_df["X"], ref_df["Y"], c="black", s=12, alpha=0.15, label="Template", linewidths=0)
    ax.scatter(query_df["X"], query_df["Y"], c="black", s=12, alpha=0.15, label="Query", linewidths=0, marker="P")

    x0_all = []
    y0_all = []
    dx_all = []
    dy_all = []
    for qname in ["top_left", "top_right", "bottom_right", "bottom_left"]:
        q = quadrants.get(qname)
        if q is None:
            continue
        ref_pts = q.get("ref_points")
        query_pts = q.get("query_points")
        gt = q.get("ground_truth_ref_idx")
        if ref_pts is None or query_pts is None or gt is None:
            continue

        gt = np.asarray(gt)
        if len(query_pts) == 0 or len(ref_pts) == 0 or gt.size == 0:
            continue

        valid = (gt >= 0) & (gt < len(ref_pts))
        if not np.any(valid):
            continue

        q_idx = np.where(valid)[0]
        if arrow_stride and arrow_stride > 1:
            q_idx = q_idx[::arrow_stride]

        r_idx = gt[q_idx].astype(int)
        origins = ref_pts[r_idx]
        targets = query_pts[q_idx]
        deltas = targets - origins

        x0_all.append(origins[:, 0])
        y0_all.append(origins[:, 1])
        dx_all.append(deltas[:, 0])
        dy_all.append(deltas[:, 1])

    if len(x0_all) > 0:
        x0 = np.concatenate(x0_all)
        y0 = np.concatenate(y0_all)
        dx = np.concatenate(dx_all)
        dy = np.concatenate(dy_all)
        ax.quiver(
            x0,
            y0,
            dx,
            dy,
            angles="xy",
            scale_units="xy",
            scale=1,
            color="tab:blue",
            alpha=0.6,
            width=0.003,
        )

    ax.set_title(r"$\mathbf{c.}$ Template → Query deformation", fontsize=14, loc="left")
    ax.legend(labels=["Template", "Query"])
    ax.set_aspect("equal")

    # -------------------------
    # Panel d: NN class match
    # -------------------------
    ax = axes[3]
    ref_coords = ref_df[["X", "Y"]].values
    query_coords = query_df[["X", "Y"]].values
    tree = cKDTree(ref_coords)
    _, nearest_idx = tree.query(query_coords, k=1)

    query_classes = query_df["cell_type"].values
    nearest_ref_classes = ref_df["cell_type"].values[nearest_idx]
    class_match = query_classes == nearest_ref_classes
    match_rate = 100 * class_match.sum() / len(class_match)

    match_mask = class_match
    mismatch_mask = ~class_match

    ax.scatter(
        query_df.loc[match_mask, "X"],
        query_df.loc[match_mask, "Y"],
        c="green",
        s=40,
        alpha=0.7,
        label=f"Match ({match_mask.sum()})",
        edgecolors="darkgreen",
        linewidth=0.5,
    )
    ax.scatter(
        query_df.loc[mismatch_mask, "X"],
        query_df.loc[mismatch_mask, "Y"],
        c="red",
        s=40,
        alpha=0.7,
        marker="x",
        label=f"Mismatch ({mismatch_mask.sum()})",
        linewidths=1.5,
    )
    ax.set_title(r"$\mathbf{d.}$ NN Class Match (" + f"{match_rate:.1f}%)", fontsize=14, loc="left")
    ax.legend()
    ax.set_aspect("equal")

    # Common axis styling
    for ax in axes.flatten():
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.set_xticks([0, 5, 10])
        ax.set_yticks([0, 5, 10])

    plt.tight_layout()
    return fig




def visualize_space_tearing(quadrants, q_name='bottom_right', recompute_triangulation=False, min_angle_deg=10):
    """Detailed visualization of the space-tearing quadrant.
    
    Parameters:
    -----------
    quadrants : dict
        Dictionary of quadrant data from create_full_benchmark()
    q_name : str
        Name of quadrant to visualize
    recompute_triangulation : bool
        If True, recompute triangulation using only ref points that have 
        corresponding query points (useful when query has missing cells).
        Uses ground_truth_ref_idx to find the subset.
    """
    
    q = quadrants[q_name]
    
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    
    # Get ground truth mapping (query_idx -> ref_idx)
    gt = q['ground_truth_ref_idx']
    
    if recompute_triangulation:
        # Find ref indices that have valid query matches
        valid_query_idx = np.where(gt >= 0)[0]  # Query indices with valid ref match
        valid_ref_idx = gt[valid_query_idx]      # Corresponding ref indices
        valid_ref_idx = valid_ref_idx[valid_ref_idx >= 0]  # Filter out -1
        
        # Get unique ref indices that exist in query
        unique_ref_idx = np.unique(valid_ref_idx).astype(int)
        
        # Subset ref and query points to matched pairs only
        ref_points_subset = q['ref_points'][unique_ref_idx]
        ref_classes_subset = q['ref_classes'][unique_ref_idx]
        
        # Create mapping: old ref index -> new index in subset
        old_to_new = {old_idx: new_idx for new_idx, old_idx in enumerate(unique_ref_idx)}
        
        # For each query point, find its position based on ground truth
        query_points_matched = []
        query_classes_matched = []
        for query_idx in range(len(q['query_points'])):
            ref_idx = gt[query_idx]
            if ref_idx >= 0 and ref_idx in old_to_new:
                query_points_matched.append(q['query_points'][query_idx])
                query_classes_matched.append(q['query_classes'][query_idx])
        
        query_points_subset = np.array(query_points_matched) if query_points_matched else np.empty((0, 2))
        query_classes_subset = np.array(query_classes_matched) if query_classes_matched else np.array([])
        
        # Recompute triangulation on subset
        simplices = compute_filtered_delaunay(ref_points_subset, min_angle_deg=min_angle_deg)
        
        # Use subset for visualization
        ref_points_viz = ref_points_subset
        ref_classes_viz = ref_classes_subset
        query_points_viz = query_points_subset
        query_classes_viz = query_classes_subset
        
        #title_suffix = f" (recomputed, {len(unique_ref_idx)} matched points)"
        title_suffix = ""
    else:
        # Original behavior: use all points
        simplices = compute_filtered_delaunay(q['ref_points'], min_angle_deg=min_angle_deg)
        ref_points_viz = q['ref_points']
        ref_classes_viz = q['ref_classes']
        query_points_viz = q['query_points']
        query_classes_viz = q['query_classes']
        title_suffix = ""
    
    # Detect flipped triangles
    def signed_area(p1, p2, p3):
        return 0.5 * ((p2[0] - p1[0]) * (p3[1] - p1[1]) - (p3[0] - p1[0]) * (p2[1] - p1[1]))
    
    flipped = []
    preserved = []
    skipped = 0
    for i, simplex in enumerate(simplices):
        # Check if all vertices exist in query
        if max(simplex) >= len(query_points_viz):
            skipped += 1
            continue
            
        area_ref = signed_area(ref_points_viz[simplex[0]], 
                               ref_points_viz[simplex[1]], 
                               ref_points_viz[simplex[2]])
        area_query = signed_area(query_points_viz[simplex[0]], 
                                 query_points_viz[simplex[1]], 
                                 query_points_viz[simplex[2]])
        if area_ref * area_query < 0:
            flipped.append(i)
        else:
            preserved.append(i)
    
    # Plot 1: Reference
    ax = axes[0]
    ax.triplot(ref_points_viz[:, 0], ref_points_viz[:, 1], simplices, 'k-', alpha=0.3)
    # Use CLASS_COLORS for consistent coloring
    ref_colors = [CLASS_COLORS[CLASS_NAMES[c]] for c in ref_classes_viz]
    ax.scatter(ref_points_viz[:, 0], ref_points_viz[:, 1], 
               c=ref_colors, s=60, zorder=5, edgecolors='black')
    ax.set_title(f'a. Template{title_suffix}', fontsize=12)
    #ax.set_aspect('equal')
    
    # Plot 2: Query with flipped triangles
    ax = axes[1]
    if len(preserved) > 0:
        ax.triplot(query_points_viz[:, 0], query_points_viz[:, 1], 
                   simplices[preserved], color='gray', alpha=0.3, linestyle='-')
    if len(flipped) > 0:
        ax.triplot(query_points_viz[:, 0], query_points_viz[:, 1], 
                   simplices[flipped], color='magenta', linestyle='-', linewidth=2)
    query_colors = [CLASS_COLORS[CLASS_NAMES[c]] for c in query_classes_viz]
    ax.scatter(query_points_viz[:, 0], query_points_viz[:, 1], 
               c=query_colors, s=60, zorder=5, 
               edgecolors='black', marker='P')
    
    if 'swapped_pairs' in q and not recompute_triangulation:
        for i, j in q['swapped_pairs']:
            ax.annotate('', xy=q['query_points'][j], xytext=q['query_points'][i],
                       arrowprops=dict(arrowstyle='<->', color='purple', lw=2))

    ax.set_title(f'b. Query - {len(flipped)} flipped triangles{title_suffix}', fontsize=12)
    #ax.set_aspect('equal')
    
    # Plot 3: Deformation vectors (use matched points if recompute_triangulation)
    ax = axes[2]
    if recompute_triangulation and len(ref_points_viz) == len(query_points_viz):
        # Use matched subset
        ax.quiver(ref_points_viz[:, 0], ref_points_viz[:, 1],
                  query_points_viz[:, 0] - ref_points_viz[:, 0],
                  query_points_viz[:, 1] - ref_points_viz[:, 1],
                  angles='xy', scale_units='xy', scale=1, alpha=0.7)
        ax.scatter(ref_points_viz[:, 0], ref_points_viz[:, 1], c='blue', s=30, label='Template')
        ax.scatter(query_points_viz[:, 0], query_points_viz[:, 1], c='red', s=30, marker='P', label='Query')
    elif len(q['ref_points']) == len(q['query_points']):
        # Original: same number of points
        ax.quiver(q['ref_points'][:, 0], q['ref_points'][:, 1],
                  q['query_points'][:, 0] - q['ref_points'][:, 0],
                  q['query_points'][:, 1] - q['ref_points'][:, 1],
                  angles='xy', scale_units='xy', scale=1, alpha=0.7)
        ax.scatter(q['ref_points'][:, 0], q['ref_points'][:, 1], c='blue', s=30, label='Template')
        ax.scatter(q['query_points'][:, 0], q['query_points'][:, 1], c='red', s=30, marker='P', label='Query')
    else:
        # Different counts: just show points without vectors
        ax.scatter(q['ref_points'][:, 0], q['ref_points'][:, 1], c='blue', s=30, label=f'Template ({len(q["ref_points"])})')
        ax.scatter(q['query_points'][:, 0], q['query_points'][:, 1], c='red', s=30, marker='x', label=f'Query ({len(q["query_points"])})')
        ax.set_title('Points (different counts, no vectors)', fontsize=10)
    ax.set_title(f'c. Deformation Vectors{title_suffix}', fontsize=12)
    ax.legend()
    #ax.set_aspect('equal')
    for ax in axes:
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
    plt.tight_layout()
    return fig


def visualize_topological_merger(quadrants):
    """Detailed visualization of the topological split quadrant (ref=ellipse, query=2 rings)."""
    
    q = quadrants['bottom_left']
    
    fig, axes = plt.subplots(1, 2, figsize=(7, 3.5))
    
    # Reference: background + ONE ellipse
    ax = axes[0]
    c1_mask = q['ref_classes'] == 0
    c2_mask = q['ref_classes'] == 1
    c3_mask = q['ref_classes'] == 2
    ax.scatter(q['ref_points'][c1_mask, 0], q['ref_points'][c1_mask, 1],
               c=CLASS_COLORS['c1'], s=50, alpha=0.9, label='c1', 
               edgecolors='black', linewidth=0.3)
    ax.scatter(q['ref_points'][c2_mask, 0], q['ref_points'][c2_mask, 1],
               c=CLASS_COLORS['c2'], s=50, alpha=0.9, label='c2', 
               edgecolors='black', linewidth=1)
    ax.scatter(q['ref_points'][c3_mask, 0], q['ref_points'][c3_mask, 1],
               c=CLASS_COLORS['c3'], s=50, alpha=0.9, label='c3', 
               edgecolors='black', linewidth=1)
    # Draw ellipse
    from matplotlib.patches import Ellipse as EllipsePatch
    ellipse_a = q.get('ellipse_a', 1.8)
    ellipse_b = q.get('ellipse_b', 0.8)
    ellipse_center = q.get('ellipse_center', np.array([3.5, 3.5]))
    ellipse = EllipsePatch(ellipse_center, 2*ellipse_a, 2*ellipse_b, fill=False, color='orange', linestyle='--', linewidth=2)
    ax.add_patch(ellipse)
    
    ax.set_title(f'a. Template ({len(q["ref_points"])} cells)', fontsize=12)
    ax.legend(loc='upper right')
    ax.set_aspect('equal')
    # ax.set_xlim(0, 7.5)
    # ax.set_ylim(0, 7.5)
    
    # Query: background + TWO rings
    ax = axes[1]
    c1_mask = q['query_classes'] == 0
    c2_mask = q['query_classes'] == 1
    c3_mask = q['query_classes'] == 2
    ax.scatter(q['query_points'][c1_mask, 0], q['query_points'][c1_mask, 1],
               c=CLASS_COLORS['c1'], s=50, alpha=0.9, label='c1', 
               marker='P', edgecolors='black', linewidth=0.3)
    ax.scatter(q['query_points'][c2_mask, 0], q['query_points'][c2_mask, 1],
               c=CLASS_COLORS['c2'], s=50, alpha=0.9, label='c2', 
               marker='P', edgecolors='black', linewidth=1)
    ax.scatter(q['query_points'][c3_mask, 0], q['query_points'][c3_mask, 1],
               c=CLASS_COLORS['c3'], s=50, alpha=0.9, label='c3', 
               marker='P', edgecolors='black', linewidth=1)
    # Draw circles around ring centers
    if 'ring1_center' in q:
        r = q.get('ring_radius', 0.7)
        circle1 = plt.Circle(q['ring1_center'], r, fill=False, color='orange', linestyle='--', linewidth=2)
        circle2 = plt.Circle(q['ring2_center'], r, fill=False, color='orange', linestyle='--', linewidth=2)
        ax.add_patch(circle1)
        ax.add_patch(circle2)
    
    ax.set_title(f'b. Query ({len(q["query_points"])} cells)', fontsize=12)
    ax.legend(loc='upper right')
    ax.set_aspect('equal')
    # ax.set_xlim(0, 7.5)
    # ax.set_ylim(0, 7.5)
    # Remove top and right spines from all axes
    for ax in axes.flatten():
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
    plt.tight_layout()
    return fig


def print_statistics(ref_df, query_df, quadrants):
    """Print statistics about the benchmark."""
    
    print("=" * 60)
    print("SYNTHETIC 4-QUADRANT BENCHMARK STATISTICS")
    print("=" * 60)
    
    print(f"\nTotal Reference cells: {len(ref_df)}")
    print(f"Total Query cells: {len(query_df)}")
    print(f"Cell difference: {len(ref_df) - len(query_df)} (ref has more)")
    
    print("\nClass distribution in Reference:")
    print(ref_df['cell_type'].value_counts().sort_index())
    
    print("\nClass distribution in Query:")
    print(query_df['cell_type'].value_counts().sort_index())
    
    print("\n" + "-" * 60)
    print("PER-QUADRANT STATISTICS:")
    print("-" * 60)
    
    for qname, q in quadrants.items():
        print(f"\n{qname.upper().replace('_', ' ')}:")
        print(f"  Description: {q['description']}")
        print(f"  Ref points: {len(q['ref_points'])}")
        print(f"  Query points: {len(q['query_points'])}")
        print(f"  Point difference: {len(q['ref_points']) - len(q['query_points'])}")
        print(f"  Ref c1: {(q['ref_classes']==0).sum()}, c2: {(q['ref_classes']==1).sum()}, c3: {(q['ref_classes']==2).sum()}")
        print(f"  Query c1: {(q['query_classes']==0).sum()}, c2: {(q['query_classes']==1).sum()}, c3: {(q['query_classes']==2).sum()}")


# =============================================================================
# MAIN
# =============================================================================
if __name__ == "__main__":
    # Create benchmark
    ref_df, query_df, quadrants, ground_truth_df, _expression_dict = create_full_benchmark()
    
    # Print statistics
    print_statistics(ref_df, query_df, quadrants)
    
    # Visualize
    fig1 = visualize_benchmark(ref_df, query_df, quadrants)
    fig1.savefig('benchmark_overview.png', dpi=150, bbox_inches='tight')
    
    fig2 = visualize_space_tearing(quadrants)
    fig2.savefig('benchmark_space_tearing.png', dpi=150, bbox_inches='tight')
    
    fig3 = visualize_topological_merger(quadrants)
    fig3.savefig('benchmark_topological_merger.png', dpi=150, bbox_inches='tight')
    
    plt.show()
    
    # Save data
    ref_df.to_csv('benchmark_ref.csv', index=False)
    query_df.to_csv('benchmark_query.csv', index=False)
    ground_truth_df.to_csv('benchmark_ground_truth.csv', index=False)
    
    print("\n" + "=" * 60)
    print("Files saved:")
    print("  - benchmark_ref.csv")
    print("  - benchmark_query.csv")
    print("  - benchmark_ground_truth.csv")
    print("  - benchmark_overview.png")
    print("  - benchmark_space_tearing.png")
    print("  - benchmark_topological_merger.png")
    print("=" * 60)
import numpy as np
from scipy.spatial import Delaunay

import numpy as np

def check_triangle_violations_within_quadrants(matches_df, mc_align):
    """
    Check triangle violations only within quadrants (ignore cross-quadrant triangles).
    
    Parameters:
    -----------
    matches_df : pd.DataFrame
        SAME matching results with columns: aligned_idx, ref_idx, X, Y, ref_X, ref_Y
    mc_align : MetaCell object
        Contains metacell_delaunay (array of simplices) and metacell_df with 'quadrant' column
    
    Returns:
    --------
    matches_df : pd.DataFrame
        Updated with 'triangle_violation' column (True if violation, False otherwise)
    """
    # Get Delaunay simplices (array of triangles)
    simplices = mc_align.metacell_delaunay
    metacell_df = mc_align.metacell_df
    
    # Create mapping: aligned_idx -> metacell_id (from matches_df)
    # aligned_idx is the index in query_df, which corresponds to metacell_id
    aligned_to_metacell = dict(zip(matches_df['aligned_idx'], matches_df.get('Aligned_metacell_id', matches_df['aligned_idx'])))
    
    # If Aligned_metacell_id doesn't exist, assume aligned_idx IS the metacell_id
    if 'Aligned_metacell_id' not in matches_df.columns:
        aligned_to_metacell = dict(zip(matches_df['aligned_idx'], matches_df['aligned_idx']))
    
    # Get quadrant for each metacell_id
    metacell_to_quadrant = dict(zip(metacell_df.index, metacell_df['quadrant']))
    
    # Function to compute signed area
    def signed_area(p1, p2, p3):
        return 0.5 * ((p2[0] - p1[0]) * (p3[1] - p1[1]) - (p3[0] - p1[0]) * (p2[1] - p1[1]))
    
    # Track violations per matched point
    violations = np.zeros(len(matches_df), dtype=bool)
    
    # Check each triangle
    for simplex in simplices:
        # Get metacell_ids (indices in metacell_df)
        metacell_id1, metacell_id2, metacell_id3 = simplex
        
        # Get quadrants for all three points
        q1 = metacell_to_quadrant.get(metacell_id1)
        q2 = metacell_to_quadrant.get(metacell_id2)
        q3 = metacell_to_quadrant.get(metacell_id3)
        
        # Skip if any quadrant is missing or if they're not all the same
        if q1 is None or q2 is None or q3 is None:
            continue
        if q1 != q2 or q2 != q3:
            continue  # Cross-quadrant triangle, skip
        
        # Find aligned_idx for these metacell_ids (reverse lookup)
        aligned_idx1 = None
        aligned_idx2 = None
        aligned_idx3 = None
        
        for aligned_idx, mc_id in aligned_to_metacell.items():
            if mc_id == metacell_id1:
                aligned_idx1 = aligned_idx
            if mc_id == metacell_id2:
                aligned_idx2 = aligned_idx
            if mc_id == metacell_id3:
                aligned_idx3 = aligned_idx
        
        # Skip if any point is not in matches_df
        if aligned_idx1 is None or aligned_idx2 is None or aligned_idx3 is None:
            continue
        
        # Get positions from matches_df
        match1 = matches_df[matches_df['aligned_idx'] == aligned_idx1].iloc[0]
        match2 = matches_df[matches_df['aligned_idx'] == aligned_idx2].iloc[0]
        match3 = matches_df[matches_df['aligned_idx'] == aligned_idx3].iloc[0]
        
        # Ref positions (old)
        ref_pos1 = np.array([match1['ref_X'], match1['ref_Y']])
        ref_pos2 = np.array([match2['ref_X'], match2['ref_Y']])
        ref_pos3 = np.array([match3['ref_X'], match3['ref_Y']])
        
        # Query positions (new)
        query_pos1 = np.array([match1['X'], match1['Y']])
        query_pos2 = np.array([match2['X'], match2['Y']])
        query_pos3 = np.array([match3['X'], match3['Y']])
        
        # Compute signed areas
        area_ref = signed_area(ref_pos1, ref_pos2, ref_pos3)
        area_query = signed_area(query_pos1, query_pos2, query_pos3)
        
        # Violation if areas have opposite signs (triangle flipped)
        if area_ref * area_query < 0:
            # Mark all three points as having violations
            idx1 = matches_df[matches_df['aligned_idx'] == aligned_idx1].index[0]
            idx2 = matches_df[matches_df['aligned_idx'] == aligned_idx2].index[0]
            idx3 = matches_df[matches_df['aligned_idx'] == aligned_idx3].index[0]
            violations[idx1] = True
            violations[idx2] = True
            violations[idx3] = True
    
    # Add violation column
    matches_df = matches_df.copy()
    matches_df['triangle_violation'] = violations
    
    return matches_df
