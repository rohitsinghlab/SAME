import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import Delaunay
from shapely.geometry import MultiPolygon, Polygon, GeometryCollection
from alphashape import alphashape

def is_triangle_within_alpha(triangle, alpha_shape):
    # Convert the triangle to a Shapely Polygon
    triangle_polygon = Polygon(triangle)
    
    # Check if the triangle is within the alpha shape
    return alpha_shape.contains(triangle_polygon)

def compute_filtered_delaunay(old_coords, alpha=0.05):
    """
    Computes the filtered Delaunay triangulation based on an alpha shape.

    Parameters:
    - old_coords: array-like, shape (n_points, 2)
        The coordinates of the points to triangulate.
    - alpha: float, optional (default=0.05)
        The alpha value for the alpha shape. This value might need tuning.

    Returns:
    - filtered_triangles: list of simplices
        The list of filtered triangles that are within the alpha shape.
    """
    # Convert old_coords to a list of tuples
    points = [tuple(coord) for coord in old_coords]

    # Compute alpha shape
    alpha_shape = alphashape(points, alpha)

    # Compute Delaunay triangulation
    delaunay = Delaunay(points)

    # Filter triangles based on alpha shape
    filtered_triangles = []
    for simplex in delaunay.simplices:
        triangle = [points[index] for index in simplex]  # Correctly index into points
        if is_triangle_within_alpha(triangle, alpha_shape):
            filtered_triangles.append(simplex)
    print(len(filtered_triangles), len(delaunay.simplices))
    return filtered_triangles
def compute_smallest_angle(triangle):
    a = np.linalg.norm(triangle[0] - triangle[1])
    b = np.linalg.norm(triangle[1] - triangle[2])
    c = np.linalg.norm(triangle[2] - triangle[0])
    return min(180-np.degrees(np.arccos((a**2 + b**2 - c**2) / (2 * a * b))), np.degrees(np.arccos((a**2 + b**2 - c**2) / (2 * a * b))))

MIN_ANGLE = 10
def compute_triangle_radius(old_coords, r):
    """
    Finds all triangles with edges of length at least 'r', using each node once.

    Parameters:
    - old_coords: array-like, shape (n_points, 2)
        The coordinates of the points to form triangles.
    - r: float
        The minimum desired length of triangle edges.

    Returns:
    - valid_triangles: list of tuples
        The list of valid triangles, where each triangle is represented by indices of its vertices.
    """
    points = np.array(old_coords)
    n = len(points)
    valid_triangles = []
    used_points = set()

    def distance(p1, p2):
        return np.linalg.norm(p1 - p2)

    for i in range(n - 2):
        if i in used_points:
            continue
        for j in range(i + 1, n - 1):
            if j in used_points or distance(points[i], points[j]) < r:
                continue
            for k in range(j + 1, n):
                if k in used_points:
                    continue
                if (distance(points[i], points[k]) >= r and 
                    distance(points[j], points[k]) >= r and 
                    compute_smallest_angle([points[i], points[j], points[k]]) > MIN_ANGLE and
                    calculate_signed_area(*[points[i], points[j], points[k]]) <=2*r):
                    valid_triangles.append((i, j, k))
                    used_points.update([i, j, k])
                    break
            if i in used_points:
                break
        if i in used_points:
            continue
    print(len(used_points), len(points))
    print(f"Found {len(valid_triangles)} valid triangles")
    return valid_triangles
def create_point_subsets(points, r, n_subsets):
    """
    Create n subsets of points with minimal overlap, ensuring points in each subset are at least r distance apart.

    Parameters:
    - points: array-like, shape (n_points, 2)
        The coordinates of the points.
    - r: float
        The minimum distance between points in a subset.
    - n_subsets: int
        The number of subsets to create.

    Returns:
    - subsets: list of arrays
        A list containing the subsets of points.
    """
    subsets = []
    #print(r)
    for _ in range(n_subsets):
        subset = []
        for point_ix in np.random.choice(range(len(points)), len(points)):
            point = points[point_ix]
            for existing_point_ix in subset:
                if np.linalg.norm(np.array(point) - np.array(points[existing_point_ix])) < r:
                    break
            else:
                subset.append(point_ix)
        subsets.append(np.array(subset))
    return subsets
from tqdm.notebook import tqdm

def compute_triangle_radiusv2(old_coords, r, min_triangles, n_subsets=1000):
    """
    Finds all triangles with edges of length at least 'r', using each node at most twice.

    Parameters:
    - old_coords: array-like, shape (n_points, 2)
        The coordinates of the points to form triangles.
    - r: float
        The minimum desired length of triangle edges.
    - min_triangles: int
        The minimum number of triangles to find.
    - n_subsets: int, optional (default=5)
        The number of subsets to create.

    Returns:
    - valid_triangles: list of tuples
        The list of valid triangles, where each triangle is represented by indices of its vertices.
    """
    points = np.array(old_coords)
    valid_triangles = []

    # Create subsets of points
    #plt.scatter(points[:,0], points[:,1])
    #plt.scatter(subsets[0][:,0], subsets[0][:,1], c='r')
    #plt.show()
    # Iterate over each subset and compute filtered Delaunay

    for ix in tqdm(range(n_subsets)):
        subset = create_point_subsets(points, r, 1)[0]
        #plt.scatter(points[subset][:,0], points[subset][:,1], c='r')
        #print('Subset', ix, '. # of points:', len(subset))
        if len(valid_triangles) >= min_triangles:
            print(f"Found {len(valid_triangles)} valid triangles")
            break
        delaunay_triangles = Delaunay(points[subset])#compute_filtered_delaunay(points[subset], alpha=0.1)
        for simplex in delaunay_triangles.simplices:
            new_simplex = [subset[i] for i in simplex]
            # plt.plot([points[new_simplex[0]][0], points[new_simplex[1]][0]], [points[new_simplex[0]][1], points[new_simplex[1]][1]], 'k-', alpha=0.5, linewidth=1)
            # plt.plot([points[new_simplex[1]][0], points[new_simplex[2]][0]], [points[new_simplex[1]][1], points[new_simplex[2]][1]], 'k-', alpha=0.5, linewidth=1)
            # plt.plot([points[new_simplex[2]][0], points[new_simplex[0]][0]], [points[new_simplex[2]][1], points[new_simplex[0]][1]], 'k-', alpha=0.5, linewidth=1)
            # plt.show()
            if compute_smallest_angle([points[new_simplex[0]], points[new_simplex[1]], points[new_simplex[2]]]) > MIN_ANGLE:
                if new_simplex not in valid_triangles:
                    valid_triangles.append(new_simplex)

        #print(f"Found {len(valid_triangles)} valid triangles")
    # Ensure we only return the required number of triangles
    valid_triangles = list(valid_triangles)
    #print(valid_triangles[0])
    return valid_triangles[:min_triangles]

def calculate_signed_area(p1, p2, p3):
    x1, y1 = p1
    x2, y2 = p2
    x3, y3 = p3
    return 0.5 * (x1 * (y2 - y3) + x2 * (y3 - y1) + x3 * (y1 - y2))

def check_mesh_orientation(df, x1='X', y1='Y', x2='ref_X', y2='ref_Y', triangle='delaunay', radius=50, alpha=0.1,delaunay=None):
    # Extract coordinates
    old_coords = df[[x1, y1]].values
    new_coords = df[[x2, y2]].values
    MIN_ANGLE = 20
    # Perform Delaunay triangulation on the old coordinates
    if delaunay is None:
        delaunay = Delaunay(old_coords)
    df['violation'] = 0
    # Initialize violation count
    violation_count = np.zeros(len(df))
    if triangle == 'delaunay':
        delaunay = compute_filtered_delaunay(old_coords, alpha)
    elif triangle == 'radius':
        delaunay = compute_filtered_delaunay(old_coords, alpha)
        min_triangles = len(delaunay)
        delaunay = compute_triangle_radiusv2(old_coords, radius, min_triangles)
    elif triangle == 'custom':
        if delaunay is None:
            raise ValueError("Custom triangle type requires a pre-computed delaunay triangulation")
    else:
        raise ValueError(f"Invalid triangle type: {triangle}")
    # Check each triangle in the triangulation
    
    for simplex in tqdm(delaunay):
        #print(simplex)
        old_triangle = old_coords[simplex]
        new_triangle = new_coords[simplex]
        
        # Calculate signed areas
        old_area = calculate_signed_area(*old_triangle)
        new_area = calculate_signed_area(*new_triangle)
        #print(simplex, old_area, new_area)
        # Check if the orientation has changed
        #print(old_area, new_area, compute_smallest_angle(new_triangle))
        MIN_ANGLE = 0
        if old_area * new_area < 0 and compute_smallest_angle(new_triangle)>MIN_ANGLE:
            #print(old_area, new_area)
            #print(df.iloc[[simplex[0], simplex[1], simplex[2]]][['X', 'Y', 'ref_X', 'ref_Y', 'window_id']])
            #print(df.iloc[[simplex[0], simplex[1], simplex[2]]][['X', 'Y', 'ref_X', 'ref_Y']])
            #plt.plot(df.iloc[[simplex[0], simplex[1], simplex[2]]]['X'], df.iloc[[simplex[0], simplex[1], simplex[2]]]['Y'], c='r')
            #plt.plot(df.iloc[[simplex[0], simplex[1], simplex[2]]]['ref_X'], df.iloc[[simplex[0], simplex[1], simplex[2]]]['ref_Y'], c='b')
            
            # Label the vertices
            #for idx in simplex:
            #    plt.text(df.iloc[idx]['X'], df.iloc[idx]['Y'], f'{idx}', fontsize=9, ha='right', color='red')
            #    plt.text(df.iloc[idx]['ref_X'], df.iloc[idx]['ref_Y'], f'{idx}', fontsize=9, ha='right', color='blue')
            #plt.gca().invert_yaxis()
            #plt.show()
            #print(compute_smallest_angle(old_triangle), compute_smallest_angle(new_triangle))
            #violation_count[simplex[0]] = 1
            #violation_count[simplex[1]] = 1
            #violation_count[simplex[2]] = 1
            df.iloc[simplex[0], df.columns.get_loc('violation')] = 1
            df.iloc[simplex[1], df.columns.get_loc('violation')] = 1
            df.iloc[simplex[2], df.columns.get_loc('violation')] = 1
    return df, delaunay

def check_mesh_bounds(df, x1='X', y1='Y', x2='ref_X', y2='ref_Y', delaunay=None):
    df_new = df.copy()
    # Ensure delaunay is provided
    if delaunay is None:
        raise ValueError("Delaunay triangulation must be provided")

    # Initialize a new column for tracking changes
    df_new['bound_violation'] = 0

    for simplex in tqdm(delaunay):
        # Extract the vertices of the current triangle
        old_triangle = np.round(df.iloc[simplex][[x1, y1]].values, 4)
        new_triangle = np.round(df.iloc[simplex][[x2, y2]].values, 4)
        # Find the vertices corresponding to max_x, min_x, max_y, min_y using x1, y1 coordinates
        max_x_indices = np.where(old_triangle[:, 0] == old_triangle[:, 0].max())[0]
        min_x_indices = np.where(old_triangle[:, 0] == old_triangle[:, 0].min())[0]
        max_y_indices = np.where(old_triangle[:, 1] == old_triangle[:, 1].max())[0]
        min_y_indices = np.where(old_triangle[:, 1] == old_triangle[:, 1].min())[0]

        # Check if these vertices are still the same with x2, y2 coordinates
        max_x_indices_new = np.where(new_triangle[:, 0] == new_triangle[:, 0].max())[0]
        min_x_indices_new = np.where(new_triangle[:, 0] == new_triangle[:, 0].min())[0]
        max_y_indices_new = np.where(new_triangle[:, 1] == new_triangle[:, 1].max())[0]
        min_y_indices_new = np.where(new_triangle[:, 1] == new_triangle[:, 1].min())[0]
        # Count the number of true conditions
        true_conditions_x = sum([
            bool(set(max_x_indices).intersection(max_x_indices_new)),
            bool(set(min_x_indices).intersection(min_x_indices_new))
        ])
        true_conditions_y = sum([
            bool(set(max_y_indices).intersection(max_y_indices_new)),
            bool(set(min_y_indices).intersection(min_y_indices_new))
        ])
        
        # Check if at least two conditions are false
        if true_conditions_x < 1 and true_conditions_y < 1 and len(df.iloc[simplex].window_id.unique()) == 1:     
            # print( df.iloc[simplex].window_id)
            # print('Max_x', simplex[max_x_indices][0], simplex[max_x_indices_new][0], df.iloc[simplex[max_x_indices]][x1].values[0], df.iloc[simplex[max_x_indices_new]][x2].values[0])
            # print('Min_x', simplex[min_x_indices][0], simplex[min_x_indices_new][0], df.iloc[simplex[min_x_indices]][x1].values[0], df.iloc[simplex[min_x_indices_new]][x2].values[0])
            # print('Max_y', simplex[max_y_indices][0], simplex[max_y_indices_new][0], df.iloc[simplex[max_y_indices]][y1].values[0], df.iloc[simplex[max_y_indices_new]][y2].values[0])
            # print('Min_y', simplex[min_y_indices][0], simplex[min_y_indices_new][0], df.iloc[simplex[min_y_indices]][y1].values[0], df.iloc[simplex[min_y_indices_new]][y2].values[0],)   
            # plt.plot(df.iloc[[simplex[0], simplex[1], simplex[2], simplex[0]]][x1], df.iloc[[simplex[0], simplex[1], simplex[2], simplex[0]]][y1], c='r')
            # plt.plot(df.iloc[[simplex[0], simplex[1], simplex[2], simplex[0]]][x2], df.iloc[[simplex[0], simplex[1], simplex[2], simplex[0]]][y2], c='b')
            
            # for idx in simplex:
            #    plt.text(df.iloc[idx]['X'], df.iloc[idx]['Y'], f'{idx}', fontsize=9, ha='right', color='red')
            #    plt.text(df.iloc[idx]['ref_X'], df.iloc[idx]['ref_Y'], f'{idx}', fontsize=9, ha='right', color='blue')
            # #plt.gca().invert_yaxis()
            # plt.show()
            
            # old_area = calculate_signed_area((df_new.iloc[simplex[0]][x1], df_new.iloc[simplex[0]][y1]), (df_new.iloc[simplex[1]][x1],df_new.iloc[simplex[1]][y1]), (df_new.iloc[simplex[2]][x1], df_new.iloc[simplex[2]][y1]))
            # new_area = calculate_signed_area((df_new.iloc[simplex[0]][x2], df_new.iloc[simplex[0]][y2]), (df_new.iloc[simplex[1]][x2],df_new.iloc[simplex[1]][y2]), (df_new.iloc[simplex[2]][x2], df_new.iloc[simplex[2]][y2]))
            # if old_area*new_area > 0:                
            #     print('Max_x', simplex[max_x_indices][0], simplex[max_x_indices_new][0], df.iloc[simplex[max_x_indices]][x1].values[0], df.iloc[simplex[max_x_indices_new]][x2].values[0])
            #     print('Min_x', simplex[min_x_indices][0], simplex[min_x_indices_new][0], df.iloc[simplex[min_x_indices]][x1].values[0], df.iloc[simplex[min_x_indices_new]][x2].values[0])
            #     print('Max_y', simplex[max_y_indices][0], simplex[max_y_indices_new][0], df.iloc[simplex[max_y_indices]][y1].values[0], df.iloc[simplex[max_y_indices_new]][y2].values[0])
            #     print('Min_y', simplex[min_y_indices][0], simplex[min_y_indices_new][0], df.iloc[simplex[min_y_indices]][y1].values[0], df.iloc[simplex[min_y_indices_new]][y2].values[0])   
            #     plt.plot(df.iloc[[simplex[0], simplex[1], simplex[2], simplex[0]]][x1], df.iloc[[simplex[0], simplex[1], simplex[2], simplex[0]]][y1], c='r')
            #     plt.plot(df.iloc[[simplex[0], simplex[1], simplex[2], simplex[0]]][x2], df.iloc[[simplex[0], simplex[1], simplex[2], simplex[0]]][y2], c='b')
                
            #     for idx in simplex:
            #        plt.text(df.iloc[idx]['X'], df.iloc[idx]['Y'], f'{idx}', fontsize=9, ha='right', color='red')
            #        plt.text(df.iloc[idx]['ref_X'], df.iloc[idx]['ref_Y'], f'{idx}', fontsize=9, ha='right', color='blue')
            #     plt.gca().invert_yaxis()
            #     plt.show()
    
            # Mark the vertices of the triangle as changed
            df_new.iloc[simplex[0], df_new.columns.get_loc('bound_violation')] = 1
            df_new.iloc[simplex[1], df_new.columns.get_loc('bound_violation')] = 1
            df_new.iloc[simplex[2], df_new.columns.get_loc('bound_violation')] = 1

    return df_new

def compute_celltype_match(df, x1='X', y1='Y', x2='ref_X', y2='ref_Y'):
    df_new = df.copy()
    df_new['celltype_match'] = 0