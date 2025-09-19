from collections import defaultdict
import numpy as np
import scanpy as sc

def find_knn_with_cell_type_priority(aligned_df, ref_df, radius, knn=5):
    """
    Find KNN pairs with priority for same cell type matches.
    If the closest neighbor has the same cell type and isn't already matched, only return that match.
    If the reference point is already matched, return all pairs.
    """
    from .utils import find_knn_within_radius
    
    # First get all potential pairs within radius
    aligned_df, ref_df, all_pairs = find_knn_within_radius(aligned_df, ref_df, radius, knn=knn)
    
    # Group pairs by aligned point index
    pairs_by_aligned = defaultdict(list)
    for i, j in all_pairs:
        pairs_by_aligned[i].append(j)
    
    # Filter pairs based on cell type
    filtered_pairs = []
    same_type_matches = 0
    points_with_all_pairs = 0
    points_with_no_matches = 0
    
    # Track which reference points have been matched with priority
    ref_points_matched = set()
    
    # Process each aligned point
    for aligned_idx in range(len(aligned_df)):
        if aligned_idx not in pairs_by_aligned:
            points_with_no_matches += 1
            continue
            
        ref_indices = pairs_by_aligned[aligned_idx]
        aligned_type = aligned_df.iloc[aligned_idx]['cell_type']
        
        # Sort ref points by distance
        ref_points = [(j, ref_df.iloc[j]['cell_type'],
                      np.sqrt((aligned_df.iloc[aligned_idx]['X'] - ref_df.iloc[j]['X'])**2 +
                             (aligned_df.iloc[aligned_idx]['Y'] - ref_df.iloc[j]['Y'])**2))
                     for j in ref_indices]
        
        if not ref_points:
            points_with_no_matches += 1
            continue
            
        ref_points.sort(key=lambda x: x[2])  # Sort by distance
        
        # Check if closest point has same type
        closest_ref_idx = ref_points[0][0]
        closest_ref_type = ref_points[0][1]
        
        # If closest point has same type and hasn't been matched yet
        if closest_ref_type == aligned_type and closest_ref_idx not in ref_points_matched:
            filtered_pairs.append((aligned_idx, closest_ref_idx))
            ref_points_matched.add(closest_ref_idx)
            same_type_matches += 1
        else:
            # Keep all pairs if either:
            # 1. Closest point is different type
            # 2. Closest point is already matched
            filtered_pairs.extend((aligned_idx, j) for j, _, _ in ref_points)
            points_with_all_pairs += 1
    
    print(f"\nPair filtering summary:")
    print(f"Total aligned points: {len(aligned_df)}")
    print(f"Total reference points: {len(ref_df)}")
    print(f"Total pairs before filtering: {len(all_pairs)}")
    print(f"Points with unique same-type closest match: {same_type_matches}")
    print(f"Points keeping all matches: {points_with_all_pairs}")
    print(f"Points with no matches within radius: {points_with_no_matches}")
    print(f"Total pairs after filtering: {len(filtered_pairs)}")
    print(f"Reference points used in priority matches: {len(ref_points_matched)}")
    print(f"Average pairs per matched point: {len(filtered_pairs)/(same_type_matches + points_with_all_pairs):.2f}")
    
    return aligned_df, ref_df, filtered_pairs