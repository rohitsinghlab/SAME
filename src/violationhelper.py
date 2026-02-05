def verify_spatial_preservation(aligned_df, ref_df, matches_df, triangle_info, tolerance=1e-6):
    """
    Verify if spatial relationships are preserved in the matched points.
    
    Parameters:
    -----------
    aligned_df : pd.DataFrame
        Original aligned points with 'X', 'Y' coordinates
    ref_df : pd.DataFrame
        Reference points with 'X', 'Y' coordinates
    matches_df : pd.DataFrame
        Dataframe containing the matches with columns:
        ['aligned_idx', 'ref_idx', 'X', 'Y', 'ref_X', 'ref_Y']
    triangle_info : dict
        Dictionary containing triangle information from Delaunay triangulation
    tolerance : float
        Tolerance for floating point comparisons
        
    Returns:
    --------
    dict
        Detailed report of spatial preservation violations
    """
    violations = {
        'x_order_violations': [],
        'y_order_violations': [],
        'triangles_with_violations': set(),
        'points_with_violations': set(),
        'violation_summary': {
            'total_triangles': len(triangle_info),
            'violated_triangles': 0,
            'total_comparisons': 0,
            'total_violations': 0
        }
    }
    
    # Create lookup for matched points
    match_lookup = {row['aligned_idx']: row['ref_idx'] 
                   for _, row in matches_df.iterrows()}
    
    # Create direct lookups for coordinates using positional indices (iloc)
    # triangle_info vertices are positional indices into aligned_df
    aligned_coords = {
        idx: {'X': aligned_df.iloc[idx]['X'], 'Y': aligned_df.iloc[idx]['Y']} 
        for idx in range(len(aligned_df))
    }
    ref_coords = {
        idx: {'X': ref_df.iloc[idx]['X'], 'Y': ref_df.iloc[idx]['Y']} 
        for idx in range(len(ref_df))
    }
    
    # Process each triangle
    for simplex_idx, info in triangle_info.items():
        triangle_violated = False
        vertices = info['vertices']
        
        # Check if all vertices in this triangle are matched
        matched_vertices = [v for v in vertices if v in match_lookup]
        if len(matched_vertices) < 2:
            continue  # Need at least 2 points to check relative positions
            
        # Check relative X positions
        for i, v1 in enumerate(matched_vertices):
            for v2 in matched_vertices[i+1:]:
                violations['violation_summary']['total_comparisons'] += 1
                
                # Original relative positions
                orig_x_order = aligned_coords[v1]['X'] < aligned_coords[v2]['X']
                orig_y_order = aligned_coords[v1]['Y'] < aligned_coords[v2]['Y']
                
                # Matched relative positions
                ref_idx1 = match_lookup[v1]
                ref_idx2 = match_lookup[v2]
                matched_x_order = ref_coords[ref_idx1]['X'] < ref_coords[ref_idx2]['X']
                matched_y_order = ref_coords[ref_idx1]['Y'] < ref_coords[ref_idx2]['Y']
                
                # Check X order preservation
                if orig_x_order != matched_x_order:
                    violations['x_order_violations'].append({
                        'triangle_idx': simplex_idx,
                        'point1': {
                            'aligned_idx': v1,
                            'ref_idx': ref_idx1,
                            'orig_x': aligned_coords[v1]['X'],
                            'matched_x': ref_coords[ref_idx1]['X']
                        },
                        'point2': {
                            'aligned_idx': v2,
                            'ref_idx': ref_idx2,
                            'orig_x': aligned_coords[v2]['X'],
                            'matched_x': ref_coords[ref_idx2]['X']
                        }
                    })
                    triangle_violated = True
                    violations['points_with_violations'].update([v1, v2])
                    violations['violation_summary']['total_violations'] += 1
                
                # Check Y order preservation
                if orig_y_order != matched_y_order:
                    violations['y_order_violations'].append({
                        'triangle_idx': simplex_idx,
                        'point1': {
                            'aligned_idx': v1,
                            'ref_idx': ref_idx1,
                            'orig_y': aligned_coords[v1]['Y'],
                            'matched_y': ref_coords[ref_idx1]['Y']
                        },
                        'point2': {
                            'aligned_idx': v2,
                            'ref_idx': ref_idx2,
                            'orig_y': aligned_coords[v2]['Y'],
                            'matched_y': ref_coords[ref_idx2]['Y']
                        }
                    })
                    triangle_violated = True
                    violations['points_with_violations'].update([v1, v2])
                    violations['violation_summary']['total_violations'] += 1
        
        if triangle_violated:
            violations['triangles_with_violations'].add(simplex_idx)
            violations['violation_summary']['violated_triangles'] += 1
    
    # Convert sets to lists for JSON serialization
    violations['triangles_with_violations'] = list(violations['triangles_with_violations'])
    violations['points_with_violations'] = list(violations['points_with_violations'])
    
    # Add percentage summaries
    summary = violations['violation_summary']
    summary['percent_triangles_violated'] = (summary['violated_triangles'] / summary['total_triangles'] * 100 
                                           if summary['total_triangles'] > 0 else 0)
    summary['percent_violations'] = (summary['total_violations'] / summary['total_comparisons'] * 100 
                                   if summary['total_comparisons'] > 0 else 0)
    
    return violations

def print_violation_report(violations):
    """
    Print a human-readable report of the violations.
    """
    summary = violations['violation_summary']
    print("\nSpatial Preservation Violation Report")
    print("=====================================")
    print(f"Total triangles analyzed: {summary['total_triangles']}")
    print(f"Triangles with violations: {summary['violated_triangles']} ({summary['percent_triangles_violated']:.2f}%)")
    print(f"Total position comparisons: {summary['total_comparisons']}")
    print(f"Total violations found: {summary['total_violations']} ({summary['percent_violations']:.2f}%)")
    print(f"Number of points involved in violations: {len(violations['points_with_violations'])}")
    
    # print("\nDetailed X-order violations:")
    # for v in violations['x_order_violations'][:5]:  # Show first 5 violations
    #     print(f"\nTriangle {v['triangle_idx']}:")
    #     p1, p2 = v['point1'], v['point2']
    #     print(f"  Point {p1['aligned_idx']} -> {p1['ref_idx']}: {p1['orig_x']:.2f} -> {p1['matched_x']:.2f}")
    #     print(f"  Point {p2['aligned_idx']} -> {p2['ref_idx']}: {p2['orig_x']:.2f} -> {p2['matched_x']:.2f}")
    
    # if len(violations['x_order_violations']) > 5:
    #     print(f"... and {len(violations['x_order_violations'])-5} more X-order violations")
    
    # print("\nDetailed Y-order violations:")
    # for v in violations['y_order_violations'][:5]:  # Show first 5 violations
    #     print(f"\nTriangle {v['triangle_idx']}:")
    #     p1, p2 = v['point1'], v['point2']
    #     print(f"  Point {p1['aligned_idx']} -> {p1['ref_idx']}: {p1['orig_y']:.2f} -> {p1['matched_y']:.2f}")
    #     print(f"  Point {p2['aligned_idx']} -> {p2['ref_idx']}: {p2['orig_y']:.2f} -> {p2['matched_y']:.2f}")
    
    # if len(violations['y_order_violations']) > 5:
    #     print(f"... and {len(violations['y_order_violations'])-5} more Y-order violations")