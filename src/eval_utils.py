from scipy.spatial import cKDTree
from tqdm import tqdm
import numpy as np
import pandas as pd

def check_alignment(queryDF, templateDF, xcol, ycol, ctype_col = 'cell_type', kNN=1):
    """
    Check if the dominant cell type of queryDF is among the nearest k cells in templateDF.
    
    Args:
        queryDF: First input DataFrame containing coordinates and cell types.
        templateDF: Second input DataFrame containing coordinates and cell types.
        xcol: Name of the X coordinate column.
        ycol: Name of the Y coordinate column.
        kNN: Number of nearest neighbors to consider.
    
    Returns:
        inDF1 with an additional column indicating if the dominant type is among the nearest k.
    """
    
    # Make a copy to avoid SettingWithCopyWarning if queryDF is a view/slice
    queryDF = queryDF.copy()
    
    # Ensure the necessary columns are present
    required_columns = {xcol, ycol, ctype_col}
    if not required_columns.issubset(queryDF.columns) or not required_columns.issubset(templateDF.columns):
        raise ValueError(f"Both DataFrames must contain the columns: {required_columns}")
    
    # Create KDTree for efficient nearest neighbor search
    tree = cKDTree(templateDF[[xcol, ycol]])
    
    # Find nearest k neighbors
    distances, indices = tree.query(queryDF[[xcol, ycol]], k=kNN)

    
    # Check if the dominant cell type is among the nearest k neighbors
    if kNN == 1:
        # When k=1, indices is 1D
        queryDF.loc[:,'_'+str(kNN)+'NN_match'] = [
            queryDF.iloc[i][ctype_col] == templateDF.iloc[indices[i]][ctype_col]
            for i in range(len(queryDF))
        ]
        queryDF.loc[:,'_'+str(kNN)+'NN_match_ctype'] = templateDF.iloc[indices][ctype_col].values
    else:
        # When kNN>1, indices is 2D
        queryDF.loc[:,'_'+str(kNN)+'NN_match'] = [
            queryDF.iloc[i][ctype_col] in templateDF.iloc[indices[i]][ctype_col].values
            for i in range(len(queryDF))]
       #queryDF.loc[:,'_'+str(kNN)+'NN_match_ctype'] = templateDF.iloc[indices][ctype_col].values
    # Calculate alignment score
    alignment_score = queryDF['_'+str(kNN)+'NN_match'].mean()
    
    return queryDF, alignment_score


def signed_area(p1, p2, p3):
    """Compute the signed area (orientation) of the triangle p1, p2, p3 in 2D."""
    return 0.5 * (
        (p2[0] - p1[0]) * (p3[1] - p1[1]) -
        (p2[1] - p1[1]) * (p3[0] - p1[0])
    )


# source: adapted from user's check_triangle_violations in @src/eval_utils.py

def check_triangle_violations(
 outputDF,mc_align,
    aligned_id_col='aligned_metacell_index',
    ref_id_col='matched_ref_index',
    mapped_x_col='mapped_x',
    mapped_y_col='mapped_y',
    cell_type_col='cell_type',
    ignore_same_type_triangles=True,
    node_local=False,
    majority_threshold=0.5,
    min_flips=1,
    verbose=False,
):
    """
    Check for triangle orientation flips (space-tearing violations) after alignment.

    node_local:
        If False (default SAME behavior):
            A node is violating if it appears in ANY flipped triangle.

        If True:
            A node is violating only if a majority of its incident triangles flip,
            and it appears in at least `min_flips` flipped triangles.
    """

    import numpy as np

    outputDF = outputDF.copy()
    triangles = mc_align.metacell_delaunay

    align_idx_to_output_idx = {
        row[aligned_id_col]: idx
        for idx, row in outputDF.iterrows()
    }

    aligned_to_ref_map = {
        row[aligned_id_col]: row[ref_id_col]
        for _, row in outputDF.iterrows()
    }

    # --- tracking ---
    node_in_violating_triangle = {x: False for x in outputDF[aligned_id_col].unique()}

    # when node_local=True
    node_tri_counts = {x: 0 for x in node_in_violating_triangle}
    node_flip_counts = {x: 0 for x in node_in_violating_triangle}

    sign_flips = []
    tri_with_matched = tri_processed = tri_same_type_skipped = 0


    def _signed_area(p1, p2, p3):
        return 0.5 * (
            p1[0]*(p2[1]-p3[1])
            + p2[0]*(p3[1]-p1[1])
            + p3[0]*(p1[1]-p2[1])
        )


    for tri in triangles:
        idx0, idx1, idx2 = tri

        if not all(idx in aligned_to_ref_map for idx in [idx0, idx1, idx2]):
            continue

        tri_with_matched += 1

        # --- same-type skipping ---
        is_same_type = False
        if ignore_same_type_triangles:
            dom_types = [
                outputDF.loc[align_idx_to_output_idx[idx0], cell_type_col],
                outputDF.loc[align_idx_to_output_idx[idx1], cell_type_col],
                outputDF.loc[align_idx_to_output_idx[idx2], cell_type_col],
            ]
            if dom_types[0] == dom_types[1] == dom_types[2]:
                is_same_type = True
                tri_same_type_skipped += 1
            else:
                is_same_type = False

        tri_processed += 1

        try:
            aligned_coords = np.array([
                mc_align.metacell_df.loc[idx0, ['X', 'Y']].values,
                mc_align.metacell_df.loc[idx1, ['X', 'Y']].values,
                mc_align.metacell_df.loc[idx2, ['X', 'Y']].values,
            ])
            mapped_coords = np.array([
                outputDF.loc[align_idx_to_output_idx[idx0], [mapped_x_col, mapped_y_col]].values,
                outputDF.loc[align_idx_to_output_idx[idx1], [mapped_x_col, mapped_y_col]].values,
                outputDF.loc[align_idx_to_output_idx[idx2], [mapped_x_col, mapped_y_col]].values,
            ])
        except Exception:
            continue

        sign_before = np.sign(_signed_area(*aligned_coords))
        sign_after  = np.sign(_signed_area(*mapped_coords))
        is_flipped = (
            (sign_before != sign_after)
            and (sign_before != 0)
            and (sign_after != 0)
        )

        if not is_same_type:
            sign_flips.append(is_flipped)

            # --- always count incident triangles ---
            for nid in [idx0, idx1, idx2]:
                if nid in node_tri_counts:
                    node_tri_counts[nid] += 1
                    if is_flipped:
                        node_flip_counts[nid] += 1

            # --- legacy behavior when node_local=False ---
            if (not node_local) and is_flipped:
                for nid in [idx0, idx1, idx2]:
                    node_in_violating_triangle[nid] = True


    # --- node-local evaluation ---
    if node_local:
        for nid in node_tri_counts:
            n_tri = node_tri_counts[nid]
            n_flip = node_flip_counts[nid]

            if n_tri == 0:
                node_in_violating_triangle[nid] = False
                continue

            frac = n_flip / n_tri
            node_in_violating_triangle[nid] = (
                (n_flip >= min_flips) and (frac >= majority_threshold)
            )

    outputDF['in_violating_triangle'] = (
        outputDF[aligned_id_col].map(node_in_violating_triangle).fillna(False)
    )

    stats = {
        'total_triangles': len(triangles),
        'triangles_with_all_matched': tri_with_matched,
        'triangles_processed': tri_processed,
        'triangles_same_type_skipped': tri_same_type_skipped,
        'triangles_flipped': int(np.sum(sign_flips)) if len(sign_flips) else 0,
        'percent_flipped': (
            100.0 * np.sum(sign_flips) / len(sign_flips)
            if len(sign_flips) else 0.0
        ),
        'nodes_in_violating_triangles': int(outputDF['in_violating_triangle'].sum()),
        'percent_nodes_violating': 100.0*outputDF['in_violating_triangle'].mean(),
    }

    if verbose:
        print(stats)

    return outputDF, stats
