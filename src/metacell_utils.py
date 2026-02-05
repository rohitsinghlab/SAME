"""
Metacell creation utilities for graph simplification.

This module provides functions to simplify spatial graphs by collapsing
same-type triangles into metacells, reducing the problem size while
preserving boundary structure.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd
from scipy.spatial import Delaunay

try:
    get_ipython()
    from tqdm.notebook import tqdm
except:
    from tqdm import tqdm


@dataclass
class MetaCell:
    """
    Container for metacell collapse results + reproducibility metadata.

    Key conventions:
    - `original_delaunay_*` triangles refer to vertices in the *original* input.
    - `metacell_delaunay` triangles refer to vertices in the returned `metacell_df`
      and use `metacell_idx_col` (typically 0..n_metacells-1).
    - `metacell_df["members"]` stores a list of *original IDs* (values from
      `original_idx_col`) that were merged into each metacell.
    """

    # Inputs / metadata
    original_df: pd.DataFrame
    params: Dict[str, Any]
    x_col: str
    y_col: str
    cell_type_col: str
    original_idx_col: str
    metacell_idx_col: str

    # Triangulation (filtered) in original-ID space
    # Shape: (n_triangles, 3); vertex values are from `original_df[original_idx_col]`.
    original_delaunay: np.ndarray

    # Outputs
    metacell_df: pd.DataFrame
    metacell_delaunay: np.ndarray

    def metacell_members(self, metacell_idx: int) -> List[Any]:
        """Return list of original IDs that form this metacell."""
        return list(self.metacell_df.iloc[int(metacell_idx)]["members"])

    def original_delaunay_to_row_indices(
        self,
        triangles: Optional[np.ndarray] = None,
        *,
        on_missing: str = "drop",
    ) -> np.ndarray:
        """
        Convert original-ID-space triangles to row indices (0..n_original-1).

        Parameters
        ----------
        triangles : np.ndarray, optional
            Triangle array in original-ID space. If None, uses self.original_delaunay.
        on_missing : {"drop", "error"}, default="drop"
            What to do if a triangle references an original ID not present in original_df.
            - "drop": drop those triangles
            - "error": raise KeyError
        """
        tri = self.original_delaunay if triangles is None else np.asarray(triangles)
        if tri.size == 0:
            return np.array([], dtype=int).reshape(0, 3)
        if tri.ndim != 2 or tri.shape[1] != 3:
            raise ValueError(f"triangles must have shape (n, 3); got {tri.shape}")

        ids = self.original_df[self.original_idx_col].to_numpy()
        id_to_pos = {v: i for i, v in enumerate(ids)}

        flat = tri.reshape(-1)
        remapped = np.fromiter((id_to_pos.get(v, -1) for v in flat), dtype=int, count=flat.size)
        remapped = remapped.reshape(tri.shape)
        if (remapped < 0).any():
            if on_missing == "error":
                missing = set(flat[remapped.reshape(-1) < 0].tolist())
                raise KeyError(f"Found triangle vertices not in original_df[{self.original_idx_col}]: {list(missing)[:10]}")
            remapped = remapped[(remapped >= 0).all(axis=1)]
        return remapped

    def original_delaunay_to_pos(
        self,
        triangles: Optional[np.ndarray] = None,
        *,
        on_missing: str = "drop",
    ) -> np.ndarray:
        """
        Backwards-compatible alias for `original_delaunay_to_row_indices`.

        NOTE: This returns *row indices*, not X/Y coordinates.
        """
        return self.original_delaunay_to_row_indices(triangles=triangles, on_missing=on_missing)

    def original_delaunay_to_xy(
        self,
        triangles: Optional[np.ndarray] = None,
        *,
        on_missing: str = "drop",
    ) -> np.ndarray:
        """
        Convert original-ID-space triangles to their X/Y coordinates.

        Returns
        -------
        np.ndarray
            Shape (n_triangles, 3, 2), where the last dimension is (x, y).
        """
        tri_pos = self.original_delaunay_to_row_indices(triangles=triangles, on_missing=on_missing)
        if tri_pos.size == 0:
            return np.array([], dtype=float).reshape(0, 3, 2)
        coords = self.original_df[[self.x_col, self.y_col]].to_numpy(dtype=float, copy=False)
        return coords[tri_pos]

    def metacell_delaunay_to_xy(self) -> np.ndarray:
        """
        Convert metacell triangles (row-index space) to their X/Y coordinates.

        Returns
        -------
        np.ndarray
            Shape (n_triangles, 3, 2), where the last dimension is (x, y).
        """
        tri = np.asarray(self.metacell_delaunay)
        if tri.size == 0:
            return np.array([], dtype=float).reshape(0, 3, 2)
        coords = self.metacell_df[[self.x_col, self.y_col]].to_numpy(dtype=float, copy=False)
        return coords[tri.astype(int, copy=False)]

    def to_summary_dict(self) -> Dict[str, Any]:
        """Small JSON-serializable-ish summary (avoids embedding full dataframes)."""
        return {
            "n_original": int(len(self.original_df)),
            "n_metacells": int(len(self.metacell_df)),
            "params": dict(self.params),
            "x_col": self.x_col,
            "y_col": self.y_col,
            "cell_type_col": self.cell_type_col,
            "original_idx_col": self.original_idx_col,
            "metacell_idx_col": self.metacell_idx_col,
            "n_original_triangles": int(getattr(self.original_delaunay, "shape", [0])[0]),
            "n_metacell_triangles": int(getattr(self.metacell_delaunay, "shape", [0])[0]),
        }


def greedy_triangle_collapse(aligned_df, max_metacell_size=3, max_iterations=1000,r_max=None, min_angle_deg=10, use_alpha_shape=False,
                             alpha=0.05,
                             *,
                             original_idx_col: str = "Cell_Num_Old",
                             metacell_idx_col: str = "metacell_id",
                             x_col: str = "X",
                             y_col: str = "Y",
                             cell_type_col: str = "cell_type",
                             return_object: bool = False):
    """
    Iteratively collapse same-type triangles into metacells.

    This function simplifies a spatial graph by merging cells that form
    homogeneous triangles (all vertices same cell type). The result is a
    coarser graph with metacells that preserve boundary structure while
    reducing the number of nodes.

    Parameters
    ----------
    aligned_df : pd.DataFrame
        Input dataframe with spatial cells.
        Required columns: x_col, y_col, cell_type_col, original_idx_col
        Optional numeric columns will be averaged (e.g., cell type proportions)
        Note: ID columns (Cell_Num, Cell_Num_Old, etc.) are NOT averaged;
              metacells get new sequential IDs in `metacell_idx_col` (0, 1, 2, ...)
    max_metacell_size : int, default=10
        Maximum number of original cells in a metacell
    max_iterations : int, default=1000
        Maximum number of collapse iterations
    r_max : float, optional
        Maximum edge length - triangles with any edge > r_max are removed
    min_angle_deg : float, default=10
        Minimum angle in degrees - triangles with smaller angles are degenerate
    use_alpha_shape : bool, default=False
        If True, filter triangles to only those within alpha shape
    alpha : float, default=0.05
        Alpha parameter for alpha shape (smaller = tighter boundary)
        Only used if use_alpha_shape=True

    Returns
    -------
    If return_object is False (default):
        metacell_df : pd.DataFrame
            Simplified graph where each row is a metacell.
            Columns: x_col, y_col, cell_type_col, size, members (list of original IDs),
            metacell_idx_col (new sequential IDs), plus averaged numeric columns.
        metacell_delaunay : np.ndarray
            Filtered Delaunay triangulation on metacells, shape (n_triangles, 3)

    If return_object is True:
        MetaCell
            Object containing original_df, original_delaunay (filtered), metacell_df,
            metacell_delaunay, and all parameters used.

    Examples
    --------
    >>> # Basic usage with edge filtering
    >>> metacell_df, delaunay = greedy_triangle_collapse(
    ...     aligned_df,
    ...     max_metacell_size=10,
    ...     r_max=500
    ... )

    >>> # With alpha shape filtering
    >>> metacell_df, delaunay = greedy_triangle_collapse(
    ...     aligned_df,
    ...     max_metacell_size=10,
    ...     r_max=500,
    ...     min_angle_deg=15,
    ...     use_alpha_shape=True,
    ...     alpha=0.05
    ... )
    """

    def compute_angle(p1, p2, p3):
        """Compute angle at p2 in triangle (p1, p2, p3) in degrees."""
        v1 = p1 - p2
        v2 = p3 - p2
        cos_angle = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
        cos_angle = np.clip(cos_angle, -1, 1)  # Numerical stability
        return np.degrees(np.arccos(cos_angle))

    def is_triangle_valid(tri_coords, r_max, min_angle_deg):
        """Check if triangle meets edge length and angle constraints."""
        p1, p2, p3 = tri_coords

        # Check edge lengths
        if r_max is not None:
            edge1 = np.linalg.norm(p2 - p1)
            edge2 = np.linalg.norm(p3 - p2)
            edge3 = np.linalg.norm(p1 - p3)
            if max(edge1, edge2, edge3) > r_max:
                return False

        # Check minimum angle
        if min_angle_deg is not None:
            angle1 = compute_angle(p2, p1, p3)
            angle2 = compute_angle(p1, p2, p3)
            angle3 = compute_angle(p1, p3, p2)
            if min(angle1, angle2, angle3) < min_angle_deg:
                return False

        return True

    def filter_triangles(coords, triangles, r_max, min_angle_deg, use_alpha_shape, alpha):
        """Filter triangles by geometry constraints."""
        if use_alpha_shape:
            try:
                from alphashape import alphashape
                from shapely.geometry import Polygon

                # Compute alpha shape
                points = [tuple(coord) for coord in coords]
                alpha_shape = alphashape(points, alpha)
            except ImportError:
                print("Warning: alphashape not available, skipping alpha shape filtering")
                use_alpha_shape = False

        valid_triangles = []
        for tri in triangles:
            tri_coords = coords[tri]

            # Check edge/angle constraints
            if not is_triangle_valid(tri_coords, r_max, min_angle_deg):
                continue

            # Check alpha shape containment
            if use_alpha_shape:
                tri_polygon = Polygon(tri_coords)
                if not alpha_shape.contains(tri_polygon):
                    continue

            valid_triangles.append(tri)

        return np.array(valid_triangles) if valid_triangles else np.array([]).reshape(0, 3)

    # ---- Validate and normalize input ----
    required = [x_col, y_col, cell_type_col, original_idx_col]
    missing = [c for c in required if c not in aligned_df.columns]
    if missing:
        raise ValueError(f"Input dataframe missing required columns: {missing}")

    aligned_df = aligned_df.copy()
    if aligned_df[original_idx_col].duplicated().any():
        dups = aligned_df.loc[aligned_df[original_idx_col].duplicated(), original_idx_col].head(5).tolist()
        raise ValueError(
            f"'{original_idx_col}' must be unique per original cell. "
            f"Found duplicates (examples): {dups}"
        )

    # Index the original df by original_idx_col for fast member lookups
    aligned_df_indexed = aligned_df.set_index(original_idx_col, drop=False)

    # ---- Compute filtered original Delaunay (pre-collapse) ----
    original_coords = aligned_df[[x_col, y_col]].to_numpy()
    if len(original_coords) >= 4:
        original_delaunay_raw = Delaunay(original_coords).simplices
        original_delaunay_pos = filter_triangles(
            original_coords,
            original_delaunay_raw,
            r_max,
            min_angle_deg,
            use_alpha_shape,
            alpha,
        )
    else:
        original_delaunay_pos = np.array([], dtype=int).reshape(0, 3)

    # Map original triangle vertices from positional indices -> original IDs
    original_ids_by_pos = aligned_df[original_idx_col].to_numpy()
    if original_delaunay_pos.size == 0:
        original_delaunay = np.array([], dtype=original_ids_by_pos.dtype).reshape(0, 3)
    else:
        original_delaunay = original_ids_by_pos[original_delaunay_pos]

    # Identify ID columns that should NOT be averaged
    id_columns = ['Cell_Num', 'Cell_Num_Old', 'cell_id', 'Cell_ID', 'ID', 'id']
    id_cols_present = [col for col in aligned_df.columns if col in id_columns]
    # Always exclude the user-specified ID columns too
    if original_idx_col not in id_cols_present:
        id_cols_present.append(original_idx_col)
    if metacell_idx_col in aligned_df.columns and metacell_idx_col not in id_cols_present:
        id_cols_present.append(metacell_idx_col)

    # Initialize: each cell is a metacell of size 1
    metacells = []
    for _, row in aligned_df.iterrows():
        orig_id = row[original_idx_col]
        metacells.append({
            x_col: row[x_col],
            y_col: row[y_col],
            cell_type_col: row[cell_type_col],
            'size': 1,
            # members are ORIGINAL IDs (not dataframe indices)
            'members': [orig_id],
            **{col: row[col] for col in aligned_df.columns
               if col not in [x_col, y_col, cell_type_col] + id_cols_present}
        })

    metacell_df = pd.DataFrame(metacells)

    # Add new sequential metacell IDs
    metacell_df[metacell_idx_col] = range(len(metacell_df))

    print(f"Starting greedy triangle collapse:")
    print(f"  Initial cells: {len(aligned_df)}")
    print(f"  Max metacell size: {max_metacell_size}")

    pbar = tqdm(total=max_iterations, desc="Collapsing triangles",
                bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]')

    for iteration in range(max_iterations):
        # Compute Delaunay on current metacells
        coords = metacell_df[[x_col, y_col]].values
        if len(coords) < 4:
            pbar.close()
            print(f"\nToo few points ({len(coords)}) for Delaunay at iteration {iteration}")
            break

        delaunay = Delaunay(coords)
        triangles_raw = delaunay.simplices

        # Filter triangles by geometry
        triangles = filter_triangles(coords, triangles_raw, r_max, min_angle_deg,
                                    use_alpha_shape, alpha)

        if iteration == 0:
            print(f"\nInitial triangle filtering: {len(triangles_raw)} → {len(triangles)} triangles")
            if r_max:
                print(f"  Max edge length: {r_max}")
            if min_angle_deg:
                print(f"  Min angle: {min_angle_deg}°")
            if use_alpha_shape:
                print(f"  Alpha shape: α={alpha}")

        if len(triangles) == 0:
            pbar.close()
            print(f"\nNo valid triangles remaining at iteration {iteration}")
            break

        # Find collapsible triangles
        candidates = []
        for tri in triangles:
            a, b, c = tri

            # Check if all same type
            if not (metacell_df.iloc[a][cell_type_col] ==
                    metacell_df.iloc[b][cell_type_col] ==
                    metacell_df.iloc[c][cell_type_col]):
                continue

            # Check if merged size would exceed limit
            total_size = (metacell_df.iloc[a]['size'] +
                         metacell_df.iloc[b]['size'] +
                         metacell_df.iloc[c]['size'])
            if total_size > max_metacell_size:
                continue

            # Compute priority (lower = collapse first)
            # Prefer small triangles (less distortion)
            perimeter = (
                np.linalg.norm(coords[a] - coords[b]) +
                np.linalg.norm(coords[b] - coords[c]) +
                np.linalg.norm(coords[c] - coords[a])
            )

            candidates.append({
                'priority': perimeter,
                'indices': [a, b, c],
                'total_size': total_size
            })

        if not candidates:
            pbar.close()
            print(f"\nNo more collapsible triangles at iteration {iteration}")
            break

        # BATCH MODE: Pick all non-overlapping candidates
        candidates.sort(key=lambda x: x['priority'])

        to_collapse_batch = []
        used_vertices = set()

        for candidate in candidates:
            a, b, c = candidate['indices']
            # Only add if vertices not already used
            if a not in used_vertices and b not in used_vertices and c not in used_vertices:
                to_collapse_batch.append(candidate)
                used_vertices.update([a, b, c])

        # Collapse all selected triangles
        merged_metacells = []
        vertices_to_remove = []

        for to_collapse in to_collapse_batch:
            a, b, c = to_collapse['indices']
            vertices_to_remove.extend([a, b, c])

            # Get all original member cells
            all_members = (metacell_df.iloc[a]['members'] +
                          metacell_df.iloc[b]['members'] +
                          metacell_df.iloc[c]['members'])

            # Compute true centroid from original cells
            member_coords = aligned_df_indexed.loc[all_members, [x_col, y_col]]
            true_centroid_x = member_coords[x_col].mean()
            true_centroid_y = member_coords[y_col].mean()

            # Create merged metacell
            merged = {
                x_col: true_centroid_x,
                y_col: true_centroid_y,
                cell_type_col: metacell_df.iloc[a][cell_type_col],
                'size': to_collapse['total_size'],
                'members': all_members
            }

            # Average other columns (e.g., cell type proportions)
            # For numeric columns, compute true average from original cells
            # For non-numeric, take first value
            # SKIP ID columns - they will get new sequential IDs after merge
            for col in metacell_df.columns:
                if col not in [x_col, y_col, cell_type_col, 'size', 'members', metacell_idx_col] + id_cols_present:
                    # Check if column is numeric
                    if pd.api.types.is_numeric_dtype(metacell_df[col]):
                        # True average from original cells if column exists in original
                        if col in aligned_df.columns:
                            merged[col] = aligned_df_indexed.loc[all_members, col].mean()
                        else:
                            # Column doesn't exist in original, weighted average of metacell values
                            sizes = [metacell_df.iloc[idx]['size'] for idx in [a, b, c]]
                            values = [metacell_df.iloc[idx][col] for idx in [a, b, c]]
                            merged[col] = np.average(values, weights=sizes)
                    else:
                        # For non-numeric, take first value (or could do mode)
                        merged[col] = metacell_df.iloc[a][col]

            merged_metacells.append(merged)

        # Remove old vertices and add all new ones
        metacell_df = metacell_df.drop(vertices_to_remove).reset_index(drop=True)
        if merged_metacells:
            metacell_df = pd.concat([metacell_df, pd.DataFrame(merged_metacells)],
                                    ignore_index=True)

        # Reassign sequential metacell IDs to all metacells
        metacell_df[metacell_idx_col] = range(len(metacell_df))

        # Update progress bar with batch info
        pbar.update(1)
        pbar.set_postfix({
            'metacells': len(metacell_df),
            'reduction': f"{100*(1-len(metacell_df)/len(aligned_df)):.1f}%",
            'batch': len(to_collapse_batch)
        })

    # Close progress bar
    pbar.close()

    # Final filtered Delaunay on result
    final_coords = metacell_df[[x_col, y_col]].values
    if len(final_coords) >= 4:
        final_delaunay_raw = Delaunay(final_coords).simplices
        final_delaunay = filter_triangles(final_coords, final_delaunay_raw,
                                          r_max, min_angle_deg,
                                          use_alpha_shape, alpha)
    else:
        final_delaunay = np.array([]).reshape(0, 3)

    print(f"\nCollapse complete:")
    print(f"  Original cells: {len(aligned_df)}")
    print(f"  Final metacells: {len(metacell_df)}")
    print(f"  Reduction: {100*(1 - len(metacell_df)/len(aligned_df)):.1f}%")
    print(f"  Avg metacell size: {metacell_df['size'].mean():.1f}")
    print(f"  Max metacell size: {metacell_df['size'].max()}")
    print(f"  Final triangles: {len(final_delaunay)}")
    if id_cols_present:
        print(f"  Note: ID columns {id_cols_present} were excluded; metacells assigned new Cell_Num 0-{len(metacell_df)-1}")

    if return_object:
        params = {
            "max_metacell_size": max_metacell_size,
            "max_iterations": max_iterations,
            "r_max": r_max,
            "min_angle_deg": min_angle_deg,
            "use_alpha_shape": use_alpha_shape,
            "alpha": alpha,
        }
        return MetaCell(
            original_df=aligned_df,
            params=params,
            x_col=x_col,
            y_col=y_col,
            cell_type_col=cell_type_col,
            original_idx_col=original_idx_col,
            metacell_idx_col=metacell_idx_col,
            original_delaunay=original_delaunay,
            metacell_df=metacell_df,
            metacell_delaunay=final_delaunay,
        )

    return metacell_df, final_delaunay


def unpack_metacell_matches(
    metacell_matches,
    metacell_aligned_df,
    metacell_ref_df,
    aligned_df=None,
    ref_df=None,
    strategy="distribute",
    aligned_original_idx_col: Optional[str] = None,
    ref_original_idx_col: Optional[str] = None,
    x_col: str = "X",
    y_col: str = "Y",
):
    """
    Unpack metacell-level matches to individual cell matches.

    Handles two cases:
    1. Only aligned has metacells: ref_df is individual cells (simple unpacking)
    2. Both have metacells: both need unpacking using nearest neighbor

    Parameters
    ----------
    metacell_matches : pd.DataFrame
        Matches at metacell level (output from run_same on metacells)
        Must have columns: aligned_idx, ref_idx
    metacell_aligned_df : pd.DataFrame
        Aligned metacell dataframe (output from greedy_triangle_collapse)
        Must have column: members (list of original cell indices)
    metacell_ref_df : pd.DataFrame
        Reference dataframe - can be either:
        - Individual cells (no 'members' column): simple case
        - Metacells (has 'members' column): requires unpacking both sides
    aligned_df : pd.DataFrame, optional
        Original aligned cells with X, Y coordinates
        Required if strategy='nearest' or if ref has metacells
    ref_df : pd.DataFrame, optional
        Original reference cells with X, Y coordinates
        Required if metacell_ref_df has metacells
    aligned_original_idx_col : str, optional
        If provided, interpret members in metacell_aligned_df as values from this
        column and use aligned_df.set_index(aligned_original_idx_col) to look up
        coordinates. If not provided, members are assumed to be valid aligned_df
        index values (legacy behavior).
    ref_original_idx_col : str, optional
        Analogous to aligned_original_idx_col, for ref_df lookups when ref has metacells.
    x_col, y_col : str
        Coordinate column names in aligned_df/ref_df.
    strategy : str, default='distribute'
        How to distribute matches:
        - 'distribute': all aligned members → same ref (only valid if ref is individual cells)
        - 'nearest': each aligned member → nearest ref member (required if both are metacells)

    Returns
    -------
    individual_matches : pd.DataFrame
        Matches at individual cell level
        Columns: aligned_idx, ref_idx

    Examples
    --------
    # Case 1: Only aligned has metacells
    >>> metacell_aligned, _ = greedy_triangle_collapse(aligned_df)
    >>> metacell_matches, _ = run_same(metacell_aligned, ref_df, ...)
    >>> individual_matches = unpack_metacell_matches(
    ...     metacell_matches, metacell_aligned, ref_df
    ... )

    # Case 2: Both have metacells
    >>> metacell_aligned, _ = greedy_triangle_collapse(aligned_df)
    >>> metacell_ref, _ = greedy_triangle_collapse(ref_df)
    >>> metacell_matches, _ = run_same(metacell_aligned, metacell_ref, ...)
    >>> individual_matches = unpack_metacell_matches(
    ...     metacell_matches, metacell_aligned, metacell_ref,
    ...     aligned_df=aligned_df, ref_df=ref_df, strategy='nearest'
    ... )
    """
    from scipy.spatial.distance import cdist

    aligned_df_indexed = None
    ref_df_indexed = None
    if aligned_df is not None and aligned_original_idx_col is not None:
        if aligned_original_idx_col not in aligned_df.columns:
            raise ValueError(f"aligned_df missing aligned_original_idx_col='{aligned_original_idx_col}'")
        aligned_df_indexed = aligned_df.set_index(aligned_original_idx_col, drop=False)
    if ref_df is not None and ref_original_idx_col is not None:
        if ref_original_idx_col not in ref_df.columns:
            raise ValueError(f"ref_df missing ref_original_idx_col='{ref_original_idx_col}'")
        ref_df_indexed = ref_df.set_index(ref_original_idx_col, drop=False)

    # Detect if ref has metacells
    ref_has_metacells = ('members' in metacell_ref_df.columns and
                        metacell_ref_df['members'].apply(lambda x: isinstance(x, list)).any())

    # Validate parameters
    if ref_has_metacells:
        if strategy == 'nearest' and (aligned_df is None or ref_df is None):
            raise ValueError(
                "When ref has metacells and strategy='nearest', must provide both aligned_df and ref_df "
                "for nearest neighbor unpacking."
            )

    if strategy == 'nearest' and aligned_df is None:
        raise ValueError("strategy='nearest' requires aligned_df parameter")

    individual_matches = []

    for _, match_row in metacell_matches.iterrows():
        metacell_aligned_idx = match_row['Aligned_metacell_id']
        metacell_ref_idx = match_row['Ref_metacell_id']

        # Get aligned members
        aligned_members = metacell_aligned_df.iloc[metacell_aligned_idx]['members']

        if not ref_has_metacells:
            # Simple case: ref is individual cells
            if strategy == 'distribute':
                # All aligned members → same ref point
                for member in aligned_members:
                    individual_matches.append({
                        'Aligned_cell_id': member,
                        'Ref_cell_id': metacell_ref_idx,
                    })

            elif strategy == 'nearest':
                # Each aligned member → same ref (since ref is single point)
                # 'nearest' doesn't add value here, but allow it for consistency
                for member in aligned_members:
                    individual_matches.append({
                        'Aligned_cell_id': member,
                        'Ref_cell_id': metacell_ref_idx,
                    })

        else:
            # Complex case: both sides are metacells
            ref_members = metacell_ref_df.iloc[metacell_ref_idx]['members']

            if strategy == 'distribute':
                # Round-robin distribute: cycle through ref members
                # Each ref cell gets used once before any is reused (like dealing cards)
                # e.g., aligned [a1,a2,a3,a4,a5] → ref [r1,r2] gives: a1→r1, a2→r2, a3→r1, a4→r2, a5→r1
                n_ref = len(ref_members)
                for i, aligned_member in enumerate(aligned_members):
                    ref_idx = i % n_ref
                    individual_matches.append({
                        'Aligned_cell_id': aligned_member,
                        'Ref_cell_id': ref_members[ref_idx],
                    })

            elif strategy == 'nearest':
                # Hungarian assignment: optimal matching minimizing total distance
                # When N aligned > M ref, replicate ref cells for balanced assignment
                from scipy.optimize import linear_sum_assignment
                
                if aligned_df is None or ref_df is None:
                    raise ValueError("strategy='nearest' with metacell ref requires aligned_df and ref_df")

                if aligned_df_indexed is not None:
                    aligned_coords = aligned_df_indexed.loc[aligned_members, [x_col, y_col]].values
                else:
                    aligned_coords = aligned_df.loc[aligned_members, [x_col, y_col]].values

                if ref_df_indexed is not None:
                    ref_coords = ref_df_indexed.loc[ref_members, [x_col, y_col]].values
                else:
                    ref_coords = ref_df.loc[ref_members, [x_col, y_col]].values

                n_aligned = len(aligned_members)
                n_ref = len(ref_members)

                # Compute pairwise distances
                distances = cdist(aligned_coords, ref_coords)  # shape: (n_aligned, n_ref)

                if n_aligned <= n_ref:
                    # More ref than aligned: simple Hungarian, some ref unused
                    row_ind, col_ind = linear_sum_assignment(distances)
                    for aligned_idx, ref_idx in zip(row_ind, col_ind):
                        individual_matches.append({
                            'Aligned_cell_id': aligned_members[aligned_idx],
                            'Ref_cell_id': ref_members[ref_idx],
                        })
                else:
                    # More aligned than ref: replicate ref cells for balanced usage
                    # Each ref cell can be used ceil(n_aligned / n_ref) times
                    n_copies = int(np.ceil(n_aligned / n_ref))
                    
                    # Tile the distance matrix: replicate ref columns
                    # Shape becomes (n_aligned, n_ref * n_copies)
                    tiled_distances = np.tile(distances, (1, n_copies))
                    
                    # Run Hungarian on tiled matrix
                    row_ind, col_ind = linear_sum_assignment(tiled_distances)
                    
                    # Map back: col_ind % n_ref gives original ref index
                    for aligned_idx, tiled_ref_idx in zip(row_ind, col_ind):
                        original_ref_idx = tiled_ref_idx % n_ref
                        individual_matches.append({
                            'Aligned_cell_id': aligned_members[aligned_idx],
                            'Ref_cell_id': ref_members[original_ref_idx],
                        })

            else:
                raise ValueError(f"Unknown strategy: {strategy}")

    return pd.DataFrame(individual_matches)
