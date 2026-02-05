"""
Helpers for initializing Gurobi MIP starts.

This module provides optional heuristics to generate a good initial feasible
solution (MIP start) for the binary match variables in SAME.

Supported methods
-----------------
- greedy: global greedy over allowed edges, enforcing one-to-one assignment.
- hungarian: Hungarian assignment with a per-row dummy column for unmatched.

Notes
-----
These initializers are most appropriate when max_matches == 1 (one match per
aligned point). They can improve time-to-good-incumbent substantially.
"""

from __future__ import annotations

from typing import Iterable, List, Optional, Sequence, Set, Tuple

import numpy as np


def _hungarian_linear_sum_assignment():
    """
    Import scipy.optimize.linear_sum_assignment at runtime.

    Returns
    -------
    callable or None
        The linear_sum_assignment function if available, else None.
    """
    try:
        import importlib
    except Exception:
        return None

    try:
        optimize_mod = importlib.import_module("scipy.optimize")
        return getattr(optimize_mod, "linear_sum_assignment")
    except Exception:
        return None


def compute_mip_start_pairs(
    *,
    valid_pairs: Sequence[Tuple[int, int]],
    costs: Sequence[float],
    n_aligned: int,
    n_ref: int,
    aligned_sizes: np.ndarray,
    no_match_penalty: float,
    max_matches: int,
    init_method: str,
    init_big_m: float = 1e9,
    init_hungarian_max_n: int = 2000,
    verbose: bool = True,
) -> Tuple[List[Tuple[int, int, int]], Set[int]]:
    """
    Compute a MIP start as (chosen match edges, chosen unmatched aligned indices).

    Parameters
    ----------
    valid_pairs
        List of allowed (aligned_i, ref_j) pairs corresponding 1:1 to `costs`.
    costs
        Per-pair costs used in the objective, aligned with `valid_pairs`.
    n_aligned, n_ref
        Number of aligned and reference points in the current window.
    aligned_sizes
        Array of aligned metacell sizes, shape (n_aligned,).
    no_match_penalty
        Penalty coefficient used for no_match_vars in the objective.
    max_matches
        Maximum matches per aligned point (SAME parameter).
    init_method
        'greedy' or 'hungarian'.
    init_big_m
        Large cost used to represent forbidden edges in Hungarian.
    init_hungarian_max_n
        Safety limit for Hungarian initialization to avoid huge dense matrices.
    verbose
        If True, prints brief status.

    Returns
    -------
    chosen_pairs
        List of (aligned_i, ref_j, var_idx) where var_idx indexes into valid_pairs/costs.
    chosen_unmatched
        Set of aligned indices assigned to "unmatched".
    """
    method = str(init_method).lower()
    if method not in {"greedy", "hungarian"}:
        raise ValueError(f"Unknown init_method={init_method!r}. Use 'greedy' or 'hungarian'.")

    if method == "hungarian" and max_matches != 1:
        raise ValueError("init_method='hungarian' requires max_matches == 1.")

    if len(valid_pairs) != len(costs):
        raise ValueError("valid_pairs and costs must have the same length.")

    costs_arr = np.asarray(costs, dtype=float)
    unmatched_cost = float(no_match_penalty) * np.asarray(aligned_sizes, dtype=float)

    chosen_pairs: List[Tuple[int, int, int]] = []
    chosen_unmatched: Set[int] = set()

    if method == "greedy":
        # Global greedy over all candidate edges sorted by cost
        edges = [(float(costs_arr[idx]), i, j, idx) for idx, (i, j) in enumerate(valid_pairs)]
        edges.sort(key=lambda t: t[0])

        used_aligned: Set[int] = set()
        used_ref: Set[int] = set()

        # Only try to match i if its best available edge is better than leaving unmatched
        best_cost_per_i = np.full(n_aligned, np.inf, dtype=float)
        for cost, i, _j, _idx in edges:
            if cost < best_cost_per_i[i]:
                best_cost_per_i[i] = cost
        prefer_match = best_cost_per_i < unmatched_cost

        for cost, i, j, var_idx in edges:
            if i in used_aligned or j in used_ref:
                continue
            if not prefer_match[i]:
                continue
            chosen_pairs.append((i, j, var_idx))
            used_aligned.add(i)
            used_ref.add(j)

        chosen_unmatched = set(range(n_aligned)) - used_aligned

    elif method == "hungarian":
        if (n_aligned + n_ref) > int(init_hungarian_max_n):
            if verbose:
                print(
                    f"Skipping Hungarian init: n_aligned+n_ref={n_aligned+n_ref} > "
                    f"init_hungarian_max_n={init_hungarian_max_n}"
                )
            return [], set()

        linear_sum_assignment = _hungarian_linear_sum_assignment()
        if linear_sum_assignment is None:
            if verbose:
                print("Skipping Hungarian init: scipy.optimize.linear_sum_assignment not available")
            return [], set()

        # Dense assignment with per-row dummy for unmatched
        cost_mat = np.full((n_aligned, n_ref + n_aligned), float(init_big_m), dtype=float)
        for idx, (i, j) in enumerate(valid_pairs):
            cost_mat[i, j] = float(costs_arr[idx])
        for i in range(n_aligned):
            cost_mat[i, n_ref + i] = float(unmatched_cost[i])

        row_ind, col_ind = linear_sum_assignment(cost_mat)
        used_ref: Set[int] = set()

        # Map (i, j) -> variable index
        pair_to_var_idx = {(i, j): idx for idx, (i, j) in enumerate(valid_pairs)}

        for i, col in zip(row_ind, col_ind):
            i = int(i)
            col = int(col)
            if col < n_ref and cost_mat[i, col] < float(init_big_m) * 0.5:
                j = col
                if j in used_ref:
                    continue
                used_ref.add(j)
                var_idx = pair_to_var_idx.get((i, j))
                if var_idx is not None:
                    chosen_pairs.append((i, j, int(var_idx)))
            else:
                chosen_unmatched.add(i)

    return chosen_pairs, chosen_unmatched


def apply_mip_start(
    *,
    x_vars,
    no_match_vars,
    valid_pairs: Sequence[Tuple[int, int]],
    costs: Sequence[float],
    n_aligned: int,
    n_ref: int,
    aligned_sizes: np.ndarray,
    no_match_penalty: float,
    max_matches: int,
    init_method: Optional[str],
    init_big_m: float = 1e9,
    init_hungarian_max_n: int = 2000,
    verbose: bool = True,
) -> None:
    """
    Apply a MIP start to Gurobi variables x and no_match_vars.

    Parameters
    ----------
    x_vars
        Gurobi variable container (e.g., tupledict) for match variables indexed
        by integer var_idx (0..len(valid_pairs)-1).
    no_match_vars
        Gurobi variable container for unmatched indicators indexed by aligned i.
    valid_pairs, costs, n_aligned, n_ref, aligned_sizes, no_match_penalty, max_matches
        See compute_mip_start_pairs.
    init_method
        None (skip) or 'greedy'/'hungarian'.
    """
    if init_method is None:
        return

    chosen_pairs, chosen_unmatched = compute_mip_start_pairs(
        valid_pairs=valid_pairs,
        costs=costs,
        n_aligned=n_aligned,
        n_ref=n_ref,
        aligned_sizes=aligned_sizes,
        no_match_penalty=no_match_penalty,
        max_matches=max_matches,
        init_method=init_method,
        init_big_m=init_big_m,
        init_hungarian_max_n=init_hungarian_max_n,
        verbose=verbose,
    )

    if not chosen_pairs and not chosen_unmatched:
        return

    for var_idx in range(len(valid_pairs)):
        x_vars[var_idx].Start = 0.0
    for i in range(n_aligned):
        no_match_vars[i].Start = 1.0 if i in chosen_unmatched else 0.0
    for i, _j, var_idx in chosen_pairs:
        x_vars[var_idx].Start = 1.0
        no_match_vars[int(i)].Start = 0.0

    if verbose:
        method = str(init_method).lower()
        print(
            f"Initialized MIP start ({method}): "
            f"{len(chosen_pairs)} matches, {len(chosen_unmatched)} unmatched"
        )


