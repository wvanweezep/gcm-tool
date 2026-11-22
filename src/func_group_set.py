import re

from func_group import FunctionalGroup
from rdkit import Chem


type FunctionalGroupSet = dict[int, dict[str, FunctionalGroup]]


def decompose_mol1(
    mol: Chem.Mol,
    group_set: FunctionalGroupSet,
    strict: bool = False
) -> list[tuple[int, FunctionalGroup]]:
    """
    Decompose a molecule into non-overlapping functional groups.

    Functional groups are matched in descending order, priority,
    and atom count. Once an atom has been assigned to a functional
    group, it cannot be assigned again.

    :param mol: Molecule to decompose.
    :param group_set: Set of functional groups to use.
    :param strict: Require every molecule atom to be assigned.
    :return: List of (match_count, FunctionalGroup) tuples.

    :raises ValueError: If strict decomposition fails.
    """

    if mol is None:
        raise ValueError("Cannot decompose a None molecule")

    # Collect all functional groups
    groups = [
        group
        for order_groups in group_set.values()
        for group in order_groups.values()
    ]

    # Sort groups by order, priority, and atom count
    sorted_groups = sorted(
        groups,
        key=lambda g: (g.order, g.priority, g.num_atoms),
        reverse=True
    )

    # Match groups without reusing atoms
    matched: set[int] = set()
    decomposition: list[tuple[int, FunctionalGroup]] = []

    for group in sorted_groups:
        count = group.match(mol, matched)

        if count > 0:
            decomposition.append((count, group))

    # Validate strict decomposition
    if strict and len(matched) != mol.GetNumAtoms():
        raise ValueError(
            f"Failed to strictly decompose molecule into functional groups:\n"
            f"Matched atoms: {len(matched)}/{mol.GetNumAtoms()}\n"
            f"Decomposition: {decomposition}"
        )

    return decomposition



import numpy as np

from scipy.optimize import milp, Bounds, LinearConstraint
from scipy.sparse import coo_matrix


def decompose_mol(
    mol: Chem.Mol,
    group_set: FunctionalGroupSet,
    strict: bool = False
) -> list[tuple[int, FunctionalGroup]]:
    """
    Decompose a molecule into optimal non-overlapping functional groups.

    Uses lexicographic set-packing optimization:
      1. Maximize occurrences of highest-order groups.
      2. Maximize occurrences of highest-priority groups.
      3. Prefer groups with more atoms.
      4. Continue with lower-ranked groups.

    Each atom can belong to at most one selected group occurrence.

    When strict=True, every molecule atom must be assigned exactly once.

    Args:
        mol: Molecule to decompose.
        group_set: Functional groups organized by order.
        strict: Require complete atom coverage.

    Returns:
        List of (match_count, FunctionalGroup) tuples.

    Raises:
        ValueError: If strict decomposition is impossible.
        RuntimeError: If optimization fails.
    """

    if mol is None:
        raise ValueError("Cannot decompose a None molecule")

    # Collect unique functional groups
    groups = list(dict.fromkeys(
        group
        for order_groups in group_set.values()
        for group in order_groups.values()
    ))

    # Sort groups by their hierarchy
    groups.sort(
        key=lambda g: (g.order, g.priority, g.num_atoms),
        reverse=True
    )

    n_atoms = mol.GetNumAtoms()

    # Handle empty molecules
    if n_atoms == 0:
        return []

    # ---------------------------------------------------------
    # 1. Generate all candidate group occurrences
    # ---------------------------------------------------------

    candidates = []

    for group in groups:
        for match in group.get_matches(mol):
            candidates.append({
                "group": group,
                "atoms": frozenset(match),
                "rank": (group.order, group.priority, group.num_atoms)
            })

    n_candidates = len(candidates)

    if n_candidates == 0:
        if strict:
            raise ValueError(
                "No functional groups matched the molecule."
            )
        return []

    # ---------------------------------------------------------
    # 2. Build atom-to-candidate relationships
    # ---------------------------------------------------------

    atom_candidates = [[] for _ in range(n_atoms)]

    for i, candidate in enumerate(candidates):
        for atom_idx in candidate["atoms"]:
            atom_candidates[atom_idx].append(i)

    # ---------------------------------------------------------
    # 3. Construct set-packing constraints
    # ---------------------------------------------------------

    # Every atom can be assigned at most once.
    # In strict mode, every atom must be assigned exactly once.

    base_rows = []
    base_cols = []
    base_data = []

    for atom_idx, candidate_indices in enumerate(atom_candidates):
        for candidate_idx in candidate_indices:
            base_rows.append(atom_idx)
            base_cols.append(candidate_idx)
            base_data.append(1.0)

    n_base_constraints = n_atoms

    if strict:
        base_lb = np.ones(n_atoms)
    else:
        base_lb = np.full(n_atoms, -np.inf)

    base_ub = np.ones(n_atoms)

    # Additional constraints preserve optimal counts
    # from previously optimized hierarchy levels.
    fixed_constraints = []

    # ---------------------------------------------------------
    # 4. Lexicographic optimization
    # ---------------------------------------------------------

    ranks = sorted(
        {candidate["rank"] for candidate in candidates},
        reverse=True
    )

    result = None

    for rank in ranks:

        # Candidates belonging to the current hierarchy level
        rank_indices = [
            i
            for i, candidate in enumerate(candidates)
            if candidate["rank"] == rank
        ]

        # Construct constraint matrix
        rows = list(base_rows)
        cols = list(base_cols)
        data = list(base_data)

        constraint_lb = list(base_lb)
        constraint_ub = list(base_ub)

        # Preserve optimal solutions from previous levels
        for row_idx, (indices, optimal_count) in enumerate(
            fixed_constraints,
            start=n_base_constraints
        ):
            for candidate_idx in indices:
                rows.append(row_idx)
                cols.append(candidate_idx)
                data.append(1.0)

            constraint_lb.append(optimal_count)
            constraint_ub.append(optimal_count)

        n_constraints = (
            n_base_constraints + len(fixed_constraints)
        )

        A = coo_matrix(
            (data, (rows, cols)),
            shape=(n_constraints, n_candidates)
        ).tocsr()

        constraints = LinearConstraint(
            A,
            lb=np.asarray(constraint_lb),
            ub=np.asarray(constraint_ub)
        )

        # Maximize the number of occurrences at this rank.
        # scipy.milp minimizes, hence the negative objective.
        objective = np.zeros(n_candidates)
        objective[rank_indices] = -1.0

        result = milp(
            c=objective,
            integrality=np.ones(n_candidates),
            bounds=Bounds(
                lb=np.zeros(n_candidates),
                ub=np.ones(n_candidates)
            ),
            constraints=constraints,
            options={"presolve": True}
        )

        if not result.success:
            if strict:
                raise ValueError(
                    "Failed to strictly decompose molecule "
                    "into functional groups.\n"
                    f"Matched candidates: {n_candidates}\n"
                    f"Optimizer message: {result.message}"
                )

            raise RuntimeError(
                f"Set-packing optimization failed: {result.message}"
            )

        # Record the optimal number of occurrences at this rank.
        optimal_count = int(
            round(np.sum(result.x[rank_indices]))
        )

        # Fix this optimal count for subsequent optimization levels.
        fixed_constraints.append(
            (rank_indices, optimal_count)
        )

    # ---------------------------------------------------------
    # 5. Extract selected functional group occurrences
    # ---------------------------------------------------------

    selected = [
        candidate
        for candidate, value in zip(candidates, result.x)
        if value > 0.5
    ]

    # Count occurrences per functional group
    counts = {}

    for candidate in selected:
        group = candidate["group"]
        counts[group] = counts.get(group, 0) + 1

    # Preserve original hierarchy ordering
    decomposition = [
        (counts[group], group)
        for group in groups
        if group in counts
    ]

    return decomposition




def load_fgc(path: str) -> 'FunctionalGroupSet':
    """
    Load functional groups from an FGC file.

    Each #DEFINE_GROUPS directive introduces a new order,
    starting at 1. The optional 'strict' keyword determines
    whether groups in that order must be fully decomposable.

    Each group definition has the format:
        "SMARTS" [priority] # optional comment

    :param path: Path to the FGC file.
    :return: Constructed FunctionalGroupSet.
    """
    groups: dict[int, dict[str, FunctionalGroup]] = {}
    strict_orders: set[int] = set()

    current_order = 0

    # Match a quoted SMARTS string followed by an optional integer priority
    group_pattern = re.compile(r'^\s*"([^"]+)"(?:\s+(-?\d+))?\s*(?:#.*)?$')

    with open(path, "r", encoding="utf-8") as f:

        for line_number, line in enumerate(f, start=1):
            line = line.strip()

            # Skip empty lines and comments
            if not line or line.startswith("#") and not line.startswith("#DEFINE_GROUPS"):
                continue

            # Start a new functional group order
            if line.startswith("#DEFINE_GROUPS"):
                directive = line.split()

                if directive[0] != "#DEFINE_GROUPS" or len(directive) > 2:
                    raise ValueError(
                        f"Invalid directive at line {line_number}: {line}"
                    )

                current_order += 1
                groups[current_order] = {}

                if len(directive) == 2:
                    if directive[1] != "strict":
                        raise ValueError(
                            f"Unknown directive option at line {line_number}: "
                            f"{directive[1]}"
                        )
                    strict_orders.add(current_order)

                continue

            # Reject group definitions outside a directive
            if current_order == 0:
                raise ValueError(
                    f"Group definition outside #DEFINE_GROUPS "
                    f"at line {line_number}"
                )

            # Parse group definition
            match = group_pattern.match(line)

            if not match:
                raise ValueError(
                    f"Invalid group definition at line {line_number}: {line}"
                )

            smarts = match.group(1)
            priority = int(match.group(2) or 0)

            # Reject duplicate SMARTS within the same order
            if smarts in groups[current_order]:
                raise ValueError(
                    f"Duplicate functional group at line {line_number}: "
                    f"{smarts}"
                )

            # Construct functional group
            try:
                group = FunctionalGroup(
                    smarts=smarts,
                    order=current_order,
                    priority=priority
                )
            except ValueError as e:
                raise ValueError(
                    f"Invalid SMARTS at line {line_number}: {smarts}"
                ) from e

            groups[current_order][smarts] = group

    if not groups:
        raise ValueError(f"No functional groups found in {path}")

    # Decompose each order into all previously defined orders
    for order in sorted(groups):

        if order == 1:
            continue

        lower_order_groups = [
            group
            for lower_order in range(1, order)
            for group in groups[lower_order].values()
        ]

        for group in groups[order].values():
            group.decompose(
                lower_order_groups,
                strict=(order in strict_orders)
            )

    return groups
