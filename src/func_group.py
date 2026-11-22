from rdkit import Chem


class FunctionalGroup:
    """
    Dataclass representing a functional group defined by a SMARTS string.
    Allows for auto-decomposition and provides the `match` method to find occurrences
    of the functional group in a molecule.
    """

    def __init__(self, smarts: str, order: int, priority: int = 0):
        """
        Initialize a new functional group from a SMARTS string.

        :param smarts: Unique SMARTS string defining the functional group.
        :param order: Order in functional group hierarchy.
        :param priority: Priority value defining order of matching.

        :raises ValueError: Raised when the SMARTS string is invalid.
        """
        self._smarts: str = smarts
        self._order: int = order
        self._priority: int = priority

        self._query: Chem.Mol = Chem.MolFromSmarts(smarts)
        if not self._query: raise ValueError(
            f"Invalid SMARTS ({smarts}) for molecule")

        self._decomp: dict[str, tuple[FunctionalGroup, int]] | None = None


    @property
    def smarts(self) -> str:
        return self._smarts

    @property
    def order(self) -> int:
        return self._order

    @property
    def priority(self) -> int:
        return self._priority

    @property
    def num_atoms(self) -> int:
        return self._query.GetNumAtoms()

    @property
    def decomposition(self) -> dict[str, tuple['FunctionalGroup', int]] | None:
        return self._decomp.copy() if self._decomp is not None else None


    def decompose(self, groups: list['FunctionalGroup'], strict: bool = False) -> None:
        """
        Decomposes functional group into the provided functional groups.

        :param groups: Functional groups to decompose this functional group into.
        :param strict: `True` when this functional group should be fully decomposable
                       into the provided groups.
        :return: None

        :raises Exception: Raised when the functional group has already been decomposed.
        :raises ValueError: Raised when strict matching failed.
        """
        # Validate no decomposition exists for this functional group
        if self._decomp is not None: raise Exception(
            f"Functional group {self._smarts} has already been decomposed")
        decomp: dict[str, tuple[FunctionalGroup, int]] = {}

        # Validate all groups provided are of a lower order
        for group in groups:
            if group.order >= self._order: raise ValueError(
                f"Cannot decompose {self._order}-order functional group into "
                f"{group.order}-order groups: {group!r}")

        # Sort groups on order (desc), priority (desc), and atom count (desc)
        sorted_groups = sorted(groups, key=lambda g:
            (g.order, g.priority, g.num_atoms), reverse=True)

        # Match all groups in the sorted order (with replacement)
        matched: set[int] = set()
        for group in sorted_groups:
            matches = group.match(self._query, matched, use_qq_match=True)
            if matches > 0: decomp[group.smarts] = (group, matches)

        # Evaluate whether the decomposition is strict if required
        if strict and len(matched) != self._query.GetNumAtoms(): raise ValueError(
            f"Failed to strictly decompose {self!r} into provided groups: \n{decomp}")

        # Commit constructed decomposition
        self._decomp = decomp


    def match(self, mol: Chem.Mol, matched: set[int], use_qq_match: bool = False) -> int:
        """
        Find the number of occurrences of the functional group in the unmatched atoms of a molecule,
        automatically updating the set of matched atoms after completion.

        :param mol: Molecule to use for the functional group matching.
        :param matched: Atoms of the molecule to exclude from the search.
        :return: Number of matches found in the unmatched atoms of the molecule.
        """
        # Find all (unique) substructure matches
        matches = set()
        matches.update(mol.GetSubstructMatches(self._query, uniquify=True, useQueryQueryMatches=False))
        if use_qq_match: matches.update(mol.GetSubstructMatches(self._query, uniquify=True, useQueryQueryMatches=True))

        # Evaluate every substructure match found
        count: int = 0
        for match in matches:

            # Skip matches with already matched atoms
            if matched.intersection(match): continue

            # Update matched atoms set and increment counter
            matched.update(match)
            count += 1

        return count

    def get_matches(self, mol: Chem.Mol) -> list[tuple[int, ...]]:
        """
        Get all unique substructure matches without assigning atoms.

        Returns:
            List of unique atom-index tuples.
        """
        matches = set()

        for query_query in (False, True):
            matches.update(
                mol.GetSubstructMatches(
                    self._query,
                    uniquify=True,
                    useQueryQueryMatches=query_query
                )
            )

        return sorted(matches)


    def __str__(self) -> str:
        return self._smarts

    def __repr__(self) -> str:
        return (f"FunctionalGroup("
                    f"smarts: {self._smarts}, "
                    f"order: {self._order}, "
                    f"priority: {self._priority})")
