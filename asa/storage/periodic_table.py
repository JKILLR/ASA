"""
PeriodicTable - O(1) lookup for semantic atoms, organized by semantic properties.

Provides fast lookup by:
- Atom ID
- Token string
- Valence group (semantic family)
- Phase state
"""

from typing import Dict, List, Optional

from ..core.config import AtomConfig
from ..core.atoms import SemanticAtom, ConceptPhase


class PeriodicTable:
    """
    O(1) lookup for semantic atoms, organized by semantic properties.

    Analogous to the chemical periodic table, atoms are organized by:
    - Valence (group): Semantic bonding capacity
    - Phase: Crystallization state
    - Charge: Polarity
    """

    def __init__(self, config: AtomConfig):
        """
        Initialize periodic table.

        Args:
            config: Atom configuration
        """
        self.config = config

        # Primary storage
        self.atoms: Dict[int, SemanticAtom] = {}
        self.token_to_id: Dict[str, int] = {}

        # Indices for fast lookup
        self.group_index: Dict[int, List[int]] = {}  # valence -> atom_ids
        self.phase_index: Dict[ConceptPhase, List[int]] = {}  # phase -> atom_ids
        self.charge_index: Dict[str, List[int]] = {
            "positive": [],
            "negative": [],
            "neutral": [],
        }

    def register(self, atom: SemanticAtom) -> None:
        """
        Register an atom in the table.

        Args:
            atom: Atom to register
        """
        # Primary storage
        self.atoms[atom.atom_id] = atom
        self.token_to_id[atom.token] = atom.atom_id

        # Index by valence (group)
        group = atom.valence_count
        if group not in self.group_index:
            self.group_index[group] = []
        self.group_index[group].append(atom.atom_id)

        # Index by phase
        if atom.phase not in self.phase_index:
            self.phase_index[atom.phase] = []
        self.phase_index[atom.phase].append(atom.atom_id)

        # Index by charge
        charge = atom.effective_charge
        if charge > 0.2:
            self.charge_index["positive"].append(atom.atom_id)
        elif charge < -0.2:
            self.charge_index["negative"].append(atom.atom_id)
        else:
            self.charge_index["neutral"].append(atom.atom_id)

    def unregister(self, atom_id: int) -> Optional[SemanticAtom]:
        """
        Remove an atom from the table.

        Args:
            atom_id: ID of atom to remove

        Returns:
            Removed atom, or None if not found
        """
        if atom_id not in self.atoms:
            return None

        atom = self.atoms.pop(atom_id)
        self.token_to_id.pop(atom.token, None)

        # Remove from indices
        if atom.valence_count in self.group_index:
            self.group_index[atom.valence_count] = [
                aid for aid in self.group_index[atom.valence_count]
                if aid != atom_id
            ]

        if atom.phase in self.phase_index:
            self.phase_index[atom.phase] = [
                aid for aid in self.phase_index[atom.phase]
                if aid != atom_id
            ]

        for key in self.charge_index:
            self.charge_index[key] = [
                aid for aid in self.charge_index[key]
                if aid != atom_id
            ]

        return atom

    def lookup(self, atom_id: int) -> Optional[SemanticAtom]:
        """
        Lookup atom by ID.

        Args:
            atom_id: Atom ID

        Returns:
            Atom if found, None otherwise
        """
        return self.atoms.get(atom_id)

    def lookup_token(self, token: str) -> Optional[SemanticAtom]:
        """
        Lookup atom by token string.

        Args:
            token: Token string

        Returns:
            Atom if found, None otherwise
        """
        atom_id = self.token_to_id.get(token)
        return self.atoms.get(atom_id) if atom_id is not None else None

    def get_family(self, atom_id: int) -> List[SemanticAtom]:
        """
        Get semantically similar atoms (same valence group).

        Args:
            atom_id: Atom to find family for

        Returns:
            List of atoms in same valence group
        """
        atom = self.atoms.get(atom_id)
        if not atom:
            return []
        group_id = atom.valence_count
        return [
            self.atoms[aid]
            for aid in self.group_index.get(group_id, [])
        ]

    def get_by_phase(self, phase: ConceptPhase) -> List[SemanticAtom]:
        """
        Get all atoms in a phase state.

        Args:
            phase: Phase to filter by

        Returns:
            List of atoms in that phase
        """
        return [
            self.atoms[aid]
            for aid in self.phase_index.get(phase, [])
        ]

    def get_by_charge(self, charge_type: str) -> List[SemanticAtom]:
        """
        Get atoms by charge type.

        Args:
            charge_type: "positive", "negative", or "neutral"

        Returns:
            List of atoms with that charge type
        """
        return [
            self.atoms[aid]
            for aid in self.charge_index.get(charge_type, [])
        ]

    def get_compatible_partners(self, atom_id: int) -> List[SemanticAtom]:
        """
        Get atoms that could bond with the given atom.

        Compatibility based on:
        - Having open valence
        - Compatible charge (opposite or neutral)

        Args:
            atom_id: Atom to find partners for

        Returns:
            List of potentially compatible atoms
        """
        atom = self.atoms.get(atom_id)
        if not atom or atom.valence_count <= 0:
            return []

        partners = []
        atom_charge = atom.effective_charge

        for other_id, other in self.atoms.items():
            if other_id == atom_id:
                continue
            if other.valence_count <= 0:
                continue

            # Check charge compatibility
            other_charge = other.effective_charge
            if atom_charge * other_charge <= 0:  # Opposite or neutral
                partners.append(other)

        return partners

    def __len__(self) -> int:
        return len(self.atoms)

    def __contains__(self, item) -> bool:
        if isinstance(item, int):
            return item in self.atoms
        elif isinstance(item, str):
            return item in self.token_to_id
        return False

    def get_statistics(self) -> dict:
        """Get table statistics."""
        return {
            "total_atoms": len(self.atoms),
            "groups": {k: len(v) for k, v in self.group_index.items()},
            "phases": {k.value: len(v) for k, v in self.phase_index.items()},
            "charges": {k: len(v) for k, v in self.charge_index.items()},
        }
