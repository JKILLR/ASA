"""
SemanticAtom, SemanticBond, SemanticMolecule - Core semantic entities.

These classes model meaning using atomic physics analogies:
- Atoms: Basic semantic units with identity, shells, charge, and valence
- Bonds: Typed connections between atoms
- Molecules: Composed semantic structures
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import List, Optional, Tuple
import torch

from .charge import ChargeState
from .shells import Shell


class ConceptPhase(Enum):
    """
    Phase state of a concept.

    Concepts crystallize from gas (vague) through liquid (adaptive)
    to solid (stable) based on their association patterns.
    """

    GAS = "gas"  # Vague, dispersed, context-dependent
    LIQUID = "liquid"  # Adaptive, flowing, moderate stability
    SOLID = "solid"  # Crystallized, stable, resists change


@dataclass
class SemanticAtom:
    """
    Complete semantic atom with all components.

    A semantic atom is the fundamental unit of meaning, analogous
    to a chemical atom:
    - Nuclear vector: Core identity (like atomic number)
    - Shells: Relational features at different abstraction levels
    - Charge: Truth/existence state (polarity and conviction)
    - Valence: Bonding capacity and interface

    Attributes:
        atom_id: Unique identifier
        token: The token this atom represents
        nuclear_vector: Core identity embedding
        shells: List of shells with associations
        charge: Truth/existence state
        valence_count: Number of bonding slots
        bond_vectors: Valence interface vectors
        mass: Semantic weight (affects composition)
        phase: Crystallization state
        isotope_id: Variant identifier (for polysemy)
    """

    atom_id: int
    token: str
    nuclear_vector: torch.Tensor
    shells: List[Shell]
    charge: ChargeState
    valence_count: int
    bond_vectors: torch.Tensor  # (valence_max, bond_dim)
    mass: float = 1.0
    phase: ConceptPhase = None
    isotope_id: int = 0

    def __post_init__(self):
        if self.phase is None:
            self.phase = ConceptPhase.LIQUID

    @property
    def effective_charge(self) -> float:
        """Get effective charge value."""
        return self.charge.effective

    def to_flat_vector(self) -> torch.Tensor:
        """
        Concatenate all components into a flat vector.

        Returns:
            Tensor of shape (nuclear_dim + sum(shell_dims),)
        """
        shell_vecs = [s.get_vector() for s in self.shells]
        return torch.cat([self.nuclear_vector] + shell_vecs)

    def shell_occupancy(self) -> Tuple[int, int]:
        """
        Get shell occupancy statistics.

        Returns:
            (filled_slots, total_capacity)
        """
        filled = sum(len(s.associations) for s in self.shells)
        total = sum(s.capacity for s in self.shells)
        return filled, total

    def get_shell(self, level: int) -> Optional[Shell]:
        """Get shell by level index."""
        if 0 <= level < len(self.shells):
            return self.shells[level]
        return None

    def decay_associations(self, factor: float = 0.99) -> None:
        """Apply time decay to all shell associations."""
        for shell in self.shells:
            shell.decay_all(factor)

    def __repr__(self) -> str:
        filled, total = self.shell_occupancy()
        return (
            f"SemanticAtom(id={self.atom_id}, token='{self.token}', "
            f"charge={self.charge.effective:.2f}, shells={filled}/{total}, "
            f"phase={self.phase.value})"
        )


@dataclass
class SemanticBond:
    """
    A bond between two atoms.

    Bonds connect atoms with typed relationships, analogous
    to chemical bonds:
    - Covalent: Strong shared meaning
    - Ionic: Oppositely charged attraction
    - Hydrogen: Weak contextual link
    - Modifier: One atom modifies the other
    - Argument: Predicate-argument relationship

    Attributes:
        atom_a: First atom in bond
        atom_b: Second atom in bond
        bond_type: Type of bond
        strength: Bond strength [0, 1]
        role_a: Semantic role of atom_a
        role_b: Semantic role of atom_b
        catalyzed_by: List of catalyst tokens (if any)
    """

    atom_a: SemanticAtom
    atom_b: SemanticAtom
    bond_type: str  # 'covalent', 'ionic', 'hydrogen', 'modifier', 'argument'
    strength: float
    role_a: str = ""
    role_b: str = ""
    catalyzed_by: Optional[List[str]] = None

    def __repr__(self) -> str:
        cat = f", catalyzed_by={self.catalyzed_by}" if self.catalyzed_by else ""
        return (
            f"SemanticBond({self.atom_a.token} --[{self.bond_type}:{self.strength:.2f}]--> "
            f"{self.atom_b.token}{cat})"
        )


@dataclass
class SemanticMolecule:
    """
    A composed semantic structure.

    Molecules are collections of atoms connected by bonds,
    representing complex meanings like phrases or sentences.

    Attributes:
        atoms: List of atoms in molecule
        bonds: List of bonds connecting atoms
        composed_vector: Aggregated embedding
        net_charge: Overall charge of molecule
    """

    atoms: List[SemanticAtom]
    bonds: List[SemanticBond]
    composed_vector: torch.Tensor = None
    net_charge: float = 0.0

    def __post_init__(self):
        if self.composed_vector is None and self.atoms:
            self.composed_vector = self._compose()
            self.net_charge = self._compute_charge()

    def _compose(self) -> torch.Tensor:
        """Compose atoms into molecular vector (mass-weighted average)."""
        if not self.atoms:
            return torch.zeros(1)
        vecs = torch.stack([a.to_flat_vector() for a in self.atoms])
        weights = torch.tensor([a.mass for a in self.atoms])
        weights = weights / (weights.sum() + 1e-8)
        return (vecs * weights.unsqueeze(-1)).sum(dim=0)

    def _compute_charge(self) -> float:
        """Compute net charge (mass-weighted average of effective charges)."""
        if not self.atoms:
            return 0.0
        total = sum(a.effective_charge * a.mass for a in self.atoms)
        mass = sum(a.mass for a in self.atoms)
        return total / mass if mass > 0 else 0.0

    def get_atom(self, atom_id: int) -> Optional[SemanticAtom]:
        """Get atom by ID."""
        for atom in self.atoms:
            if atom.atom_id == atom_id:
                return atom
        return None

    def get_bonds_for(self, atom_id: int) -> List[SemanticBond]:
        """Get all bonds involving a specific atom."""
        return [
            b for b in self.bonds
            if b.atom_a.atom_id == atom_id or b.atom_b.atom_id == atom_id
        ]

    def total_bond_strength(self) -> float:
        """Get sum of all bond strengths."""
        return sum(b.strength for b in self.bonds)

    def average_bond_strength(self) -> float:
        """Get average bond strength."""
        if not self.bonds:
            return 0.0
        return self.total_bond_strength() / len(self.bonds)

    def __repr__(self) -> str:
        tokens = [a.token for a in self.atoms]
        return (
            f"SemanticMolecule(atoms={tokens}, bonds={len(self.bonds)}, "
            f"charge={self.net_charge:.2f})"
        )
