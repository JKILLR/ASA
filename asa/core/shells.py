"""
Shell and ShellAssociation - Layered relational features.

Shells model semantic associations at different abstraction levels:
- Shell 1 (inner): Core/definitional associations
- Shell 2 (middle): Common contextual associations
- Shell 3 (outer): Peripheral/situational associations

Each shell has capacity limits, and associations compete for slots
based on strength and recency.
"""

from dataclasses import dataclass, field
from typing import List, Optional
import torch


@dataclass
class ShellAssociation:
    """
    A single association within a shell.

    Represents a connection from this atom to another atom,
    stored at a specific shell level based on association strength.

    Attributes:
        target_atom_id: ID of the associated atom
        vector: Embedding vector for this association
        strength: Strength of association [0, 1]
        created_at: Timestamp of creation (for recency)
        source_quality: Quality/reliability of the source (v1.1)
    """

    target_atom_id: int
    vector: torch.Tensor
    strength: float
    created_at: int = 0
    source_quality: float = 0.5  # v1.1: Track source reliability

    def __lt__(self, other: "ShellAssociation") -> bool:
        """For heap operations: weaker = smaller = evicted first."""
        if self.strength != other.strength:
            return self.strength < other.strength
        # Newer associations are evicted first if strengths are equal
        return self.created_at > other.created_at

    def decay(self, factor: float = 0.99) -> None:
        """Apply time decay to strength."""
        self.strength *= factor

    def reinforce(self, amount: float = 0.1) -> None:
        """Reinforce this association."""
        self.strength = min(1.0, self.strength + amount)

    def __repr__(self) -> str:
        return f"ShellAssociation(target={self.target_atom_id}, strength={self.strength:.3f})"


@dataclass
class Shell:
    """
    A single semantic shell with capacity limits.

    Shells organize associations by strength/importance:
    - Inner shells: High-strength, core associations
    - Outer shells: Lower-strength, peripheral associations

    When a shell is full, weak associations are displaced to outer shells.

    Attributes:
        level: Shell index (0 = innermost)
        capacity: Maximum associations in this shell
        dimension: Embedding dimension for this shell
        associations: List of current associations
    """

    level: int
    capacity: int
    dimension: int
    associations: List[ShellAssociation] = field(default_factory=list)

    @property
    def is_full(self) -> bool:
        """Check if shell is at capacity."""
        return len(self.associations) >= self.capacity

    @property
    def occupancy(self) -> float:
        """Fraction of capacity used."""
        return len(self.associations) / self.capacity if self.capacity > 0 else 0

    @property
    def count(self) -> int:
        """Number of associations in this shell."""
        return len(self.associations)

    def weakest(self) -> Optional[ShellAssociation]:
        """Get the weakest association in this shell."""
        if not self.associations:
            return None
        return min(self.associations)

    def strongest(self) -> Optional[ShellAssociation]:
        """Get the strongest association in this shell."""
        if not self.associations:
            return None
        return max(self.associations, key=lambda a: a.strength)

    def add(self, assoc: ShellAssociation) -> Optional[ShellAssociation]:
        """
        Add association to shell.

        If shell is full, displaces weakest if new association is stronger.

        Args:
            assoc: Association to add

        Returns:
            Displaced association if shell was full, or the rejected
            association if it was too weak. None if successfully added.
        """
        if not self.is_full:
            self.associations.append(assoc)
            return None

        weakest = self.weakest()
        if assoc.strength > weakest.strength:
            self.associations.remove(weakest)
            self.associations.append(assoc)
            return weakest
        else:
            return assoc  # Rejected

    def remove(self, target_id: int) -> Optional[ShellAssociation]:
        """
        Remove association by target atom ID.

        Args:
            target_id: ID of target atom to remove

        Returns:
            Removed association, or None if not found
        """
        for assoc in self.associations:
            if assoc.target_atom_id == target_id:
                self.associations.remove(assoc)
                return assoc
        return None

    def get(self, target_id: int) -> Optional[ShellAssociation]:
        """Get association by target atom ID."""
        for assoc in self.associations:
            if assoc.target_atom_id == target_id:
                return assoc
        return None

    def get_vector(self) -> torch.Tensor:
        """
        Get aggregated shell vector.

        Computes weighted average of all association vectors,
        weighted by their strengths.

        Returns:
            Aggregated vector of shape (dimension,)
        """
        if not self.associations:
            return torch.zeros(self.dimension)

        vectors = torch.stack([a.vector for a in self.associations])
        weights = torch.tensor([a.strength for a in self.associations])
        weights = weights / (weights.sum() + 1e-8)
        return (vectors * weights.unsqueeze(-1)).sum(dim=0)

    def decay_all(self, factor: float = 0.99) -> None:
        """Apply time decay to all associations."""
        for assoc in self.associations:
            assoc.decay(factor)

    def average_strength(self) -> float:
        """Get average strength of associations."""
        if not self.associations:
            return 0.0
        return sum(a.strength for a in self.associations) / len(self.associations)

    def __repr__(self) -> str:
        return f"Shell(level={self.level}, count={self.count}/{self.capacity}, dim={self.dimension})"
