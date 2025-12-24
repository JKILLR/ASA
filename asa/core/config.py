"""
AtomConfig - Master configuration for atomic embeddings.
"""

from dataclasses import dataclass
from typing import Tuple


@dataclass
class AtomConfig:
    """
    Master configuration for atomic embeddings.

    The atomic semantic architecture models meaning using the structure of matter:
    - Nuclear vector: Core identity embedding (like proton count)
    - Shells: Layered relational features at different abstraction levels
    - Charge: Truth/existence state with polarity and magnitude
    - Valence: Bonding interface for semantic composition

    Attributes:
        vocab_size: Size of the vocabulary
        nuclear_dim: Dimension of nuclear (identity) vector
        shell_dims: Dimensions per shell (inner to outer)
        shell_capacities: Max associations per shell
        charge_dim: Dimension of charge vector representation
        valence_max: Maximum number of valence bonds
        bond_dim: Dimension of bond interface vectors
        max_cascade_depth: Maximum migration cascade depth (v1.1)
        overflow_strategy: How to handle cascade overflow (v1.1)
    """

    # Vocabulary
    vocab_size: int = 50000

    # Nuclear (identity)
    nuclear_dim: int = 64

    # Shells (relational) - dimensions per shell
    shell_dims: Tuple[int, ...] = (128, 256, 512)

    # Shell capacity limits (max associations per shell)
    shell_capacities: Tuple[int, ...] = (4, 12, 32, 64)

    # Charge system
    charge_dim: int = 8

    # Valence/bonding
    valence_max: int = 4
    bond_dim: int = 32

    # Migration bounds (v1.1)
    max_cascade_depth: int = 3
    overflow_strategy: str = "discard"  # "discard", "compress", "archive"

    @property
    def total_dim(self) -> int:
        """Total embedding dimension (nuclear + all shells)."""
        return self.nuclear_dim + sum(self.shell_dims)

    @property
    def num_shells(self) -> int:
        """Number of shells."""
        return len(self.shell_dims)

    def __post_init__(self):
        """Validate configuration."""
        assert self.nuclear_dim > 0, "Nuclear dimension must be positive"
        assert len(self.shell_dims) > 0, "Must have at least one shell"
        assert all(d > 0 for d in self.shell_dims), "All shell dimensions must be positive"
        assert self.charge_dim > 0, "Charge dimension must be positive"
        assert self.valence_max > 0, "Valence max must be positive"
        assert self.bond_dim > 0, "Bond dimension must be positive"
        assert self.max_cascade_depth > 0, "Max cascade depth must be positive"
        assert self.overflow_strategy in ["discard", "compress", "archive"], \
            f"Unknown overflow strategy: {self.overflow_strategy}"
