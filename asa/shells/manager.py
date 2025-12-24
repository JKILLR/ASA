"""
BoundedShellManager - Shell management with bounded cascade migration.

v1.1 Change: Migration cascades are bounded to O(1) with configurable
overflow handling. This ensures predictable performance regardless of
association volume.

Key changes from v1.0:
1. max_cascade_depth limits displacement chains
2. Overflow handling when cascade limit reached
3. Optional compression of weak associations
4. Migration statistics for monitoring
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import torch

from ..core.config import AtomConfig
from ..core.shells import Shell, ShellAssociation
from ..thermodynamics.learnable import LearnableThermodynamics


@dataclass
class MigrationConfig:
    """Configuration for bounded migration."""

    max_cascade_depth: int = 3
    overflow_strategy: str = "discard"  # "discard", "compress", "archive"
    compression_threshold: float = 0.3
    track_migrations: bool = True


class BoundedShellManager:
    """
    Shell manager with bounded cascade migration.

    Key features:
    - O(1) insertion complexity (bounded by max_cascade_depth)
    - Configurable overflow strategies
    - Migration statistics for monitoring
    - Hysteresis-based strength updates

    Complexity Analysis:
        Insertion (best case): O(1)
        Insertion (worst case): O(max_cascade_depth) = O(1)
        Update with hysteresis: O(N) but bounded moves
        Space: O(total_capacity)
    """

    def __init__(
        self,
        config: AtomConfig,
        thermo: LearnableThermodynamics,
        migration_config: MigrationConfig = None,
    ):
        """
        Initialize shell manager.

        Args:
            config: Atom configuration
            thermo: Thermodynamics module
            migration_config: Migration configuration
        """
        self.config = config
        self.thermo = thermo
        self.migration = migration_config if migration_config is not None else MigrationConfig()

        # Initialize shells based on config
        self.shells = [
            Shell(
                level=i,
                capacity=(
                    config.shell_capacities[i]
                    if i < len(config.shell_capacities)
                    else 64
                ),
                dimension=(
                    config.shell_dims[i] if i < len(config.shell_dims) else 512
                ),
            )
            for i in range(config.num_shells)
        ]

        # Migration statistics
        self.stats = {
            "insertions": 0,
            "cascades": 0,
            "max_cascade_seen": 0,
            "overflows": 0,
            "compressions": 0,
            "promotions": 0,
            "demotions": 0,
        }

    def add_association(
        self,
        target_atom_id: int,
        vector: torch.Tensor,
        strength: float,
        timestamp: int = 0,
        source_quality: float = 0.5,
    ) -> Tuple[int, dict]:
        """
        Add association with bounded cascade.

        Args:
            target_atom_id: ID of target atom
            vector: Association vector
            strength: Association strength [0, 1]
            timestamp: Creation timestamp
            source_quality: Quality of information source

        Returns:
            (shell_level, migration_info)
            shell_level = -1 if discarded, -2 if archived
        """
        self.stats["insertions"] += 1

        assoc = ShellAssociation(
            target_atom_id=target_atom_id,
            vector=vector,
            strength=strength,
            created_at=timestamp,
            source_quality=source_quality,
        )

        # Determine target shell based on strength
        target_level = self._strength_to_level(strength)

        migration_info = {
            "initial_target": target_level,
            "cascade_depth": 0,
            "displaced": [],
            "overflow": False,
            "final_level": None,
        }

        current_level = target_level
        current_assoc = assoc
        cascade_depth = 0

        # Bounded cascade loop
        while (
            current_level < len(self.shells)
            and cascade_depth < self.migration.max_cascade_depth
        ):
            displaced = self.shells[current_level].add(current_assoc)

            if displaced is None:
                # Successfully added
                migration_info["final_level"] = current_level
                migration_info["cascade_depth"] = cascade_depth
                self._update_cascade_stats(cascade_depth)
                return current_level, migration_info

            if displaced is current_assoc:
                # Rejected (too weak), try next shell
                current_level += 1
                cascade_depth += 1
            else:
                # Displaced existing association
                migration_info["displaced"].append(
                    {
                        "atom_id": displaced.target_atom_id,
                        "from_level": current_level,
                        "strength": displaced.strength,
                    }
                )
                current_assoc = displaced
                current_level += 1
                cascade_depth += 1

        # Cascade limit reached or fell off edge
        migration_info["overflow"] = True
        migration_info["cascade_depth"] = cascade_depth
        self.stats["overflows"] += 1

        final_level = self._handle_overflow(current_assoc, current_level)
        migration_info["final_level"] = final_level

        self._update_cascade_stats(cascade_depth)
        return final_level, migration_info

    def _handle_overflow(
        self, assoc: ShellAssociation, attempted_level: int
    ) -> int:
        """
        Handle association that couldn't be placed within cascade limit.

        Args:
            assoc: Association that overflowed
            attempted_level: Level that was attempted

        Returns:
            Final level (-1 = discarded, -2 = archived)
        """
        if self.migration.overflow_strategy == "discard":
            return -1

        elif self.migration.overflow_strategy == "compress":
            # Try to make room by compressing outer shells
            self._compress_outer_shells()
            if attempted_level < len(self.shells):
                displaced = self.shells[-1].add(assoc)
                if displaced is None:
                    self.stats["compressions"] += 1
                    return len(self.shells) - 1
            return -1

        elif self.migration.overflow_strategy == "archive":
            # Mark for external archival
            return -2

        return -1

    def _compress_outer_shells(self) -> None:
        """Remove weak associations from outer shells to free space."""
        for shell in reversed(self.shells[2:]):
            weak = [
                a
                for a in shell.associations
                if a.strength < self.migration.compression_threshold
            ]
            # Remove half of weak associations
            for a in weak[: len(weak) // 2]:
                shell.associations.remove(a)
                self.stats["compressions"] += 1

    def _strength_to_level(self, strength: float) -> int:
        """
        Map strength to target shell level using learned thresholds.

        Args:
            strength: Association strength

        Returns:
            Target shell level (0 = innermost)
        """
        promo = self.thermo.promotion_thresholds.detach()
        for i, thresh in enumerate(promo):
            if strength >= thresh.item():
                return i
        return len(self.shells) - 1

    def _update_cascade_stats(self, depth: int) -> None:
        """Update cascade statistics."""
        if depth > 0:
            self.stats["cascades"] += 1
        self.stats["max_cascade_seen"] = max(
            self.stats["max_cascade_seen"], depth
        )

    def update_strengths(self, updates: Dict[int, float]) -> List[dict]:
        """
        Update strengths with hysteresis rules.

        Applies promotion/demotion based on new strengths,
        respecting hysteresis thresholds.

        Args:
            updates: Dict mapping atom_id to new strength

        Returns:
            List of moves performed
        """
        moves = []

        for level, shell in enumerate(self.shells):
            for assoc in list(shell.associations):
                if assoc.target_atom_id not in updates:
                    continue

                old_strength = assoc.strength
                assoc.strength = updates[assoc.target_atom_id]

                # Check for demotion (to outer shell)
                if self.thermo.should_demote(
                    level, assoc.strength, len(self.shells)
                ):
                    moves.append(
                        {
                            "atom_id": assoc.target_atom_id,
                            "from": level,
                            "to": level + 1,
                            "type": "demotion",
                            "old_strength": old_strength,
                            "new_strength": assoc.strength,
                        }
                    )
                    self.stats["demotions"] += 1

                # Check for promotion (to inner shell)
                elif self.thermo.can_promote(level, assoc.strength):
                    moves.append(
                        {
                            "atom_id": assoc.target_atom_id,
                            "from": level,
                            "to": level - 1,
                            "type": "promotion",
                            "old_strength": old_strength,
                            "new_strength": assoc.strength,
                        }
                    )
                    self.stats["promotions"] += 1

        # Execute moves
        for move in moves:
            assoc = self.shells[move["from"]].remove(move["atom_id"])
            if assoc and 0 <= move["to"] < len(self.shells):
                self.shells[move["to"]].add(assoc)

        return moves

    def remove_association(
        self, target_atom_id: int, available_energy: float = float("inf")
    ) -> Tuple[bool, Optional[ShellAssociation]]:
        """
        Remove association if enough energy available.

        Inner shells require more energy to modify (ionization energy).

        Args:
            target_atom_id: ID of target atom to remove
            available_energy: Energy available for removal

        Returns:
            (success, removed_association)
        """
        for level, shell in enumerate(self.shells):
            for assoc in shell.associations:
                if assoc.target_atom_id == target_atom_id:
                    required = self.thermo.ionization_energy(
                        level, len(self.shells)
                    ).item()
                    if available_energy < required:
                        return False, None
                    shell.associations.remove(assoc)
                    return True, assoc
        return False, None

    def get_association(self, target_atom_id: int) -> Optional[Tuple[int, ShellAssociation]]:
        """
        Find association by target atom ID.

        Args:
            target_atom_id: ID to search for

        Returns:
            (shell_level, association) or None
        """
        for level, shell in enumerate(self.shells):
            assoc = shell.get(target_atom_id)
            if assoc is not None:
                return level, assoc
        return None

    def decay_all(self, factor: float = 0.99) -> None:
        """Apply time decay to all associations."""
        for shell in self.shells:
            shell.decay_all(factor)

    def get_efficiency_report(self) -> dict:
        """
        Report on migration efficiency.

        Returns:
            Dict with efficiency metrics
        """
        total = self.stats["insertions"]
        if total == 0:
            return {"status": "no_insertions"}

        return {
            "total_insertions": total,
            "cascade_rate": self.stats["cascades"] / total,
            "overflow_rate": self.stats["overflows"] / total,
            "compression_rate": self.stats["compressions"] / total,
            "max_cascade_depth": self.stats["max_cascade_seen"],
            "promotion_count": self.stats["promotions"],
            "demotion_count": self.stats["demotions"],
            "avg_shell_occupancy": [
                len(s.associations) / s.capacity if s.capacity > 0 else 0
                for s in self.shells
            ],
        }

    def total_occupancy(self) -> Tuple[int, int]:
        """
        Get total occupancy across all shells.

        Returns:
            (filled_slots, total_capacity)
        """
        filled = sum(len(s.associations) for s in self.shells)
        total = sum(s.capacity for s in self.shells)
        return filled, total

    def get_shell_vectors(self) -> List[torch.Tensor]:
        """Get aggregated vectors for all shells."""
        return [shell.get_vector() for shell in self.shells]

    def clear(self) -> None:
        """Clear all associations."""
        for shell in self.shells:
            shell.associations.clear()
        self.stats = {k: 0 for k in self.stats}
