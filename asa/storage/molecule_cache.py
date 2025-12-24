"""
MoleculeCache - LRU cache for frequently-used semantic molecules.

Provides fast lookup of pre-composed molecular structures with:
- LRU eviction policy
- Hit rate tracking
- Compute-on-miss pattern
"""

from collections import OrderedDict
from typing import Callable, Optional, Tuple

from ..core.atoms import SemanticMolecule


class MoleculeCache:
    """
    LRU cache for frequently-used semantic molecules.

    Caches composed molecular structures keyed by atom ID tuples.
    Uses LRU eviction to maintain memory bounds.
    """

    def __init__(self, max_size: int = 100000):
        """
        Initialize molecule cache.

        Args:
            max_size: Maximum number of molecules to cache
        """
        self.cache: OrderedDict[Tuple[int, ...], SemanticMolecule] = OrderedDict()
        self.max_size = max_size
        self.hits = 0
        self.misses = 0

    def get(self, atom_ids: Tuple[int, ...]) -> Optional[SemanticMolecule]:
        """
        Get molecule from cache.

        Args:
            atom_ids: Tuple of atom IDs (sorted for consistency)

        Returns:
            Cached molecule, or None if not found
        """
        # Normalize key
        key = tuple(sorted(atom_ids))

        if key in self.cache:
            # Move to end (most recently used)
            self.cache.move_to_end(key)
            self.hits += 1
            return self.cache[key]

        self.misses += 1
        return None

    def put(self, atom_ids: Tuple[int, ...], molecule: SemanticMolecule) -> None:
        """
        Put molecule in cache.

        Args:
            atom_ids: Tuple of atom IDs
            molecule: Molecule to cache
        """
        # Normalize key
        key = tuple(sorted(atom_ids))

        self.cache[key] = molecule

        # Evict oldest if over capacity
        while len(self.cache) > self.max_size:
            self.cache.popitem(last=False)

    def get_or_compute(
        self,
        atom_ids: Tuple[int, ...],
        compute_fn: Callable[[], SemanticMolecule],
    ) -> SemanticMolecule:
        """
        Get from cache or compute and cache.

        Args:
            atom_ids: Tuple of atom IDs
            compute_fn: Function to compute molecule if not cached

        Returns:
            Cached or newly computed molecule
        """
        cached = self.get(atom_ids)
        if cached is not None:
            return cached

        molecule = compute_fn()
        self.put(atom_ids, molecule)
        return molecule

    @property
    def hit_rate(self) -> float:
        """Get cache hit rate."""
        total = self.hits + self.misses
        return self.hits / total if total > 0 else 0.0

    @property
    def miss_rate(self) -> float:
        """Get cache miss rate."""
        return 1.0 - self.hit_rate

    @property
    def size(self) -> int:
        """Get current cache size."""
        return len(self.cache)

    def clear(self) -> None:
        """Clear the cache."""
        self.cache.clear()
        self.hits = 0
        self.misses = 0

    def invalidate(self, atom_id: int) -> int:
        """
        Invalidate all cache entries containing an atom.

        Args:
            atom_id: Atom ID to invalidate

        Returns:
            Number of entries invalidated
        """
        to_remove = [
            key for key in self.cache if atom_id in key
        ]
        for key in to_remove:
            del self.cache[key]
        return len(to_remove)

    def get_statistics(self) -> dict:
        """Get cache statistics."""
        return {
            "size": len(self.cache),
            "max_size": self.max_size,
            "hits": self.hits,
            "misses": self.misses,
            "hit_rate": self.hit_rate,
            "utilization": len(self.cache) / self.max_size if self.max_size > 0 else 0,
        }

    def __len__(self) -> int:
        return len(self.cache)

    def __contains__(self, atom_ids: Tuple[int, ...]) -> bool:
        key = tuple(sorted(atom_ids))
        return key in self.cache


class TieredMoleculeCache:
    """
    Two-tier cache with hot and cold layers.

    Hot layer: Small, fast, frequently accessed
    Cold layer: Larger, less frequently accessed
    """

    def __init__(
        self,
        hot_size: int = 10000,
        cold_size: int = 100000,
    ):
        """
        Initialize tiered cache.

        Args:
            hot_size: Size of hot layer
            cold_size: Size of cold layer
        """
        self.hot = MoleculeCache(hot_size)
        self.cold = MoleculeCache(cold_size)
        self.promotions = 0

    def get(self, atom_ids: Tuple[int, ...]) -> Optional[SemanticMolecule]:
        """
        Get molecule from cache.

        Checks hot layer first, then cold. Promotes from cold to hot.

        Args:
            atom_ids: Tuple of atom IDs

        Returns:
            Cached molecule, or None
        """
        # Check hot layer
        result = self.hot.get(atom_ids)
        if result is not None:
            return result

        # Check cold layer
        result = self.cold.get(atom_ids)
        if result is not None:
            # Promote to hot layer
            self.hot.put(atom_ids, result)
            self.promotions += 1
            return result

        return None

    def put(
        self,
        atom_ids: Tuple[int, ...],
        molecule: SemanticMolecule,
        hot: bool = False,
    ) -> None:
        """
        Put molecule in cache.

        Args:
            atom_ids: Tuple of atom IDs
            molecule: Molecule to cache
            hot: If True, put directly in hot layer
        """
        if hot:
            self.hot.put(atom_ids, molecule)
        else:
            self.cold.put(atom_ids, molecule)

    def get_statistics(self) -> dict:
        """Get combined statistics."""
        return {
            "hot": self.hot.get_statistics(),
            "cold": self.cold.get_statistics(),
            "promotions": self.promotions,
            "combined_hit_rate": (
                (self.hot.hits + self.cold.hits)
                / max(1, self.hot.hits + self.hot.misses)
            ),
        }

    def clear(self) -> None:
        """Clear both layers."""
        self.hot.clear()
        self.cold.clear()
        self.promotions = 0
