"""
CompositionNetwork - Composes atoms into molecules with charge propagation.

Uses neural networks to:
- Aggregate atomic representations
- Propagate charge through structure
- Produce molecular embeddings
"""

from typing import Dict, List

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..core.config import AtomConfig


class CompositionNetwork(nn.Module):
    """
    Composes atoms into molecules with charge propagation.

    Takes a list of atom representations and bonds, produces
    a unified molecular representation.
    """

    def __init__(self, config: AtomConfig):
        """
        Initialize composition network.

        Args:
            config: Atom configuration
        """
        super().__init__()
        self.config = config

        # Message network for bond-based aggregation
        self.message_net = nn.Sequential(
            nn.Linear(config.total_dim + 32, 128),
            nn.GELU(),
            nn.Linear(128, config.total_dim),
        )

        # Composition network
        self.compose_net = nn.Sequential(
            nn.Linear(config.total_dim, 256),
            nn.LayerNorm(256),
            nn.GELU(),
            nn.Linear(256, config.total_dim),
        )

        # Charge aggregation
        self.charge_aggregator = nn.Sequential(
            nn.Linear(3, 16),  # polarity, magnitude, stability
            nn.GELU(),
            nn.Linear(16, 1),
        )

    def forward(
        self,
        atoms: List[Dict[str, torch.Tensor]],
        bonds: List[Dict] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Compose atoms into molecule.

        Args:
            atoms: List of atom representation dicts
            bonds: Optional list of bond dicts

        Returns:
            Dict with composed vector and net charge
        """
        if not atoms:
            raise ValueError("Empty atom list")

        # Stack flat vectors
        vecs = torch.stack([a["flat"] for a in atoms])

        # Simple weighted average composition
        weights = F.softmax(torch.ones(len(atoms)), dim=0)
        composed = (vecs * weights.unsqueeze(-1)).sum(dim=0)

        # Apply composition network
        composed = self.compose_net(composed)

        # Propagate charge
        net_charge = self._propagate_charge(atoms)

        return {
            "composed_vector": composed,
            "net_charge": net_charge,
        }

    def _propagate_charge(self, atoms: List[Dict[str, torch.Tensor]]) -> float:
        """
        Propagate charge through atoms.

        Args:
            atoms: List of atom representations

        Returns:
            Net charge value
        """
        if not atoms:
            return 0.0

        total = 0.0
        weight_sum = 0.0

        for atom in atoms:
            p = atom.get("polarity", torch.tensor(0.0))
            m = atom.get("magnitude", torch.tensor(1.0))
            s = atom.get("stability", torch.tensor(1.0))

            if isinstance(p, torch.Tensor):
                p = p.mean().item()
            if isinstance(m, torch.Tensor):
                m = m.mean().item()
            if isinstance(s, torch.Tensor):
                s = s.mean().item()

            total += p * m * s
            weight_sum += 1.0

        return total / weight_sum if weight_sum > 0 else 0.0


class AttentiveCompositionNetwork(nn.Module):
    """
    Composition network with self-attention.

    Uses multi-head attention to compute importance weights
    for each atom in the composition.
    """

    def __init__(
        self,
        config: AtomConfig,
        num_heads: int = 4,
        dropout: float = 0.1,
    ):
        """
        Initialize attentive composition network.

        Args:
            config: Atom configuration
            num_heads: Number of attention heads
            dropout: Dropout rate
        """
        super().__init__()
        self.config = config

        # Self-attention
        self.self_attn = nn.MultiheadAttention(
            embed_dim=config.total_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )

        # Layer norm
        self.norm1 = nn.LayerNorm(config.total_dim)
        self.norm2 = nn.LayerNorm(config.total_dim)

        # FFN
        self.ffn = nn.Sequential(
            nn.Linear(config.total_dim, config.total_dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(config.total_dim * 4, config.total_dim),
            nn.Dropout(dropout),
        )

        # Pooling attention
        self.pool_query = nn.Parameter(torch.randn(1, 1, config.total_dim))
        self.pool_attn = nn.MultiheadAttention(
            embed_dim=config.total_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )

    def forward(
        self,
        atoms: torch.Tensor,
        attention_mask: torch.Tensor = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Compose atoms with attention.

        Args:
            atoms: Atom representations (batch_size, num_atoms, dim)
            attention_mask: Optional attention mask

        Returns:
            Dict with composed vector and attention weights
        """
        # Self-attention
        attn_out, attn_weights = self.self_attn(
            atoms, atoms, atoms,
            key_padding_mask=attention_mask,
        )
        atoms = self.norm1(atoms + attn_out)

        # FFN
        ffn_out = self.ffn(atoms)
        atoms = self.norm2(atoms + ffn_out)

        # Pool to single vector
        batch_size = atoms.shape[0]
        query = self.pool_query.expand(batch_size, -1, -1)
        pooled, pool_weights = self.pool_attn(
            query, atoms, atoms,
            key_padding_mask=attention_mask,
        )
        pooled = pooled.squeeze(1)

        return {
            "composed_vector": pooled,
            "attention_weights": attn_weights,
            "pool_weights": pool_weights,
        }


class HierarchicalCompositionNetwork(nn.Module):
    """
    Hierarchical composition at multiple granularities.

    Composes atoms into phrases, phrases into sentences, etc.
    """

    def __init__(
        self,
        config: AtomConfig,
        num_levels: int = 3,
    ):
        """
        Initialize hierarchical composition.

        Args:
            config: Atom configuration
            num_levels: Number of hierarchy levels
        """
        super().__init__()
        self.config = config
        self.num_levels = num_levels

        # Level-specific composition networks
        self.level_composers = nn.ModuleList([
            nn.Sequential(
                nn.Linear(config.total_dim, config.total_dim),
                nn.LayerNorm(config.total_dim),
                nn.GELU(),
            )
            for _ in range(num_levels)
        ])

        # Pooling for each level
        self.level_poolers = nn.ModuleList([
            nn.AdaptiveAvgPool1d(1) for _ in range(num_levels)
        ])

    def forward(
        self,
        atoms: torch.Tensor,
        level_boundaries: List[List[int]] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Compose hierarchically.

        Args:
            atoms: Atom representations (batch_size, num_atoms, dim)
            level_boundaries: Optional boundaries for each level

        Returns:
            Dict with composed vectors at each level
        """
        batch_size, num_atoms, dim = atoms.shape

        # Default: equal-sized chunks at each level
        if level_boundaries is None:
            level_boundaries = []
            chunk_size = num_atoms
            for level in range(self.num_levels):
                chunk_size = max(1, chunk_size // 2)
                boundaries = list(range(0, num_atoms, chunk_size))
                boundaries.append(num_atoms)
                level_boundaries.append(boundaries)

        outputs = {"level_vectors": []}

        current_repr = atoms
        for level, (composer, pooler) in enumerate(
            zip(self.level_composers, self.level_poolers)
        ):
            # Apply composition
            composed = composer(current_repr)

            # Pool within chunks
            if level < len(level_boundaries):
                bounds = level_boundaries[level]
                chunks = []
                for i in range(len(bounds) - 1):
                    start, end = bounds[i], bounds[i + 1]
                    chunk = composed[:, start:end, :]
                    # Pool
                    chunk = chunk.transpose(1, 2)
                    pooled = pooler(chunk)
                    pooled = pooled.transpose(1, 2)
                    chunks.append(pooled)
                if chunks:
                    current_repr = torch.cat(chunks, dim=1)
                else:
                    current_repr = composed
            else:
                current_repr = composed

            outputs["level_vectors"].append(current_repr.mean(dim=1))

        # Final composed vector is from top level
        outputs["composed_vector"] = outputs["level_vectors"][-1]

        return outputs
