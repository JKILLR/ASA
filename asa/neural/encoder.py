"""
AtomicEncoder - Encodes tokens into structured semantic atoms.

Produces:
- Nuclear vector (core identity)
- Shell vectors (relational features)
- Charge components (polarity, magnitude, stability)
- Valence vectors (bonding interfaces)
"""

from typing import Dict

import torch
import torch.nn as nn

from ..core.config import AtomConfig


class AtomicEncoder(nn.Module):
    """
    Encodes tokens into structured semantic atoms.

    Architecture:
    - Shared base embedding layer
    - Separate heads for nuclear, shells, charge, valence
    - Produces complete atomic representation
    """

    def __init__(self, config: AtomConfig):
        """
        Initialize encoder.

        Args:
            config: Atom configuration
        """
        super().__init__()
        self.config = config

        # Base embedding
        self.embedding = nn.Embedding(config.vocab_size, 256)

        # Nuclear encoder
        self.nuclear_encoder = nn.Sequential(
            nn.Linear(256, 128),
            nn.LayerNorm(128),
            nn.GELU(),
            nn.Linear(128, config.nuclear_dim),
        )

        # Shell encoders (one per shell)
        self.shell_encoders = nn.ModuleList([
            nn.Sequential(
                nn.Linear(256, dim * 2),
                nn.LayerNorm(dim * 2),
                nn.GELU(),
                nn.Linear(dim * 2, dim),
            )
            for dim in config.shell_dims
        ])

        # Charge encoder
        self.charge_encoder = nn.Sequential(
            nn.Linear(256, 64),
            nn.GELU(),
            nn.Linear(64, config.charge_dim),
        )
        self.polarity_head = nn.Linear(config.charge_dim, 1)
        self.magnitude_head = nn.Linear(config.charge_dim, 1)
        self.stability_head = nn.Linear(config.charge_dim, 1)

        # Valence encoder
        self.valence_head = nn.Sequential(
            nn.Linear(256, 32),
            nn.GELU(),
            nn.Linear(32, config.valence_max + 1),  # +1 for 0 valence
        )

        # Bond vector encoder
        self.bond_encoder = nn.Sequential(
            nn.Linear(256, 128),
            nn.GELU(),
            nn.Linear(128, config.valence_max * config.bond_dim),
        )

    def forward(self, token_ids: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Encode tokens into atomic representations.

        Args:
            token_ids: Token IDs of shape (batch, seq_len) or (seq_len,)

        Returns:
            Dict with all atomic components
        """
        squeeze = False
        if token_ids.dim() == 1:
            token_ids = token_ids.unsqueeze(0)
            squeeze = True

        batch_size, seq_len = token_ids.shape

        # Base embedding
        base = self.embedding(token_ids)

        # Nuclear
        nuclear = self.nuclear_encoder(base)

        # Shells
        shells = [enc(base) for enc in self.shell_encoders]

        # Charge
        charge_vec = self.charge_encoder(base)
        polarity = torch.tanh(self.polarity_head(charge_vec)).squeeze(-1)
        magnitude = torch.sigmoid(self.magnitude_head(charge_vec)).squeeze(-1)
        stability = torch.sigmoid(self.stability_head(charge_vec)).squeeze(-1)

        # Valence
        valence_logits = self.valence_head(base)
        valence_count = valence_logits.argmax(dim=-1)

        # Bond vectors
        bond_vecs = self.bond_encoder(base).view(
            batch_size, seq_len, self.config.valence_max, self.config.bond_dim
        )

        # Flat vector (nuclear + all shells)
        flat = torch.cat([nuclear] + shells, dim=-1)

        output = {
            "nuclear": nuclear,
            "shells": shells,
            "charge_vector": charge_vec,
            "polarity": polarity,
            "magnitude": magnitude,
            "stability": stability,
            "valence_logits": valence_logits,
            "valence_count": valence_count,
            "bond_vectors": bond_vecs,
            "flat": flat,
        }

        if squeeze:
            output = {
                k: v.squeeze(0) if isinstance(v, torch.Tensor) else [s.squeeze(0) for s in v]
                for k, v in output.items()
            }

        return output

    def encode_single(self, token_id: int) -> Dict[str, torch.Tensor]:
        """
        Encode a single token.

        Args:
            token_id: Token ID

        Returns:
            Atomic representation dict
        """
        token_ids = torch.tensor([token_id])
        result = self.forward(token_ids)
        return {
            k: v.squeeze(0) if isinstance(v, torch.Tensor) else [s.squeeze(0) for s in v]
            for k, v in result.items()
        }


class ContextualAtomicEncoder(nn.Module):
    """
    Atomic encoder with contextual attention.

    Produces context-aware atomic representations by attending
    to surrounding tokens.
    """

    def __init__(
        self,
        config: AtomConfig,
        num_heads: int = 4,
        num_layers: int = 2,
    ):
        """
        Initialize contextual encoder.

        Args:
            config: Atom configuration
            num_heads: Number of attention heads
            num_layers: Number of transformer layers
        """
        super().__init__()
        self.config = config

        # Base encoder
        self.base_encoder = AtomicEncoder(config)

        # Contextual layers
        self.context_layers = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=config.total_dim,
                nhead=num_heads,
                dim_feedforward=config.total_dim * 4,
                dropout=0.1,
                activation="gelu",
                batch_first=True,
            ),
            num_layers=num_layers,
        )

        # Output projection
        self.output_proj = nn.Linear(config.total_dim, config.total_dim)

    def forward(
        self,
        token_ids: torch.Tensor,
        attention_mask: torch.Tensor = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Encode with context.

        Args:
            token_ids: Token IDs
            attention_mask: Optional attention mask

        Returns:
            Context-aware atomic representations
        """
        # Get base representations
        base_output = self.base_encoder(token_ids)

        # Apply contextual attention
        flat = base_output["flat"]
        contextualized = self.context_layers(
            flat,
            src_key_padding_mask=attention_mask,
        )
        contextualized = self.output_proj(contextualized)

        # Update flat representation
        base_output["flat"] = contextualized
        base_output["contextual"] = True

        return base_output
