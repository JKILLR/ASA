"""
LearnedCatalystDetector - Neural network for detecting catalyst effects.

v1.1 Change: Catalysts are detected by a neural network trained with
symbolic supervision, enabling cross-lingual generalization.

Catalysts are words that facilitate or modify bond formation:
- "because" facilitates causal bonds
- "but" facilitates contrastive bonds
- "like" facilitates analogical bonds
"""

from typing import Dict, List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class LearnedCatalystDetector(nn.Module):
    """
    Learns to detect catalyst effects from atom embeddings.

    Uses hard-coded catalysts as SUPERVISION signal, then generalizes
    to unseen words based on embedding patterns. This enables:
    - Cross-lingual transfer (e.g., learns "because" → understands "porque")
    - Novel catalyst discovery
    - Graduated catalyst effects

    Outputs:
    - threshold_multipliers: How much to lower bonding thresholds per bond type
    - strength_bonuses: Bonus strength to add to formed bonds
    - scope_probs: How far the catalyst effect extends
    - is_catalyst: Overall catalyst score
    """

    def __init__(
        self,
        atom_dim: int,
        num_bond_types: int = 5,
        hidden_dim: int = 64,
        known_catalysts: Dict[str, Dict[str, float]] = None,
    ):
        """
        Initialize catalyst detector.

        Args:
            atom_dim: Dimension of input atom embeddings
            num_bond_types: Number of bond types to predict effects for
            hidden_dim: Hidden dimension
            known_catalysts: Dict of known catalysts for supervision
        """
        super().__init__()

        self.atom_dim = atom_dim
        self.num_bond_types = num_bond_types

        # Main detector network
        self.detector = nn.Sequential(
            nn.Linear(atom_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, num_bond_types * 2),  # threshold_mult + strength_bonus
        )

        # Scope predictor: adjacent, sentence, paragraph
        self.scope_predictor = nn.Sequential(
            nn.Linear(atom_dim, 32),
            nn.GELU(),
            nn.Linear(32, 3),
        )

        self.known_catalysts = known_catalysts or self._default_catalysts()
        self.bond_types = [
            "causal",
            "inferential",
            "analogical",
            "contrastive",
            "general",
        ]

    def _default_catalysts(self) -> Dict[str, Dict[str, float]]:
        """
        Default catalyst definitions for supervision.

        Values are threshold multipliers (lower = easier bonding).
        """
        return {
            # Causal connectives
            "because": {"causal": 0.6, "inferential": 0.8},
            "since": {"causal": 0.6, "inferential": 0.7},
            "therefore": {"inferential": 0.5, "causal": 0.7},
            "thus": {"inferential": 0.6},
            "hence": {"inferential": 0.6},
            "so": {"causal": 0.7, "inferential": 0.8},
            "consequently": {"inferential": 0.5, "causal": 0.6},
            "as": {"causal": 0.7, "analogical": 0.5},
            # Analogical connectives
            "like": {"analogical": 0.4},
            "similarly": {"analogical": 0.5},
            "likewise": {"analogical": 0.5},
            "compared": {"analogical": 0.6},
            "resembles": {"analogical": 0.5},
            # Contrastive connectives
            "but": {"contrastive": 0.5},
            "however": {"contrastive": 0.5},
            "although": {"contrastive": 0.6},
            "yet": {"contrastive": 0.6},
            "whereas": {"contrastive": 0.5},
            "while": {"contrastive": 0.7},
            "despite": {"contrastive": 0.5},
            "nevertheless": {"contrastive": 0.5},
            # General connectives
            "and": {"general": 0.9},
            "or": {"general": 0.85},
            "with": {"general": 0.8},
            "also": {"general": 0.8},
            "moreover": {"general": 0.7, "inferential": 0.8},
        }

    def forward(self, atom_embedding: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Detect catalyst effects for an atom.

        Args:
            atom_embedding: Atom embedding of shape (..., atom_dim)

        Returns:
            Dict with:
            - threshold_multipliers: (batch, num_bond_types) - multiply threshold
            - strength_bonuses: (batch, num_bond_types) - add to strength
            - scope_probs: (batch, 3) - adjacent/sentence/paragraph
            - is_catalyst: (batch,) - overall catalyst score
        """
        squeeze = atom_embedding.dim() == 1
        if squeeze:
            atom_embedding = atom_embedding.unsqueeze(0)

        raw = self.detector(atom_embedding)

        # Split into threshold multipliers and strength bonuses
        threshold_raw = raw[..., : self.num_bond_types]
        strength_raw = raw[..., self.num_bond_types :]

        # Threshold multipliers: [0.3, 1.0] where < 1.0 makes bonding easier
        threshold_mult = 0.3 + 0.7 * torch.sigmoid(threshold_raw)

        # Strength bonuses: [0, 0.3]
        strength_bonus = 0.3 * torch.sigmoid(strength_raw)

        # Scope probabilities
        scope_logits = self.scope_predictor(atom_embedding)
        scope_probs = F.softmax(scope_logits, dim=-1)

        # Overall catalyst score: deviation from neutral (1.0)
        catalyst_score = (1.0 - threshold_mult).abs().mean(dim=-1)

        result = {
            "threshold_multipliers": threshold_mult,
            "strength_bonuses": strength_bonus,
            "scope_probs": scope_probs,
            "is_catalyst": catalyst_score,
        }

        if squeeze:
            result = {k: v.squeeze(0) for k, v in result.items()}

        return result

    def get_effect_for_bond_type(
        self, atom_embedding: torch.Tensor, bond_type_idx: int
    ) -> Tuple[float, float]:
        """
        Get (threshold_multiplier, strength_bonus) for specific bond type.

        Args:
            atom_embedding: Atom embedding
            bond_type_idx: Index of bond type

        Returns:
            (threshold_multiplier, strength_bonus)
        """
        effects = self.forward(atom_embedding)
        return (
            effects["threshold_multipliers"][..., bond_type_idx].item(),
            effects["strength_bonuses"][..., bond_type_idx].item(),
        )

    def get_known_effect(self, token: str) -> Dict[str, float]:
        """
        Get known catalyst effect for a token (if any).

        Args:
            token: Token string

        Returns:
            Dict of bond_type -> threshold_multiplier
        """
        return self.known_catalysts.get(token.lower(), {})


class CatalystSupervisionLoss(nn.Module):
    """
    Train catalyst detector to reproduce known catalysts.

    Uses hard-coded catalysts as supervision signal while allowing
    the network to generalize to unseen patterns.
    """

    def __init__(self, detector: LearnedCatalystDetector):
        """
        Initialize supervision loss.

        Args:
            detector: Catalyst detector to train
        """
        super().__init__()
        self.detector = detector
        self.bond_type_to_idx = {
            bt: i for i, bt in enumerate(detector.bond_types)
        }

    def forward(
        self,
        atom_embeddings: torch.Tensor,
        tokens: List[str],
    ) -> torch.Tensor:
        """
        Compute supervision loss.

        Args:
            atom_embeddings: Batch of atom embeddings (batch_size, atom_dim)
            tokens: Corresponding tokens

        Returns:
            Supervision loss
        """
        effects = self.detector(atom_embeddings)

        losses = []

        for i, token in enumerate(tokens):
            token_lower = token.lower()

            if token_lower in self.detector.known_catalysts:
                # Known catalyst: supervise toward known effects
                known_effects = self.detector.known_catalysts[token_lower]

                for bond_type, target_mult in known_effects.items():
                    if bond_type in self.bond_type_to_idx:
                        idx = self.bond_type_to_idx[bond_type]
                        pred_mult = effects["threshold_multipliers"][i, idx]
                        losses.append((pred_mult - target_mult) ** 2)
            else:
                # Non-catalyst: should have neutral effect (~1.0)
                neutral_loss = (
                    (effects["threshold_multipliers"][i] - 1.0).pow(2).mean()
                )
                # Weight non-catalyst loss lower to allow discovery
                losses.append(neutral_loss * 0.1)

        if not losses:
            return torch.tensor(0.0, device=atom_embeddings.device)

        return torch.stack(losses).mean()


class CatalystAwareBonder(nn.Module):
    """
    Bond formation with learned catalyst effects.

    Wraps a base bonding network and applies catalyst effects
    from context atoms.
    """

    def __init__(
        self,
        base_bonder: nn.Module,
        catalyst_detector: LearnedCatalystDetector,
    ):
        """
        Initialize catalyst-aware bonder.

        Args:
            base_bonder: Base bonding network
            catalyst_detector: Catalyst detector
        """
        super().__init__()
        self.bonder = base_bonder
        self.catalyst_detector = catalyst_detector

    def forward(
        self,
        atom_a: Dict[str, torch.Tensor],
        atom_b: Dict[str, torch.Tensor],
        context_atoms: List[Dict[str, torch.Tensor]],
        base_threshold: float = 0.5,
    ) -> Dict[str, torch.Tensor]:
        """
        Predict bond with catalyst effects.

        Args:
            atom_a: First atom data
            atom_b: Second atom data
            context_atoms: List of context atom data (potential catalysts)
            base_threshold: Base bond threshold

        Returns:
            Dict with bond prediction and catalyst info
        """
        # Get base bond prediction
        base_pred = self.bonder(atom_a, atom_b)
        base_strength = base_pred["strength"]
        bond_type_idx = base_pred["type_logits"].argmax(dim=-1).item()

        # Accumulate catalyst effects
        threshold_mult = 1.0
        strength_bonus = 0.0
        catalysts_found = []

        for ctx_atom in context_atoms:
            effects = self.catalyst_detector(ctx_atom["flat"])

            if effects["is_catalyst"].item() > 0.1:
                threshold_mult *= effects["threshold_multipliers"][
                    bond_type_idx
                ].item()
                strength_bonus += effects["strength_bonuses"][
                    bond_type_idx
                ].item()
                catalysts_found.append(
                    {
                        "is_catalyst": effects["is_catalyst"].item(),
                        "effect": 1.0 - threshold_mult,
                    }
                )

        # Apply effects
        effective_threshold = base_threshold * threshold_mult
        effective_strength = torch.clamp(base_strength + strength_bonus, 0.0, 1.0)

        can_bond = (effective_strength > effective_threshold).float()

        return {
            "strength": effective_strength,
            "type_logits": base_pred["type_logits"],
            "can_bond": can_bond,
            "threshold_used": effective_threshold,
            "catalyst_effect": 1.0 - threshold_mult,
            "catalysts_found": len(catalysts_found),
        }
