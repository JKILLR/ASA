"""
ASE Loss Functions - Combined losses for training ASA v1.1.

Includes:
- Alignment loss (composed vector vs teacher embedding)
- Charge loss (net charge vs sentiment)
- Belief resistance loss (ionization energy training)
- Thermodynamic consistency losses
- Catalyst supervision loss
"""

from typing import Dict, List

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..thermodynamics.learnable import LearnableThermodynamics
from ..thermodynamics.losses import ThermodynamicsLoss
from ..bonding.catalyst import LearnedCatalystDetector, CatalystSupervisionLoss


class ASELossV1_1(nn.Module):
    """
    Combined loss with thermodynamic and catalyst terms (v1.1).

    Loss components:
    - alignment: Composed vector similarity to teacher embeddings
    - charge: Net charge accuracy for sentiment
    - belief_resistance: Ionization energy behavior
    - hysteresis_consistency: Promotion > demotion thresholds
    - phase_separation: Clear phase boundaries
    - catalyst_supervision: Known catalyst reproduction
    """

    def __init__(
        self,
        thermo: LearnableThermodynamics,
        catalyst_detector: LearnedCatalystDetector,
        lambdas: Dict[str, float] = None,
    ):
        """
        Initialize combined loss.

        Args:
            thermo: Thermodynamics module
            catalyst_detector: Catalyst detector module
            lambdas: Loss component weights
        """
        super().__init__()
        self.thermo = thermo
        self.catalyst_detector = catalyst_detector
        self.thermo_loss = ThermodynamicsLoss()
        self.catalyst_loss = CatalystSupervisionLoss(catalyst_detector)

        self.lambdas = lambdas or {
            "alignment": 1.0,
            "charge": 0.5,
            "belief_resistance": 0.3,
            "hysteresis_consistency": 0.1,
            "phase_separation": 0.1,
            "catalyst_supervision": 0.2,
        }

    def forward(
        self,
        output: Dict[str, torch.Tensor],
        targets: Dict[str, torch.Tensor],
    ) -> Dict[str, torch.Tensor]:
        """
        Compute combined loss.

        Args:
            output: Model output dict
            targets: Target dict with optional keys:
                - sentence_embedding: Teacher embeddings
                - sentiment_label: Binary sentiment labels
                - belief_resistance_data: Dict for belief resistance
                - tokens: Token strings for catalyst supervision

        Returns:
            Dict with individual losses and total
        """
        losses = {}

        # Alignment loss: composed vector vs teacher embedding
        if "sentence_embedding" in targets:
            cos_sim = F.cosine_similarity(
                output["composed"],
                targets["sentence_embedding"],
                dim=-1,
            )
            losses["alignment"] = (1 - cos_sim).mean()

        # Charge loss: net charge vs sentiment
        if "sentiment_label" in targets:
            # Convert 0/1 labels to -1/+1 target charges
            target_charge = targets["sentiment_label"].float() * 2 - 1
            losses["charge"] = F.mse_loss(output["net_charges"], target_charge)

        # Belief resistance loss
        if "belief_resistance_data" in targets:
            brd = targets["belief_resistance_data"]
            losses["belief_resistance"] = self.thermo_loss.belief_resistance_loss(
                self.thermo,
                brd["strength"],
                brd["evidence"],
                brd["shell_level"],
                brd["should_update"],
            )

        # Thermodynamic consistency losses
        losses["hysteresis_consistency"] = self.thermo_loss.hysteresis_consistency_loss(
            self.thermo
        )
        losses["phase_separation"] = self.thermo_loss.phase_separation_loss(
            self.thermo
        )

        # Catalyst supervision
        if "tokens" in targets and "atom_data" in output:
            flat = output["atom_data"]["flat"]
            if flat.dim() == 3:
                # Flatten batch and sequence
                flat = flat.view(-1, flat.shape[-1])

            # Handle nested token lists
            tokens = targets["tokens"]
            if isinstance(tokens[0], list):
                tokens_flat = [t for batch in tokens for t in batch]
            else:
                tokens_flat = tokens

            # Match dimensions
            min_len = min(len(tokens_flat), flat.shape[0])
            losses["catalyst_supervision"] = self.catalyst_loss(
                flat[:min_len], tokens_flat[:min_len]
            )

        # Weighted sum
        total = torch.tensor(0.0, device=self._get_device(output))
        for name, loss in losses.items():
            if isinstance(loss, torch.Tensor):
                weight = self.lambdas.get(name, 1.0)
                total = total + weight * loss

        losses["total"] = total

        return losses

    def _get_device(self, output: Dict) -> torch.device:
        """Get device from output tensors."""
        for v in output.values():
            if isinstance(v, torch.Tensor):
                return v.device
            elif isinstance(v, dict):
                for vv in v.values():
                    if isinstance(vv, torch.Tensor):
                        return vv.device
        return torch.device("cpu")


class ContrastiveLoss(nn.Module):
    """
    Contrastive loss for learning atomic similarities.

    Pulls similar atoms together, pushes different atoms apart.
    """

    def __init__(self, margin: float = 0.5, temperature: float = 0.07):
        """
        Initialize contrastive loss.

        Args:
            margin: Margin for triplet loss
            temperature: Temperature for InfoNCE
        """
        super().__init__()
        self.margin = margin
        self.temperature = temperature

    def triplet_loss(
        self,
        anchor: torch.Tensor,
        positive: torch.Tensor,
        negative: torch.Tensor,
    ) -> torch.Tensor:
        """
        Triplet margin loss.

        Args:
            anchor: Anchor embeddings
            positive: Positive (similar) embeddings
            negative: Negative (dissimilar) embeddings

        Returns:
            Triplet loss
        """
        pos_dist = F.pairwise_distance(anchor, positive)
        neg_dist = F.pairwise_distance(anchor, negative)
        loss = F.relu(pos_dist - neg_dist + self.margin)
        return loss.mean()

    def infonce_loss(
        self,
        query: torch.Tensor,
        positive_key: torch.Tensor,
        negative_keys: torch.Tensor = None,
    ) -> torch.Tensor:
        """
        InfoNCE contrastive loss.

        Args:
            query: Query embeddings (batch, dim)
            positive_key: Positive key embeddings (batch, dim)
            negative_keys: Negative keys (batch, num_neg, dim) or None for in-batch

        Returns:
            InfoNCE loss
        """
        query = F.normalize(query, dim=-1)
        positive_key = F.normalize(positive_key, dim=-1)

        # Positive logits
        pos_logits = (query * positive_key).sum(dim=-1) / self.temperature

        if negative_keys is None:
            # In-batch negatives
            all_logits = torch.matmul(query, positive_key.T) / self.temperature
            labels = torch.arange(query.shape[0], device=query.device)
            return F.cross_entropy(all_logits, labels)
        else:
            # Explicit negatives
            negative_keys = F.normalize(negative_keys, dim=-1)
            neg_logits = torch.matmul(
                query.unsqueeze(1), negative_keys.transpose(-1, -2)
            ).squeeze(1) / self.temperature

            logits = torch.cat([pos_logits.unsqueeze(-1), neg_logits], dim=-1)
            labels = torch.zeros(query.shape[0], dtype=torch.long, device=query.device)
            return F.cross_entropy(logits, labels)


class ChargeConsistencyLoss(nn.Module):
    """
    Loss for charge consistency in compositions.

    Ensures that charge propagation follows expected rules.
    """

    def __init__(self):
        super().__init__()

    def forward(
        self,
        atom_charges: torch.Tensor,
        molecule_charge: torch.Tensor,
        bond_strengths: torch.Tensor = None,
    ) -> torch.Tensor:
        """
        Compute charge consistency loss.

        Args:
            atom_charges: Individual atom charges (batch, num_atoms)
            molecule_charge: Composed molecule charge (batch,)
            bond_strengths: Optional bond strengths for weighting

        Returns:
            Consistency loss
        """
        # Simple version: molecule charge should be mean of atom charges
        expected = atom_charges.mean(dim=-1)
        return F.mse_loss(molecule_charge, expected)


class ValenceConstraintLoss(nn.Module):
    """
    Loss for enforcing valence constraints.

    Penalizes over/under-bonding.
    """

    def __init__(self, penalty_weight: float = 0.5):
        """
        Initialize valence constraint loss.

        Args:
            penalty_weight: Weight for constraint violations
        """
        super().__init__()
        self.penalty_weight = penalty_weight

    def forward(
        self,
        valence_counts: torch.Tensor,
        bond_counts: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute valence constraint loss.

        Args:
            valence_counts: Expected valence per atom (batch, num_atoms)
            bond_counts: Actual bond count per atom (batch, num_atoms)

        Returns:
            Constraint violation loss
        """
        # Penalize both over and under bonding
        diff = bond_counts - valence_counts
        over_bonding = F.relu(diff)
        under_bonding = F.relu(-diff)

        loss = over_bonding.sum() + self.penalty_weight * under_bonding.sum()
        return loss / valence_counts.numel()
