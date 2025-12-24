"""
ThermodynamicsLoss - Loss functions to train thermodynamic parameters.

Key insight: We supervise the BEHAVIOR, not the constants directly.
The system learns appropriate ionization energies by training on
belief update tasks, learning thresholds by training on bonding tasks, etc.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from .learnable import LearnableThermodynamics


class ThermodynamicsLoss(nn.Module):
    """
    Loss functions to train thermodynamic parameters.

    These losses train the learnable thermodynamic constants by
    supervising the downstream behaviors they control:
    - Belief resistance: Core beliefs should resist weak contradictions
    - Hysteresis consistency: Promotion > demotion thresholds
    - Phase separation: Clear boundaries between phases
    """

    def __init__(self):
        super().__init__()

    def belief_resistance_loss(
        self,
        thermo: LearnableThermodynamics,
        core_belief_strength: torch.Tensor,
        contradictory_evidence: torch.Tensor,
        shell_level: torch.Tensor,
        should_update: torch.Tensor,
    ) -> torch.Tensor:
        """
        Train ionization energy to produce correct update behavior.

        Core beliefs (inner shell, high strength) should resist weak contradictions.
        Peripheral beliefs should update easily.

        Args:
            thermo: Thermodynamics module
            core_belief_strength: Strength of existing belief [0, 1]
            contradictory_evidence: Strength of contradicting evidence [0, 1]
            shell_level: Shell level of belief (0 = core)
            should_update: Target: 1 if belief should update, 0 otherwise

        Returns:
            Binary cross-entropy loss for update prediction
        """
        # Get ionization energy for each sample's shell level
        batch_size = shell_level.shape[0]
        ionization = torch.stack([
            thermo.ionization_energy(int(level.item()), max_shells=4)
            for level in shell_level
        ])

        # Update probability: evidence / (evidence + ionization * belief_strength)
        # High ionization + high strength = low update probability
        update_prob = contradictory_evidence / (
            contradictory_evidence + ionization * core_belief_strength + 1e-8
        )

        return F.binary_cross_entropy(update_prob, should_update.float())

    def hysteresis_consistency_loss(
        self, thermo: LearnableThermodynamics
    ) -> torch.Tensor:
        """
        Ensure promotion thresholds > demotion thresholds.

        Hysteresis requires a gap between thresholds to prevent
        oscillation between shells.

        Args:
            thermo: Thermodynamics module

        Returns:
            Loss penalizing insufficient gap between thresholds
        """
        promo = thermo.promotion_thresholds
        demo = thermo.demotion_thresholds

        # Want at least 0.15 gap between promotion and demotion
        margin_violation = F.relu(0.15 - (promo - demo))
        return margin_violation.sum()

    def phase_separation_loss(
        self, thermo: LearnableThermodynamics
    ) -> torch.Tensor:
        """
        Ensure phase thresholds are well-separated.

        Solid and liquid thresholds should have clear separation
        to avoid ambiguous phase states.

        Args:
            thermo: Thermodynamics module

        Returns:
            Loss penalizing insufficient phase separation
        """
        solid = thermo.solid_threshold
        liquid = thermo.liquid_threshold
        return F.relu(0.2 - (solid - liquid))

    def threshold_monotonicity_loss(
        self, thermo: LearnableThermodynamics
    ) -> torch.Tensor:
        """
        Ensure thresholds are properly ordered.

        Promotion thresholds should be descending (inner shells need
        higher strength to reach).

        Args:
            thermo: Thermodynamics module

        Returns:
            Loss for non-monotonic thresholds
        """
        promo = thermo.promotion_thresholds

        # Check that promo[i] > promo[i+1] (descending)
        violations = F.relu(promo[1:] - promo[:-1] + 0.05)
        return violations.sum()

    def ionization_scaling_loss(
        self, thermo: LearnableThermodynamics, expected_ratio: float = 2.5
    ) -> torch.Tensor:
        """
        Encourage consistent ionization energy scaling.

        Inner shells should require proportionally more energy.

        Args:
            thermo: Thermodynamics module
            expected_ratio: Expected ratio between shell levels

        Returns:
            Loss for deviating from expected scaling
        """
        actual_ratio = thermo.ionization_multiplier
        return (actual_ratio - expected_ratio) ** 2

    def combined_thermodynamics_loss(
        self,
        thermo: LearnableThermodynamics,
        belief_data: dict = None,
        weights: dict = None,
    ) -> torch.Tensor:
        """
        Combined loss for all thermodynamic constraints.

        Args:
            thermo: Thermodynamics module
            belief_data: Optional dict with belief resistance training data
            weights: Optional dict of loss weights

        Returns:
            Weighted sum of all thermodynamic losses
        """
        weights = weights or {
            "hysteresis": 0.3,
            "phase": 0.2,
            "monotonicity": 0.2,
            "scaling": 0.1,
            "belief": 0.2,
        }

        losses = {}

        losses["hysteresis"] = self.hysteresis_consistency_loss(thermo)
        losses["phase"] = self.phase_separation_loss(thermo)
        losses["monotonicity"] = self.threshold_monotonicity_loss(thermo)
        losses["scaling"] = self.ionization_scaling_loss(thermo)

        if belief_data is not None:
            losses["belief"] = self.belief_resistance_loss(
                thermo,
                belief_data["strength"],
                belief_data["evidence"],
                belief_data["shell_level"],
                belief_data["should_update"],
            )
        else:
            losses["belief"] = torch.tensor(0.0)

        total = sum(weights.get(k, 1.0) * v for k, v in losses.items())
        losses["total"] = total

        return losses
