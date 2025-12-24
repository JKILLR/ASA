"""
LearnableThermodynamics - All thermodynamic constants as trainable parameters.

v1.1 Change: All thermodynamic constants are now trainable nn.Parameter
with physical constraints enforced via clamping.

Key parameters:
- Temperature effects: Bond threshold and charge volatility
- Ionization energy: Belief resistance based on shell depth
- Hysteresis: Promotion/demotion thresholds for shell migration
- Phase transitions: Crystallization scoring weights
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..core.atoms import ConceptPhase


class LearnableThermodynamics(nn.Module):
    """
    All thermodynamic constants as learnable parameters.

    Initialized with spec values, but trainable through backprop.
    Constraints enforced via clamping/sigmoid to maintain physical validity.

    This enables the system to learn optimal values for:
    - Bond formation thresholds at different temperatures
    - Ionization energies for belief resistance
    - Shell migration thresholds (promotion/demotion)
    - Phase transition boundaries
    - Crystallization scoring weights
    """

    def __init__(self):
        super().__init__()

        # === TEMPERATURE PARAMETERS ===
        # bond_threshold = base - (scale * temperature)
        self._threshold_base = nn.Parameter(torch.tensor(0.9))
        self._threshold_scale = nn.Parameter(torch.tensor(0.8))

        # charge_volatility = base + (scale * temperature)
        self._volatility_base = nn.Parameter(torch.tensor(0.2))
        self._volatility_scale = nn.Parameter(torch.tensor(0.8))

        # === IONIZATION PARAMETERS ===
        # ionization = base * (multiplier ^ depth)
        self._ionization_base = nn.Parameter(torch.tensor(1.0))
        self._ionization_multiplier = nn.Parameter(torch.tensor(2.5))

        # === HYSTERESIS PARAMETERS ===
        # Learned per-shell thresholds for migration
        self._promotion_thresholds = nn.Parameter(
            torch.tensor([0.85, 0.65, 0.45, 0.25])
        )
        self._demotion_thresholds = nn.Parameter(
            torch.tensor([0.55, 0.40, 0.25, 0.10])
        )

        # === PHASE TRANSITION PARAMETERS ===
        self._solid_threshold = nn.Parameter(torch.tensor(0.7))
        self._liquid_threshold = nn.Parameter(torch.tensor(0.3))

        # === CRYSTALLIZATION WEIGHTS ===
        # score = w1*occupancy + w2*strength + w3*stability + w4*concentration
        self._crystal_weights = nn.Parameter(
            torch.tensor([0.25, 0.30, 0.25, 0.20])
        )

        # === CAPACITY PRESSURE ===
        self._capacity_pressure = nn.Parameter(torch.tensor(1.0))

    # === CONSTRAINED PROPERTY ACCESS ===

    @property
    def threshold_base(self) -> torch.Tensor:
        """Bond threshold base, clamped to [0.5, 1.0]."""
        return torch.clamp(self._threshold_base, 0.5, 1.0)

    @property
    def threshold_scale(self) -> torch.Tensor:
        """Bond threshold scale, clamped to [0.3, 0.9]."""
        return torch.clamp(self._threshold_scale, 0.3, 0.9)

    @property
    def ionization_base(self) -> torch.Tensor:
        """Ionization base energy, must be positive."""
        return F.softplus(self._ionization_base)

    @property
    def ionization_multiplier(self) -> torch.Tensor:
        """Ionization multiplier per shell, clamped to [1.5, 4.0]."""
        return torch.clamp(self._ionization_multiplier, 1.5, 4.0)

    @property
    def promotion_thresholds(self) -> torch.Tensor:
        """Promotion thresholds, sorted descending and clamped."""
        raw = torch.sigmoid(self._promotion_thresholds)
        sorted_desc, _ = torch.sort(raw, descending=True)
        return sorted_desc

    @property
    def demotion_thresholds(self) -> torch.Tensor:
        """Demotion thresholds, must be below promotion thresholds."""
        promo = self.promotion_thresholds
        raw = torch.sigmoid(self._demotion_thresholds)
        return torch.clamp(raw, max=promo - 0.1)

    @property
    def solid_threshold(self) -> torch.Tensor:
        """Solid phase threshold."""
        return torch.clamp(self._solid_threshold, 0.5, 0.9)

    @property
    def liquid_threshold(self) -> torch.Tensor:
        """Liquid phase threshold."""
        return torch.clamp(self._liquid_threshold, 0.1, self.solid_threshold - 0.1)

    @property
    def crystal_weights(self) -> torch.Tensor:
        """Crystallization weights, normalized to sum to 1."""
        return F.softmax(self._crystal_weights, dim=0)

    @property
    def capacity_pressure(self) -> torch.Tensor:
        """Capacity pressure coefficient, must be positive."""
        return F.softplus(self._capacity_pressure)

    # === COMPUTED VALUES ===

    def bond_threshold(self, temperature: float) -> torch.Tensor:
        """
        Compute bond threshold at given temperature.

        Higher temperature = lower threshold = easier bonding.

        Args:
            temperature: Temperature value [0, 1]

        Returns:
            Bond formation threshold
        """
        return self.threshold_base - (self.threshold_scale * temperature)

    def charge_volatility(self, temperature: float) -> torch.Tensor:
        """
        Compute charge volatility at given temperature.

        Higher temperature = higher volatility = easier charge changes.

        Args:
            temperature: Temperature value [0, 1]

        Returns:
            Charge volatility coefficient
        """
        base = torch.clamp(self._volatility_base, 0.1, 0.5)
        scale = torch.clamp(self._volatility_scale, 0.3, 0.9)
        return base + (scale * temperature)

    def ionization_energy(
        self, shell_level: int, max_shells: int = 4
    ) -> torch.Tensor:
        """
        Compute ionization energy for a shell level.

        Inner shells have higher ionization energy (harder to modify).
        This implements the "belief resistance" mechanism.

        Args:
            shell_level: Shell index (0 = innermost)
            max_shells: Maximum number of shells

        Returns:
            Ionization energy for this shell
        """
        depth = max_shells - 1 - shell_level
        return self.ionization_base * (self.ionization_multiplier ** depth)

    def crystallization_score(
        self,
        occupancy: float,
        avg_strength: float,
        charge_stability: float,
        inner_concentration: float,
    ) -> torch.Tensor:
        """
        Compute crystallization score with learned weights.

        Higher score = more crystallized (solid-like).

        Args:
            occupancy: Fraction of shell capacity used
            avg_strength: Average association strength
            charge_stability: Charge stability value
            inner_concentration: Fraction of associations in inner shells

        Returns:
            Crystallization score [0, 1]
        """
        features = torch.tensor(
            [occupancy, avg_strength, charge_stability, inner_concentration]
        )
        return (self.crystal_weights * features).sum()

    def determine_phase(self, crystal_score: torch.Tensor) -> ConceptPhase:
        """
        Determine phase from crystallization score.

        Args:
            crystal_score: Crystallization score

        Returns:
            ConceptPhase (GAS, LIQUID, or SOLID)
        """
        if crystal_score >= self.solid_threshold:
            return ConceptPhase.SOLID
        elif crystal_score >= self.liquid_threshold:
            return ConceptPhase.LIQUID
        else:
            return ConceptPhase.GAS

    def can_promote(self, current_level: int, strength: float) -> bool:
        """
        Check if association can move to inner shell.

        Args:
            current_level: Current shell level
            strength: Association strength

        Returns:
            True if promotion is allowed
        """
        if current_level == 0:
            return False  # Already at innermost
        thresh = self.promotion_thresholds[current_level - 1].item()
        return strength >= thresh

    def should_demote(
        self, current_level: int, strength: float, max_shells: int = 4
    ) -> bool:
        """
        Check if association should move to outer shell.

        Args:
            current_level: Current shell level
            strength: Association strength
            max_shells: Maximum number of shells

        Returns:
            True if demotion should occur
        """
        if current_level >= max_shells - 1:
            return False  # Already at outermost
        thresh = self.demotion_thresholds[current_level].item()
        return strength < thresh

    def get_state_dict_readable(self) -> dict:
        """Get human-readable state of all parameters."""
        return {
            "threshold_base": self.threshold_base.item(),
            "threshold_scale": self.threshold_scale.item(),
            "ionization_base": self.ionization_base.item(),
            "ionization_multiplier": self.ionization_multiplier.item(),
            "promotion_thresholds": self.promotion_thresholds.tolist(),
            "demotion_thresholds": self.demotion_thresholds.tolist(),
            "solid_threshold": self.solid_threshold.item(),
            "liquid_threshold": self.liquid_threshold.item(),
            "crystal_weights": self.crystal_weights.tolist(),
        }
