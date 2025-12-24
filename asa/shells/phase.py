"""
PhaseAnalyzer - Analyzes and manages phase state of concepts.

Concepts crystallize from gas (vague) through liquid (adaptive)
to solid (stable) based on their association patterns.

Phase affects behavior:
- GAS: High bond formation rate, high context sensitivity
- LIQUID: Balanced behavior, moderate stability
- SOLID: Low bond formation, resists change
"""

from typing import Dict

from ..core.atoms import ConceptPhase, SemanticAtom
from ..thermodynamics.learnable import LearnableThermodynamics


class PhaseAnalyzer:
    """
    Analyzes and manages phase state of concepts.

    Phase is determined by a crystallization score computed from:
    - Shell occupancy
    - Average association strength
    - Charge stability
    - Inner shell concentration
    """

    def __init__(self, thermo: LearnableThermodynamics):
        """
        Initialize phase analyzer.

        Args:
            thermo: Thermodynamics module for phase thresholds
        """
        self.thermo = thermo

    def compute_crystallization_score(self, atom: SemanticAtom) -> float:
        """
        Compute crystallization score for an atom.

        Higher score = more crystallized = more solid-like.

        Args:
            atom: Semantic atom to analyze

        Returns:
            Crystallization score [0, 1]
        """
        filled, capacity = atom.shell_occupancy()
        occupancy_score = filled / capacity if capacity > 0 else 0

        # Average strength weighted toward inner shells
        total_strength = 0.0
        total_weight = 0.0
        for level, shell in enumerate(atom.shells):
            # Inner shells have higher weight
            weight = 2 ** (len(atom.shells) - level)
            for assoc in shell.associations:
                total_strength += assoc.strength * weight
                total_weight += weight
        avg_strength = total_strength / total_weight if total_weight > 0 else 0

        charge_stability = atom.charge.stability

        # Inner shell concentration
        inner_concentration = 0.0
        if filled > 0:
            inner_filled = sum(
                len(atom.shells[i].associations)
                for i in range(min(2, len(atom.shells)))
            )
            inner_concentration = inner_filled / filled

        return self.thermo.crystallization_score(
            occupancy_score, avg_strength, charge_stability, inner_concentration
        ).item()

    def determine_phase(self, atom: SemanticAtom) -> ConceptPhase:
        """
        Determine phase state of an atom.

        Args:
            atom: Semantic atom to analyze

        Returns:
            ConceptPhase (GAS, LIQUID, or SOLID)
        """
        import torch
        score = self.compute_crystallization_score(atom)
        return self.thermo.determine_phase(torch.tensor(score))

    def update_phase(self, atom: SemanticAtom) -> ConceptPhase:
        """
        Update atom's phase based on current state.

        Args:
            atom: Semantic atom to update

        Returns:
            New phase (also updates atom.phase)
        """
        new_phase = self.determine_phase(atom)
        atom.phase = new_phase
        return new_phase

    def phase_modifiers(self, phase: ConceptPhase) -> Dict[str, float]:
        """
        Get behavior modifiers based on phase.

        These modifiers affect how the atom interacts:
        - bond_formation_rate: Multiplier for bond threshold
        - bond_breaking_rate: Ease of breaking existing bonds
        - charge_volatility: How easily charge changes
        - context_sensitivity: How much context affects behavior

        Args:
            phase: Phase state

        Returns:
            Dict of modifier names to values
        """
        if phase == ConceptPhase.SOLID:
            return {
                "bond_formation_rate": 0.2,  # Hard to form new bonds
                "bond_breaking_rate": 0.1,  # Very hard to break bonds
                "charge_volatility": 0.1,  # Stable charge
                "context_sensitivity": 0.3,  # Low context influence
            }
        elif phase == ConceptPhase.LIQUID:
            return {
                "bond_formation_rate": 1.0,  # Normal bonding
                "bond_breaking_rate": 0.5,  # Moderate bond breaking
                "charge_volatility": 0.5,  # Moderate volatility
                "context_sensitivity": 1.0,  # Normal context influence
            }
        else:  # GAS
            return {
                "bond_formation_rate": 1.5,  # Easy to form bonds
                "bond_breaking_rate": 1.5,  # Easy to break bonds
                "charge_volatility": 1.0,  # High volatility
                "context_sensitivity": 2.0,  # High context influence
            }

    def get_phase_report(self, atom: SemanticAtom) -> dict:
        """
        Get detailed phase analysis report.

        Args:
            atom: Semantic atom to analyze

        Returns:
            Dict with phase analysis details
        """
        filled, capacity = atom.shell_occupancy()

        return {
            "token": atom.token,
            "current_phase": atom.phase.value,
            "crystallization_score": self.compute_crystallization_score(atom),
            "shell_occupancy": f"{filled}/{capacity}",
            "charge_stability": atom.charge.stability,
            "modifiers": self.phase_modifiers(atom.phase),
        }
