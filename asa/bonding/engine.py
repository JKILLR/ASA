"""
BondingEngine - Handles bond formation with full thermodynamic and catalyst support.

Bonds connect atoms with typed relationships:
- COVALENT: Strong shared meaning
- IONIC: Oppositely charged attraction
- HYDROGEN: Weak contextual link
- MODIFIER: One atom modifies the other
- ARGUMENT: Predicate-argument relationship
"""

from enum import Enum
from typing import List, Optional

import torch
import torch.nn.functional as F

from ..core.atoms import SemanticAtom, SemanticBond
from ..core.config import AtomConfig
from ..thermodynamics.learnable import LearnableThermodynamics
from ..thermodynamics.context import LearnableSemanticContext
from ..shells.phase import PhaseAnalyzer
from .catalyst import LearnedCatalystDetector


class BondType(Enum):
    """Types of semantic bonds."""

    COVALENT = "covalent"  # Strong shared meaning
    IONIC = "ionic"  # Oppositely charged attraction
    HYDROGEN = "hydrogen"  # Weak contextual link
    MODIFIER = "modifier"  # One atom modifies the other
    ARGUMENT = "argument"  # Predicate-argument relationship


class BondingEngine:
    """
    Handles bond formation with full thermodynamic and catalyst support.

    Bond formation considers:
    - Valence compatibility
    - Charge interactions
    - Temperature-dependent thresholds
    - Phase modifiers
    - Catalyst effects
    """

    def __init__(
        self,
        config: AtomConfig,
        thermo: LearnableThermodynamics,
        catalyst_detector: LearnedCatalystDetector = None,
        phase_analyzer: PhaseAnalyzer = None,
    ):
        """
        Initialize bonding engine.

        Args:
            config: Atom configuration
            thermo: Thermodynamics module
            catalyst_detector: Optional catalyst detector
            phase_analyzer: Optional phase analyzer
        """
        self.config = config
        self.thermo = thermo
        self.catalyst_detector = catalyst_detector
        self.phase_analyzer = phase_analyzer or PhaseAnalyzer(thermo)

    def attempt_bond(
        self,
        atom_a: SemanticAtom,
        atom_b: SemanticAtom,
        context: LearnableSemanticContext,
        context_atoms: List[SemanticAtom] = None,
    ) -> Optional[SemanticBond]:
        """
        Attempt to form a bond with all thermodynamic effects.

        Args:
            atom_a: First atom
            atom_b: Second atom
            context: Semantic context (provides temperature)
            context_atoms: Optional context atoms (potential catalysts)

        Returns:
            SemanticBond if successful, None otherwise
        """
        # Check valence availability
        if atom_a.valence_count <= 0 or atom_b.valence_count <= 0:
            return None

        # Calculate raw bond strength
        raw_strength = self._compute_strength(atom_a, atom_b)

        # Determine bond type
        bond_type = self._classify_bond(atom_a, atom_b)

        # Get base threshold from context
        threshold = context.bond_threshold

        # Apply catalyst effects
        strength_bonus = 0.0
        catalysts_used = []

        if self.catalyst_detector and context_atoms:
            for ctx_atom in context_atoms:
                effects = self.catalyst_detector(ctx_atom.to_flat_vector())
                if effects["is_catalyst"].item() > 0.1:
                    bond_idx = list(BondType).index(bond_type)
                    if bond_idx < len(effects["threshold_multipliers"]):
                        threshold *= effects["threshold_multipliers"][
                            bond_idx
                        ].item()
                        strength_bonus += effects["strength_bonuses"][
                            bond_idx
                        ].item()
                        catalysts_used.append(ctx_atom.token)

        # Apply phase modifiers
        phase_a = self.phase_analyzer.determine_phase(atom_a)
        phase_b = self.phase_analyzer.determine_phase(atom_b)
        mods_a = self.phase_analyzer.phase_modifiers(phase_a)
        mods_b = self.phase_analyzer.phase_modifiers(phase_b)
        avg_rate = (
            mods_a["bond_formation_rate"] + mods_b["bond_formation_rate"]
        ) / 2
        threshold /= max(0.1, avg_rate)  # Avoid division by zero

        # Final strength
        final_strength = min(1.0, raw_strength + strength_bonus)

        # Check threshold
        if final_strength < threshold:
            return None

        # Check charge compatibility at low temperature
        charge_product = atom_a.effective_charge * atom_b.effective_charge
        if charge_product < -0.7 and context.temperature < 0.7:
            return None

        return SemanticBond(
            atom_a=atom_a,
            atom_b=atom_b,
            bond_type=bond_type.value,
            strength=final_strength,
            catalyzed_by=catalysts_used if catalysts_used else None,
        )

    def _compute_strength(
        self, atom_a: SemanticAtom, atom_b: SemanticAtom
    ) -> float:
        """
        Compute raw bond strength from atom properties.

        Considers:
        - Valence vector similarity
        - Shell vector similarity
        - Charge interaction
        """
        # Valence compatibility
        val_sim = F.cosine_similarity(
            atom_a.bond_vectors[0].unsqueeze(0),
            atom_b.bond_vectors[0].unsqueeze(0),
        ).item()

        # Shell compatibility
        shell_a = (
            atom_a.shells[0].get_vector()
            if atom_a.shells
            else atom_a.nuclear_vector
        )
        shell_b = (
            atom_b.shells[0].get_vector()
            if atom_b.shells
            else atom_b.nuclear_vector
        )
        shell_sim = F.cosine_similarity(
            shell_a.unsqueeze(0), shell_b.unsqueeze(0)
        ).item()

        # Charge interaction
        charge_int = abs(atom_a.effective_charge * atom_b.effective_charge)

        # Weighted combination
        return (
            0.4 * max(0, val_sim)
            + 0.4 * max(0, shell_sim)
            + 0.2 * charge_int
        )

    def _classify_bond(
        self, atom_a: SemanticAtom, atom_b: SemanticAtom
    ) -> BondType:
        """
        Classify bond type based on atom properties.

        Rules:
        - Strong charge opposition → IONIC
        - High valence similarity → COVALENT
        - Otherwise → HYDROGEN
        """
        charge_a = atom_a.effective_charge
        charge_b = atom_b.effective_charge

        # Ionic: opposite charges
        if charge_a * charge_b < -0.5:
            return BondType.IONIC

        # Covalent: similar valence profiles
        val_sim = F.cosine_similarity(
            atom_a.bond_vectors[0].unsqueeze(0),
            atom_b.bond_vectors[0].unsqueeze(0),
        ).item()

        if val_sim > 0.6:
            return BondType.COVALENT

        return BondType.HYDROGEN

    def compute_bond_energy(self, bond: SemanticBond) -> float:
        """
        Compute energy of a bond (for breaking calculations).

        Higher strength + inner shell atoms = higher energy required to break.
        """
        # Base energy from strength
        base_energy = bond.strength

        # Modify by phase
        phase_a = bond.atom_a.phase
        phase_b = bond.atom_b.phase

        # Solid atoms have higher bond energy
        phase_mult = 1.0
        from ..core.atoms import ConceptPhase
        if phase_a == ConceptPhase.SOLID:
            phase_mult *= 1.5
        if phase_b == ConceptPhase.SOLID:
            phase_mult *= 1.5

        return base_energy * phase_mult

    def can_break_bond(
        self,
        bond: SemanticBond,
        context: LearnableSemanticContext,
        available_energy: float = float("inf"),
    ) -> bool:
        """
        Check if a bond can be broken given available energy.

        Args:
            bond: Bond to potentially break
            context: Semantic context
            available_energy: Energy available for breaking

        Returns:
            True if bond can be broken
        """
        required_energy = self.compute_bond_energy(bond)
        prob = context.boltzmann_probability(required_energy)
        return available_energy >= required_energy or prob > 0.5
