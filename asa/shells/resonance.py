"""
ResonanceState - Superposition of interpretations for ambiguous structures.

Ambiguous structures exist in superposition of valid interpretations
until context collapses them. This models semantic ambiguity and
polysemy in a principled way.
"""

import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F

from ..core.atoms import SemanticAtom, SemanticBond


@dataclass
class InterpretationStructure:
    """
    A single possible interpretation.

    Represents one way to understand an ambiguous structure,
    including its bond graph, charge states, and semantic vector.

    Attributes:
        bond_graph: Dict mapping (atom_a_id, atom_b_id) to bond
        charge_states: Dict mapping atom_id to effective charge
        semantic_vector: Aggregated embedding for this interpretation
        prior_weight: Prior probability of this interpretation
    """

    bond_graph: Dict[Tuple[int, int], SemanticBond]
    charge_states: Dict[int, float]
    semantic_vector: torch.Tensor
    prior_weight: float = 1.0


class ResonanceState:
    """
    Superposition of interpretations for ambiguous structures.

    Maintains multiple possible interpretations with weights,
    allowing gradual collapse as context accumulates.

    Key operations:
    - add_interpretation: Add a possible interpretation
    - update_with_context: Update weights based on context
    - collapse: Force selection of single interpretation
    - get_dominant: Get most probable interpretation
    """

    def __init__(
        self, max_structures: int = 5, collapse_threshold: float = 0.8
    ):
        """
        Initialize resonance state.

        Args:
            max_structures: Maximum interpretations to maintain
            collapse_threshold: Weight at which to auto-collapse
        """
        self.max_structures = max_structures
        self.collapse_threshold = collapse_threshold
        self.structures: List[InterpretationStructure] = []
        self.weights: torch.Tensor = torch.tensor([])

    def add_interpretation(self, structure: InterpretationStructure) -> None:
        """
        Add a possible interpretation.

        If at capacity, replaces the lowest-weight interpretation
        if new one has higher prior.

        Args:
            structure: Interpretation to add
        """
        if len(self.structures) >= self.max_structures:
            min_idx = self.weights.argmin().item()
            if structure.prior_weight > self.weights[min_idx]:
                self.structures[min_idx] = structure
                self.weights[min_idx] = structure.prior_weight
        else:
            self.structures.append(structure)
            self.weights = torch.cat(
                [self.weights, torch.tensor([structure.prior_weight])]
            )

        # Normalize weights
        self.weights = self.weights / (self.weights.sum() + 1e-8)

    @property
    def is_ambiguous(self) -> bool:
        """Check if structure is still ambiguous."""
        return len(self.structures) > 1 and self.entropy > 0.5

    @property
    def is_collapsed(self) -> bool:
        """Check if structure has collapsed to single interpretation."""
        return len(self.structures) == 1 or self.weights.max().item() >= self.collapse_threshold

    @property
    def entropy(self) -> float:
        """
        Shannon entropy of interpretation distribution.

        Higher entropy = more ambiguous.
        """
        if len(self.weights) <= 1:
            return 0.0
        probs = self.weights.clamp(min=1e-10)
        return -(probs * probs.log()).sum().item()

    @property
    def normalized_entropy(self) -> float:
        """Entropy as fraction of maximum possible."""
        if len(self.structures) <= 1:
            return 0.0
        max_ent = math.log(len(self.structures))
        return self.entropy / max_ent if max_ent > 0 else 0.0

    def update_with_context(
        self, context_atoms: List[SemanticAtom], weight: float = 1.0
    ) -> None:
        """
        Update weights based on context (partial measurement).

        Interpretations more compatible with context get higher weights.

        Args:
            context_atoms: Context atoms to condition on
            weight: Strength of context influence
        """
        if not self.structures or not context_atoms:
            return

        # Compute context vector
        context_vec = torch.stack(
            [a.to_flat_vector() for a in context_atoms]
        ).mean(0)

        # Compute compatibility of each interpretation with context
        compatibilities = []
        for structure in self.structures:
            compat = F.cosine_similarity(
                structure.semantic_vector.unsqueeze(0),
                context_vec.unsqueeze(0),
            ).item()
            compatibilities.append(compat)

        # Convert to likelihood
        likelihood = F.softmax(
            torch.tensor(compatibilities) * weight, dim=0
        )

        # Bayesian update: posterior ∝ prior × likelihood
        self.weights = self.weights * likelihood
        self.weights = self.weights / (self.weights.sum() + 1e-8)

        # Auto-collapse if dominant weight exceeds threshold
        if self.weights.max().item() >= self.collapse_threshold:
            self._auto_collapse()

    def _auto_collapse(self) -> None:
        """Automatically collapse to dominant interpretation."""
        max_idx = self.weights.argmax().item()
        self.structures = [self.structures[max_idx]]
        self.weights = torch.tensor([1.0])

    def collapse(self) -> InterpretationStructure:
        """
        Force collapse to single interpretation.

        Selects the highest-weight interpretation and discards others.

        Returns:
            Selected interpretation
        """
        max_idx = self.weights.argmax().item()
        selected = self.structures[max_idx]
        self.structures = [selected]
        self.weights = torch.tensor([1.0])
        return selected

    def get_dominant(self) -> Tuple[InterpretationStructure, float]:
        """
        Get most probable interpretation and its probability.

        Returns:
            (interpretation, probability)
        """
        if not self.structures:
            return None, 0.0
        max_idx = self.weights.argmax().item()
        return self.structures[max_idx], self.weights[max_idx].item()

    def get_expectation(self) -> torch.Tensor:
        """
        Get expected semantic vector (weighted average).

        Returns:
            Weighted average of interpretation vectors
        """
        if not self.structures:
            return torch.zeros(1)

        vectors = torch.stack([s.semantic_vector for s in self.structures])
        return (vectors * self.weights.unsqueeze(-1)).sum(dim=0)

    def get_expected_charge(self, atom_id: int) -> float:
        """
        Get expected charge for an atom across interpretations.

        Args:
            atom_id: Atom to get expected charge for

        Returns:
            Weighted average charge
        """
        total = 0.0
        for structure, weight in zip(self.structures, self.weights):
            if atom_id in structure.charge_states:
                total += structure.charge_states[atom_id] * weight.item()
        return total

    def get_report(self) -> dict:
        """Get status report of resonance state."""
        dominant, prob = self.get_dominant()
        return {
            "num_interpretations": len(self.structures),
            "is_ambiguous": self.is_ambiguous,
            "is_collapsed": self.is_collapsed,
            "entropy": self.entropy,
            "normalized_entropy": self.normalized_entropy,
            "dominant_probability": prob,
            "weights": self.weights.tolist() if len(self.weights) > 0 else [],
        }


class ResonanceBuilder:
    """Helper class to build resonance states from ambiguous inputs."""

    @staticmethod
    def from_polysemy(
        atom: SemanticAtom, sense_vectors: List[torch.Tensor], sense_priors: List[float] = None
    ) -> ResonanceState:
        """
        Create resonance state from polysemous word with multiple senses.

        Args:
            atom: Base atom
            sense_vectors: Embedding for each sense
            sense_priors: Prior probabilities for senses

        Returns:
            ResonanceState representing ambiguity
        """
        if sense_priors is None:
            sense_priors = [1.0 / len(sense_vectors)] * len(sense_vectors)

        resonance = ResonanceState()

        for i, (vec, prior) in enumerate(zip(sense_vectors, sense_priors)):
            structure = InterpretationStructure(
                bond_graph={},
                charge_states={atom.atom_id: atom.effective_charge},
                semantic_vector=vec,
                prior_weight=prior,
            )
            resonance.add_interpretation(structure)

        return resonance

    @staticmethod
    def from_ambiguous_parse(
        atoms: List[SemanticAtom],
        parse_options: List[List[Tuple[int, int, str]]],  # List of (a, b, bond_type)
        priors: List[float] = None,
    ) -> ResonanceState:
        """
        Create resonance state from ambiguous syntactic parse.

        Args:
            atoms: Atoms involved in ambiguous structure
            parse_options: List of possible bond configurations
            priors: Prior probabilities for each parse

        Returns:
            ResonanceState representing structural ambiguity
        """
        if priors is None:
            priors = [1.0 / len(parse_options)] * len(parse_options)

        resonance = ResonanceState()
        atom_dict = {a.atom_id: a for a in atoms}

        for parse, prior in zip(parse_options, priors):
            bond_graph = {}
            for a_id, b_id, bond_type in parse:
                if a_id in atom_dict and b_id in atom_dict:
                    bond = SemanticBond(
                        atom_a=atom_dict[a_id],
                        atom_b=atom_dict[b_id],
                        bond_type=bond_type,
                        strength=0.5,  # Default strength
                    )
                    bond_graph[(a_id, b_id)] = bond

            # Compute semantic vector for this parse
            vecs = torch.stack([a.to_flat_vector() for a in atoms])
            semantic_vec = vecs.mean(dim=0)

            structure = InterpretationStructure(
                bond_graph=bond_graph,
                charge_states={a.atom_id: a.effective_charge for a in atoms},
                semantic_vector=semantic_vec,
                prior_weight=prior,
            )
            resonance.add_interpretation(structure)

        return resonance
