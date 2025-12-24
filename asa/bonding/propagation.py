"""
Charge propagation - Propagate charge through molecular structure.

Implements compositional negation logic:
- NOT flips charge polarity
- Double negatives cancel
- Charge flows through bonds weighted by strength
"""

from typing import Dict, List

from ..core.atoms import SemanticMolecule, SemanticAtom, SemanticBond


def propagate_charge(molecule: SemanticMolecule) -> float:
    """
    Propagate charge through molecular structure.

    Rules:
    - Charges flow through bonds weighted by strength
    - Negation modifiers flip downstream charge
    - Double negatives cancel (odd flips = flip, even flips = no change)

    Args:
        molecule: Semantic molecule to process

    Returns:
        Net charge of the molecule
    """
    if not molecule.atoms:
        return 0.0

    # Count negations affecting each atom
    negation_count: Dict[int, int] = {atom.atom_id: 0 for atom in molecule.atoms}

    for bond in molecule.bonds:
        # Check if either atom in the bond is a negator
        if bond.role_a == "NEGATOR":
            negation_count[bond.atom_b.atom_id] += 1
        if bond.role_b == "NEGATOR":
            negation_count[bond.atom_a.atom_id] += 1

    # Compute net charge with negation effects
    total_charge = 0.0
    total_weight = 0.0

    for atom in molecule.atoms:
        base_charge = atom.effective_charge

        # Apply negation: odd count flips, even count preserves
        negations = negation_count.get(atom.atom_id, 0)
        if negations % 2 == 1:
            base_charge = -base_charge

        # Weight by mass and stability
        weight = atom.mass * atom.charge.stability
        total_charge += base_charge * weight
        total_weight += weight

    return total_charge / total_weight if total_weight > 0 else 0.0


def propagate_charge_graph(
    atoms: List[SemanticAtom],
    bonds: List[SemanticBond],
    negator_tokens: set = None,
) -> Dict[int, float]:
    """
    Propagate charge through a bond graph.

    More detailed version that tracks charge at each atom.

    Args:
        atoms: List of atoms
        bonds: List of bonds
        negator_tokens: Set of tokens that act as negators

    Returns:
        Dict mapping atom_id to effective charge after propagation
    """
    if negator_tokens is None:
        negator_tokens = {
            "not", "n't", "never", "no", "none", "neither", "nor",
            "without", "lack", "absent", "missing",
        }

    # Build adjacency
    adj: Dict[int, List[tuple]] = {a.atom_id: [] for a in atoms}
    for bond in bonds:
        adj[bond.atom_a.atom_id].append((bond.atom_b.atom_id, bond.strength))
        adj[bond.atom_b.atom_id].append((bond.atom_a.atom_id, bond.strength))

    # Identify negators
    negator_ids = {
        a.atom_id for a in atoms if a.token.lower() in negator_tokens
    }

    # BFS propagation from negators
    negation_depth: Dict[int, int] = {}
    for neg_id in negator_ids:
        visited = {neg_id}
        frontier = [(neg_id, 0)]

        while frontier:
            current, depth = frontier.pop(0)
            for neighbor, strength in adj.get(current, []):
                if neighbor not in visited and strength > 0.3:
                    visited.add(neighbor)
                    # Only propagate negation to adjacent non-negators
                    if neighbor not in negator_ids:
                        current_depth = negation_depth.get(neighbor, 0)
                        negation_depth[neighbor] = current_depth + 1
                        # Only propagate one hop from negator
                        if depth == 0:
                            frontier.append((neighbor, depth + 1))

    # Compute effective charges
    result = {}
    for atom in atoms:
        base_charge = atom.effective_charge
        flips = negation_depth.get(atom.atom_id, 0)
        if flips % 2 == 1:
            result[atom.atom_id] = -base_charge
        else:
            result[atom.atom_id] = base_charge

    return result


def compute_sentiment_from_molecule(molecule: SemanticMolecule) -> Dict[str, float]:
    """
    Compute sentiment analysis from molecular charge.

    Args:
        molecule: Semantic molecule

    Returns:
        Dict with sentiment scores
    """
    net_charge = propagate_charge(molecule)

    # Compute charge variance for confidence
    charges = [a.effective_charge for a in molecule.atoms]
    if len(charges) > 1:
        mean_charge = sum(charges) / len(charges)
        variance = sum((c - mean_charge) ** 2 for c in charges) / len(charges)
    else:
        variance = 0.0

    # Convert to sentiment
    if net_charge > 0.3:
        sentiment = "positive"
    elif net_charge < -0.3:
        sentiment = "negative"
    else:
        sentiment = "neutral"

    confidence = 1.0 - min(1.0, variance)

    return {
        "sentiment": sentiment,
        "net_charge": net_charge,
        "confidence": confidence,
        "variance": variance,
    }


class ChargePropagator:
    """
    Stateful charge propagation with configurable rules.
    """

    def __init__(
        self,
        negator_tokens: set = None,
        intensifier_tokens: set = None,
        hedge_tokens: set = None,
    ):
        """
        Initialize charge propagator.

        Args:
            negator_tokens: Tokens that flip charge
            intensifier_tokens: Tokens that amplify charge
            hedge_tokens: Tokens that attenuate charge
        """
        self.negator_tokens = negator_tokens or {
            "not", "n't", "never", "no", "none", "neither", "nor",
            "without", "lack", "absent", "missing", "fail", "failed",
        }
        self.intensifier_tokens = intensifier_tokens or {
            "very", "extremely", "absolutely", "totally", "completely",
            "really", "highly", "deeply", "incredibly", "utterly",
        }
        self.hedge_tokens = hedge_tokens or {
            "maybe", "perhaps", "possibly", "might", "could",
            "somewhat", "slightly", "rather", "fairly", "kind of",
        }

    def propagate(
        self, molecule: SemanticMolecule
    ) -> Dict[int, Dict[str, float]]:
        """
        Propagate charge with modifiers.

        Args:
            molecule: Semantic molecule

        Returns:
            Dict mapping atom_id to charge info dict
        """
        result = {}

        for atom in molecule.atoms:
            token_lower = atom.token.lower()

            # Start with base charge
            polarity = atom.charge.polarity
            magnitude = atom.charge.magnitude
            stability = atom.charge.stability

            # Check for self-modification
            if token_lower in self.intensifier_tokens:
                magnitude = min(1.0, magnitude * 1.3)
            elif token_lower in self.hedge_tokens:
                magnitude *= 0.5

            # Look at bonded atoms for modification
            bonds = molecule.get_bonds_for(atom.atom_id)
            for bond in bonds:
                other = (
                    bond.atom_b if bond.atom_a.atom_id == atom.atom_id
                    else bond.atom_a
                )
                other_token = other.token.lower()

                if other_token in self.negator_tokens:
                    polarity = -polarity
                    stability *= 0.9
                elif other_token in self.intensifier_tokens:
                    magnitude = min(1.0, magnitude * (1.0 + 0.2 * bond.strength))
                elif other_token in self.hedge_tokens:
                    magnitude *= (1.0 - 0.3 * bond.strength)

            result[atom.atom_id] = {
                "polarity": polarity,
                "magnitude": magnitude,
                "stability": stability,
                "effective": polarity * magnitude * stability,
            }

        return result
