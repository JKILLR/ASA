"""
Molecular Composition - Compose atoms into molecular vectors.

Provides multiple composition strategies:
- ADDITIVE: Mass-weighted average
- MULTIPLICATIVE: Element-wise product
- TENSOR: Tensor interaction
- ATTENTION: Self-attention aggregation
"""

import math
from enum import Enum
from typing import List

import torch
import torch.nn.functional as F

from ..core.atoms import SemanticAtom, SemanticBond


class CompositionStrategy(Enum):
    """Strategies for composing atoms into molecules."""

    ADDITIVE = "additive"  # Mass-weighted average
    MULTIPLICATIVE = "multiplicative"  # Element-wise product
    TENSOR = "tensor"  # Tensor interaction
    ATTENTION = "attention"  # Self-attention aggregation


def compose_molecule(
    atoms: List[SemanticAtom],
    bonds: List[SemanticBond],
    strategy: CompositionStrategy = CompositionStrategy.ADDITIVE,
) -> torch.Tensor:
    """
    Compose atoms into molecular vector.

    Args:
        atoms: List of atoms to compose
        bonds: List of bonds (used by some strategies)
        strategy: Composition strategy

    Returns:
        Composed molecular vector

    Raises:
        ValueError: If atoms list is empty
    """
    if not atoms:
        raise ValueError("Cannot compose empty atom list")

    vecs = torch.stack([a.to_flat_vector() for a in atoms])
    weights = torch.tensor([a.mass for a in atoms])

    if strategy == CompositionStrategy.ADDITIVE:
        # Mass-weighted average
        weights = weights / (weights.sum() + 1e-8)
        return (vecs * weights.unsqueeze(-1)).sum(dim=0)

    elif strategy == CompositionStrategy.MULTIPLICATIVE:
        # Element-wise product (with normalization)
        result = vecs[0]
        for v in vecs[1:]:
            result = result * v
        # Normalize to prevent explosion/vanishing
        return F.normalize(result, dim=-1)

    elif strategy == CompositionStrategy.TENSOR:
        # Tensor interaction through bonds
        result = vecs.mean(dim=0)
        for bond in bonds:
            # Compute outer product interaction
            interaction = torch.outer(
                bond.atom_a.nuclear_vector, bond.atom_b.nuclear_vector
            ).flatten()
            # Truncate/pad to match dimension
            target_dim = result.shape[0]
            if len(interaction) >= target_dim:
                interaction = interaction[:target_dim]
            else:
                interaction = F.pad(
                    interaction, (0, target_dim - len(interaction))
                )
            result = result + bond.strength * interaction

        return result / (len(bonds) + 1)

    elif strategy == CompositionStrategy.ATTENTION:
        # Self-attention aggregation
        d_k = vecs.shape[-1]
        scores = torch.matmul(vecs, vecs.T) / math.sqrt(d_k)
        attn = F.softmax(scores, dim=-1)
        return torch.matmul(attn, vecs).mean(dim=0)

    # Default fallback
    return vecs.mean(dim=0)


def compose_with_bonds(
    atoms: List[SemanticAtom],
    bonds: List[SemanticBond],
    bond_weight: float = 0.3,
) -> torch.Tensor:
    """
    Compose atoms with explicit bond information.

    Creates a composed vector that incorporates bond interactions.

    Args:
        atoms: List of atoms
        bonds: List of bonds
        bond_weight: Weight of bond interactions

    Returns:
        Composed vector
    """
    if not atoms:
        raise ValueError("Cannot compose empty atom list")

    # Base composition
    vecs = torch.stack([a.to_flat_vector() for a in atoms])
    base = vecs.mean(dim=0)

    if not bonds:
        return base

    # Add bond interactions
    bond_contribution = torch.zeros_like(base)
    for bond in bonds:
        # Interaction vector: difference weighted by strength
        vec_a = bond.atom_a.to_flat_vector()
        vec_b = bond.atom_b.to_flat_vector()
        interaction = (vec_a + vec_b) / 2 * bond.strength
        bond_contribution = bond_contribution + interaction

    bond_contribution = bond_contribution / len(bonds)

    return (1 - bond_weight) * base + bond_weight * bond_contribution


def compose_hierarchical(
    atoms: List[SemanticAtom],
    bonds: List[SemanticBond],
    levels: int = 3,
) -> List[torch.Tensor]:
    """
    Compose atoms hierarchically with multiple granularity levels.

    Args:
        atoms: List of atoms
        bonds: List of bonds
        levels: Number of hierarchy levels

    Returns:
        List of vectors from fine to coarse granularity
    """
    if not atoms:
        raise ValueError("Cannot compose empty atom list")

    results = []

    # Level 0: Individual atom vectors
    vecs = torch.stack([a.to_flat_vector() for a in atoms])
    results.append(vecs.mean(dim=0))

    # Level 1: Local compositions (connected components)
    if levels >= 2 and bonds:
        # Build adjacency
        adj = {}
        for bond in bonds:
            a_id, b_id = bond.atom_a.atom_id, bond.atom_b.atom_id
            adj.setdefault(a_id, []).append(b_id)
            adj.setdefault(b_id, []).append(a_id)

        # Find connected pairs and compose
        seen = set()
        pair_vecs = []
        for bond in bonds:
            pair_key = (
                min(bond.atom_a.atom_id, bond.atom_b.atom_id),
                max(bond.atom_a.atom_id, bond.atom_b.atom_id),
            )
            if pair_key not in seen:
                seen.add(pair_key)
                pair_vec = (
                    bond.atom_a.to_flat_vector() + bond.atom_b.to_flat_vector()
                ) / 2
                pair_vecs.append(pair_vec * bond.strength)

        if pair_vecs:
            results.append(torch.stack(pair_vecs).mean(dim=0))
        else:
            results.append(results[0])

    # Level 2: Global composition
    if levels >= 3:
        results.append(compose_molecule(atoms, bonds, CompositionStrategy.ATTENTION))

    return results


def compose_with_charge(
    atoms: List[SemanticAtom],
    bonds: List[SemanticBond],
) -> tuple:
    """
    Compose atoms and return both vector and charge.

    Args:
        atoms: List of atoms
        bonds: List of bonds

    Returns:
        (composed_vector, net_charge)
    """
    if not atoms:
        raise ValueError("Cannot compose empty atom list")

    # Compose vector
    composed = compose_molecule(atoms, bonds, CompositionStrategy.ADDITIVE)

    # Compute net charge
    total_charge = 0.0
    total_mass = 0.0
    for atom in atoms:
        total_charge += atom.effective_charge * atom.mass
        total_mass += atom.mass

    net_charge = total_charge / total_mass if total_mass > 0 else 0.0

    return composed, net_charge
