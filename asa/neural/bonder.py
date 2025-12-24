"""
BondingNetwork - Predicts bonds between atom pairs.

Outputs:
- Bond strength [0, 1]
- Bond type probabilities
- Can-bond binary prediction
"""

from typing import Dict

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..core.config import AtomConfig
from ..bonding.engine import BondType


class BondingNetwork(nn.Module):
    """
    Predicts bonds between atom pairs.

    Takes two atom representations and predicts:
    - Whether they can bond
    - The strength of the bond
    - The type of bond
    """

    def __init__(self, config: AtomConfig):
        """
        Initialize bonding network.

        Args:
            config: Atom configuration
        """
        super().__init__()
        self.config = config

        # Pair encoder
        pair_dim = config.total_dim * 2
        self.pair_encoder = nn.Sequential(
            nn.Linear(pair_dim, 256),
            nn.LayerNorm(256),
            nn.GELU(),
            nn.Linear(256, 128),
        )

        # Valence compatibility via bilinear
        self.valence_compat = nn.Bilinear(
            config.bond_dim, config.bond_dim, 64
        )

        # Charge interaction via bilinear
        self.charge_interact = nn.Bilinear(
            config.charge_dim, config.charge_dim, 32
        )

        # Combined head
        combined_dim = 128 + 64 + 32
        self.bond_head = nn.Sequential(
            nn.Linear(combined_dim, 64),
            nn.GELU(),
            nn.Linear(64, 32),
        )

        # Output heads
        self.strength_out = nn.Linear(32, 1)
        self.type_out = nn.Linear(32, len(BondType))

    def forward(
        self,
        atom_a: Dict[str, torch.Tensor],
        atom_b: Dict[str, torch.Tensor],
    ) -> Dict[str, torch.Tensor]:
        """
        Predict bond between two atoms.

        Args:
            atom_a: First atom representation
            atom_b: Second atom representation

        Returns:
            Dict with bond predictions
        """
        # Handle batch dimension
        flat_a = atom_a["flat"]
        flat_b = atom_b["flat"]

        if flat_a.dim() == 1:
            flat_a = flat_a.unsqueeze(0)
            flat_b = flat_b.unsqueeze(0)
            squeeze = True
        else:
            squeeze = False

        # Pair encoding
        pair = torch.cat([flat_a, flat_b], dim=-1)
        pair_enc = self.pair_encoder(pair)

        # Valence compatibility
        val_a = atom_a["bond_vectors"]
        val_b = atom_b["bond_vectors"]

        if val_a.dim() == 2:
            val_a = val_a.unsqueeze(0)
            val_b = val_b.unsqueeze(0)

        # Use first valence slot
        val_a = val_a[..., 0, :]
        val_b = val_b[..., 0, :]
        val_compat = self.valence_compat(val_a, val_b)

        # Charge interaction
        charge_a = atom_a["charge_vector"]
        charge_b = atom_b["charge_vector"]

        if charge_a.dim() == 1:
            charge_a = charge_a.unsqueeze(0)
            charge_b = charge_b.unsqueeze(0)

        charge_int = self.charge_interact(charge_a, charge_b)

        # Combined
        combined = torch.cat([pair_enc, val_compat, charge_int], dim=-1)
        hidden = self.bond_head(combined)

        # Outputs
        strength = torch.sigmoid(self.strength_out(hidden))
        type_logits = self.type_out(hidden)

        result = {
            "strength": strength.squeeze(-1),
            "type_logits": type_logits,
            "type_probs": F.softmax(type_logits, dim=-1),
            "can_bond": (strength > 0.3).float().squeeze(-1),
        }

        if squeeze:
            result = {k: v.squeeze(0) for k, v in result.items()}

        return result

    def predict_batch(
        self,
        atoms: Dict[str, torch.Tensor],
        pair_indices: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """
        Predict bonds for multiple pairs.

        Args:
            atoms: Batch of atom representations
            pair_indices: (num_pairs, 2) tensor of indices

        Returns:
            Bond predictions for each pair
        """
        num_pairs = pair_indices.shape[0]

        # Gather atom representations for each pair
        flat = atoms["flat"]
        bond_vecs = atoms["bond_vectors"]
        charge_vecs = atoms["charge_vector"]

        flat_a = flat[pair_indices[:, 0]]
        flat_b = flat[pair_indices[:, 1]]

        # Create atom dicts for each pair
        atom_a = {
            "flat": flat_a,
            "bond_vectors": bond_vecs[pair_indices[:, 0]],
            "charge_vector": charge_vecs[pair_indices[:, 0]],
        }
        atom_b = {
            "flat": flat_b,
            "bond_vectors": bond_vecs[pair_indices[:, 1]],
            "charge_vector": charge_vecs[pair_indices[:, 1]],
        }

        return self.forward(atom_a, atom_b)


class GraphBondingNetwork(nn.Module):
    """
    Graph neural network for iterative bond refinement.

    Uses message passing to refine bond predictions based
    on graph structure.
    """

    def __init__(
        self,
        config: AtomConfig,
        num_iterations: int = 3,
    ):
        """
        Initialize graph bonding network.

        Args:
            config: Atom configuration
            num_iterations: Number of message passing iterations
        """
        super().__init__()
        self.config = config
        self.num_iterations = num_iterations

        # Base bonder
        self.base_bonder = BondingNetwork(config)

        # Message passing layers
        self.message_net = nn.Sequential(
            nn.Linear(config.total_dim * 2 + 1, 128),
            nn.GELU(),
            nn.Linear(128, config.total_dim),
        )

        self.update_net = nn.GRUCell(config.total_dim, config.total_dim)

    def forward(
        self,
        atoms: Dict[str, torch.Tensor],
        adjacency: torch.Tensor = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Predict bonds with message passing.

        Args:
            atoms: Atom representations (batch_size, num_atoms, dim)
            adjacency: Optional adjacency matrix (batch_size, num_atoms, num_atoms)

        Returns:
            Bond predictions for all pairs
        """
        flat = atoms["flat"]
        batch_size, num_atoms, dim = flat.shape

        # Initial bond predictions
        if adjacency is None:
            # Create all-pairs adjacency
            adjacency = torch.ones(batch_size, num_atoms, num_atoms)
            adjacency = adjacency - torch.eye(num_atoms).unsqueeze(0)

        # Iterate
        node_states = flat
        for _ in range(self.num_iterations):
            # Compute messages
            messages = torch.zeros_like(node_states)
            for i in range(num_atoms):
                for j in range(num_atoms):
                    if i != j:
                        # Compute message from j to i
                        pair = torch.cat([node_states[:, i], node_states[:, j]], dim=-1)
                        edge_weight = adjacency[:, i, j].unsqueeze(-1)
                        msg_input = torch.cat([pair, edge_weight], dim=-1)
                        msg = self.message_net(msg_input) * edge_weight
                        messages[:, i] += msg

            # Update node states
            messages = messages.view(-1, dim)
            node_states_flat = node_states.view(-1, dim)
            updated = self.update_net(messages, node_states_flat)
            node_states = updated.view(batch_size, num_atoms, dim)

        # Final bond predictions
        atoms["flat"] = node_states
        return self._predict_all_pairs(atoms, num_atoms)

    def _predict_all_pairs(
        self,
        atoms: Dict[str, torch.Tensor],
        num_atoms: int,
    ) -> Dict[str, torch.Tensor]:
        """Predict bonds for all pairs."""
        strengths = []
        type_logits = []

        flat = atoms["flat"]
        bond_vecs = atoms["bond_vectors"]
        charge_vecs = atoms["charge_vector"]

        for i in range(num_atoms):
            for j in range(i + 1, num_atoms):
                atom_a = {
                    "flat": flat[:, i],
                    "bond_vectors": bond_vecs[:, i],
                    "charge_vector": charge_vecs[:, i],
                }
                atom_b = {
                    "flat": flat[:, j],
                    "bond_vectors": bond_vecs[:, j],
                    "charge_vector": charge_vecs[:, j],
                }
                pred = self.base_bonder(atom_a, atom_b)
                strengths.append(pred["strength"])
                type_logits.append(pred["type_logits"])

        return {
            "strengths": torch.stack(strengths, dim=1),
            "type_logits": torch.stack(type_logits, dim=1),
        }
