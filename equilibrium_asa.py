"""
EQUILIBRIUM ASA: Attention as Energy Minimization
==================================================

Core Insight: Attention shouldn't be computed from energy.
             Attention should EMERGE from energy minimization.

Traditional ASA:
    energy = f(atoms)
    attention = softmax(-energy)    <- Bolted on

Equilibrium ASA:
    energy = E(system_state)
    dynamics = -∇E                  <- Physics
    equilibrium = fixed_point(dynamics)
    attention = emergent_from(equilibrium)

Key Principles:
1. Sparsity emerges from energy landscape, not masks
2. Information flows through field interactions, not path enumeration
3. System finds equilibrium through gradient descent on energy
4. O(N) computation via field decomposition

Mathematical Foundation:
- Hopfield Networks: attention IS energy minimization
- Mean Field Theory: aggregate effects without O(N²)
- Deep Equilibrium Models: implicit layers finding fixed points

Run: python equilibrium_asa.py
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import numpy as np
import time
from typing import Optional, Tuple, NamedTuple
import math

# Device
if torch.backends.mps.is_available():
    DEVICE = torch.device("mps")
elif torch.cuda.is_available():
    DEVICE = torch.device("cuda")
else:
    DEVICE = torch.device("cpu")
print(f"Using {DEVICE}")


# =============================================================================
# PART 1: ATOMIC STRUCTURE (What tokens ARE)
# =============================================================================

class AtomicState(NamedTuple):
    """
    The state of semantic atoms in the system.

    Unlike previous ASA where properties were fixed after embedding,
    here the state EVOLVES toward equilibrium.
    """
    charge: torch.Tensor      # (B, N, 1) - polarity, evolves
    position: torch.Tensor    # (B, N, D_pos) - location in semantic space, evolves
    momentum: torch.Tensor    # (B, N, D_pos) - for damped dynamics
    mass: torch.Tensor        # (B, N, 1) - inertia, fixed after embedding
    valence: torch.Tensor     # (B, N, 1) - bonding capacity, fixed
    identity: torch.Tensor    # (B, N, D_id) - core meaning, fixed (like nucleus)


class AtomicEmbedding(nn.Module):
    """
    Convert tokens to initial atomic states.

    Key insight: position in semantic space is LEARNED and EVOLVES.
    The embedding provides initial conditions; dynamics find equilibrium.
    """

    def __init__(
        self,
        vocab_size: int,
        position_dim: int = 64,    # Semantic space dimension
        identity_dim: int = 128,   # Core meaning dimension
        max_seq_len: int = 512
    ):
        super().__init__()

        self.position_dim = position_dim
        self.identity_dim = identity_dim

        # Initial position in semantic space (will evolve)
        self.position_embed = nn.Embedding(vocab_size, position_dim)

        # Fixed properties
        self.charge_embed = nn.Embedding(vocab_size, 1)
        self.mass_embed = nn.Embedding(vocab_size, 1)
        self.valence_embed = nn.Embedding(vocab_size, 1)
        self.identity_embed = nn.Embedding(vocab_size, identity_dim)

        # Sequence position encoding (added to semantic position)
        self.seq_position = nn.Embedding(max_seq_len, position_dim)

    def forward(self, token_ids: torch.Tensor) -> AtomicState:
        B, N = token_ids.shape
        device = token_ids.device

        # Sequence positions
        seq_pos = torch.arange(N, device=device).unsqueeze(0).expand(B, -1)

        # Initial semantic position (token embedding + sequence encoding)
        position = self.position_embed(token_ids) + 0.1 * self.seq_position(seq_pos)

        # Fixed properties
        charge = torch.tanh(self.charge_embed(token_ids))
        mass = F.softplus(self.mass_embed(token_ids)) + 0.5
        valence = F.softplus(self.valence_embed(token_ids)) + 1.0
        identity = self.identity_embed(token_ids)

        # Initial momentum is zero
        momentum = torch.zeros_like(position)

        return AtomicState(
            charge=charge,
            position=position,
            momentum=momentum,
            mass=mass,
            valence=valence,
            identity=identity
        )


# =============================================================================
# PART 2: SEMANTIC FIELDS (How atoms create and feel fields)
# =============================================================================

class SemanticFieldComputer(nn.Module):
    """
    Compute aggregate semantic fields from atomic ensemble.

    Key insight: Instead of O(N²) pairwise interactions,
    atoms create fields that other atoms respond to.

    This is O(N) to create the field, O(N) to read from it.

    Physics analogy:
    - Electric field from charges: φ(r) = Σ_i q_i / |r - r_i|
    - We use soft attention to aggregate: φ(r) = Σ_i w_i(r) * q_i
    """

    def __init__(self, position_dim: int, num_field_components: int = 4):
        super().__init__()

        self.position_dim = position_dim
        self.num_components = num_field_components

        # Field "sensors" - what aspects of the field we compute
        # These are like basis functions for the field
        self.field_keys = nn.Parameter(torch.randn(num_field_components, position_dim) * 0.1)

        # How atoms contribute to each field component
        self.contribution_proj = nn.Linear(position_dim + 1, num_field_components)  # +1 for charge

    def compute_fields(self, state: AtomicState) -> torch.Tensor:
        """
        Compute aggregate semantic fields.

        Returns: (B, num_components, position_dim) - the field values
        """
        B, N, D = state.position.shape

        # What each atom contributes (based on position and charge)
        atom_features = torch.cat([state.position, state.charge], dim=-1)  # (B, N, D+1)
        contributions = self.contribution_proj(atom_features)  # (B, N, num_components)

        # Weighted sum to create field (contribution-weighted average of positions)
        # This is like computing the "center of charge" for each field component
        contributions_softmax = F.softmax(contributions, dim=1)  # Normalize over atoms

        # Field value = weighted average of atom positions
        # (B, num_components, D) = (B, N, num_components)^T @ (B, N, D)
        fields = torch.bmm(contributions_softmax.transpose(1, 2), state.position)

        return fields  # (B, num_components, D)

    def field_at_position(self, fields: torch.Tensor, query_pos: torch.Tensor) -> torch.Tensor:
        """
        Read field value at specific positions.

        Args:
            fields: (B, num_components, D) - the computed fields
            query_pos: (B, N, D) - positions to query

        Returns: (B, N, num_components) - field values at each position
        """
        B, N, D = query_pos.shape

        # How much each field component affects each position
        # Based on alignment between query position and field keys
        # (B, N, D) @ (num_components, D)^T -> (B, N, num_components)
        alignment = torch.matmul(query_pos, self.field_keys.T)  # (B, N, num_components)
        alignment = alignment / math.sqrt(D)
        weights = F.softmax(alignment, dim=-1)

        # Interpolate field values based on weights
        # fields: (B, num_components, D)
        # We want the "potential" at each query position
        # Simplified: dot product of query with each field

        # (B, N, D) @ (B, D, num_components) -> (B, N, num_components)
        field_values = torch.bmm(query_pos, fields.transpose(1, 2))

        return field_values * weights


# =============================================================================
# PART 3: ENERGY FUNCTIONAL (What the system minimizes)
# =============================================================================

class EnergyFunctional(nn.Module):
    """
    The energy of the atomic system.

    E_total = E_field + E_repulsion + E_valence + E_regularization

    Key insight: Design energy so that:
    - MOST pairs have high energy (weak interaction -> sparse)
    - RELEVANT pairs have low energy (strong interaction)
    - Sparsity EMERGES from the energy landscape
    """

    def __init__(
        self,
        position_dim: int,
        baseline_energy: float = 5.0,     # Default: weak interaction
        charge_strength: float = 2.0,      # How much charge matters
        distance_strength: float = 1.0,    # Distance penalty
        valence_strength: float = 1.0,     # Valence satisfaction
    ):
        super().__init__()

        self.position_dim = position_dim
        self.baseline = baseline_energy
        self.charge_strength = charge_strength
        self.distance_strength = distance_strength
        self.valence_strength = valence_strength

        self.field_computer = SemanticFieldComputer(position_dim)

    def compute_energy(
        self,
        state: AtomicState,
        return_components: bool = False
    ) -> torch.Tensor:
        """
        Compute total system energy.

        Design principle: Energy should be HIGH by default (baseline),
        and only LOW for semantically relevant interactions.
        """
        B, N, D = state.position.shape

        # === Component 1: Field Energy ===
        # Atoms in favorable field regions have lower energy
        fields = self.field_computer.compute_fields(state)
        field_values = self.field_computer.field_at_position(fields, state.position)

        # Charge-field interaction: opposite charges in same field = low energy
        charge_field = field_values * state.charge  # (B, N, num_components)
        E_field = -self.charge_strength * charge_field.sum(dim=-1)  # (B, N)

        # === Component 2: Pairwise Distance Energy ===
        # Compute pairwise distances (this is O(N²) but sparse in practice)
        # For efficiency, we'll use a sampling-based approximation later
        pos_i = state.position.unsqueeze(2)  # (B, N, 1, D)
        pos_j = state.position.unsqueeze(1)  # (B, 1, N, D)
        distances = (pos_i - pos_j).norm(dim=-1)  # (B, N, N)

        # Short-range repulsion, long-range decay
        # E_dist = baseline - attraction * exp(-distance)
        attraction = torch.exp(-distances / math.sqrt(D))  # Decays with distance
        E_distance_pairwise = self.baseline - self.distance_strength * attraction

        # Mask self-interactions
        mask = torch.eye(N, device=state.position.device).bool()
        E_distance_pairwise = E_distance_pairwise.masked_fill(mask.unsqueeze(0), 0)

        E_distance = E_distance_pairwise.sum(dim=-1) / N  # (B, N) - average over pairs

        # === Component 3: Charge-Charge Interaction ===
        # Opposite charges attract (low energy), same charges repel
        charge_i = state.charge.squeeze(-1).unsqueeze(2)  # (B, N, 1)
        charge_j = state.charge.squeeze(-1).unsqueeze(1)  # (B, 1, N)
        charge_product = charge_i * charge_j  # (B, N, N)

        # Weight by distance (closer = stronger interaction)
        charge_interaction = charge_product * attraction
        E_charge_pairwise = self.charge_strength * charge_interaction
        E_charge_pairwise = E_charge_pairwise.masked_fill(mask.unsqueeze(0), 0)

        E_charge = E_charge_pairwise.sum(dim=-1) / N  # (B, N)

        # === Component 4: Valence Satisfaction ===
        # Atoms want to form the right number of bonds
        # "Bonds" = significant attention weights
        # Approximate by counting low-energy neighbors
        bond_strength = F.softmax(-E_distance_pairwise / 2.0, dim=-1)  # Soft bonds
        bond_count = bond_strength.sum(dim=-1)  # (B, N) - how many bonds formed

        # Energy is low when bond_count ≈ valence
        valence = state.valence.squeeze(-1)  # (B, N)
        valence_error = (bond_count - valence).abs()
        E_valence = self.valence_strength * valence_error

        # === Total Energy per Atom ===
        E_total_per_atom = E_field + E_distance + E_charge + E_valence

        if return_components:
            return E_total_per_atom, {
                'E_field': E_field.mean(),
                'E_distance': E_distance.mean(),
                'E_charge': E_charge.mean(),
                'E_valence': E_valence.mean(),
            }

        return E_total_per_atom  # (B, N)

    def compute_pairwise_energy(self, state: AtomicState) -> torch.Tensor:
        """
        Compute pairwise energies for attention derivation.

        This is the energy of the bond between atoms i and j.
        Used after equilibrium to extract attention weights.
        """
        B, N, D = state.position.shape

        # Distance-based energy
        pos_i = state.position.unsqueeze(2)
        pos_j = state.position.unsqueeze(1)
        distances = (pos_i - pos_j).norm(dim=-1)

        # Charge interaction
        charge_i = state.charge.squeeze(-1).unsqueeze(2)
        charge_j = state.charge.squeeze(-1).unsqueeze(1)
        charge_product = charge_i * charge_j

        # Combined pairwise energy
        # High baseline + charge interaction + distance decay
        attraction = torch.exp(-distances / math.sqrt(D))
        E_pair = self.baseline + self.charge_strength * charge_product - self.distance_strength * attraction

        return E_pair  # (B, N, N)


# =============================================================================
# PART 4: DYNAMICS (How the system evolves)
# =============================================================================

class EquilibriumDynamics(nn.Module):
    """
    Evolve atomic state toward energy minimum.

    Uses damped gradient descent on the energy functional.

    dx/dt = -∇E - γ * v  (gradient descent with friction)
    dv/dt = (dx/dt) / m  (momentum update)

    Key insight: The dynamics naturally find equilibrium.
    "Attention" is implicit in which atoms end up close together.
    """

    def __init__(
        self,
        energy_fn: EnergyFunctional,
        num_steps: int = 5,
        step_size: float = 0.1,
        damping: float = 0.5,
    ):
        super().__init__()

        self.energy_fn = energy_fn
        self.num_steps = num_steps
        self.step_size = step_size
        self.damping = damping

    def forward(self, initial_state: AtomicState) -> AtomicState:
        """
        Evolve state toward equilibrium.
        """
        # Start with initial state
        position = initial_state.position.clone()
        momentum = initial_state.momentum.clone()
        charge = initial_state.charge.clone()

        # Fixed properties don't evolve
        mass = initial_state.mass
        valence = initial_state.valence
        identity = initial_state.identity

        for step in range(self.num_steps):
            # Create current state
            current_state = AtomicState(
                charge=charge,
                position=position,
                momentum=momentum,
                mass=mass,
                valence=valence,
                identity=identity
            )

            # Compute energy gradient w.r.t. position
            position.requires_grad_(True)
            state_for_grad = AtomicState(
                charge=charge.detach(),
                position=position,
                momentum=momentum.detach(),
                mass=mass,
                valence=valence,
                identity=identity
            )

            energy = self.energy_fn.compute_energy(state_for_grad)
            total_energy = energy.sum()

            # Gradient of energy w.r.t. position
            grad_E = torch.autograd.grad(total_energy, position, create_graph=True)[0]
            position.requires_grad_(False)

            # Damped dynamics update
            # F = -∇E - γv
            force = -grad_E - self.damping * momentum

            # Update momentum and position
            momentum = momentum + self.step_size * force / mass
            position = position + self.step_size * momentum

            # Also evolve charge slightly (allows polarity adjustment)
            charge_grad = torch.autograd.grad(
                total_energy, charge.requires_grad_(True),
                create_graph=True
            )[0]
            charge.requires_grad_(False)
            charge = torch.tanh(charge - 0.01 * charge_grad)  # Small charge update

        return AtomicState(
            charge=charge,
            position=position,
            momentum=momentum,
            mass=mass,
            valence=valence,
            identity=identity
        )


# =============================================================================
# PART 5: ATTENTION EXTRACTION (From equilibrium to output)
# =============================================================================

class EquilibriumAttention(nn.Module):
    """
    Extract attention weights from equilibrium state.

    After dynamics, atoms have moved in semantic space.
    Atoms that are close together (low energy bond) attend to each other.

    This is where "attention" becomes explicit again,
    but now it's DERIVED from physics, not computed directly.
    """

    def __init__(self, position_dim: int, identity_dim: int, output_dim: int):
        super().__init__()

        self.energy_fn = EnergyFunctional(position_dim)

        # Value projection from identity
        self.v_proj = nn.Linear(identity_dim, output_dim)
        self.out_proj = nn.Linear(output_dim, output_dim)

    def forward(
        self,
        equilibrium_state: AtomicState,
        causal: bool = True
    ) -> torch.Tensor:
        """
        Extract output from equilibrium state.
        """
        B, N, D = equilibrium_state.position.shape

        # Compute pairwise energies at equilibrium
        pair_energy = self.energy_fn.compute_pairwise_energy(equilibrium_state)

        # Apply causal mask
        if causal:
            causal_mask = torch.triu(torch.ones(N, N, device=pair_energy.device), diagonal=1).bool()
            pair_energy = pair_energy.masked_fill(causal_mask.unsqueeze(0), float('inf'))

        # Energy to attention (lower energy = higher attention)
        # This is where sparsity emerges: high-energy pairs get ~0 attention
        attn = F.softmax(-pair_energy, dim=-1)

        # Value from identity (the fixed core meaning)
        V = self.v_proj(equilibrium_state.identity)  # (B, N, output_dim)

        # Apply attention
        output = torch.bmm(attn, V)  # (B, N, output_dim)

        return self.out_proj(output)


# =============================================================================
# PART 6: EQUILIBRIUM ASA LAYER
# =============================================================================

class EquilibriumASALayer(nn.Module):
    """
    A transformer layer based on equilibrium dynamics.

    1. Embed tokens as atoms with initial positions
    2. Evolve toward equilibrium (energy minimization)
    3. Extract attention from equilibrium configuration
    4. Standard FFN for feature transformation
    """

    def __init__(
        self,
        embed_dim: int,
        position_dim: int = 64,
        identity_dim: int = 128,
        num_steps: int = 5,
    ):
        super().__init__()

        self.embed_dim = embed_dim
        self.position_dim = position_dim
        self.identity_dim = identity_dim

        # Energy and dynamics
        self.energy_fn = EnergyFunctional(position_dim)
        self.dynamics = EquilibriumDynamics(self.energy_fn, num_steps=num_steps)

        # Attention extraction
        self.attention = EquilibriumAttention(position_dim, identity_dim, embed_dim)

        # Project input to initial atomic state
        self.to_position = nn.Linear(embed_dim, position_dim)
        self.to_identity = nn.Linear(embed_dim, identity_dim)
        self.to_charge = nn.Linear(embed_dim, 1)
        self.to_mass = nn.Linear(embed_dim, 1)
        self.to_valence = nn.Linear(embed_dim, 1)

        # Standard transformer components
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 4),
            nn.GELU(),
            nn.Linear(embed_dim * 4, embed_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, N, D = x.shape

        # Create initial atomic state from input
        x_norm = self.norm1(x)

        initial_state = AtomicState(
            charge=torch.tanh(self.to_charge(x_norm)),
            position=self.to_position(x_norm),
            momentum=torch.zeros(B, N, self.position_dim, device=x.device),
            mass=F.softplus(self.to_mass(x_norm)) + 0.5,
            valence=F.softplus(self.to_valence(x_norm)) + 1.0,
            identity=self.to_identity(x_norm),
        )

        # Evolve toward equilibrium
        equilibrium_state = self.dynamics(initial_state)

        # Extract attention-based output
        attn_out = self.attention(equilibrium_state)

        # Residual connection
        x = x + attn_out
        x = x + self.ffn(self.norm2(x))

        return x


# =============================================================================
# PART 7: FULL EQUILIBRIUM ASA TRANSFORMER
# =============================================================================

class EquilibriumASATransformer(nn.Module):
    """
    Complete transformer using equilibrium-based attention.
    """

    def __init__(
        self,
        vocab_size: int,
        embed_dim: int = 256,
        num_layers: int = 4,
        position_dim: int = 64,
        identity_dim: int = 128,
        num_steps: int = 5,
        max_seq_len: int = 512,
    ):
        super().__init__()

        self.embed_dim = embed_dim

        # Token embedding
        self.token_embed = nn.Embedding(vocab_size, embed_dim)
        self.pos_embed = nn.Embedding(max_seq_len, embed_dim)

        # Equilibrium layers
        self.layers = nn.ModuleList([
            EquilibriumASALayer(embed_dim, position_dim, identity_dim, num_steps)
            for _ in range(num_layers)
        ])

        self.norm = nn.LayerNorm(embed_dim)
        self.head = nn.Linear(embed_dim, vocab_size, bias=False)
        self.head.weight = self.token_embed.weight

    def forward(self, input_ids: torch.Tensor, targets: Optional[torch.Tensor] = None):
        B, N = input_ids.shape

        positions = torch.arange(N, device=input_ids.device).unsqueeze(0).expand(B, -1)
        x = self.token_embed(input_ids) + self.pos_embed(positions)

        for layer in self.layers:
            x = layer(x)

        x = self.norm(x)
        logits = self.head(x)

        loss = None
        if targets is not None:
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)),
                targets.view(-1),
                ignore_index=0
            )

        return logits, loss

    @torch.no_grad()
    def generate(self, prompt: torch.Tensor, max_new: int = 20, temperature: float = 0.8):
        self.eval()
        for _ in range(max_new):
            logits, _ = self(prompt[:, -512:])
            logits = logits[:, -1, :] / temperature
            probs = F.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, 1)
            prompt = torch.cat([prompt, next_token], dim=1)
        return prompt


# =============================================================================
# PART 8: STANDARD TRANSFORMER BASELINE
# =============================================================================

class StandardTransformer(nn.Module):
    """Standard transformer for comparison."""

    def __init__(
        self,
        vocab_size: int,
        embed_dim: int = 256,
        num_layers: int = 4,
        num_heads: int = 4,
        max_seq_len: int = 512,
    ):
        super().__init__()

        self.token_embed = nn.Embedding(vocab_size, embed_dim)
        self.pos_embed = nn.Embedding(max_seq_len, embed_dim)

        self.layers = nn.ModuleList([
            nn.TransformerEncoderLayer(
                d_model=embed_dim,
                nhead=num_heads,
                dim_feedforward=embed_dim * 4,
                batch_first=True,
            )
            for _ in range(num_layers)
        ])

        self.norm = nn.LayerNorm(embed_dim)
        self.head = nn.Linear(embed_dim, vocab_size, bias=False)
        self.head.weight = self.token_embed.weight

    def forward(self, x: torch.Tensor, targets: Optional[torch.Tensor] = None):
        B, N = x.shape

        positions = torch.arange(N, device=x.device).unsqueeze(0).expand(B, -1)
        h = self.token_embed(x) + self.pos_embed(positions)

        causal_mask = torch.triu(torch.ones(N, N, device=x.device), diagonal=1).bool()

        for layer in self.layers:
            h = layer(h, src_mask=causal_mask, is_causal=True)

        h = self.norm(h)
        logits = self.head(h)

        loss = None
        if targets is not None:
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)),
                targets.view(-1),
                ignore_index=0
            )

        return logits, loss

    @torch.no_grad()
    def generate(self, prompt: torch.Tensor, max_new: int = 20, temperature: float = 0.8):
        self.eval()
        for _ in range(max_new):
            logits, _ = self(prompt[:, -512:])
            logits = logits[:, -1, :] / temperature
            probs = F.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, 1)
            prompt = torch.cat([prompt, next_token], dim=1)
        return prompt


# =============================================================================
# DATASET
# =============================================================================

class PatternDataset(Dataset):
    def __init__(self, vocab_size: int = 256, seq_len: int = 64, num_samples: int = 10000):
        self.data = []
        for _ in range(num_samples):
            pattern = np.random.randint(4)
            if pattern == 0:
                a, b = np.random.randint(2, vocab_size, size=2)
                seq = [a, b] * (seq_len // 2)
            elif pattern == 1:
                start = np.random.randint(2, vocab_size - seq_len)
                seq = list(range(start, start + seq_len))
            elif pattern == 2:
                length = seq_len // 3
                segment = np.random.randint(2, vocab_size, size=length).tolist()
                seq = segment + [1] + segment + [0] * (seq_len - 2*length - 1)
            else:
                length = seq_len // 3
                segment = np.random.randint(2, vocab_size, size=length).tolist()
                seq = segment + [1] + segment[::-1] + [0] * (seq_len - 2*length - 1)
            self.data.append(seq[:seq_len])
        self.data = torch.tensor(self.data, dtype=torch.long)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        seq = self.data[idx]
        return seq[:-1], seq[1:]


# =============================================================================
# TRAINING
# =============================================================================

def train_epoch(model, loader, optimizer, device, clip_grad=1.0):
    model.train()
    total_loss = 0
    for inputs, targets in loader:
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        _, loss = model(inputs, targets)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), clip_grad)
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(loader)


def evaluate(model, loader, device):
    model.eval()
    total_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, targets in loader:
            inputs, targets = inputs.to(device), targets.to(device)
            logits, loss = model(inputs, targets)
            total_loss += loss.item()
            preds = logits.argmax(dim=-1)
            mask = targets != 0
            correct += ((preds == targets) & mask).sum().item()
            total += mask.sum().item()
    return total_loss / len(loader), correct / total if total > 0 else 0


def count_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


# =============================================================================
# VISUALIZATION OF ENERGY LANDSCAPE
# =============================================================================

def visualize_energy_landscape(model, sample_input, device):
    """Visualize how energy evolves during dynamics."""
    model.eval()

    with torch.no_grad():
        # Get the first layer
        layer = model.layers[0]

        # Create initial state
        B, N = sample_input.shape
        positions = torch.arange(N, device=device).unsqueeze(0).expand(B, -1)
        x = model.token_embed(sample_input) + model.pos_embed(positions)
        x_norm = layer.norm1(x)

        initial_state = AtomicState(
            charge=torch.tanh(layer.to_charge(x_norm)),
            position=layer.to_position(x_norm),
            momentum=torch.zeros(B, N, layer.position_dim, device=device),
            mass=F.softplus(layer.to_mass(x_norm)) + 0.5,
            valence=F.softplus(layer.to_valence(x_norm)) + 1.0,
            identity=layer.to_identity(x_norm),
        )

        # Track energy over dynamics
        energies = []
        state = initial_state

        for step in range(layer.dynamics.num_steps + 1):
            E, components = layer.energy_fn.compute_energy(state, return_components=True)
            energies.append({
                'step': step,
                'total': E.mean().item(),
                **{k: v.item() for k, v in components.items()}
            })

            if step < layer.dynamics.num_steps:
                state = layer.dynamics(state)

        return energies


# =============================================================================
# MAIN
# =============================================================================

def main():
    print("=" * 70)
    print("EQUILIBRIUM ASA: Attention as Energy Minimization")
    print("=" * 70)

    VOCAB_SIZE = 256
    EMBED_DIM = 256
    NUM_LAYERS = 4
    SEQ_LEN = 48  # Slightly shorter for memory with dynamics
    BATCH_SIZE = 24
    NUM_EPOCHS = 15
    LR = 3e-4

    print("\nPhilosophy:")
    print("  - Attention EMERGES from energy minimization")
    print("  - Sparsity EMERGES from energy landscape")
    print("  - No masks, no ripples - just physics")

    print("\n1. Creating datasets...")
    train_data = PatternDataset(VOCAB_SIZE, SEQ_LEN, 6000)
    val_data = PatternDataset(VOCAB_SIZE, SEQ_LEN, 800)
    train_loader = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_data, batch_size=BATCH_SIZE)

    print("\n2. Creating models...")

    equilibrium = EquilibriumASATransformer(
        VOCAB_SIZE, EMBED_DIM, NUM_LAYERS,
        position_dim=48,
        identity_dim=96,
        num_steps=3,
        max_seq_len=SEQ_LEN+1
    ).to(DEVICE)

    standard = StandardTransformer(
        VOCAB_SIZE, EMBED_DIM, NUM_LAYERS,
        num_heads=4,
        max_seq_len=SEQ_LEN+1
    ).to(DEVICE)

    print(f"   Equilibrium ASA: {count_params(equilibrium):,} params")
    print(f"   Standard:        {count_params(standard):,} params")

    # Optimizers
    eq_opt = torch.optim.AdamW(equilibrium.parameters(), lr=LR)
    std_opt = torch.optim.AdamW(standard.parameters(), lr=LR)

    print("\n3. Training...")
    print("-" * 70)

    for epoch in range(NUM_EPOCHS):
        t0 = time.time()

        eq_loss = train_epoch(equilibrium, train_loader, eq_opt, DEVICE)
        std_loss = train_epoch(standard, train_loader, std_opt, DEVICE)

        eq_val_loss, eq_acc = evaluate(equilibrium, val_loader, DEVICE)
        std_val_loss, std_acc = evaluate(standard, val_loader, DEVICE)

        elapsed = time.time() - t0

        print(f"\nEpoch {epoch+1:2d}/{NUM_EPOCHS} ({elapsed:.1f}s)")
        print(f"  Equilibrium: train={eq_loss:.4f}, val={eq_val_loss:.4f}, acc={eq_acc:.1%}")
        print(f"  Standard:    train={std_loss:.4f}, val={std_val_loss:.4f}, acc={std_acc:.1%}")

    # Final results
    print("\n" + "=" * 70)
    print("FINAL RESULTS")
    print("=" * 70)

    eq_val_loss, eq_acc = evaluate(equilibrium, val_loader, DEVICE)
    std_val_loss, std_acc = evaluate(standard, val_loader, DEVICE)

    print(f"\nEquilibrium ASA: {eq_acc:.1%}")
    print(f"Standard:        {std_acc:.1%}")
    print(f"Difference:      {eq_acc - std_acc:+.1%}")

    # Energy landscape visualization
    print("\n" + "=" * 70)
    print("ENERGY LANDSCAPE (First Layer)")
    print("=" * 70)

    sample = torch.tensor([[10, 20, 10, 20, 10, 20, 10, 20]], device=DEVICE)
    energies = visualize_energy_landscape(equilibrium, sample, DEVICE)

    print("\nEnergy evolution during dynamics:")
    for e in energies:
        print(f"  Step {e['step']}: total={e['total']:.3f} "
              f"(field={e['E_field']:.3f}, dist={e['E_distance']:.3f}, "
              f"charge={e['E_charge']:.3f}, valence={e['E_valence']:.3f})")

    # Generation test
    print("\n" + "=" * 70)
    print("GENERATION TEST")
    print("=" * 70)

    prompts = [
        ([10, 20, 10, 20, 10], "Repetition"),
        ([50, 51, 52, 53, 54], "Counting"),
    ]

    for prompt_list, name in prompts:
        prompt = torch.tensor([prompt_list], device=DEVICE)
        print(f"\n{name}: {prompt_list}")

        eq_gen = equilibrium.generate(prompt.clone(), max_new=10)
        std_gen = standard.generate(prompt.clone(), max_new=10)

        print(f"  Equilibrium: {eq_gen[0].tolist()}")
        print(f"  Standard:    {std_gen[0].tolist()}")

    print("\n" + "=" * 70)
    print("COMPLETE")
    print("=" * 70)

    return equilibrium, standard


if __name__ == "__main__":
    main()
