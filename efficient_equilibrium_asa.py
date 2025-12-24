"""
EFFICIENT EQUILIBRIUM ASA: O(N) Scaling via Kernel Approximations
===================================================================

Building on equilibrium_asa.py, this adds:

1. Random Fourier Features for O(N) kernel approximation
2. Hopfield-inspired energy formulation
3. Linear attention derivation from equilibrium

Key insight from Hopfield Networks is All You Need:
    softmax(QK^T)V ≈ σ(Q)σ(K)^T V = σ(Q)[σ(K)^T V]
                                     \_____O(N)_____/

We adapt this: instead of learned Q,K, use physics-derived features.

The energy between atoms i,j can be written as:
    E_ij = φ(atom_i)^T ψ(atom_j) + baseline

If we choose φ, ψ to be feature maps, we get:
    Attention ∝ exp(-E) = exp(-φ^T ψ) × exp(-baseline)

Using Random Fourier Features:
    exp(-||x-y||²/2σ²) ≈ z(x)^T z(y)
    where z(x) = [cos(ωx + b), sin(ωx + b)] / √D

This gives us O(N) attention!

Run: python efficient_equilibrium_asa.py
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import numpy as np
import time
import math
from typing import Optional, Tuple, NamedTuple


# Device
if torch.backends.mps.is_available():
    DEVICE = torch.device("mps")
elif torch.cuda.is_available():
    DEVICE = torch.device("cuda")
else:
    DEVICE = torch.device("cpu")
print(f"Using {DEVICE}")


# =============================================================================
# RANDOM FOURIER FEATURES FOR EFFICIENT KERNELS
# =============================================================================

class RandomFourierFeatures(nn.Module):
    """
    Approximate Gaussian kernel with random Fourier features.

    k(x, y) = exp(-||x-y||² / 2σ²) ≈ φ(x)^T φ(y)

    where φ(x) = √(2/D) * [cos(ω₁x + b₁), sin(ω₁x + b₁), ...]

    This allows O(N) computation of kernel sums:
        Σⱼ k(xᵢ, xⱼ) yⱼ = φ(xᵢ)^T [Σⱼ φ(xⱼ) yⱼ]
                                  \____O(N)____/
    """

    def __init__(
        self,
        input_dim: int,
        num_features: int = 64,
        sigma: float = 1.0,
        learnable: bool = True
    ):
        super().__init__()

        self.input_dim = input_dim
        self.num_features = num_features

        # Random frequencies from N(0, 1/σ²)
        omega = torch.randn(num_features, input_dim) / sigma

        # Random phases from U(0, 2π)
        bias = torch.rand(num_features) * 2 * math.pi

        if learnable:
            self.omega = nn.Parameter(omega)
            self.bias = nn.Parameter(bias)
        else:
            self.register_buffer('omega', omega)
            self.register_buffer('bias', bias)

        self.scale = math.sqrt(2.0 / num_features)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Compute random Fourier features.

        Args:
            x: (..., input_dim)

        Returns:
            φ(x): (..., num_features * 2)
        """
        # Project: ωx + b
        projection = torch.matmul(x, self.omega.T) + self.bias  # (..., num_features)

        # Cos and sin features
        cos_feat = torch.cos(projection) * self.scale
        sin_feat = torch.sin(projection) * self.scale

        return torch.cat([cos_feat, sin_feat], dim=-1)


# =============================================================================
# EFFICIENT ENERGY COMPUTATION
# =============================================================================

class EfficientEnergyComputer(nn.Module):
    """
    Compute energies efficiently using kernel approximations.

    Key insight: The energy function can be decomposed as:
        E_ij = charge_interaction + distance_kernel + shell_similarity

    Each component can be computed in O(N) using:
    1. Aggregated statistics (mean, sum)
    2. Random Fourier Features for distance kernels
    """

    def __init__(
        self,
        position_dim: int,
        num_rff_features: int = 32,
        baseline: float = 2.0,
        charge_weight: float = 1.0,
        distance_weight: float = 1.0,
    ):
        super().__init__()

        self.baseline = baseline
        self.charge_weight = charge_weight
        self.distance_weight = distance_weight

        # RFF for distance-based interaction
        self.rff = RandomFourierFeatures(
            position_dim,
            num_features=num_rff_features,
            sigma=math.sqrt(position_dim),
            learnable=True
        )

    def compute_field_statistics(
        self,
        position: torch.Tensor,
        charge: torch.Tensor,
        values: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute aggregated field statistics (O(N)).

        Returns:
            - charge_field: aggregate charge contribution
            - kernel_field: RFF-weighted aggregate
        """
        B, N, D = position.shape

        # Charge field: weighted sum of positions by charge
        # (B, D) = sum over N of position * charge
        charge_field = (position * charge).sum(dim=1)  # (B, D)

        # Kernel field using RFF
        phi = self.rff(position)  # (B, N, num_features*2)
        # Aggregate: Σⱼ φ(xⱼ) * valueⱼ
        kernel_field = torch.bmm(phi.transpose(1, 2), values)  # (B, num_features*2, value_dim)

        return charge_field, kernel_field

    def compute_energy_from_fields(
        self,
        position: torch.Tensor,
        charge: torch.Tensor,
        charge_field: torch.Tensor,
        kernel_field: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute per-atom energy from pre-computed fields (O(N)).
        """
        B, N, D = position.shape

        # Energy from charge field: how much each atom aligns with aggregate charge
        charge_energy = self.charge_weight * charge.squeeze(-1) * (
            (position * charge_field.unsqueeze(1)).sum(dim=-1) / N
        )

        # Energy from kernel field (distance-based)
        phi = self.rff(position)  # (B, N, num_features*2)
        # φ(xᵢ)^T [Σⱼ φ(xⱼ) vⱼ]
        kernel_response = torch.bmm(phi, kernel_field)  # (B, N, value_dim)
        kernel_energy = -self.distance_weight * kernel_response.mean(dim=-1)

        # Total per-atom energy
        E_atom = self.baseline + charge_energy + kernel_energy

        return E_atom  # (B, N)


# =============================================================================
# HOPFIELD-INSPIRED ATTENTION
# =============================================================================

class HopfieldASAAttention(nn.Module):
    """
    Attention derived from Hopfield energy minimization.

    Modern Hopfield Network energy:
        E = -Σᵢ log(Σⱼ exp(xᵢ^T ξⱼ))

    The update rule minimizing this energy IS attention:
        x_new = softmax(x ξ^T) ξ

    We modify this with ASA physics:
        E = -Σᵢ log(Σⱼ exp(-E_physics(atomᵢ, atomⱼ)))

    Where E_physics includes charge, distance, etc.
    """

    def __init__(
        self,
        embed_dim: int,
        position_dim: int = 64,
        num_rff: int = 32,
        temperature: float = 1.0,
    ):
        super().__init__()

        self.embed_dim = embed_dim
        self.position_dim = position_dim
        self.temperature = temperature

        # Project to atomic properties
        self.to_position = nn.Linear(embed_dim, position_dim)
        self.to_charge = nn.Linear(embed_dim, 1)

        # RFF for efficient distance computation
        self.rff = RandomFourierFeatures(position_dim, num_rff, learnable=True)

        # Value and output projections
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        self.out_proj = nn.Linear(embed_dim, embed_dim)

        # Learnable energy weights
        self.charge_scale = nn.Parameter(torch.tensor(1.0))
        self.distance_scale = nn.Parameter(torch.tensor(1.0))

    def forward(self, x: torch.Tensor, causal: bool = True) -> torch.Tensor:
        """
        Hopfield-ASA attention.

        Uses physics-based similarity instead of QK dot product.
        """
        B, N, D = x.shape

        # Extract atomic properties
        position = self.to_position(x)  # (B, N, position_dim)
        charge = torch.tanh(self.to_charge(x))  # (B, N, 1)

        # Compute RFF features
        phi = self.rff(position)  # (B, N, num_rff * 2)

        # === Efficient Attention via RFF ===
        # Standard: attn = softmax(QK^T) = softmax(φ(Q) φ(K)^T)
        # Linear: attn ≈ φ(Q) [φ(K)^T]  (unnormalized)

        # For physics-based attention, we want:
        # score_ij = -E_distance - E_charge
        # E_distance ∝ ||pos_i - pos_j||² → approximated by φ(pos_i)^T φ(pos_j)
        # E_charge ∝ charge_i * charge_j

        # Distance similarity via RFF (higher = closer = lower energy)
        dist_score = torch.bmm(phi, phi.transpose(1, 2))  # (B, N, N)
        dist_score = dist_score * F.softplus(self.distance_scale)

        # Charge interaction (opposite attracts)
        charge_score = -torch.bmm(charge, charge.transpose(1, 2))  # (B, N, N)
        charge_score = charge_score * F.softplus(self.charge_scale)

        # Combined physics score (higher = stronger bond = more attention)
        scores = dist_score + charge_score

        # Apply causal mask
        if causal:
            causal_mask = torch.triu(torch.ones(N, N, device=x.device), diagonal=1).bool()
            scores = scores.masked_fill(causal_mask.unsqueeze(0), float('-inf'))

        # Softmax attention
        attn = F.softmax(scores / self.temperature, dim=-1)

        # Apply to values
        V = self.v_proj(x)  # (B, N, D)
        out = torch.bmm(attn, V)

        return self.out_proj(out)


# =============================================================================
# LINEAR HOPFIELD-ASA ATTENTION (TRUE O(N))
# =============================================================================

class LinearHopfieldASA(nn.Module):
    """
    True O(N) attention using linearized Hopfield formulation.

    Key insight: For positive features, we can use:
        softmax(QK^T)V ≈ elu(Q)+1 @ [elu(K)+1]^T @ V
                       = elu(Q)+1 @ [(elu(K)+1)^T V]
                                    \_____ O(N) ____/

    We adapt this with physics features:
        ψ(atom) = [elu(φ_charge) + 1, elu(φ_position) + 1, ...]

    Then:
        output_i = ψ(atom_i)^T [Σⱼ ψ(atom_j) ⊗ value_j] / [Σⱼ ψ(atom_j)]
    """

    def __init__(
        self,
        embed_dim: int,
        feature_dim: int = 64,
        eps: float = 1e-6,
    ):
        super().__init__()

        self.embed_dim = embed_dim
        self.feature_dim = feature_dim
        self.eps = eps

        # Project to physics-inspired features
        self.query_proj = nn.Linear(embed_dim, feature_dim)
        self.key_proj = nn.Linear(embed_dim, feature_dim)
        self.value_proj = nn.Linear(embed_dim, embed_dim)
        self.out_proj = nn.Linear(embed_dim, embed_dim)

    def feature_map(self, x: torch.Tensor) -> torch.Tensor:
        """Positive feature map for linear attention."""
        return F.elu(x) + 1  # Always positive

    def forward(self, x: torch.Tensor, causal: bool = True) -> torch.Tensor:
        B, N, D = x.shape

        Q = self.feature_map(self.query_proj(x))  # (B, N, feature_dim)
        K = self.feature_map(self.key_proj(x))    # (B, N, feature_dim)
        V = self.value_proj(x)                     # (B, N, embed_dim)

        if causal:
            # Causal linear attention using cumsum
            # For each position i: output_i = (Σⱼ≤ᵢ Kⱼ ⊗ Vⱼ) @ Qᵢ / (Σⱼ≤ᵢ Kⱼ @ Qᵢ)

            # Cumulative sum of K^T V
            KV = torch.bmm(K.transpose(1, 2), V)  # Would be (B, F, D) but need causal

            # For true causal, we need cumsum trick
            # KV[i] = Σⱼ≤ᵢ K[j]^T V[j]
            KV_cumsum = torch.zeros(B, self.feature_dim, D, device=x.device)
            K_cumsum = torch.zeros(B, self.feature_dim, device=x.device)

            outputs = []
            for i in range(N):
                Ki = K[:, i, :]  # (B, F)
                Vi = V[:, i, :]  # (B, D)
                Qi = Q[:, i, :]  # (B, F)

                # Update cumulative sums
                KV_cumsum = KV_cumsum + Ki.unsqueeze(-1) * Vi.unsqueeze(1)  # (B, F, D)
                K_cumsum = K_cumsum + Ki  # (B, F)

                # Compute output
                numerator = torch.bmm(Qi.unsqueeze(1), KV_cumsum).squeeze(1)  # (B, D)
                denominator = (Qi * K_cumsum).sum(dim=-1, keepdim=True) + self.eps  # (B, 1)

                outputs.append(numerator / denominator)

            out = torch.stack(outputs, dim=1)  # (B, N, D)

        else:
            # Non-causal: full linear attention
            # KV = K^T V: (B, F, D)
            KV = torch.bmm(K.transpose(1, 2), V)

            # QKV = Q (K^T V): (B, N, D)
            numerator = torch.bmm(Q, KV)

            # Normalization: Q K^T 1
            K_sum = K.sum(dim=1, keepdim=True)  # (B, 1, F)
            denominator = (Q * K_sum).sum(dim=-1, keepdim=True) + self.eps  # (B, N, 1)

            out = numerator / denominator

        return self.out_proj(out)


# =============================================================================
# EFFICIENT EQUILIBRIUM LAYER
# =============================================================================

class EfficientEquilibriumLayer(nn.Module):
    """
    Transformer layer with efficient equilibrium-based attention.

    Combines:
    1. Physics-inspired feature extraction
    2. Hopfield-style energy formulation
    3. Linear attention for O(N) scaling
    """

    def __init__(
        self,
        embed_dim: int,
        attention_type: str = "hopfield",  # "hopfield", "linear", or "hybrid"
        position_dim: int = 64,
    ):
        super().__init__()

        self.attention_type = attention_type

        if attention_type == "hopfield":
            self.attn = HopfieldASAAttention(embed_dim, position_dim)
        elif attention_type == "linear":
            self.attn = LinearHopfieldASA(embed_dim, position_dim)
        elif attention_type == "hybrid":
            # Use Hopfield for first half, linear for second
            self.attn_hopfield = HopfieldASAAttention(embed_dim, position_dim)
            self.attn_linear = LinearHopfieldASA(embed_dim, position_dim)
            self.gate = nn.Linear(embed_dim, 1)
        else:
            raise ValueError(f"Unknown attention type: {attention_type}")

        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 4),
            nn.GELU(),
            nn.Linear(embed_dim * 4, embed_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_norm = self.norm1(x)

        if self.attention_type == "hybrid":
            h_out = self.attn_hopfield(x_norm)
            l_out = self.attn_linear(x_norm)
            gate = torch.sigmoid(self.gate(x_norm))
            attn_out = gate * h_out + (1 - gate) * l_out
        else:
            attn_out = self.attn(x_norm)

        x = x + attn_out
        x = x + self.ffn(self.norm2(x))
        return x


# =============================================================================
# EFFICIENT EQUILIBRIUM TRANSFORMER
# =============================================================================

class EfficientEquilibriumTransformer(nn.Module):
    """Transformer using efficient equilibrium attention."""

    def __init__(
        self,
        vocab_size: int,
        embed_dim: int = 256,
        num_layers: int = 4,
        attention_type: str = "hopfield",
        position_dim: int = 64,
        max_seq_len: int = 512,
    ):
        super().__init__()

        self.embed_dim = embed_dim
        self.attention_type = attention_type

        self.token_embed = nn.Embedding(vocab_size, embed_dim)
        self.pos_embed = nn.Embedding(max_seq_len, embed_dim)

        self.layers = nn.ModuleList([
            EfficientEquilibriumLayer(embed_dim, attention_type, position_dim)
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
# STANDARD TRANSFORMER BASELINE
# =============================================================================

class StandardBlock(nn.Module):
    def __init__(self, embed_dim: int, num_heads: int = 4):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.scale = self.head_dim ** -0.5

        self.qkv = nn.Linear(embed_dim, embed_dim * 3)
        self.out_proj = nn.Linear(embed_dim, embed_dim)

        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 4),
            nn.GELU(),
            nn.Linear(embed_dim * 4, embed_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, N, D = x.shape
        H = self.num_heads

        x_norm = self.norm1(x)
        qkv = self.qkv(x_norm).reshape(B, N, 3, H, self.head_dim).permute(2, 0, 3, 1, 4)
        Q, K, V = qkv[0], qkv[1], qkv[2]

        attn = torch.matmul(Q, K.transpose(-2, -1)) * self.scale
        causal = torch.triu(torch.ones(N, N, device=x.device), diagonal=1).bool()
        attn = attn.masked_fill(causal.unsqueeze(0).unsqueeze(0), float('-inf'))
        attn = F.softmax(attn, dim=-1)

        out = torch.matmul(attn, V)
        out = out.permute(0, 2, 1, 3).reshape(B, N, D)

        x = x + self.out_proj(out)
        x = x + self.ffn(self.norm2(x))
        return x


class StandardTransformer(nn.Module):
    def __init__(self, vocab_size: int, embed_dim: int = 256, num_layers: int = 4, max_seq_len: int = 512):
        super().__init__()
        self.token_embed = nn.Embedding(vocab_size, embed_dim)
        self.pos_embed = nn.Embedding(max_seq_len, embed_dim)
        self.layers = nn.ModuleList([StandardBlock(embed_dim) for _ in range(num_layers)])
        self.norm = nn.LayerNorm(embed_dim)
        self.head = nn.Linear(embed_dim, vocab_size, bias=False)
        self.head.weight = self.token_embed.weight

    def forward(self, x: torch.Tensor, targets: Optional[torch.Tensor] = None):
        B, N = x.shape
        positions = torch.arange(N, device=x.device).unsqueeze(0).expand(B, -1)
        h = self.token_embed(x) + self.pos_embed(positions)
        for layer in self.layers:
            h = layer(h)
        logits = self.head(self.norm(h))
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=0)
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
        return self.data[idx][:-1], self.data[idx][1:]


# =============================================================================
# TRAINING
# =============================================================================

def train_epoch(model, loader, optimizer, device):
    model.train()
    total_loss = 0
    for inputs, targets in loader:
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        _, loss = model(inputs, targets)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
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


def measure_speed(model, device, batch_size=32, seq_len=64, num_trials=10):
    """Measure forward pass speed."""
    model.eval()
    x = torch.randint(2, 256, (batch_size, seq_len), device=device)

    # Warmup
    with torch.no_grad():
        for _ in range(3):
            model(x)

    # Measure
    torch.cuda.synchronize() if device.type == 'cuda' else None
    t0 = time.time()
    with torch.no_grad():
        for _ in range(num_trials):
            model(x)
    torch.cuda.synchronize() if device.type == 'cuda' else None
    elapsed = time.time() - t0

    return elapsed / num_trials * 1000  # ms per forward


# =============================================================================
# MAIN
# =============================================================================

def main():
    print("=" * 70)
    print("EFFICIENT EQUILIBRIUM ASA")
    print("Hopfield-inspired attention with O(N) linear variant")
    print("=" * 70)

    VOCAB_SIZE = 256
    EMBED_DIM = 256
    NUM_LAYERS = 4
    SEQ_LEN = 64
    BATCH_SIZE = 32
    NUM_EPOCHS = 15
    LR = 3e-4

    print("\n1. Creating datasets...")
    train_data = PatternDataset(VOCAB_SIZE, SEQ_LEN, 8000)
    val_data = PatternDataset(VOCAB_SIZE, SEQ_LEN, 1000)
    train_loader = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_data, batch_size=BATCH_SIZE)

    print("\n2. Creating models...")

    models = {
        "Standard (baseline)": StandardTransformer(
            VOCAB_SIZE, EMBED_DIM, NUM_LAYERS, SEQ_LEN+1
        ),
        "Hopfield-ASA (O(N²))": EfficientEquilibriumTransformer(
            VOCAB_SIZE, EMBED_DIM, NUM_LAYERS, "hopfield", max_seq_len=SEQ_LEN+1
        ),
        "Linear-ASA (O(N))": EfficientEquilibriumTransformer(
            VOCAB_SIZE, EMBED_DIM, NUM_LAYERS, "linear", max_seq_len=SEQ_LEN+1
        ),
        "Hybrid-ASA": EfficientEquilibriumTransformer(
            VOCAB_SIZE, EMBED_DIM, NUM_LAYERS, "hybrid", max_seq_len=SEQ_LEN+1
        ),
    }

    for name, model in models.items():
        model.to(DEVICE)
        print(f"   {name}: {count_params(model):,} params")

    optimizers = {name: torch.optim.AdamW(m.parameters(), lr=LR) for name, m in models.items()}

    # Measure speed before training
    print("\n3. Speed comparison...")
    for name, model in models.items():
        speed = measure_speed(model, DEVICE)
        print(f"   {name}: {speed:.1f} ms/forward")

    print("\n4. Training...")
    print("-" * 70)

    for epoch in range(NUM_EPOCHS):
        t0 = time.time()

        results = {}
        for name, model in models.items():
            train_loss = train_epoch(model, train_loader, optimizers[name], DEVICE)
            val_loss, acc = evaluate(model, val_loader, DEVICE)
            results[name] = acc

        elapsed = time.time() - t0

        print(f"\nEpoch {epoch+1:2d}/{NUM_EPOCHS} ({elapsed:.1f}s)")
        for name, acc in results.items():
            print(f"  {name:25s}: {acc:.1%}")

    # Final results
    print("\n" + "=" * 70)
    print("FINAL RESULTS")
    print("=" * 70)

    for name, model in models.items():
        val_loss, acc = evaluate(model, val_loader, DEVICE)
        speed = measure_speed(model, DEVICE)
        print(f"{name:25s}: acc={acc:.1%}, speed={speed:.1f}ms")

    # Generation test
    print("\n" + "=" * 70)
    print("GENERATION TEST")
    print("=" * 70)

    prompts = [([10, 20, 10, 20, 10], "Repetition"), ([50, 51, 52, 53, 54], "Counting")]

    for prompt_list, pattern in prompts:
        prompt = torch.tensor([prompt_list], device=DEVICE)
        print(f"\n{pattern}: {prompt_list}")
        for name in ["Standard (baseline)", "Hopfield-ASA (O(N²))", "Linear-ASA (O(N))"]:
            gen = models[name].generate(prompt.clone(), max_new=10)
            print(f"  {name:25s}: {gen[0].tolist()}")

    print("\n" + "=" * 70)
    print("COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()
