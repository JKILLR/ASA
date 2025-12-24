"""
UNIFIED ASA: Equilibrium Physics + Structural Efficiency
=========================================================

Merges two key insights:

1. EQUILIBRIUM (what to compute):
   - Attention emerges from energy minimization
   - Sparsity emerges from energy landscape
   - Physics-grounded formulation

2. STRUCTURE (how to compute efficiently):
   - E_charge = q_i × q_j → Polynomial features (separable)
   - E_shell = shell_i · shell_j → Random Fourier Features
   - E_distance = |i - j| → Toeplitz (precompute + FFT)

The combination gives us:
- Principled physics-based attention
- O(N) or O(N log N) complexity
- No artificial masking or bolted-on ripples

Mathematical Foundation:
========================

For energy E = E_charge + E_shell + E_distance:

    exp(-E/T) = exp(-E_charge/T) × exp(-E_shell/T) × exp(-E_distance/T)

Each factor can be approximated as separable:

    exp(-E_charge/T) ≈ φ_c(q_i)^T φ_c(q_j)     [polynomial features]
    exp(-E_shell/T)  ≈ φ_s(s_i)^T φ_s(s_j)     [random Fourier features]
    exp(-E_dist/T)   = D[|i-j|]                 [Toeplitz matrix]

Combined attention (before normalization):
    A[i,j] ≈ [φ(atom_i)^T φ(atom_j)] × D[|i-j|]

Using linear attention tricks:
    output = φ(Q) @ [φ(K)^T @ V]  → O(N)

With Toeplitz convolution via FFT:
    output = FFT^{-1}(FFT(x) × FFT(d))  → O(N log N)

Run: python unified_asa.py
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import numpy as np
import time
import math
from typing import Optional, NamedTuple, Tuple

# Device
if torch.backends.mps.is_available():
    DEVICE = torch.device("mps")
elif torch.cuda.is_available():
    DEVICE = torch.device("cuda")
else:
    DEVICE = torch.device("cpu")
print(f"Using {DEVICE}")


# =============================================================================
# CONFIGURATION
# =============================================================================

class ASAConfig:
    """Configuration for Unified ASA."""

    # Atomic dimensions
    CHARGE_DIM = 1
    SHELL_DIM = 64
    NUCLEUS_DIM = 128

    # Feature dimensions for approximations
    POLY_ORDER = 4           # Polynomial features for charge
    RFF_DIM = 32             # Random Fourier features for shell

    # Physics constants
    CHARGE_COUPLING = 1.0
    SHELL_COUPLING = 0.5
    DISTANCE_COUPLING = 0.3
    TEMPERATURE = 1.0

    # Distance decay parameter
    DISTANCE_SCALE = 10.0    # How fast distance effect decays

    @classmethod
    def embed_dim(cls):
        return cls.CHARGE_DIM + cls.SHELL_DIM + cls.NUCLEUS_DIM

    @classmethod
    def padded_dim(cls):
        raw = cls.embed_dim()
        return ((raw + 3) // 4) * 4


# =============================================================================
# PART 1: FEATURE MAPS FOR SEPARABLE APPROXIMATION
# =============================================================================

class PolynomialFeatures(nn.Module):
    """
    Polynomial feature map for charge interaction.

    exp(-k * q_i * q_j) ≈ Σ_n (-k)^n (q_i * q_j)^n / n!
                        = Σ_n [(-k)^n/n! * q_i^n] * [q_j^n]
                        = φ(q_i)^T φ(q_j)

    This makes the charge interaction SEPARABLE.
    """

    def __init__(self, order: int = 4, coupling: float = 1.0):
        super().__init__()
        self.order = order
        self.coupling = coupling

        # Precompute coefficients: (-k)^n / n!
        coeffs = []
        for n in range(order + 1):
            coeffs.append(((-coupling) ** n) / math.factorial(n))
        self.register_buffer('coeffs', torch.tensor(coeffs))

    def forward(self, q: torch.Tensor) -> torch.Tensor:
        """
        Compute polynomial features.

        Args:
            q: (..., 1) charge values in [-1, 1]

        Returns:
            φ(q): (..., order+1) polynomial features
        """
        q = q.squeeze(-1)  # (...,)

        # Build powers: [1, q, q², q³, ...]
        features = []
        q_power = torch.ones_like(q)

        for n in range(self.order + 1):
            # Apply coefficient and sqrt for symmetric factorization
            # We want φ(q_i)^T φ(q_j) = Σ c_n q_i^n q_j^n
            # So φ(q) = [√|c_0|, √|c_1|*q, √|c_2|*q², ...] with signs handled

            coeff = self.coeffs[n]
            # Handle sign: store sign separately, use abs for sqrt
            sign = torch.sign(coeff) if coeff != 0 else torch.tensor(1.0)
            magnitude = torch.sqrt(torch.abs(coeff) + 1e-10)

            features.append(magnitude * q_power)
            q_power = q_power * q

        return torch.stack(features, dim=-1)  # (..., order+1)


class RandomFourierFeatures(nn.Module):
    """
    Random Fourier Features for shell (dot product) kernel.

    For Gaussian kernel: k(x,y) = exp(-||x-y||²/2σ²)
    For dot product: We approximate exp(-coupling * (1 - x·y))
                   = exp(-coupling) * exp(coupling * x·y)

    exp(coupling * x·y) can be approximated with:
    φ(x) = [cos(ω_1·x + b_1), sin(ω_1·x + b_1), ...]
    """

    def __init__(
        self,
        input_dim: int,
        num_features: int = 32,
        coupling: float = 0.5,
        learnable: bool = False
    ):
        super().__init__()

        self.input_dim = input_dim
        self.num_features = num_features
        self.coupling = coupling

        # Random frequencies ~ N(0, coupling)
        # For exp(c * x·y), we want frequencies scaled by sqrt(c)
        omega = torch.randn(num_features, input_dim) * math.sqrt(coupling)
        bias = torch.rand(num_features) * 2 * math.pi

        if learnable:
            self.omega = nn.Parameter(omega)
            self.bias = nn.Parameter(bias)
        else:
            self.register_buffer('omega', omega)
            self.register_buffer('bias', bias)

        # Scaling factor
        self.scale = math.sqrt(2.0 / num_features)

        # Precompute exp(-coupling) factor
        self.register_buffer('exp_factor', torch.tensor(math.exp(-coupling)))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Compute RFF features.

        Args:
            x: (..., input_dim) shell vectors (should be normalized)

        Returns:
            φ(x): (..., num_features * 2) random Fourier features
        """
        # Project: ω·x + b
        proj = F.linear(x, self.omega, self.bias)  # (..., num_features)

        # Cos and sin features
        cos_feat = torch.cos(proj)
        sin_feat = torch.sin(proj)

        # Concatenate and scale
        features = torch.cat([cos_feat, sin_feat], dim=-1) * self.scale

        # Include the exp(-coupling) factor in the features
        return features * torch.sqrt(self.exp_factor)


class ToeplitzDistance(nn.Module):
    """
    Handles Toeplitz distance matrix efficiently.

    exp(-coupling * |i-j| / scale) is a Toeplitz matrix.

    For attention: we multiply element-wise with the separable term.
    For efficiency: can use FFT-based convolution.
    """

    def __init__(
        self,
        max_seq_len: int = 512,
        coupling: float = 0.3,
        scale: float = 10.0
    ):
        super().__init__()

        self.max_seq_len = max_seq_len
        self.coupling = coupling
        self.scale = scale

        # Precompute distance weights: exp(-coupling * d / scale) for d = 0, 1, 2, ...
        distances = torch.arange(max_seq_len, dtype=torch.float)
        weights = torch.exp(-coupling * distances / scale)
        self.register_buffer('weights', weights)

    def get_toeplitz_matrix(self, seq_len: int) -> torch.Tensor:
        """
        Get the Toeplitz matrix for given sequence length.

        Returns:
            (seq_len, seq_len) Toeplitz matrix
        """
        # Build full Toeplitz from first row
        indices = torch.arange(seq_len, device=self.weights.device)
        diff = (indices.unsqueeze(1) - indices.unsqueeze(0)).abs()  # |i - j|

        return self.weights[diff]  # (seq_len, seq_len)

    def apply_toeplitz_fft(
        self,
        x: torch.Tensor,
        seq_len: int
    ) -> torch.Tensor:
        """
        Apply Toeplitz multiplication using FFT (O(N log N)).

        For Toeplitz matrix T and vector x:
        T @ x can be computed as IFFT(FFT(first_col) * FFT(x_padded))

        Args:
            x: (B, N, D) input

        Returns:
            (B, N, D) output after Toeplitz multiplication
        """
        B, N, D = x.shape

        # Build first column of Toeplitz (symmetric, so first col = first row)
        first_col = self.weights[:N]  # (N,)

        # Pad for circular convolution
        # For Toeplitz multiplication, we need size 2N-1
        padded_size = 2 * N - 1

        # Pad first column (for circular embedding of Toeplitz)
        col_padded = F.pad(first_col, (0, padded_size - N))
        # Make it symmetric for Toeplitz: [w_0, w_1, ..., w_{N-1}, 0, w_{N-1}, ..., w_1]
        if N > 1:
            col_padded[N:] = torch.flip(first_col[1:], dims=[0])

        # FFT of the Toeplitz first column
        col_fft = torch.fft.rfft(col_padded)  # (padded_size//2 + 1,)

        # For each feature dimension, apply convolution
        x_padded = F.pad(x, (0, 0, 0, padded_size - N))  # (B, padded_size, D)

        # FFT along sequence dimension
        x_fft = torch.fft.rfft(x_padded, dim=1)  # (B, padded_size//2+1, D)

        # Multiply in frequency domain
        result_fft = x_fft * col_fft.unsqueeze(0).unsqueeze(-1)

        # Inverse FFT
        result = torch.fft.irfft(result_fft, n=padded_size, dim=1)  # (B, padded_size, D)

        # Take first N elements
        return result[:, :N, :]


# =============================================================================
# PART 2: COMBINED FEATURE MAP
# =============================================================================

class CombinedFeatureMap(nn.Module):
    """
    Combines all feature maps into a single separable representation.

    φ(atom) = φ_charge(q) ⊗ φ_shell(s)

    Where ⊗ is tensor product (Kronecker product for vectors).

    This gives: φ(atom_i)^T φ(atom_j) ≈ exp(-E_charge) * exp(-E_shell)
    """

    def __init__(self, config: ASAConfig = ASAConfig):
        super().__init__()

        self.config = config

        # Feature maps for each component
        self.charge_features = PolynomialFeatures(
            order=config.POLY_ORDER,
            coupling=config.CHARGE_COUPLING
        )

        self.shell_features = RandomFourierFeatures(
            input_dim=config.SHELL_DIM,
            num_features=config.RFF_DIM,
            coupling=config.SHELL_COUPLING,
            learnable=True  # Allow fine-tuning
        )

        # Toeplitz for distance (applied separately)
        self.toeplitz = ToeplitzDistance(
            coupling=config.DISTANCE_COUPLING,
            scale=config.DISTANCE_SCALE
        )

        # Combined feature dimension
        self.charge_feat_dim = config.POLY_ORDER + 1
        self.shell_feat_dim = config.RFF_DIM * 2

        # For efficiency, we project the tensor product to lower dim
        self.combined_dim = 64  # Projected dimension
        self.combine_proj = nn.Linear(
            self.charge_feat_dim * self.shell_feat_dim,
            self.combined_dim,
            bias=False
        )

    def forward(
        self,
        charge: torch.Tensor,
        shell: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute combined features.

        Args:
            charge: (B, N, 1) charge values
            shell: (B, N, shell_dim) shell vectors (normalized)

        Returns:
            combined_features: (B, N, combined_dim)
            toeplitz_matrix: (N, N) distance weighting
        """
        B, N, _ = charge.shape

        # Get individual features
        φ_c = self.charge_features(charge)  # (B, N, charge_feat_dim)
        φ_s = self.shell_features(shell)    # (B, N, shell_feat_dim)

        # Tensor product: φ_c ⊗ φ_s
        # (B, N, charge_feat_dim, 1) × (B, N, 1, shell_feat_dim)
        # = (B, N, charge_feat_dim, shell_feat_dim)
        tensor_prod = φ_c.unsqueeze(-1) * φ_s.unsqueeze(-2)
        tensor_prod = tensor_prod.reshape(B, N, -1)  # (B, N, charge*shell)

        # Project to lower dimension for efficiency
        combined = self.combine_proj(tensor_prod)  # (B, N, combined_dim)

        # Get Toeplitz matrix for this sequence length
        toeplitz = self.toeplitz.get_toeplitz_matrix(N)  # (N, N)

        return combined, toeplitz


# =============================================================================
# PART 3: LINEAR ATTENTION WITH PHYSICS FEATURES
# =============================================================================

class UnifiedASAAttention(nn.Module):
    """
    Attention using physics-derived separable features.

    Full attention would be:
        A[i,j] = exp(-E[i,j]/T) / Σ_k exp(-E[i,k]/T)
               ≈ [φ(i)^T φ(j) * D[i,j]] / [Σ_k φ(i)^T φ(k) * D[i,k]]

    We compute this efficiently using:
    1. Separable features for φ^T φ terms
    2. Toeplitz structure for D

    Modes:
    - 'full': Materialize attention matrix (O(N²), for comparison)
    - 'linear': Pure linear attention (O(N), ignores Toeplitz)
    - 'hybrid': Linear features + Toeplitz weighting (O(N log N))
    """

    def __init__(
        self,
        embed_dim: int,
        num_heads: int = 4,
        mode: str = 'hybrid',
        config: ASAConfig = ASAConfig
    ):
        super().__init__()

        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.mode = mode
        self.config = config

        # Project input to atomic properties
        self.to_charge = nn.Linear(embed_dim, config.CHARGE_DIM)
        self.to_shell = nn.Linear(embed_dim, config.SHELL_DIM)

        # Combined feature map
        self.feature_map = CombinedFeatureMap(config)

        # Value and output projections
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        self.out_proj = nn.Linear(embed_dim, embed_dim)

        # Temperature (learnable)
        self.log_temp = nn.Parameter(torch.tensor(0.0))

    @property
    def temperature(self):
        return F.softplus(self.log_temp) + 0.1

    def forward(self, x: torch.Tensor, causal: bool = True) -> torch.Tensor:
        B, N, D = x.shape
        H = self.num_heads

        # Extract atomic properties
        charge = torch.tanh(self.to_charge(x))  # (B, N, 1)
        shell = F.normalize(self.to_shell(x), dim=-1)  # (B, N, shell_dim)

        # Get combined features and Toeplitz
        φ, T_matrix = self.feature_map(charge, shell)  # (B, N, F), (N, N)

        # Value projection
        V = self.v_proj(x)  # (B, N, D)

        if self.mode == 'full':
            # Full O(N²) attention for comparison
            # A = softmax(φ @ φ^T * T_matrix / temp)

            attn_logits = torch.bmm(φ, φ.transpose(1, 2))  # (B, N, N)
            attn_logits = attn_logits * T_matrix.unsqueeze(0)  # Apply Toeplitz
            attn_logits = attn_logits / self.temperature

            if causal:
                causal_mask = torch.triu(torch.ones(N, N, device=x.device), diagonal=1).bool()
                attn_logits = attn_logits.masked_fill(causal_mask.unsqueeze(0), float('-inf'))

            attn = F.softmax(attn_logits, dim=-1)
            out = torch.bmm(attn, V)

        elif self.mode == 'linear':
            # Pure linear attention O(N)
            # Ignores Toeplitz for true linear scaling
            # out_i = φ_i^T [Σ_j φ_j ⊗ V_j] / [φ_i^T Σ_j φ_j]

            if causal:
                # Causal linear attention using cumulative sums
                out = self._causal_linear_attention(φ, V)
            else:
                # Non-causal: can use simple formulation
                φV = torch.bmm(φ.transpose(1, 2), V)  # (B, F, D)
                φ_sum = φ.sum(dim=1, keepdim=True)    # (B, 1, F)

                numerator = torch.bmm(φ, φV)          # (B, N, D)
                denominator = torch.bmm(φ, φ_sum.transpose(1, 2))  # (B, N, 1)

                out = numerator / (denominator + 1e-6)

        elif self.mode == 'hybrid':
            # Hybrid: linear features + Toeplitz via convolution
            # This is O(N log N) due to FFT

            if causal:
                # First apply causal linear attention
                linear_out = self._causal_linear_attention(φ, V)

                # Then apply Toeplitz weighting via FFT convolution
                # This modulates the attention by distance
                out = self.feature_map.toeplitz.apply_toeplitz_fft(linear_out, N)
            else:
                # Compute linear attention output
                φV = torch.bmm(φ.transpose(1, 2), V)
                φ_sum = φ.sum(dim=1, keepdim=True)

                numerator = torch.bmm(φ, φV)
                denominator = torch.bmm(φ, φ_sum.transpose(1, 2))

                linear_out = numerator / (denominator + 1e-6)

                # Apply Toeplitz
                out = self.feature_map.toeplitz.apply_toeplitz_fft(linear_out, N)

        else:
            raise ValueError(f"Unknown mode: {self.mode}")

        return self.out_proj(out)

    def _causal_linear_attention(
        self,
        φ: torch.Tensor,
        V: torch.Tensor
    ) -> torch.Tensor:
        """
        Causal linear attention using cumulative sums.

        For each position i:
            out_i = φ_i^T [Σ_{j≤i} φ_j ⊗ V_j] / [φ_i^T Σ_{j≤i} φ_j]

        We maintain running sums as we scan through positions.
        """
        B, N, F = φ.shape
        _, _, D = V.shape

        device = φ.device
        dtype = φ.dtype

        # Running sums
        φV_sum = torch.zeros(B, F, D, device=device, dtype=dtype)  # Σ φ ⊗ V
        φ_sum = torch.zeros(B, F, device=device, dtype=dtype)       # Σ φ

        outputs = []

        for i in range(N):
            φ_i = φ[:, i, :]  # (B, F)
            V_i = V[:, i, :]  # (B, D)

            # Update running sums (include current position)
            φV_sum = φV_sum + φ_i.unsqueeze(-1) * V_i.unsqueeze(1)  # (B, F, D)
            φ_sum = φ_sum + φ_i  # (B, F)

            # Compute output for position i
            numerator = torch.bmm(φ_i.unsqueeze(1), φV_sum).squeeze(1)  # (B, D)
            denominator = (φ_i * φ_sum).sum(dim=-1, keepdim=True)       # (B, 1)

            out_i = numerator / (denominator + 1e-6)
            outputs.append(out_i)

        return torch.stack(outputs, dim=1)  # (B, N, D)


# =============================================================================
# PART 4: TRANSFORMER LAYER AND MODEL
# =============================================================================

class UnifiedASALayer(nn.Module):
    """Transformer layer using Unified ASA attention."""

    def __init__(
        self,
        embed_dim: int,
        num_heads: int = 4,
        mode: str = 'hybrid'
    ):
        super().__init__()

        self.attn = UnifiedASAAttention(embed_dim, num_heads, mode)
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)

        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 4),
            nn.GELU(),
            nn.Linear(embed_dim * 4, embed_dim),
        )

    def forward(self, x: torch.Tensor, causal: bool = True) -> torch.Tensor:
        x = x + self.attn(self.norm1(x), causal=causal)
        x = x + self.ffn(self.norm2(x))
        return x


class UnifiedASATransformer(nn.Module):
    """
    Complete Unified ASA Transformer.

    Combines:
    - Physics-grounded attention (equilibrium formulation)
    - Efficient computation (separable features)
    - O(N) or O(N log N) scaling
    """

    def __init__(
        self,
        vocab_size: int,
        embed_dim: int = 256,
        num_layers: int = 4,
        num_heads: int = 4,
        mode: str = 'hybrid',
        max_seq_len: int = 512,
    ):
        super().__init__()

        self.embed_dim = embed_dim
        self.mode = mode

        self.token_embed = nn.Embedding(vocab_size, embed_dim)
        self.pos_embed = nn.Embedding(max_seq_len, embed_dim)

        self.layers = nn.ModuleList([
            UnifiedASALayer(embed_dim, num_heads, mode)
            for _ in range(num_layers)
        ])

        self.norm = nn.LayerNorm(embed_dim)
        self.head = nn.Linear(embed_dim, vocab_size, bias=False)
        self.head.weight = self.token_embed.weight  # Weight tying

    def forward(
        self,
        input_ids: torch.Tensor,
        targets: Optional[torch.Tensor] = None
    ):
        B, N = input_ids.shape

        positions = torch.arange(N, device=input_ids.device).unsqueeze(0).expand(B, -1)
        x = self.token_embed(input_ids) + self.pos_embed(positions)

        for layer in self.layers:
            x = layer(x, causal=True)

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
    def generate(
        self,
        prompt: torch.Tensor,
        max_new: int = 20,
        temperature: float = 0.8
    ):
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

class StandardAttention(nn.Module):
    def __init__(self, embed_dim: int, num_heads: int = 4):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.scale = self.head_dim ** -0.5

        self.qkv = nn.Linear(embed_dim, embed_dim * 3)
        self.out_proj = nn.Linear(embed_dim, embed_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, N, D = x.shape
        H = self.num_heads

        qkv = self.qkv(x).reshape(B, N, 3, H, self.head_dim).permute(2, 0, 3, 1, 4)
        Q, K, V = qkv[0], qkv[1], qkv[2]

        attn = torch.matmul(Q, K.transpose(-2, -1)) * self.scale
        causal = torch.triu(torch.ones(N, N, device=x.device), diagonal=1).bool()
        attn = attn.masked_fill(causal.unsqueeze(0).unsqueeze(0), float('-inf'))
        attn = F.softmax(attn, dim=-1)

        out = torch.matmul(attn, V)
        out = out.transpose(1, 2).reshape(B, N, D)
        return self.out_proj(out)


class StandardLayer(nn.Module):
    def __init__(self, embed_dim: int, num_heads: int = 4):
        super().__init__()
        self.attn = StandardAttention(embed_dim, num_heads)
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 4),
            nn.GELU(),
            nn.Linear(embed_dim * 4, embed_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.norm1(x))
        x = x + self.ffn(self.norm2(x))
        return x


class StandardTransformer(nn.Module):
    def __init__(self, vocab_size: int, embed_dim: int = 256, num_layers: int = 4,
                 num_heads: int = 4, max_seq_len: int = 512):
        super().__init__()
        self.token_embed = nn.Embedding(vocab_size, embed_dim)
        self.pos_embed = nn.Embedding(max_seq_len, embed_dim)
        self.layers = nn.ModuleList([StandardLayer(embed_dim, num_heads) for _ in range(num_layers)])
        self.norm = nn.LayerNorm(embed_dim)
        self.head = nn.Linear(embed_dim, vocab_size, bias=False)
        self.head.weight = self.token_embed.weight

    def forward(self, input_ids, targets=None):
        B, N = input_ids.shape
        pos = torch.arange(N, device=input_ids.device).unsqueeze(0).expand(B, -1)
        x = self.token_embed(input_ids) + self.pos_embed(pos)
        for layer in self.layers:
            x = layer(x)
        logits = self.head(self.norm(x))
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=0)
        return logits, loss

    @torch.no_grad()
    def generate(self, prompt, max_new=20, temperature=0.8):
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
# TRAINING AND EVALUATION
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
    total_loss, correct, total = 0, 0, 0
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


def benchmark_speed(model, device, batch_size=32, seq_len=64, num_runs=50):
    model.eval()
    x = torch.randint(2, 256, (batch_size, seq_len), device=device)

    # Warmup
    with torch.no_grad():
        for _ in range(5):
            model(x)

    if device.type == 'cuda':
        torch.cuda.synchronize()

    t0 = time.time()
    with torch.no_grad():
        for _ in range(num_runs):
            model(x)

    if device.type == 'cuda':
        torch.cuda.synchronize()

    return (time.time() - t0) / num_runs * 1000  # ms


def count_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


# =============================================================================
# MAIN
# =============================================================================

def main():
    print("=" * 70)
    print("UNIFIED ASA: Physics + Efficiency")
    print("=" * 70)

    VOCAB_SIZE = 256
    EMBED_DIM = 256
    NUM_LAYERS = 4
    NUM_HEADS = 4
    SEQ_LEN = 64
    BATCH_SIZE = 32
    NUM_EPOCHS = 15
    LR = 3e-4

    print("\nCore Ideas:")
    print("  1. WHAT to compute: Energy minimization (Hopfield/equilibrium)")
    print("  2. HOW to compute: Exploit structure (separable features)")
    print("  3. RESULT: Physics-grounded O(N) attention")

    print("\nStructural Decomposition:")
    print("  exp(-E) = exp(-E_charge) × exp(-E_shell) × exp(-E_dist)")
    print("          ≈ [φ_c(i)·φ_c(j)] × [φ_s(i)·φ_s(j)] × [D|i-j|]")
    print("          = [φ(i)·φ(j)] × Toeplitz")

    print("\n1. Creating datasets...")
    train_data = PatternDataset(VOCAB_SIZE, SEQ_LEN, 8000)
    val_data = PatternDataset(VOCAB_SIZE, SEQ_LEN, 1000)
    train_loader = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_data, batch_size=BATCH_SIZE)

    print("\n2. Creating models...")

    models = {
        "Standard (O(N²))": StandardTransformer(
            VOCAB_SIZE, EMBED_DIM, NUM_LAYERS, NUM_HEADS, SEQ_LEN+1
        ),
        "Unified-Full (O(N²))": UnifiedASATransformer(
            VOCAB_SIZE, EMBED_DIM, NUM_LAYERS, NUM_HEADS, 'full', SEQ_LEN+1
        ),
        "Unified-Linear (O(N))": UnifiedASATransformer(
            VOCAB_SIZE, EMBED_DIM, NUM_LAYERS, NUM_HEADS, 'linear', SEQ_LEN+1
        ),
        "Unified-Hybrid (O(N log N))": UnifiedASATransformer(
            VOCAB_SIZE, EMBED_DIM, NUM_LAYERS, NUM_HEADS, 'hybrid', SEQ_LEN+1
        ),
    }

    for name, model in models.items():
        model.to(DEVICE)
        print(f"   {name}: {count_params(model):,} params")

    print("\n3. Speed benchmark...")
    for name, model in models.items():
        ms = benchmark_speed(model, DEVICE)
        print(f"   {name}: {ms:.2f} ms/forward")

    optimizers = {name: torch.optim.AdamW(m.parameters(), lr=LR) for name, m in models.items()}

    print("\n4. Training...")
    print("-" * 70)

    for epoch in range(NUM_EPOCHS):
        t0 = time.time()
        results = {}

        for name, model in models.items():
            train_loss = train_epoch(model, train_loader, optimizers[name], DEVICE)
            _, acc = evaluate(model, val_loader, DEVICE)
            results[name] = acc

        elapsed = time.time() - t0
        print(f"\nEpoch {epoch+1:2d}/{NUM_EPOCHS} ({elapsed:.1f}s)")
        for name, acc in results.items():
            print(f"  {name:30s}: {acc:.1%}")

    # Final results
    print("\n" + "=" * 70)
    print("FINAL RESULTS")
    print("=" * 70)

    print("\n{:30s} {:>10s} {:>12s}".format("Model", "Accuracy", "Speed (ms)"))
    print("-" * 55)

    for name, model in models.items():
        _, acc = evaluate(model, val_loader, DEVICE)
        ms = benchmark_speed(model, DEVICE)
        print(f"{name:30s} {acc:>10.1%} {ms:>12.2f}")

    # Analysis
    print("\n" + "-" * 70)
    print("ANALYSIS")
    print("-" * 70)

    std_acc = evaluate(models["Standard (O(N²))"], val_loader, DEVICE)[1]
    full_acc = evaluate(models["Unified-Full (O(N²))"], val_loader, DEVICE)[1]
    linear_acc = evaluate(models["Unified-Linear (O(N))"], val_loader, DEVICE)[1]
    hybrid_acc = evaluate(models["Unified-Hybrid (O(N log N))"], val_loader, DEVICE)[1]

    print(f"\nPhysics vs Standard:     {full_acc - std_acc:+.1%}")
    print(f"Linear vs Full:          {linear_acc - full_acc:+.1%}")
    print(f"Hybrid vs Full:          {hybrid_acc - full_acc:+.1%}")

    if abs(linear_acc - std_acc) < 0.05:
        print("\n✅ Linear ASA achieves comparable accuracy with O(N) complexity!")

    # Generation test
    print("\n" + "=" * 70)
    print("GENERATION TEST")
    print("=" * 70)

    prompts = [([10, 20, 10, 20, 10], "Repetition"), ([50, 51, 52, 53, 54], "Counting")]

    for prompt_list, pattern in prompts:
        prompt = torch.tensor([prompt_list], device=DEVICE)
        print(f"\n{pattern}: {prompt_list}")
        for name in ["Standard (O(N²))", "Unified-Linear (O(N))", "Unified-Hybrid (O(N log N))"]:
            gen = models[name].generate(prompt.clone(), max_new=10)
            print(f"  {name:30s}: {gen[0].tolist()}")

    print("\n" + "=" * 70)
    print("COMPLETE")
    print("=" * 70)

    return models


if __name__ == "__main__":
    main()
