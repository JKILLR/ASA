"""
TRUE ASA TRANSFORMER - IMPROVED
===============================

Philosophy: "Fixed physics, learned embeddings"
- Physics rules are CONSTANTS (not nn.Parameters)
- Only embeddings learn to represent tokens as atoms
- The universe has fixed laws; matter learns to exist within them

Fixes applied:
1. Shell compatibility: different shells attract (complementary)
2. No reconstructors: same atoms throughout, physics doesn't change
3. Valence constraint: limits bond formation
4. Fixed temperature: Boltzmann-like constant
5. Principled dimensions: based on physical meaning
6. Simplified embedding: single embedding, split by physics

Run: python true_asa_transformer.py
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import numpy as np
import time
from typing import Optional, Dict, NamedTuple
from dataclasses import dataclass

# Device
if torch.backends.mps.is_available():
    DEVICE = torch.device("mps")
elif torch.cuda.is_available():
    DEVICE = torch.device("cuda")
else:
    DEVICE = torch.device("cpu")
print(f"Using {DEVICE}")


# =============================================================================
# PHYSICS CONSTANTS - FIXED, NOT LEARNED
# =============================================================================

class PhysicsConstants:
    """
    The laws of semantic physics. These are FIXED.
    Only embeddings learn - physics rules don't change.
    """

    # Dimensions with physical meaning
    CHARGE_DIM = 1         # Scalar polarity (like real electric charge)
    SHELL_1_DIM = 16       # Inner shell (core features, few)
    SHELL_2_DIM = 32       # Middle shell (intermediate features)
    SHELL_3_DIM = 64       # Outer shell (valence features, most complex)
    NUCLEUS_DIM = 128      # Core identity (largest - defines the atom)
    MASS_DIM = 1           # Scalar semantic inertia
    VALENCE_DIM = 1        # Scalar bonding capacity

    # Energy coupling constants (like physical constants)
    CHARGE_COUPLING = 1.0      # Coulomb-like coupling
    SHELL_COUPLING = 0.5       # Shell interaction strength
    DISTANCE_COUPLING = 0.3    # Position-based decay
    MASS_COUPLING = 0.1        # Inertia effect

    # Thermodynamics
    TEMPERATURE = 1.0          # Boltzmann-like constant

    @classmethod
    def total_dim(cls) -> int:
        """Total embedding dimension."""
        raw = (cls.CHARGE_DIM + cls.SHELL_1_DIM + cls.SHELL_2_DIM +
               cls.SHELL_3_DIM + cls.NUCLEUS_DIM + cls.MASS_DIM + cls.VALENCE_DIM)
        # Pad to multiple of 4 for efficient computation
        return ((raw + 3) // 4) * 4

    @classmethod
    def raw_dim(cls) -> int:
        """Unpadded dimension."""
        return (cls.CHARGE_DIM + cls.SHELL_1_DIM + cls.SHELL_2_DIM +
                cls.SHELL_3_DIM + cls.NUCLEUS_DIM + cls.MASS_DIM + cls.VALENCE_DIM)


# =============================================================================
# ATOMIC STRUCTURE
# =============================================================================

class AtomicStructure(NamedTuple):
    """
    A semantic atom's structure.
    Each token is represented as an atom with these components.
    """
    charge: torch.Tensor      # (B, N, 1) - polarity [-1, 1]
    shell_1: torch.Tensor     # (B, N, 16) - inner shell (normalized)
    shell_2: torch.Tensor     # (B, N, 32) - middle shell (normalized)
    shell_3: torch.Tensor     # (B, N, 64) - outer shell (normalized)
    nucleus: torch.Tensor     # (B, N, 128) - core identity
    mass: torch.Tensor        # (B, N, 1) - semantic inertia
    valence: torch.Tensor     # (B, N, 1) - bonding capacity
    position: torch.Tensor    # (B, N) - sequence position

    def to_flat(self) -> torch.Tensor:
        """Flatten to single tensor for FFN processing."""
        return torch.cat([
            self.charge,
            self.shell_1,
            self.shell_2,
            self.shell_3,
            self.nucleus,
            self.mass,
            self.valence,
        ], dim=-1)


# =============================================================================
# FIXED PHYSICS ENGINE
# =============================================================================

class FixedPhysicsEngine:
    """
    NO nn.Module - NO learnable parameters.
    This is pure physics with fixed laws.
    """

    @staticmethod
    def charge_interaction(atoms: AtomicStructure) -> torch.Tensor:
        """
        Coulomb-like charge interaction.
        Opposite charges attract (negative energy).
        Same charges repel (positive energy).

        E = k * q_i * q_j
        """
        charge = atoms.charge.squeeze(-1)  # (B, N)

        # Pairwise product: (B, N, N)
        E = PhysicsConstants.CHARGE_COUPLING * torch.bmm(
            charge.unsqueeze(-1),
            charge.unsqueeze(-2)
        )

        return E

    @staticmethod
    def shell_compatibility(atoms: AtomicStructure) -> torch.Tensor:
        """
        Shell compatibility energy.
        DIFFERENT/COMPLEMENTARY shells attract (like puzzle pieces).
        SIMILAR shells have higher energy (less favorable).

        E = k * similarity(shell_i, shell_j)
        High similarity = high energy = repulsion
        Low similarity = low energy = attraction
        """
        outer = atoms.shell_3  # (B, N, D) - normalized

        # Similarity: high when shells are identical
        similarity = torch.bmm(outer, outer.transpose(-1, -2))  # (B, N, N)

        # Energy: similar shells repel (high E), different attract (low E)
        E = PhysicsConstants.SHELL_COUPLING * similarity

        return E

    @staticmethod
    def distance_decay(atoms: AtomicStructure) -> torch.Tensor:
        """
        Distance-based energy decay.
        Closer atoms have stronger interactions (lower energy barrier).

        E = k * |pos_i - pos_j|
        """
        pos = atoms.position  # (B, N)

        # Pairwise distance
        dist = (pos.unsqueeze(-1) - pos.unsqueeze(-2)).abs()  # (B, N, N)

        # Further apart = higher energy barrier
        E = PhysicsConstants.DISTANCE_COUPLING * dist

        return E

    @staticmethod
    def mass_inertia(atoms: AtomicStructure) -> torch.Tensor:
        """
        Mass-based interaction barrier.
        Heavier atoms are harder to bond (higher barrier).

        E = k * sqrt(m_i * m_j)
        """
        mass = atoms.mass.squeeze(-1)  # (B, N)

        # Geometric mean of masses
        mass_product = torch.bmm(
            mass.unsqueeze(-1),
            mass.unsqueeze(-2)
        ).sqrt()  # (B, N, N)

        E = PhysicsConstants.MASS_COUPLING * mass_product

        return E

    @staticmethod
    def total_energy(atoms: AtomicStructure) -> torch.Tensor:
        """
        Total bond energy between all atom pairs.
        Lower energy = more favorable bond.
        """
        E_charge = FixedPhysicsEngine.charge_interaction(atoms)
        E_shell = FixedPhysicsEngine.shell_compatibility(atoms)
        E_dist = FixedPhysicsEngine.distance_decay(atoms)
        E_mass = FixedPhysicsEngine.mass_inertia(atoms)

        return E_charge + E_shell + E_dist + E_mass

    @staticmethod
    def bond_strength(energy: torch.Tensor) -> torch.Tensor:
        """
        Convert energy to attention/bond strength.
        Lower energy = stronger bond = higher attention.

        Uses Boltzmann-like distribution with fixed temperature.
        """
        return F.softmax(-energy / PhysicsConstants.TEMPERATURE, dim=-1)

    @staticmethod
    def apply_valence_constraint(
        atoms: AtomicStructure,
        attn: torch.Tensor
    ) -> torch.Tensor:
        """
        Limit bonds based on valence capacity.
        Atoms with low valence can't form many strong bonds.
        """
        valence = atoms.valence.squeeze(-1)  # (B, N)

        # Current bond load (sum of attention given)
        bond_load = attn.sum(dim=-1, keepdim=True)  # (B, N, 1)

        # Capacity (valence determines max bonds)
        capacity = valence.unsqueeze(-1)  # (B, N, 1)

        # Scale down attention if over capacity
        scale = torch.clamp(capacity / (bond_load + 1e-6), max=1.0)

        return attn * scale


# =============================================================================
# ATOMIC EMBEDDING
# =============================================================================

class AtomicEmbedding(nn.Module):
    """
    Convert tokens to atomic structures.
    This is the ONLY learnable component for attention.
    """

    def __init__(self, vocab_size: int, max_seq_len: int = 512):
        super().__init__()

        C = PhysicsConstants
        self.total_dim = C.total_dim()
        self.raw_dim = C.raw_dim()

        # Single embedding that gets split into atomic components
        self.token_embed = nn.Embedding(vocab_size, self.total_dim)
        self.pos_embed = nn.Embedding(max_seq_len, self.total_dim)

        # Split sizes for each component
        self.splits = [
            C.CHARGE_DIM,
            C.SHELL_1_DIM,
            C.SHELL_2_DIM,
            C.SHELL_3_DIM,
            C.NUCLEUS_DIM,
            C.MASS_DIM,
            C.VALENCE_DIM,
        ]

        # Padding dimension (if any)
        self.pad_dim = self.total_dim - self.raw_dim

    def forward(self, token_ids: torch.Tensor) -> AtomicStructure:
        """Convert token IDs to atomic structure."""
        B, N = token_ids.shape

        # Create positions
        positions = torch.arange(N, device=token_ids.device)
        positions = positions.unsqueeze(0).expand(B, -1)

        # Get embeddings
        tok_emb = self.token_embed(token_ids)
        pos_emb = self.pos_embed(positions)

        # Combine (position has smaller influence)
        x = tok_emb + 0.1 * pos_emb

        # Remove padding if present
        if self.pad_dim > 0:
            x = x[..., :self.raw_dim]

        # Split into atomic components
        parts = x.split(self.splits, dim=-1)

        return AtomicStructure(
            charge=torch.tanh(parts[0]),                    # Bounded [-1, 1]
            shell_1=F.normalize(parts[1], dim=-1),          # Unit sphere
            shell_2=F.normalize(parts[2], dim=-1),          # Unit sphere
            shell_3=F.normalize(parts[3], dim=-1),          # Unit sphere
            nucleus=parts[4],                                # Unbounded identity
            mass=F.softplus(parts[5]) + 0.5,                # Positive, min 0.5
            valence=F.softplus(parts[6]) + 1.0,             # Positive, min 1.0
            position=positions.float(),
        )


# =============================================================================
# TRUE ASA ATTENTION
# =============================================================================

class TrueASAAttention(nn.Module):
    """
    Attention using fixed physics.

    The ONLY learnable parts are:
    - Value projection (what information to pass)
    - Output projection (how to combine)

    Energy computation uses FIXED physics rules.
    """

    def __init__(self, embed_dim: int, num_heads: int = 4):
        super().__init__()

        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads

        # Only V and output are learnable
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        self.out_proj = nn.Linear(embed_dim, embed_dim)

    def forward(
        self,
        atoms: AtomicStructure,
        x: torch.Tensor,
        causal: bool = True
    ) -> torch.Tensor:
        """
        Apply attention using fixed physics.

        Args:
            atoms: Atomic structure (from embedding)
            x: Flat representation for values
            causal: Whether to apply causal mask
        """
        B, N, D = x.shape
        H = self.num_heads

        # Compute energy using FIXED physics
        energy = FixedPhysicsEngine.total_energy(atoms)  # (B, N, N)

        # Convert to attention weights
        attn = FixedPhysicsEngine.bond_strength(energy)  # (B, N, N)

        # Apply valence constraint
        attn = FixedPhysicsEngine.apply_valence_constraint(atoms, attn)

        # Apply causal mask
        if causal:
            causal_mask = torch.triu(
                torch.ones(N, N, device=x.device),
                diagonal=1
            ).bool()
            # Re-normalize after masking
            attn = attn.masked_fill(causal_mask.unsqueeze(0), 0.0)
            attn = attn / (attn.sum(dim=-1, keepdim=True) + 1e-8)

        # Value projection (learnable)
        V = self.v_proj(x)
        V = V.view(B, N, H, self.head_dim).permute(0, 2, 1, 3)

        # Apply attention (expand for heads)
        attn_expanded = attn.unsqueeze(1).expand(-1, H, -1, -1)
        out = torch.matmul(attn_expanded, V)

        # Reshape and project output
        out = out.permute(0, 2, 1, 3).reshape(B, N, D)

        return self.out_proj(out)


# =============================================================================
# TRUE ASA BLOCK
# =============================================================================

class TrueASABlock(nn.Module):
    """
    Transformer block with true ASA attention.
    No reconstructors - atoms stay fixed throughout.
    """

    def __init__(self, embed_dim: int, num_heads: int = 4):
        super().__init__()

        self.attn = TrueASAAttention(embed_dim, num_heads)
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)

        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 4),
            nn.GELU(),
            nn.Linear(embed_dim * 4, embed_dim),
        )

    def forward(
        self,
        atoms: AtomicStructure,
        x: torch.Tensor,
        causal: bool = True
    ) -> torch.Tensor:
        """Forward with fixed atoms."""
        x = x + self.attn(atoms, self.norm1(x), causal=causal)
        x = x + self.ffn(self.norm2(x))
        return x


# =============================================================================
# TRUE ASA TRANSFORMER
# =============================================================================

class TrueASATransformer(nn.Module):
    """
    Complete transformer using fixed semantic physics.

    Architecture:
    1. Embedding converts tokens to atoms (learnable)
    2. Each layer applies fixed physics for attention
    3. FFN layers learn transformations (learnable)
    4. Same atoms used throughout - physics doesn't change
    """

    def __init__(
        self,
        vocab_size: int,
        num_layers: int = 6,
        num_heads: int = 4,
        max_seq_len: int = 512,
    ):
        super().__init__()

        self.embed_dim = PhysicsConstants.total_dim()

        # Atomic embedding (learnable)
        self.embedding = AtomicEmbedding(vocab_size, max_seq_len)

        # Transformer layers
        self.layers = nn.ModuleList([
            TrueASABlock(self.embed_dim, num_heads)
            for _ in range(num_layers)
        ])

        self.norm = nn.LayerNorm(self.embed_dim)
        self.head = nn.Linear(self.embed_dim, vocab_size, bias=False)

        # Weight tying
        self.head.weight = self.embedding.token_embed.weight

    def forward(
        self,
        input_ids: torch.Tensor,
        targets: Optional[torch.Tensor] = None
    ):
        B, N = input_ids.shape

        # Convert to atoms ONCE
        atoms = self.embedding(input_ids)

        # Get flat representation for processing
        x = atoms.to_flat()

        # Pad if needed
        if x.size(-1) < self.embed_dim:
            pad = torch.zeros(B, N, self.embed_dim - x.size(-1), device=x.device)
            x = torch.cat([x, pad], dim=-1)

        # Apply layers with SAME atoms
        for layer in self.layers:
            x = layer(atoms, x, causal=True)

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
# STANDARD TRANSFORMER (for comparison)
# =============================================================================

class StandardBlock(nn.Module):
    """Standard transformer block for comparison."""

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
        causal_mask = torch.triu(torch.ones(N, N, device=x.device), diagonal=1).bool()
        attn = attn.masked_fill(causal_mask.unsqueeze(0).unsqueeze(0), float('-inf'))
        attn = F.softmax(attn, dim=-1)

        out = torch.matmul(attn, V)
        out = out.permute(0, 2, 1, 3).reshape(B, N, D)

        x = x + self.out_proj(out)
        x = x + self.ffn(self.norm2(x))
        return x


class StandardTransformer(nn.Module):
    """Standard transformer for fair comparison."""

    def __init__(
        self,
        vocab_size: int,
        embed_dim: int = 256,
        num_layers: int = 6,
        num_heads: int = 4,
        max_seq_len: int = 512,
    ):
        super().__init__()

        self.embed_dim = embed_dim
        self.token_embed = nn.Embedding(vocab_size, embed_dim)
        self.pos_embed = nn.Embedding(max_seq_len, embed_dim)

        self.layers = nn.ModuleList([
            StandardBlock(embed_dim, num_heads)
            for _ in range(num_layers)
        ])

        self.norm = nn.LayerNorm(embed_dim)
        self.head = nn.Linear(embed_dim, vocab_size, bias=False)
        self.head.weight = self.token_embed.weight

    def forward(self, x: torch.Tensor, targets: Optional[torch.Tensor] = None):
        B, N = x.shape
        positions = torch.arange(N, device=x.device).unsqueeze(0).expand(B, -1)
        h = self.token_embed(x) + self.pos_embed(positions)

        for layer in self.layers:
            h = layer(h)

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
    """Pattern dataset for testing."""

    def __init__(self, vocab_size: int = 256, seq_len: int = 64, num_samples: int = 10000):
        self.vocab_size = vocab_size
        self.seq_len = seq_len
        self.data = []

        for _ in range(num_samples):
            pattern = np.random.randint(4)

            if pattern == 0:  # Repetition
                a, b = np.random.randint(2, vocab_size, size=2)
                seq = [a, b] * (seq_len // 2)
            elif pattern == 1:  # Counting
                start = np.random.randint(2, vocab_size - seq_len)
                seq = list(range(start, start + seq_len))
            elif pattern == 2:  # Copy
                length = seq_len // 3
                segment = np.random.randint(2, vocab_size, size=length).tolist()
                seq = segment + [1] + segment + [0] * (seq_len - 2*length - 1)
            else:  # Reversal
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


# =============================================================================
# MAIN
# =============================================================================

def main():
    print("=" * 60)
    print("TRUE ASA TRANSFORMER")
    print("Fixed Physics, Learned Embeddings")
    print("=" * 60)

    # Config
    VOCAB_SIZE = 256
    NUM_LAYERS = 6
    NUM_HEADS = 4
    SEQ_LEN = 64
    BATCH_SIZE = 32
    NUM_EPOCHS = 20
    LR = 3e-4

    # Show physics constants
    print("\nPhysics Constants (FIXED):")
    print(f"  Charge coupling:   {PhysicsConstants.CHARGE_COUPLING}")
    print(f"  Shell coupling:    {PhysicsConstants.SHELL_COUPLING}")
    print(f"  Distance coupling: {PhysicsConstants.DISTANCE_COUPLING}")
    print(f"  Mass coupling:     {PhysicsConstants.MASS_COUPLING}")
    print(f"  Temperature:       {PhysicsConstants.TEMPERATURE}")
    print(f"  Embed dimension:   {PhysicsConstants.total_dim()}")

    # Data
    print("\n1. Creating datasets...")
    train_data = PatternDataset(VOCAB_SIZE, SEQ_LEN, 10000)
    val_data = PatternDataset(VOCAB_SIZE, SEQ_LEN, 1000)
    train_loader = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_data, batch_size=BATCH_SIZE)

    # Models
    print("\n2. Creating models...")

    true_asa = TrueASATransformer(
        VOCAB_SIZE, NUM_LAYERS, NUM_HEADS, max_seq_len=SEQ_LEN+1
    ).to(DEVICE)

    # Standard transformer with same embed_dim for fair comparison
    standard = StandardTransformer(
        VOCAB_SIZE,
        embed_dim=PhysicsConstants.total_dim(),
        num_layers=NUM_LAYERS,
        num_heads=NUM_HEADS,
        max_seq_len=SEQ_LEN+1
    ).to(DEVICE)

    print(f"   True ASA params:  {count_params(true_asa):,}")
    print(f"   Standard params:  {count_params(standard):,}")

    print("\n   True ASA Philosophy:")
    print("   - Embeddings learn to represent tokens as atoms")
    print("   - Physics rules are FIXED (not learned)")
    print("   - Same atoms used throughout all layers")
    print("   - Only V projection and FFN are learnable")

    # Optimizers
    asa_opt = torch.optim.AdamW(true_asa.parameters(), lr=LR)
    std_opt = torch.optim.AdamW(standard.parameters(), lr=LR)

    # Training
    print("\n3. Training...")
    print("-" * 60)

    for epoch in range(NUM_EPOCHS):
        t0 = time.time()

        asa_loss = train_epoch(true_asa, train_loader, asa_opt, DEVICE)
        std_loss = train_epoch(standard, train_loader, std_opt, DEVICE)

        asa_val_loss, asa_acc = evaluate(true_asa, val_loader, DEVICE)
        std_val_loss, std_acc = evaluate(standard, val_loader, DEVICE)

        elapsed = time.time() - t0

        print(f"\nEpoch {epoch+1:2d}/{NUM_EPOCHS} ({elapsed:.1f}s)")
        print(f"  True-ASA: train={asa_loss:.4f}, val={asa_val_loss:.4f}, acc={asa_acc:.1%}")
        print(f"  Standard: train={std_loss:.4f}, val={std_val_loss:.4f}, acc={std_acc:.1%}")

    # Final results
    print("\n" + "=" * 60)
    print("FINAL RESULTS")
    print("=" * 60)

    asa_val_loss, asa_acc = evaluate(true_asa, val_loader, DEVICE)
    std_val_loss, std_acc = evaluate(standard, val_loader, DEVICE)

    print(f"\nTrue ASA Transformer (Fixed Physics):")
    print(f"  Val loss: {asa_val_loss:.4f}")
    print(f"  Accuracy: {asa_acc:.1%}")

    print(f"\nStandard Transformer:")
    print(f"  Val loss: {std_val_loss:.4f}")
    print(f"  Accuracy: {std_acc:.1%}")

    print(f"\nDifference: {asa_acc - std_acc:+.1%}")

    # Generation test
    print("\n" + "=" * 60)
    print("GENERATION TEST")
    print("=" * 60)

    prompts = [
        ([10, 20, 10, 20, 10], "Repetition"),
        ([50, 51, 52, 53, 54], "Counting"),
    ]

    for prompt_list, name in prompts:
        prompt = torch.tensor([prompt_list], device=DEVICE)
        print(f"\n{name}: {prompt_list}")

        asa_gen = true_asa.generate(prompt.clone(), max_new=10)
        std_gen = standard.generate(prompt.clone(), max_new=10)

        print(f"  True-ASA: {asa_gen[0].tolist()}")
        print(f"  Standard: {std_gen[0].tolist()}")

    print("\n" + "=" * 60)
    print("DONE")
    print("=" * 60)

    return true_asa, standard


if __name__ == "__main__":
    main()
