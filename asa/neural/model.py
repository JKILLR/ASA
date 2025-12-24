"""
AtomicSemanticModel - Complete Atomic Semantic Embedding model v1.1.

Combines:
- AtomicEncoder for token → atom conversion
- BondingNetwork for bond prediction
- CompositionNetwork for molecular composition
- LearnableThermodynamics for dynamic behavior
- LearnedCatalystDetector for catalyst effects
"""

from typing import Dict, List

import torch
import torch.nn as nn

from ..core.config import AtomConfig
from ..thermodynamics.learnable import LearnableThermodynamics
from ..thermodynamics.context import LearnableSemanticContext
from ..bonding.catalyst import LearnedCatalystDetector
from ..bonding.engine import BondType
from .encoder import AtomicEncoder
from .bonder import BondingNetwork
from .composer import CompositionNetwork


class AtomicSemanticModel(nn.Module):
    """
    Complete Atomic Semantic Embedding model v1.1.

    This is the main model class that integrates all ASA components.

    Features:
    - Encodes tokens into structured atomic representations
    - Predicts bonds between atoms
    - Composes atoms into molecular vectors
    - Uses learnable thermodynamics for dynamic behavior
    - Detects and applies catalyst effects
    """

    def __init__(self, config: AtomConfig):
        """
        Initialize the complete ASA model.

        Args:
            config: Atom configuration
        """
        super().__init__()
        self.config = config

        # Core components
        self.encoder = AtomicEncoder(config)
        self.bonder = BondingNetwork(config)
        self.composer = CompositionNetwork(config)

        # v1.1: Learnable thermodynamics
        self.thermo = LearnableThermodynamics()
        self.context = LearnableSemanticContext(self.thermo)

        # v1.1: Learned catalyst detection
        self.catalyst_detector = LearnedCatalystDetector(
            atom_dim=config.total_dim,
            num_bond_types=len(BondType),
        )

    def encode_tokens(self, token_ids: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Encode tokens into atomic representations.

        Args:
            token_ids: Token IDs (batch_size, seq_len) or (seq_len,)

        Returns:
            Dict with all atomic components
        """
        return self.encoder(token_ids)

    def predict_bond(
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
            Bond prediction dict
        """
        return self.bonder(atom_a, atom_b)

    def compose(
        self,
        atoms: List[Dict[str, torch.Tensor]],
        bonds: List[Dict] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Compose atoms into molecule.

        Args:
            atoms: List of atom representations
            bonds: Optional list of bonds

        Returns:
            Composition result dict
        """
        return self.composer(atoms, bonds)

    def forward(self, token_ids: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Full forward pass: encode, compose, analyze.

        Args:
            token_ids: Token IDs (batch_size, seq_len)

        Returns:
            Dict with atom data, composed vectors, charges, and thermo params
        """
        # Handle 1D input
        if token_ids.dim() == 1:
            token_ids = token_ids.unsqueeze(0)

        batch_size, seq_len = token_ids.shape

        # Encode all tokens
        atom_data = self.encoder(token_ids)

        # Compose each sequence
        composed_vectors = []
        net_charges = []

        for b in range(batch_size):
            # Extract atoms for this batch element
            atoms_b = []
            for s in range(seq_len):
                atom = {}
                for k, v in atom_data.items():
                    if isinstance(v, torch.Tensor) and v.dim() >= 2:
                        atom[k] = v[b, s]
                    elif isinstance(v, list):
                        # Handle shell list
                        atom[k] = [shell[b, s] for shell in v]
                atoms_b.append(atom)

            result = self.composer(atoms_b, [])
            composed_vectors.append(result["composed_vector"])
            net_charges.append(result["net_charge"])

        return {
            "atom_data": atom_data,
            "composed": torch.stack(composed_vectors),
            "net_charges": torch.tensor(net_charges),
            "thermo_params": {
                "threshold_base": self.thermo.threshold_base,
                "threshold_scale": self.thermo.threshold_scale,
                "ionization_base": self.thermo.ionization_base,
                "ionization_mult": self.thermo.ionization_multiplier,
            },
        }

    def get_thermodynamic_state(self) -> dict:
        """Get current thermodynamic parameter values."""
        return self.thermo.get_state_dict_readable()

    def set_temperature(self, temperature: float) -> None:
        """Set the processing temperature."""
        self.context.temperature = temperature

    def get_catalyst_effects(
        self,
        token_id: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """
        Get catalyst effects for a token.

        Args:
            token_id: Token ID

        Returns:
            Catalyst effect dict
        """
        atom_data = self.encoder(token_id)
        return self.catalyst_detector(atom_data["flat"])


class MinimalASE(nn.Module):
    """
    Simplest version capturing core value. ~50 lines.

    Good for quick experiments or as a baseline.
    """

    def __init__(self, vocab_size: int = 30000):
        """
        Initialize minimal ASE.

        Args:
            vocab_size: Vocabulary size
        """
        super().__init__()

        self.embedding = nn.Embedding(vocab_size, 256)
        self.nuclear = nn.Linear(256, 64)
        self.shell = nn.Linear(256, 128)
        self.charge = nn.Linear(256, 2)

    def forward(self, token_ids: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Encode tokens.

        Args:
            token_ids: Token IDs

        Returns:
            Minimal atomic representation
        """
        x = self.embedding(token_ids)
        nuclear = self.nuclear(x)
        shell = self.shell(x)
        charge_out = self.charge(x)

        return {
            "nuclear": nuclear,
            "shell": shell,
            "polarity": torch.tanh(charge_out[..., 0]),
            "magnitude": torch.sigmoid(charge_out[..., 1]),
            "flat": torch.cat([nuclear, shell], dim=-1),
        }


class ASEForClassification(nn.Module):
    """
    ASE model with classification head.

    Good for sentiment analysis, text classification, etc.
    """

    def __init__(
        self,
        config: AtomConfig,
        num_classes: int,
        use_charge: bool = True,
    ):
        """
        Initialize classification model.

        Args:
            config: Atom configuration
            num_classes: Number of output classes
            use_charge: Whether to use charge features
        """
        super().__init__()
        self.config = config
        self.use_charge = use_charge

        self.ase = AtomicSemanticModel(config)

        # Classification head
        input_dim = config.total_dim
        if use_charge:
            input_dim += 1  # Add net charge

        self.classifier = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(128, num_classes),
        )

    def forward(self, token_ids: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Forward pass with classification.

        Args:
            token_ids: Token IDs

        Returns:
            Dict with logits and ASE outputs
        """
        ase_output = self.ase(token_ids)

        features = ase_output["composed"]
        if self.use_charge:
            charge = ase_output["net_charges"].unsqueeze(-1)
            features = torch.cat([features, charge], dim=-1)

        logits = self.classifier(features)

        return {
            "logits": logits,
            "probs": torch.softmax(logits, dim=-1),
            "composed": ase_output["composed"],
            "net_charges": ase_output["net_charges"],
        }


class ASEForSequenceLabeling(nn.Module):
    """
    ASE model for sequence labeling tasks.

    Good for NER, POS tagging, etc.
    """

    def __init__(
        self,
        config: AtomConfig,
        num_labels: int,
    ):
        """
        Initialize sequence labeling model.

        Args:
            config: Atom configuration
            num_labels: Number of labels
        """
        super().__init__()
        self.config = config

        self.ase = AtomicSemanticModel(config)

        # Token-level classifier
        self.classifier = nn.Sequential(
            nn.Linear(config.total_dim, 128),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(128, num_labels),
        )

    def forward(self, token_ids: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Forward pass with token-level classification.

        Args:
            token_ids: Token IDs (batch_size, seq_len)

        Returns:
            Dict with token logits
        """
        atom_data = self.ase.encode_tokens(token_ids)

        # Use flat vectors for each token
        token_features = atom_data["flat"]
        logits = self.classifier(token_features)

        return {
            "logits": logits,  # (batch, seq_len, num_labels)
            "atom_data": atom_data,
        }
