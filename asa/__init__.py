"""
Atomic Semantic Architecture (ASA) v1.1

A novel semantic embedding system that models meaning using the structure of matter.

Key concepts:
- Atoms: Basic semantic units with nuclear identity and shell relationships
- Charge: Polarity (+/-) represents truth/existence state
- Shells: Layered relational features at different abstraction levels
- Bonds: Typed connections between atoms with strength
- Molecules: Composite semantic structures
- Temperature: Cognitive mode affecting bond formation
- Ionization: Belief resistance based on shell depth

Usage:
    from asa import AtomConfig, AtomicSemanticModel

    config = AtomConfig(vocab_size=30000)
    model = AtomicSemanticModel(config)

    token_ids = torch.tensor([[1, 2, 3, 4, 5]])
    output = model(token_ids)

    print(f"Net charge: {output['net_charges']}")
"""

from .core.config import AtomConfig
from .core.charge import ChargeState
from .core.shells import Shell, ShellAssociation
from .core.atoms import SemanticAtom, SemanticBond, SemanticMolecule, ConceptPhase
from .thermodynamics.learnable import LearnableThermodynamics
from .thermodynamics.context import LearnableSemanticContext, TemperatureSchedule
from .shells.manager import BoundedShellManager, MigrationConfig
from .bonding.engine import BondingEngine, BondType
from .bonding.propagation import propagate_charge
from .bonding.composition import CompositionStrategy, compose_molecule
from .neural.encoder import AtomicEncoder
from .neural.bonder import BondingNetwork
from .neural.composer import CompositionNetwork
from .neural.model import AtomicSemanticModel
from .storage.periodic_table import PeriodicTable
from .storage.vector_store import AtomicVectorStore
from .storage.molecule_cache import MoleculeCache
from .training.losses import ASELossV1_1, ThermodynamicsLoss
from .training.trainer import ASETrainerV1_1

__version__ = "1.1.0"
__all__ = [
    # Config
    "AtomConfig",
    # Core
    "ChargeState",
    "Shell",
    "ShellAssociation",
    "SemanticAtom",
    "SemanticBond",
    "SemanticMolecule",
    "ConceptPhase",
    # Thermodynamics
    "LearnableThermodynamics",
    "LearnableSemanticContext",
    "TemperatureSchedule",
    # Shells
    "BoundedShellManager",
    "MigrationConfig",
    # Bonding
    "BondingEngine",
    "BondType",
    "propagate_charge",
    "CompositionStrategy",
    "compose_molecule",
    # Neural
    "AtomicEncoder",
    "BondingNetwork",
    "CompositionNetwork",
    "AtomicSemanticModel",
    # Storage
    "PeriodicTable",
    "AtomicVectorStore",
    "MoleculeCache",
    # Training
    "ASELossV1_1",
    "ThermodynamicsLoss",
    "ASETrainerV1_1",
]
