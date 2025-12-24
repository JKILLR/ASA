# Atomic Semantic Architecture (ASA) v1.1

A novel semantic embedding system that models meaning using the structure of matter.

## Overview

ASA represents semantic concepts as "atoms" with:
- **Nuclear vector**: Core identity embedding (like proton count)
- **Shells**: Layered relational features at different abstraction levels
- **Charge**: Truth/existence state with polarity and magnitude
- **Valence**: Bonding interface for semantic composition

## Key Features

### What's Novel

| Component | What It Is | Why It's New |
|-----------|------------|--------------|
| Hierarchical shells | Separate vector spaces per abstraction level | Query "identity" vs "context" separately |
| Explicit charge | Learned polarity + magnitude | Compositional negation logic |
| Valence slots | Typed bonding interfaces | Semantic role expectations per slot |
| Charge propagation | Algebraic truth-state flow | NOT flips charge, double negatives cancel |
| Temperature | Learnable bonding threshold modifier | Single dial for analytical ↔ creative |
| Ionization energy | Learnable shell-weighted resistance | Core beliefs protected, periphery flexible |
| Belief robustness | Distinguishes "just read" from "verified truth" | Solves gullible RAG problem |

### v1.1 Improvements

- **Learnable thermodynamics**: All constants are trainable parameters
- **Bounded cascades**: O(1) insertion with configurable overflow
- **Neural catalyst detection**: Generalizes beyond hard-coded word lists
- **Belief robustness testing**: Validation suite for ionization energy

## Installation

```bash
pip install -e .
```

## Quick Start

```python
import torch
from asa import AtomConfig, AtomicSemanticModel

# Create configuration
config = AtomConfig(vocab_size=30000)

# Initialize model
model = AtomicSemanticModel(config)

# Encode tokens
token_ids = torch.tensor([[1, 2, 3, 4, 5]])
output = model(token_ids)

print(f"Net charge: {output['net_charges']}")
print(f"Composed vector shape: {output['composed'].shape}")
print(f"Threshold base: {model.thermo.threshold_base.item():.3f}")
```

## Architecture

```
Token IDs
    │
    ▼
┌──────────────────┐
│  AtomicEncoder   │ → Nuclear, Shells, Charge, Valence
└────────┬─────────┘
         │
    ▼    ▼    ▼
┌────┐ ┌────┐ ┌────┐
│Atom│─│Bond│─│Atom│
└────┘ └────┘ └────┘
         │
         ▼
┌──────────────────┐
│ CompositionNet   │ → Molecular vector + Net charge
└──────────────────┘
```

## Core Components

### Data Structures

- `AtomConfig`: Master configuration
- `ChargeState`: Truth/existence state
- `Shell`, `ShellAssociation`: Layered relationships
- `SemanticAtom`, `SemanticBond`, `SemanticMolecule`: Main entities

### Thermodynamics

- `LearnableThermodynamics`: Trainable physical constants
- `LearnableSemanticContext`: Runtime temperature and derived values
- `TemperatureSchedule`: Annealing strategies

### Shell Management

- `BoundedShellManager`: O(1) insertion with hysteresis
- `PhaseAnalyzer`: Gas/Liquid/Solid phase detection
- `ResonanceState`: Ambiguity handling

### Bonding

- `BondingEngine`: Full thermodynamic bond formation
- `LearnedCatalystDetector`: Neural catalyst effects
- `StabilityAnalyzer`: Contradiction detection

### Neural Networks

- `AtomicEncoder`: Token → Atom
- `BondingNetwork`: Atom pair → Bond prediction
- `CompositionNetwork`: Atoms → Molecule
- `AtomicSemanticModel`: Complete model

## Training

```python
from asa import AtomicSemanticModel, ASETrainerV1_1
from torch.utils.data import DataLoader

model = AtomicSemanticModel(AtomConfig())
trainer = ASETrainerV1_1(model, lr=1e-4)

for epoch in range(10):
    metrics = trainer.train_epoch(train_loader)
    print(f"Epoch {epoch}: loss={metrics['total_loss']:.4f}")
```

## Belief Robustness Testing

```python
from asa.testing import BeliefRobustnessTest

test = BeliefRobustnessTest()
results = test.run_all()
test.print_report(results)
```

## Key Equations

### Charge Algebra
```
effective_charge = polarity × magnitude × stability
NOT(charge) = -charge × 0.9
double_negative: NOT(NOT(x)) = x × 0.81
```

### Temperature Effects
```
bond_threshold = threshold_base - (threshold_scale × temperature)
charge_volatility = volatility_base + (volatility_scale × temperature)
```

### Ionization Energy
```
ionization_energy(level) = base × (multiplier ^ (max_shells - 1 - level))
```

## License

MIT License - see LICENSE file for details.

## Citation

If you use ASA in your research, please cite:

```bibtex
@software{asa2024,
  title = {Atomic Semantic Architecture},
  version = {1.1.0},
  year = {2024},
}
```
