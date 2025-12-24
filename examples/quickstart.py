#!/usr/bin/env python3
"""
Quick Start Example for Atomic Semantic Architecture (ASA).

This example demonstrates the basic usage of ASA:
1. Creating and configuring a model
2. Encoding tokens into atomic representations
3. Predicting bonds between atoms
4. Composing atoms into molecules
5. Working with charge and temperature
"""

import torch

# Import ASA components
from asa import (
    AtomConfig,
    AtomicSemanticModel,
    ChargeState,
)


def main():
    print("=" * 60)
    print("Atomic Semantic Architecture (ASA) - Quick Start")
    print("=" * 60)

    # 1. Create configuration
    print("\n1. Creating configuration...")
    config = AtomConfig(
        vocab_size=30000,
        nuclear_dim=64,
        shell_dims=(128, 256, 512),
        shell_capacities=(4, 12, 32, 64),
        valence_max=4,
    )
    print(f"   Total embedding dimension: {config.total_dim}")
    print(f"   Number of shells: {config.num_shells}")

    # 2. Initialize model
    print("\n2. Initializing model...")
    model = AtomicSemanticModel(config)
    print(f"   Parameters: {sum(p.numel() for p in model.parameters()):,}")

    # 3. Encode tokens
    print("\n3. Encoding tokens...")
    token_ids = torch.tensor([[1, 42, 100, 256, 500]])
    output = model(token_ids)

    print(f"   Composed vector shape: {output['composed'].shape}")
    print(f"   Net charge: {output['net_charges'].item():.3f}")

    # 4. Access atomic data
    print("\n4. Atomic data structure:")
    atom_data = output["atom_data"]
    print(f"   Nuclear shape: {atom_data['nuclear'].shape}")
    print(f"   Polarity range: [{atom_data['polarity'].min():.2f}, {atom_data['polarity'].max():.2f}]")
    print(f"   Valence counts: {atom_data['valence_count'].tolist()}")

    # 5. Work with thermodynamics
    print("\n5. Thermodynamic parameters:")
    thermo = model.thermo
    print(f"   Threshold base: {thermo.threshold_base.item():.3f}")
    print(f"   Ionization multiplier: {thermo.ionization_multiplier.item():.3f}")
    print(f"   Promotion thresholds: {thermo.promotion_thresholds.tolist()}")

    # 6. Temperature effects
    print("\n6. Temperature effects:")
    for temp in [0.1, 0.5, 0.9]:
        model.set_temperature(temp)
        threshold = model.context.bond_threshold
        volatility = model.context.charge_volatility
        print(f"   T={temp}: threshold={threshold:.3f}, volatility={volatility:.3f}")

    # 7. Charge state manipulation
    print("\n7. Charge state examples:")
    charge = ChargeState(polarity=0.8, magnitude=0.9, stability=0.7)
    print(f"   Original: {charge}")
    print(f"   Effective: {charge.effective:.3f}")

    flipped = charge.flip()
    print(f"   Flipped: {flipped}")
    print(f"   Flipped effective: {flipped.effective:.3f}")

    # 8. Modifier application
    print("\n8. Linguistic modifiers:")
    base_charge = ChargeState.from_modifiers(1.0, [])
    print(f"   Base 'good': {base_charge.effective:.3f}")

    negated = ChargeState.from_modifiers(1.0, ["not"])
    print(f"   'not good': {negated.effective:.3f}")

    double_neg = ChargeState.from_modifiers(1.0, ["not", "not"])
    print(f"   'not not good': {double_neg.effective:.3f}")

    intensified = ChargeState.from_modifiers(1.0, ["very", "very"])
    print(f"   'very very good': {intensified.effective:.3f}")

    hedged = ChargeState.from_modifiers(1.0, ["maybe"])
    print(f"   'maybe good': {hedged.effective:.3f}")

    # 9. Bond prediction
    print("\n9. Bond prediction:")
    # Create two atoms
    tokens_pair = torch.tensor([[42, 100]])
    atom_output = model.encode_tokens(tokens_pair)

    atom_a = {k: v[:, 0] if isinstance(v, torch.Tensor) and v.dim() > 1
              else ([s[:, 0] for s in v] if isinstance(v, list) else v)
              for k, v in atom_output.items()}
    atom_b = {k: v[:, 1] if isinstance(v, torch.Tensor) and v.dim() > 1
              else ([s[:, 1] for s in v] if isinstance(v, list) else v)
              for k, v in atom_output.items()}

    bond_pred = model.predict_bond(atom_a, atom_b)
    print(f"   Strength: {bond_pred['strength'].item():.3f}")
    print(f"   Can bond: {bond_pred['can_bond'].item()}")
    print(f"   Type probs: {bond_pred['type_probs'].tolist()}")

    # 10. Catalyst detection
    print("\n10. Catalyst detection:")
    test_tokens = torch.tensor([[1]])  # Some token
    catalyst_effects = model.get_catalyst_effects(test_tokens)
    print(f"   Is catalyst: {catalyst_effects['is_catalyst'].item():.3f}")
    print(f"   Threshold multipliers: {catalyst_effects['threshold_multipliers'].tolist()}")

    print("\n" + "=" * 60)
    print("Quick start complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
