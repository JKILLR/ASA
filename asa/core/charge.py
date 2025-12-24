"""
ChargeState - Represents the truth/existence state of a concept.

The charge system models semantic polarity:
- Positive charge: Affirmation, truth, existence
- Negative charge: Negation, falsity, absence
- Magnitude: Conviction strength
- Stability: Resistance to change
"""

from dataclasses import dataclass
from typing import List


@dataclass
class ChargeState:
    """
    Represents the truth/existence state of a concept.

    Polarity:
        +1.0 = Strong affirmation (IS, EXISTS, TRUE)
        +0.5 = Weak affirmation (probably, seems)
         0.0 = Neutral/uncommitted
        -0.5 = Weak negation (unlikely, doubtful)
        -1.0 = Strong negation (IS NOT, ABSENT, FALSE)

    Magnitude:
        1.0 = Absolute certainty
        0.5 = Moderate confidence
        0.0 = Complete uncertainty

    Stability:
        1.0 = Fixed charge (lexically determined)
        0.5 = Context-dependent
        0.0 = Highly volatile
    """

    polarity: float = 0.0
    magnitude: float = 0.5
    stability: float = 0.5

    def __post_init__(self):
        """Clamp values to valid ranges."""
        self.polarity = max(-1.0, min(1.0, self.polarity))
        self.magnitude = max(0.0, min(1.0, self.magnitude))
        self.stability = max(0.0, min(1.0, self.stability))

    @property
    def effective(self) -> float:
        """Net charge considering all factors."""
        return self.polarity * self.magnitude * self.stability

    def flip(self) -> "ChargeState":
        """Return negated charge (for NOT operations)."""
        return ChargeState(
            polarity=-self.polarity,
            magnitude=self.magnitude,
            stability=self.stability * 0.9,  # Negation slightly destabilizes
        )

    def amplify(self, factor: float = 1.2) -> "ChargeState":
        """Return amplified charge (for intensifiers)."""
        return ChargeState(
            polarity=self.polarity,
            magnitude=min(1.0, self.magnitude * factor),
            stability=min(1.0, self.stability * 1.1),
        )

    def attenuate(self, factor: float = 0.5) -> "ChargeState":
        """Return attenuated charge (for hedges)."""
        return ChargeState(
            polarity=self.polarity,
            magnitude=self.magnitude * factor,
            stability=self.stability * 0.7,
        )

    @staticmethod
    def from_modifiers(base: float, modifiers: List[str]) -> "ChargeState":
        """
        Apply linguistic modifiers to base charge.

        Args:
            base: Base polarity value
            modifiers: List of modifier words

        Returns:
            ChargeState with modifiers applied
        """
        polarity = base
        magnitude = 1.0
        stability = 0.7

        for mod in modifiers:
            mod_lower = mod.lower()
            # Negation
            if mod_lower in ["not", "n't", "never", "no", "none", "neither", "nor"]:
                polarity *= -1
            # Intensifiers
            elif mod_lower in ["very", "extremely", "absolutely", "totally", "completely"]:
                magnitude = min(1.0, magnitude * 1.3)
                stability = min(1.0, stability * 1.1)
            # Hedges
            elif mod_lower in ["maybe", "perhaps", "possibly", "might", "could"]:
                magnitude *= 0.5
                stability *= 0.7
            # Confidence boosters
            elif mod_lower in ["definitely", "certainly", "surely", "clearly"]:
                magnitude = min(1.0, magnitude * 1.2)
                stability = min(1.0, stability * 1.2)
            # Doubt markers
            elif mod_lower in ["unlikely", "doubtful", "questionable"]:
                magnitude *= 0.6
                stability *= 0.5

        return ChargeState(polarity, magnitude, stability)

    @staticmethod
    def neutral() -> "ChargeState":
        """Return a neutral charge state."""
        return ChargeState(polarity=0.0, magnitude=0.5, stability=0.5)

    @staticmethod
    def positive(strength: float = 1.0) -> "ChargeState":
        """Return a positive charge state."""
        return ChargeState(polarity=1.0, magnitude=strength, stability=0.7)

    @staticmethod
    def negative(strength: float = 1.0) -> "ChargeState":
        """Return a negative charge state."""
        return ChargeState(polarity=-1.0, magnitude=strength, stability=0.7)

    def __repr__(self) -> str:
        sign = "+" if self.polarity >= 0 else ""
        return f"ChargeState({sign}{self.polarity:.2f}, mag={self.magnitude:.2f}, stab={self.stability:.2f})"
