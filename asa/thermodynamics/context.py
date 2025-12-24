"""
LearnableSemanticContext and TemperatureSchedule - Runtime context for semantic processing.

The semantic context provides:
- Current temperature (cognitive mode)
- Bond threshold derived from temperature
- Ionization energies for belief modification
- Boltzmann probabilities for energy barriers
"""

import math
from typing import Callable

import torch.nn as nn

from .learnable import LearnableThermodynamics


class LearnableSemanticContext(nn.Module):
    """
    Semantic context with learnable thermodynamics.

    Temperature is a runtime state; thermodynamic parameters are learned.

    Temperature interpretation:
    - 0.0 = Cold/analytical: High thresholds, stable charges
    - 0.5 = Moderate: Balanced bonding and stability
    - 1.0 = Hot/creative: Low thresholds, volatile charges
    """

    def __init__(self, thermo: LearnableThermodynamics = None):
        super().__init__()
        self.thermo = thermo if thermo is not None else LearnableThermodynamics()
        self._temperature = 0.5  # Runtime state, not a parameter

    @property
    def temperature(self) -> float:
        """Get current temperature."""
        return self._temperature

    @temperature.setter
    def temperature(self, value: float):
        """Set temperature, clamped to [0, 1]."""
        self._temperature = max(0.0, min(1.0, value))

    @property
    def bond_threshold(self) -> float:
        """Get current bond formation threshold."""
        return self.thermo.bond_threshold(self._temperature).item()

    @property
    def charge_volatility(self) -> float:
        """Get current charge volatility."""
        return self.thermo.charge_volatility(self._temperature).item()

    def ionization_energy(self, shell_level: int, max_shells: int = 4) -> float:
        """
        Get ionization energy for a shell level.

        Args:
            shell_level: Shell index (0 = innermost)
            max_shells: Maximum number of shells

        Returns:
            Ionization energy for modifying associations at this level
        """
        return self.thermo.ionization_energy(shell_level, max_shells).item()

    def boltzmann_probability(self, energy_barrier: float) -> float:
        """
        Probability of overcoming an energy barrier.

        Uses Boltzmann-like probability: P = exp(-E/T)

        Args:
            energy_barrier: Energy barrier to overcome

        Returns:
            Probability [0, 1] of overcoming the barrier
        """
        if self._temperature < 0.01:
            return 0.0 if energy_barrier > 0 else 1.0
        return math.exp(-energy_barrier / self._temperature)

    def can_modify_shell(self, shell_level: int, max_shells: int = 4) -> bool:
        """
        Check if modification at shell level is thermodynamically possible.

        Args:
            shell_level: Shell index to modify
            max_shells: Maximum number of shells

        Returns:
            True if modification is likely given current temperature
        """
        energy = self.ionization_energy(shell_level, max_shells)
        prob = self.boltzmann_probability(energy)
        return prob > 0.5

    def set_mode(self, mode: str) -> None:
        """
        Set processing mode via temperature.

        Args:
            mode: One of 'analytical', 'balanced', 'creative'
        """
        modes = {
            "analytical": 0.1,
            "balanced": 0.5,
            "creative": 0.9,
        }
        if mode in modes:
            self._temperature = modes[mode]

    def get_state(self) -> dict:
        """Get current state for logging/debugging."""
        return {
            "temperature": self._temperature,
            "bond_threshold": self.bond_threshold,
            "charge_volatility": self.charge_volatility,
            "thermo_params": self.thermo.get_state_dict_readable(),
        }


class TemperatureSchedule:
    """
    Annealing schedules for semantic processing.

    Provides various scheduling strategies for temperature:
    - Constant: Fixed temperature
    - Linear decay: Gradual cooling
    - Cosine annealing: Smooth oscillation
    - Exploration-exploitation: Start hot, cool down
    """

    @staticmethod
    def constant(temp: float) -> Callable[[int], float]:
        """
        Constant temperature schedule.

        Args:
            temp: Fixed temperature value

        Returns:
            Schedule function
        """
        return lambda step: temp

    @staticmethod
    def linear_decay(
        start: float, end: float, steps: int
    ) -> Callable[[int], float]:
        """
        Linear temperature decay.

        Args:
            start: Starting temperature
            end: Ending temperature
            steps: Number of steps for decay

        Returns:
            Schedule function
        """

        def schedule(step: int) -> float:
            progress = min(1.0, step / steps)
            return start + (end - start) * progress

        return schedule

    @staticmethod
    def cosine_annealing(
        high: float, low: float, period: int
    ) -> Callable[[int], float]:
        """
        Cosine annealing schedule.

        Smoothly oscillates between high and low temperatures.

        Args:
            high: Maximum temperature
            low: Minimum temperature
            period: Period of oscillation in steps

        Returns:
            Schedule function
        """

        def schedule(step: int) -> float:
            progress = (step % period) / period
            return low + (high - low) * (1 + math.cos(math.pi * progress)) / 2

        return schedule

    @staticmethod
    def exploration_exploitation(steps: int) -> Callable[[int], float]:
        """
        Start hot (explore), cool down (exploit).

        Good for learning phases where you want to discover
        new associations first, then consolidate.

        Args:
            steps: Total steps for schedule

        Returns:
            Schedule function
        """
        return TemperatureSchedule.linear_decay(0.9, 0.2, steps)

    @staticmethod
    def warmup_decay(
        warmup_steps: int, peak: float, decay_steps: int, end: float
    ) -> Callable[[int], float]:
        """
        Warm up to peak, then decay.

        Args:
            warmup_steps: Steps to reach peak
            peak: Peak temperature
            decay_steps: Steps for decay after peak
            end: Final temperature

        Returns:
            Schedule function
        """

        def schedule(step: int) -> float:
            if step < warmup_steps:
                # Warmup phase
                return (step / warmup_steps) * peak
            else:
                # Decay phase
                decay_step = step - warmup_steps
                progress = min(1.0, decay_step / decay_steps)
                return peak + (end - peak) * progress

        return schedule

    @staticmethod
    def step_decay(
        initial: float, factor: float, step_size: int
    ) -> Callable[[int], float]:
        """
        Step-wise temperature decay.

        Args:
            initial: Initial temperature
            factor: Multiplicative factor at each step
            step_size: Steps between decays

        Returns:
            Schedule function
        """

        def schedule(step: int) -> float:
            num_decays = step // step_size
            return max(0.01, initial * (factor ** num_decays))

        return schedule
