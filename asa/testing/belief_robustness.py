"""
BeliefRobustnessTest - Test framework for ionization energy validation.

v1.1 Addition: Explicit test framework validating that the system
can distinguish between "a fact I just read" and "a core truth I have verified".

Key tests:
1. Core beliefs resist weak contradictions
2. Peripheral beliefs update easily
3. Sufficient evidence updates even core beliefs
4. System tracks contradictions even when not updating
"""

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional

import torch

from ..neural.model import AtomicSemanticModel
from ..storage.vector_store import AtomicVectorStore
from ..shells.manager import BoundedShellManager, MigrationConfig
from ..thermodynamics.learnable import LearnableThermodynamics
from ..core.config import AtomConfig


@dataclass
class TestResult:
    """Result of a single test."""

    test_name: str
    passed: bool
    details: Dict[str, Any] = field(default_factory=dict)
    error: Optional[str] = None


@dataclass
class TestSuiteResult:
    """Result of complete test suite."""

    total: int
    passed: int
    failed: int
    tests: List[TestResult]
    success_rate: float


class BeliefRobustnessTest:
    """
    Test framework for ionization energy validation.

    The key question: Can the system distinguish between
    "a fact I just read" and "a core truth I have verified"?

    Tests validate that:
    - Core beliefs (inner shell, high strength) resist weak contradictions
    - Peripheral beliefs (outer shell, low strength) update easily
    - Sufficient evidence can update even core beliefs
    - Contradictions are tracked even when beliefs don't update
    """

    def __init__(
        self,
        model: AtomicSemanticModel = None,
        store: AtomicVectorStore = None,
        config: AtomConfig = None,
    ):
        """
        Initialize test framework.

        Args:
            model: ASE model (optional, creates minimal if not provided)
            store: Vector store (optional, creates empty if not provided)
            config: Atom configuration
        """
        self.config = config or AtomConfig()
        self.model = model
        self.store = store or AtomicVectorStore(self.config)
        self.thermo = LearnableThermodynamics()

        # Belief storage for testing
        self.beliefs: Dict[str, Dict[str, Any]] = {}
        self.contradictions: Dict[str, List[str]] = {}

    def _establish_belief(
        self,
        concept: str,
        value: str,
        repetitions: int,
        quality: float,
    ) -> int:
        """
        Establish a belief through repeated exposure.

        Args:
            concept: Concept name
            value: Belief value
            repetitions: Number of exposures
            quality: Source quality [0, 1]

        Returns:
            Belief ID
        """
        # Create shell manager for this concept
        manager = BoundedShellManager(
            self.config,
            self.thermo,
            MigrationConfig(max_cascade_depth=3),
        )

        # Simulate repeated exposure
        strength = min(0.95, 0.3 + 0.05 * repetitions + 0.2 * quality)

        # Create mock association vector
        vector = torch.randn(self.config.shell_dims[0])

        for i in range(repetitions):
            manager.add_association(
                target_atom_id=hash(value) % 10000,
                vector=vector,
                strength=strength,
                timestamp=i,
                source_quality=quality,
            )

        # Store belief
        belief_id = len(self.beliefs)
        self.beliefs[concept] = {
            "id": belief_id,
            "value": value,
            "strength": strength,
            "manager": manager,
            "repetitions": repetitions,
            "quality": quality,
        }

        return belief_id

    def _present_evidence(
        self,
        concept: str,
        value: str,
        repetitions: int,
        quality: float,
    ) -> None:
        """
        Present evidence (possibly contradictory) to the system.

        Args:
            concept: Concept name
            value: Evidence value
            repetitions: Number of exposures
            quality: Evidence quality
        """
        if concept not in self.beliefs:
            # New concept
            self._establish_belief(concept, value, repetitions, quality)
            return

        existing = self.beliefs[concept]
        manager = existing["manager"]

        # Calculate evidence strength
        evidence_strength = min(0.95, 0.3 + 0.05 * repetitions + 0.2 * quality)

        # Check if this contradicts existing belief
        if existing["value"] != value:
            # Track contradiction
            if concept not in self.contradictions:
                self.contradictions[concept] = []
            self.contradictions[concept].append(value)

            # Get ionization energy for update resistance
            filled, total = manager.total_occupancy()
            avg_shell = 1 if filled < 10 else (2 if filled < 30 else 3)
            ionization = self.thermo.ionization_energy(avg_shell, 4).item()

            # Calculate update probability
            existing_strength = existing["strength"]
            update_prob = evidence_strength / (
                evidence_strength + ionization * existing_strength + 1e-8
            )

            # Decide whether to update
            if update_prob > 0.6:
                # Update belief
                existing["value"] = value
                existing["strength"] = evidence_strength
        else:
            # Reinforcing evidence
            existing["strength"] = min(
                1.0, existing["strength"] + 0.05 * repetitions * quality
            )

    def _query_belief(self, concept: str) -> Dict[str, Any]:
        """
        Query current belief state.

        Args:
            concept: Concept to query

        Returns:
            Belief state dict
        """
        if concept not in self.beliefs:
            return {"dominant": None, "confidence": 0.0}

        belief = self.beliefs[concept]
        return {
            "dominant": belief["value"],
            "confidence": belief["strength"],
            "repetitions": belief["repetitions"],
        }

    def _get_shell_level(self, concept: str, value: str) -> int:
        """
        Get shell level of a belief.

        Args:
            concept: Concept name
            value: Belief value

        Returns:
            Shell level (0 = innermost)
        """
        if concept not in self.beliefs:
            return -1

        belief = self.beliefs[concept]
        if belief["value"] != value:
            return -1

        # Estimate shell from strength
        strength = belief["strength"]
        if strength >= 0.8:
            return 0
        elif strength >= 0.6:
            return 1
        elif strength >= 0.4:
            return 2
        else:
            return 3

    def _get_contradictions(self, concept: str) -> List[str]:
        """
        Get tracked contradictions for a concept.

        Args:
            concept: Concept name

        Returns:
            List of contradicting values
        """
        return self.contradictions.get(concept, [])

    def test_core_belief_resistance(self) -> TestResult:
        """
        Test: Core beliefs should resist weak contradictions.

        Establishes a strong core belief, attacks with weak evidence,
        verifies the belief remains unchanged.
        """
        result = TestResult(
            test_name="core_belief_resistance",
            passed=False,
            details={},
        )

        try:
            # Clear state
            self.beliefs.clear()
            self.contradictions.clear()

            # Establish strong core belief
            self._establish_belief(
                "sky_color", "blue", repetitions=100, quality=0.95
            )
            initial_shell = self._get_shell_level("sky_color", "blue")
            result.details["initial_shell"] = initial_shell

            # Attack with weak contradiction
            self._present_evidence(
                "sky_color", "green", repetitions=1, quality=0.2
            )

            # Verify resistance
            query = self._query_belief("sky_color")
            result.details["post_attack_belief"] = query["dominant"]
            result.details["confidence"] = query["confidence"]

            result.passed = (
                query["dominant"] == "blue" and query["confidence"] > 0.8
            )

        except Exception as e:
            result.error = str(e)

        return result

    def test_peripheral_update(self) -> TestResult:
        """
        Test: Peripheral beliefs should update easily.

        Establishes a weak peripheral belief, presents moderate evidence,
        verifies the belief updates.
        """
        result = TestResult(
            test_name="peripheral_update",
            passed=False,
            details={},
        )

        try:
            self.beliefs.clear()
            self.contradictions.clear()

            # Establish weak belief
            self._establish_belief(
                "restaurant", "good", repetitions=2, quality=0.5
            )
            initial_shell = self._get_shell_level("restaurant", "good")
            result.details["initial_shell"] = initial_shell

            # Present moderate contradicting evidence
            self._present_evidence(
                "restaurant", "bad", repetitions=5, quality=0.6
            )

            query = self._query_belief("restaurant")
            result.details["final_belief"] = query["dominant"]

            result.passed = query["dominant"] == "bad"

        except Exception as e:
            result.error = str(e)

        return result

    def test_sufficient_evidence_updates_core(self) -> TestResult:
        """
        Test: Even core beliefs should update with overwhelming evidence.

        Establishes a core belief, presents overwhelming contradicting evidence,
        verifies the belief eventually updates.
        """
        result = TestResult(
            test_name="sufficient_evidence",
            passed=False,
            details={},
        )

        try:
            self.beliefs.clear()
            self.contradictions.clear()

            # Establish moderately strong belief
            self._establish_belief(
                "earth_shape", "flat", repetitions=50, quality=0.9
            )

            # Overwhelming evidence
            self._present_evidence(
                "earth_shape", "round", repetitions=200, quality=0.98
            )

            query = self._query_belief("earth_shape")
            result.details["final_belief"] = query["dominant"]

            result.passed = query["dominant"] == "round"

        except Exception as e:
            result.error = str(e)

        return result

    def test_contradiction_awareness(self) -> TestResult:
        """
        Test: System should track contradictions even when not updating.

        Establishes a core belief, presents weak contradiction,
        verifies belief stays but contradiction is tracked.
        """
        result = TestResult(
            test_name="contradiction_awareness",
            passed=False,
            details={},
        )

        try:
            self.beliefs.clear()
            self.contradictions.clear()

            # Establish core belief
            self._establish_belief(
                "vaccine", "safe", repetitions=100, quality=0.95
            )

            # Weak contradiction
            self._present_evidence(
                "vaccine", "unsafe", repetitions=5, quality=0.3
            )

            query = self._query_belief("vaccine")
            contradictions = self._get_contradictions("vaccine")

            result.details["dominant"] = query["dominant"]
            result.details["contradictions"] = len(contradictions)
            result.details["contradiction_values"] = contradictions

            # Should maintain belief but track contradiction
            result.passed = (
                query["dominant"] == "safe" and "unsafe" in contradictions
            )

        except Exception as e:
            result.error = str(e)

        return result

    def run_all(self) -> TestSuiteResult:
        """
        Run complete test suite.

        Returns:
            TestSuiteResult with all test results
        """
        tests = [
            self.test_core_belief_resistance,
            self.test_peripheral_update,
            self.test_sufficient_evidence_updates_core,
            self.test_contradiction_awareness,
        ]

        results = []
        passed = 0
        failed = 0

        for test_fn in tests:
            try:
                result = test_fn()
                results.append(result)
                if result.passed:
                    passed += 1
                else:
                    failed += 1
            except Exception as e:
                results.append(
                    TestResult(
                        test_name=test_fn.__name__,
                        passed=False,
                        error=str(e),
                    )
                )
                failed += 1

        return TestSuiteResult(
            total=len(tests),
            passed=passed,
            failed=failed,
            tests=results,
            success_rate=passed / len(tests),
        )

    def print_report(self, suite_result: TestSuiteResult) -> None:
        """Print formatted test report."""
        print("\n" + "=" * 60)
        print("BELIEF ROBUSTNESS TEST REPORT")
        print("=" * 60)
        print(f"\nTotal: {suite_result.total}")
        print(f"Passed: {suite_result.passed}")
        print(f"Failed: {suite_result.failed}")
        print(f"Success Rate: {suite_result.success_rate:.1%}")
        print("\n" + "-" * 60)

        for test in suite_result.tests:
            status = "✓ PASS" if test.passed else "✗ FAIL"
            print(f"\n{status}: {test.test_name}")

            if test.details:
                for k, v in test.details.items():
                    print(f"    {k}: {v}")

            if test.error:
                print(f"    ERROR: {test.error}")

        print("\n" + "=" * 60)
