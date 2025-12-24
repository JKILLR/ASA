#!/usr/bin/env python3
"""
Belief Robustness Testing Demo for ASA.

This example demonstrates ASA's ability to distinguish between:
- Core beliefs that resist weak contradictions
- Peripheral beliefs that update easily
- How sufficient evidence can update even core beliefs

The key insight: ASA uses ionization energy (based on shell depth)
to protect core beliefs while allowing peripheral beliefs to update.
"""

from asa.testing import BeliefRobustnessTest


def main():
    print("=" * 60)
    print("ASA Belief Robustness Testing Demo")
    print("=" * 60)

    print("\nThe key question: Can ASA distinguish between")
    print("'a fact I just read' and 'a core truth I have verified'?")
    print()

    # Initialize test framework
    test = BeliefRobustnessTest()

    # Run individual tests with explanations
    print("\n" + "-" * 60)
    print("TEST 1: Core Belief Resistance")
    print("-" * 60)
    print("Scenario: Establish strong belief that 'sky is blue' (100 exposures, 95% quality)")
    print("Then attack with weak evidence that 'sky is green' (1 exposure, 20% quality)")
    print("Expected: Belief should remain 'blue' with high confidence")
    result1 = test.test_core_belief_resistance()
    print(f"Result: {'PASS' if result1.passed else 'FAIL'}")
    print(f"Details: {result1.details}")

    print("\n" + "-" * 60)
    print("TEST 2: Peripheral Update")
    print("-" * 60)
    print("Scenario: Establish weak belief that 'restaurant is good' (2 exposures, 50% quality)")
    print("Then present moderate evidence that 'restaurant is bad' (5 exposures, 60% quality)")
    print("Expected: Belief should update to 'bad'")
    result2 = test.test_peripheral_update()
    print(f"Result: {'PASS' if result2.passed else 'FAIL'}")
    print(f"Details: {result2.details}")

    print("\n" + "-" * 60)
    print("TEST 3: Sufficient Evidence Updates Core")
    print("-" * 60)
    print("Scenario: Establish moderately strong belief that 'earth is flat' (50 exposures, 90% quality)")
    print("Then present overwhelming evidence that 'earth is round' (200 exposures, 98% quality)")
    print("Expected: Even the core belief should update to 'round'")
    result3 = test.test_sufficient_evidence_updates_core()
    print(f"Result: {'PASS' if result3.passed else 'FAIL'}")
    print(f"Details: {result3.details}")

    print("\n" + "-" * 60)
    print("TEST 4: Contradiction Awareness")
    print("-" * 60)
    print("Scenario: Establish core belief that 'vaccine is safe' (100 exposures, 95% quality)")
    print("Then present weak contradiction that 'vaccine is unsafe' (5 exposures, 30% quality)")
    print("Expected: Belief stays 'safe' BUT contradiction is tracked")
    result4 = test.test_contradiction_awareness()
    print(f"Result: {'PASS' if result4.passed else 'FAIL'}")
    print(f"Details: {result4.details}")

    # Run full test suite
    print("\n" + "=" * 60)
    print("FULL TEST SUITE RESULTS")
    print("=" * 60)

    results = test.run_all()
    test.print_report(results)

    # Explain the mechanism
    print("\n" + "=" * 60)
    print("HOW IT WORKS")
    print("=" * 60)
    print("""
ASA uses "ionization energy" to protect core beliefs:

1. SHELL DEPTH:
   - Inner shells (0, 1): High ionization energy → hard to modify
   - Outer shells (2, 3): Low ionization energy → easy to modify

2. BELIEF STRENGTH:
   - Strong beliefs: High repetition, high quality sources
   - Weak beliefs: Low repetition, low quality sources

3. UPDATE PROBABILITY:
   update_prob = evidence / (evidence + ionization × belief_strength)

   - Core belief + weak evidence → low update probability
   - Peripheral belief + moderate evidence → high update probability
   - Core belief + overwhelming evidence → eventual update

4. KEY INSIGHT:
   Unlike traditional embeddings, ASA tracks WHERE information came from
   and HOW strongly it was established. This prevents the "gullible RAG"
   problem where any new information overwrites existing knowledge.
""")


if __name__ == "__main__":
    main()
