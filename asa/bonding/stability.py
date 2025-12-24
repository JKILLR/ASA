"""
StabilityAnalyzer - Analyzes structural stability of semantic molecules.

Detects and resolves stability issues:
- Contradictions: Conflicting charges in bonded atoms
- Circular references: Cycles in bond graph
- Valence gaps: Unfilled valence slots
- Overcrowding: Too many bonds on an atom
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Tuple

from ..core.atoms import SemanticMolecule, SemanticAtom, SemanticBond


@dataclass
class StabilityIssue:
    """A single stability issue found in a molecule."""

    issue_type: str  # "contradiction", "circular", "valence_gap", "overcrowding"
    severity: float  # 0.0 to 1.0
    involved_atoms: List[int]
    description: str
    resolution: Optional[str] = None


@dataclass
class StabilityReport:
    """Complete stability analysis report."""

    is_stable: bool
    overall_score: float  # 1.0 = perfectly stable
    issues: List[StabilityIssue] = field(default_factory=list)

    def critical_issues(self) -> List[StabilityIssue]:
        """Get issues with severity > 0.7."""
        return [i for i in self.issues if i.severity > 0.7]

    def warnings(self) -> List[StabilityIssue]:
        """Get issues with severity between 0.3 and 0.7."""
        return [i for i in self.issues if 0.3 < i.severity <= 0.7]

    def info(self) -> List[StabilityIssue]:
        """Get issues with severity <= 0.3."""
        return [i for i in self.issues if i.severity <= 0.3]

    def __repr__(self) -> str:
        return (
            f"StabilityReport(stable={self.is_stable}, score={self.overall_score:.2f}, "
            f"issues={len(self.issues)})"
        )


class StabilityAnalyzer:
    """Analyzes structural stability of semantic molecules."""

    def __init__(self, contradiction_threshold: float = 0.5):
        """
        Initialize stability analyzer.

        Args:
            contradiction_threshold: Threshold for charge conflict detection
        """
        self.contradiction_threshold = contradiction_threshold

    def analyze(self, molecule: SemanticMolecule) -> StabilityReport:
        """
        Analyze molecule for stability issues.

        Args:
            molecule: Molecule to analyze

        Returns:
            StabilityReport with all issues found
        """
        issues = []

        # Check for various issue types
        issues.extend(self._find_contradictions(molecule))
        issues.extend(self._find_circular_refs(molecule))
        issues.extend(self._find_valence_gaps(molecule))
        issues.extend(self._find_overcrowding(molecule))

        if not issues:
            return StabilityReport(True, 1.0, [])

        # Calculate overall score
        total_severity = sum(i.severity for i in issues)
        score = max(0.0, 1.0 - total_severity / max(1, len(issues)))

        # Molecule is unstable if any critical issues exist
        is_stable = len([i for i in issues if i.severity > 0.5]) == 0

        return StabilityReport(is_stable, score, issues)

    def _find_contradictions(
        self, molecule: SemanticMolecule
    ) -> List[StabilityIssue]:
        """Find charge contradictions in bonded atoms."""
        issues = []

        for bond in molecule.bonds:
            if bond.strength < 0.5:
                continue  # Weak bonds don't matter as much

            charge_a = bond.atom_a.effective_charge
            charge_b = bond.atom_b.effective_charge
            product = charge_a * charge_b

            # Strong opposite charges in a strong bond = contradiction
            if product < -self.contradiction_threshold:
                severity = min(1.0, abs(product) * bond.strength)
                issues.append(
                    StabilityIssue(
                        "contradiction",
                        severity,
                        [bond.atom_a.atom_id, bond.atom_b.atom_id],
                        f"Charge conflict: {bond.atom_a.token}({charge_a:.2f}) — "
                        f"{bond.atom_b.token}({charge_b:.2f})",
                        "Weaken bond or align charges",
                    )
                )

        return issues

    def _find_circular_refs(
        self, molecule: SemanticMolecule
    ) -> List[StabilityIssue]:
        """Find circular reference chains (cycles) in bond graph."""
        # Build adjacency list
        graph: Dict[int, List[int]] = {}
        for bond in molecule.bonds:
            a, b = bond.atom_a.atom_id, bond.atom_b.atom_id
            graph.setdefault(a, []).append(b)
            graph.setdefault(b, []).append(a)

        visited: Set[int] = set()
        issues = []

        def dfs(node: int, path: List[int]) -> Optional[List[int]]:
            visited.add(node)
            for neighbor in graph.get(node, []):
                if neighbor in path and len(path) > 2:
                    # Found cycle
                    cycle_start = path.index(neighbor)
                    return path[cycle_start:]
                if neighbor not in visited:
                    result = dfs(neighbor, path + [neighbor])
                    if result:
                        return result
            return None

        for node in graph:
            if node not in visited:
                cycle = dfs(node, [node])
                if cycle and len(cycle) >= 3:
                    issues.append(
                        StabilityIssue(
                            "circular",
                            0.6,  # Moderate severity for cycles
                            cycle,
                            f"Circular reference chain of length {len(cycle)}",
                            "Break one bond in cycle",
                        )
                    )

        return issues

    def _find_valence_gaps(
        self, molecule: SemanticMolecule
    ) -> List[StabilityIssue]:
        """Find atoms with unfilled valence slots."""
        # Count bonds per atom
        bond_counts: Dict[int, int] = {}
        for bond in molecule.bonds:
            bond_counts[bond.atom_a.atom_id] = (
                bond_counts.get(bond.atom_a.atom_id, 0) + 1
            )
            bond_counts[bond.atom_b.atom_id] = (
                bond_counts.get(bond.atom_b.atom_id, 0) + 1
            )

        issues = []
        for atom in molecule.atoms:
            expected = atom.valence_count
            actual = bond_counts.get(atom.atom_id, 0)
            if actual < expected:
                gap = expected - actual
                severity = min(1.0, gap / max(1, expected)) * 0.5
                issues.append(
                    StabilityIssue(
                        "valence_gap",
                        severity,
                        [atom.atom_id],
                        f"{atom.token} has {gap} unfilled valence slot(s)",
                        "Find compatible bonding partners",
                    )
                )

        return issues

    def _find_overcrowding(
        self, molecule: SemanticMolecule
    ) -> List[StabilityIssue]:
        """Find atoms with too many bonds."""
        # Count bonds per atom
        bond_counts: Dict[int, int] = {}
        for bond in molecule.bonds:
            bond_counts[bond.atom_a.atom_id] = (
                bond_counts.get(bond.atom_a.atom_id, 0) + 1
            )
            bond_counts[bond.atom_b.atom_id] = (
                bond_counts.get(bond.atom_b.atom_id, 0) + 1
            )

        issues = []
        for atom in molecule.atoms:
            max_valence = atom.valence_count
            actual = bond_counts.get(atom.atom_id, 0)
            if actual > max_valence:
                excess = actual - max_valence
                severity = min(1.0, excess / max(1, max_valence)) * 0.7
                issues.append(
                    StabilityIssue(
                        "overcrowding",
                        severity,
                        [atom.atom_id],
                        f"{atom.token} has {excess} excess bond(s)",
                        "Break weakest bonds",
                    )
                )

        return issues


class StabilityResolver:
    """Attempts to resolve stability issues."""

    def __init__(self, analyzer: StabilityAnalyzer = None):
        """
        Initialize resolver.

        Args:
            analyzer: Stability analyzer to use
        """
        self.analyzer = analyzer or StabilityAnalyzer()

    def auto_resolve(
        self,
        molecule: SemanticMolecule,
        max_iterations: int = 10,
    ) -> Tuple[SemanticMolecule, List[str]]:
        """
        Automatically resolve stability issues.

        Args:
            molecule: Molecule to stabilize
            max_iterations: Maximum resolution attempts

        Returns:
            (stabilized_molecule, list_of_actions_taken)
        """
        actions = []

        for iteration in range(max_iterations):
            report = self.analyzer.analyze(molecule)

            if report.is_stable:
                break

            critical = report.critical_issues()
            if not critical:
                break

            # Resolve most severe issue
            issue = max(critical, key=lambda i: i.severity)
            action = self._resolve_issue(molecule, issue)

            if action:
                actions.append(f"Iteration {iteration + 1}: {action}")
            else:
                break

        return molecule, actions

    def _resolve_issue(
        self, molecule: SemanticMolecule, issue: StabilityIssue
    ) -> Optional[str]:
        """Resolve a single stability issue."""
        if issue.issue_type == "contradiction":
            return self._resolve_contradiction(molecule, issue)
        elif issue.issue_type == "circular":
            return self._break_cycle(molecule, issue)
        elif issue.issue_type == "overcrowding":
            return self._reduce_overcrowding(molecule, issue)
        return None

    def _resolve_contradiction(
        self, molecule: SemanticMolecule, issue: StabilityIssue
    ) -> Optional[str]:
        """Resolve a charge contradiction by weakening the bond."""
        if len(issue.involved_atoms) < 2:
            return None

        a_id, b_id = issue.involved_atoms[:2]
        for bond in molecule.bonds:
            if {bond.atom_a.atom_id, bond.atom_b.atom_id} == {a_id, b_id}:
                old_strength = bond.strength
                bond.strength *= 0.5
                return (
                    f"Weakened contradictory bond between atoms {a_id} and {b_id} "
                    f"from {old_strength:.2f} to {bond.strength:.2f}"
                )
        return None

    def _break_cycle(
        self, molecule: SemanticMolecule, issue: StabilityIssue
    ) -> Optional[str]:
        """Break a cycle by removing the weakest bond."""
        cycle_atoms = set(issue.involved_atoms)
        weakest_bond = None
        weakest_strength = float("inf")

        for bond in molecule.bonds:
            a_id = bond.atom_a.atom_id
            b_id = bond.atom_b.atom_id
            if a_id in cycle_atoms and b_id in cycle_atoms:
                if bond.strength < weakest_strength:
                    weakest_strength = bond.strength
                    weakest_bond = bond

        if weakest_bond:
            molecule.bonds.remove(weakest_bond)
            return f"Broke cycle by removing bond (strength {weakest_strength:.2f})"
        return None

    def _reduce_overcrowding(
        self, molecule: SemanticMolecule, issue: StabilityIssue
    ) -> Optional[str]:
        """Reduce overcrowding by removing excess bonds."""
        if not issue.involved_atoms:
            return None

        atom_id = issue.involved_atoms[0]
        atom_bonds = [
            b
            for b in molecule.bonds
            if b.atom_a.atom_id == atom_id or b.atom_b.atom_id == atom_id
        ]

        atom = next(
            (a for a in molecule.atoms if a.atom_id == atom_id), None
        )
        if not atom:
            return None

        excess = len(atom_bonds) - atom.valence_count
        if excess <= 0:
            return None

        # Remove weakest bonds
        atom_bonds.sort(key=lambda b: b.strength)
        removed = 0
        for bond in atom_bonds[:excess]:
            molecule.bonds.remove(bond)
            removed += 1

        return f"Removed {removed} excess bond(s) from atom {atom_id}"
