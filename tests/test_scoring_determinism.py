"""
tests/test_scoring_determinism.py

Test suite for scoring service determinism.
Validates:
- Same input produces same output every time
- Scoring is reproducible across multiple runs
- Score bounds are always respected [0, 100]
- Different inputs produce different outputs (when appropriate)
"""

import pytest
from typing import List

from components.scoring_service import ScoringService, ScoringRules
from components.models import AgentFinding, Severity, FindingType


class TestScoringDeterminism:
    """Test that scoring is deterministic."""

    @pytest.fixture
    def scoring_service(self):
        """Initialize scoring service."""
        return ScoringService()

    @pytest.fixture
    def sample_findings(self) -> List[AgentFinding]:
        """Create consistent sample findings."""
        return [
            AgentFinding(
                description="Accessibility issue",
                severity=Severity.CRITICAL,
                finding_type=FindingType.ACCESSIBILITY,
                impact_score=0.9
            ),
            AgentFinding(
                description="Inconsistent color usage",
                severity=Severity.HIGH,
                finding_type=FindingType.DESIGN_CONSISTENCY,
                impact_score=0.7
            ),
            AgentFinding(
                description="Good spacing hierarchy",
                severity=Severity.LOW,
                finding_type=FindingType.TYPOGRAPHY,
                impact_score=0.2
            ),
        ]

    def test_same_findings_same_score(self, scoring_service, sample_findings):
        """Same findings should always produce the same score."""
        score1 = scoring_service.calculate_score(sample_findings)
        score2 = scoring_service.calculate_score(sample_findings)
        score3 = scoring_service.calculate_score(sample_findings)

        # All three scores should be identical
        assert score1 == score2 == score3, \
            f"Scores differ: {score1}, {score2}, {score3}"

    def test_score_bounds_respected(self, scoring_service, sample_findings):
        """Scores should always be in [0, 100] range."""
        for _ in range(10):
            score = scoring_service.calculate_score(sample_findings)
            assert 0 <= score <= 100, f"Score out of bounds: {score}"

    def test_empty_findings_score(self, scoring_service):
        """Empty findings list should produce consistent score."""
        score1 = scoring_service.calculate_score([])
        score2 = scoring_service.calculate_score([])

        assert score1 == score2
        assert 0 <= score1 <= 100

    def test_single_finding_determinism(self, scoring_service):
        """Single finding should always score the same."""
        finding = AgentFinding(
            description="Single test finding",
            severity=Severity.MEDIUM,
            finding_type=FindingType.USABILITY,
            impact_score=0.5
        )

        scores = [scoring_service.calculate_score([finding]) for _ in range(5)]

        # All scores should be identical
        assert all(s == scores[0] for s in scores), \
            f"Scores differ: {scores}"

    def test_order_independence(self, scoring_service, sample_findings):
        """Score should not depend on finding order."""
        # Create reversed list
        reversed_findings = list(reversed(sample_findings))

        score_original = scoring_service.calculate_score(sample_findings)
        score_reversed = scoring_service.calculate_score(reversed_findings)

        # Scores should be identical (order shouldn't matter)
        assert score_original == score_reversed, \
            f"Order affected score: {score_original} vs {score_reversed}"

    def test_different_findings_different_scores(self, scoring_service):
        """Different findings should typically produce different scores."""
        findings1 = [
            AgentFinding(
                description="Critical issue",
                severity=Severity.CRITICAL,
                finding_type=FindingType.ACCESSIBILITY,
                impact_score=0.9
            )
        ]

        findings2 = [
            AgentFinding(
                description="Minor issue",
                severity=Severity.LOW,
                finding_type=FindingType.STYLING,
                impact_score=0.1
            )
        ]

        score1 = scoring_service.calculate_score(findings1)
        score2 = scoring_service.calculate_score(findings2)

        # Different findings should produce different scores
        # (unless by coincidence they're the same, but unlikely)
        assert score1 != score2, \
            f"Different findings produced same score: {score1}"

    def test_severity_determinism(self, scoring_service):
        """Severity classification should be deterministic."""
        findings = [
            AgentFinding(
                description="Issue 1",
                severity=Severity.CRITICAL,
                finding_type=FindingType.ACCESSIBILITY,
                impact_score=0.9
            ),
            AgentFinding(
                description="Issue 2",
                severity=Severity.HIGH,
                finding_type=FindingType.DESIGN_CONSISTENCY,
                impact_score=0.7
            ),
        ]

        # Run scoring multiple times
        scores = [scoring_service.calculate_score(findings) for _ in range(10)]

        # All scores should match
        assert len(set(scores)) == 1, \
            f"Severity classification not deterministic: {scores}"

    def test_impact_score_scaling_determinism(self, scoring_service):
        """Impact score scaling should be deterministic."""
        findings_low_impact = [
            AgentFinding(
                description="Test",
                severity=Severity.MEDIUM,
                finding_type=FindingType.TYPOGRAPHY,
                impact_score=0.1
            )
        ]

        findings_high_impact = [
            AgentFinding(
                description="Test",
                severity=Severity.MEDIUM,
                finding_type=FindingType.TYPOGRAPHY,
                impact_score=0.9
            )
        ]

        # Run multiple times to ensure consistency
        scores_low = [scoring_service.calculate_score(
            findings_low_impact) for _ in range(5)]
        scores_high = [scoring_service.calculate_score(
            findings_high_impact) for _ in range(5)]

        # All low-impact scores should be identical
        assert len(set(scores_low)) == 1
        # All high-impact scores should be identical
        assert len(set(scores_high)) == 1
        # High-impact should generally score lower (more negative impact)
        assert sum(scores_high) / \
            len(scores_high) < sum(scores_low) / len(scores_low)

    def test_large_finding_set_determinism(self, scoring_service):
        """Large finding sets should still be deterministic."""
        findings = [
            AgentFinding(
                description=f"Finding {i}",
                severity=Severity.MEDIUM if i % 2 == 0 else Severity.HIGH,
                finding_type=FindingType.ACCESSIBILITY,
                impact_score=0.5 + (i % 10) * 0.05
            )
            for i in range(50)
        ]

        scores = [scoring_service.calculate_score(findings) for _ in range(3)]

        # All scores should be identical
        assert all(s == scores[0] for s in scores), \
            f"Large set scoring not deterministic: {scores}"


class TestScoringEdgeCases:
    """Test edge cases in scoring."""

    @pytest.fixture
    def scoring_service(self):
        """Initialize scoring service."""
        return ScoringService()

    def test_max_impact_score_finding(self, scoring_service):
        """Finding with max impact (1.0) should score consistently."""
        finding = AgentFinding(
            description="Maximum impact finding",
            severity=Severity.CRITICAL,
            finding_type=FindingType.ACCESSIBILITY,
            impact_score=1.0
        )

        score1 = scoring_service.calculate_score([finding])
        score2 = scoring_service.calculate_score([finding])

        assert score1 == score2
        assert 0 <= score1 <= 100

    def test_min_impact_score_finding(self, scoring_service):
        """Finding with min impact (0.0) should score consistently."""
        finding = AgentFinding(
            description="No impact finding",
            severity=Severity.LOW,
            finding_type=FindingType.STYLING,
            impact_score=0.0
        )

        score1 = scoring_service.calculate_score([finding])
        score2 = scoring_service.calculate_score([finding])

        assert score1 == score2
        assert 0 <= score1 <= 100

    def test_all_severity_levels(self, scoring_service):
        """Each severity level should score deterministically."""
        for severity in [Severity.CRITICAL, Severity.HIGH, Severity.MEDIUM, Severity.LOW]:
            finding = AgentFinding(
                description=f"Finding with {severity.name} severity",
                severity=severity,
                finding_type=FindingType.ACCESSIBILITY,
                impact_score=0.5
            )

            scores = [scoring_service.calculate_score(
                [finding]) for _ in range(5)]

            # All scores for this severity should be identical
            assert all(s == scores[0] for s in scores), \
                f"Determinism failed for {severity.name}: {scores}"

    def test_mixed_severities_determinism(self, scoring_service):
        """Mixed severity findings should score deterministically."""
        findings = [
            AgentFinding(
                description="Critical",
                severity=Severity.CRITICAL,
                finding_type=FindingType.ACCESSIBILITY,
                impact_score=0.8
            ),
            AgentFinding(
                description="Low",
                severity=Severity.LOW,
                finding_type=FindingType.STYLING,
                impact_score=0.2
            ),
            AgentFinding(
                description="Medium",
                severity=Severity.MEDIUM,
                finding_type=FindingType.TYPOGRAPHY,
                impact_score=0.5
            ),
        ]

        scores = [scoring_service.calculate_score(findings) for _ in range(10)]

        assert len(set(scores)) == 1, \
            f"Mixed severity scoring not deterministic: {scores}"
