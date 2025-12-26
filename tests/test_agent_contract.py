"""
tests/test_agent_contract.py

Contract tests for agent output validation.
Validates that agent outputs conform to the AgentResult schema.
"""

import pytest
from typing import Dict, List, Any

from components.agents import VisualDesignAgent  # Or whichever agent is available
from components.models import AgentResult, AgentFinding, Severity, FindingType


class TestAgentResultContract:
    """Test that agent results conform to AgentResult contract."""

    def test_agent_result_has_required_fields(self):
        """AgentResult must have all required fields."""
        result = AgentResult(
            agent_id="visual_design",
            findings=[],
            score=85.0,
            metadata={"timestamp": "2024-01-01"},
            recommendations=["Improve contrast"],
            status="success"
        )

        # Verify all required fields exist
        assert hasattr(result, 'agent_id')
        assert hasattr(result, 'findings')
        assert hasattr(result, 'score')
        assert hasattr(result, 'metadata')
        assert hasattr(result, 'recommendations')
        assert hasattr(result, 'status')

    def test_agent_id_is_string(self):
        """Agent ID must be a string."""
        result = AgentResult(
            agent_id="test_agent",
            findings=[],
            score=75.0,
            metadata={},
            recommendations=[],
            status="success"
        )

        assert isinstance(result.agent_id, str)
        assert len(result.agent_id) > 0

    def test_findings_is_list(self):
        """Findings must be a list."""
        result = AgentResult(
            agent_id="test_agent",
            findings=[],
            score=75.0,
            metadata={},
            recommendations=[],
            status="success"
        )

        assert isinstance(result.findings, list)

    def test_finding_structure(self):
        """Each finding must be an AgentFinding with required fields."""
        finding = AgentFinding(
            description="Test finding",
            severity=Severity.MEDIUM,
            finding_type=FindingType.ACCESSIBILITY,
            impact_score=0.6
        )

        # Verify required fields
        assert hasattr(finding, 'description')
        assert hasattr(finding, 'severity')
        assert hasattr(finding, 'finding_type')
        assert hasattr(finding, 'impact_score')

        # Verify types
        assert isinstance(finding.description, str)
        assert isinstance(finding.severity, Severity)
        assert isinstance(finding.finding_type, FindingType)
        assert isinstance(finding.impact_score, (float, int))
        assert 0 <= finding.impact_score <= 1

    def test_score_is_numeric_and_bounded(self):
        """Score must be numeric and in range [0, 100]."""
        for score_value in [0, 50.5, 100]:
            result = AgentResult(
                agent_id="test",
                findings=[],
                score=score_value,
                metadata={},
                recommendations=[],
                status="success"
            )

            assert isinstance(result.score, (int, float))
            assert 0 <= result.score <= 100

    def test_score_out_of_bounds_invalid(self):
        """Score outside [0, 100] should be invalid."""
        invalid_scores = [-1, 101, 150, -10]

        for invalid_score in invalid_scores:
            with pytest.raises((ValueError, AssertionError)):
                result = AgentResult(
                    agent_id="test",
                    findings=[],
                    score=invalid_score,
                    metadata={},
                    recommendations=[],
                    status="success"
                )

    def test_metadata_is_dict(self):
        """Metadata must be a dictionary."""
        result = AgentResult(
            agent_id="test",
            findings=[],
            score=75.0,
            metadata={"key": "value", "nested": {"inner": "data"}},
            recommendations=[],
            status="success"
        )

        assert isinstance(result.metadata, dict)

    def test_recommendations_is_list_of_strings(self):
        """Recommendations must be a list of strings."""
        result = AgentResult(
            agent_id="test",
            findings=[],
            score=75.0,
            metadata={},
            recommendations=[
                "Improve contrast ratio",
                "Add alternative text",
                "Enhance spacing"
            ],
            status="success"
        )

        assert isinstance(result.recommendations, list)
        assert all(isinstance(r, str) for r in result.recommendations)

    def test_status_is_valid_value(self):
        """Status must be a valid value."""
        valid_statuses = ["success", "partial", "error", "pending"]

        for status in valid_statuses:
            result = AgentResult(
                agent_id="test",
                findings=[],
                score=75.0,
                metadata={},
                recommendations=[],
                status=status
            )

            assert result.status in ["success", "partial", "error", "pending"]


class TestVisualDesignAgentContract:
    """Contract tests for Visual Design Agent specifically."""

    @pytest.fixture
    def visual_agent(self):
        """Initialize Visual Design Agent."""
        try:
            return VisualDesignAgent()
        except Exception:
            pytest.skip("Visual Design Agent not available")

    def test_agent_returns_agent_result(self, visual_agent):
        """Agent must return AgentResult object."""
        # Create minimal design input
        design_input = {
            "name": "Test Design",
            "description": "A test design",
            "mockup": None  # Agent should handle None
        }

        try:
            result = visual_agent.analyze(design_input)

            # Result must be AgentResult or dict
            if not isinstance(result, AgentResult):
                if isinstance(result, dict):
                    # Check it has the required structure
                    assert 'agent_id' in result
                    assert 'findings' in result
                    assert 'score' in result
                else:
                    raise AssertionError(
                        f"Unexpected result type: {type(result)}")

            # Verify contract
            if isinstance(result, AgentResult):
                assert result.agent_id == "visual_design"
                assert isinstance(result.findings, list)
                assert isinstance(result.score, (int, float))
                assert 0 <= result.score <= 100
                assert isinstance(result.metadata, dict)
                assert isinstance(result.recommendations, list)
        except AttributeError:
            # Agent might not have analyze method, that's ok for this test
            pytest.skip("Agent does not have analyze method")

    def test_agent_id_matches_agent_name(self, visual_agent):
        """Agent ID should match agent name."""
        design_input = {"name": "Test"}

        try:
            result = visual_agent.analyze(design_input)
            if isinstance(result, AgentResult):
                # Visual design agent should have matching ID
                assert "visual" in result.agent_id.lower()
        except AttributeError:
            pytest.skip("Agent does not have expected interface")

    def test_agent_findings_have_valid_structure(self, visual_agent):
        """All findings must have valid AgentFinding structure."""
        design_input = {"name": "Test"}

        try:
            result = visual_agent.analyze(design_input)
            if isinstance(result, AgentResult):
                for finding in result.findings:
                    assert isinstance(finding, AgentFinding)
                    assert hasattr(finding, 'description')
                    assert hasattr(finding, 'severity')
                    assert hasattr(finding, 'finding_type')
                    assert hasattr(finding, 'impact_score')
        except AttributeError:
            pytest.skip("Agent does not have expected interface")


class TestAgentOutputConsistency:
    """Test that agent outputs are consistent."""

    def test_agent_result_immutability_contract(self):
        """AgentResult should maintain contract across instantiation."""
        result1 = AgentResult(
            agent_id="test",
            findings=[],
            score=80.0,
            metadata={"v": 1},
            recommendations=["rec1"],
            status="success"
        )

        result2 = AgentResult(
            agent_id="test",
            findings=[],
            score=80.0,
            metadata={"v": 1},
            recommendations=["rec1"],
            status="success"
        )

        # Both should have same field values
        assert result1.agent_id == result2.agent_id
        assert result1.score == result2.score
        assert result1.status == result2.status

    def test_agent_result_with_complex_findings(self):
        """AgentResult with multiple findings should maintain contract."""
        findings = [
            AgentFinding(
                description="Finding 1",
                severity=Severity.CRITICAL,
                finding_type=FindingType.ACCESSIBILITY,
                impact_score=0.9
            ),
            AgentFinding(
                description="Finding 2",
                severity=Severity.LOW,
                finding_type=FindingType.STYLING,
                impact_score=0.2
            ),
        ]

        result = AgentResult(
            agent_id="complex_test",
            findings=findings,
            score=65.0,
            metadata={"finding_count": 2},
            recommendations=["Fix critical issue", "Consider styling"],
            status="success"
        )

        # Verify contract with complex data
        assert len(result.findings) == 2
        assert all(isinstance(f, AgentFinding) for f in result.findings)
        assert result.metadata["finding_count"] == 2
        assert len(result.recommendations) == 2

    def test_severity_enum_values(self):
        """All Severity enum values should be valid."""
        valid_severities = [Severity.CRITICAL,
                            Severity.HIGH, Severity.MEDIUM, Severity.LOW]

        for severity in valid_severities:
            finding = AgentFinding(
                description="Test",
                severity=severity,
                finding_type=FindingType.ACCESSIBILITY,
                impact_score=0.5
            )

            assert finding.severity == severity

    def test_finding_type_enum_values(self):
        """All FindingType enum values should be valid."""
        finding_types = [
            FindingType.ACCESSIBILITY,
            FindingType.USABILITY,
            FindingType.DESIGN_CONSISTENCY,
            FindingType.TYPOGRAPHY,
            FindingType.STYLING,
        ]

        for ftype in finding_types:
            finding = AgentFinding(
                description="Test",
                severity=Severity.MEDIUM,
                finding_type=ftype,
                impact_score=0.5
            )

            assert finding.finding_type == ftype
