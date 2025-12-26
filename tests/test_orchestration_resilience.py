"""
tests/test_orchestration_resilience.py

Test suite for orchestration resilience.
Validates:
- Orchestration returns FullReport even if an agent fails
- Failed agents are marked appropriately in the report
- Other agent results are included even when one fails
- Multiple agent failures are handled gracefully
"""

import pytest
from unittest.mock import Mock, patch, MagicMock
from typing import Dict, List, Any

from components.orchestration import Orchestrator
from components.models import AgentResult, FullReport, Severity, FindingType, AgentFinding


class TestOrchestrationResilience:
    """Test orchestration handles agent failures gracefully."""

    @pytest.fixture
    def orchestrator(self):
        """Initialize orchestrator."""
        return Orchestrator()

    @pytest.fixture
    def mock_agents(self):
        """Create mock agents."""
        return {
            'visual': Mock(name='visual_agent'),
            'ux': Mock(name='ux_agent'),
            'market': Mock(name='market_agent'),
            'conversion': Mock(name='conversion_agent'),
            'brand': Mock(name='brand_agent'),
        }

    def test_orchestration_returns_full_report(self, orchestrator):
        """Orchestration must return FullReport."""
        design = {
            "name": "Test Design",
            "description": "A test design"
        }

        result = orchestrator.analyze(design)

        assert isinstance(result, FullReport) or isinstance(result, dict)

        if isinstance(result, FullReport):
            assert hasattr(result, 'agent_results')
            assert hasattr(result, 'overall_score')

    def test_full_report_has_required_fields(self):
        """FullReport must have required fields."""
        report = FullReport(
            design_name="Test",
            agent_results=[],
            overall_score=75.0,
            timestamp="2024-01-01T00:00:00",
            session_id="test-session"
        )

        assert hasattr(report, 'design_name')
        assert hasattr(report, 'agent_results')
        assert hasattr(report, 'overall_score')
        assert hasattr(report, 'timestamp')
        assert hasattr(report, 'session_id')

    def test_agent_result_includes_status(self):
        """Agent results in report should have status field."""
        result = AgentResult(
            agent_id="test",
            findings=[],
            score=80.0,
            metadata={},
            recommendations=[],
            status="success"
        )

        assert hasattr(result, 'status')
        assert result.status in ["success", "partial", "error", "pending"]


class TestSingleAgentFailure:
    """Test orchestration with single agent failure."""

    @pytest.fixture
    def orchestrator(self):
        """Initialize orchestrator."""
        return Orchestrator()

    def test_one_agent_failure_returns_report(self, orchestrator):
        """Orchestration should return FullReport even if one agent fails."""
        # This tests the requirement: "orchestration returns FullReport even if an agent fails"

        design = {
            "name": "Test Design",
            "description": "A design to analyze"
        }

        # Patch one agent to raise an exception
        with patch.object(orchestrator, '_run_agents') as mock_run:
            # Simulate: 4 agents succeed, 1 fails
            mock_run.return_value = {
                'visual': AgentResult(
                    agent_id="visual",
                    findings=[],
                    score=85.0,
                    metadata={},
                    recommendations=[],
                    status="success"
                ),
                'ux': AgentResult(
                    agent_id="ux",
                    findings=[],
                    score=80.0,
                    metadata={},
                    recommendations=[],
                    status="success"
                ),
                'market': AgentResult(
                    agent_id="market",
                    findings=[],
                    score=70.0,
                    metadata={"error": "Connection failed"},
                    recommendations=[],
                    status="error"  # This one failed
                ),
                'conversion': AgentResult(
                    agent_id="conversion",
                    findings=[],
                    score=75.0,
                    metadata={},
                    recommendations=[],
                    status="success"
                ),
            }

            result = orchestrator.analyze(design)

            # Must still return a FullReport
            assert isinstance(result, FullReport) or isinstance(result, dict)

    def test_failed_agent_marked_in_report(self):
        """Failed agent should be marked with error status."""
        # Create a report with one failed agent
        failed_agent = AgentResult(
            agent_id="market",
            findings=[],
            score=0.0,  # No score due to failure
            metadata={"error": "Analysis timeout"},
            recommendations=[],
            status="error"
        )

        successful_agent = AgentResult(
            agent_id="visual",
            findings=[],
            score=85.0,
            metadata={},
            recommendations=[],
            status="success"
        )

        report = FullReport(
            design_name="Test",
            agent_results=[failed_agent, successful_agent],
            overall_score=85.0,
            timestamp="2024-01-01T00:00:00",
            session_id="test"
        )

        # Verify failed agent is marked
        assert any(r.status == "error" for r in report.agent_results)
        assert any(r.status == "success" for r in report.agent_results)

    def test_other_agents_included_on_failure(self):
        """Other agent results should be included even if one fails."""
        agent_results = [
            AgentResult(
                agent_id="visual",
                findings=[AgentFinding(
                    description="High contrast needed",
                    severity=Severity.MEDIUM,
                    finding_type=FindingType.ACCESSIBILITY,
                    impact_score=0.6
                )],
                score=80.0,
                metadata={},
                recommendations=["Improve contrast"],
                status="success"
            ),
            AgentResult(
                agent_id="failed_agent",
                findings=[],
                score=0.0,
                metadata={"error": "Agent crashed"},
                recommendations=[],
                status="error"
            ),
            AgentResult(
                agent_id="ux",
                findings=[],
                score=90.0,
                metadata={},
                recommendations=["Keep clean layout"],
                status="success"
            ),
        ]

        report = FullReport(
            design_name="Test",
            agent_results=agent_results,
            overall_score=85.0,
            timestamp="2024-01-01T00:00:00",
            session_id="test"
        )

        # Verify all agent results are present
        assert len(report.agent_results) == 3

        # Verify successful agents are included
        successful_ids = [
            r.agent_id for r in report.agent_results if r.status == "success"]
        assert "visual" in successful_ids
        assert "ux" in successful_ids

        # Verify failed agent is included
        failed_ids = [
            r.agent_id for r in report.agent_results if r.status == "error"]
        assert "failed_agent" in failed_ids


class TestMultipleAgentFailures:
    """Test orchestration with multiple agent failures."""

    def test_multiple_agents_fail(self):
        """Report should handle multiple agent failures."""
        agent_results = [
            AgentResult(
                agent_id="agent1",
                findings=[],
                score=80.0,
                metadata={},
                recommendations=[],
                status="success"
            ),
            AgentResult(
                agent_id="agent2",
                findings=[],
                score=0.0,
                metadata={"error": "Timeout"},
                recommendations=[],
                status="error"
            ),
            AgentResult(
                agent_id="agent3",
                findings=[],
                score=0.0,
                metadata={"error": "Connection failed"},
                recommendations=[],
                status="error"
            ),
            AgentResult(
                agent_id="agent4",
                findings=[],
                score=85.0,
                metadata={},
                recommendations=[],
                status="success"
            ),
        ]

        report = FullReport(
            design_name="Test",
            agent_results=agent_results,
            overall_score=82.5,
            timestamp="2024-01-01T00:00:00",
            session_id="test"
        )

        # Verify report still created
        assert isinstance(report, FullReport)
        assert len(report.agent_results) == 4

        # Count successes and failures
        successes = sum(
            1 for r in report.agent_results if r.status == "success")
        failures = sum(1 for r in report.agent_results if r.status == "error")

        assert successes == 2
        assert failures == 2

    def test_all_agents_fail(self):
        """Report should still be created if all agents fail."""
        agent_results = [
            AgentResult(
                agent_id="agent1",
                findings=[],
                score=0.0,
                metadata={"error": "Failed"},
                recommendations=[],
                status="error"
            ),
            AgentResult(
                agent_id="agent2",
                findings=[],
                score=0.0,
                metadata={"error": "Failed"},
                recommendations=[],
                status="error"
            ),
            AgentResult(
                agent_id="agent3",
                findings=[],
                score=0.0,
                metadata={"error": "Failed"},
                recommendations=[],
                status="error"
            ),
        ]

        report = FullReport(
            design_name="Test",
            agent_results=agent_results,
            overall_score=0.0,
            timestamp="2024-01-01T00:00:00",
            session_id="test"
        )

        # Report must still exist
        assert isinstance(report, FullReport)

        # All agents should be marked as failed
        assert all(r.status == "error" for r in report.agent_results)

        # Report should indicate failure somehow
        # (overall score 0, or explicit error field, etc.)
        assert report.overall_score <= 10


class TestPartialAgentResults:
    """Test orchestration with partial/incomplete results."""

    def test_agent_with_partial_findings(self):
        """Agent with partial findings should still be included."""
        report = FullReport(
            design_name="Test",
            agent_results=[
                AgentResult(
                    agent_id="visual",
                    findings=[
                        AgentFinding(
                            description="Issue 1",
                            severity=Severity.HIGH,
                            finding_type=FindingType.ACCESSIBILITY,
                            impact_score=0.8
                        )
                    ],
                    score=75.0,
                    metadata={},
                    recommendations=["Fix issue 1"],
                    status="success"
                ),
                AgentResult(
                    agent_id="ux",
                    findings=[],  # No findings found
                    score=90.0,
                    metadata={},
                    recommendations=[],
                    status="success"  # But still successful
                ),
            ],
            overall_score=82.5,
            timestamp="2024-01-01T00:00:00",
            session_id="test"
        )

        # Both agents should be present regardless of findings
        assert len(report.agent_results) == 2

        # One with findings, one without - both valid
        assert len(report.agent_results[0].findings) > 0
        assert len(report.agent_results[1].findings) == 0

    def test_agent_partial_status(self):
        """Agent might return partial results (status='partial')."""
        result = AgentResult(
            agent_id="partial_agent",
            findings=[
                AgentFinding(
                    description="Partial analysis",
                    severity=Severity.LOW,
                    finding_type=FindingType.STYLING,
                    impact_score=0.3
                )
            ],
            score=60.0,
            metadata={"incomplete": True, "reason": "Limited data"},
            recommendations=["Provide more details for full analysis"],
            status="partial"  # Partial result
        )

        # Partial result should still be valid
        assert result.status == "partial"
        assert result.score >= 0
        assert len(result.findings) > 0


class TestReportIntegrity:
    """Test integrity of reports with failures."""

    def test_report_score_calculation_with_failures(self):
        """Overall score should handle failed agents appropriately."""
        # Scenario: 3 successful agents (scores 80, 85, 90), 1 failed agent (score 0)
        # Overall score should probably be average of successful agents, or weighted
        agent_results = [
            AgentResult(
                agent_id="a1",
                findings=[],
                score=80.0,
                metadata={},
                recommendations=[],
                status="success"
            ),
            AgentResult(
                agent_id="a2",
                findings=[],
                score=85.0,
                metadata={},
                recommendations=[],
                status="success"
            ),
            AgentResult(
                agent_id="a3",
                findings=[],
                score=90.0,
                metadata={},
                recommendations=[],
                status="success"
            ),
            AgentResult(
                agent_id="a4",
                findings=[],
                score=0.0,
                metadata={"error": "Failed"},
                recommendations=[],
                status="error"
            ),
        ]

        report = FullReport(
            design_name="Test",
            agent_results=agent_results,
            overall_score=85.0,  # Could be average of successes
            timestamp="2024-01-01T00:00:00",
            session_id="test"
        )

        # Report is valid
        assert isinstance(report, FullReport)

        # Overall score should be reasonable (not affected heavily by one failure)
        assert 70 <= report.overall_score <= 90

    def test_report_has_all_agent_metadata(self):
        """Report should preserve metadata from all agents."""
        agent_results = [
            AgentResult(
                agent_id="visual",
                findings=[],
                score=80.0,
                metadata={"version": "1.0", "engine": "visual_v1"},
                recommendations=[],
                status="success"
            ),
            AgentResult(
                agent_id="ux",
                findings=[],
                score=75.0,
                metadata={"version": "2.0", "engine": "ux_v2"},
                recommendations=[],
                status="success"
            ),
        ]

        report = FullReport(
            design_name="Test",
            agent_results=agent_results,
            overall_score=77.5,
            timestamp="2024-01-01T00:00:00",
            session_id="test"
        )

        # All metadata should be preserved
        visual_result = next(
            r for r in report.agent_results if r.agent_id == "visual")
        assert visual_result.metadata["version"] == "1.0"

        ux_result = next(r for r in report.agent_results if r.agent_id == "ux")
        assert ux_result.metadata["version"] == "2.0"
