"""
tests/conftest.py

Pytest configuration and shared fixtures for all tests.
Provides:
- Common test data fixtures
- Mock objects for testing
- Database setup/teardown
- Temporary file handling
"""

import pytest
import json
import tempfile
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Any
from unittest.mock import Mock, MagicMock

from components.models import (
    AgentFinding, AgentResult, FullReport,
    Severity, FindingType, RAGCitation
)


# ==================== Image Testing Fixtures ====================

@pytest.fixture
def temp_dir():
    """Create temporary directory for test files."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


@pytest.fixture
def sample_json_file(temp_dir):
    """Create a sample JSON file for testing."""
    data = {
        "name": "Test Design",
        "description": "A test design pattern",
        "rules": ["rule1", "rule2"]
    }
    file_path = temp_dir / "sample.json"
    with open(file_path, 'w') as f:
        json.dump(data, f)
    return file_path


# ==================== Model Fixtures ====================

@pytest.fixture
def sample_finding() -> AgentFinding:
    """Create a sample AgentFinding."""
    return AgentFinding(
        description="Sample accessibility finding",
        severity=Severity.MEDIUM,
        finding_type=FindingType.ACCESSIBILITY,
        impact_score=0.6
    )


@pytest.fixture
def sample_findings() -> List[AgentFinding]:
    """Create a list of sample findings."""
    return [
        AgentFinding(
            description="Critical accessibility issue",
            severity=Severity.CRITICAL,
            finding_type=FindingType.ACCESSIBILITY,
            impact_score=0.9
        ),
        AgentFinding(
            description="Minor styling issue",
            severity=Severity.LOW,
            finding_type=FindingType.STYLING,
            impact_score=0.2
        ),
        AgentFinding(
            description="Design consistency problem",
            severity=Severity.HIGH,
            finding_type=FindingType.DESIGN_CONSISTENCY,
            impact_score=0.7
        ),
    ]


@pytest.fixture
def sample_agent_result() -> AgentResult:
    """Create a sample AgentResult."""
    return AgentResult(
        agent_id="test_agent",
        findings=[],
        score=75.0,
        metadata={
            "timestamp": datetime.now().isoformat(),
            "version": "1.0"
        },
        recommendations=[
            "Improve contrast ratio",
            "Add alternative text"
        ],
        status="success"
    )


@pytest.fixture
def successful_agent_results() -> List[AgentResult]:
    """Create a list of successful agent results."""
    return [
        AgentResult(
            agent_id="visual_design",
            findings=[],
            score=85.0,
            metadata={"engine": "visual_v1"},
            recommendations=["Improve spacing"],
            status="success"
        ),
        AgentResult(
            agent_id="ux_critique",
            findings=[],
            score=80.0,
            metadata={"engine": "ux_v1"},
            recommendations=["Better navigation"],
            status="success"
        ),
        AgentResult(
            agent_id="market_research",
            findings=[],
            score=75.0,
            metadata={"engine": "market_v1"},
            recommendations=["Target younger audience"],
            status="success"
        ),
    ]


@pytest.fixture
def mixed_agent_results() -> List[AgentResult]:
    """Create agent results with mixed success/failure."""
    return [
        AgentResult(
            agent_id="visual_design",
            findings=[],
            score=85.0,
            metadata={},
            recommendations=[],
            status="success"
        ),
        AgentResult(
            agent_id="ux_critique",
            findings=[],
            score=0.0,
            metadata={"error": "Analysis timeout"},
            recommendations=[],
            status="error"
        ),
        AgentResult(
            agent_id="market_research",
            findings=[],
            score=75.0,
            metadata={},
            recommendations=[],
            status="success"
        ),
    ]


@pytest.fixture
def sample_full_report() -> FullReport:
    """Create a sample FullReport."""
    return FullReport(
        design_name="Test Design",
        agent_results=[
            AgentResult(
                agent_id="visual_design",
                findings=[],
                score=85.0,
                metadata={},
                recommendations=[],
                status="success"
            ),
            AgentResult(
                agent_id="ux_critique",
                findings=[],
                score=80.0,
                metadata={},
                recommendations=[],
                status="success"
            ),
        ],
        overall_score=82.5,
        timestamp=datetime.now().isoformat(),
        session_id="test-session-123"
    )


# ==================== RAG Testing Fixtures ====================

@pytest.fixture
def sample_rag_citation() -> RAGCitation:
    """Create a sample RAG citation."""
    return RAGCitation(
        source="design_patterns.json",
        pattern_name="Minimalist Design",
        relevance_score=0.95,
        section="design_principles"
    )


@pytest.fixture
def sample_citations() -> List[RAGCitation]:
    """Create a list of sample citations."""
    return [
        RAGCitation(
            source="design_patterns.json",
            pattern_name="Minimalist Design",
            relevance_score=0.95,
            section="design_principles"
        ),
        RAGCitation(
            source="design_patterns.json",
            pattern_name="Color Theory",
            relevance_score=0.87,
            section="color_palettes"
        ),
        RAGCitation(
            source="best_practices.json",
            pattern_name="Typography Standards",
            relevance_score=0.82,
            section="typography"
        ),
    ]


# ==================== Mock Fixtures ====================

@pytest.fixture
def mock_agent():
    """Create a mock agent."""
    agent = Mock()
    agent.analyze = Mock(return_value=AgentResult(
        agent_id="mock_agent",
        findings=[],
        score=75.0,
        metadata={},
        recommendations=[],
        status="success"
    ))
    return agent


@pytest.fixture
def mock_rag_service():
    """Create a mock RAG service."""
    service = Mock()
    service.retrieve = Mock(return_value=[])
    service.embed = Mock(return_value=[0.1] * 768)  # 768-dim embedding
    return service


@pytest.fixture
def mock_scoring_service():
    """Create a mock scoring service."""
    service = Mock()
    service.calculate_score = Mock(return_value=75.0)
    return service


@pytest.fixture
def mock_storage_service(temp_dir):
    """Create a mock storage service."""
    service = Mock()
    service.save = Mock(return_value=str(temp_dir / "report.json"))
    service.load = Mock(return_value={})
    service.list_history = Mock(return_value=[])
    return service


# ==================== Design Input Fixtures ====================

@pytest.fixture
def sample_design_input() -> Dict[str, Any]:
    """Create a sample design input."""
    return {
        "name": "Modern Minimalist App",
        "description": "A clean, modern app design with minimalist principles",
        "mockup": None,  # Could be image data in real tests
        "metadata": {
            "industry": "SaaS",
            "target_audience": "professionals"
        }
    }


@pytest.fixture
def minimal_design_input() -> Dict[str, Any]:
    """Create minimal design input."""
    return {
        "name": "Test Design",
        "description": "A test design"
    }


# ==================== Configuration Fixtures ====================

@pytest.fixture
def test_env_vars(monkeypatch):
    """Set up test environment variables."""
    monkeypatch.setenv("APP_ENV", "test")
    monkeypatch.setenv("DEBUG", "true")
    monkeypatch.setenv("AUTH_ENABLED", "false")
    monkeypatch.setenv("STORAGE_BACKEND", "json")
    return monkeypatch


# ==================== Pytest Hooks ====================

def pytest_configure(config):
    """Configure pytest."""
    # Register custom markers
    config.addinivalue_line(
        "markers", "unit: mark test as a unit test"
    )
    config.addinivalue_line(
        "markers", "integration: mark test as an integration test"
    )
    config.addinivalue_line(
        "markers", "slow: mark test as slow"
    )
    config.addinivalue_line(
        "markers", "contract: mark test as a contract test"
    )


# ==================== Test Utilities ====================

@pytest.fixture
def assert_valid_score():
    """Helper function to assert valid score."""
    def _assert(score):
        assert isinstance(score, (int, float)
                          ), f"Score must be numeric, got {type(score)}"
        assert 0 <= score <= 100, f"Score must be in [0, 100], got {score}"
        return True
    return _assert


@pytest.fixture
def assert_valid_finding():
    """Helper function to assert valid finding."""
    def _assert(finding):
        assert hasattr(finding, 'description'), "Finding must have description"
        assert hasattr(finding, 'severity'), "Finding must have severity"
        assert hasattr(
            finding, 'finding_type'), "Finding must have finding_type"
        assert hasattr(
            finding, 'impact_score'), "Finding must have impact_score"
        assert 0 <= finding.impact_score <= 1, f"Impact score must be in [0, 1], got {finding.impact_score}"
        return True
    return _assert


@pytest.fixture
def assert_valid_agent_result():
    """Helper function to assert valid agent result."""
    def _assert(result):
        assert isinstance(
            result, AgentResult), f"Result must be AgentResult, got {type(result)}"
        assert isinstance(result.agent_id, str), "Agent ID must be string"
        assert isinstance(result.findings, list), "Findings must be list"
        assert isinstance(result.score, (int, float)), "Score must be numeric"
        assert 0 <= result.score <= 100, "Score must be in [0, 100]"
        assert isinstance(result.metadata, dict), "Metadata must be dict"
        assert isinstance(result.recommendations,
                          list), "Recommendations must be list"
        assert result.status in ["success", "partial",
                                 "error", "pending"], "Status must be valid"
        return True
    return _assert
