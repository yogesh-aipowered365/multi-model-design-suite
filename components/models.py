"""
Data models for type-safe data structures across the application.
Uses Python dataclasses instead of Pydantic to avoid compilation issues on Streamlit Cloud.
"""

from dataclasses import dataclass, field
from typing import List, Dict, Optional, Any, Literal
from datetime import datetime
from enum import Enum


# ============================================================================
# Input Models
# ============================================================================

@dataclass
class DesignInput:
    """Input design file information."""
    id: str
    filename: str
    file_hash: str
    image_base64: str
    platform: str
    creative_type: str
    metadata: Dict[str, Any] = field(default_factory=dict)
    is_cached: bool = False
    cache_key: Optional[str] = None


# ============================================================================
# Agent Finding & Result Models
# ============================================================================

class AnalysisMode(str, Enum):
    """Analysis modes with different token/cost profiles."""
    FAST = "fast"
    STANDARD = "standard"
    COMPREHENSIVE = "comprehensive"


class SeverityLevel(str, Enum):
    """Severity levels for findings."""
    CRITICAL = "critical"
    WARNING = "warning"
    INFO = "info"


class PriorityLevel(str, Enum):
    """Priority levels for recommendations."""
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"


@dataclass
class AgentFinding:
    """Individual finding from an agent analysis."""
    category: str
    severity: SeverityLevel
    evidence: str
    recommendation: str
    confidence: float = 0.8
    expected_impact: Optional[str] = None
    supporting_data: Optional[Dict[str, Any]] = None


@dataclass
class AgentResult:
    """Result from a single agent evaluation."""
    agent_name: str
    summary: str
    score: float
    findings: List[AgentFinding] = field(default_factory=list)
    subscores: Dict[str, float] = field(default_factory=dict)
    errors: List[str] = field(default_factory=list)
    raw_response: Optional[Dict[str, Any]] = None
    latency_ms: float = 0.0
    tokens_used: Optional[Dict[str, int]] = None
    was_cached: bool = False
    cache_age_seconds: Optional[float] = None


# ============================================================================
# Comparison Models
# ============================================================================

@dataclass
class DesignRanking:
    """Ranking of a single design in comparison."""
    design_id: str
    rank: int
    overall_score: float
    visual_score: float
    ux_score: float
    market_score: float
    conversion_score: float
    brand_score: float


@dataclass
class DesignDifference:
    """Key difference between designs."""
    aspect: str
    winner: str
    loser: str
    reason: str
    impact: str


@dataclass
class ABTestPlan:
    """A/B test recommendation for comparison."""
    test_type: str
    variant_a: str
    variant_b: str
    sample_size: int
    predicted_winner: str
    confidence_percentage: float
    success_metric: str
    duration_days: int = 14
    expected_lift: str = ""


@dataclass
class ComparisonResult:
    """Result from comparing multiple designs."""
    design_ids: List[str]
    rankings: List[DesignRanking]
    similarity_matrix: Dict[str, Dict[str, float]]
    synthesis_recommendation: str
    key_differences: List[DesignDifference] = field(default_factory=list)
    ab_test_plans: List[ABTestPlan] = field(default_factory=list)
    composite_image_base64: Optional[str] = None


# ============================================================================
# Score Card Models
# ============================================================================

@dataclass
class ScoreCardDeltas:
    """Score changes/trends."""
    visual_delta: float = 0.0
    ux_delta: float = 0.0
    market_delta: float = 0.0
    conversion_delta: float = 0.0
    brand_delta: float = 0.0
    overall_delta: float = 0.0


@dataclass
class ScoreCard:
    """Summary scorecard for design."""
    overall: float
    visual: float
    ux: float
    market: float
    conversion: float
    brand: float
    deltas: Optional[ScoreCardDeltas] = None
    timestamp: datetime = field(default_factory=datetime.utcnow)


# ============================================================================
# Visual Feedback Models
# ============================================================================

@dataclass
class Annotation:
    """Single annotation on a design."""
    id: str
    x: float
    y: float
    label: str
    category: str
    severity: SeverityLevel
    width: float = 0.1
    height: float = 0.1
    color: str = "#FF0000"


@dataclass
class VisualFeedback:
    """Visual feedback with annotations and mockups."""
    annotations: List[Annotation] = field(default_factory=list)
    annotated_image_base64: Optional[str] = None
    before_image_base64: Optional[str] = None
    after_image_base64: Optional[str] = None
    heatmap_image_base64: Optional[str] = None
    heatmap_type: str = "f_pattern"
    color_palette_image_base64: Optional[str] = None
    detected_colors: List[str] = field(default_factory=list)
    recommended_colors: List[str] = field(default_factory=list)


# ============================================================================
# Report Models
# ============================================================================

@dataclass
class RAGCitation:
    """Citation of a retrieved design pattern."""
    pattern_id: str
    pattern_title: str
    category: str
    relevance_score: float
    used_by: List[str] = field(default_factory=list)


@dataclass
class FullReport:
    """Complete analysis report."""
    report_id: str
    design_inputs: List[DesignInput]
    analysis_mode: str
    scores: ScoreCard
    agent_results: List[AgentResult]
    visual_feedback: VisualFeedback
    platform: str
    creative_type: str
    enabled_agents: List[str]
    
    created_at: datetime = field(default_factory=datetime.utcnow)
    app_version: str = "0.1.0"
    comparison_result: Optional[ComparisonResult] = None
    rag_citations: List[RAGCitation] = field(default_factory=list)
    rag_top_k: int = 3
    execution_mode: str = "standard"
    charts_data: Dict[str, Any] = field(default_factory=dict)
    top_recommendations: List[Dict[str, Any]] = field(default_factory=list)
    total_latency_ms: float = 0.0
    total_tokens_used: Optional[Dict[str, int]] = None
    estimated_cost_usd: Optional[float] = None
    cache_stats: Dict[str, Any] = field(default_factory=dict)
    num_cached_results: int = 0


# ============================================================================
# Session & Cost Tracking Models
# ============================================================================

@dataclass
class TokenUsage:
    """Token usage from a single API call."""
    input_tokens: int = 0
    output_tokens: int = 0
    total_tokens: int = 0


@dataclass
class APICallMetrics:
    """Metrics from a single API call."""
    model: str
    agent: str
    latency_ms: float
    cost_usd: float = 0.0
    tokens: Optional[TokenUsage] = None
    timestamp: datetime = field(default_factory=datetime.utcnow)


@dataclass
class SessionMetrics:
    """Aggregated metrics for an analysis session."""
    session_id: str
    api_calls: List[APICallMetrics] = field(default_factory=list)
    @property
    def total_tokens(self) -> int:
        """Calculate total tokens used in session."""
        return sum(
            call.tokens.total_tokens for call in self.api_calls
            if call.tokens
        )

    @property
    def total_cost_usd(self) -> float:
        """Calculate total cost in session."""
        return sum(call.cost_usd for call in self.api_calls)

    @property
    def total_latency_ms(self) -> float:
        """Calculate total latency."""
        return sum(call.latency_ms for call in self.api_calls)
