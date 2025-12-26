"""
Pydantic models for type-safe data structures across the application.
"""

from pydantic import BaseModel, Field
from typing import List, Dict, Optional, Any, Literal
from datetime import datetime
from enum import Enum


# ============================================================================
# Input Models
# ============================================================================

class DesignInput(BaseModel):
    """Input design file information."""
    id: str = Field(..., description="Unique design ID")
    filename: str = Field(..., description="Original filename")
    file_hash: str = Field(..., description="SHA256 hash of file bytes")
    image_base64: str = Field(..., description="Base64-encoded image data")
    platform: str = Field(
        ..., description="Target platform (Instagram, Facebook, LinkedIn, Twitter, Pinterest)")
    creative_type: str = Field(
        ..., description="Type of creative (Marketing Creative, Product UI/App Screen, etc)")
    metadata: Dict[str, Any] = Field(
        default_factory=dict, description="Image metadata (width, height, aspect_ratio, format)")
    is_cached: bool = Field(
        default=False, description="Whether result was retrieved from cache")
    cache_key: Optional[str] = Field(
        None, description="Cache key for this design analysis")

    class Config:
        json_schema_extra = {
            "example": {
                "id": "design_001",
                "filename": "post.png",
                "file_hash": "abc123...",
                "image_base64": "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNk+M9QDwADhgGAWjR9awAAAABJRU5ErkJggg==",
                "platform": "Instagram",
                "creative_type": "Marketing Creative",
                "metadata": {"width": 1080, "height": 1350, "aspect_ratio": 0.8, "format": "PNG"},
                "is_cached": False,
                "cache_key": None
            }
        }


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


class AgentFinding(BaseModel):
    """Individual finding from an agent analysis."""
    category: str = Field(
        ..., description="Finding category (e.g., 'Color Harmony', 'CTA Visibility')")
    severity: SeverityLevel = Field(...,
                                    description="Severity level of this finding")
    evidence: str = Field(...,
                          description="Evidence/description of the finding")
    recommendation: str = Field(..., description="Recommended action")
    expected_impact: Optional[str] = Field(
        None, description="Expected impact if recommendation is implemented (e.g., '+15% engagement')")
    confidence: float = Field(
        default=0.8, ge=0.0, le=1.0, description="Confidence level (0-1)")
    supporting_data: Optional[Dict[str, Any]] = Field(
        None, description="Additional data supporting the finding")


class AgentResult(BaseModel):
    """Result from a single agent evaluation."""
    agent_name: str = Field(
        ..., description="Name of the agent (visual, ux, market, conversion, brand)")
    summary: str = Field(..., description="High-level summary of findings")
    findings: List[AgentFinding] = Field(
        default_factory=list, description="Detailed findings from this agent")
    score: float = Field(..., ge=0.0, le=100.0,
                         description="Overall score for this agent (0-100)")
    subscores: Dict[str, float] = Field(
        default_factory=dict, description="Subscores by category (e.g., color: 85, layout: 92)")
    errors: List[str] = Field(default_factory=list,
                              description="Errors encountered during analysis")
    raw_response: Optional[Dict[str, Any]] = Field(
        None, description="Raw API response for debugging")
    latency_ms: float = Field(
        default=0.0, description="API call latency in milliseconds")
    tokens_used: Optional[Dict[str, int]] = Field(
        None, description="Token usage (input, output)")
    was_cached: bool = Field(
        default=False, description="Whether this result came from cache")
    cache_age_seconds: Optional[float] = Field(
        None, description="Age of cached result in seconds")

    class Config:
        json_schema_extra = {
            "example": {
                "agent_name": "visual",
                "summary": "Strong color harmony and layout, minor typography improvements needed",
                "findings": [
                    {
                        "category": "Color Harmony",
                        "severity": "info",
                        "evidence": "Palette uses complementary colors effectively",
                        "recommendation": "Maintain current palette for brand consistency",
                        "expected_impact": None,
                        "confidence": 0.95
                    }
                ],
                "score": 82.5,
                "subscores": {"color": 85, "layout": 80, "typography": 82},
                "errors": [],
                "latency_ms": 1250.5,
                "tokens_used": {"input": 1500, "output": 200}
            }
        }


# ============================================================================
# Comparison Models
# ============================================================================

class DesignRanking(BaseModel):
    """Ranking of a single design in comparison."""
    design_id: str = Field(..., description="ID of the design")
    rank: int = Field(..., ge=1, description="Rank position (1 = best)")
    overall_score: float = Field(..., ge=0.0, le=100.0)
    visual_score: float = Field(..., ge=0.0, le=100.0)
    ux_score: float = Field(..., ge=0.0, le=100.0)
    market_score: float = Field(..., ge=0.0, le=100.0)
    conversion_score: float = Field(..., ge=0.0, le=100.0)
    brand_score: float = Field(..., ge=0.0, le=100.0)


class DesignDifference(BaseModel):
    """Key difference between designs."""
    aspect: str = Field(
        ..., description="Aspect being compared (e.g., 'Color Palette', 'CTA Clarity')")
    winner: str = Field(..., description="ID of winning design")
    loser: str = Field(..., description="ID of losing design")
    reason: str = Field(..., description="Explanation of difference")
    impact: str = Field(..., description="Significance of the difference")


class ABTestPlan(BaseModel):
    """A/B test recommendation for comparison."""
    test_type: str = Field(...,
                           description="Type of test (e.g., 'Color Variation', 'CTA Button')")
    variant_a: str = Field(..., description="Design ID for variant A")
    variant_b: str = Field(..., description="Design ID for variant B")
    duration_days: int = Field(
        default=14, ge=1, description="Recommended test duration in days")
    sample_size: int = Field(..., description="Recommended sample size")
    predicted_winner: str = Field(...,
                                  description="Predicted winner based on analysis")
    confidence_percentage: float = Field(..., ge=0.0,
                                         le=100.0, description="Confidence in prediction")
    success_metric: str = Field(...,
                                description="Primary success metric to measure")
    expected_lift: str = Field(
        default="", description="Expected improvement (e.g., '+5% CTR')")


class ComparisonResult(BaseModel):
    """Result from comparing multiple designs."""
    design_ids: List[str] = Field(...,
                                  description="IDs of all designs compared")
    rankings: List[DesignRanking] = Field(...,
                                          description="Rankings from best to worst")
    similarity_matrix: Dict[str, Dict[str, float]
                            ] = Field(..., description="Pairwise similarity scores (0-1)")
    key_differences: List[DesignDifference] = Field(
        default_factory=list, description="Major differences between designs")
    synthesis_recommendation: str = Field(
        ..., description="Recommendation for hybrid approach or winner selection")
    ab_test_plans: List[ABTestPlan] = Field(
        default_factory=list, description="Recommended A/B tests")
    composite_image_base64: Optional[str] = Field(
        None, description="Side-by-side comparison image as base64")


# ============================================================================
# Score Card Models
# ============================================================================

class ScoreCardDeltas(BaseModel):
    """Score changes/trends."""
    visual_delta: float = Field(
        default=0.0, description="Change in visual score")
    ux_delta: float = Field(default=0.0, description="Change in UX score")
    market_delta: float = Field(
        default=0.0, description="Change in market score")
    conversion_delta: float = Field(
        default=0.0, description="Change in conversion score")
    brand_delta: float = Field(
        default=0.0, description="Change in brand score")
    overall_delta: float = Field(
        default=0.0, description="Change in overall score")


class ScoreCard(BaseModel):
    """Summary scorecard for design."""
    overall: float = Field(..., ge=0.0, le=100.0, description="Overall score")
    visual: float = Field(..., ge=0.0, le=100.0,
                          description="Visual analysis score")
    ux: float = Field(..., ge=0.0, le=100.0, description="UX critique score")
    market: float = Field(..., ge=0.0, le=100.0,
                          description="Market research score")
    conversion: float = Field(..., ge=0.0, le=100.0,
                              description="Conversion optimization score")
    brand: float = Field(..., ge=0.0, le=100.0,
                         description="Brand consistency score")
    deltas: Optional[ScoreCardDeltas] = Field(
        None, description="Score changes from previous version")
    timestamp: datetime = Field(default_factory=datetime.utcnow)


# ============================================================================
# Visual Feedback Models
# ============================================================================

class Annotation(BaseModel):
    """Single annotation on a design."""
    id: str = Field(..., description="Unique annotation ID")
    x: float = Field(..., ge=0.0, le=1.0,
                     description="X position as fraction of image width")
    y: float = Field(..., ge=0.0, le=1.0,
                     description="Y position as fraction of image height")
    width: float = Field(default=0.1, ge=0.0, le=1.0,
                         description="Annotation width as fraction")
    height: float = Field(default=0.1, ge=0.0, le=1.0,
                          description="Annotation height as fraction")
    label: str = Field(..., description="Label text")
    category: str = Field(...,
                          description="Category (issue, strength, suggestion)")
    severity: SeverityLevel = Field(..., description="Severity of the issue")
    color: str = Field(default="#FF0000",
                       description="Hex color for annotation")


class VisualFeedback(BaseModel):
    """Visual feedback with annotations and mockups."""
    annotated_image_base64: Optional[str] = Field(
        None, description="Image with numbered annotations as base64")
    annotations: List[Annotation] = Field(
        default_factory=list, description="List of annotations")
    before_image_base64: Optional[str] = Field(
        None, description="Original design as base64")
    after_image_base64: Optional[str] = Field(
        None, description="Simulated improvements as base64")
    heatmap_image_base64: Optional[str] = Field(
        None, description="Attention heatmap as base64")
    heatmap_type: Literal["f_pattern", "z_pattern", "density"] = Field(
        default="f_pattern", description="Type of heatmap")
    color_palette_image_base64: Optional[str] = Field(
        None, description="Color palette visualization as base64")
    detected_colors: List[str] = Field(
        default_factory=list, description="Detected hex colors from design")
    recommended_colors: List[str] = Field(
        default_factory=list, description="Recommended hex colors")


# ============================================================================
# Report Models
# ============================================================================

class RAGCitation(BaseModel):
    """Citation of a retrieved design pattern."""
    pattern_id: str = Field(..., description="ID of the design pattern")
    pattern_title: str = Field(..., description="Title of the pattern")
    category: str = Field(..., description="Pattern category")
    relevance_score: float = Field(..., ge=0.0, le=1.0,
                                   description="Relevance to analysis (0-1)")
    used_by: List[str] = Field(
        default_factory=list, description="Agent names that used this pattern")


class FullReport(BaseModel):
    """Complete analysis report."""
    report_id: str = Field(..., description="Unique report ID")
    created_at: datetime = Field(default_factory=datetime.utcnow)
    app_version: str = Field(
        default="0.1.0", description="Application version")

    # Inputs
    design_inputs: List[DesignInput] = Field(...,
                                             description="Input designs analyzed")
    analysis_mode: Literal["single",
                           "compare"] = Field(..., description="Analysis mode used")

    # Results
    scores: ScoreCard = Field(..., description="Score summary")
    agent_results: List[AgentResult] = Field(...,
                                             description="Results from all agents")
    comparison_result: Optional[ComparisonResult] = Field(
        None, description="Comparison results if compare mode")

    # Visual feedback
    visual_feedback: VisualFeedback = Field(...,
                                            description="Visual feedback and annotations")

    # RAG context
    rag_citations: List[RAGCitation] = Field(
        default_factory=list, description="Design patterns used in analysis")
    rag_top_k: int = Field(
        default=3, description="Number of RAG patterns retrieved")

    # Metadata
    platform: str = Field(..., description="Target platform")
    creative_type: str = Field(..., description="Creative type")
    enabled_agents: List[str] = Field(...,
                                      description="Agents that were enabled")
    execution_mode: str = Field(
        default="standard", description="Execution mode (fast/standard/comprehensive)")

    # Charts data (for easy frontend rendering)
    charts_data: Dict[str, Any] = Field(
        default_factory=dict, description="Pre-computed chart data")

    # Recommendations
    top_recommendations: List[Dict[str, Any]] = Field(
        default_factory=list, description="Top 10 prioritized recommendations")

    # Processing info
    total_latency_ms: float = Field(
        default=0.0, description="Total processing time in milliseconds")
    total_tokens_used: Optional[Dict[str, int]] = Field(
        None, description="Total tokens used across all agents")
    estimated_cost_usd: Optional[float] = Field(
        None, description="Estimated cost in USD")

    # Cache statistics
    cache_stats: Dict[str, Any] = Field(
        default_factory=dict, description="Cache hit/miss statistics")
    num_cached_results: int = Field(
        default=0, description="Number of results retrieved from cache")


# ============================================================================
# Session & Cost Tracking Models
# ============================================================================

class TokenUsage(BaseModel):
    """Token usage from a single API call."""
    input_tokens: int = Field(default=0)
    output_tokens: int = Field(default=0)
    total_tokens: int = Field(default=0)


class APICallMetrics(BaseModel):
    """Metrics from a single API call."""
    model: str = Field(...)
    agent: str = Field(...)
    latency_ms: float = Field(...)
    tokens: Optional[TokenUsage] = Field(None)
    cost_usd: float = Field(default=0.0)
    timestamp: datetime = Field(default_factory=datetime.utcnow)


class SessionMetrics(BaseModel):
    """Aggregated metrics for an analysis session."""
    session_id: str = Field(...)
    api_calls: List[APICallMetrics] = Field(default_factory=list)

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
