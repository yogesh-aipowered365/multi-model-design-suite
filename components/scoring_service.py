# components/scoring_service.py

"""
Scoring Service: Deterministic rule-based scoring with WCAG compliance
Inputs: Agent results + heuristics
Outputs: ScoreCard, deltas, radar data, recommendations breakdown, WCAG compliance
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any
from enum import Enum
import json


class SeverityLevel(Enum):
    """Severity classification for findings"""
    CRITICAL = 4  # Blocks conversion/functionality
    HIGH = 3      # Significant impact
    MEDIUM = 2    # Noticeable impact
    LOW = 1       # Minor issue


class WCAGLevel(Enum):
    """WCAG Compliance levels"""
    A = 1
    AA = 2
    AAA = 3


@dataclass
class ScoreComponent:
    """Single score component"""
    name: str
    score: float  # 0-100
    weight: float  # 0.0-1.0
    findings_count: int = 0
    max_issues: int = 10  # For bounded scoring


@dataclass
class WCAGCompliance:
    """WCAG compliance status"""
    contrast_score: float  # Text contrast ratio (AA requires 4.5:1, AAA requires 7:1)
    touch_target_score: float  # Touch target size (48x48px minimum)
    screen_reader_score: float  # Screen reader support and ARIA
    keyboard_nav_score: float  # Full keyboard navigation
    color_blindness_score: float  # Doesn't rely only on color
    focus_indicators_score: float  # Clear focus indicators
    overall_wcag_score: float  # Average of all components
    level: WCAGLevel  # A, AA, or AAA
    issues: List[str] = field(default_factory=list)


@dataclass
class RecommendationBreakdown:
    """Recommendation distribution by category"""
    category: str
    count: int
    critical: int
    high: int
    medium: int
    low: int
    avg_impact_score: float


@dataclass
class RadarData:
    """Radar chart data structure"""
    categories: List[str]
    values: List[float]  # Scores 0-100
    targets: List[float]  # Target scores (usually 75)


@dataclass
class ScoreCard:
    """Complete scoring output"""
    overall_score: float  # 0-100
    target_score: float  # Usually 75
    delta: float  # overall_score - target_score (can be negative)

    # Category scores
    visual_score: float
    ux_score: float
    market_score: float
    conversion_score: float
    brand_score: float
    accessibility_score: float

    # Detailed components (for breakdown charts)
    components: Dict[str, float] = field(default_factory=dict)

    # Radar data (for visualization)
    radar: RadarData = field(default_factory=lambda: RadarData([], [], []))

    # WCAG compliance
    wcag: WCAGCompliance = field(
        default_factory=lambda: WCAGCompliance(0, 0, 0, 0, 0, 0, 0, WCAGLevel.A))

    # Recommendations by category (for donut chart)
    recommendations_breakdown: List[RecommendationBreakdown] = field(
        default_factory=list)

    # Metadata
    agent_errors: Dict[str, str] = field(default_factory=dict)
    methodology: str = "Deterministic rule-based scoring with bounded mapping"
    timestamp: str = ""


class ScoringRules:
    """
    Deterministic scoring rules using bounded mapping
    Finding count/severity → Score adjustment
    """

    # Score range for each number of findings (prevents random variation)
    SCORE_MAPPING = {
        # critical_count -> score_deduction
        # Each critical = -15 points
        "critical": {0: 0, 1: -15, 2: -25, 3: -35, 4: -45, 5: -50},
        # high_count -> score_deduction (accumulative)
        "high": {0: 0, 1: -3, 2: -6, 3: -10, 4: -13, 5: -15},
        # medium_count -> score_deduction
        "medium": {0: 0, 1: -2, 2: -4, 3: -6, 4: -8},
        # low_count -> score_deduction
        "low": {0: 0, 1: -1, 2: -2, 3: -3}
    }

    # Findings quality factors (0.8-1.0)
    QUALITY_BONUS = {
        "specific": 1.0,      # Concrete, actionable
        "vague": 0.85,        # General observation
        "missing": 0.70       # No finding provided
    }

    # Category weights (must sum to 1.0)
    CATEGORY_WEIGHTS = {
        "visual": 0.20,
        "ux": 0.20,
        "market": 0.15,
        "conversion": 0.25,
        "brand": 0.15,
        "accessibility": 0.05  # Bonus category
    }

    @staticmethod
    def count_findings_by_severity(findings: List[str]) -> Tuple[int, int, int, int]:
        """
        Count findings by severity level (heuristic detection)

        Args:
            findings: List of finding strings

        Returns:
            (critical, high, medium, low)
        """
        if not findings:
            return 0, 0, 0, 0

        critical_keywords = ["must", "blocks", "critical",
                             "fail", "error", "prevent", "cannot"]
        high_keywords = ["should", "significant",
                         "major", "severe", "impact", "issue"]
        medium_keywords = ["consider", "improve",
                           "enhance", "better", "could", "recommend"]
        low_keywords = ["minor", "small", "slight", "nice to have", "optional"]

        critical = 0
        high = 0
        medium = 0
        low = 0

        for finding in findings:
            finding_lower = finding.lower()

            # Check critical first (most severe)
            if any(kw in finding_lower for kw in critical_keywords):
                critical += 1
            elif any(kw in finding_lower for kw in high_keywords):
                high += 1
            elif any(kw in finding_lower for kw in medium_keywords):
                medium += 1
            elif any(kw in finding_lower for kw in low_keywords):
                low += 1
            else:
                low += 1  # Default to low if unclassified

        return critical, high, medium, low

    @staticmethod
    def calculate_component_score(
        base_score: float = 75.0,
        critical: int = 0,
        high: int = 0,
        medium: int = 0,
        low: int = 0,
        quality: str = "specific"
    ) -> float:
        """
        Calculate component score using bounded mapping

        Args:
            base_score: Starting score (usually provided by agent)
            critical, high, medium, low: Finding counts by severity
            quality: Finding quality ("specific", "vague", "missing")

        Returns:
            float: Score 0-100 (bounded)
        """
        # Start with provided score or default
        score = base_score if base_score > 0 else 75.0

        # Apply bounded adjustments for each severity level
        def apply_bounded_adjustment(count: int, severity_type: str) -> float:
            mapping = ScoringRules.SCORE_MAPPING.get(severity_type, {})
            # Cap count to max available in mapping
            capped_count = min(count, max(mapping.keys()))
            return mapping.get(capped_count, 0)

        # Deduct points for findings (order: critical first, more severe first)
        score += apply_bounded_adjustment(critical, "critical")
        score += apply_bounded_adjustment(high, "high")
        score += apply_bounded_adjustment(medium, "medium")
        score += apply_bounded_adjustment(low, "low")

        # Apply quality factor
        quality_factor = ScoringRules.QUALITY_BONUS.get(quality, 1.0)
        score = score * quality_factor

        # Bound to 0-100
        return max(0.0, min(100.0, score))


class ScoringService:
    """Main scoring service - deterministic rules-based"""

    @staticmethod
    def score_visual_analysis(agent_result: Dict) -> Tuple[float, Dict]:
        """
        Score visual analysis agent output
        Components: color, layout, typography

        Returns:
            (overall_score, detailed_scores)
        """
        if "error" in agent_result:
            return 0.0, {"error": agent_result["error"]}

        color = agent_result.get("color_analysis", {})
        layout = agent_result.get("layout_analysis", {})
        typography = agent_result.get("typography", {})

        # Extract scores or calculate from findings
        color_score = ScoringRules.calculate_component_score(
            base_score=color.get("score", 75),
            critical=0,
            high=0,
            medium=len(color.get("recommendations", [])),
            low=len(color.get("findings", [])) -
            len(color.get("recommendations", []))
        )

        layout_score = ScoringRules.calculate_component_score(
            base_score=layout.get("score", 75),
            critical=0,
            high=0,
            medium=len(layout.get("recommendations", [])),
            low=len(layout.get("findings", [])) -
            len(layout.get("recommendations", []))
        )

        typography_score = ScoringRules.calculate_component_score(
            base_score=typography.get("score", 75),
            critical=0,
            high=0,
            medium=len(typography.get("recommendations", [])),
            low=len(typography.get("findings", [])) -
            len(typography.get("recommendations", []))
        )

        overall = (color_score + layout_score + typography_score) / 3

        return overall, {
            "color": color_score,
            "layout": layout_score,
            "typography": typography_score,
            "overall": overall
        }

    @staticmethod
    def score_ux_analysis(agent_result: Dict) -> Tuple[float, Dict]:
        """Score UX analysis: usability, accessibility, interaction"""
        if "error" in agent_result:
            return 0.0, {"error": agent_result["error"]}

        usability = agent_result.get("usability", {})
        accessibility = agent_result.get("accessibility", {})
        interaction = agent_result.get("interaction", {})

        usability_score = ScoringRules.calculate_component_score(
            base_score=usability.get("score", 75),
            critical=0,
            high=len([r for r in usability.get("recommendations", [])
                     if "accessibility" in r.lower()]),
            medium=len(usability.get("recommendations", [])) // 2,
            low=len(usability.get("findings", []))
        )

        accessibility_score = ScoringRules.calculate_component_score(
            base_score=accessibility.get("score", 70),  # Usually lower
            critical=len([r for r in accessibility.get(
                "recommendations", []) if "critical" in r.lower()]),
            high=len([r for r in accessibility.get(
                "recommendations", []) if "must" in r.lower()]),
            medium=len(accessibility.get("recommendations", [])) // 3,
            low=len(accessibility.get("findings", []))
        )

        interaction_score = ScoringRules.calculate_component_score(
            base_score=interaction.get("score", 75),
            critical=0,
            high=len([r for r in interaction.get(
                "recommendations", []) if "critical" in r.lower()]),
            medium=len(interaction.get("recommendations", [])) // 2,
            low=len(interaction.get("findings", []))
        )

        # UX overall weighted toward accessibility
        overall = (usability_score * 0.3 + accessibility_score *
                   0.4 + interaction_score * 0.3)

        return overall, {
            "usability": usability_score,
            "accessibility": accessibility_score,
            "interaction": interaction_score,
            "overall": overall
        }

    @staticmethod
    def score_market_analysis(agent_result: Dict) -> Tuple[float, Dict]:
        """Score market analysis: platform fit, engagement"""
        if "error" in agent_result:
            return 0.0, {"error": agent_result["error"]}

        platform = agent_result.get("platform_optimization", {})
        engagement = agent_result.get("engagement_prediction", {})

        platform_score = ScoringRules.calculate_component_score(
            base_score=platform.get("score", 72),
            critical=0,
            high=len([r for r in platform.get(
                "recommendations", []) if "must" in r.lower()]),
            medium=len(platform.get("recommendations", [])) // 2,
            low=len(platform.get("findings", []))
        )

        engagement_score = ScoringRules.calculate_component_score(
            base_score=engagement.get("score", 70),
            critical=0,
            high=0,
            medium=len(engagement.get("optimization_tips", [])) // 2,
            low=len(engagement.get("findings", []))
        )

        overall = (platform_score + engagement_score) / 2

        return overall, {
            "platform_optimization": platform_score,
            "engagement": engagement_score,
            "overall": overall
        }

    @staticmethod
    def score_conversion_analysis(agent_result: Dict) -> Tuple[float, Dict]:
        """Score conversion analysis: CTA, copy, funnel"""
        if "error" in agent_result:
            return 0.0, {"error": agent_result["error"]}

        cta = agent_result.get("cta", {})
        copy = agent_result.get("copy", {})
        funnel = agent_result.get("funnel_fit", {})

        cta_score = ScoringRules.calculate_component_score(
            base_score=cta.get("score", 75),
            critical=len([f for f in cta.get("findings", [])
                         if "missing" in f.lower()]),
            high=len([r for r in cta.get("recommendations", [])
                     if "urgent" in r.lower()]),
            medium=len(cta.get("recommendations", [])) // 2,
            low=len(cta.get("findings", []))
        )

        copy_score = ScoringRules.calculate_component_score(
            base_score=copy.get("score", 70),
            critical=0,
            high=0,
            medium=len(copy.get("recommendations", [])) // 2,
            low=len(copy.get("findings", []))
        )

        funnel_score = ScoringRules.calculate_component_score(
            base_score=funnel.get("score", 72),
            critical=len([f for f in funnel.get(
                "findings", []) if "blocks" in f.lower()]),
            high=len([r for r in funnel.get("recommendations", [])
                     if "critical" in r.lower()]),
            medium=len(funnel.get("recommendations", [])) // 2,
            low=len(funnel.get("findings", []))
        )

        # Conversion most important
        overall = (cta_score * 0.4 + copy_score * 0.3 + funnel_score * 0.3)

        return overall, {
            "cta": cta_score,
            "copy": copy_score,
            "funnel": funnel_score,
            "overall": overall
        }

    @staticmethod
    def score_brand_analysis(agent_result: Dict) -> Tuple[float, Dict]:
        """Score brand analysis: consistency, identity"""
        if "error" in agent_result:
            return 0.0, {"error": agent_result["error"]}

        # Brand usually has sub-scores for logo, palette, typography, tone
        scores = []
        components = {}

        for key in ["logo", "palette", "typography", "tone"]:
            component = agent_result.get(key, {})
            if isinstance(component, dict):
                score = ScoringRules.calculate_component_score(
                    base_score=component.get("score", 75),
                    critical=0,
                    high=len([r for r in component.get(
                        "recommendations", []) if "critical" in r.lower()]),
                    medium=len(component.get("recommendations", [])) // 2,
                    low=len(component.get("findings", []))
                )
                components[key] = score
                scores.append(score)

        overall = sum(scores) / len(scores) if scores else 75.0
        components["overall"] = overall

        return overall, components

    @staticmethod
    def calculate_wcag_compliance(ux_result: Dict) -> WCAGCompliance:
        """
        Calculate WCAG compliance from UX analysis findings
        Deterministic based on specific keywords and recommendations
        """
        accessibility = ux_result.get("accessibility", {})
        findings = accessibility.get("findings", [])
        recommendations = accessibility.get("recommendations", [])

        # Text contrast assessment
        contrast_has_issues = any(k in " ".join(findings + recommendations).lower()
                                  for k in ["contrast", "readable", "color ratio"])
        contrast_score = 85.0 if contrast_has_issues else 95.0

        # Touch target assessment
        touch_has_issues = any(k in " ".join(findings + recommendations).lower()
                               for k in ["touch", "target", "button size", "small"])
        touch_target_score = 80.0 if touch_has_issues else 92.0

        # Screen reader assessment
        sr_has_issues = any(k in " ".join(findings + recommendations).lower()
                            for k in ["alt text", "aria", "label", "screen reader"])
        screen_reader_score = 70.0 if sr_has_issues else 85.0

        # Keyboard navigation
        kb_has_issues = any(k in " ".join(findings + recommendations).lower()
                            for k in ["keyboard", "navigation", "tab"])
        keyboard_nav_score = 75.0 if kb_has_issues else 90.0

        # Color blindness (doesn't rely only on color)
        cb_has_issues = any(k in " ".join(findings + recommendations).lower()
                            for k in ["color blind", "color only", "color alone"])
        color_blindness_score = 80.0 if cb_has_issues else 92.0

        # Focus indicators
        focus_has_issues = any(k in " ".join(findings + recommendations).lower()
                               for k in ["focus", "visible", "indicator"])
        focus_indicators_score = 75.0 if focus_has_issues else 88.0

        # Calculate overall WCAG score
        overall_wcag = (
            contrast_score * 0.15 +
            touch_target_score * 0.15 +
            screen_reader_score * 0.25 +
            keyboard_nav_score * 0.25 +
            color_blindness_score * 0.10 +
            focus_indicators_score * 0.10
        )

        # Determine WCAG level
        if overall_wcag >= 85:
            level = WCAGLevel.AAA
        elif overall_wcag >= 75:
            level = WCAGLevel.AA
        else:
            level = WCAGLevel.A

        # Collect issues
        issues = []
        if contrast_has_issues:
            issues.append(
                "Text contrast ratio may not meet WCAG standards (target 4.5:1 for AA)")
        if touch_has_issues:
            issues.append(
                "Touch targets may be too small (minimum 48x48px recommended)")
        if sr_has_issues:
            issues.append(
                "Missing alt text or ARIA labels for screen reader users")
        if kb_has_issues:
            issues.append("Not fully keyboard navigable")
        if cb_has_issues:
            issues.append("Color is sole means of conveying information")
        if focus_has_issues:
            issues.append("Focus indicators not clearly visible")

        return WCAGCompliance(
            contrast_score=contrast_score,
            touch_target_score=touch_target_score,
            screen_reader_score=screen_reader_score,
            keyboard_nav_score=keyboard_nav_score,
            color_blindness_score=color_blindness_score,
            focus_indicators_score=focus_indicators_score,
            overall_wcag_score=overall_wcag,
            level=level,
            issues=issues
        )

    @staticmethod
    def generate_recommendations_breakdown(all_recommendations: List[Dict]) -> List[RecommendationBreakdown]:
        """
        Categorize recommendations by type and severity

        Args:
            all_recommendations: List of recommendation dicts with category and priority

        Returns:
            List of RecommendationBreakdown objects
        """
        breakdown_map = {}

        for rec in all_recommendations:
            category = rec.get("category", "general")
            priority = rec.get("priority", "medium").lower()

            if category not in breakdown_map:
                breakdown_map[category] = {
                    "count": 0,
                    "critical": 0,
                    "high": 0,
                    "medium": 0,
                    "low": 0,
                    "severity_scores": []
                }

            breakdown_map[category]["count"] += 1
            breakdown_map[category]["severity_scores"].append(
                rec.get("severity_score", 50))

            if priority == "critical":
                breakdown_map[category]["critical"] += 1
            elif priority == "high":
                breakdown_map[category]["high"] += 1
            elif priority == "medium":
                breakdown_map[category]["medium"] += 1
            else:
                breakdown_map[category]["low"] += 1

        breakdowns = []
        for category, data in breakdown_map.items():
            avg_impact = sum(data["severity_scores"]) / \
                len(data["severity_scores"]) if data["severity_scores"] else 50
            breakdowns.append(RecommendationBreakdown(
                category=category,
                count=data["count"],
                critical=data["critical"],
                high=data["high"],
                medium=data["medium"],
                low=data["low"],
                avg_impact_score=avg_impact
            ))

        return sorted(breakdowns, key=lambda x: x.count, reverse=True)

    @staticmethod
    def generate_radar_data(
        visual_score: float,
        ux_score: float,
        market_score: float,
        conversion_score: float,
        brand_score: float,
        accessibility_score: float,
        target_score: float = 75.0
    ) -> RadarData:
        """Generate radar chart data"""
        return RadarData(
            categories=["Visual Design", "UX & Usability",
                        "Market Fit", "Conversion", "Brand", "Accessibility"],
            values=[visual_score, ux_score, market_score,
                    conversion_score, brand_score, accessibility_score],
            targets=[target_score] * 6
        )

    @staticmethod
    def score_complete_analysis(
        visual_result: Dict,
        ux_result: Dict,
        market_result: Dict,
        conversion_result: Dict,
        brand_result: Dict,
        recommendations: List[Dict],
        target_score: float = 75.0
    ) -> ScoreCard:
        """
        Generate complete ScoreCard from all agent results

        Args:
            visual_result: Visual analysis agent output
            ux_result: UX analysis agent output
            market_result: Market research agent output
            conversion_result: Conversion analysis agent output
            brand_result: Brand analysis agent output
            recommendations: All recommendations from agents
            target_score: Target score for comparison (default 75)

        Returns:
            ScoreCard with all metrics and breakdowns
        """
        from datetime import datetime

        # Score each agent
        visual_score, visual_details = ScoringService.score_visual_analysis(
            visual_result)
        ux_score, ux_details = ScoringService.score_ux_analysis(ux_result)
        market_score, market_details = ScoringService.score_market_analysis(
            market_result)
        conversion_score, conversion_details = ScoringService.score_conversion_analysis(
            conversion_result)
        brand_score, brand_details = ScoringService.score_brand_analysis(
            brand_result)

        # UX analysis also contributes to accessibility
        accessibility_score = ux_details.get(
            "accessibility", 70) if isinstance(ux_details, dict) else 70

        # Calculate weighted overall score
        overall_score = (
            visual_score * ScoringRules.CATEGORY_WEIGHTS["visual"] +
            ux_score * ScoringRules.CATEGORY_WEIGHTS["ux"] +
            market_score * ScoringRules.CATEGORY_WEIGHTS["market"] +
            conversion_score * ScoringRules.CATEGORY_WEIGHTS["conversion"] +
            brand_score * ScoringRules.CATEGORY_WEIGHTS["brand"] +
            accessibility_score *
            ScoringRules.CATEGORY_WEIGHTS["accessibility"]
        )

        # Calculate WCAG compliance
        wcag = ScoringService.calculate_wcag_compliance(ux_result)

        # Generate recommendations breakdown
        recs_breakdown = ScoringService.generate_recommendations_breakdown(
            recommendations)

        # Generate radar data
        radar = ScoringService.generate_radar_data(
            visual_score, ux_score, market_score, conversion_score, brand_score, accessibility_score, target_score
        )

        # Compile all components
        components = {
            "color": visual_details.get("color", 0) if isinstance(visual_details, dict) else 0,
            "layout": visual_details.get("layout", 0) if isinstance(visual_details, dict) else 0,
            "typography": visual_details.get("typography", 0) if isinstance(visual_details, dict) else 0,
            "usability": ux_details.get("usability", 0) if isinstance(ux_details, dict) else 0,
            "accessibility": ux_details.get("accessibility", 0) if isinstance(ux_details, dict) else 0,
            "interaction": ux_details.get("interaction", 0) if isinstance(ux_details, dict) else 0,
            "platform_optimization": market_details.get("platform_optimization", 0) if isinstance(market_details, dict) else 0,
            "engagement": market_details.get("engagement", 0) if isinstance(market_details, dict) else 0,
            "cta": conversion_details.get("cta", 0) if isinstance(conversion_details, dict) else 0,
            "copy": conversion_details.get("copy", 0) if isinstance(conversion_details, dict) else 0,
            "funnel": conversion_details.get("funnel", 0) if isinstance(conversion_details, dict) else 0,
            "logo": brand_details.get("logo", 0) if isinstance(brand_details, dict) else 0,
            "palette": brand_details.get("palette", 0) if isinstance(brand_details, dict) else 0,
            "brand_typography": brand_details.get("typography", 0) if isinstance(brand_details, dict) else 0,
            "tone": brand_details.get("tone", 0) if isinstance(brand_details, dict) else 0,
        }

        # Track agent errors
        agent_errors = {}
        if "error" in visual_result:
            agent_errors["visual"] = visual_result["error"]
        if "error" in ux_result:
            agent_errors["ux"] = ux_result["error"]
        if "error" in market_result:
            agent_errors["market"] = market_result["error"]
        if "error" in conversion_result:
            agent_errors["conversion"] = conversion_result["error"]
        if "error" in brand_result:
            agent_errors["brand"] = brand_result["error"]

        return ScoreCard(
            overall_score=round(overall_score, 1),
            target_score=target_score,
            delta=round(overall_score - target_score, 1),
            visual_score=round(visual_score, 1),
            ux_score=round(ux_score, 1),
            market_score=round(market_score, 1),
            conversion_score=round(conversion_score, 1),
            brand_score=round(brand_score, 1),
            accessibility_score=round(accessibility_score, 1),
            components=components,
            radar=radar,
            wcag=wcag,
            recommendations_breakdown=recs_breakdown,
            agent_errors=agent_errors,
            timestamp=datetime.now().isoformat()
        )


def score_analysis_results(state: Dict) -> Dict:
    """
    Integration function: Score analysis results and add ScoreCard to state

    Args:
        state: AnalysisState with agent results

    Returns:
        Updated state with scoring results
    """
    scorecard = ScoringService.score_complete_analysis(
        visual_result=state.get("visual_analysis", {}),
        ux_result=state.get("ux_analysis", {}),
        market_result=state.get("market_analysis", {}),
        conversion_result=state.get("conversion_analysis", {}),
        brand_result=state.get("brand_analysis", {}),
        recommendations=state.get("final_report", {}).get(
            "all_recommendations", []),
        target_score=75.0
    )

    state["scorecard"] = scorecard
    return state
