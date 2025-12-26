# tests/test_scoring_service.py

"""
Test suite for scoring service
Validates: determinism, boundedness, correctness, WCAG compliance
"""

import pytest
from components.scoring_service import (
    ScoringService,
    ScoringRules,
    ScoreCard,
    WCAGCompliance,
    RadarData,
    RecommendationBreakdown,
    WCAGLevel
)


class TestSeverityClassification:
    """Test finding severity classification"""

    def test_critical_keywords(self):
        """Critical findings should be identified correctly"""
        findings = ["Must fix critical issue", "This blocks functionality"]
        critical, high, medium, low = ScoringRules.count_findings_by_severity(
            findings)
        assert critical == 2, f"Expected 2 critical, got {critical}"

    def test_high_keywords(self):
        """High severity findings"""
        findings = ["Should improve", "This has major impact"]
        critical, high, medium, low = ScoringRules.count_findings_by_severity(
            findings)
        assert high >= 1, f"Expected at least 1 high, got {high}"

    def test_medium_keywords(self):
        """Medium severity findings"""
        findings = ["Consider improving", "Could enhance design"]
        critical, high, medium, low = ScoringRules.count_findings_by_severity(
            findings)
        assert medium >= 1, f"Expected at least 1 medium, got {medium}"

    def test_low_keywords(self):
        """Low severity findings"""
        findings = ["Minor typo", "Small spacing issue"]
        critical, high, medium, low = ScoringRules.count_findings_by_severity(
            findings)
        assert low >= 1, f"Expected at least 1 low, got {low}"

    def test_empty_findings(self):
        """Empty findings list"""
        critical, high, medium, low = ScoringRules.count_findings_by_severity([
        ])
        assert critical == 0 and high == 0 and medium == 0 and low == 0


class TestScoreCalculation:
    """Test deterministic score calculation"""

    def test_no_findings(self):
        """No findings with default base should give 75"""
        score = ScoringRules.calculate_component_score()
        assert score == 75.0, f"Expected 75.0, got {score}"

    def test_base_score_preserved(self):
        """Explicit base score with no findings should be preserved"""
        base = 82.0
        score = ScoringRules.calculate_component_score(base_score=base)
        assert score == base, f"Expected {base}, got {score}"

    def test_one_critical(self):
        """One critical finding should deduct 15 points"""
        base = 75.0
        score = ScoringRules.calculate_component_score(
            base_score=base, critical=1)
        assert score < base, f"Score should decrease for critical finding"
        assert score == 60.0, f"Expected 60, got {score}"

    def test_multiple_findings(self):
        """Multiple findings should deduct cumulatively"""
        base = 75.0
        score = ScoringRules.calculate_component_score(
            base_score=base,
            critical=1,
            high=2,
            medium=1,
            low=1
        )
        # 75 - 15 (critical) - 6 (high x2) - 2 (medium) - 1 (low) = 51
        assert score == 51.0, f"Expected 51, got {score}"

    def test_bounded_to_zero(self):
        """Negative scores should be bounded to 0"""
        score = ScoringRules.calculate_component_score(
            base_score=10.0,
            critical=5,
            high=5
        )
        assert score == 0.0, f"Score should be bounded to 0, got {score}"

    def test_bounded_to_100(self):
        """Scores should be bounded to 100"""
        score = ScoringRules.calculate_component_score(base_score=150.0)
        assert score <= 100.0, f"Score should be bounded to 100, got {score}"

    def test_quality_factor_specific(self):
        """Specific findings should have quality factor 1.0"""
        base = 80.0
        score = ScoringRules.calculate_component_score(
            base_score=base,
            critical=0,
            quality="specific"
        )
        assert score == base, f"Expected {base}, got {score}"

    def test_quality_factor_vague(self):
        """Vague findings should have quality factor 0.85"""
        base = 80.0
        score = ScoringRules.calculate_component_score(
            base_score=base,
            critical=0,
            quality="vague"
        )
        expected = base * 0.85
        assert abs(
            score - expected) < 0.1, f"Expected ~{expected}, got {score}"

    def test_quality_factor_missing(self):
        """Missing findings should have quality factor 0.70"""
        base = 80.0
        score = ScoringRules.calculate_component_score(
            base_score=base,
            critical=0,
            quality="missing"
        )
        expected = base * 0.70
        assert abs(
            score - expected) < 0.1, f"Expected ~{expected}, got {score}"

    def test_determinism(self):
        """Same input should produce same output"""
        score1 = ScoringRules.calculate_component_score(
            base_score=75,
            critical=1,
            high=2,
            medium=1,
            low=0
        )
        score2 = ScoringRules.calculate_component_score(
            base_score=75,
            critical=1,
            high=2,
            medium=1,
            low=0
        )
        assert score1 == score2, f"Scores should be identical: {score1} vs {score2}"


class TestVisualScoring:
    """Test visual analysis scoring"""

    def test_color_only(self):
        """Score color analysis alone"""
        result = {
            "color_analysis": {
                "score": 80,
                "findings": ["Good palette"],
                "recommendations": []
            },
            "layout_analysis": {
                "score": 75,
                "findings": [],
                "recommendations": []
            },
            "typography": {
                "score": 75,
                "findings": [],
                "recommendations": []
            }
        }
        score, details = ScoringService.score_visual_analysis(result)
        assert 0 <= score <= 100
        assert "overall" in details

    def test_visual_with_error(self):
        """Visual analysis with error"""
        result = {"error": "API timeout"}
        score, details = ScoringService.score_visual_analysis(result)
        assert score == 0.0
        assert "error" in details


class TestUXScoring:
    """Test UX analysis scoring"""

    def test_ux_component_breakdown(self):
        """UX should break down into usability, accessibility, interaction"""
        result = {
            "usability": {"score": 80, "findings": [], "recommendations": []},
            "accessibility": {
                "score": 70,
                "findings": ["Missing alt text"],
                "recommendations": ["Add alt text"]
            },
            "interaction": {"score": 75, "findings": [], "recommendations": []}
        }
        score, details = ScoringService.score_ux_analysis(result)
        assert 0 <= score <= 100
        assert "usability" in details
        assert "accessibility" in details
        assert "interaction" in details

    def test_ux_weights_accessibility_higher(self):
        """Accessibility should have higher weight in UX"""
        result = {
            "usability": {"score": 100, "findings": [], "recommendations": []},
            "accessibility": {"score": 60, "findings": ["Missing alt text", "No screen reader support"], "recommendations": []},
            "interaction": {"score": 100, "findings": [], "recommendations": []}
        }
        score, details = ScoringService.score_ux_analysis(result)
        # Accessibility has 40% weight, so lower accessibility should pull down overall
        # Rough calculation: 100*0.3 + 60*0.4 + 100*0.3 = 30 + 24 + 30 = 84
        # But accessibility starts at 60 and gets deductions, so overall should be lower than simple avg
        simple_avg = (100 + 60 + 100) / 3
        assert score <= simple_avg + \
            5, f"Accessibility weight should lower overall, got {score}"


class TestWCAGCompliance:
    """Test WCAG compliance calculation"""

    def test_wcag_all_clear(self):
        """Good accessibility practices should result in high WCAG score"""
        ux_result = {
            "accessibility": {
                "findings": ["Excellent color contrast (7:1 ratio throughout)", "Fully keyboard navigable",
                             "All images have high quality descriptive alt text", "Clear visible focus indicators",
                             "No color-only reliance"],
                "recommendations": []
            }
        }
        wcag = ScoringService.calculate_wcag_compliance(ux_result)
        # With these positive findings, WCAG score should be reasonably high
        assert wcag.overall_wcag_score > 75, f"Good accessibility should score >75, got {wcag.overall_wcag_score}"
        assert wcag.level in [
            WCAGLevel.AA, WCAGLevel.AAA], f"Expected AA or AAA, got {wcag.level.name}"

    def test_wcag_with_contrast_issue(self):
        """Contrast issue detected"""
        ux_result = {
            "accessibility": {
                "findings": ["Poor color contrast ratio"],
                "recommendations": []
            }
        }
        wcag = ScoringService.calculate_wcag_compliance(ux_result)
        assert wcag.contrast_score < 90
        assert any("contrast" in issue.lower() for issue in wcag.issues)

    def test_wcag_with_alt_text_issue(self):
        """Missing alt text detected"""
        ux_result = {
            "accessibility": {
                "findings": ["Missing alt text on images"],
                "recommendations": ["Add alt text"]
            }
        }
        wcag = ScoringService.calculate_wcag_compliance(ux_result)
        assert wcag.screen_reader_score < 85
        assert any("alt text" in issue.lower() for issue in wcag.issues)

    def test_wcag_level_assignment(self):
        """WCAG levels assigned correctly"""
        # Create scenarios that produce different levels
        ux_result_good = {
            "accessibility": {
                "findings": ["Good throughout"],
                "recommendations": []
            }
        }
        wcag_good = ScoringService.calculate_wcag_compliance(ux_result_good)
        assert wcag_good.overall_wcag_score > 80

        ux_result_poor = {
            "accessibility": {
                "findings": ["Missing alt text", "Poor contrast", "No keyboard nav"],
                "recommendations": []
            }
        }
        wcag_poor = ScoringService.calculate_wcag_compliance(ux_result_poor)
        assert wcag_poor.overall_wcag_score < wcag_good.overall_wcag_score


class TestRecommendationsBreakdown:
    """Test recommendations categorization"""

    def test_empty_recommendations(self):
        """No recommendations"""
        breakdown = ScoringService.generate_recommendations_breakdown([])
        assert len(breakdown) == 0

    def test_single_category(self):
        """All recommendations in one category"""
        recs = [
            {"category": "visual", "priority": "high", "severity_score": 70},
            {"category": "visual", "priority": "medium", "severity_score": 50},
            {"category": "visual", "priority": "low", "severity_score": 30}
        ]
        breakdown = ScoringService.generate_recommendations_breakdown(recs)
        assert len(breakdown) == 1
        assert breakdown[0].category == "visual"
        assert breakdown[0].count == 3
        assert breakdown[0].high == 1
        assert breakdown[0].medium == 1
        assert breakdown[0].low == 1

    def test_multiple_categories(self):
        """Recommendations in multiple categories"""
        recs = [
            {"category": "visual", "priority": "high", "severity_score": 80},
            {"category": "ux", "priority": "critical", "severity_score": 90},
            {"category": "conversion", "priority": "medium", "severity_score": 50}
        ]
        breakdown = ScoringService.generate_recommendations_breakdown(recs)
        assert len(breakdown) == 3

        # Check totals
        assert sum(b.count for b in breakdown) == 3
        assert sum(b.critical for b in breakdown) == 1

    def test_avg_impact_score(self):
        """Average impact score calculated correctly"""
        recs = [
            {"category": "visual", "priority": "high", "severity_score": 100},
            {"category": "visual", "priority": "high", "severity_score": 50}
        ]
        breakdown = ScoringService.generate_recommendations_breakdown(recs)
        visual = breakdown[0]
        assert abs(visual.avg_impact_score - 75.0) < 0.1


class TestRadarData:
    """Test radar chart data generation"""

    def test_radar_structure(self):
        """Radar should have 6 categories"""
        radar = ScoringService.generate_radar_data(
            visual_score=78,
            ux_score=72,
            market_score=75,
            conversion_score=82,
            brand_score=70,
            accessibility_score=68,
            target_score=75
        )
        assert len(radar.categories) == 6
        assert len(radar.values) == 6
        assert len(radar.targets) == 6
        assert all(t == 75 for t in radar.targets)

    def test_radar_values_bounded(self):
        """All radar values should be 0-100"""
        radar = ScoringService.generate_radar_data(
            visual_score=78,
            ux_score=72,
            market_score=75,
            conversion_score=82,
            brand_score=70,
            accessibility_score=68
        )
        for value in radar.values:
            assert 0 <= value <= 100


class TestCompleteScoring:
    """Test complete scorecard generation"""

    @pytest.fixture
    def sample_agent_results(self):
        """Sample agent results for testing"""
        return {
            "visual": {
                "overall_score": 78.5,
                "color_analysis": {
                    "score": 82,
                    "findings": ["Good palette"],
                    "recommendations": ["Improve secondary colors"]
                },
                "layout_analysis": {
                    "score": 75,
                    "findings": [],
                    "recommendations": []
                },
                "typography": {
                    "score": 76,
                    "findings": [],
                    "recommendations": []
                }
            },
            "ux": {
                "overall_score": 72.5,
                "usability": {"score": 74, "findings": [], "recommendations": []},
                "accessibility": {
                    "score": 68,
                    "findings": ["Missing alt text"],
                    "recommendations": ["Add alt text"]
                },
                "interaction": {"score": 75, "findings": [], "recommendations": []}
            },
            "market": {
                "overall_score": 70.0,
                "platform_optimization": {"score": 72, "findings": [], "recommendations": []},
                "engagement_prediction": {"score": 68, "findings": [], "recommendations": []}
            },
            "conversion": {
                "overall_score": 71.0,
                "cta": {"score": 75, "findings": [], "recommendations": []},
                "copy": {"score": 68, "findings": [], "recommendations": []},
                "funnel_fit": {"score": 70, "findings": [], "recommendations": []}
            },
            "brand": {
                "overall_score": 72.0,
                "logo": {"score": 75, "findings": [], "recommendations": []},
                "palette": {"score": 72, "findings": [], "recommendations": []},
                "typography": {"score": 71, "findings": [], "recommendations": []},
                "tone": {"score": 70, "findings": [], "recommendations": []}
            },
            "recommendations": [
                {"category": "visual", "priority": "medium", "severity_score": 50},
                {"category": "ux", "priority": "critical", "severity_score": 85}
            ]
        }

    def test_complete_scorecard(self, sample_agent_results):
        """Generate complete scorecard"""
        scorecard = ScoringService.score_complete_analysis(
            visual_result=sample_agent_results["visual"],
            ux_result=sample_agent_results["ux"],
            market_result=sample_agent_results["market"],
            conversion_result=sample_agent_results["conversion"],
            brand_result=sample_agent_results["brand"],
            recommendations=sample_agent_results["recommendations"],
            target_score=75.0
        )

        # Verify structure
        assert isinstance(scorecard, ScoreCard)
        assert 0 <= scorecard.overall_score <= 100
        assert scorecard.target_score == 75.0
        assert abs(scorecard.delta) <= 100

    def test_scorecard_category_scores(self, sample_agent_results):
        """All category scores should be 0-100"""
        scorecard = ScoringService.score_complete_analysis(
            visual_result=sample_agent_results["visual"],
            ux_result=sample_agent_results["ux"],
            market_result=sample_agent_results["market"],
            conversion_result=sample_agent_results["conversion"],
            brand_result=sample_agent_results["brand"],
            recommendations=sample_agent_results["recommendations"]
        )

        for score in [scorecard.visual_score, scorecard.ux_score, scorecard.market_score,
                      scorecard.conversion_score, scorecard.brand_score, scorecard.accessibility_score]:
            assert 0 <= score <= 100, f"Score {score} out of bounds"

    def test_scorecard_components(self, sample_agent_results):
        """Components should be present and bounded"""
        scorecard = ScoringService.score_complete_analysis(
            visual_result=sample_agent_results["visual"],
            ux_result=sample_agent_results["ux"],
            market_result=sample_agent_results["market"],
            conversion_result=sample_agent_results["conversion"],
            brand_result=sample_agent_results["brand"],
            recommendations=sample_agent_results["recommendations"]
        )

        assert len(scorecard.components) > 0
        for name, score in scorecard.components.items():
            assert isinstance(score, (int, float))
            assert 0 <= score <= 100

    def test_scorecard_radar(self, sample_agent_results):
        """Radar data should be complete"""
        scorecard = ScoringService.score_complete_analysis(
            visual_result=sample_agent_results["visual"],
            ux_result=sample_agent_results["ux"],
            market_result=sample_agent_results["market"],
            conversion_result=sample_agent_results["conversion"],
            brand_result=sample_agent_results["brand"],
            recommendations=sample_agent_results["recommendations"]
        )

        assert len(scorecard.radar.categories) == 6
        assert len(scorecard.radar.values) == 6
        assert len(scorecard.radar.targets) == 6

    def test_scorecard_wcag(self, sample_agent_results):
        """WCAG data should be present"""
        scorecard = ScoringService.score_complete_analysis(
            visual_result=sample_agent_results["visual"],
            ux_result=sample_agent_results["ux"],
            market_result=sample_agent_results["market"],
            conversion_result=sample_agent_results["conversion"],
            brand_result=sample_agent_results["brand"],
            recommendations=sample_agent_results["recommendations"]
        )

        assert isinstance(scorecard.wcag, WCAGCompliance)
        assert 0 <= scorecard.wcag.overall_wcag_score <= 100
        assert scorecard.wcag.level in [
            WCAGLevel.A, WCAGLevel.AA, WCAGLevel.AAA]

    def test_scorecard_recommendations_breakdown(self, sample_agent_results):
        """Recommendations breakdown should be present"""
        scorecard = ScoringService.score_complete_analysis(
            visual_result=sample_agent_results["visual"],
            ux_result=sample_agent_results["ux"],
            market_result=sample_agent_results["market"],
            conversion_result=sample_agent_results["conversion"],
            brand_result=sample_agent_results["brand"],
            recommendations=sample_agent_results["recommendations"]
        )

        assert isinstance(scorecard.recommendations_breakdown, list)
        for breakdown in scorecard.recommendations_breakdown:
            assert isinstance(breakdown, RecommendationBreakdown)
            assert breakdown.count == (breakdown.critical + breakdown.high +
                                       breakdown.medium + breakdown.low)


class TestDeterminism:
    """Test determinism guarantees"""

    def test_same_input_same_output(self):
        """Same input should always produce same output"""
        agent_results = {
            "visual": {"error": None, "overall_score": 78.5, "color_analysis": {"score": 82, "findings": [], "recommendations": []}},
            "ux": {"error": None, "overall_score": 72.5, "usability": {"score": 74, "findings": [], "recommendations": []}},
            "market": {"error": None, "overall_score": 70.0},
            "conversion": {"error": None, "overall_score": 71.0},
            "brand": {"error": None, "overall_score": 72.0}
        }

        result1 = ScoringService.score_complete_analysis(
            visual_result=agent_results["visual"],
            ux_result=agent_results["ux"],
            market_result=agent_results["market"],
            conversion_result=agent_results["conversion"],
            brand_result=agent_results["brand"],
            recommendations=[]
        )

        result2 = ScoringService.score_complete_analysis(
            visual_result=agent_results["visual"],
            ux_result=agent_results["ux"],
            market_result=agent_results["market"],
            conversion_result=agent_results["conversion"],
            brand_result=agent_results["brand"],
            recommendations=[]
        )

        assert result1.overall_score == result2.overall_score
        assert result1.visual_score == result2.visual_score
        assert result1.delta == result2.delta


class TestCategoryWeights:
    """Test category weights"""

    def test_weights_sum_to_one(self):
        """Category weights must sum to 1.0"""
        total = sum(ScoringRules.CATEGORY_WEIGHTS.values())
        assert abs(
            total - 1.0) < 0.01, f"Weights should sum to 1.0, got {total}"

    def test_conversion_highest_weight(self):
        """Conversion should have highest weight"""
        weights = ScoringRules.CATEGORY_WEIGHTS
        conversion_weight = weights["conversion"]
        for category, weight in weights.items():
            if category != "conversion":
                assert conversion_weight >= weight, \
                    f"Conversion weight {conversion_weight} should be >= {category} weight {weight}"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
