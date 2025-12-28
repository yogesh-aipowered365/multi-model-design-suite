# components/agents_real.py

"""
REAL Agent Implementation with OpenRouter API calls
Each agent makes actual vision LLM calls to analyze designs
"""

import json
import time
from typing import Optional, List, Dict, Any
from components.models import AgentResult, AgentFinding, SeverityLevel
from components.llm_client import run_llm, parse_json_response


def analyze_visual_design(image_base64: str, platform: str, creative_type: str,
                          api_key: Optional[str] = None,
                          rag_citations: Optional[List] = None) -> AgentResult:
    """
    REAL Visual Design Agent - Analyzes image using OpenRouter LLM
    
    Makes actual vision API call to analyze:
    - Color palette and harmony
    - Layout and composition  
    - Typography and hierarchy
    - Visual balance
    """
    start_time = time.time()
    try:
        prompt = f"""You are an expert visual design analyst. Analyze this {platform} {creative_type} design image.

Provide analysis in JSON format with:
{{
  "overall_score": <0-100>,
  "color_analysis": {{
    "score": <0-100>,
    "dominant_colors": ["hex1", "hex2", ...],
    "harmony_type": "complementary|analogous|triadic|monochromatic",
    "contrast_score": <0-100>,
    "findings": ["finding1", "finding2"],
    "recommendations": ["rec1", "rec2"]
  }},
  "layout_analysis": {{
    "score": <0-100>,
    "hierarchy_clarity": <0-100>,
    "balance_score": <0-100>,
    "findings": ["finding1"],
    "recommendations": ["rec1"]
  }},
  "typography": {{
    "score": <0-100>,
    "readability": <0-100>,
    "findings": ["finding1"],
    "recommendations": ["rec1"]
  }},
  "critical_issues": ["issue1", "issue2"],
  "summary": "brief summary"
}}

Focus on actionable feedback."""

        response = run_llm(
            task_name="visual_analysis",
            messages=[{"role": "user", "content": prompt}],
            images=[image_base64],
            temperature=0.7,
            max_tokens=1000,
            json_mode=True,
            api_key=api_key
        )

        if response.error:
            return AgentResult(
                agent_name="visual_analysis",
                summary=f"Visual analysis failed: {response.error}",
                findings=[],
                score=0.0,
                subscores={},
                errors=[response.error],
                latency_ms=response.latency_ms
            )

        # Parse LLM response
        try:
            data = json.loads(response.text)
        except json.JSONDecodeError:
            return AgentResult(
                agent_name="visual_analysis",
                summary="Failed to parse LLM response",
                findings=[],
                score=0.0,
                subscores={},
                errors=["JSON parse error"],
                latency_ms=response.latency_ms
            )

        # Build findings from LLM response
        findings = []
        
        # Color analysis findings
        if "color_analysis" in data and data["color_analysis"].get("findings"):
            for rec in data["color_analysis"]["findings"]:
                findings.append(AgentFinding(
                    category="Color Analysis",
                    severity=SeverityLevel.INFO,
                    evidence=f"Harmony: {data['color_analysis'].get('harmony_type', 'unknown')}",
                    recommendation=rec,
                    expected_impact="+10-20% visual impact",
                    confidence=0.85,
                    supporting_data={"colors": data["color_analysis"].get("dominant_colors", [])}
                ))

        # Layout findings
        if "layout_analysis" in data and data["layout_analysis"].get("findings"):
            for rec in data["layout_analysis"]["findings"]:
                findings.append(AgentFinding(
                    category="Layout & Composition",
                    severity=SeverityLevel.INFO,
                    evidence=f"Hierarchy: {data['layout_analysis'].get('hierarchy_clarity', 0)}/100",
                    recommendation=rec,
                    expected_impact="+5-15% clarity",
                    confidence=0.85
                ))

        # Critical issues
        critical_issues = data.get("critical_issues", [])
        for issue in critical_issues:
            findings.append(AgentFinding(
                category="Critical Issue",
                severity=SeverityLevel.CRITICAL,
                evidence="Visual analysis detected critical issue",
                recommendation=issue,
                expected_impact="Must fix before launch",
                confidence=0.90
            ))

        overall_score = data.get("overall_score", 70)
        latency_ms = response.latency_ms

        return AgentResult(
            agent_name="visual_analysis",
            summary=data.get("summary", "Visual design analysis complete"),
            findings=findings,
            score=overall_score,
            subscores={
                "color": data.get("color_analysis", {}).get("score", 70),
                "layout": data.get("layout_analysis", {}).get("score", 70),
                "typography": data.get("typography", {}).get("score", 70)
            },
            latency_ms=latency_ms
        )

    except Exception as e:
        latency_ms = (time.time() - start_time) * 1000
        return AgentResult(
            agent_name="visual_analysis",
            summary=f"Visual analysis error: {str(e)}",
            findings=[],
            score=0.0,
            subscores={},
            errors=[str(e)],
            latency_ms=latency_ms
        )


def analyze_ux_design(image_base64: str, platform: str, creative_type: str,
                      api_key: Optional[str] = None,
                      rag_citations: Optional[List] = None) -> AgentResult:
    """
    REAL UX Analysis Agent - Analyzes usability using OpenRouter LLM
    
    Makes actual vision API call to evaluate:
    - Navigation clarity
    - Touch target sizes
    - Readability
    - Accessibility compliance
    """
    start_time = time.time()
    try:
        prompt = f"""You are a UX/Usability expert. Analyze this {platform} {creative_type} design for user experience.

Evaluate in JSON format:
{{
  "overall_score": <0-100>,
  "usability": {{
    "score": <0-100>,
    "nav_clarity": <0-100>,
    "findings": ["finding1"],
    "recommendations": ["rec1"]
  }},
  "accessibility": {{
    "score": <0-100>,
    "wcag_score": <0-100>,
    "findings": ["finding1"],
    "recommendations": ["rec1"]
  }},
  "interaction": {{
    "score": <0-100>,
    "findings": ["finding1"],
    "recommendations": ["rec1"]
  }},
  "critical_ux_issues": ["issue1"],
  "summary": "brief summary"
}}

Focus on usability problems that reduce conversions or cause friction."""

        response = run_llm(
            task_name="ux_analysis",
            messages=[{"role": "user", "content": prompt}],
            images=[image_base64],
            temperature=0.7,
            max_tokens=1000,
            json_mode=True,
            api_key=api_key
        )

        if response.error:
            return AgentResult(
                agent_name="ux_critique",
                summary=f"UX analysis failed: {response.error}",
                findings=[],
                score=0.0,
                subscores={},
                errors=[response.error],
                latency_ms=response.latency_ms
            )

        try:
            data = json.loads(response.text)
        except json.JSONDecodeError:
            return AgentResult(
                agent_name="ux_critique",
                summary="Failed to parse UX response",
                findings=[],
                score=0.0,
                subscores={},
                errors=["JSON parse error"],
                latency_ms=response.latency_ms
            )

        findings = []

        # Usability findings
        if "usability" in data:
            for rec in data["usability"].get("recommendations", []):
                findings.append(AgentFinding(
                    category="Usability",
                    severity=SeverityLevel.WARNING,
                    evidence="UX analysis",
                    recommendation=rec,
                    expected_impact="+5-15% usability",
                    confidence=0.85
                ))

        # Accessibility findings
        if "accessibility" in data:
            for rec in data["accessibility"].get("recommendations", []):
                findings.append(AgentFinding(
                    category="Accessibility",
                    severity=SeverityLevel.WARNING,
                    evidence=f"WCAG Score: {data['accessibility'].get('wcag_score', 0)}/100",
                    recommendation=rec,
                    expected_impact="Compliance + wider reach",
                    confidence=0.85
                ))

        # Critical UX issues
        for issue in data.get("critical_ux_issues", []):
            findings.append(AgentFinding(
                category="Critical UX Issue",
                severity=SeverityLevel.CRITICAL,
                evidence="UX analysis detected critical issue",
                recommendation=issue,
                expected_impact="High impact on conversions",
                confidence=0.90
            ))

        overall_score = data.get("overall_score", 70)

        return AgentResult(
            agent_name="ux_critique",
            summary=data.get("summary", "UX analysis complete"),
            findings=findings,
            score=overall_score,
            subscores={
                "usability": data.get("usability", {}).get("score", 70),
                "accessibility": data.get("accessibility", {}).get("score", 70),
                "interaction": data.get("interaction", {}).get("score", 70)
            },
            latency_ms=response.latency_ms
        )

    except Exception as e:
        latency_ms = (time.time() - start_time) * 1000
        return AgentResult(
            agent_name="ux_critique",
            summary=f"UX analysis error: {str(e)}",
            findings=[],
            score=0.0,
            subscores={},
            errors=[str(e)],
            latency_ms=latency_ms
        )


def analyze_market_positioning(image_base64: str, platform: str, creative_type: str,
                               api_key: Optional[str] = None,
                               rag_citations: Optional[List] = None) -> AgentResult:
    """
    REAL Market Research Agent - Analyzes market fit using OpenRouter LLM
    
    Makes actual vision API call to evaluate:
    - Platform optimization
    - Audience fit
    - Competitive positioning
    - Engagement potential
    """
    start_time = time.time()
    try:
        prompt = f"""You are a market strategist. Analyze this {platform} {creative_type} for market positioning.

Provide in JSON format:
{{
  "overall_score": <0-100>,
  "platform_optimization": {{
    "score": <0-100>,
    "findings": ["finding1"],
    "recommendations": ["rec1"]
  }},
  "audience_fit": {{
    "score": <0-100>,
    "target_segments": ["segment1"],
    "findings": ["finding1"],
    "recommendations": ["rec1"]
  }},
  "competitive_positioning": {{
    "score": <0-100>,
    "findings": ["finding1"],
    "recommendations": ["rec1"]
  }},
  "engagement_potential": {{
    "score": <0-100>,
    "predictions": ["pred1"]
  }},
  "strategic_issues": ["issue1"],
  "summary": "brief summary"
}}

Assess how well this creative fits the market and platform."""

        response = run_llm(
            task_name="market_analysis",
            messages=[{"role": "user", "content": prompt}],
            images=[image_base64],
            temperature=0.7,
            max_tokens=1000,
            json_mode=True,
            api_key=api_key
        )

        if response.error:
            return AgentResult(
                agent_name="market_research",
                summary=f"Market analysis failed: {response.error}",
                findings=[],
                score=0.0,
                subscores={},
                errors=[response.error],
                latency_ms=response.latency_ms
            )

        try:
            data = json.loads(response.text)
        except json.JSONDecodeError:
            return AgentResult(
                agent_name="market_research",
                summary="Failed to parse market response",
                findings=[],
                score=0.0,
                subscores={},
                errors=["JSON parse error"],
                latency_ms=response.latency_ms
            )

        findings = []

        # Platform optimization
        if "platform_optimization" in data:
            for rec in data["platform_optimization"].get("recommendations", []):
                findings.append(AgentFinding(
                    category="Platform Fit",
                    severity=SeverityLevel.WARNING,
                    evidence="Market analysis",
                    recommendation=rec,
                    expected_impact="+10-20% platform relevance",
                    confidence=0.85
                ))

        # Audience fit
        if "audience_fit" in data:
            for rec in data["audience_fit"].get("recommendations", []):
                findings.append(AgentFinding(
                    category="Audience Alignment",
                    severity=SeverityLevel.INFO,
                    evidence=f"Target: {', '.join(data['audience_fit'].get('target_segments', []))}",
                    recommendation=rec,
                    expected_impact="+15-25% audience resonance",
                    confidence=0.85
                ))

        # Strategic issues
        for issue in data.get("strategic_issues", []):
            findings.append(AgentFinding(
                category="Strategic Issue",
                severity=SeverityLevel.CRITICAL,
                evidence="Market positioning",
                recommendation=issue,
                expected_impact="Must address for market success",
                confidence=0.90
            ))

        overall_score = data.get("overall_score", 70)

        return AgentResult(
            agent_name="market_research",
            summary=data.get("summary", "Market analysis complete"),
            findings=findings,
            score=overall_score,
            subscores={
                "platform_optimization": data.get("platform_optimization", {}).get("score", 70),
                "audience_fit": data.get("audience_fit", {}).get("score", 70),
                "competitive_positioning": data.get("competitive_positioning", {}).get("score", 70),
                "engagement_potential": data.get("engagement_potential", {}).get("score", 70)
            },
            latency_ms=response.latency_ms
        )

    except Exception as e:
        latency_ms = (time.time() - start_time) * 1000
        return AgentResult(
            agent_name="market_research",
            summary=f"Market analysis error: {str(e)}",
            findings=[],
            score=0.0,
            subscores={},
            errors=[str(e)],
            latency_ms=latency_ms
        )


def analyze_conversion_optimization(image_base64: str, platform: str, creative_type: str,
                                   api_key: Optional[str] = None,
                                   rag_citations: Optional[List] = None) -> AgentResult:
    """
    REAL Conversion CTA Agent - Analyzes CTA and conversion using OpenRouter LLM
    
    Makes actual vision API call to evaluate:
    - CTA visibility and clarity
    - Urgency elements
    - Offer framing
    - Conversion friction points
    """
    start_time = time.time()
    try:
        prompt = f"""You are a conversion rate optimization expert. Analyze this {platform} {creative_type} for conversion potential.

Evaluate in JSON format:
{{
  "overall_score": <0-100>,
  "cta_analysis": {{
    "score": <0-100>,
    "visibility": <0-100>,
    "clarity": <0-100>,
    "findings": ["finding1"],
    "cta_variants": ["variant1", "variant2"]
  }},
  "urgency_elements": {{
    "score": <0-100>,
    "recommendations": ["rec1"]
  }},
  "offer_clarity": {{
    "score": <0-100>,
    "recommendations": ["rec1"]
  }},
  "friction_points": ["friction1"],
  "conversion_recommendations": ["rec1"],
  "summary": "brief summary"
}}

Focus on conversion barriers and CTA optimization."""

        response = run_llm(
            task_name="conversion_analysis",
            messages=[{"role": "user", "content": prompt}],
            images=[image_base64],
            temperature=0.7,
            max_tokens=1000,
            json_mode=True,
            api_key=api_key
        )

        if response.error:
            return AgentResult(
                agent_name="conversion_cta",
                summary=f"Conversion analysis failed: {response.error}",
                findings=[],
                score=0.0,
                subscores={},
                errors=[response.error],
                latency_ms=response.latency_ms
            )

        try:
            data = json.loads(response.text)
        except json.JSONDecodeError:
            return AgentResult(
                agent_name="conversion_cta",
                summary="Failed to parse conversion response",
                findings=[],
                score=0.0,
                subscores={},
                errors=["JSON parse error"],
                latency_ms=response.latency_ms
            )

        findings = []

        # CTA findings
        if "cta_analysis" in data:
            for rec in data["cta_analysis"].get("recommendations", []):
                findings.append(AgentFinding(
                    category="CTA Optimization",
                    severity=SeverityLevel.CRITICAL,
                    evidence=f"Visibility: {data['cta_analysis'].get('visibility', 0)}/100",
                    recommendation=rec,
                    expected_impact="+20-35% CTR",
                    confidence=0.90
                ))

        # Friction points
        for friction in data.get("friction_points", []):
            findings.append(AgentFinding(
                category="Conversion Friction",
                severity=SeverityLevel.WARNING,
                evidence="Friction point identified",
                recommendation=f"Remove or reduce: {friction}",
                expected_impact="+10-20% conversions",
                confidence=0.85
            ))

        # Conversion recommendations
        for rec in data.get("conversion_recommendations", []):
            findings.append(AgentFinding(
                category="Conversion Strategy",
                severity=SeverityLevel.INFO,
                evidence="Optimization analysis",
                recommendation=rec,
                expected_impact="+5-15% conversion rate",
                confidence=0.80
            ))

        overall_score = data.get("overall_score", 70)

        return AgentResult(
            agent_name="conversion_cta",
            summary=data.get("summary", "Conversion analysis complete"),
            findings=findings,
            score=overall_score,
            subscores={
                "cta": data.get("cta_analysis", {}).get("score", 70),
                "urgency": data.get("urgency_elements", {}).get("score", 70),
                "offer_clarity": data.get("offer_clarity", {}).get("score", 70)
            },
            latency_ms=response.latency_ms
        )

    except Exception as e:
        latency_ms = (time.time() - start_time) * 1000
        return AgentResult(
            agent_name="conversion_cta",
            summary=f"Conversion analysis error: {str(e)}",
            findings=[],
            score=0.0,
            subscores={},
            errors=[str(e)],
            latency_ms=latency_ms
        )


def analyze_brand_consistency_check(image_base64: str, platform: str, creative_type: str,
                                    api_key: Optional[str] = None,
                                    rag_citations: Optional[List] = None) -> AgentResult:
    """
    REAL Brand Consistency Agent - Analyzes brand alignment using OpenRouter LLM
    
    Makes actual vision API call to evaluate:
    - Logo and visual identity
    - Color consistency
    - Brand messaging alignment
    - Voice and tone
    """
    start_time = time.time()
    try:
        prompt = f"""You are a brand strategist. Analyze this {platform} {creative_type} for brand consistency.

Evaluate in JSON format:
{{
  "overall_score": <0-100>,
  "visual_identity": {{
    "score": <0-100>,
    "elements_detected": ["element1"],
    "recommendations": ["rec1"]
  }},
  "color_consistency": {{
    "score": <0-100>,
    "recommendations": ["rec1"]
  }},
  "messaging_alignment": {{
    "score": <0-100>,
    "findings": ["finding1"],
    "recommendations": ["rec1"]
  }},
  "brand_issues": ["issue1"],
  "brand_strengths": ["strength1"],
  "summary": "brief summary"
}}

Assess brand consistency and identity alignment."""

        response = run_llm(
            task_name="brand_analysis",
            messages=[{"role": "user", "content": prompt}],
            images=[image_base64],
            temperature=0.7,
            max_tokens=1000,
            json_mode=True,
            api_key=api_key
        )

        if response.error:
            return AgentResult(
                agent_name="brand_consistency",
                summary=f"Brand analysis failed: {response.error}",
                findings=[],
                score=0.0,
                subscores={},
                errors=[response.error],
                latency_ms=response.latency_ms
            )

        try:
            data = json.loads(response.text)
        except json.JSONDecodeError:
            return AgentResult(
                agent_name="brand_consistency",
                summary="Failed to parse brand response",
                findings=[],
                score=0.0,
                subscores={},
                errors=["JSON parse error"],
                latency_ms=response.latency_ms
            )

        findings = []

        # Brand strengths
        for strength in data.get("brand_strengths", []):
            findings.append(AgentFinding(
                category="Brand Strength",
                severity=SeverityLevel.INFO,
                evidence="Brand analysis",
                recommendation=f"Maintain and amplify: {strength}",
                expected_impact="Stronger brand recall",
                confidence=0.85
            ))

        # Visual identity
        if "visual_identity" in data:
            for rec in data["visual_identity"].get("recommendations", []):
                findings.append(AgentFinding(
                    category="Visual Identity",
                    severity=SeverityLevel.WARNING,
                    evidence=f"Elements: {', '.join(data['visual_identity'].get('elements_detected', []))}",
                    recommendation=rec,
                    expected_impact="+10-20% brand recognition",
                    confidence=0.85
                ))

        # Brand issues
        for issue in data.get("brand_issues", []):
            findings.append(AgentFinding(
                category="Brand Inconsistency",
                severity=SeverityLevel.CRITICAL,
                evidence="Brand analysis",
                recommendation=f"Fix: {issue}",
                expected_impact="Critical for brand integrity",
                confidence=0.90
            ))

        overall_score = data.get("overall_score", 70)

        return AgentResult(
            agent_name="brand_consistency",
            summary=data.get("summary", "Brand analysis complete"),
            findings=findings,
            score=overall_score,
            subscores={
                "visual_identity": data.get("visual_identity", {}).get("score", 70),
                "color_consistency": data.get("color_consistency", {}).get("score", 70),
                "messaging_alignment": data.get("messaging_alignment", {}).get("score", 70)
            },
            latency_ms=response.latency_ms
        )

    except Exception as e:
        latency_ms = (time.time() - start_time) * 1000
        return AgentResult(
            agent_name="brand_consistency",
            summary=f"Brand analysis error: {str(e)}",
            findings=[],
            score=0.0,
            subscores={},
            errors=[str(e)],
            latency_ms=latency_ms
        )
