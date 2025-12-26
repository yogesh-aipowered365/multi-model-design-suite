"""
Brand Compliance Evaluation
Lightweight helpers for assessing brand rule compliance.
"""

from typing import Dict, List


def compute_brand_compliance_score(brand_rules: Dict, violations: List[Dict]) -> int:
    """
    Compute a brand compliance score (0-100) based on violations.
    
    Args:
        brand_rules: Normalized rules dict
        violations: List of violations found
    
    Returns:
        Score 0-100
    """
    if not brand_rules or not brand_rules.get('rules'):
        return 100  # No rules = compliant
    
    total_rules = len(brand_rules.get('rules', []))
    if total_rules == 0:
        return 100
    
    # Count violations by priority
    high_violations = sum(1 for v in violations if v.get('severity') == 'high')
    medium_violations = sum(1 for v in violations if v.get('severity') == 'medium')
    low_violations = sum(1 for v in violations if v.get('severity') == 'low')
    
    # Calculate score: each high violation = -20pts, medium = -10pts, low = -5pts
    penalty = (high_violations * 20) + (medium_violations * 10) + (low_violations * 5)
    score = max(0, 100 - penalty)
    
    return int(score)


def create_mock_brand_compliance(brand_rules: Dict, design_name: str = "Design") -> Dict:
    """
    Create mock brand compliance results for demo/testing.
    
    Args:
        brand_rules: Normalized rules dict
        design_name: Name of design being evaluated
    
    Returns:
        Brand compliance dict
    """
    if not brand_rules or not brand_rules.get('rules'):
        return {
            "enabled": False,
            "score": 100,
            "level": "Not Evaluated",
            "passed_checks": [],
            "violations": [],
            "summary": "No brand rules provided."
        }
    
    rules = brand_rules.get('rules', [])
    
    # Mock: assume ~70% compliance
    num_rules = len(rules)
    num_violations = max(1, int(num_rules * 0.3))
    
    # Select some rules as violations (mostly medium/low for demo)
    violations = []
    for i, rule in enumerate(rules[:num_violations]):
        severity = 'high' if i == 0 else 'medium' if i % 2 == 0 else 'low'
        violations.append({
            "rule_id": rule['id'],
            "severity": severity,
            "category": rule.get('category', 'General'),
            "issue": f"{design_name} does not fully comply with: {rule['rule'][:80]}",
            "evidence": "Observed in visual inspection",
            "fix": f"Adjust design to align with {rule['id']}"
        })
    
    # Mock: assume ~70% of rules are passed
    passed_checks = []
    for rule in rules[num_violations:]:
        passed_checks.append({
            "rule_id": rule['id'],
            "category": rule.get('category', 'General'),
            "note": f"{design_name} complies with: {rule['rule'][:80]}"
        })
    
    score = compute_brand_compliance_score(brand_rules, violations)
    
    level = "Excellent" if score >= 90 else "Good" if score >= 75 else "Fair" if score >= 60 else "Poor"
    
    return {
        "enabled": True,
        "score": score,
        "level": level,
        "passed_checks": passed_checks[:5],  # Show top 5
        "violations": violations,
        "summary": f"{design_name} has {len(violations)} violations across {len(set(v['category'] for v in violations))} categories."
    }


def compare_brand_compliance(
    compliance_a: Dict, 
    compliance_b: Dict, 
    design_a_name: str = "Design A",
    design_b_name: str = "Design B"
) -> Dict:
    """
    Compare brand compliance between two designs.
    
    Args:
        compliance_a: Brand compliance for design A
        compliance_b: Brand compliance for design B
        design_a_name: Name of design A
        design_b_name: Name of design B
    
    Returns:
        Comparison dict
    """
    score_a = compliance_a.get('score', 0)
    score_b = compliance_b.get('score', 0)
    
    if score_a > score_b:
        winner = design_a_name
        delta = score_a - score_b
    elif score_b > score_a:
        winner = design_b_name
        delta = score_b - score_a
    else:
        winner = "Tied"
        delta = 0
    
    return {
        "winner": winner,
        "delta": delta,
        "design_a": {
            "name": design_a_name,
            "score": score_a,
            "level": compliance_a.get('level', 'Unknown'),
            "violations_count": len(compliance_a.get('violations', []))
        },
        "design_b": {
            "name": design_b_name,
            "score": score_b,
            "level": compliance_b.get('level', 'Unknown'),
            "violations_count": len(compliance_b.get('violations', []))
        }
    }
