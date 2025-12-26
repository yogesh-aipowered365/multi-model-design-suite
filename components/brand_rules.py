"""
Brand Rules Management
Handles parsing, normalization, and validation of user-provided brand rules.
"""

import json
import re
from typing import Dict, List, Optional, Tuple


def extract_text_from_upload(uploaded_file) -> Tuple[str, str]:
    """
    Extract text from uploaded file (TXT, JSON, DOCX, PDF).
    
    Args:
        uploaded_file: Streamlit UploadedFile object
    
    Returns:
        Tuple of (extracted_text, format_note)
    """
    if uploaded_file is None:
        return "", ""
    
    filename = uploaded_file.name.lower()
    
    try:
        if filename.endswith('.txt'):
            # Read TXT
            text = uploaded_file.read().decode('utf-8', errors='replace')
            return text, "TXT file"
        
        elif filename.endswith('.json'):
            # Read JSON
            content = json.loads(uploaded_file.read().decode('utf-8'))
            if isinstance(content, dict):
                # Try to extract rules or pretty-print
                if 'rules' in content:
                    rules = content['rules']
                    if isinstance(rules, list):
                        text = "\n".join([str(r) for r in rules])
                    else:
                        text = json.dumps(content, indent=2)
                else:
                    text = json.dumps(content, indent=2)
            else:
                text = json.dumps(content, indent=2)
            return text, "JSON file"
        
        elif filename.endswith('.docx'):
            # Read DOCX
            try:
                from docx import Document
                doc = Document(uploaded_file)
                text = "\n".join([para.text for para in doc.paragraphs if para.text.strip()])
                return text, "DOCX file"
            except ImportError:
                return "", "DOCX support not available (python-docx not installed)"
        
        elif filename.endswith('.pdf'):
            # Read PDF
            try:
                import PyPDF2
                pdf_reader = PyPDF2.PdfReader(uploaded_file)
                text = ""
                for page_num in range(len(pdf_reader.pages)):
                    page = pdf_reader.pages[page_num]
                    text += page.extract_text() + "\n"
                return text, "PDF file"
            except ImportError:
                return "", "PDF support not available (PyPDF2 not installed)"
        
        else:
            return "", f"Unsupported file type: {filename}"
    
    except Exception as e:
        return "", f"Error reading file: {str(e)}"


def normalize_brand_rules(raw_text: str, max_length: int = 30000) -> Dict:
    """
    Normalize raw brand rules text into structured format.
    
    Args:
        raw_text: Raw text from any source
        max_length: Max chars to store (warn if truncated)
    
    Returns:
        Dict with keys: raw_text, rules, voice_tone, colors, typography
    """
    
    if not raw_text or not raw_text.strip():
        return {
            "raw_text": "",
            "rules": [],
            "voice_tone": {},
            "colors": {},
            "typography": {},
            "truncated": False
        }
    
    # Truncate if needed
    truncated = len(raw_text) > max_length
    if truncated:
        raw_text = raw_text[:max_length] + "\n[... truncated ...]"
    
    # Parse rules from text
    rules = _parse_rules_from_text(raw_text)
    
    # Detect sections
    voice_tone = _extract_section(raw_text, ["tone", "voice", "language", "writing"])
    colors = _extract_section(raw_text, ["color", "hex", "palette"])
    typography = _extract_section(raw_text, ["font", "typography", "type", "typeface"])
    
    return {
        "raw_text": raw_text,
        "rules": rules,
        "voice_tone": voice_tone,
        "colors": colors,
        "typography": typography,
        "truncated": truncated
    }


def _parse_rules_from_text(text: str) -> List[Dict]:
    """
    Parse individual rules from text.
    Heuristics: split by bullets/numbers/newlines, assign IDs.
    """
    rules = []
    rule_id_counter = 1
    
    # Split by common delimiters
    lines = text.split('\n')
    
    for line in lines:
        line = line.strip()
        if not line or len(line) < 10:
            continue
        
        # Remove common prefixes (numbers, bullets)
        clean_line = re.sub(r'^[\d\-\*\•\.]+\s+', '', line)
        clean_line = clean_line.strip()
        
        if not clean_line:
            continue
        
        # Detect category by keywords
        category = _detect_category(clean_line)
        
        # Detect priority by keywords
        priority = _detect_priority(clean_line)
        
        rule = {
            "id": f"BR-{rule_id_counter:03d}",
            "category": category,
            "rule": clean_line,
            "priority": priority
        }
        rules.append(rule)
        rule_id_counter += 1
    
    return rules[:100]  # Limit to 100 rules to avoid explosion


def _detect_category(text: str) -> str:
    """Detect category by keywords."""
    text_lower = text.lower()
    
    if any(k in text_lower for k in ['logo', 'brand mark', 'symbol']):
        return "Logo"
    elif any(k in text_lower for k in ['color', 'hex', 'palette', 'primary', 'secondary']):
        return "Color"
    elif any(k in text_lower for k in ['font', 'typography', 'typeface', 'serif', 'sans']):
        return "Typography"
    elif any(k in text_lower for k in ['tone', 'voice', 'language', 'writing']):
        return "Tone & Voice"
    elif any(k in text_lower for k in ['imagery', 'image', 'photo', 'illustration']):
        return "Imagery"
    elif any(k in text_lower for k in ['spacing', 'margin', 'padding', 'layout']):
        return "Layout & Spacing"
    elif any(k in text_lower for k in ['cta', 'call', 'button', 'action']):
        return "CTA"
    elif any(k in text_lower for k in ['legal', 'compliance', 'gdpr', 'terms', 'privacy']):
        return "Legal"
    elif any(k in text_lower for k in ['accessibility', 'a11y', 'wcag', 'contrast']):
        return "Accessibility"
    else:
        return "General"


def _detect_priority(text: str) -> str:
    """Detect priority level by keywords."""
    text_lower = text.lower()
    
    if any(k in text_lower for k in ['must', 'never', 'required', 'mandatory', 'always']):
        return "high"
    elif any(k in text_lower for k in ['should', 'recommended', 'prefer']):
        return "medium"
    else:
        return "low"


def _extract_section(text: str, keywords: List[str]) -> Dict:
    """
    Extract a section from text by keywords.
    Returns dict with extracted content.
    """
    text_lower = text.lower()
    found_keywords = [kw for kw in keywords if kw.lower() in text_lower]
    
    if not found_keywords:
        return {}
    
    # Try to find section headers and content
    result = {}
    for kw in found_keywords:
        pattern = rf'{kw}[:\s]+(.*?)(?=\n[A-Z]|$)'
        matches = re.findall(pattern, text, re.IGNORECASE | re.DOTALL)
        if matches:
            result[kw] = matches[0].strip()[:200]  # Limit to 200 chars per section
    
    return result


def format_rules_for_prompt(brand_rules: Dict, max_rules: int = 25) -> str:
    """
    Format brand rules for inclusion in analysis prompt.
    
    Args:
        brand_rules: Normalized rules dict
        max_rules: Max rules to include (to avoid token explosion)
    
    Returns:
        Formatted string for prompt
    """
    if not brand_rules or not brand_rules.get('rules'):
        return ""
    
    prompt_text = "BRAND RULES:\n\n"
    
    rules = brand_rules['rules'][:max_rules]
    
    # Group by category
    by_category = {}
    for rule in rules:
        cat = rule.get('category', 'General')
        if cat not in by_category:
            by_category[cat] = []
        by_category[cat].append(rule)
    
    # Format by category
    for category, category_rules in sorted(by_category.items()):
        prompt_text += f"### {category}\n"
        for rule in category_rules:
            priority_marker = "🔴" if rule['priority'] == 'high' else "🟡" if rule['priority'] == 'medium' else "🟢"
            prompt_text += f"  {priority_marker} [{rule['id']}] {rule['rule']}\n"
        prompt_text += "\n"
    
    return prompt_text
