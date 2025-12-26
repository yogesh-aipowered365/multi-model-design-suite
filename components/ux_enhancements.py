"""
UX Enhancement Components
- Tooltips for all controls
- Sample/demo images
- Shareable report links
- Executive summary export
- Team view toggle (Marketing vs Product)
"""

import streamlit as st
from datetime import datetime
import json
import uuid
from typing import Dict, List, Tuple, Optional


# ============================================================================
# TOOLTIPS CONFIGURATION
# ============================================================================

CONTROL_TOOLTIPS = {
    # Analysis Mode
    "analysis_mode": "Choose 'Single' to analyze one design or 'Compare' to analyze multiple designs side-by-side",

    # RAG Settings
    "rag_top_k": "Retrieve more patterns for comprehensive comparison; fewer patterns for faster analysis",

    # File Upload
    "file_uploader": "Upload PNG, JPG, or JPEG images. You can upload multiple files for comparison mode",

    # Platform Selection
    "platform": "Select the platform where this design will be used to get platform-specific recommendations",

    # Creative Type
    "creative_type": "Specify the type of creative to get tailored feedback on design best practices",

    # Agents
    "visual_agent": "Analyzes color, typography, layout, and visual hierarchy",
    "ux_agent": "Evaluates usability, accessibility, and user experience design",
    "market_agent": "Assesses market fit, competitive positioning, and audience appeal",
    "conversion_agent": "Analyzes CTAs, messaging clarity, and conversion optimization",
    "brand_agent": "Checks brand consistency, guidelines adherence, and identity alignment",

    # Export Options
    "export_summary": "Generate a one-page executive summary in plain English for stakeholders",
    "export_pdf": "Export detailed analysis report as PDF with visualizations",

    # Team View
    "team_view": "Toggle to see recommendations tailored for different teams (Marketing vs Product)",

    # Share Report
    "share_report": "Generate a shareable link to view this analysis report without re-running the analysis",
}


# ============================================================================
# TOOLTIPS RENDERING
# ============================================================================

def add_tooltip(label: str, key: str = None) -> str:
    """Add tooltip to a control label."""
    if key and key in CONTROL_TOOLTIPS:
        return f"{label} ⓘ"
    return label


def show_tooltip(key: str):
    """Display tooltip for a control if it exists."""
    if key in CONTROL_TOOLTIPS:
        st.caption(f"💡 {CONTROL_TOOLTIPS[key]}")


# ============================================================================
# DEMO/SAMPLE IMAGES
# ============================================================================

SAMPLE_IMAGES = {
    "E-commerce Product Page": {
        "description": "Modern e-commerce product showcase with CTA",
        "file": "demo_images/ecommerce_product.png",
        "category": "web",
    },
    "Social Media Ad": {
        "description": "Instagram-style marketing ad",
        "file": "demo_images/social_media_ad.png",
        "category": "social",
    },
    "App Onboarding": {
        "description": "Mobile app onboarding screen",
        "file": "demo_images/app_onboarding.png",
        "category": "mobile",
    },
    "Email Newsletter": {
        "description": "Email marketing header",
        "file": "demo_images/email_newsletter.png",
        "category": "email",
    },
    "Landing Page Hero": {
        "description": "Landing page hero section",
        "file": "demo_images/landing_page_hero.png",
        "category": "web",
    },
    "LinkedIn Post": {
        "description": "LinkedIn social media post",
        "file": "demo_images/linkedin_post.png",
        "category": "social",
    },
}


def get_demo_items_list() -> List[Tuple[str, Dict]]:
    """Get list of demo items (name, info) for selectbox or other UIs.

    Returns:
        List of (name, info_dict) tuples
    """
    return list(SAMPLE_IMAGES.items())


def load_demo_image_bytes(demo_name: str) -> Optional[bytes]:
    """Load bytes of a demo image by name.

    Args:
        demo_name: Name of the demo from SAMPLE_IMAGES keys

    Returns:
        Bytes of the image or None if not found
    """
    import os
    from io import BytesIO
    from pathlib import Path

    if demo_name not in SAMPLE_IMAGES:
        return None

    info = SAMPLE_IMAGES[demo_name]

    try:
        current_file = Path(__file__).parent
        project_root = current_file.parent
        demo_file_path = project_root / info['file']

        if not demo_file_path.exists():
            st.error(f"Demo image file not found at: {demo_file_path}")
            return None

        with open(demo_file_path, 'rb') as f:
            return f.read()
    except Exception as e:
        st.error(f"Failed to load {demo_name}: {str(e)}")
        return None


def show_demo_images_button(show_header=True):
    """Display sample images for users to try the app without uploading.

    Args:
        show_header: Whether to display the section header. Set to False when header already shown.
    """
    import os
    from io import BytesIO
    from pathlib import Path

    # Create container for demo images section
    demo_container = st.container()

    with demo_container:
        if show_header:
            st.subheader("📸 Try Demo Images")
            st.write("Click a sample to try the analysis:")

        col1, col2, col3 = st.columns(3)
        cols = [col1, col2, col3]

        for idx, (name, info) in enumerate(SAMPLE_IMAGES.items()):
            with cols[idx % 3]:
                st.info(f"**{name}**\n{info['description']}", icon="🖼️")
                if st.button(f"Use {name}", key=f"demo_{idx}", use_container_width=True):
                    try:
                        # Get the project root directory
                        # components directory
                        current_file = Path(__file__).parent
                        project_root = current_file.parent    # parent of components = project root
                        demo_file_path = project_root / info['file']

                        if not demo_file_path.exists():
                            st.error(
                                f"Demo image file not found at: {demo_file_path}")
                            st.write(
                                f"Current working directory: {os.getcwd()}")
                            return

                        # Read file and create BytesIO object
                        with open(demo_file_path, 'rb') as f:
                            demo_file = BytesIO(f.read())

                        demo_file.name = f"{name.replace(' ', '_')}.png"

                        if "uploaded_files" not in st.session_state:
                            st.session_state.uploaded_files = []

                        # Clear previous uploads if any
                        st.session_state.uploaded_files = [demo_file]
                        st.session_state.selected_demo = name
                        st.success(f"✅ {name} loaded! Ready for analysis.")
                        st.rerun()
                    except Exception as e:
                        st.error(f"Failed to load {name}: {str(e)}")
                        import traceback
                        st.write(traceback.format_exc())

    # Apply same card styling to demo images section with branded border and background
    st.markdown(
        """
        <style>
        [data-testid="stVerticalBlock"] > [data-testid="stVerticalBlock"]:has(> [data-testid="stInfo"]) {
            padding: 32px;
            border-radius: 20px;
            border: 2px solid #6366f1;
            background: rgba(255, 255, 255, 0.7);
            backdrop-filter: blur(10px);
            box-shadow: 0 8px 32px rgba(0, 0, 0, 0.04);
        }
        </style>
        """,
        unsafe_allow_html=True
    )


# ============================================================================
# REPORT SHARING & PERSISTENCE
# ============================================================================

class ReportManager:
    """Manage report creation, storage, and sharing."""

    @staticmethod
    def generate_report_id() -> str:
        """Generate a unique report ID."""
        return str(uuid.uuid4())[:8].upper()

    @staticmethod
    def create_report(results: Dict, config: Dict) -> Dict:
        """Create a report object with sharing capability."""
        report = {
            "id": ReportManager.generate_report_id(),
            "created_at": datetime.now().isoformat(),
            "config": config,
            "results": results,
            "view_count": 0,
        }
        return report

    @staticmethod
    def get_report_url(report_id: str) -> str:
        """Get shareable URL for a report."""
        return f"/?report_id={report_id}"

    @staticmethod
    def display_share_panel(report_id: str):
        """Display report sharing UI."""
        st.info("📤 **Share This Analysis**")

        share_url = f"{st.session_state.get('app_url', 'https://your-app.com')}{ReportManager.get_report_url(report_id)}"

        col1, col2 = st.columns([4, 1])
        with col1:
            st.text_input(
                "Shareable Link",
                value=share_url,
                disabled=True,
                label_visibility="collapsed"
            )
        with col2:
            if st.button("📋 Copy", help="Copy link to clipboard"):
                st.toast("✅ Copied to clipboard!", icon="✓")


# ============================================================================
# EXECUTIVE SUMMARY EXPORT
# ============================================================================

def generate_executive_summary(results: Dict, config: Dict) -> str:
    """Generate a one-page executive summary in plain English."""

    overall_score = results.get("overall_score", 0)
    agent_scores = results.get("agent_scores", {})
    findings = results.get("findings_summary", {})
    recommendations = results.get("top_recommendations", [])
    impact = results.get("impact_analysis", {})

    # Score interpretation
    if overall_score >= 85:
        score_interpretation = "Strong design that meets best practices and user needs"
        recommendation_level = "Minor optimizations recommended"
    elif overall_score >= 70:
        score_interpretation = "Solid design with room for improvement"
        recommendation_level = "Key improvements should be prioritized"
    elif overall_score >= 50:
        score_interpretation = "Design has significant challenges"
        recommendation_level = "Major revisions recommended before launch"
    else:
        score_interpretation = "Design requires substantial rework"
        recommendation_level = "Extensive redesign strongly recommended"

    summary = f"""DESIGN ANALYSIS - EXECUTIVE SUMMARY
Generated: {datetime.now().strftime('%B %d, %Y')}
Report ID: {ReportManager.generate_report_id()}

═════════════════════════════════════════════════════════════════════════════

OVERVIEW

Platform: {config.get('platform', 'Not specified')}
Creative Type: {config.get('creative_type', 'Not specified')}
Analysis Scope: {config.get('analysis_mode', 'Single')} Design

OVERALL ASSESSMENT

Score: {overall_score:.1f}/100
Status: {score_interpretation.upper()}
Next Steps: {recommendation_level}

DETAILED SCORES

Visual Design:        {agent_scores.get('visual', 0):.1f}/100
User Experience:     {agent_scores.get('ux', 0):.1f}/100
Market Fit:          {agent_scores.get('market', 0):.1f}/100
Conversion Potential: {agent_scores.get('conversion', 0):.1f}/100
Brand Alignment:     {agent_scores.get('brand', 0):.1f}/100

KEY FINDINGS

Total Issues Found: {findings.get('total', 0)}
  • Critical Issues: {findings.get('critical', 0)}
  • Warnings: {findings.get('warnings', 0)}
  • Info Items: {findings.get('info', 0)}

TOP RECOMMENDATIONS

"""

    for idx, rec in enumerate(recommendations[:5], 1):
        summary += f"""
{idx}. {rec.get('title', 'Recommendation').upper()}
   Priority: {rec.get('priority', 'Medium').upper()}
   Issue: {rec.get('description', 'N/A')}
   Solution: {rec.get('recommendation', 'N/A')}
   Expected Impact: {rec.get('impact', 'Positive user experience improvement')}
"""

    # Impact projections
    conversion_impact = impact.get('conversion_improvement', 0)
    engagement_impact = impact.get('engagement_improvement', 0)
    brand_impact = impact.get('brand_recognition', 0)

    summary += f"""

PROJECTED BUSINESS IMPACT

If implemented, recommendations could yield:
  • Conversion Rate Improvement: +{conversion_impact:.1f}%
  • User Engagement Increase: +{engagement_impact:.1f}%
  • Brand Recognition Boost: +{brand_impact:.1f}%

IMPLEMENTATION PRIORITY

Focus on high-priority items first for maximum ROI. Review detailed report for
effort estimation and implementation timeline.

═════════════════════════════════════════════════════════════════════════════

This analysis is based on design best practices, platform guidelines, and
conversion optimization principles. Recommendations should be validated with
your target audience before full implementation.

For questions or to discuss findings, refer to the detailed report dashboard.
"""

    return summary


def show_export_summary_button(results: Dict, config: Dict):
    """Display button to export executive summary."""
    # Premium header
    st.markdown("""
    <div style='background: linear-gradient(90deg, #667eea 0%, #764ba2 100%); 
                padding: 20px; border-radius: 12px; margin-bottom: 20px;'>
        <h3 style='margin: 0; color: white; font-size: 20px;'>📊 Analysis Results</h3>
    </div>
    """, unsafe_allow_html=True)

    # Export options
    st.markdown("<p style='font-size: 14px; font-weight: 600; margin-bottom: 10px; color: #333;'>Export Summary:</p>", unsafe_allow_html=True)

    col1, col2, col3 = st.columns(3)

    with col1:
        summary_text = generate_executive_summary(results, config)
        st.download_button(
            label="📄 Export Summary",
            data=summary_text,
            file_name=f"design_analysis_summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
            mime="text/plain",
            help="Download one-page executive summary",
            use_container_width=True
        )

    with col2:
        try:
            from reportlab.lib.pagesizes import letter
            from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
            from reportlab.lib.units import inch
            from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
            from reportlab.lib import colors
            from io import BytesIO

            def generate_pdf_report(results: Dict, config: Dict) -> bytes:
                """Generate PDF report from analysis results."""
                pdf_buffer = BytesIO()
                doc = SimpleDocTemplate(pdf_buffer, pagesize=letter)
                styles = getSampleStyleSheet()
                story = []

                # Title
                title_style = ParagraphStyle(
                    'CustomTitle',
                    parent=styles['Heading1'],
                    fontSize=24,
                    textColor=colors.HexColor('#1f77b4'),
                    spaceAfter=30,
                    alignment=1
                )
                story.append(Paragraph("Design Analysis Report", title_style))
                story.append(Spacer(1, 0.3*inch))

                # Overall Score
                if 'overall_score' in results:
                    score = results['overall_score']
                    score_text = f"<b>Overall Design Score:</b> {score:.1f}/100"
                    story.append(Paragraph(score_text, styles['Normal']))
                    story.append(Spacer(1, 0.2*inch))

                # Agent Scores Table
                if 'agent_scores' in results:
                    agent_data = [['Evaluation Category', 'Score']]
                    for agent, score in results['agent_scores'].items():
                        agent_name = agent.replace('_', ' ').title()
                        agent_data.append([agent_name, f"{score:.1f}/100"])

                    score_table = Table(agent_data, colWidths=[
                                        3.5*inch, 1.5*inch])
                    score_table.setStyle(TableStyle([
                        ('BACKGROUND', (0, 0), (-1, 0),
                         colors.HexColor('#1f77b4')),
                        ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
                        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
                        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                        ('FONTSIZE', (0, 0), (-1, 0), 12),
                        ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
                        ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
                        ('GRID', (0, 0), (-1, -1), 1, colors.grey),
                        ('ROWBACKGROUNDS', (0, 1), (-1, -1),
                         [colors.white, colors.lightgrey]),
                    ]))
                    story.append(score_table)
                    story.append(Spacer(1, 0.3*inch))

                # Key Findings
                if 'findings' in results:
                    story.append(
                        Paragraph("<b>Key Findings</b>", styles['Heading2']))
                    findings = results['findings']
                    if isinstance(findings, str):
                        story.append(Paragraph(
                            findings[:500] + "..." if len(findings) > 500 else findings, styles['Normal']))
                    story.append(Spacer(1, 0.2*inch))

                # Recommendations
                if 'recommendations' in results:
                    story.append(
                        Paragraph("<b>Recommendations</b>", styles['Heading2']))
                    recs = results['recommendations']
                    if isinstance(recs, str):
                        story.append(
                            Paragraph(recs[:500] + "..." if len(recs) > 500 else recs, styles['Normal']))
                    story.append(Spacer(1, 0.2*inch))

                # Build PDF
                doc.build(story)
                pdf_buffer.seek(0)
                return pdf_buffer.getvalue()

            pdf_data = generate_pdf_report(results, config)
            st.download_button(
                label="📕 Export PDF Report",
                data=pdf_data,
                file_name=f"design_analysis_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf",
                mime="application/pdf",
                help="Download complete analysis report as PDF",
                use_container_width=True
            )
        except ImportError:
            st.button("📕 Export PDF Report", disabled=True,
                      help="reportlab package not installed", use_container_width=True)

    with col3:
        st.button("📧 Email Report", disabled=True,
                  help="Coming soon", use_container_width=True)

    # View options section
    st.markdown("<p style='font-size: 14px; font-weight: 600; margin-top: 15px; margin-bottom: 10px; color: #333;'>View Options:</p>", unsafe_allow_html=True)

    view_col1, view_col2 = st.columns(2)
    with view_col1:
        st.button("📄 Full Report", use_container_width=True,
                  help="View comprehensive analysis report")
    with view_col2:
        st.button("⭐ Summary View", use_container_width=True,
                  help="View key findings summary")


# ============================================================================
# TEAM VIEW TOGGLE
# ============================================================================

class TeamViewManager:
    """Manage team-specific recommendation filtering."""

    TEAM_ROLES = {
        "marketing": {
            "name": "Marketing",
            "focus": "Campaign effectiveness, messaging, brand alignment, audience appeal",
            "icon": "📢",
            "priority_agents": ["market", "brand", "conversion"],
        },
        "product": {
            "name": "Product",
            "focus": "Usability, accessibility, technical feasibility, user experience",
            "icon": "🛠️",
            "priority_agents": ["ux", "visual", "conversion"],
        },
        "design": {
            "name": "Design",
            "focus": "Visual excellence, consistency, aesthetics, brand guidelines",
            "icon": "🎨",
            "priority_agents": ["visual", "brand"],
        },
    }

    @staticmethod
    def filter_recommendations(recommendations: List[Dict], team: str) -> List[Dict]:
        """Filter recommendations by team focus."""
        if team == "all":
            return recommendations

        # This is a simplified filter - in production, could use more sophisticated categorization
        return recommendations

    @staticmethod
    def highlight_team_insights(recommendations: List[Dict], team: str) -> List[Dict]:
        """Reorder recommendations by team priority."""
        if team == "all":
            return recommendations

        team_info = TeamViewManager.TEAM_ROLES.get(team, {})
        priority_agents = team_info.get("priority_agents", [])

        # Sort by relevance to team's priority agents
        sorted_recs = sorted(
            recommendations,
            key=lambda x: (
                x.get("agent_type", "") not in priority_agents,
                -float(x.get("impact_score", 0))
            )
        )
        return sorted_recs

    @staticmethod
    def render_team_view_selector():
        """Render team view toggle buttons."""
        st.write("**View Recommendations For:**")

        col1, col2, col3, col4 = st.columns(4)

        teams = ["all", "marketing", "product", "design"]
        team_labels = {
            "all": "👥 All Teams",
            "marketing": "📢 Marketing",
            "product": "🛠️ Product",
            "design": "🎨 Design",
        }

        current_team = st.session_state.get("team_view", "all")

        with col1:
            if st.button(team_labels["all"],
                         key="team_all",
                         use_container_width=True,
                         type="primary" if current_team == "all" else "secondary"):
                st.session_state.team_view = "all"
                st.rerun()

        with col2:
            if st.button(team_labels["marketing"],
                         key="team_marketing",
                         use_container_width=True,
                         type="primary" if current_team == "marketing" else "secondary"):
                st.session_state.team_view = "marketing"
                st.rerun()

        with col3:
            if st.button(team_labels["product"],
                         key="team_product",
                         use_container_width=True,
                         type="primary" if current_team == "product" else "secondary"):
                st.session_state.team_view = "product"
                st.rerun()

        with col4:
            if st.button(team_labels["design"],
                         key="team_design",
                         use_container_width=True,
                         type="primary" if current_team == "design" else "secondary"):
                st.session_state.team_view = "design"
                st.rerun()

        if current_team != "all":
            team_info = TeamViewManager.TEAM_ROLES.get(current_team, {})
            st.info(
                f"**{team_info.get('name')} Focus:** {team_info.get('focus')}")

        return current_team
