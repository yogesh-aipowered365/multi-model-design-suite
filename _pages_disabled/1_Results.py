"""Results Page - Displays analysis output after wizard completion"""
import streamlit as st
from pathlib import Path

# Import theme
from app.ui.theme import apply_aip365_theme
from app.ui.components import hero_panel, status_banner, helper_text

# Import visualization and export
from components.enhanced_output import (
    generate_score_gauge_chart,
    generate_comparison_radar_chart,
    generate_category_breakdown_chart,
)
from components.ux_enhancements import (
    ReportManager,
    show_export_summary_button,
    TeamViewManager,
)

# Apply theme
apply_aip365_theme("AIpowered365 Labs - Results", "📊")

# Initialize session state check
if "analysis_complete" not in st.session_state or not st.session_state.analysis_complete:
    status_banner(
        "⚠️ No analysis to display. Please complete the wizard to generate results.",
        "warning"
    )

    col1, col2 = st.columns([0.6, 0.4])
    with col1:
        if st.button("← Start New Analysis", use_container_width=True, type="primary"):
            st.session_state.analysis_complete = False
            st.session_state.current_step = 1
            st.session_state.context_confirmed = False
            st.session_state.selected_agents = [
                "visual_analysis", "ux_critique"]
            st.switch_page(
                "pages/app.py" if Path("pages/app.py").exists() else "app.py")
    st.stop()

# Get analysis results
results = st.session_state.get("analysis_results", {})
if not results:
    status_banner("❌ No analysis results found.", "error")
    st.stop()

# Render hero
hero_panel(
    title="Analysis Complete",
    subtitle="Your design has been analyzed by our AI engines. Review findings below.",
    pills=["Detailed Insights", "Actionable Recommendations", "Impact Metrics"],
    show_logo=False
)

st.markdown("")

# Status banner
status_banner(
    f"✅ Analysis complete! {len(st.session_state.selected_agents)} agents analyzed {len(st.session_state.uploads)} design(s).",
    "success"
)

st.markdown("")

# Export section
st.markdown("### 📤 Export & Share")
col1, col2 = st.columns(2)
with col1:
    st.markdown("**Share Report:**")
    storage_config = {
        "platform": st.session_state.platform,
        "creative_type": st.session_state.creative_type,
        "analysis_mode": "Compare" if st.session_state.compare_mode else "Single",
        "enabled_agents": st.session_state.selected_agents
    }
    report = ReportManager.create_report(results, storage_config)
    ReportManager.display_share_panel(report["id"])

with col2:
    st.markdown("**Download Report:**")
    show_export_summary_button(results, storage_config)

st.divider()

# Results tabs
st.markdown("### 📊 Detailed Results")

tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "🎯 Overview",
    "✨ Recommendations",
    "📈 Impact Analysis",
    "👁️ Visual Feedback",
    "📋 Detailed Data"
])

with tab1:
    st.markdown("#### Overall Score")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Overall Score",
                  f"{results.get('overall_score', 0):.1f}/100")
    with col2:
        findings = results.get("findings_summary", {})
        st.metric("Total Findings", findings.get("total", 0))
    with col3:
        st.metric("Critical Issues", findings.get("critical", 0))

    st.markdown("#### Agent Scores")
    col1, col2 = st.columns(2)
    with col1:
        st.plotly_chart(
            generate_score_gauge_chart(
                results.get("agent_scores", {}).get("visual", 0),
                "Visual Analysis"
            ),
            use_container_width=True
        )
    with col2:
        st.plotly_chart(
            generate_score_gauge_chart(
                results.get("agent_scores", {}).get("ux", 0),
                "UX Critique"
            ),
            use_container_width=True
        )

with tab2:
    st.markdown("#### Top Recommendations")
    recommendations = results.get("top_recommendations", [])
    for idx, rec in enumerate(recommendations, 1):
        with st.expander(f"{idx}. {rec.get('title', 'Recommendation')} - {rec.get('priority', 'medium').upper()}"):
            st.markdown(f"**Description:** {rec.get('description', '')}")
            st.markdown(f"**Impact:** {rec.get('impact', '')}")
            st.markdown(f"**Recommendation:** {rec.get('recommendation', '')}")

with tab3:
    st.markdown("#### Impact Projections")
    impact = results.get("impact_analysis", {})
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric(
            "Conversion Improvement",
            f"+{impact.get('conversion_improvement', 0):.1f}%"
        )
    with col2:
        st.metric(
            "Engagement Improvement",
            f"+{impact.get('engagement_improvement', 0):.1f}%"
        )
    with col3:
        st.metric(
            "Brand Recognition",
            f"+{impact.get('brand_recognition', 0):.1f}%"
        )

with tab4:
    st.markdown("#### Visual Feedback")
    feedback = results.get("visual_feedback", {})
    if feedback:
        st.json(feedback)
    else:
        st.info("No visual feedback annotations available")

with tab5:
    st.markdown("#### Raw Data")
    st.json(results)

st.divider()

# Navigation
st.markdown("### 🎯 Next Steps")
col1, col2 = st.columns([0.6, 0.4])
with col1:
    if st.button("← Start New Analysis", use_container_width=True, type="secondary"):
        # Reset wizard state
        st.session_state.analysis_complete = False
        st.session_state.current_step = 1
        st.session_state.context_confirmed = False
        st.session_state.selected_agents = ["visual_analysis", "ux_critique"]
        st.session_state.uploads = []
        st.session_state.analysis_results = None
        st.switch_page("app.py")
