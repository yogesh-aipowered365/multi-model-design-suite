"""
Streamlit UI components for caching and guardrails.
Ready-to-use functions for integrating into app.py
"""

import streamlit as st
from typing import Dict, List, Tuple, Optional, Any

from components.caching import get_cache_manager
from components.guardrails import Guardrails, AnalysisMode, validate_and_downscale_images
from components.cache_integration import AnalysisOrchestrator


def show_guardrails_sidebar() -> Dict[str, Any]:
    """
    Display guardrails and mode selection in sidebar.

    Returns:
        Dict with user selections:
        - analysis_mode: 'fast', 'standard', or 'comprehensive'
        - enable_cache: Whether to use caching
        - show_estimates: Whether to show token/cost estimates
    """
    st.sidebar.header("⚙️ Analysis Settings")

    # Analysis mode selector
    analysis_mode = st.sidebar.selectbox(
        "Analysis Mode",
        options=["fast", "standard", "comprehensive"],
        help="Determines scope and cost of analysis",
        key="analysis_mode_selector"
    )

    # Show mode info
    if analysis_mode == "fast":
        fast_info = Guardrails.get_fast_mode_summary()
        st.sidebar.info(
            f"⚡ **Fast Mode**\n\n"
            f"**Agents ({len(fast_info['agents'])}):** {', '.join(fast_info['agents'])}\n\n"
            f"**Cost:** ~${fast_info['estimated_cost']:.3f}/image\n\n"
            f"**Time:** ~{fast_info['estimated_time_seconds']}s\n\n"
            f"**Benefits:**\n" +
            "\n".join(fast_info['benefits']) + "\n\n" +
            f"**Tradeoffs:**\n" +
            "\n".join(fast_info['tradeoffs'])
        )
    elif analysis_mode == "comprehensive":
        st.sidebar.info(
            f"🔬 **Comprehensive Mode**\n\n"
            f"All agents with deep analysis\n\n"
            f"**Cost:** ~$0.20/image\n\n"
            f"**Time:** ~120s"
        )
    else:  # standard
        std_info = Guardrails.get_standard_mode_summary()
        st.sidebar.info(
            f"⚙️ **Standard Mode** (Recommended)\n\n"
            f"**Agents ({len(std_info['agents'])}):** All 5 agents\n\n"
            f"**Cost:** ~${std_info['estimated_cost']:.3f}/image\n\n"
            f"**Time:** ~{std_info['estimated_time_seconds']}s\n\n" +
            "**Benefits:**\n" +
            "\n".join(std_info['benefits'])
        )

    # Caching toggle
    st.sidebar.divider()
    enable_cache = st.sidebar.checkbox(
        "🚀 Enable Caching",
        value=True,
        help="Cache embeddings, RAG results, and agent outputs for faster re-analysis"
    )

    # Show estimates toggle
    st.sidebar.divider()
    show_estimates = st.sidebar.checkbox(
        "📊 Show Cost Estimates",
        value=True,
        help="Display token and cost estimates before running analysis"
    )

    # Image limits info
    st.sidebar.divider()
    st.sidebar.caption(f"📸 Max {Guardrails.MAX_IMAGES} images per analysis")
    st.sidebar.caption(
        f"🖼️ Resolution: {Guardrails.MIN_WIDTH}x{Guardrails.MIN_HEIGHT} - {Guardrails.MAX_WIDTH}x{Guardrails.MAX_HEIGHT}px")
    st.sidebar.caption(f"💾 Max {Guardrails.MAX_FILE_SIZE_MB}MB per file")

    return {
        "analysis_mode": analysis_mode,
        "enable_cache": enable_cache,
        "show_estimates": show_estimates,
    }


def show_cache_statistics() -> None:
    """Display cache statistics in sidebar."""
    st.sidebar.divider()
    st.sidebar.subheader("📦 Cache Statistics")

    cache_manager = get_cache_manager()
    stats = cache_manager.get_cache_stats()

    # Summary metrics
    col1, col2 = st.sidebar.columns(2)
    with col1:
        st.metric(
            "Total Items",
            stats["total"]["count"],
            f"{stats['total']['size_mb']:.1f}MB"
        )
    with col2:
        st.metric("TTL", "24 hours")

    # Breakdown
    with st.sidebar.expander("Cache Breakdown", expanded=False):
        st.write(
            f"**Embeddings:** {stats['embeddings']['count']} items ({stats['embeddings']['size_mb']:.1f}MB)")
        st.write(
            f"**RAG Results:** {stats['rag']['count']} items ({stats['rag']['size_mb']:.1f}MB)")
        st.write(
            f"**Agent Results:** {stats['agents']['count']} items ({stats['agents']['size_mb']:.1f}MB)")

    # Clear cache button
    if st.sidebar.button("🗑️ Clear All Cache", key="clear_cache_button"):
        cache_manager.clear_cache()
        st.sidebar.success("✓ Cache cleared!")
        st.rerun()


def show_cost_estimate(
    num_images: int,
    analysis_mode: str,
    is_comparison: bool = False,
    enabled_agents: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """
    Show token and cost estimate before analysis.

    Args:
        num_images: Number of images to analyze
        analysis_mode: 'fast', 'standard', or 'comprehensive'
        is_comparison: Whether in comparison mode
        enabled_agents: Specific agents (if None, use mode default)

    Returns:
        Estimate dict for reference
    """
    try:
        mode = AnalysisMode(analysis_mode)
    except ValueError:
        mode = AnalysisMode.STANDARD

    estimate = Guardrails.estimate_token_usage(
        num_images=num_images,
        analysis_mode=mode,
        is_comparison=is_comparison,
        enabled_agents=enabled_agents,
    )

    # Display estimate in three columns
    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric(
            "📊 Estimated Tokens",
            Guardrails.format_token_display(estimate["total_tokens"]),
            f"Input: {estimate['total_image_tokens']}"
        )

    with col2:
        st.metric(
            "💵 Estimated Cost",
            f"${estimate['estimated_cost']:.4f}",
            f"Range: ${estimate['estimated_cost_range']['low']:.4f} - ${estimate['estimated_cost_range']['high']:.4f}"
        )

    with col3:
        st.metric(
            "⏱️ Est. Time",
            f"{estimate['processing_time_seconds']}s",
            f"{len(estimate['agents'])} agents"
        )

    st.divider()
    return estimate


def validate_and_process_images(
    uploaded_files: List,
) -> Tuple[bool, Dict[str, Any], List[str]]:
    """
    Validate and process uploaded images.

    Args:
        uploaded_files: List of Streamlit UploadedFile objects

    Returns:
        Tuple of (success, processed_images_dict, warnings)
    """
    if not uploaded_files:
        return False, {}, ["Please upload at least one image"]

    # Convert to (filename, bytes) tuples
    images_list = [(f.name, f.getvalue()) for f in uploaded_files]

    # Validate
    is_valid, errors, warnings = Guardrails.validate_images(images_list)

    if not is_valid:
        return False, {}, errors

    # Process and downscale
    try:
        processed = {}
        images_dict = {f.name: f.getvalue() for f in uploaded_files}
        processed_images, downscale_warnings = validate_and_downscale_images(
            images_dict)

        # Convert PIL images to bytes
        from PIL import Image
        import io

        processed_bytes = {}
        for filename, image in processed_images.items():
            buf = io.BytesIO()
            image.save(buf, format=image.format or "PNG")
            processed_bytes[filename] = buf.getvalue()

        return True, processed_bytes, warnings + downscale_warnings

    except Exception as e:
        return False, {}, [f"Processing error: {str(e)}"]


def show_analysis_confirmation(
    num_images: int,
    analysis_mode: str,
    is_comparison: bool = False,
    enable_cache: bool = True,
) -> Tuple[bool, Dict[str, Any]]:
    """
    Show confirmation dialog with estimate and settings.

    Args:
        num_images: Number of images
        analysis_mode: Analysis mode
        is_comparison: Whether in comparison mode
        enable_cache: Whether cache is enabled

    Returns:
        Tuple of (confirmed, estimate_dict)
    """
    # Get estimate
    estimate = Guardrails.estimate_token_usage(
        num_images=num_images,
        analysis_mode=analysis_mode,
        is_comparison=is_comparison,
    )

    # Show confirmation
    st.warning("📋 Please review before analysis:")

    col1, col2, col3 = st.columns(3)
    with col1:
        st.write(f"**Images:** {num_images}")
    with col2:
        st.write(f"**Mode:** {analysis_mode.title()}")
    with col3:
        st.write(f"**Cache:** {'✓ Enabled' if enable_cache else '✗ Disabled'}")

    st.divider()

    # Cost/token breakdown
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Est. Tokens", Guardrails.format_token_display(
            estimate["total_tokens"]))
    with col2:
        st.metric("Est. Cost", f"${estimate['estimated_cost']:.4f}")
    with col3:
        st.metric("Est. Time", f"{estimate['processing_time_seconds']}s")

    st.divider()

    # Confirmation buttons
    col1, col2 = st.columns([3, 1])
    with col1:
        st.info("✓ Ready to analyze. Click button to proceed.")
    with col2:
        confirmed = st.button(
            "▶️ Analyze", use_container_width=True, type="primary")

    return confirmed, estimate


def show_execution_summary(
    orchestrator: AnalysisOrchestrator,
    total_time_seconds: float,
) -> None:
    """
    Show execution summary with cache stats and actual costs.

    Args:
        orchestrator: AnalysisOrchestrator instance
        total_time_seconds: Total execution time
    """
    summary = orchestrator.get_execution_summary()

    st.divider()
    st.subheader("✅ Analysis Complete")

    col1, col2, col3 = st.columns(3)

    with col1:
        cache_stats = summary.get("cache_stats", {})
        hit_rate = cache_stats.get("cache_hit_rate", 0)
        st.metric(
            "🚀 Cache Hit Rate",
            f"{hit_rate:.1%}",
            f"{cache_stats.get('cache_hits', 0)} hits"
        )

    with col2:
        token_stats = summary.get("token_summary", {})
        actual = token_stats.get("actual", 0)
        st.metric(
            "📊 Actual Tokens",
            Guardrails.format_token_display(actual),
            f"Est: {Guardrails.format_token_display(token_stats.get('estimated', 0))}"
        )

    with col3:
        st.metric(
            "⏱️ Total Time",
            f"{total_time_seconds:.1f}s",
            f"Est: {summary.get('token_summary', {}).get('estimated', 0) // 100}s"
        )

    # Show if cache was used
    if cache_stats.get("cache_hits", 0) > 0:
        st.success(
            f"💾 Saved ~{cache_stats.get('cache_hits', 0) * 5} seconds with caching "
            f"({cache_stats.get('cache_hit_rate', 0):.1%} hit rate)"
        )


def show_analysis_insights(
    analysis_results: Dict[str, Any],
    enable_cache: bool = True,
) -> None:
    """
    Show insights about the analysis (cache usage, fast mode benefits, etc.).

    Args:
        analysis_results: Results dict from analysis
        enable_cache: Whether cache was enabled
    """
    with st.expander("💡 Analysis Insights", expanded=False):
        # Check for cache hits
        cached_count = analysis_results.get("num_cached_results", 0)
        if cached_count > 0 and enable_cache:
            st.success(
                f"✓ **{cached_count} cached result(s) used** - Analysis was {cached_count * 5}-{cached_count * 10}s faster!"
            )

        # Show cache stats
        cache_stats = analysis_results.get("cache_stats", {})
        if cache_stats:
            st.write("**Cache Statistics:**")
            st.json(cache_stats)

        # Fast mode insights
        if analysis_results.get("execution_mode") == "fast":
            st.info(
                "⚡ **Fast Mode Analysis**\n\n"
                "This analysis used only 2 core agents (Market Research & Conversion CTA) "
                "for speed. Consider running in Standard Mode for comprehensive feedback."
            )

        # Token usage
        token_info = analysis_results.get("token_summary", {})
        if token_info:
            st.write("**Token Usage:**")
            st.write(
                f"- Estimated: {Guardrails.format_token_display(token_info.get('estimated', 0))}")
            st.write(
                f"- Actual: {Guardrails.format_token_display(token_info.get('actual', 0))}")
            if token_info.get("estimate_accuracy"):
                st.write(
                    f"- Accuracy: {token_info.get('estimate_accuracy'):.1%}")


# ============================================================================
# Helper Components
# ============================================================================

def show_image_constraints() -> None:
    """Show image constraints in a collapsible section."""
    with st.expander("📸 Image Constraints", expanded=False):
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.write(f"**Max Images:** {Guardrails.MAX_IMAGES}")
        with col2:
            st.write(
                f"**Min Resolution:** {Guardrails.MIN_WIDTH}x{Guardrails.MIN_HEIGHT}px")
        with col3:
            st.write(
                f"**Max Resolution:** {Guardrails.MAX_WIDTH}x{Guardrails.MAX_HEIGHT}px")
        with col4:
            st.write(f"**Max File Size:** {Guardrails.MAX_FILE_SIZE_MB}MB")


def show_mode_comparison() -> None:
    """Show comparison of all analysis modes."""
    st.subheader("Analysis Mode Comparison")

    modes = {
        "Fast": Guardrails.get_fast_mode_summary(),
        "Standard": Guardrails.get_standard_mode_summary(),
        "Comprehensive": {
            "name": "Comprehensive",
            "agents": Guardrails.STANDARD_MODE_AGENTS,
            "estimated_cost": 0.20,
            "estimated_time_seconds": 120,
        }
    }

    # Create comparison table
    comparison_data = {
        "Mode": list(modes.keys()),
        "Agents": [len(m["agents"]) for m in modes.values()],
        "Cost/Image": [f"${m['estimated_cost']:.3f}" for m in modes.values()],
        "Time": [f"{m['estimated_time_seconds']}s" for m in modes.values()],
    }

    try:
        import pandas as pd
        df = pd.DataFrame(comparison_data)
        st.dataframe(df, use_container_width=True, hide_index=True)
    except ImportError:
        # Fallback: display as text
        for key, values in comparison_data.items():
            st.write(f"{key}: {values}")
