"""3-Step Wizard Flow for AIpowered365 Labs - FIXED VERSION"""
import streamlit as st
from pathlib import Path
from typing import List, Tuple
import hashlib
import json
from datetime import datetime
import urllib.parse
from app.ui.components import (
    status_banner,
    helper_text,
    primary_button,
    info_grid,
    render_upload_preview_panel,
    spacer,
    render_context_summary_card
)
from app.wizard.state import is_step_complete, advance_to_step
from app.wizard.agents import get_available_agents, get_preset_agents
from components.ux_enhancements import show_demo_images_button, get_demo_items_list, load_demo_image_bytes
from components.enhanced_output import generate_score_gauge_chart


PLATFORM_OPTIONS = ["Instagram", "LinkedIn", "Twitter/X",
                    "Facebook", "Pinterest", "Web", "App Store", "Email"]
CREATIVE_TYPE_OPTIONS = [
    "Marketing Creative",
    "Product UI/App Screen",
    "Landing Page",
    "Email Header",
    "Social Media Ad",
    "Banner",
    "Other"
]


def _calculate_file_hash(file_bytes: bytes) -> str:
    """Calculate SHA256 hash of file bytes."""
    return hashlib.sha256(file_bytes).hexdigest()[:8]


def render_multi_chip_group(
    title: str,
    options: List[str],
    state_key: str,
    allow_empty: bool = True,
    wrapper_class: str = ""
) -> List[str]:
    """
    Render a multi-select chip group with branded pill styling using st.multiselect().
    
    Renders as BaseWeb tags which can be styled via CSS selectors.

    Args:
        title: Group title to display
        options: List of option strings
        state_key: Session state key for storing selections (e.g., 'selected_platforms')
        allow_empty: Whether to allow no selection (default True)
        wrapper_class: Optional CSS class to wrap the multiselect (e.g., 'aip365-step2-platform')

    Returns:
        List of selected options
    """
    # Initialize state if needed
    if state_key not in st.session_state:
        st.session_state[state_key] = []

    # Create outer wrapper if class provided
    if wrapper_class:
        st.markdown(f'<div class="{wrapper_class}">', unsafe_allow_html=True)

    # Render title and clear button in columns
    col_title, col_clear = st.columns([0.90, 0.10])

    with col_title:
        st.markdown(f"#### {title}")

    with col_clear:
        # "Clear" button on the right
        if st.button(
            "Clear",
            key=f"{state_key}_clear_btn",
            help=f"Clear all {title.lower()} selections"
        ):
            st.session_state[state_key] = []
            st.rerun()

    # Use multiselect which renders as BaseWeb tags (styleable via CSS)
    selected = st.multiselect(
        label=title,
        options=options,
        default=st.session_state.get(state_key, []),
        key=f"{state_key}_multiselect",
        label_visibility="collapsed",
        placeholder="Select one or more…"
    )

    # Update session state with selected values
    st.session_state[state_key] = selected

    # Close wrapper if class provided
    if wrapper_class:
        st.markdown('</div>', unsafe_allow_html=True)

    # Add spacing
    st.markdown("<div style='height: 8px'></div>", unsafe_allow_html=True)

    return selected


def render_upload_step() -> Tuple[bool, bool]:
    """
    Render Step 1: Upload Designs with two-column layout.

    Left column (40%): Upload controls, analysis mode, demo selection
    Right column (60%): Image preview panel

    Returns:
        Tuple of (is_complete, should_advance)
    """
    # Initialize preview state if needed
    if "preview_image_bytes" not in st.session_state:
        st.session_state.preview_image_bytes = None
    if "preview_image_name" not in st.session_state:
        st.session_state.preview_image_name = None

    # Two-column layout: left for controls, right for preview
    left, right = st.columns([0.42, 0.58], gap="large")

    # ========================================================================
    # LEFT COLUMN: UPLOAD CONTROLS & DEMO SELECTION
    # ========================================================================
    with left:
        # === Analysis Mode Section ===
        analysis_container = st.container()
        with analysis_container:
            st.markdown("#### Analysis Mode")
            single_mode = st.radio(
                "Analysis Mode",
                ["Analyze Single Design", "Compare Designs"],
                key="analysis_mode_radio",
                label_visibility="collapsed",
                index=0 if not st.session_state.get("compare_mode") else 1
            )
            st.session_state.compare_mode = single_mode == "Compare Designs"

        # Vertical spacing between sections
        st.markdown("<div style='height: 24px'></div>", unsafe_allow_html=True)

        # === Upload Images Section ===
        upload_container = st.container()
        with upload_container:
            st.markdown("#### Upload Images")
            uploaded_files = st.file_uploader(
                "Choose design files",
                type=["png", "jpg", "jpeg", "webp"],
                accept_multiple_files=st.session_state.compare_mode,
                key="file_uploader_widget",
                label_visibility="collapsed"
            )

            # Process uploaded files into structured format
            if uploaded_files:
                files_list = uploaded_files if isinstance(
                    uploaded_files, list) else [uploaded_files]
                st.session_state.uploads = []
                for file in files_list:
                    file_bytes = file.read()
                    file.seek(0)

                    upload_item = {
                        "name": file.name,
                        "bytes": file_bytes,
                        "size": len(file_bytes),
                        "mime": file.type,
                        "hash": _calculate_file_hash(file_bytes)
                    }
                    st.session_state.uploads.append(upload_item)

                st.session_state.uploaded_files = uploaded_files  # Legacy
                status_banner(
                    f"✅ {len(files_list)} design(s) uploaded successfully", "success")

                # Update preview with first uploaded image
                st.session_state.preview_image_bytes = st.session_state.uploads[0]["bytes"]
                st.session_state.preview_image_name = st.session_state.uploads[0]["name"]

        # Vertical spacing between sections
        st.markdown("<div style='height: 24px'></div>", unsafe_allow_html=True)

        # === Demo Selection Section ===
        demo_container = st.container()
        with demo_container:
            st.markdown("#### Demo Images")
            helper_text("Choose a demo design to get started quickly.")

            demo_items = get_demo_items_list()
            demo_names = [name for name, _ in demo_items]

            demo_choice = st.selectbox(
                "Select a demo design",
                options=demo_names,
                key="demo_selectbox",
                label_visibility="collapsed"
            )

            # Add breathing room before action button
            st.markdown("<div style='height: 12px'></div>",
                        unsafe_allow_html=True)

            if st.button("📸 Use Selected Demo", use_container_width=True, key="use_demo_btn"):
                # Load demo image bytes
                demo_bytes = load_demo_image_bytes(demo_choice)
                if demo_bytes:
                    # Create upload structure for consistency
                    st.session_state.uploads = [{
                        "name": f"{demo_choice.replace(' ', '_')}.png",
                        "bytes": demo_bytes,
                        "size": len(demo_bytes),
                        "mime": "image/png",
                        "hash": _calculate_file_hash(demo_bytes)
                    }]
                    st.session_state.uploaded_files = []  # Demo, not uploaded but maintain list
                    st.session_state.selected_demo = demo_choice

                    # Update preview
                    st.session_state.preview_image_bytes = demo_bytes
                    st.session_state.preview_image_name = demo_choice

                    status_banner(
                        f"✅ {demo_choice} loaded! Ready for analysis.", "success")
                    st.rerun()

    # ========================================================================
    # RIGHT COLUMN: PREVIEW PANEL (sticky feel)
    # ========================================================================
    with right:
        # Preview card container
        preview_container = st.container()
        with preview_container:
            st.markdown("#### Preview")
            st.markdown("<div style='height: 8px'></div>",
                        unsafe_allow_html=True)

            # Preview container with light styling
            preview_box = st.container(border=True)

            with preview_box:
                if st.session_state.preview_image_bytes:
                    try:
                        st.image(
                            st.session_state.preview_image_bytes,
                            use_container_width=True,
                            caption=st.session_state.preview_image_name
                        )

                        # Show thumbnails in compare mode
                        if st.session_state.compare_mode and len(st.session_state.uploads) > 1:
                            st.markdown(
                                "<div style='height: 16px'></div>", unsafe_allow_html=True)
                            st.markdown("##### Other Images")
                            thumb_cols = st.columns(2)
                            for idx, upload in enumerate(st.session_state.uploads[1:], 1):
                                with thumb_cols[idx % 2]:
                                    st.image(
                                        upload["bytes"],
                                        use_container_width=True,
                                        caption=upload["name"]
                                    )

                            # Count badge
                            st.markdown(
                                "<div style='height: 12px'></div>", unsafe_allow_html=True)
                            st.caption(
                                f"📊 Total: {len(st.session_state.uploads)} images selected")

                    except Exception as e:
                        st.error(f"Could not load preview: {str(e)}")
                else:
                    # Empty state
                    st.markdown(
                        """
                        <div style="text-align: center; padding: 40px 20px; color: #888;">
                            <p style="font-size: 48px; margin: 0;">📷</p>
                            <p style="font-size: 14px; margin-top: 16px;">
                                Upload a design image or select a demo to preview
                            </p>
                        </div>
                        """,
                        unsafe_allow_html=True
                    )

    # ========================================================================
    # NAVIGATION (FULL WIDTH, BELOW COLUMNS)
    # ========================================================================
    st.markdown("<div style='height: 32px'></div>", unsafe_allow_html=True)
    is_complete = is_step_complete(1)

    col1, col2 = st.columns([0.6, 0.4])
    with col1:
        next_clicked = st.button(
            "→ Next: Set Context",
            key="step1_next_btn",
            disabled=not is_complete,
            use_container_width=True,
            type="primary" if is_complete else "secondary"
        )

    if next_clicked and is_complete:
        st.session_state.current_step = 2
        st.rerun()

    return is_complete, next_clicked


def render_step_2_context() -> Tuple[bool, bool]:
    """
    Render Step 2: Set Context with two-column layout.

    Left column (42%): Platform, Creative Type, RAG, API Key, Confirm button
    Right column (58%): Uploaded Images Preview panel

    Returns:
        Tuple of (is_complete, should_advance)
    """
    st.markdown("## 🎯 Set Context")
    helper_text(
        "Configure your design analysis parameters.")

    # Two-column layout: left for controls, right for preview
    left, right = st.columns([0.42, 0.58], gap="large")

    # ========================================================================
    # LEFT COLUMN: CONFIGURATION CONTROLS
    # ========================================================================
    with left:
        # WRAPPER FOR STEP 2 PILL STYLING - wraps both Platform and Creative Type sections
        st.markdown('<div class="aip365-step2-pillwrap">',
                    unsafe_allow_html=True)

        # Platform section - Multi-select chips
        selected_platforms = render_multi_chip_group(
            "Platform",
            PLATFORM_OPTIONS,
            "selected_platforms",
            allow_empty=True,
            wrapper_class="aip365-step2-platform"
        )
        # Update backward-compatible single selection
        st.session_state.platform = selected_platforms[0] if selected_platforms else "Instagram"

        # Creative Type section - Multi-select chips
        selected_creative_types = render_multi_chip_group(
            "Creative Type",
            CREATIVE_TYPE_OPTIONS,
            "selected_creative_types",
            allow_empty=True,
            wrapper_class="aip365-step2-creative"
        )
        # Update backward-compatible single selection
        st.session_state.creative_type = selected_creative_types[
            0] if selected_creative_types else "Marketing Creative"

        # CLOSE STEP 2 PILL STYLING WRAPPER
        st.markdown('</div>', unsafe_allow_html=True)

        st.markdown("<div style='height: 24px'></div>", unsafe_allow_html=True)

        # Brand Rules section (Optional)
        st.markdown("#### Brand Rules (Optional)")
        with st.expander("📋 Upload or Paste Brand Guidelines", expanded=False):
            st.caption(
                "Provide brand rules as PDF, TXT, DOCX, or JSON (or paste text)")

            # File uploader
            brand_rules_file = st.file_uploader(
                "Upload brand guidelines file",
                type=["pdf", "txt", "docx", "json"],
                key="brand_rules_file",
                label_visibility="collapsed",
                help="PDF, TXT, DOCX, or JSON with brand rules"
            )

            # Text paste option
            brand_rules_text = st.text_area(
                "Or paste brand rules directly",
                key="brand_rules_text",
                height=160,
                label_visibility="collapsed",
                placeholder="Paste your brand guidelines, rules, or tone-of-voice here..."
            )

            # Process brand rules if provided
            if brand_rules_file or brand_rules_text:
                from components.brand_rules import extract_text_from_upload, normalize_brand_rules

                # Extract text from file if provided
                if brand_rules_file:
                    extracted_text, format_note = extract_text_from_upload(
                        brand_rules_file)
                    if extracted_text:
                        st.success(f"✅ {format_note} loaded successfully")
                        brand_rules_source = brand_rules_file.name
                    else:
                        st.warning(f"⚠️ {format_note}")
                        extracted_text = brand_rules_text if brand_rules_text else ""
                        brand_rules_source = "pasted text" if brand_rules_text else "none"
                else:
                    extracted_text = brand_rules_text
                    brand_rules_source = "pasted text"

                # Normalize rules
                if extracted_text:
                    normalized = normalize_brand_rules(extracted_text)
                    st.session_state.brand_rules = normalized
                    st.session_state.brand_rules_source = brand_rules_source
                    st.session_state.brand_rules_enabled = True

                    # Show summary
                    num_rules = len(normalized.get('rules', []))
                    if num_rules > 0:
                        st.info(
                            f"📌 Parsed {num_rules} brand rules from {brand_rules_source}")
                    if normalized.get('truncated'):
                        st.warning(
                            "⚠️ Brand rules were truncated (30k char limit)")
                else:
                    st.session_state.brand_rules = None
                    st.session_state.brand_rules_enabled = False
            else:
                st.session_state.brand_rules = None
                st.session_state.brand_rules_enabled = False

        st.markdown("<div style='height: 24px'></div>", unsafe_allow_html=True)

        # Advanced settings section
        st.markdown("#### Advanced Settings")
        with st.expander("⚙️ RAG Configuration", expanded=False):
            rag_k = st.slider(
                "Search depth (higher = more thorough but slower)",
                1,
                10,
                st.session_state.rag_top_k,
                help="Number of design patterns to reference",
                key="rag_slider_widget"
            )
            st.session_state.rag_top_k = rag_k
            helper_text(
                f"Will reference top {rag_k} design patterns from knowledge base")

        st.markdown("<div style='height: 24px'></div>", unsafe_allow_html=True)

        # API Key section
        st.markdown("#### API Configuration")
        with st.expander("🔑 Bring Your Own API Key (Optional)", expanded=False):
            api_key_input = st.text_input(
                "API Key",
                type="password",
                key="api_key_input_field",
                label_visibility="collapsed",
                help="Leave blank to use demo mode (limited analysis)"
            )
            if api_key_input:
                st.session_state.api_key = api_key_input
                status_banner(
                    "✅ API Key configured for this session", "success")

        st.markdown("<div style='height: 32px'></div>", unsafe_allow_html=True)

        # Confirmation button
        st.markdown("#### Confirm & Proceed")
        if st.button(
            "✓ Confirm Context",
            key="confirm_context_btn",
            use_container_width=True,
            type="primary"
        ):
            st.session_state.context_confirmed = True
            st.session_state.current_step = 3
            st.rerun()

    # ========================================================================
    # RIGHT COLUMN: UPLOADED IMAGES PREVIEW PANEL
    # ========================================================================
    with right:
        st.markdown("#### Uploaded Images Preview")

        preview_container = st.container(border=True)

        with preview_container:
            # Get uploads from session state
            uploads = st.session_state.get("uploads", [])

            if uploads:
                try:
                    if st.session_state.compare_mode and len(uploads) >= 2:
                        # Compare mode: show Design A and Design B stacked
                        st.markdown("**Design A**")
                        st.image(
                            uploads[0]["bytes"],
                            use_container_width=True,
                            caption=uploads[0].get("name", "Design A")
                        )

                        st.markdown("<div style='height: 16px'></div>",
                                    unsafe_allow_html=True)
                        st.markdown("**Design B**")
                        st.image(
                            uploads[1]["bytes"],
                            use_container_width=True,
                            caption=uploads[1].get("name", "Design B")
                        )

                        # Show remaining images as thumbnails if any
                        if len(uploads) > 2:
                            st.markdown(
                                "<div style='height: 16px'></div>", unsafe_allow_html=True)
                            st.markdown("##### Additional Images")
                            thumb_cols = st.columns(2)
                            for idx, upload in enumerate(uploads[2:], 0):
                                with thumb_cols[idx % 2]:
                                    st.image(
                                        upload["bytes"],
                                        use_container_width=True,
                                        caption=upload.get(
                                            "name", f"Design {idx+3}")
                                    )

                        st.markdown("<div style='height: 12px'></div>",
                                    unsafe_allow_html=True)
                        st.caption(f"📊 Total: {len(uploads)} images selected")
                    else:
                        # Single mode: show first image large
                        st.image(
                            uploads[0]["bytes"],
                            use_container_width=True,
                            caption=uploads[0].get("name", "Uploaded Design")
                        )

                        # Show file size and dimensions if available
                        size_kb = uploads[0].get("size", 0) / 1024
                        if size_kb > 0:
                            st.caption(f"📦 {size_kb:.1f} KB")

                except Exception as e:
                    st.error(f"Could not load preview: {str(e)}")
            else:
                # Empty state
                st.markdown(
                    """
                    <div style="text-align: center; padding: 40px 20px; color: #888;">
                        <p style="font-size: 48px; margin: 0;">📷</p>
                        <p style="font-size: 14px; margin-top: 16px;">
                            No images uploaded yet. Go back to Step 1 to upload or select a demo.
                        </p>
                    </div>
                    """,
                    unsafe_allow_html=True
                )

    # ========================================================================
    # SUMMARY & NAVIGATION (FULL WIDTH, BELOW COLUMNS)
    # ========================================================================

    # Show summary if confirmed
    if st.session_state.context_confirmed:
        st.markdown("<div style='height: 24px'></div>", unsafe_allow_html=True)
        st.markdown("### ✅ Context Summary")

        # Format platform selection
        platforms_str = ", ".join(
            st.session_state.selected_platforms) if st.session_state.selected_platforms else "None selected"
        creative_types_str = ", ".join(
            st.session_state.selected_creative_types) if st.session_state.selected_creative_types else "None selected"

        summary_items = {
            "🌐 Platforms": platforms_str,
            "📝 Creative Types": creative_types_str,
            "🔍 RAG Depth": str(st.session_state.rag_top_k),
            "🔐 API Key": "Configured" if st.session_state.api_key else "Demo Mode"
        }
        info_grid(summary_items)

        st.markdown("<div style='height: 32px'></div>", unsafe_allow_html=True)
        col1, col2 = st.columns([0.6, 0.4])
        with col1:
            if st.button(
                "→ Next: Select Agents",
                key="step2_next_confirmed",
                use_container_width=True,
                type="primary"
            ):
                st.session_state.current_step = 3
                st.rerun()

    # Back button (always visible)
    st.markdown("<div style='height: 24px'></div>", unsafe_allow_html=True)
    col1, col2 = st.columns([0.4, 0.6])
    with col1:
        if st.button(
            "← Back",
            key="step2_back_btn",
            use_container_width=True,
            type="secondary"
        ):
            st.session_state.context_confirmed = False
            st.session_state.current_step = 1
            st.rerun()

    is_complete = st.session_state.context_confirmed
    return is_complete, is_complete


def render_step_3_analysis() -> Tuple[bool, List[str]]:
    """
    Render Step 3: Select Agents & Trigger Analysis with two-column layout.

    Left column (1.6x): Agent selection controls
    Right column (1x): Uploaded images preview panel

    Returns:
        Tuple of (should_run_analysis, selected_agents)
    """
    # Initialize agent session state keys if not present
    agent_keys = ["agent_visual", "agent_ux",
                  "agent_market", "agent_conversion", "agent_brand"]
    for key in agent_keys:
        if key not in st.session_state:
            st.session_state[key] = False

    if "selected_preset" not in st.session_state:
        st.session_state.selected_preset = "custom"

    st.markdown("## 🤖 Select Analysis Engines")
    st.caption(
        "Choose which AI agents should analyze your design. Each brings unique expertise.")

    st.markdown("<div style='height: 16px'></div>", unsafe_allow_html=True)

    # Two-column layout: left for controls, right for preview
    left, right = st.columns([1.6, 1])

    # ========================================================================
    # LEFT COLUMN: AGENT SELECTION & CONTROLS
    # ========================================================================
    with left:
        # Get available agents
        try:
            agents = get_available_agents()
            if not agents:
                status_banner(
                    "⚠️ No agents available. Using fallback defaults.", "warning")
                agents = []
        except Exception as e:
            status_banner(
                f"⚠️ Agent registry error: {str(e)}. Using fallback.", "warning")
            agents = []

        # Quick Start Presets
        st.markdown("#### Quick Start Presets")
        preset_cols = st.columns(3)

        with preset_cols[0]:
            if st.button("⚡ Quick Review", use_container_width=True, key="preset_quick_btn"):
                # Set quick review: visual + ux only
                st.session_state.agent_visual = True
                st.session_state.agent_ux = True
                st.session_state.agent_market = False
                st.session_state.agent_conversion = False
                st.session_state.agent_brand = False
                st.session_state.selected_preset = "quick_review"
                st.rerun()

        with preset_cols[1]:
            if st.button("🔍 Full Analysis", use_container_width=True, key="preset_full_btn"):
                # Set full analysis: all agents
                st.session_state.agent_visual = True
                st.session_state.agent_ux = True
                st.session_state.agent_market = True
                st.session_state.agent_conversion = True
                st.session_state.agent_brand = True
                st.session_state.selected_preset = "full_analysis"
                st.rerun()

        with preset_cols[2]:
            if st.button("✏️ Custom", use_container_width=True, key="preset_custom_btn"):
                # Set custom: just clear the preset, don't force values
                st.session_state.selected_preset = "custom"
                st.rerun()

        st.markdown("<div style='height: 24px'></div>", unsafe_allow_html=True)

        # Individual Agent Selection - ALWAYS RENDER, don't depend on agents list
        st.markdown("#### Individual Agents")

        # Define agent configurations directly (not dependent on get_available_agents)
        agent_configs = [
            ("agent_visual", "👁️ Visual Analyst",
             "Analyzes visual design, composition, color theory, and aesthetic appeal"),
            ("agent_ux", "🎯 UX Critique",
             "Evaluates user experience, usability, and interaction patterns"),
            ("agent_market", "📊 Market Fit",
             "Assesses market positioning and competitive analysis"),
            ("agent_conversion", "💰 Conversion",
             "Examines conversion optimization and call-to-action effectiveness"),
            ("agent_brand", "🎨 Brand Alignment",
             "Ensures consistency with brand guidelines and identity"),
        ]

        for state_key, label, description in agent_configs:
            col1, col2 = st.columns([0.9, 0.1])

            with col1:
                # Use session state key directly for checkbox
                # Streamlit automatically manages state via the key parameter
                # Do NOT provide value= parameter as that causes a conflict
                st.checkbox(
                    label,
                    key=state_key,
                    label_visibility="visible"
                )
                helper_text(description)

            st.markdown("<div style='height: 12px'></div>",
                        unsafe_allow_html=True)

        st.markdown("<div style='height: 24px'></div>", unsafe_allow_html=True)

        # Build selected_agents list from checkbox states
        selected_agents = []
        if st.session_state.get("agent_visual", False):
            selected_agents.append("visual")
        if st.session_state.get("agent_ux", False):
            selected_agents.append("ux")
        if st.session_state.get("agent_market", False):
            selected_agents.append("market")
        if st.session_state.get("agent_conversion", False):
            selected_agents.append("conversion")
        if st.session_state.get("agent_brand", False):
            selected_agents.append("brand")

        st.session_state.selected_agents = selected_agents

        # Analysis Summary (Streamlit-native, no raw HTML)
        st.markdown("#### Analysis Summary")

        uploads = st.session_state.get("uploads", [])
        summary_cols = st.columns(4)

        with summary_cols[0]:
            st.metric("📸 Designs", len(uploads))
        with summary_cols[1]:
            st.metric("🎯 Platform", st.session_state.platform[:10])
        with summary_cols[2]:
            st.metric("📝 Type", st.session_state.creative_type[:10])
        with summary_cols[3]:
            st.metric("🤖 Engines", len(selected_agents))

        st.markdown("<div style='height: 32px'></div>", unsafe_allow_html=True)

        # Primary CTA
        is_complete = len(selected_agents) > 0 and len(uploads) > 0

        if not is_complete:
            if not uploads:
                st.info(
                    "📸 No designs uploaded. Go back to Step 1 to upload or select a demo.", icon="ℹ️")
            if not selected_agents:
                st.info("🤖 Select at least one agent to run analysis.", icon="ℹ️")

        st.markdown("<div style='height: 16px'></div>", unsafe_allow_html=True)

        analyze_clicked = st.button(
            "🚀 Analyze Now",
            key="step3_analyze_btn",
            use_container_width=True,
            type="primary" if is_complete else "secondary",
            disabled=not is_complete
        )

    # ========================================================================
    # RIGHT COLUMN: UPLOADED IMAGES PREVIEW PANEL
    # ========================================================================
    with right:
        render_upload_preview_panel()

    # ========================================================================
    # NAVIGATION (FULL WIDTH, BELOW COLUMNS)
    # ========================================================================
    st.markdown("<div style='height: 24px'></div>", unsafe_allow_html=True)
    col1, col2, col3 = st.columns([0.4, 0.3, 0.3])

    with col1:
        if st.button("← Back", key="step3_back_btn", use_container_width=True, type="secondary"):
            st.session_state.context_confirmed = False
            st.session_state.current_step = 2
            st.rerun()

    return analyze_clicked, st.session_state.selected_agents


def render_compare_results(results: dict) -> Tuple[bool, bool]:
    """
    Render comparison results for 2+ designs.

    Args:
        results: dict with mode="compare" and comparison data

    Returns:
        Tuple[bool, bool]: (step_complete, should_advance)
    """
    import pandas as pd

    # Extract comparison data
    winner = results.get("winner", "Design A")
    confidence = results.get("confidence", "medium")
    rankings = results.get("rankings", [])
    designs = results.get("designs", {})
    key_differences = results.get("key_differences", [])
    why_winner_won = results.get("why_winner_won", [])
    trade_offs = results.get("trade_offs", [])

    # ====================================================================
    # WINNER BANNER
    # ====================================================================
    st.markdown("")
    winner_color = "🟢" if confidence == "high" else "🟡"
    st.markdown(f"""
    ### {winner_color} **🏆 Winner: {winner}** ({confidence.upper()} confidence)
    
    This design was ranked highest by our analysis engines across the key metrics.
    """)

    # ====================================================================
    # RANKINGS TABLE
    # ====================================================================
    st.markdown("### 📊 Design Rankings")

    ranking_data = []
    for idx, design_name in enumerate(rankings, 1):
        design_data = designs.get(design_name, {})
        overall_score = design_data.get("overall_score", 0)
        ranking_data.append({
            "Rank": f"#{idx} 🥇" if idx == 1 else f"#{idx}",
            "Design": design_name,
            "Overall Score": f"{overall_score:.1f}/100",
            "Visual": f"{design_data.get('agent_scores', {}).get('visual', 0):.1f}",
            "UX": f"{design_data.get('agent_scores', {}).get('ux', 0):.1f}",
            "Market": f"{design_data.get('agent_scores', {}).get('market', 0):.1f}",
            "Conversion": f"{design_data.get('agent_scores', {}).get('conversion', 0):.1f}",
            "Brand": f"{design_data.get('agent_scores', {}).get('brand', 0):.1f}",
        })

    df_rankings = pd.DataFrame(ranking_data)
    st.dataframe(df_rankings, use_container_width=True, hide_index=True)

    st.markdown("")

    # ====================================================================
    # WHY WINNER WON
    # ====================================================================
    st.markdown("### ✨ Why This Design Won")
    st.markdown("Key strengths that made the difference:")

    for point in why_winner_won:
        st.markdown(f"• {point}")

    st.markdown("")

    # ====================================================================
    # KEY DIFFERENCES
    # ====================================================================
    if key_differences:
        st.markdown("### 🔍 Key Differences")
        st.markdown("How the designs compare across critical aspects:")

        for diff in key_differences:
            aspect = diff.get("aspect", "Unknown")
            winner_adv = diff.get("winner_advantage", "")
            details = diff.get("details", "")

            st.markdown(f"""
**{aspect}**
- {winner_adv}
- {details}
""")

    st.markdown("")

    # ====================================================================
    # TRADE-OFFS
    # ====================================================================
    if trade_offs:
        st.markdown("### ⚖️ Trade-Offs & Considerations")

        for tradeoff in trade_offs:
            st.markdown(f"• {tradeoff}")

    st.markdown("")

    # ====================================================================
    # BRAND COMPLIANCE COMPARISON (if applicable)
    # ====================================================================
    brand_compliance = results.get("brand_compliance", {})
    if brand_compliance and brand_compliance.get("enabled"):
        st.divider()
        st.markdown("### 🏛️ Brand Compliance Comparison")

        comparison = brand_compliance.get("compliance_comparison", {})
        if comparison and comparison.get("winner") != "Tied":
            st.markdown(
                f"**Brand Winner:** {comparison['winner']} (Δ {comparison['delta']} points)")

            col1, col2 = st.columns(2)

            with col1:
                st.markdown("**Design A**")
                st.metric("Compliance Score",
                          f"{comparison['design_a']['score']}/100")
                st.metric("Level", comparison['design_a']['level'])
                st.metric("Violations",
                          comparison['design_a']['violations_count'])

            with col2:
                st.markdown("**Design B**")
                st.metric("Compliance Score",
                          f"{comparison['design_b']['score']}/100")
                st.metric("Level", comparison['design_b']['level'])
                st.metric("Violations",
                          comparison['design_b']['violations_count'])

        st.markdown("")

    if designs:
        design_tabs = st.tabs(list(designs.keys()))

        for tab, (design_name, design_data) in zip(design_tabs, designs.items()):
            with tab:
                st.markdown(f"#### {design_name} Analysis")

                overall_score = design_data.get("overall_score", 0)
                agent_scores = design_data.get("agent_scores", {})
                findings = design_data.get("findings_summary", {})
                recommendations = design_data.get("top_recommendations", [])

                # Metrics
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("Overall Score", f"{overall_score:.1f}/100")
                with col2:
                    st.metric("Total Findings", findings.get("total", 0))
                with col3:
                    st.metric("Critical Issues", findings.get("critical", 0))
                with col4:
                    st.metric("Warnings", findings.get("warnings", 0))

                st.markdown("")

                # Brand Compliance (if available)
                brand_compliance = results.get("brand_compliance", {})
                if brand_compliance and brand_compliance.get("enabled"):
                    design_compliance = brand_compliance.get(
                        "by_design", {}).get(design_name)
                    if design_compliance:
                        st.markdown("**Brand Compliance**")

                        col1, col2, col3 = st.columns(3)
                        with col1:
                            st.metric(
                                "Score", f"{design_compliance['score']}/100")
                        with col2:
                            st.metric("Level", design_compliance['level'])
                        with col3:
                            st.metric("Violations", len(
                                design_compliance.get('violations', [])))

                        if design_compliance.get('violations'):
                            with st.expander("View Violations"):
                                for violation in design_compliance.get('violations', [])[:5]:
                                    severity_icon = "🔴" if violation.get(
                                        'severity') == 'high' else "🟡" if violation.get('severity') == 'medium' else "🟢"
                                    st.markdown(
                                        f"{severity_icon} **[{violation.get('rule_id')}]** {violation.get('issue', 'N/A')}")
                                    st.caption(
                                        f"Fix: {violation.get('fix', 'N/A')}")

                        st.divider()

                # Agent scores
                if agent_scores:
                    st.markdown("**Agent Scores**")
                    agent_names = {
                        "visual": "Visual Design",
                        "ux": "UX Critique",
                        "market": "Market Research",
                        "conversion": "Conversion/CTA",
                        "brand": "Brand Consistency",
                    }

                    score_cols = st.columns(5)
                    for col, (agent_id, score) in zip(score_cols, agent_scores.items()):
                        with col:
                            agent_label = agent_names.get(
                                agent_id, agent_id.title())
                            st.metric(agent_label, f"{score:.1f}")

                st.divider()

                # Recommendations
                if recommendations:
                    st.markdown("**Top Recommendations**")
                    for rec in recommendations[:3]:  # Show top 3
                        with st.expander(f"📌 {rec.get('title', 'Recommendation')}"):
                            st.write(
                                f"**Issue:** {rec.get('description', 'N/A')}")
                            st.write(f"**Impact:** {rec.get('impact', 'N/A')}")
                            st.write(
                                f"**Action:** {rec.get('recommendation', 'N/A')}")

    st.markdown("")
    st.divider()
    st.markdown("")

    # ====================================================================
    # EXPORT & SHARE
    # ====================================================================
    st.markdown("### 📤 Export & Share Results")

    col1, col2, col3 = st.columns(3)

    with col1:
        report_json = json.dumps(results, indent=2, default=str)
        st.download_button(
            label="📋 Download as JSON",
            data=report_json,
            file_name=f"comparison_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
            mime="application/json",
            key="export_compare_json"
        )

    with col2:
        # Simple text summary
        summary_lines = [
            f"Design Comparison Report",
            f"========================",
            f"",
            f"Winner: {winner} ({confidence} confidence)",
            f"",
            f"Rankings:",
        ]
        for idx, design_name in enumerate(rankings, 1):
            score = designs.get(design_name, {}).get("overall_score", 0)
            summary_lines.append(f"{idx}. {design_name} - {score:.1f}/100")

        summary_lines.extend([
            f"",
            f"Why {winner} Won:",
        ])
        summary_lines.extend(why_winner_won)

        summary_text = "\n".join(summary_lines)
        st.download_button(
            label="📝 Download as TXT",
            data=summary_text.encode('utf-8'),
            file_name=f"comparison_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
            mime="text/plain",
            key="export_compare_txt"
        )

    with col3:
        if st.button("🔄 New Comparison", use_container_width=True):
            st.session_state.current_step = 1
            st.session_state.context_confirmed = False
            st.session_state.analysis_complete = False
            st.session_state.selected_agents = [
                "visual_analysis", "ux_critique"]
            st.session_state.uploads = []
            st.session_state.analysis_results = None
            st.rerun()

    st.markdown("")
    st.divider()
    st.markdown("")

    # Navigation
    st.markdown("### 🎯 Next Steps")
    col1, col2 = st.columns([0.5, 0.5])

    with col1:
        if st.button("← Back to Analysis", key="compare_back_btn", use_container_width=True, type="secondary"):
            st.session_state.current_step = 3
            st.rerun()

    with col2:
        if st.button("🏠 Start Over", key="compare_start_btn", use_container_width=True, type="primary"):
            st.session_state.current_step = 1
            st.session_state.context_confirmed = False
            st.session_state.analysis_complete = False
            st.session_state.selected_agents = [
                "visual_analysis", "ux_critique"]
            st.session_state.uploads = []
            st.session_state.analysis_results = None
            st.rerun()

    return True, False


def render_step_4_results() -> Tuple[bool, bool]:
    """
    Render Step 4: Results display from completed analysis.

    Returns:
        Tuple[bool, bool]: (is_complete, should_advance)
        - is_complete: Whether results are ready to display
        - should_advance: Whether to advance to next step (not used for Step 4)
    """
    # ====================================================================
    # HELPER FUNCTIONS - Local implementations for reliability
    # ====================================================================
    def build_summary_text(results: dict) -> str:
        """Build a text summary from results. Always returns non-empty string."""
        try:
            lines = []
            lines.append("=" * 60)
            lines.append("DESIGN ANALYSIS SUMMARY")
            lines.append("=" * 60)
            lines.append("")

            # Overall Score
            overall = results.get("overall_score", 0)
            lines.append(f"OVERALL SCORE: {overall:.1f}/100")
            lines.append("")

            # Agent Scores
            agents = results.get("agent_scores", {}) or {}
            if agents:
                lines.append("AGENT SCORES:")
                lines.append("-" * 40)
                agent_names = {
                    "visual": "Visual Design",
                    "ux": "User Experience",
                    "market": "Market Fit",
                    "conversion": "Conversion/CTA",
                    "brand": "Brand Consistency"
                }
                for agent_id, score in agents.items():
                    agent_name = agent_names.get(
                        agent_id, agent_id.replace("_", " ").title())
                    lines.append(f"  {agent_name:.<35} {score:>6.1f}/100")
                lines.append("")

            # Findings
            findings = results.get("findings_summary", {}) or {}
            if findings:
                lines.append("FINDINGS SUMMARY:")
                lines.append("-" * 40)
                lines.append(
                    f"  Total Issues:........ {findings.get('total', 0)}")
                lines.append(
                    f"  Critical Issues:..... {findings.get('critical', 0)}")
                lines.append(
                    f"  Warnings:............ {findings.get('warnings', 0)}")
                lines.append(
                    f"  Info Items:.......... {findings.get('info', 0)}")
                lines.append("")

            # Top Recommendations
            recs = results.get("top_recommendations", []) or []
            if recs:
                lines.append("TOP RECOMMENDATIONS:")
                lines.append("-" * 40)
                for i, rec in enumerate(recs[:5], 1):
                    title = rec.get("title", "Recommendation")
                    priority = rec.get("priority", "medium").upper()
                    desc = rec.get("description", "")
                    lines.append(f"  {i}. [{priority}] {title}")
                    if desc:
                        lines.append(f"     {desc[:80]}")
                lines.append("")

            lines.append("=" * 60)
            lines.append(
                f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
            lines.append("=" * 60)

            return "\n".join(lines)
        except Exception as e:
            return f"DESIGN ANALYSIS SUMMARY\n(Error building summary: {str(e)})\n\nPlease contact support."

    def build_pdf_bytes(summary_text: str) -> bytes:
        """Build PDF bytes from summary text. Falls back to text if reportlab unavailable."""
        try:
            from reportlab.lib.pagesizes import letter
            from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
            from reportlab.lib.units import inch
            from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
            from reportlab.lib.enums import TA_LEFT
            from reportlab.lib import colors
            import io

            pdf_buffer = io.BytesIO()
            doc = SimpleDocTemplate(pdf_buffer, pagesize=letter,
                                    topMargin=0.5*inch, bottomMargin=0.5*inch)
            styles = getSampleStyleSheet()

            # Custom style
            style = ParagraphStyle(
                'CustomBody',
                parent=styles['Normal'],
                fontSize=10,
                leading=12,
                textColor=colors.black,
                alignment=TA_LEFT,
                fontName='Courier'
            )

            story = []
            # Split summary into paragraphs and add to PDF
            for para_text in summary_text.split('\n'):
                if para_text.strip():
                    story.append(Paragraph(para_text, style))
                else:
                    story.append(Spacer(1, 0.1*inch))

            doc.build(story)
            return pdf_buffer.getvalue()
        except ImportError:
            # Reportlab not available - return summary as bytes
            return summary_text.encode('utf-8')

    # Check if analysis is complete
    if not st.session_state.get("analysis_complete", False):
        status_banner(
            "⚠️ No analysis to display. Please complete Step 3 to generate results.", "warning")
        st.stop()

    # Get analysis results
    results = st.session_state.get("analysis_results", {})
    if not results:
        status_banner("❌ No analysis results found.", "error")
        st.stop()

    # Display step title
    st.header("📊 Step 4: Results")
    st.markdown("Review your analysis results and recommendations below.")
    st.markdown("")

    # Status banner
    uploads = st.session_state.get("uploads", [])
    selected_agents = st.session_state.get("selected_agents", [])
    status_banner(
        f"✅ Analysis complete! {len(selected_agents)} agents analyzed {len(uploads)} design(s).",
        "success"
    )

    st.markdown("")

    # ========================================================================
    # COMPARE MODE RESULTS (if applicable)
    # ========================================================================
    if results.get("mode") == "compare":
        return render_compare_results(results)

    # ========================================================================
    # EXPORT & SHARE ACTIONS (Top Bar)
    # ========================================================================
    st.subheader("📤 Export & Share Results")
    st.caption("Download analysis or share with your team")

    col1, col2, col3, col4, col5 = st.columns(5, gap="small")

    # Build summary text using local function (always succeeds)
    summary_text = build_summary_text(results)

    # 1. Download as JSON
    with col1:
        report_json = json.dumps(results, indent=2, default=str)
        st.download_button(
            label="📋 JSON",
            data=report_json,
            file_name=f"analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
            mime="application/json",
            key="export_json"
        )

    # 2. Download Summary (Text) - ALWAYS ENABLED
    with col2:
        st.download_button(
            label="📝 Summary (TXT)",
            data=summary_text.encode('utf-8'),
            file_name=f"summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
            mime="text/plain",
            key="export_summary_txt"
        )

    # 3. Download Summary (PDF) - ALWAYS ENABLED
    with col3:
        pdf_data = build_pdf_bytes(summary_text)
        # Check if we got real PDF or fallback
        if pdf_data.startswith(b'%PDF'):
            # Real PDF
            file_ext = '.pdf'
            mime_type = 'application/pdf'
        else:
            # Text fallback
            file_ext = '.txt'
            mime_type = 'text/plain'
            if not st.session_state.get("_pdf_warning_shown", False):
                st.warning(
                    "📦 PDF library not available; downloading as text file instead.", icon="⚠️")
                st.session_state._pdf_warning_shown = True

        st.download_button(
            label="📄 Summary (PDF)",
            data=pdf_data,
            file_name=f"summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}{file_ext}",
            mime=mime_type,
            key="export_summary_pdf"
        )

    # 4. Email Draft - ALWAYS ENABLED
    with col4:
        if st.button("✉️ Email", key="export_email"):
            st.session_state.show_email_draft = True

    # 5. Share Link
    with col5:
        if st.button("🔗 Share", key="export_share"):
            st.session_state.show_share_link = True

    # Email Draft Expander
    if st.session_state.get("show_email_draft", False):
        with st.expander("📧 Email Draft", expanded=True):
            platform = st.session_state.get("platform", "Unknown")
            email_subject = f"Design Analysis Summary - {platform} - {datetime.now().strftime('%B %d, %Y')}"
            email_body = f"""Subject: {email_subject}
To: [recipient email]

Hi Team,

Please find the design analysis results below:

{summary_text}

---
Generated via AIpowered365 Labs Design Analysis
https://aipowered365.local
"""
            st.code(email_body, language="text")
            st.caption("Copy the text above and paste into your email client")
            st.info(
                "💡 Tip: Click the copy icon (⎘) in the code block to copy to clipboard")

    # Share Link Expander
    if st.session_state.get("show_share_link", False):
        with st.expander("🔗 Share Link", expanded=True):
            # Generate a pseudo-shareable link using session ID and timestamp
            session_id = st.session_state.get("session_id", "analysis")
            timestamp_hash = hashlib.md5(
                str(datetime.now()).encode()
            ).hexdigest()[:8]
            share_url = f"https://aipowered365.local/report/{session_id}_{timestamp_hash}"

            st.text_input(
                "Share this link with your team:",
                value=share_url,
                disabled=True,
                key="share_link_input"
            )
            st.caption("Shareable link generated for this analysis session")
            st.info("💡 Copy the link above to share results with teammates")

    st.divider()
    st.markdown("")

    # Results tabs (matching the original Results page)
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "📊 Overview",
        "🎯 Recommendations",
        "📈 Impact Analysis",
        "🎨 Visual Feedback",
        "📄 Detailed Data"
    ])

    # ========================================================================
    # TAB 1: OVERVIEW
    # ========================================================================
    with tab1:
        st.subheader("Analysis Overview")

        overall_score = results.get("overall_score", 0)
        agent_scores = results.get("agent_scores", {})
        findings = results.get("findings_summary", {})

        # ====================================================================
        # PERFORMANCE SCORES (GAUGES)
        # ====================================================================
        st.markdown("### 📈 Performance Scores")

        # Overall gauge
        try:
            fig_overall = generate_score_gauge_chart(
                float(overall_score or 0), "Overall Score")
            st.plotly_chart(
                fig_overall, use_container_width=True, key="gauge_overall")
        except Exception as e:
            st.metric("Overall Score", f"{float(overall_score or 0):.1f}/100")

        st.markdown("")

        # Agent gauges
        st.markdown("#### Agent Scores")
        name_map = {
            "visual": "Visual Design",
            "ux": "User Experience",
            "market": "Market Fit",
            "conversion": "Conversion/CTA",
            "brand": "Brand Consistency",
        }

        ordered_keys = ["visual", "ux", "market", "conversion", "brand"]
        cols = st.columns(3, gap="large")
        for i, k in enumerate(ordered_keys):
            score = float(agent_scores.get(k, 0) or 0)
            with cols[i % 3]:
                try:
                    fig = generate_score_gauge_chart(
                        score, name_map.get(k, k.title()))
                    st.plotly_chart(
                        fig, use_container_width=True, key=f"gauge_{k}")
                except Exception as e:
                    st.metric(name_map.get(k, k.title()), f"{score:.1f}/100")

        st.divider()
        st.markdown("")

        # Overall score metrics
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Overall Score", f"{overall_score:.1f}/100")
        with col2:
            st.metric("Total Findings", findings.get("total", 0))
        with col3:
            st.metric("Critical Issues", findings.get("critical", 0))

        st.divider()

        # Brand Compliance Section
        brand_compliance = results.get("brand_compliance", {})
        if brand_compliance and brand_compliance.get("enabled"):
            st.markdown("### 🏛️ Brand Compliance")

            compliance_score = brand_compliance.get("score", 0)
            compliance_level = brand_compliance.get("level", "Unknown")

            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Compliance Score", f"{compliance_score}/100")
            with col2:
                st.metric("Level", compliance_level)
            with col3:
                violations_count = len(brand_compliance.get("violations", []))
                st.metric("Violations Found", violations_count)

            st.markdown("")

            # Show top violations
            violations = brand_compliance.get("violations", [])
            if violations:
                st.markdown("#### Top Issues")
                for v in violations[:3]:
                    severity_icon = "🔴" if v.get('severity') == 'high' else "🟡" if v.get(
                        'severity') == 'medium' else "🟢"
                    st.warning(
                        f"{severity_icon} **{v.get('rule_id')}** ({v.get('category')}): {v.get('issue')}\n\n"
                        f"Fix: {v.get('fix')}"
                    )

            # Show expanded view option
            with st.expander("📋 View All Brand Rules & Compliance Details"):
                if st.session_state.get("brand_rules"):
                    st.markdown(
                        f"**Source:** {st.session_state.get('brand_rules_source', 'unknown')}")
                    raw_text = st.session_state.brand_rules.get('raw_text', '')
                    if raw_text:
                        st.text(raw_text[:1000] +
                                ("..." if len(raw_text) > 1000 else ""))

                    # Show all violations
                    all_violations = brand_compliance.get("violations", [])
                    if all_violations:
                        st.markdown("**All Violations:**")
                        for v in all_violations:
                            st.markdown(
                                f"- **{v.get('rule_id')}** ({v.get('severity').upper()}): {v.get('issue')}"
                            )

                    # Show passed checks
                    passed = brand_compliance.get("passed_checks", [])
                    if passed:
                        st.markdown("**Passed Checks:**")
                        for p in passed[:5]:
                            st.markdown(
                                f"- ✅ **{p.get('rule_id')}**: {p.get('note')}")

            st.divider()

        # Agent scores
        if agent_scores:
            st.write("#### Agent Analysis Scores")
            agent_names = {
                "visual": "Visual Design",
                "ux": "UX Critique",
                "market": "Market Research",
                "conversion": "Conversion/CTA",
                "brand": "Brand Consistency",
            }

            score_cols = st.columns(len(agent_scores))
            for col, (agent_id, score) in zip(score_cols, agent_scores.items()):
                with col:
                    agent_label = agent_names.get(agent_id, agent_id.title())
                    st.metric(agent_label, f"{score:.1f}/100")

    # ========================================================================
    # TAB 2: RECOMMENDATIONS
    # ========================================================================
    with tab2:
        st.subheader("🎯 Recommendations")

        recommendations = results.get("top_recommendations", [])

        if recommendations:
            st.write(f"**Showing {len(recommendations)} recommendations**")

            for idx, rec in enumerate(recommendations, 1):
                with st.expander(
                    f"{idx}. {rec.get('title', 'Recommendation')} - {rec.get('priority', 'medium').upper()}",
                    expanded=(idx <= 2)
                ):
                    st.write(
                        f"**What to change:** {rec.get('description', 'N/A')}")
                    st.write(f"**Why it matters:** {rec.get('impact', 'N/A')}")
                    st.write(
                        f"**Recommended action:** {rec.get('recommendation', 'N/A')}")
        else:
            st.info("No recommendations available")

    # ========================================================================
    # TAB 3: IMPACT ANALYSIS
    # ========================================================================
    with tab3:
        st.subheader("📈 Impact Projections")

        impact = results.get("impact_analysis", {})

        if impact:
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
        else:
            st.info("No impact analysis available")

    # ========================================================================
    # TAB 4: VISUAL FEEDBACK
    # ========================================================================
    with tab4:
        st.subheader("🎨 Visual Feedback")

        feedback = results.get("visual_feedback", {})
        if feedback:
            st.json(feedback)
        else:
            st.info("No visual feedback annotations available")

    # ========================================================================
    # TAB 5: DETAILED DATA
    # ========================================================================
    with tab5:
        st.subheader("📄 Detailed Data")

        # Download as JSON
        report_json = json.dumps(results, indent=2, default=str)
        st.download_button(
            label="📥 Download as JSON",
            data=report_json,
            file_name=f"analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
            mime="application/json"
        )

        st.divider()
        st.json(results)

    st.markdown("")
    st.divider()

    # Navigation buttons
    st.markdown("### 🎯 Next Steps")

    col1, col2, col3 = st.columns([0.3, 0.3, 0.4])

    with col1:
        if st.button("← Back to Analysis", key="step4_back_btn", use_container_width=True, type="secondary"):
            st.session_state.current_step = 3
            st.rerun()

    with col3:
        if st.button("🔄 Start New Analysis", key="step4_new_analysis_btn", use_container_width=True, type="primary"):
            # Reset wizard state
            st.session_state.current_step = 1
            st.session_state.context_confirmed = False
            st.session_state.analysis_complete = False
            st.session_state.selected_agents = [
                "visual_analysis", "ux_critique"]
            st.session_state.uploads = []
            st.session_state.analysis_results = None
            st.rerun()

    return True, False  # Step 4 is complete, no further advancement
