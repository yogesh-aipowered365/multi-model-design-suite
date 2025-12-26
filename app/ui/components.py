"""AIpowered365 Labs UI Components"""
import streamlit as st
from pathlib import Path
from typing import List, Optional, Callable


def hero_panel(
    title: str = "Design Analysis",
    subtitle: str = "Analyze creative designs using multiple AI experts to improve visual quality, user experience, conversions, and brand consistency — before you publish.",
    pills: Optional[List[str]] = None,
    logo_path: str = "assets/logo.png",
    show_logo: bool = True
):
    """
    Render the premium hero panel with gradient header and enhanced product messaging.

    Args:
        title: Main title text
        subtitle: Subtitle text (value statement)
        pills: List of badge texts to display
        logo_path: Path to logo file
        show_logo: Whether to show the logo
    """
    if pills is None:
        pills = ["AI-Powered", "Multi-Agent Analysis"]

    # Create hero HTML
    logo_html = ""
    if show_logo:
        logo_path_obj = Path(logo_path)
        if logo_path_obj.exists():
            logo_html = f'<div class="aip365-hero-right"><img src="app/assets/logo.png" alt="AIpowered365 Labs"></div>'

    pills_html = "".join([
        f'<div class="aip365-pill">✨ {pill}</div>'
        for pill in pills
    ])

    # Enhanced hero with additional messaging
    hero_html = f"""<div class="aip365-hero"><div class="aip365-hero-content"><div class="aip365-hero-left"><h1>{title}</h1><p class="hero-subtitle">{subtitle}</p><div class="aip365-pills">{pills_html}</div><div class="hero-details"><p class="hero-paragraph">This tool reviews marketing and product designs (ads, landing pages, social posts, emails, UI screens) using specialized AI agents. Each agent evaluates your design from a different expert perspective and delivers actionable recommendations.</p><div class="hero-bullets"><div class="hero-bullet">• Visual Design & Layout quality</div><div class="hero-bullet">• UX & Accessibility compliance</div><div class="hero-bullet">• Conversion & CTA effectiveness</div><div class="hero-bullet">• Brand consistency & market fit</div></div><p class="hero-process"><strong>How it works:</strong> Upload designs → set context → run AI analysis → review scores, insights, and comparisons.</p><p class="hero-audience">Designed for designers, marketers, product teams, and founders.</p></div></div>{logo_html}</div></div>"""

    st.markdown(hero_html, unsafe_allow_html=True)


def stepper(current_step: int, total_steps: int = 4, locked: bool = True):
    """
    Render a locked stepper/progress indicator with visual status indicators.
    Active step renders as a tab that overlaps the body container.

    Args:
        current_step: Current step number (1-indexed)
        total_steps: Total number of steps
        locked: Whether stepping is locked (cannot jump ahead)
    """
    steps = [
        {"num": 1, "label": "Upload Designs"},
        {"num": 2, "label": "Set Context"},
        {"num": 3, "label": "Run Analysis"},
        {"num": 4, "label": "Results"},
    ]

    stepper_html = '<div class="aip365-stepper-row">'

    for i, step in enumerate(steps[:total_steps]):
        step_num = step["num"]
        step_label = step["label"]

        # Determine step status class
        active_class = ""
        if step_num < current_step:
            step_class = "aip365-step-pill done"
            icon = "✓"
        elif step_num == current_step:
            step_class = "aip365-step-pill active"
            active_class = "active"
            icon = str(step_num)
        else:
            step_class = "aip365-step-pill todo"
            icon = str(step_num)

        stepper_html += f"""
        <div class="{step_class}">
            <div class="step-num">{icon}</div>
            <div class="step-label">{step_label}</div>
        </div>
        """

        # Add connector between steps (but not after last step)
        if i < total_steps - 1:
            # Connector styling based on step position
            connector_class = "aip365-connector"
            if step_num < current_step:
                connector_class += " done"
            elif step_num == current_step:
                connector_class += " active"

            stepper_html += f'<div class="{connector_class}"></div>'

    stepper_html += '</div>'
    st.markdown(stepper_html, unsafe_allow_html=True)


def card(title: str, icon: str = "📋", body_fn: Optional[Callable] = None):
    """
    Render a premium glass card with optional body content.

    Args:
        title: Card title text
        icon: Emoji icon for the title
        body_fn: Optional callback function to render card body

    Returns:
        A Streamlit container for the card body
    """
    st.markdown(f"""
    <div class="aip365-card">
        <div class="aip365-card-header">
            <span class="icon">{icon}</span>
            <h2>{title}</h2>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Return a container for body content
    if body_fn:
        body_fn()


def status_banner(text: str, state: str = "info"):
    """
    Render a status banner message.

    Args:
        text: Banner message text
        state: Banner type - 'success', 'info', 'warning', 'error'
    """
    banner_html = f"""
    <div class="aip365-banner {state}">
        {text}
    </div>
    """
    st.markdown(banner_html, unsafe_allow_html=True)


def primary_button(
    label: str,
    key: str,
    disabled: bool = False,
    help_text: Optional[str] = None,
    icon: str = "→"
) -> bool:
    """
    Render a primary action button.

    Args:
        label: Button text
        key: Unique key for the button
        disabled: Whether button is disabled
        help_text: Optional help text below button
        icon: Icon to show before text

    Returns:
        True if button clicked, False otherwise
    """
    col1, col2 = st.columns([0.6, 0.4])

    with col1:
        button_text = f"{icon} {label}" if icon else label
        clicked = st.button(
            button_text,
            key=key,
            disabled=disabled,
            use_container_width=True,
            type="primary" if not disabled else "secondary"
        )

    if help_text:
        st.markdown(
            f'<p class="aip365-helper">{help_text}</p>', unsafe_allow_html=True)

    return clicked


def helper_text(text: str):
    """
    Render small helper/instructional text.

    Args:
        text: Helper text content
    """
    st.markdown(f'<p class="aip365-helper">{text}</p>', unsafe_allow_html=True)


def chip_selector(
    label: str,
    options: List[str],
    default: Optional[str] = None,
    key: str = "chips"
) -> str:
    """
    Render a chip/badge selector (radio button alternative).

    Args:
        label: Label for the selector
        options: List of chip options
        default: Default selected option
        key: Unique key for the selector

    Returns:
        Selected option value
    """
    st.markdown(
        f'<p style="font-weight: 600; margin-bottom: 12px;">{label}</p>', unsafe_allow_html=True)

    selected = st.radio(
        label,
        options=options,
        index=options.index(default) if default and default in options else 0,
        key=key,
        horizontal=True,
        label_visibility="collapsed"
    )

    return selected


def info_grid(items: dict):
    """
    Render an info grid (label-value pairs).

    Args:
        items: Dictionary of {label: value} pairs
    """
    grid_html = '<div class="aip365-image-info-grid">'

    for label, value in items.items():
        grid_html += f"""
        <div class="aip365-info-item">
            <div class="aip365-info-label">{label}</div>
            <div class="aip365-info-value">{value}</div>
        </div>
        """

    grid_html += '</div>'
    st.markdown(grid_html, unsafe_allow_html=True)


def section_divider():
    """Render a visual section divider."""
    st.markdown('<hr style="border: none; height: 2px; background: linear-gradient(90deg, #E5E7EB 0%, #7C3AED 50%, #E5E7EB 100%); margin: 32px 0;">', unsafe_allow_html=True)


def branded_card_open(title: str = "", icon: str = ""):
    """
    Open a branded AIpowered365 card container.

    Args:
        title: Optional card title (shown in header)
        icon: Optional emoji icon for title
    """
    header_html = ""
    if title:
        header_html = f"""
        <div class="aip365-card-header">
            <span class="icon">{icon}</span>
            <h2>{title}</h2>
        </div>
        """

    card_html = f"""<div class="aip365-card">{header_html}"""
    st.markdown(card_html, unsafe_allow_html=True)


def spacer(height_px: int = 16):
    """Render a vertical spacer."""
    st.markdown(
        f"<div style='height: {height_px}px'></div>", unsafe_allow_html=True)


def render_context_summary_card():
    """Render a summary card showing current wizard context (platform, type, mode)."""
    uploads = st.session_state.get("uploads", [])
    compare_mode = st.session_state.get("compare_mode", False)
    platform = st.session_state.get("platform", "Not selected")
    creative_type = st.session_state.get("creative_type", "Not selected")

    col1, col2 = st.columns(2)
    with col1:
        st.metric("📸 Images", len(uploads))
        st.metric("🎯 Platform", platform[:12])
    with col2:
        st.metric("🔄 Mode", "Compare" if compare_mode else "Single")
        st.metric("📝 Type", creative_type[:12])


def render_upload_preview_panel(container_element=None):
    """
    Render the uploaded images preview panel (reusable for Steps 2 & 3).

    Args:
        container_element: st.container() object to render in. If None, renders in current context.

    Renders a preview panel showing:
    - Uploaded images in single or compare mode
    - File size and dimensions
    - Empty state if no uploads
    """
    # Note: We don't use context manager here - just render directly
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
                        st.markdown("<div style='height: 16px'></div>",
                                    unsafe_allow_html=True)
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
