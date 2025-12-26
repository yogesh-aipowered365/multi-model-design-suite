"""Wizard State Management for AIpowered365 Labs"""
import streamlit as st
from typing import List, Optional


def init_wizard_state():
    """Initialize session state for the 3-step wizard."""

    # Current step (1-3, or "results")
    if "current_step" not in st.session_state:
        st.session_state.current_step = 1

    # Uploaded files - structured as list of dicts {name, bytes, size, mime, hash}
    if "uploads" not in st.session_state:
        st.session_state.uploads = []

    # Legacy field for backward compatibility
    if "uploaded_files" not in st.session_state:
        st.session_state.uploaded_files = []

    # Design comparison mode flag
    if "compare_mode" not in st.session_state:
        st.session_state.compare_mode = False

    # Context confirmation status (Step 2 must be confirmed before Step 3)
    if "context_confirmed" not in st.session_state:
        st.session_state.context_confirmed = False

    # Platform selection (Step 2) - Single selection (backward compat)
    if "platform" not in st.session_state:
        st.session_state.platform = "Instagram"

    # Multi-select platforms (Step 2)
    if "selected_platforms" not in st.session_state:
        st.session_state.selected_platforms = []

    # Creative type (Step 2) - Single selection (backward compat)
    if "creative_type" not in st.session_state:
        st.session_state.creative_type = "Marketing Creative"

    # Multi-select creative types (Step 2)
    if "selected_creative_types" not in st.session_state:
        st.session_state.selected_creative_types = []

    # RAG configuration (Step 2)
    if "rag_top_k" not in st.session_state:
        st.session_state.rag_top_k = 5

    # Selected agents (Step 3)
    if "selected_agents" not in st.session_state:
        st.session_state.selected_agents = ["visual_analysis", "ux_critique"]

    # Analysis status
    if "last_run_status" not in st.session_state:
        st.session_state.last_run_status = "Initialized"

    # Cached analysis results
    if "analysis_results" not in st.session_state:
        st.session_state.analysis_results = None

    # Analysis complete flag (triggers navigation to results)
    if "analysis_complete" not in st.session_state:
        st.session_state.analysis_complete = False

    # BYOK API key (from session)
    if "api_key" not in st.session_state:
        st.session_state.api_key = None

    # Brand Rules (Step 2 - Optional)
    if "brand_rules_enabled" not in st.session_state:
        st.session_state.brand_rules_enabled = False

    if "brand_rules" not in st.session_state:
        st.session_state.brand_rules = None

    if "brand_rules_source" not in st.session_state:
        st.session_state.brand_rules_source = None

    if "brand_rules_text" not in st.session_state:
        st.session_state.brand_rules_text = ""


def is_step_complete(step: int) -> bool:
    """
    Check if a step is complete.

    Args:
        step: Step number (1, 2, or 3)

    Returns:
        True if step is complete, False otherwise
    """
    if step == 1:
        # Step 1 complete if at least one file is uploaded
        uploads = st.session_state.get("uploads", [])
        legacy_uploads = st.session_state.get("uploaded_files", [])

        # Handle case where uploads/legacy_uploads might be a single UploadedFile or a list
        try:
            uploads_count = len(uploads) if isinstance(
                uploads, list) else (1 if uploads else 0)
        except TypeError:
            uploads_count = 1 if uploads else 0

        try:
            legacy_count = len(legacy_uploads) if isinstance(
                legacy_uploads, list) else (1 if legacy_uploads else 0)
        except TypeError:
            legacy_count = 1 if legacy_uploads else 0

        return uploads_count > 0 or legacy_count > 0

    elif step == 2:
        # Step 2 complete if context is confirmed
        return st.session_state.get("context_confirmed", False)

    elif step == 3:
        # Step 3 complete if agents are selected
        return len(st.session_state.get("selected_agents", [])) > 0

    return False


def advance_to_step(step: int):
    """
    Advance to a specific step (if current step is complete).

    Args:
        step: Target step number (1-3)
    """
    if step > 1 and not is_step_complete(step - 1):
        st.error(f"⚠️ Complete Step {step - 1} before proceeding")
        return False

    st.session_state.current_step = step
    st.rerun()


def can_proceed_to_step(step: int) -> bool:
    """
    Check if user can proceed to a specific step.

    Args:
        step: Target step number

    Returns:
        True if user can proceed, False otherwise
    """
    if step <= 1:
        return True

    return is_step_complete(step - 1)


def reset_wizard():
    """Reset wizard to initial state."""
    st.session_state.current_step = 1
    st.session_state.uploaded_files = []
    st.session_state.compare_mode = False
    st.session_state.platform = "Instagram"
    st.session_state.selected_platforms = []
    st.session_state.creative_type = "Marketing Creative"
    st.session_state.selected_creative_types = []
    st.session_state.selected_agents = ["visual_analysis", "ux_critique"]
    st.session_state.last_run_status = "Initialized"
    st.session_state.analysis_results = None


def get_wizard_summary() -> dict:
    """
    Get a summary of current wizard state.

    Returns:
        Dictionary with current state values
    """
    return {
        "step": st.session_state.current_step,
        "files_count": len(st.session_state.get("uploaded_files", [])),
        "compare_mode": st.session_state.get("compare_mode", False),
        "platform": st.session_state.get("platform", "Not set"),
        "creative_type": st.session_state.get("creative_type", "Not set"),
        "agents": st.session_state.get("selected_agents", []),
        "status": st.session_state.get("last_run_status", "Initialized")
    }
