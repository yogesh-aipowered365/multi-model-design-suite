"""AIpowered365 Labs Theme System"""
import streamlit as st
from pathlib import Path


def apply_aip365_theme(app_title: str = "AIpowered365 Labs", page_icon: str = "🚀"):
    """
    Apply the AIpowered365 Labs premium theme to the Streamlit app.

    Args:
        app_title: The page title to display in browser tabs
        page_icon: Emoji icon for the browser tab
    """
    # Set page config
    st.set_page_config(
        page_title=app_title,
        page_icon=page_icon,
        layout="wide",
        initial_sidebar_state="collapsed",
    )

    # Load and inject theme CSS
    css_path = Path(__file__).parent / "styles" / "aip365.css"
    if css_path.exists():
        with open(css_path, "r", encoding="utf-8") as f:
            css_content = f.read()
        st.markdown(f"<style>{css_content}</style>", unsafe_allow_html=True)

    # Inject custom Streamlit theme overrides
    st.markdown("""
    <style>
    /* Remove default Streamlit padding/margins */
    .main {
        padding-top: 2rem;
    }
    
    /* ========================================================================
       WIZARD CONTAINER STYLING - Stepper & Body
       ======================================================================== */
    /* Stepper panel wrapper */
    div[data-testid="stVerticalBlock"]:has(.aip365-stepper-marker){
        margin-top: 10px !important;
        margin-bottom: 10px !important;  /* keeps stepper close to Step 1 */
    }

    /* Body wrapper: no extra padding/mystery blank bars */
    div[data-testid="stVerticalBlock"]:has(.aip365-body-marker){
        margin-top: 0px !important;
        padding-top: 0px !important;
    }

    /* Hidden marker divs */
    .aip365-stepper-marker, .aip365-body-marker {
        height: 0;
        overflow: hidden;
        visibility: hidden;
    }
    
    /* ========================================================================
       BRANDED PILL BUTTON STYLING - STEP 2 ONLY (Set Context)
       Matches hero pill design: light purple gradient background with subtle border
       ======================================================================== */
    
    /* Chip button container styling */
    .aip365-chips-container {
        display: flex;
        gap: 12px;
        flex-wrap: wrap;
        margin: 12px 0;
    }
    
    /* === Step 2 Pill Chips (Platform + Creative Type) === */
    .aip365-step2-pillwrap div[data-testid="stButton"] > button,
    .aip365-step2-pillwrap .stButton > button {
      border-radius: 999px !important;
      padding: 10px 16px !important;
      border: 1px solid rgba(120,110,255,0.28) !important;
      background: rgba(120,110,255,0.12) !important;
      color: #5b57ff !important;
      font-weight: 700 !important;
      box-shadow: none !important;
      transition: all 160ms ease-in-out !important;
    }

    /* Hover = silver */
    .aip365-step2-pillwrap div[data-testid="stButton"] > button:hover,
    .aip365-step2-pillwrap .stButton > button:hover {
      background: #e6e6e6 !important;
      border-color: #d0d0d0 !important;
      color: #222 !important;
    }

    /* Selected state comes from wrapper class pill-on in steps.py */
    .aip365-step2-pillwrap .pill-on div[data-testid="stButton"] > button,
    .aip365-step2-pillwrap .pill-on .stButton > button {
      background: rgba(120,110,255,0.24) !important;
      border: 1px solid rgba(120,110,255,0.78) !important;
      color: #3f3aff !important;
      font-weight: 800 !important;
    }

    /* Unselected wrapper optional, keeps contrast */
    .aip365-step2-pillwrap .pill-off div[data-testid="stButton"] > button,
    .aip365-step2-pillwrap .pill-off .stButton > button {
      opacity: 1 !important;
    }

    /* Clear button styling within Step 2 - keep it minimal and subtle */
    .aip365-step2-pillwrap [id*="clear_btn"] {
        font-size: 12px !important;
        padding: 6px 12px !important;
        border-radius: 6px !important;
        font-weight: 500 !important;
        background: transparent !important;
        border: 1px solid #d1d5db !important;
        color: #6b7280 !important;
    }

    .aip365-step2-pillwrap [id*="clear_btn"]:hover {
        background: #f3f4f6 !important;
        border-color: #9ca3af !important;
        color: #374151 !important;
    }
    
    /* Override Streamlit button styling */
    [data-testid="baseButton-primary"] {
        width: 100% !important;
        background: linear-gradient(135deg, #7C3AED 0%, #22D3EE 100%) !important;
        border-radius: 12px !important;
    }
    
    /* Override file uploader styling */
    [data-testid="stFileUploadDropzone"] {
        border: 2.5px dashed #4B5563 !important;
        border-radius: 16px !important;
        padding: 48px 32px !important;
        background: linear-gradient(135deg, #F9FAFB 0%, #F3F4F6 100%) !important;
    }
    
    /* Hide Streamlit menu and footer */
    #MainMenu {
        visibility: hidden;
    }
    
    footer {
        visibility: hidden;
    }
    
    /* Custom selectbox and input styling */
    .stSelectbox, .stTextInput {
        margin-bottom: 12px;
    }
    
    /* Tab styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 12px;
    }
    
    .stTabs [data-baseweb="tab"] {
        border-radius: 8px;
        padding: 10px 16px;
    }
    
    .stTabs [aria-selected="true"] {
        border-bottom: 3px solid #7C3AED !important;
    }
    </style>
    """, unsafe_allow_html=True)
