
from pathlib import Path

import streamlit as st
from textwrap import dedent

# ==================================================
# APPLICATION CONFIGURATION
# ==================================================

PROJECT_DIR = Path(__file__).resolve().parent

st.set_page_config(
    page_title="Water Infrastructure Intelligence",
    page_icon="💧",
    layout="wide",
    initial_sidebar_state="expanded"
)


# ==================================================
# GLOBAL STYLING
# ==================================================

CSS_PATH = PROJECT_DIR / "assets" / "style.css"

if CSS_PATH.exists():
    css = CSS_PATH.read_text(encoding="utf-8")
    st.markdown(
        f"<style>{css}</style>",
        unsafe_allow_html=True
    )
else:
    st.warning("Global stylesheet not found.")



# ==================================================
# SIDEBAR BRANDING
# ==================================================

with st.sidebar:

    st.markdown("# 💧")

    st.markdown(
        "**WATER INFRASTRUCTURE**"
    )

    st.markdown(
        "**INTELLIGENCE**"
    )

    st.caption("MACHINE LEARNING PLATFORM")

    st.divider()

    st.caption("PROJECT NAVIGATION")

# ==================================================
# MULTIPAGE NAVIGATION
# ==================================================

pages = {

    "WORKSPACE": [
        st.Page(
            PROJECT_DIR / "pages" / "1_Overview.py",
            title="Overview",
            icon="📊",
            default=True,
            help="Explore the platform and its key indicators."
        ),

        st.Page(
            PROJECT_DIR / "pages" / "2_Assessment.py",
            title="Water Point Assessment",
            icon="💧",
            help="Submit water point characteristics for ML prediction."
        ),

        st.Page(
            PROJECT_DIR / "pages" / "3_Results.py",
            title="Prediction Results",
            icon="📈",
            help="Explore prediction results and class probabilities."
        )
    ],

    "ANALYTICS": [
        st.Page(
            PROJECT_DIR / "pages" / "4_Model_Insights.py",
            title="Model Insights",
            icon="🧠",
            help="Explore model performance and explainability."
        )
    ],

    "INFORMATION": [
        st.Page(
            PROJECT_DIR / "pages" / "5_About.py",
            title="About the Project",
            icon="ℹ️",
            help="Discover the project, methodology and architecture."
        )
    ]
}


# ==================================================
# NAVIGATION EXECUTION
# ==================================================

selected_page = st.navigation(
    pages,
    position="sidebar",
    expanded=True
)

selected_page.run()
