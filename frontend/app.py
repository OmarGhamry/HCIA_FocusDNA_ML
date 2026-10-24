"""
FocusDNA Streamlit dashboard entry point.

Launch from project root:
    streamlit run frontend/app.py
"""

from __future__ import annotations

import sys
from pathlib import Path

# Ensure `frontend/` is on sys.path for `core`, `ui`, and `views` imports.
_FRONTEND_DIR = Path(__file__).resolve().parent
if str(_FRONTEND_DIR) not in sys.path:
    sys.path.insert(0, str(_FRONTEND_DIR))

import streamlit as st

from ui.theme import (
    configure_page,
    init_theme,
    inject_custom_css,
    render_sidebar_brand,
    render_theme_toggle,
)
from views import explain, explorer, home, model_lab, predictor, simulator


def main() -> None:
    configure_page()
    init_theme()
    render_sidebar_brand()
    render_theme_toggle()
    inject_custom_css(st.session_state.focusdna_theme)

    pages = [
        st.Page(home.render, title="Overview", icon="🏠", url_path="overview", default=True),
        st.Page(predictor.render, title="Predictor", icon="🎯", url_path="predictor"),
        st.Page(simulator.render, title="Focus Simulator", icon="🔬", url_path="simulator"),
        st.Page(model_lab.render, title="Model Lab", icon="📊", url_path="model-lab"),
        st.Page(explorer.render, title="Dataset Explorer", icon="🗂️", url_path="explorer"),
        st.Page(explain.render, title="Why This Prediction?", icon="💡", url_path="explain"),
    ]

    nav = st.navigation(pages, position="sidebar")
    nav.run()


if __name__ == "__main__":
    main()
