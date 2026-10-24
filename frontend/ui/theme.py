"""Page configuration and global styling."""

from __future__ import annotations

from pathlib import Path

import streamlit as st

from core.paths import assets_dir

THEME_VARS: dict[str, dict[str, str]] = {
    "dark": {
        "--bg-deep": "#0b0f1a",
        "--bg-panel": "#151c2e",
        "--bg-elevated": "#1e293b",
        "--border": "rgba(148, 163, 184, 0.22)",
        "--text": "#f1f5f9",
        "--text-muted": "#94a3b8",
        "--text-on-accent": "#ffffff",
        "--header-bg": "rgba(11, 15, 26, 0.92)",
        "--nav-text": "#cbd5e1",
        "--nav-active-bg": "rgba(139, 92, 246, 0.22)",
        "--nav-active-text": "#f8fafc",
        "--disclaimer-bg": "rgba(251, 191, 36, 0.12)",
        "--disclaimer-border": "rgba(251, 191, 36, 0.35)",
        "--disclaimer-text": "#fde68a",
        "--hero-title-start": "#f8fafc",
        "--hero-title-end": "#cbd5e1",
    },
    "light": {
        "--bg-deep": "#f1f5f9",
        "--bg-panel": "#ffffff",
        "--bg-elevated": "#f8fafc",
        "--border": "rgba(15, 23, 42, 0.12)",
        "--text": "#0f172a",
        "--text-muted": "#475569",
        "--text-on-accent": "#ffffff",
        "--header-bg": "rgba(255, 255, 255, 0.95)",
        "--nav-text": "#334155",
        "--nav-active-bg": "rgba(139, 92, 246, 0.14)",
        "--nav-active-text": "#5b21b6",
        "--disclaimer-bg": "#fffbeb",
        "--disclaimer-border": "#fcd34d",
        "--disclaimer-text": "#92400e",
        "--hero-title-start": "#0f172a",
        "--hero-title-end": "#334155",
    },
}


def configure_page() -> None:
    st.set_page_config(
        page_title="FocusDNA",
        page_icon="🧬",
        layout="wide",
        initial_sidebar_state="expanded",
    )


def init_theme() -> str:
    if "focusdna_theme" not in st.session_state:
        st.session_state.focusdna_theme = "dark"
    return st.session_state.focusdna_theme


def render_theme_toggle() -> None:
    st.sidebar.markdown("**Appearance**")
    choice = st.sidebar.segmented_control(
        "Theme",
        options=["Dark", "Light"],
        default="Dark" if st.session_state.focusdna_theme == "dark" else "Light",
        label_visibility="collapsed",
        key="focusdna_theme_control",
    )
    st.session_state.focusdna_theme = "dark" if choice == "Dark" else "light"
    st.sidebar.divider()


def inject_custom_css(theme: str | None = None) -> None:
    theme = theme or init_theme()
    vars_block = THEME_VARS.get(theme, THEME_VARS["dark"])
    var_css = "\n".join(f"  {k}: {v};" for k, v in vars_block.items())

    css_path = assets_dir() / "styles.css"
    base = css_path.read_text(encoding="utf-8") if css_path.is_file() else ""

    st.markdown(
        f"""
        <style>
        .stApp {{
        {var_css}
        }}
        </style>
        <style>{base}</style>
        """,
        unsafe_allow_html=True,
    )


def render_sidebar_brand() -> None:
    st.sidebar.markdown(
        """
        <div class="sidebar-brand">
            <div class="brand-mark">FD</div>
            <div>
                <div class="brand-title">FocusDNA</div>
                <div class="brand-sub">Study Session Predictor</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.sidebar.markdown(
        '<p class="sidebar-note">Synthetic dataset · Experimental ML demo</p>',
        unsafe_allow_html=True,
    )


def metric_card(label: str, value: str, hint: str = "") -> None:
    hint_html = f'<div class="metric-hint">{hint}</div>' if hint else ""
    st.markdown(
        f"""
        <div class="metric-card">
            <div class="metric-label">{label}</div>
            <div class="metric-value">{value}</div>
            {hint_html}
        </div>
        """,
        unsafe_allow_html=True,
    )


def section_header(title: str, subtitle: str = "") -> None:
    sub = f'<p class="section-sub">{subtitle}</p>' if subtitle else ""
    st.markdown(
        f'<div class="section-head"><h2>{title}</h2>{sub}</div>',
        unsafe_allow_html=True,
    )


def disclaimer_box(text: str) -> None:
    st.markdown(f'<div class="disclaimer">{text}</div>', unsafe_allow_html=True)
