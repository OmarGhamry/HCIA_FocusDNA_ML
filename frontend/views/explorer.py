"""Dataset explorer."""

from __future__ import annotations

import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from core.loaders import load_dataset
from ui.charts import apply_layout
from ui.theme import section_header


def render() -> None:
    section_header("Dataset Explorer", "Interactive views of focusdna_dataset.csv")

    try:
        df = load_dataset()
    except FileNotFoundError as exc:
        st.error(str(exc))
        return

    st.markdown(
        f"**{len(df):,}** rows · **{len(df.columns)}** columns · "
        "Synthetic study sessions with Focus Score and Session Success targets."
    )

    with st.expander("Filters", expanded=False):
        env_filter = st.multiselect(
            "Study environment",
            sorted(df["Study_Environment"].unique()),
            default=sorted(df["Study_Environment"].unique()),
        )
        time_filter = st.multiselect(
            "Study time",
            sorted(df["Study_Time"].unique()),
            default=sorted(df["Study_Time"].unique()),
        )
        success_filter = st.selectbox("Session success", ["All", "Successful (1)", "Unsuccessful (0)"])

    filtered = df[
        df["Study_Environment"].isin(env_filter) & df["Study_Time"].isin(time_filter)
    ]
    if success_filter == "Successful (1)":
        filtered = filtered[filtered["Session_Success"] == 1]
    elif success_filter == "Unsuccessful (0)":
        filtered = filtered[filtered["Session_Success"] == 0]

    c1, c2, c3 = st.columns(3)
    with c1:
        st.metric("Filtered rows", f"{len(filtered):,}")
    with c2:
        missing = int(filtered.isna().sum().sum())
        st.metric("Missing values", missing)
    with c3:
        st.metric("Mean focus score", f"{filtered['Focus_Score'].mean():.1f}")

    st.dataframe(filtered.head(50), use_container_width=True, hide_index=True)

    col_a, col_b = st.columns(2)
    with col_a:
        fig_succ = px.histogram(
            filtered,
            x="Session_Success",
            color="Session_Success",
            title="Session success distribution",
            labels={"Session_Success": "Success (0/1)"},
        )
        st.plotly_chart(apply_layout(fig_succ), use_container_width=True)

        fig_focus = px.histogram(
            filtered,
            x="Focus_Score",
            nbins=30,
            title="Focus score distribution",
        )
        st.plotly_chart(apply_layout(fig_focus), use_container_width=True)

    with col_b:
        fig_time = px.histogram(
            filtered,
            x="Study_Time",
            title="Study time distribution",
        )
        st.plotly_chart(apply_layout(fig_time), use_container_width=True)

        fig_env = px.pie(
            filtered,
            names="Study_Environment",
            title="Study environment share",
            hole=0.45,
        )
        st.plotly_chart(apply_layout(fig_env), use_container_width=True)

    numeric = filtered.select_dtypes(include="number")
    if len(numeric.columns) >= 2:
        corr = numeric.corr()
        fig_hm = go.Figure(
            data=go.Heatmap(
                z=corr.values,
                x=list(corr.columns),
                y=list(corr.index),
                colorscale=[[0, "#0b0f1a"], [0.5, "#6366f1"], [1, "#22d3ee"]],
                zmin=-1,
                zmax=1,
            )
        )
        st.plotly_chart(apply_layout(fig_hm, "Feature correlation (numeric)"), use_container_width=True)
