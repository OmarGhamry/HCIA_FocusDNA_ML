"""Plotly chart helpers with a consistent dark theme."""

from __future__ import annotations

import plotly.express as px
import plotly.graph_objects as go

CHART_LAYOUT = dict(
    template="plotly_dark",
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    font=dict(family="DM Sans, sans-serif", color="#cbd5e1"),
    margin=dict(l=20, r=20, t=40, b=20),
    colorway=["#a78bfa", "#22d3ee", "#34d399", "#fbbf24", "#f87171"],
)


def apply_layout(fig: go.Figure, title: str | None = None) -> go.Figure:
    fig.update_layout(**CHART_LAYOUT)
    if title:
        fig.update_layout(title=dict(text=title, x=0, xanchor="left", font=dict(size=16)))
    return fig


def focus_gauge(score: float) -> go.Figure:
    fig = go.Figure(
        go.Indicator(
            mode="gauge+number",
            value=score,
            number={"suffix": " / 100", "font": {"size": 36}},
            gauge={
                "axis": {"range": [0, 100], "tickcolor": "#64748b"},
                "bar": {"color": "#a78bfa"},
                "bgcolor": "#1a2235",
                "borderwidth": 0,
                "steps": [
                    {"range": [0, 40], "color": "rgba(248, 113, 113, 0.25)"},
                    {"range": [40, 70], "color": "rgba(251, 191, 36, 0.2)"},
                    {"range": [70, 100], "color": "rgba(52, 211, 153, 0.25)"},
                ],
            },
            title={"text": "Predicted Focus Score", "font": {"size": 14, "color": "#94a3b8"}},
        )
    )
    return apply_layout(fig)


def horizontal_bar(labels: list[str], values: list[float], title: str, color: str = "#a78bfa") -> go.Figure:
    fig = go.Figure(
        go.Bar(
            x=values,
            y=labels,
            orientation="h",
            marker=dict(color=color),
        )
    )
    fig.update_layout(yaxis=dict(autorange="reversed"))
    return apply_layout(fig, title)
