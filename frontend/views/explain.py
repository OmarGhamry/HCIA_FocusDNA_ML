"""Why this prediction? — honest model behavior notes."""

from __future__ import annotations

import streamlit as st

from core.loaders import load_artifacts
from ui.charts import horizontal_bar
from ui.theme import disclaimer_box, section_header


def _pretty_name(raw: str) -> str:
    return (
        raw.replace("cat__", "")
        .replace("num__", "")
        .replace("_", " ")
    )


def render() -> None:
    section_header(
        "Why This Prediction?",
        "What the saved models learned on synthetic data — associations, not causation.",
    )

    try:
        artifacts = load_artifacts()
    except FileNotFoundError as exc:
        st.error(str(exc))
        return

    processed = artifacts["processed_features"]
    reg = artifacts["regressor"]
    clf = artifacts["classifier"]
    lr = clf.named_steps["model"]

    st.markdown(
        """
        **Inputs** (from `input_features.pkl`) are one-hot encoded and passed through unchanged
        for numeric columns via the saved `ColumnTransformer`. The **classifier** then applies
        `StandardScaler` + logistic regression on that matrix. The **regressor** uses the same
        preprocessed matrix with linear regression — no second encoding step.
        """
    )

    st.markdown("### Linear regression — largest coefficients")
    st.markdown(
        "Coefficients apply to **preprocessed** features (one-hot columns for study time & environment). "
        "Positive values tend to increase predicted Focus Score; negative values tend to decrease it."
    )

    reg_pairs = sorted(zip(processed, reg.coef_), key=lambda x: x[1])
    top_neg = reg_pairs[:5]
    top_pos = reg_pairs[-5:][::-1]

    c1, c2 = st.columns(2)
    with c1:
        st.plotly_chart(
            horizontal_bar(
                [_pretty_name(n) for n, _ in top_neg],
                [v for _, v in top_neg],
                "Strongest negative associations",
                "#f87171",
            ),
            use_container_width=True,
        )
    with c2:
        st.plotly_chart(
            horizontal_bar(
                [_pretty_name(n) for n, _ in top_pos],
                [v for _, v in top_pos],
                "Strongest positive associations",
                "#34d399",
            ),
            use_container_width=True,
        )

    st.markdown("### Classification — logistic regression weights")
    st.markdown(
        "These are **scaled** logistic coefficients (after `StandardScaler` in the saved pipeline). "
        "They indicate direction of association with predicted success on synthetic data — "
        "not SHAP values or guaranteed real-world drivers."
    )

    clf_pairs = sorted(zip(processed, lr.coef_[0]), key=lambda x: x[1])
    clf_neg = clf_pairs[:5]
    clf_pos = clf_pairs[-5:][::-1]

    d1, d2 = st.columns(2)
    with d1:
        st.plotly_chart(
            horizontal_bar(
                [_pretty_name(n) for n, _ in clf_neg],
                [v for _, v in clf_neg],
                "Weights ↓ success odds (approx.)",
                "#f87171",
            ),
            use_container_width=True,
        )
    with d2:
        st.plotly_chart(
            horizontal_bar(
                [_pretty_name(n) for n, _ in clf_pos],
                [v for _, v in clf_pos],
                "Weights ↑ success odds (approx.)",
                "#a78bfa",
            ),
            use_container_width=True,
        )

    disclaimer_box(
        "Synthetic datasets embed intentional patterns. Model coefficients describe those patterns "
        "in training data only and must not be read as medical, psychological, or causal claims."
    )
