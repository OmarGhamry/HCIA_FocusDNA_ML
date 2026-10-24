"""Home / overview page."""

from __future__ import annotations

import streamlit as st

from core.loaders import load_classification_results, load_dataset, load_regression_results
from ui.theme import disclaimer_box, metric_card, section_header


def render() -> None:
    st.markdown(
        """
        <div class="hero">
            <div class="hero-kicker">ML-powered study analytics</div>
            <h1>FocusDNA</h1>
            <p class="hero-tagline">
                Understand your study patterns. Predict your next session.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    try:
        df = load_dataset()
        cls_results = load_classification_results()
        reg_results = load_regression_results()
    except FileNotFoundError as exc:
        st.error(str(exc))
        return

    best_cls = cls_results.loc[cls_results["F1-Score"].idxmax()]
    best_reg = reg_results.loc[reg_results["R2"].idxmax()]

    st.markdown(
        """
        FocusDNA loads your **saved scikit-learn models** to estimate **Focus Score** (regression)
        and **Session Success** (classification) from study-session inputs. All metrics below
        come from evaluation CSVs produced in the training notebook — on **synthetic data** only.
        """
    )

    section_header("At a glance", "Dataset and deployed prediction tasks")
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        metric_card("Records", f"{len(df):,}", "Synthetic sessions")
    with c2:
        metric_card("Input features", str(len(df.columns) - 2), "Excluding targets")
    with c3:
        metric_card("Classification", "Session Success", "Binary: productive or not")
    with c4:
        metric_card("Regression", "Focus Score", "Continuous 0–100")

    section_header("Best models (saved experiments)", "From classification_results.csv & regression_results.csv")
    m1, m2 = st.columns(2)
    with m1:
        st.markdown(
            f"""
            <div class="info-panel">
              <strong>Classification — highest F1 (All features)</strong>
              <ul>
                <li>Model: <strong>{best_cls['Model']}</strong> ({best_cls['Features']} features)</li>
                <li>Accuracy: <strong>{best_cls['Accuracy']:.3f}</strong></li>
                <li>Precision: <strong>{best_cls['Precision']:.3f}</strong></li>
                <li>Recall: <strong>{best_cls['Recall']:.3f}</strong></li>
                <li>F1-score: <strong>{best_cls['F1-Score']:.3f}</strong></li>
              </ul>
              <em>Production classifier: Logistic Regression pipeline (StandardScaler + LR)
              on preprocessed features.</em>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with m2:
        st.markdown(
            f"""
            <div class="info-panel">
              <strong>Regression — highest R² (All features)</strong>
              <ul>
                <li>Model: <strong>{best_reg['Model']}</strong> ({best_reg['Features']} features)</li>
                <li>MAE: <strong>{best_reg['MAE']:.3f}</strong></li>
                <li>RMSE: <strong>{best_reg['RMSE']:.3f}</strong></li>
                <li>R²: <strong>{best_reg['R2']:.3f}</strong></li>
              </ul>
              <em>Production regressor: Linear Regression on the same ColumnTransformer output.</em>
            </div>
            """,
            unsafe_allow_html=True,
        )

    disclaimer_box(
        "This application uses a <strong>synthetic</strong> dataset and experimental models. "
        "Predictions are not validated for real-world academic or clinical use."
    )
