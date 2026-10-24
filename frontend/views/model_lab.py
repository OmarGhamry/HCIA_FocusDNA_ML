"""Model Lab — metrics from saved evaluation CSVs."""

from __future__ import annotations

import plotly.express as px
import streamlit as st

from core.loaders import load_classification_results, load_regression_results
from ui.charts import apply_layout
from ui.theme import disclaimer_box, section_header


def render() -> None:
    section_header(
        "Model Lab",
        "Classification and regression benchmarks from saved experiment CSVs (synthetic data).",
    )

    try:
        cls_df = load_classification_results()
        reg_df = load_regression_results()
    except FileNotFoundError as exc:
        st.error(str(exc))
        return

    tab_cls, tab_reg = st.tabs(["Classification", "Regression"])

    with tab_cls:
        st.markdown("Metrics loaded from `classification_results.csv` — not recomputed in the app.")
        st.dataframe(cls_df, use_container_width=True, hide_index=True)

        best_f1_idx = cls_df["F1-Score"].idxmax()
        best_acc_idx = cls_df["Accuracy"].idxmax()
        st.caption(
            f"Best F1: **{cls_df.loc[best_f1_idx, 'Model']}** ({cls_df.loc[best_f1_idx, 'Features']}) · "
            f"Highest accuracy: **{cls_df.loc[best_acc_idx, 'Model']}** ({cls_df.loc[best_acc_idx, 'Features']})"
        )

        melt = cls_df.melt(
            id_vars=["Model", "Features"],
            value_vars=["Accuracy", "Precision", "Recall", "F1-Score"],
            var_name="Metric",
            value_name="Score",
        )
        melt["Label"] = melt["Model"] + " · " + melt["Features"]
        fig = px.bar(
            melt,
            x="Label",
            y="Score",
            color="Metric",
            barmode="group",
            title="Classification metrics by model",
        )
        fig.update_layout(xaxis_tickangle=-35)
        st.plotly_chart(apply_layout(fig), use_container_width=True)

    with tab_reg:
        st.markdown("Metrics loaded from `regression_results.csv` — not recomputed in the app.")
        st.dataframe(reg_df, use_container_width=True, hide_index=True)

        best_r2_idx = reg_df["R2"].idxmax()
        best_mae_idx = reg_df["MAE"].idxmin()
        st.caption(
            f"Best R²: **{reg_df.loc[best_r2_idx, 'Model']}** ({reg_df.loc[best_r2_idx, 'Features']}) · "
            f"Lowest MAE: **{reg_df.loc[best_mae_idx, 'Model']}** ({reg_df.loc[best_mae_idx, 'Features']})"
        )

        melt_reg = reg_df.melt(
            id_vars=["Model", "Features"],
            value_vars=["MAE", "RMSE"],
            var_name="Error metric",
            value_name="Value",
        )
        melt_reg["Label"] = melt_reg["Model"] + " · " + melt_reg["Features"]
        fig_err = px.bar(
            melt_reg,
            x="Label",
            y="Value",
            color="Error metric",
            barmode="group",
            title="Regression error (lower is better)",
        )
        fig_err.update_layout(xaxis_tickangle=-35)
        st.plotly_chart(apply_layout(fig_err), use_container_width=True)

        fig_r2 = px.bar(
            reg_df.assign(Label=reg_df["Model"] + " · " + reg_df["Features"]),
            x="Label",
            y="R2",
            title="R² by model (higher is better)",
            color="R2",
            color_continuous_scale=["#1a2235", "#a78bfa"],
        )
        fig_r2.update_layout(xaxis_tickangle=-35, showlegend=False)
        st.plotly_chart(apply_layout(fig_r2), use_container_width=True)

    disclaimer_box(
        "All metrics reflect model comparisons on the **synthetic** FocusDNA train/test split "
        "documented in the project notebook."
    )
