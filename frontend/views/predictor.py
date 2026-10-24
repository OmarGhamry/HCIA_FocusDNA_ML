"""Study session predictor form."""

from __future__ import annotations

import streamlit as st

from core.defaults import dataset_defaults
from core.loaders import load_artifacts, load_dataset
from core.predict import predict_focus_and_success, row_from_features
from ui.charts import focus_gauge
from ui.theme import disclaimer_box, section_header


def _render_inputs(defaults: dict, feature_order: list[str]) -> dict:
    values: dict = {}
    col_a, col_b = st.columns(2)

    numeric_specs = {
        "Sleep_Hours": (3.0, 10.0, 0.1),
        "Previous_Screen_Time": (0.0, 12.0, 0.1),
        "Stress_Level": (1, 10, 1),
        "Breaks_Taken": (0, 6, 1),
        "Task_Difficulty": (1, 10, 1),
        "Previous_Session_Score": (0.0, 100.0, 0.5),
        "Phone_Notifications": (0, 20, 1),
        "Environment_Noise": (0, 10, 1),
        "Hydration_Level": (1, 10, 1),
    }

    left_nums = ["Sleep_Hours", "Previous_Screen_Time", "Stress_Level", "Breaks_Taken", "Task_Difficulty"]
    right_nums = [
        "Previous_Session_Score",
        "Phone_Notifications",
        "Environment_Noise",
        "Hydration_Level",
    ]

    with col_a:
        values["Study_Time"] = st.selectbox(
            "Study Time",
            ["Morning", "Afternoon", "Evening", "Night"],
            index=["Morning", "Afternoon", "Evening", "Night"].index(defaults["Study_Time"]),
        )
        values["Study_Environment"] = st.selectbox(
            "Study Environment",
            ["Home", "Library", "Cafe"],
            index=["Home", "Library", "Cafe"].index(defaults["Study_Environment"]),
        )
        for name in left_nums:
            lo, hi, step = numeric_specs[name]
            default = defaults[name]
            if step == 1:
                values[name] = st.slider(name.replace("_", " "), int(lo), int(hi), int(default))
            else:
                values[name] = st.slider(
                    name.replace("_", " "), float(lo), float(hi), float(default), step=step
                )

    with col_b:
        for name in right_nums:
            lo, hi, step = numeric_specs[name]
            default = defaults[name]
            if step == 1:
                values[name] = st.slider(name.replace("_", " "), int(lo), int(hi), int(default))
            else:
                values[name] = st.slider(
                    name.replace("_", " "), float(lo), float(hi), float(default), step=step
                )

    ordered = {k: values[k] for k in feature_order if k in values}
    for k in feature_order:
        if k not in ordered:
            ordered[k] = defaults[k]
    return ordered


def render() -> None:
    section_header(
        "Study Session Predictor",
        "Enter session details — predictions come from saved preprocessor, classifier, and regressor.",
    )

    try:
        artifacts = load_artifacts()
        df = load_dataset()
    except FileNotFoundError as exc:
        st.error(str(exc))
        return

    feature_order = artifacts["input_features"]
    defaults = dataset_defaults(df, feature_order)

    with st.container(border=True):
        inputs = _render_inputs(defaults, feature_order)

    if st.button("Predict session", type="primary", use_container_width=False):
        try:
            row = row_from_features(inputs, feature_order)
            result = predict_focus_and_success(
                artifacts["preprocessor"],
                artifacts["classifier"],
                artifacts["regressor"],
                row,
            )
        except Exception as exc:
            st.error(f"Prediction failed: {exc}")
            return

        success_label = "Successful session" if result["session_success"] == 1 else "Unsuccessful session"
        success_class = "result-success" if result["session_success"] == 1 else "result-fail"
        prob_pct = result["success_probability"] * 100

        c1, c2 = st.columns([1.1, 1])
        with c1:
            st.plotly_chart(focus_gauge(result["focus_score"]), use_container_width=True)
            if result["focus_score_raw"] != result["focus_score"]:
                st.caption(
                    f"Raw model output: {result['focus_score_raw']:.2f} (display clamped to 0–100)."
                )
        with c2:
            with st.container(border=True):
                st.markdown("#### Results")
                st.markdown(
                    f'<p class="{success_class}">{success_label}</p>',
                    unsafe_allow_html=True,
                )
                st.metric("Probability of success", f"{prob_pct:.1f}%")
                st.progress(min(max(result["success_probability"], 0.0), 1.0))
                st.caption(f"Predicted class: {result['session_success']}")

        disclaimer_box(
            "Experimental prediction from models trained on synthetic data. "
            "Not a validated assessment of your real-world focus or productivity."
        )
