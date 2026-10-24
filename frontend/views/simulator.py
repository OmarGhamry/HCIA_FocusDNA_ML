"""Focus simulator — compare scenarios with the same saved models."""

from __future__ import annotations

import streamlit as st

from core.defaults import dataset_defaults
from core.loaders import load_artifacts, load_dataset
from core.predict import predict_focus_and_success, row_from_features
from ui.theme import disclaimer_box, metric_card, section_header


SIM_KEYS = [
    ("Sleep_Hours", "Sleep (hours)", 3.0, 10.0, 0.1),
    ("Stress_Level", "Stress level", 1, 10, 1),
    ("Previous_Screen_Time", "Previous screen time (hrs)", 0.0, 12.0, 0.1),
    ("Phone_Notifications", "Phone notifications", 0, 20, 1),
]


def _baseline_from_defaults(defaults: dict) -> dict:
    return dict(defaults)


def render() -> None:
    section_header(
        "Focus Simulator",
        "Adjust a few levers and compare predictions. Both scenarios use the same saved models.",
    )

    try:
        artifacts = load_artifacts()
        df = load_dataset()
    except FileNotFoundError as exc:
        st.error(str(exc))
        return

    feature_order = artifacts["input_features"]
    defaults = dataset_defaults(df, feature_order)

    if "sim_baseline" not in st.session_state:
        st.session_state.sim_baseline = _baseline_from_defaults(defaults)
    if "sim_scenario" not in st.session_state:
        st.session_state.sim_scenario = _baseline_from_defaults(defaults)

    st.markdown(
        "Other inputs stay at dataset medians / modes unless you change them on the "
        "**Study Session Predictor** page. Here you tune sleep, stress, screen time, and notifications."
    )

    col_base, col_scenario = st.columns(2)

    def _sliders(state_key: str, title: str) -> None:
        st.markdown(f"**{title}**")
        data = st.session_state[state_key]
        for field, label, lo, hi, step in SIM_KEYS:
            if step == 1:
                data[field] = st.slider(
                    label,
                    int(lo),
                    int(hi),
                    int(data[field]),
                    key=f"{state_key}_{field}",
                )
            else:
                data[field] = st.slider(
                    label,
                    float(lo),
                    float(hi),
                    float(data[field]),
                    step=step,
                    key=f"{state_key}_{field}",
                )

    with col_base:
        _sliders("sim_baseline", "Baseline scenario")
    with col_scenario:
        _sliders("sim_scenario", "What-if scenario")

    if st.button("Recalculate predictions", type="primary"):
        try:
            base_row = row_from_features(st.session_state.sim_baseline, feature_order)
            scen_row = row_from_features(st.session_state.sim_scenario, feature_order)
            base_res = predict_focus_and_success(
                artifacts["preprocessor"],
                artifacts["classifier"],
                artifacts["regressor"],
                base_row,
            )
            scen_res = predict_focus_and_success(
                artifacts["preprocessor"],
                artifacts["classifier"],
                artifacts["regressor"],
                scen_row,
            )
        except Exception as exc:
            st.error(f"Simulation failed: {exc}")
            return

        changed = [
            k
            for k, _, _, _, _ in SIM_KEYS
            if st.session_state.sim_baseline[k] != st.session_state.sim_scenario[k]
        ]
        if changed:
            st.info("Changed inputs: " + ", ".join(c.replace("_", " ") for c in changed))
        else:
            st.warning("No differences between baseline and what-if — adjust sliders to compare.")

        c1, c2, c3 = st.columns(3)
        with c1:
            metric_card("Baseline focus", f"{base_res['focus_score']:.1f}", "Saved regressor")
        with c2:
            metric_card("What-if focus", f"{scen_res['focus_score']:.1f}", "Saved regressor")
        with c3:
            delta = scen_res["focus_score"] - base_res["focus_score"]
            sign = "+" if delta >= 0 else ""
            metric_card("Focus delta", f"{sign}{delta:.1f}", "Display scores (0–100)")

        d1, d2 = st.columns(2)
        with d1:
            st.markdown("**Baseline success probability**")
            st.metric("P(success)", f"{base_res['success_probability'] * 100:.1f}%")
        with d2:
            st.markdown("**What-if success probability**")
            st.metric("P(success)", f"{scen_res['success_probability'] * 100:.1f}%")

    disclaimer_box(
        "Associations in synthetic data do not prove that changing one habit will improve "
        "a real person's focus. Use this as an exploratory what-if tool only."
    )
