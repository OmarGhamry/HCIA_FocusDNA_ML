"""Cached loading of models, datasets, and evaluation tables."""

from __future__ import annotations

import joblib
import pandas as pd
import streamlit as st

from core.paths import data_dir, models_dir


@st.cache_resource(show_spinner="Loading ML artifacts…")
def load_artifacts():
    mdir = models_dir()
    required = [
        "preprocessor.pkl",
        "classifier.pkl",
        "regressor.pkl",
        "input_features.pkl",
        "processed_features.pkl",
    ]
    missing = [name for name in required if not (mdir / name).exists()]
    if missing:
        raise FileNotFoundError(
            f"Missing model file(s) in {mdir}: {', '.join(missing)}"
        )

    return {
        "preprocessor": joblib.load(mdir / "preprocessor.pkl"),
        "classifier": joblib.load(mdir / "classifier.pkl"),
        "regressor": joblib.load(mdir / "regressor.pkl"),
        "input_features": list(joblib.load(mdir / "input_features.pkl")),
        "processed_features": list(joblib.load(mdir / "processed_features.pkl")),
    }


@st.cache_data(show_spinner="Loading dataset…")
def load_dataset() -> pd.DataFrame:
    path = data_dir() / "focusdna_dataset.csv"
    if not path.exists():
        raise FileNotFoundError(f"Dataset not found: {path}")
    return pd.read_csv(path)


@st.cache_data(show_spinner="Loading evaluation results…")
def load_classification_results() -> pd.DataFrame:
    path = data_dir() / "classification_results.csv"
    if not path.exists():
        raise FileNotFoundError(f"Classification results not found: {path}")
    return pd.read_csv(path)


@st.cache_data(show_spinner="Loading evaluation results…")
def load_regression_results() -> pd.DataFrame:
    path = data_dir() / "regression_results.csv"
    if not path.exists():
        raise FileNotFoundError(f"Regression results not found: {path}")
    return pd.read_csv(path)
