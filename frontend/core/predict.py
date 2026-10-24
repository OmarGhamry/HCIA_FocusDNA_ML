"""Run inference using saved preprocessing and models (no retraining)."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd


def row_from_features(feature_values: dict[str, Any], feature_order: list[str]) -> pd.DataFrame:
    """Build a single-row DataFrame with exact training column names and order."""
    missing = [c for c in feature_order if c not in feature_values]
    if missing:
        raise ValueError(f"Missing input features: {', '.join(missing)}")
    row = {col: feature_values[col] for col in feature_order}
    return pd.DataFrame([row], columns=feature_order)


def predict_focus_and_success(
    preprocessor,
    classifier,
    regressor,
    inputs: pd.DataFrame,
) -> dict[str, Any]:
    """
    Transform raw inputs once via the saved ColumnTransformer, then predict.

    Classifier pipeline applies StandardScaler internally on processed features.
    """
    X_processed = preprocessor.transform(inputs)
    focus_raw = float(regressor.predict(X_processed)[0])
    focus_display = float(np.clip(focus_raw, 0.0, 100.0))

    class_pred = int(classifier.predict(X_processed)[0])
    proba = classifier.predict_proba(X_processed)[0]
    classes = list(classifier.named_steps["model"].classes_)
    if 1 in classes:
        success_probability = float(proba[classes.index(1)])
    else:
        success_probability = float(proba[-1])

    return {
        "focus_score_raw": focus_raw,
        "focus_score": focus_display,
        "session_success": class_pred,
        "success_probability": success_probability,
        "failure_probability": 1.0 - success_probability,
    }
