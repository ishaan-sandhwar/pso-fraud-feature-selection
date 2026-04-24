from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

import joblib
import numpy as np
import pandas as pd


def load_model_bundle(model_path: str | Path) -> Dict[str, Any]:
    """Load a saved model bundle (pipeline + threshold + features)."""
    return joblib.load(Path(model_path))


def predict_transaction(
    model_bundle: Dict[str, Any],
    transaction: Dict[str, float],
) -> Dict[str, Any]:
    """Predict fraud probability for a single transaction dict.

    Args:
        model_bundle: dict with keys 'pipeline', 'threshold', 'features'
        transaction: dict mapping feature names to values

    Returns:
        dict with 'probability', 'prediction', 'threshold', 'confidence'
    """
    pipeline = model_bundle["pipeline"]
    threshold = float(model_bundle["threshold"])
    features = model_bundle["features"]

    # Build input row — missing features filled with 0.0
    row = {f: float(transaction.get(f, 0.0)) for f in features}
    X = pd.DataFrame([row])[features]

    proba = float(pipeline.predict_proba(X)[0, 1])
    prediction = int(proba >= threshold)

    # Confidence: how far from threshold
    confidence = abs(proba - threshold) / max(threshold, 1 - threshold)

    return {
        "probability": round(proba, 6),
        "prediction": prediction,
        "label": "FRAUD" if prediction == 1 else "LEGITIMATE",
        "threshold": round(threshold, 6),
        "confidence": round(min(confidence, 1.0), 4),
    }


def get_feature_defaults(model_bundle: Dict[str, Any]) -> Dict[str, float]:
    """Return zero-value defaults for all model features."""
    return {f: 0.0 for f in model_bundle["features"]}
