"""Model building utilities.

This module contains helpers to construct the ML pipeline used in training and
to extract feature importances for reporting. The pipeline typically includes
scaling (RobustScaler) for numeric features, optional SMOTE oversampling to
handle class imbalance, and a classifier (logistic regression or random forest).
"""

from __future__ import annotations

from typing import List

import numpy as np
import pandas as pd
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import RobustScaler


SUPPORTED_MODELS = {"logistic_regression", "random_forest"}


def create_model(model_name: str, random_state: int):
    """Factory for supported models.

    - `logistic_regression`: linear model, good baseline and interpretable coefficients.
    - `random_forest`: non-linear ensemble that provides `feature_importances_`.
    """
    if model_name not in SUPPORTED_MODELS:
        raise ValueError(f"Unsupported model '{model_name}'. Use one of: {sorted(SUPPORTED_MODELS)}")

    if model_name == "logistic_regression":
        # Balanced class weights help with skewed classes.
        return LogisticRegression(
            max_iter=1500,
            class_weight="balanced",
            solver="liblinear",
            random_state=random_state,
        )

    # Random forest configuration: moderate depth and many estimators for stability.
    return RandomForestClassifier(
        n_estimators=250,
        max_depth=10,
        min_samples_leaf=2,
        n_jobs=-1,
        class_weight="balanced_subsample",
        random_state=random_state,
    )


def build_training_pipeline(
    model_name: str,
    numeric_features: List[str],
    use_smote: bool,
    random_state: int,
):
    """Construct a training pipeline.

    Steps:
    1. Scale numeric features with `RobustScaler` to reduce influence of outliers.
    2. Optionally apply SMOTE oversampling to synthetically balance minority class.
    3. Fit the chosen classifier.
    """
    preprocessor = ColumnTransformer(
        transformers=[("scale", RobustScaler(), numeric_features)] if numeric_features else [],
        remainder="passthrough",
    )

    steps = [("preprocess", preprocessor)]
    if use_smote:
        # SMOTE increases minority class examples by synthesizing new samples.
        steps.append(("smote", SMOTE(random_state=random_state)))
    steps.append(("model", create_model(model_name, random_state=random_state)))
    return Pipeline(steps=steps)


def extract_feature_importance(
    trained_pipeline,
    selected_features: List[str],
    numeric_features: List[str],
) -> pd.DataFrame:
    """Return a DataFrame of features and importances.

    The function tries to extract importances from the underlying estimator. For
    tree-based models use `feature_importances_`. For linear models use the
    absolute value of coefficients. If the estimator does not expose either,
    zeros are returned (so the pipeline remains robust).
    """
    ordered_features = numeric_features + [f for f in selected_features if f not in numeric_features]
    model = trained_pipeline.named_steps["model"]

    if hasattr(model, "feature_importances_"):
        importances = np.asarray(model.feature_importances_, dtype=float)
    elif hasattr(model, "coef_"):
        importances = np.abs(np.asarray(model.coef_).ravel())
    else:
        importances = np.zeros(len(ordered_features), dtype=float)

    return (
        pd.DataFrame({"feature": ordered_features, "importance": importances})
        .sort_values("importance", ascending=False)
        .reset_index(drop=True)
    )
