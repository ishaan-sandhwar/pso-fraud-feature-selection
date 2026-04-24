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
    if model_name not in SUPPORTED_MODELS:
        raise ValueError(f"Unsupported model '{model_name}'. Use one of: {sorted(SUPPORTED_MODELS)}")

    if model_name == "logistic_regression":
        return LogisticRegression(
            max_iter=1500,
            class_weight="balanced",
            solver="liblinear",
            random_state=random_state,
        )

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
    preprocessor = ColumnTransformer(
        transformers=[("scale", RobustScaler(), numeric_features)] if numeric_features else [],
        remainder="passthrough",
    )

    steps = [("preprocess", preprocessor)]
    if use_smote:
        steps.append(("smote", SMOTE(random_state=random_state)))
    steps.append(("model", create_model(model_name, random_state=random_state)))
    return Pipeline(steps=steps)


def extract_feature_importance(
    trained_pipeline,
    selected_features: List[str],
    numeric_features: List[str],
) -> pd.DataFrame:
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
