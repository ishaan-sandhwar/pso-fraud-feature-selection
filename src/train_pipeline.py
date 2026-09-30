"""End-to-end training orchestration.

This script loads data, trains a baseline model, performs PSO-based feature
selection, retrains a final model on the selected features, computes metrics,
and writes reports and figures.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, Dict, List

import joblib
import pandas as pd
from sklearn.base import clone
from sklearn.model_selection import train_test_split

from src.cache import load_with_parquet_cache
from src.config import load_config
from src.data import build_data_summary, train_test_data
from src.evaluate import best_threshold_by_f1, compute_metrics, confusion_from_threshold
from src.modeling import build_training_pipeline, extract_feature_importance
from src.pso import BinaryPSOFeatureSelector
from src.visualize import (
    save_class_distribution,
    save_confusion_heatmap,
    save_feature_importance_plot,
    save_pr_comparison,
    save_pso_history,
    save_roc_comparison,
)


def _resolve(project_root: Path, path_text: str) -> Path:
    return path_text if isinstance(path_text, Path) else project_root / path_text


def _write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        json.dump(payload, file, indent=2)


def _timer(label: str, start: float) -> None:
    elapsed = time.time() - start
    print(f"[timer] {label}: {elapsed:.1f}s")


def run_training(config_path: str = "configs/config.yaml") -> Dict[str, Any]:
    """Main entry for the training pipeline.

    Steps performed:
    1. Load data (with parquet caching and optional sampling).
    2. Train baseline model and compute metrics.
    3. Run Binary PSO to select features using a weighted objective.
    4. Retrain a final model on selected features and save results.
    """
    t0 = time.time()
    project_root = Path(__file__).resolve().parents[1]
    config = load_config(project_root / config_path)

    random_state = int(config["project"]["random_state"])
    target_column = config["data"]["target_column"]
    scale_columns = list(config["data"]["scale_columns"])
    test_size = float(config["data"]["test_size"])
    validation_size = float(config["data"]["validation_size"])
    sample_size = config["data"].get("sample_size")  # None = full dataset
    use_smote = bool(config["training"]["use_smote"])
    baseline_model_name = str(config["training"]["baseline_model"])
    search_model_name = str(config["training"]["search_model"])
    final_model_name = str(config["training"]["final_model"])
    default_threshold = float(config["training"]["default_threshold"])

    raw_data_path = _resolve(project_root, config["data"]["raw_data_path"])
    parquet_cache_path = _resolve(project_root, config["data"]["parquet_cache_path"])
    model_dir = _resolve(project_root, config["artifacts"]["model_dir"])
    output_dir = _resolve(project_root, config["artifacts"]["output_dir"])
    figure_dir = _resolve(project_root, config["artifacts"]["figure_dir"])
    report_dir = _resolve(project_root, config["artifacts"]["report_dir"])

    for d in (model_dir, output_dir, figure_dir, report_dir):
        d.mkdir(parents=True, exist_ok=True)

    # ── Data loading (parquet cache + optional stratified sample) ──────────────
    t1 = time.time()
    df = load_with_parquet_cache(
        csv_path=raw_data_path,
        parquet_path=parquet_cache_path,
        sample_size=int(sample_size) if sample_size else None,
        random_state=random_state,
        target_column=target_column,
    )
    _timer("data loading", t1)

    data_summary = build_data_summary(df, target_column)
    _write_json(report_dir / "data_summary.json", data_summary)
    save_class_distribution(df, target_column, figure_dir / "class_distribution.png")

    X_train, X_test, y_train, y_test = train_test_data(
        df=df,
        target_column=target_column,
        test_size=test_size,
        random_state=random_state,
    )
    all_features = list(X_train.columns)

    # ── Validation split (carved from the training data) ───────────────────────
    # It is used twice: PSO scores candidate feature subsets on it, and every
    # model's decision threshold is tuned on it. The test set never tunes
    # anything; each model is scored on it exactly once.
    X_search_train, X_val, y_search_train, y_val = train_test_split(
        X_train,
        y_train,
        test_size=validation_size,
        stratify=y_train,
        random_state=random_state,
    )

    def tune_threshold(pipeline, features) -> float:
        """F1-optimal threshold, found on the validation split.

        Uses a copy of `pipeline` fitted without the validation rows, so the
        threshold is never chosen on data the model was trained on.
        """
        probe = clone(pipeline).fit(X_search_train[features], y_search_train)
        return best_threshold_by_f1(y_val, probe.predict_proba(X_val[features])[:, 1])

    # ── Baseline ───────────────────────────────────────────────────────────────
    t2 = time.time()
    baseline_pipeline = build_training_pipeline(
        model_name=baseline_model_name,
        numeric_features=[c for c in scale_columns if c in all_features],
        use_smote=use_smote,
        random_state=random_state,
    )
    baseline_best_threshold = tune_threshold(baseline_pipeline, all_features)
    baseline_pipeline.fit(X_train, y_train)
    baseline_proba = baseline_pipeline.predict_proba(X_test)[:, 1]
    baseline_metrics = compute_metrics(y_test, baseline_proba, threshold=baseline_best_threshold)
    _timer("baseline training", t2)

    # ── PSO feature selection ──────────────────────────────────────────────────
    # Feature subsets proposed by PSO are scored on the validation split defined above.

    metric_weights = config["pso"]["metric_weights"]
    feature_penalty = float(config["pso"]["feature_penalty"])
    early_stopping_rounds = int(config["pso"].get("early_stopping_rounds", 0))

    def objective(mask) -> float:
        """Objective function for PSO.

        For a boolean feature mask, we:
        1. Build a pipeline with only the selected features.
        2. Fit it on `X_search_train` and evaluate on `X_val`.
        3. Compute a weighted score that combines PR-AUC, recall, and F1.
        4. Subtract a penalty proportional to the fraction of selected features
           to encourage smaller feature subsets.
        """
        selected_features = [f for f, s in zip(all_features, mask) if s]
        pipeline = build_training_pipeline(
            model_name=search_model_name,
            numeric_features=[c for c in scale_columns if c in selected_features],
            use_smote=use_smote,
            random_state=random_state,
        )
        pipeline.fit(X_search_train[selected_features], y_search_train)
        val_proba = pipeline.predict_proba(X_val[selected_features])[:, 1]
        metrics = compute_metrics(y_val, val_proba, threshold=default_threshold)

        # Weighted sum of chosen metrics; configured in `config.yaml`.
        weighted_score = (
            float(metric_weights["pr_auc"]) * metrics["pr_auc"]
            + float(metric_weights["recall"]) * metrics["recall"]
            + float(metric_weights["f1"]) * metrics["f1"]
        )
        # Penalize larger feature sets to prefer parsimonious solutions.
        penalty = feature_penalty * (len(selected_features) / len(all_features))
        return float(weighted_score - penalty)

    t3 = time.time()
    pso_selector = BinaryPSOFeatureSelector(
        n_particles=int(config["pso"]["n_particles"]),
        n_iterations=int(config["pso"]["n_iterations"]),
        inertia=float(config["pso"]["inertia"]),
        cognitive=float(config["pso"]["cognitive"]),
        social=float(config["pso"]["social"]),
        min_features=int(config["pso"]["min_features"]),
        early_stopping_rounds=early_stopping_rounds,
        random_state=random_state,
    )
    pso_result = pso_selector.optimize(n_features=len(all_features), objective_fn=objective)
    selected_features = [f for f, s in zip(all_features, pso_result.best_mask) if s]
    _timer("PSO search", t3)
    print(f"[PSO] Selected {len(selected_features)} features: {selected_features}")

    # ── Final model on PSO features ────────────────────────────────────────────
    t4 = time.time()
    final_pipeline = build_training_pipeline(
        model_name=final_model_name,
        numeric_features=[c for c in scale_columns if c in selected_features],
        use_smote=use_smote,
        random_state=random_state,
    )
    pso_best_threshold = tune_threshold(final_pipeline, selected_features)
    final_pipeline.fit(X_train[selected_features], y_train)
    pso_proba = final_pipeline.predict_proba(X_test[selected_features])[:, 1]
    pso_metrics = compute_metrics(y_test, pso_proba, threshold=pso_best_threshold)
    _timer("final model training", t4)

    # ── Ablation: the same final model on ALL features ─────────────────────────
    # Baseline (logistic regression, 30 features) vs PSO model (random forest,
    # 7 features) changes the model AND the feature set. This run changes only
    # the feature set, so it isolates what feature selection costs or gains.
    t5 = time.time()
    full_pipeline = build_training_pipeline(
        model_name=final_model_name,
        numeric_features=[c for c in scale_columns if c in all_features],
        use_smote=use_smote,
        random_state=random_state,
    )
    full_best_threshold = tune_threshold(full_pipeline, all_features)
    full_pipeline.fit(X_train, y_train)
    full_proba = full_pipeline.predict_proba(X_test)[:, 1]
    full_metrics = compute_metrics(y_test, full_proba, threshold=full_best_threshold)
    _timer("all-features ablation", t5)

    # ── Save outputs ───────────────────────────────────────────────────────────
    predictions_df = pd.DataFrame(
        {
            "y_true": y_test.to_numpy(),
            "baseline_proba": baseline_proba,
            "pso_proba": pso_proba,
            "all_features_proba": full_proba,
        }
    )
    predictions_df.to_csv(report_dir / "predictions.csv", index=False)

    pso_history_df = pd.DataFrame(
        {
            "iteration": list(range(1, len(pso_result.history) + 1)),
            "best_score": pso_result.history,
        }
    )
    pso_history_df.to_csv(report_dir / "pso_history.csv", index=False)

    importance_df = extract_feature_importance(
        trained_pipeline=final_pipeline,
        selected_features=selected_features,
        numeric_features=[c for c in scale_columns if c in selected_features],
    )
    importance_df.to_csv(report_dir / "feature_importance.csv", index=False)

    # ── Plots ──────────────────────────────────────────────────────────────────
    save_pso_history(pso_result.history, figure_dir / "pso_history.png")
    save_confusion_heatmap(
        confusion_from_threshold(y_test, baseline_proba, threshold=baseline_best_threshold),
        labels=["Normal", "Fraud"],
        path=figure_dir / "baseline_confusion_matrix.png",
        title="Baseline Confusion Matrix",
    )
    save_confusion_heatmap(
        confusion_from_threshold(y_test, pso_proba, threshold=pso_best_threshold),
        labels=["Normal", "Fraud"],
        path=figure_dir / "pso_confusion_matrix.png",
        title="PSO Model Confusion Matrix",
    )
    save_roc_comparison(
        y_test,
        {"Baseline": baseline_proba, "PSO + Final Model": pso_proba, "Final model, all features": full_proba},
        figure_dir / "roc_comparison.png",
    )
    save_pr_comparison(
        y_test,
        {"Baseline": baseline_proba, "PSO + Final Model": pso_proba, "Final model, all features": full_proba},
        figure_dir / "pr_comparison.png",
    )
    save_feature_importance_plot(importance_df, figure_dir / "feature_importance.png")

    # ── JSON reports ───────────────────────────────────────────────────────────
    metrics_summary = {
        "baseline": {**baseline_metrics, "selected_feature_count": len(all_features), "model": baseline_model_name},
        "pso_model": {
            **pso_metrics,
            "selected_feature_count": len(selected_features),
            "model": final_model_name,
            "best_fitness": pso_result.best_score,
            "converged_at": pso_result.converged_at,
        },
        "all_features_final_model": {
            **full_metrics,
            "selected_feature_count": len(all_features),
            "model": final_model_name,
        },
        "selected_features": selected_features,
        "threshold_selection": "F1-optimal on a validation split of the training data (never the test set)",
        "n_test_rows": int(len(y_test)),
        "n_test_fraud": int(y_test.sum()),
    }
    _write_json(report_dir / "metrics_summary.json", metrics_summary)

    _write_json(
        report_dir / "project_summary.json",
        {
            "project_name": config["project"]["name"],
            "data_summary": data_summary,
            "baseline_model": baseline_model_name,
            "final_model": final_model_name,
            "selected_features": selected_features,
            "selected_feature_count": len(selected_features),
            "baseline_threshold": baseline_best_threshold,
            "pso_threshold": pso_best_threshold,
            "all_features_threshold": full_best_threshold,
            "pso_converged_at": pso_result.converged_at,
        },
    )

    # ── Save models ────────────────────────────────────────────────────────────
    joblib.dump(
        {"pipeline": baseline_pipeline, "threshold": baseline_best_threshold, "features": all_features},
        model_dir / "baseline_model.joblib",
    )
    joblib.dump(
        {"pipeline": final_pipeline, "threshold": pso_best_threshold, "features": selected_features},
        model_dir / "pso_selected_model.joblib",
    )

    _timer("TOTAL", t0)
    return {
        "project_root": str(project_root),
        "selected_features": selected_features,
        "baseline_metrics": baseline_metrics,
        "pso_metrics": pso_metrics,
        "report_dir": str(report_dir),
        "figure_dir": str(figure_dir),
        "pso_converged_at": pso_result.converged_at,
    }