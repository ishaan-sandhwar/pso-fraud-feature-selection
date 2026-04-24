from __future__ import annotations

from pathlib import Path
from typing import Dict, List

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.metrics import PrecisionRecallDisplay, RocCurveDisplay

sns.set_theme(style="whitegrid")


def _ensure_parent(path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def save_class_distribution(df: pd.DataFrame, target_column: str, path: str | Path) -> None:
    path = _ensure_parent(path)
    plt.figure(figsize=(7, 4))
    ax = sns.countplot(x=target_column, data=df, palette=["#1f77b4", "#d62728"])
    ax.set_title("Class Distribution")
    ax.set_xlabel("Class")
    ax.set_ylabel("Count")
    plt.tight_layout()
    plt.savefig(path, dpi=160)
    plt.close()


def save_pso_history(history: List[float], path: str | Path) -> None:
    path = _ensure_parent(path)
    plt.figure(figsize=(8, 4))
    plt.plot(range(1, len(history) + 1), history, marker="o", color="#2ca02c")
    plt.title("PSO Best Fitness per Iteration")
    plt.xlabel("Iteration")
    plt.ylabel("Best Fitness")
    plt.tight_layout()
    plt.savefig(path, dpi=160)
    plt.close()


def save_confusion_heatmap(cm, labels: List[str], path: str | Path, title: str) -> None:
    path = _ensure_parent(path)
    plt.figure(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=labels, yticklabels=labels)
    plt.title(title)
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.tight_layout()
    plt.savefig(path, dpi=160)
    plt.close()


def save_roc_comparison(y_true, proba_map: Dict[str, pd.Series], path: str | Path) -> None:
    path = _ensure_parent(path)
    fig, ax = plt.subplots(figsize=(7, 5))
    for label, probs in proba_map.items():
        RocCurveDisplay.from_predictions(y_true, probs, name=label, ax=ax)
    ax.set_title("ROC Curve Comparison")
    plt.tight_layout()
    plt.savefig(path, dpi=160)
    plt.close(fig)


def save_pr_comparison(y_true, proba_map: Dict[str, pd.Series], path: str | Path) -> None:
    path = _ensure_parent(path)
    fig, ax = plt.subplots(figsize=(7, 5))
    for label, probs in proba_map.items():
        PrecisionRecallDisplay.from_predictions(y_true, probs, name=label, ax=ax)
    ax.set_title("Precision-Recall Curve Comparison")
    plt.tight_layout()
    plt.savefig(path, dpi=160)
    plt.close(fig)


def save_feature_importance_plot(importance_df: pd.DataFrame, path: str | Path, top_n: int = 15) -> None:
    path = _ensure_parent(path)
    top_df = importance_df.head(top_n).sort_values("importance", ascending=True)
    plt.figure(figsize=(8, 6))
    plt.barh(top_df["feature"], top_df["importance"], color="#9467bd")
    plt.title(f"Top {min(top_n, len(top_df))} Feature Importances")
    plt.xlabel("Importance")
    plt.ylabel("Feature")
    plt.tight_layout()
    plt.savefig(path, dpi=160)
    plt.close()