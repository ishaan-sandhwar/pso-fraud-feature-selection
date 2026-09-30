"""Bootstrap confidence intervals for the test-set results.

The test set holds only ~100 fraud cases, so single-number metrics are noisy.
This module resamples the test rows to show how much, and whether the gaps
between models are larger than that noise.

It reads only what training wrote to ``outputs/reports`` (``predictions.csv`` and
``metrics_summary.json``), so it runs without the dataset:

    python -m src.bootstrap
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, f1_score, roc_auc_score

# model key in metrics_summary.json -> probability column in predictions.csv
MODELS = {
    "baseline": "baseline_proba",
    "all_features_final_model": "all_features_proba",
    "pso_model": "pso_proba",
}
# (a, b) -> paired difference "a - b"
COMPARISONS = [
    ("pso_model", "baseline"),
    ("all_features_final_model", "pso_model"),
]
METRICS = ("roc_auc", "pr_auc", "f1")


def _metrics(y: np.ndarray, proba: np.ndarray, threshold: float) -> tuple:
    return (
        roc_auc_score(y, proba),
        average_precision_score(y, proba),
        f1_score(y, (proba >= threshold).astype(int), zero_division=0),
    )


def run(report_dir: str | Path = "outputs/reports", n_boot: int = 1000, seed: int = 0) -> Dict:
    report_dir = Path(report_dir)
    predictions = pd.read_csv(report_dir / "predictions.csv")
    summary = json.loads((report_dir / "metrics_summary.json").read_text(encoding="utf-8"))

    y = predictions["y_true"].to_numpy()
    n = len(y)
    rng = np.random.default_rng(seed)
    # The same resampled rows are used for every model, so differences are paired.
    resamples = [rng.integers(0, n, n) for _ in range(n_boot)]

    proba = {name: predictions[col].to_numpy() for name, col in MODELS.items()}
    threshold = {name: float(summary[name]["threshold"]) for name in MODELS}

    boot: Dict[str, np.ndarray] = {}
    for name in MODELS:
        boot[name] = np.array(
            [
                _metrics(y[idx], proba[name][idx], threshold[name]) if y[idx].sum() else (np.nan,) * 3
                for idx in resamples
            ]
        )

    def interval(values: np.ndarray) -> list:
        return [float(v) for v in np.nanpercentile(values, [2.5, 97.5])]

    result: Dict = {
        "n_test_rows": int(n),
        "n_test_fraud": int(y.sum()),
        "n_boot": n_boot,
        "models": {},
        "paired_differences": {},
    }
    for name in MODELS:
        point = _metrics(y, proba[name], threshold[name])
        result["models"][name] = {
            metric: {"estimate": float(point[j]), "ci95": interval(boot[name][:, j])}
            for j, metric in enumerate(METRICS)
        }
    for a, b in COMPARISONS:
        entry = {}
        for j, metric in enumerate(METRICS):
            diff = boot[a][:, j] - boot[b][:, j]
            low, high = interval(diff)
            entry[metric] = {
                "mean_difference": float(np.nanmean(diff)),
                "ci95": [low, high],
                "ci_includes_zero": bool(low <= 0.0 <= high),
            }
        result["paired_differences"][f"{a} - {b}"] = entry
    return result


def _print(result: Dict) -> None:
    print(f"Test set: {result['n_test_rows']} rows, {result['n_test_fraud']} fraud | {result['n_boot']} bootstrap resamples\n")
    for name, metrics in result["models"].items():
        cells = "  ".join(
            f"{m} {v['estimate']:.4f} [{v['ci95'][0]:.4f}, {v['ci95'][1]:.4f}]" for m, v in metrics.items()
        )
        print(f"{name:26} {cells}")
    print("\nPaired differences (95% CI; 'includes 0' means not distinguishable from noise):")
    for label, metrics in result["paired_differences"].items():
        for m, v in metrics.items():
            verdict = "includes 0" if v["ci_includes_zero"] else "excludes 0"
            print(
                f"  {label:44} {m:8} {v['mean_difference']:+.4f} "
                f"[{v['ci95'][0]:+.4f}, {v['ci95'][1]:+.4f}]  {verdict}"
            )


if __name__ == "__main__":
    root = Path(__file__).resolve().parents[1]
    output = run(root / "outputs" / "reports")
    (root / "outputs" / "reports" / "bootstrap_ci.json").write_text(json.dumps(output, indent=2), encoding="utf-8")
    _print(output)
    print("\nSaved outputs/reports/bootstrap_ci.json")
