from __future__ import annotations

from pathlib import Path
from typing import Dict, Tuple

import pandas as pd
from sklearn.model_selection import train_test_split


REQUIRED_COLUMNS = {"Time", "Amount", "Class"}


def load_dataset(csv_path: str | Path) -> pd.DataFrame:
    """Legacy direct CSV loader — use src.cache.load_with_parquet_cache for speed."""
    csv_path = Path(csv_path)
    if not csv_path.exists():
        raise FileNotFoundError(
            f"Dataset not found at {csv_path}. Place creditcard.csv inside data/raw/."
        )
    df = pd.read_csv(csv_path)
    _validate_columns(df)
    return df


def _validate_columns(df: pd.DataFrame) -> None:
    missing_cols = REQUIRED_COLUMNS - set(df.columns)
    if missing_cols:
        raise ValueError(
            "Dataset does not look like the Kaggle credit card fraud dataset. "
            f"Missing columns: {sorted(missing_cols)}"
        )


def train_test_data(
    df: pd.DataFrame,
    target_column: str,
    test_size: float,
    random_state: int,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    X = df.drop(columns=[target_column])
    y = df[target_column]
    return train_test_split(
        X,
        y,
        test_size=test_size,
        stratify=y,
        random_state=random_state,
    )


def build_data_summary(df: pd.DataFrame, target_column: str) -> Dict[str, float | int]:
    fraud_count = int(df[target_column].sum())
    total_rows = int(len(df))
    normal_count = total_rows - fraud_count
    fraud_ratio = float(fraud_count / total_rows) if total_rows else 0.0

    return {
        "rows": total_rows,
        "columns": int(df.shape[1]),
        "fraud_cases": fraud_count,
        "normal_cases": normal_count,
        "fraud_ratio": fraud_ratio,
    }
