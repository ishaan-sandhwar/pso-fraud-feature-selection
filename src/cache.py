from __future__ import annotations

from pathlib import Path

import pandas as pd


def load_with_parquet_cache(
    csv_path: str | Path,
    parquet_path: str | Path,
    sample_size: int | None = None,
    random_state: int = 42,
    target_column: str = "Class",
) -> pd.DataFrame:
    """Load dataset using parquet cache for speed.

    First run: reads CSV → saves parquet cache → returns df.
    Later runs: reads parquet directly (5-10x faster than CSV).
    If sample_size is set, stratified sample is returned (all fraud kept).
    """
    csv_path = Path(csv_path)
    parquet_path = Path(parquet_path)

    if parquet_path.exists():
        print(f"[cache] Loading from parquet cache: {parquet_path}")
        df = pd.read_parquet(parquet_path)
    else:
        print(f"[cache] Parquet cache not found. Reading CSV: {csv_path}")
        # float32 for all V-columns to reduce memory ~50%
        col_sample = pd.read_csv(csv_path, nrows=1)
        dtype_map = {
            col: "float32"
            for col in col_sample.columns
            if col not in {target_column}
        }
        df = pd.read_csv(csv_path, dtype=dtype_map)
        parquet_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(parquet_path, index=False)
        print(f"[cache] Parquet cache saved: {parquet_path}")

    if sample_size and sample_size < len(df):
        fraud = df[df[target_column] == 1]
        n_normal = sample_size - len(fraud)
        normal = df[df[target_column] == 0].sample(n=n_normal, random_state=random_state)
        df = (
            pd.concat([fraud, normal])
            .sample(frac=1, random_state=random_state)
            .reset_index(drop=True)
        )
        print(f"[cache] Stratified sample: {len(df):,} rows ({len(fraud)} fraud kept)")

    return df
