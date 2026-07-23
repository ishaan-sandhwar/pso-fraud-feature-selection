# PSO Feature Selection — Credit Card Fraud Detection

Binary Particle Swarm Optimization (PSO) for feature selection on the [Kaggle Credit Card Fraud dataset](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud).

## Highlights
- **Binary PSO** with sigmoid transfer function, repair mask, and early stopping
- **Stratified sampling** — all 492 fraud cases always included, configurable dataset size
- **Parquet caching** — CSV read once, ~5-10× faster on subsequent runs
- Baseline (Logistic Regression) vs PSO + Random Forest comparison
- Metrics: ROC-AUC, PR-AUC, Precision, Recall, F1, MCC, Balanced Accuracy
- Streamlit dashboard: overview, PSO analysis, live threshold tuning, live prediction

## Project Structure
```
pso-credit-card-fraud/
├── app/
│   └── dashboard.py          # Streamlit dashboard (4 tabs)
├── configs/
│   └── config.yaml           # All hyperparameters here
├── data/
│   ├── processed/            # Parquet cache auto-saved here
│   └── raw/
│       └── creditcard.csv    # Place dataset here
├── models/                   # Saved .joblib model bundles
├── outputs/
│   ├── figures/              # All plots
│   └── reports/              # JSON reports + CSVs
├── src/
│   ├── cache.py              # Parquet cache + stratified sampling
│   ├── config.py             # YAML loader
│   ├── data.py               # Dataset loading + validation
│   ├── evaluate.py           # Metrics + threshold search
│   ├── inference.py          # Single-transaction predictor
│   ├── modeling.py           # Pipeline builder
│   ├── pso.py                # Binary PSO (with early stopping)
│   ├── train_pipeline.py     # Full training orchestrator
│   └── visualize.py          # All plots
├── tests/
│   └── test_pso.py           # PSO unit tests (7 tests)
├── requirements.txt
└── run_training.py
```

## Setup
```bash
pip install -r requirements.txt
```

## Dataset
Place `creditcard.csv` at `data/raw/creditcard.csv`.  
Download: https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud

The raw CSV (144 MB) is **not** committed — it exceeds GitHub's 100 MB per-file limit.
The parquet cache in `data/processed/` is also excluded; it regenerates on the first run.

## Pre-computed Artifacts
Trained models (`models/`) and results (`outputs/`) **are** committed, so you can launch
the dashboard and inspect metrics without re-training or downloading the dataset first:

```bash
streamlit run app/dashboard.py
```

## Run Training
```bash
python run_training.py
```

**Speed tip:** Set `sample_size: 50000` in `configs/config.yaml` for fast runs during development.  
First run saves a parquet cache — subsequent runs are 5-10× faster.

## Run Dashboard
```bash
streamlit run app/dashboard.py
```

## Run Tests
```bash
pytest tests/ -v
```

## Key Config Options (`configs/config.yaml`)
| Key | Default | Description |
|---|---|---|
| `data.sample_size` | `50000` | Stratified sample size (null = full dataset) |
| `data.parquet_cache_path` | `data/processed/creditcard.parquet` | Cache location |
| `pso.n_particles` | `30` | Number of PSO particles |
| `pso.n_iterations` | `50` | Max PSO iterations |
| `pso.early_stopping_rounds` | `10` | Stop if no improvement for N iterations |

## Notes
- Logistic Regression is used during PSO search (fast), Random Forest for final model (accurate)
- SMOTE oversampling applied on training set only (no leakage)
- Threshold is auto-tuned by maximizing F1 on test set
- Dashboard Tab 4 allows live fraud prediction on custom inputs
