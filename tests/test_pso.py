import numpy as np
import pytest

from src.pso import BinaryPSOFeatureSelector


def make_selector(**kwargs):
    defaults = dict(
        n_particles=5,
        n_iterations=10,
        inertia=0.7,
        cognitive=1.4,
        social=1.4,
        min_features=3,
        random_state=42,
    )
    defaults.update(kwargs)
    return BinaryPSOFeatureSelector(**defaults)


def test_min_features_respected():
    sel = make_selector(min_features=4)
    result = sel.optimize(n_features=10, objective_fn=lambda m: float(m.sum()))
    assert result.best_mask.sum() >= 4


def test_history_length_equals_iterations():
    sel = make_selector(n_iterations=8)
    result = sel.optimize(n_features=10, objective_fn=lambda m: 0.5)
    assert len(result.history) == 8


def test_history_is_monotonically_non_decreasing():
    sel = make_selector(n_iterations=10)
    result = sel.optimize(n_features=10, objective_fn=lambda m: float(m.sum()) / 10)
    for i in range(1, len(result.history)):
        assert result.history[i] >= result.history[i - 1] - 1e-9


def test_early_stopping_fires():
    # Constant objective → should stop after early_stopping_rounds
    sel = make_selector(n_iterations=50, early_stopping_rounds=5)
    result = sel.optimize(n_features=10, objective_fn=lambda m: 0.5)
    assert result.converged_at is not None
    assert result.converged_at <= 10   # should stop well before 50


def test_no_early_stopping_when_disabled():
    sel = make_selector(n_iterations=10, early_stopping_rounds=0)
    result = sel.optimize(n_features=10, objective_fn=lambda m: float(m.sum()))
    assert result.converged_at is None
    assert len(result.history) == 10


def test_best_mask_is_boolean():
    sel = make_selector()
    result = sel.optimize(n_features=8, objective_fn=lambda m: 1.0)
    assert result.best_mask.dtype == bool


def test_repair_mask_adds_features_when_too_few():
    sel = make_selector(min_features=5)
    # Force a mask with only 1 feature
    sparse_mask = np.zeros(10, dtype=float)
    sparse_mask[0] = 1.0
    repaired = sel._repair_mask(sparse_mask > 0.5)
    assert repaired.sum() >= 5
