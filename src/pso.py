from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, List

import numpy as np


@dataclass
class PSOResult:
    best_mask: np.ndarray
    best_score: float
    history: List[float]
    converged_at: int | None = field(default=None)


class BinaryPSOFeatureSelector:
    def __init__(
        self,
        n_particles: int,
        n_iterations: int,
        inertia: float,
        cognitive: float,
        social: float,
        min_features: int,
        early_stopping_rounds: int = 0,
        random_state: int = 42,
    ) -> None:
        self.n_particles = n_particles
        self.n_iterations = n_iterations
        self.inertia = inertia
        self.cognitive = cognitive
        self.social = social
        self.min_features = min_features
        self.early_stopping_rounds = early_stopping_rounds
        self.rng = np.random.default_rng(random_state)

    @staticmethod
    def _sigmoid(x: np.ndarray) -> np.ndarray:
        return 1.0 / (1.0 + np.exp(-np.clip(x, -10, 10)))

    def _repair_mask(self, mask: np.ndarray) -> np.ndarray:
        repaired = mask.copy().astype(bool)
        if repaired.sum() >= self.min_features:
            return repaired
        zero_indices = np.where(~repaired)[0]
        if len(zero_indices) == 0:
            return repaired
        pick_count = min(self.min_features - int(repaired.sum()), len(zero_indices))
        chosen = self.rng.choice(zero_indices, size=pick_count, replace=False)
        repaired[chosen] = True
        return repaired

    def optimize(self, n_features: int, objective_fn: Callable[[np.ndarray], float]) -> PSOResult:
        positions = self.rng.uniform(0.0, 1.0, size=(self.n_particles, n_features))
        velocities = self.rng.normal(0.0, 0.25, size=(self.n_particles, n_features))

        personal_best_positions = positions.copy()
        personal_best_scores = np.full(self.n_particles, -np.inf)
        global_best_position = positions[0].copy()
        global_best_score = -np.inf

        history: List[float] = []
        no_improve_count = 0
        converged_at: int | None = None

        for iteration in range(self.n_iterations):
            for idx in range(self.n_particles):
                mask = self._repair_mask(positions[idx] > 0.5)
                score = objective_fn(mask)

                if score > personal_best_scores[idx]:
                    personal_best_scores[idx] = score
                    personal_best_positions[idx] = positions[idx].copy()

                if score > global_best_score:
                    global_best_score = score
                    global_best_position = positions[idx].copy()

            prev_best = history[-1] if history else -np.inf
            history.append(float(global_best_score))

            if self.early_stopping_rounds > 0:
                if global_best_score > prev_best + 1e-6:
                    no_improve_count = 0
                else:
                    no_improve_count += 1

                if no_improve_count >= self.early_stopping_rounds:
                    converged_at = iteration + 1
                    print(
                        f"[PSO] Early stopping at iteration {converged_at} "
                        f"(no improvement for {self.early_stopping_rounds} rounds)"
                    )
                    break

            r1 = self.rng.random(size=(self.n_particles, n_features))
            r2 = self.rng.random(size=(self.n_particles, n_features))
            velocities = (
                self.inertia * velocities
                + self.cognitive * r1 * (personal_best_positions - positions)
                + self.social * r2 * (global_best_position - positions)
            )
            probabilities = self._sigmoid(velocities)
            positions = (self.rng.random(size=(self.n_particles, n_features)) < probabilities).astype(float)

        best_mask = self._repair_mask(global_best_position > 0.5)
        return PSOResult(
            best_mask=best_mask,
            best_score=float(global_best_score),
            history=history,
            converged_at=converged_at,
        )
