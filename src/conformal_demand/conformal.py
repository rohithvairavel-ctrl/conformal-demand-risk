"""Split conformal prediction for regression (absolute residual scores)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from sklearn.base import BaseEstimator, clone


@dataclass
class ConformalResult:
    y_pred: np.ndarray
    lower: np.ndarray
    upper: np.ndarray
    q_hat: float
    alpha: float
    coverage_target: float


class SplitConformalRegressor:
    """
    Inductive (split) conformal intervals around a point regressor.

    1. Fit model on training fold.
    2. Compute absolute residual scores on calibration fold.
    3. Take (1-alpha)(1+1/n) quantile as half-width q_hat.
    4. Interval = [y_hat - q_hat, y_hat + q_hat] (optionally clipped >= 0).
    """

    def __init__(
        self,
        estimator: BaseEstimator,
        alpha: float = 0.10,
        clip_nonneg: bool = True,
    ):
        if not 0 < alpha < 1:
            raise ValueError("alpha must be in (0, 1)")
        self.estimator = estimator
        self.alpha = alpha
        self.clip_nonneg = clip_nonneg
        self.model_: BaseEstimator | None = None
        self.q_hat_: float | None = None
        self.calib_scores_: np.ndarray | None = None

    @property
    def coverage_target(self) -> float:
        return 1.0 - self.alpha

    def fit(
        self,
        X_train,
        y_train,
        X_calib,
        y_calib,
    ) -> "SplitConformalRegressor":
        self.model_ = clone(self.estimator)
        self.model_.fit(X_train, y_train)
        resid = np.abs(np.asarray(y_calib) - self.model_.predict(X_calib))
        self.calib_scores_ = resid
        n = len(resid)
        level = min(1.0, np.ceil((n + 1) * (1 - self.alpha)) / n)
        self.q_hat_ = float(np.quantile(resid, level, method="higher"))
        return self

    def predict_interval(self, X) -> ConformalResult:
        if self.model_ is None or self.q_hat_ is None:
            raise RuntimeError("Call fit() first")
        y_pred = np.asarray(self.model_.predict(X), dtype=float)
        lower = y_pred - self.q_hat_
        upper = y_pred + self.q_hat_
        if self.clip_nonneg:
            lower = np.maximum(lower, 0.0)
        return ConformalResult(
            y_pred=y_pred,
            lower=lower,
            upper=upper,
            q_hat=self.q_hat_,
            alpha=self.alpha,
            coverage_target=self.coverage_target,
        )

    def score_coverage(self, X, y) -> dict[str, Any]:
        res = self.predict_interval(X)
        y = np.asarray(y, dtype=float)
        covered = (y >= res.lower) & (y <= res.upper)
        widths = res.upper - res.lower
        return {
            "empirical_coverage": float(covered.mean()),
            "nominal_coverage": res.coverage_target,
            "coverage_gap": float(covered.mean() - res.coverage_target),
            "mean_width": float(widths.mean()),
            "median_width": float(np.median(widths)),
            "q_hat": res.q_hat,
            "alpha": res.alpha,
            "n": int(len(y)),
        }
