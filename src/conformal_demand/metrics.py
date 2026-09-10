"""Point-forecast and interval evaluation metrics."""

from __future__ import annotations

import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


def point_metrics(y_true, y_pred) -> dict[str, float]:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    mae = float(mean_absolute_error(y_true, y_pred))
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    mape = float(np.mean(np.abs((y_true - y_pred) / np.clip(np.abs(y_true), 1e-6, None))) * 100)
    return {
        "mae": mae,
        "rmse": rmse,
        "mape_pct": mape,
        "r2": float(r2_score(y_true, y_pred)),
    }


def interval_metrics(y_true, lower, upper, nominal: float) -> dict[str, float]:
    y_true = np.asarray(y_true, dtype=float)
    lower = np.asarray(lower, dtype=float)
    upper = np.asarray(upper, dtype=float)
    covered = (y_true >= lower) & (y_true <= upper)
    widths = upper - lower
    return {
        "empirical_coverage": float(covered.mean()),
        "nominal_coverage": float(nominal),
        "coverage_gap": float(covered.mean() - nominal),
        "mean_width": float(widths.mean()),
        "median_width": float(np.median(widths)),
        "p90_width": float(np.percentile(widths, 90)),
    }


def coverage_curve(
    y_true,
    y_pred,
    calib_scores,
    alphas: list[float],
) -> list[dict[str, float]]:
    """Empirical coverage vs nominal for a grid of alphas (same calib scores)."""
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    scores = np.asarray(calib_scores, dtype=float)
    n = len(scores)
    out = []
    for alpha in alphas:
        level = min(1.0, np.ceil((n + 1) * (1 - alpha)) / n)
        q = float(np.quantile(scores, level, method="higher"))
        lower = np.maximum(y_pred - q, 0.0)
        upper = y_pred + q
        m = interval_metrics(y_true, lower, upper, 1 - alpha)
        m["alpha"] = float(alpha)
        m["q_hat"] = q
        out.append(m)
    return out
