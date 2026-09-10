"""Lag and calendar feature engineering for demand series."""

from __future__ import annotations

from typing import Iterable

import numpy as np
import pandas as pd

from conformal_demand.config import DEFAULT_LAGS


def add_calendar_features(dates: pd.DatetimeIndex | pd.Series) -> pd.DataFrame:
    d = pd.to_datetime(dates)
    return pd.DataFrame(
        {
            "dow": d.dayofweek.astype(int),
            "dom": d.day.astype(int),
            "month": d.month.astype(int),
            "weekofyear": d.isocalendar().week.astype(int).to_numpy(),
            "is_weekend": (d.dayofweek >= 5).astype(int),
        },
        index=d if isinstance(dates, pd.DatetimeIndex) else None,
    )


def make_supervised(
    series: pd.Series,
    lags: Iterable[int] = DEFAULT_LAGS,
) -> tuple[pd.DataFrame, pd.Series]:
    """
    Build supervised matrix from a univariate demand series.

    Features: lag_k, rolling mean/std (7/14), calendar.
    Target: next-day sales (same timestamp as features after shift).
    """
    s = series.astype(float).copy()
    s.name = "sales"
    frame = pd.DataFrame({"sales": s})
    lag_list = list(lags)
    for lag in lag_list:
        frame[f"lag_{lag}"] = s.shift(lag)

    frame["roll_mean_7"] = s.shift(1).rolling(7).mean()
    frame["roll_std_7"] = s.shift(1).rolling(7).std()
    frame["roll_mean_14"] = s.shift(1).rolling(14).mean()
    frame["roll_std_14"] = s.shift(1).rolling(14).std()

    cal = add_calendar_features(frame.index)
    cal.index = frame.index
    frame = pd.concat([frame, cal], axis=1)

    frame = frame.dropna()
    y = frame.pop("sales")
    X = frame
    return X, y


def feature_matrix_for_item(
    wide: pd.DataFrame,
    item: str,
    lags: Iterable[int] = DEFAULT_LAGS,
) -> tuple[pd.DataFrame, pd.Series]:
    if item not in wide.columns:
        raise KeyError(item)
    return make_supervised(wide[item], lags=lags)
