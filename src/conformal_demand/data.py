"""Load and reshape skforecast simulated_items_sales."""

from __future__ import annotations

from pathlib import Path
from typing import Literal

import pandas as pd

from conformal_demand.config import (
    DATA_RAW,
    DATA_SAMPLE,
    ITEM_COLS,
    RAW_FILENAME,
    SAMPLE_FILENAME,
)


def resolve_data_path(source: Literal["raw", "sample"] = "raw") -> Path:
    if source == "sample":
        path = DATA_SAMPLE / SAMPLE_FILENAME
        if path.exists():
            return path
    path = DATA_RAW / RAW_FILENAME
    if not path.exists():
        raise FileNotFoundError(
            f"Missing {path}. Run: python scripts/download_data.py"
        )
    return path


def load_wide(source: Literal["raw", "sample"] = "raw") -> pd.DataFrame:
    """Wide panel: date index x item_1/2/3."""
    path = resolve_data_path(source)
    df = pd.read_csv(path, parse_dates=["date"])
    df = df.set_index("date").sort_index()
    for c in ITEM_COLS:
        if c not in df.columns:
            raise ValueError(f"Expected column {c} in {path}")
    return df[list(ITEM_COLS)].astype(float)


def load_long(source: Literal["raw", "sample"] = "raw") -> pd.DataFrame:
    """Long format: date, item, sales."""
    wide = load_wide(source)
    long = (
        wide.reset_index()
        .melt(id_vars="date", var_name="item", value_name="sales")
        .sort_values(["item", "date"])
        .reset_index(drop=True)
    )
    return long


def temporal_split_mask(
    n: int,
    train_frac: float = 0.60,
    calib_frac: float = 0.20,
) -> tuple[slice, slice, slice]:
    """Chronological train / calibration / test index slices."""
    n_train = int(n * train_frac)
    n_calib = int(n * calib_frac)
    n_test = n - n_train - n_calib
    if min(n_train, n_calib, n_test) < 10:
        raise ValueError(f"Split too small for n={n}")
    return (
        slice(0, n_train),
        slice(n_train, n_train + n_calib),
        slice(n_train + n_calib, n),
    )
