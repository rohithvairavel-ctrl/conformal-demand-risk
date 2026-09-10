"""Project paths and defaults."""

from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DATA_RAW = ROOT / "data" / "raw"
DATA_SAMPLE = ROOT / "data" / "sample"
MODELS_DIR = ROOT / "models"
REPORTS_DIR = ROOT / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"

DATA_URL = (
    "https://raw.githubusercontent.com/skforecast/skforecast-datasets/"
    "main/data/simulated_items_sales.csv"
)
RAW_FILENAME = "simulated_items_sales.csv"
SAMPLE_FILENAME = "simulated_items_sales_sample.csv"

ITEM_COLS = ("item_1", "item_2", "item_3")
DEFAULT_LAGS = (1, 2, 3, 7, 14, 21, 28)
DEFAULT_ALPHAS = (0.05, 0.10, 0.20)  # miscoverage → 95%, 90%, 80% intervals
TRAIN_FRAC = 0.60
CALIB_FRAC = 0.20
# remainder = test
SEED = 42
