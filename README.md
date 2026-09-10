# Conformal Demand Risk

**Upgrade point demand forecasts to statistically honest prediction intervals**, then turn interval coverage into **inventory risk** (service level, fill rate, holding vs stockout cost).

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

Point forecasts alone hide uncertainty. Operations teams need intervals they can trust — not heuristic ±kσ bands. This project wraps **Ridge** and **LightGBM** demand models with **split (inductive) conformal prediction** so empirical coverage tracks the nominal level (e.g. 90%) in finite samples, without assuming Gaussian errors.

## Why this matters

| Pain | What we do |
|------|------------|
| Stockout risk next week? | Map conformal upper bounds to order-up-to policies (service level / fill rate) |
| Residual-σ intervals miscalibrated | Distribution-free conformal under exchangeability |
| Black-box uncertainty | Model-agnostic wrapper for linear and tree models |
| Demo-only notebooks | Scripts, real metrics JSON, Streamlit app |

## Method (split conformal regression)

1. Train a point regressor on the first 60% of each series (chronological).
2. Absolute residual scores `|y - ŷ|` on a calibration fold (next 20%).
3. Finite-sample corrected `(1-α)` quantile `q̂`.
4. Interval `[ŷ-q̂, ŷ+q̂]` (clipped at 0 for demand).
5. Evaluate empirical coverage on the test fold vs nominal `1-α`.

Features: lags `{1,2,3,7,14,21,28}`, rolling mean/std (7/14), calendar (DOW, DOM, month, week-of-year, weekend).

**Inventory risk:** treat point / lower / mid / upper as order-up-to levels.

## Dataset

[skforecast `simulated_items_sales`](https://github.com/skforecast/skforecast-datasets) — daily sales for 3 items (~2012-01-01 → 2015-01-01).

```bash
python scripts/download_data.py
python scripts/download_data.py --force
```

## Quickstart

```bash
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
python scripts/download_data.py
python scripts/train.py
streamlit run app/streamlit_app.py
```

## Results (α = 0.10 → 90% nominal; 60/20/20 split)

| Item | Model | MAE | RMSE | R² | Empirical cov. | Nominal | Mean width | Service (upper) | Service (point) |
|------|-------|-----|------|----|----------------|---------|------------|-----------------|-----------------|
| item_1 | ridge | 0.678 | 1.087 | 0.781 | 98.1% | 90% | 5.32 | 99.5% | 53.0% |
| item_1 | lightgbm | 0.735 | 1.148 | 0.756 | 98.1% | 90% | 5.40 | 99.5% | 57.7% |
| item_2 | ridge | 1.949 | 2.662 | 0.569 | 95.8% | 90% | 11.91 | 96.7% | 55.3% |
| item_2 | lightgbm | 2.009 | 2.678 | 0.564 | 94.4% | 90% | 11.55 | 97.2% | 53.5% |
| item_3 | ridge | 2.871 | 3.860 | 0.337 | 93.0% | 90% | 14.54 | 95.3% | 50.2% |
| item_3 | lightgbm | 3.036 | 3.925 | 0.315 | 94.0% | 90% | 15.16 | 95.8% | 49.8% |

Source: `reports/summary.csv`. **Takeaway:** stocking to the conformal upper bound lifts service level vs the point forecast — a transparent risk knob tied to α.

## Project layout

```
app/streamlit_app.py
data/{raw,sample}/
models/
notebooks/conformal_demand_demo.ipynb
reports/{metrics.json,summary.csv,figures/}
scripts/{download_data.py,train.py}
src/conformal_demand/{conformal,features,metrics,models,risk,plots}.py
```

## Resume blurb

> Built a demand-forecasting pipeline that wraps Ridge/LightGBM with **split conformal prediction** to produce finite-sample valid intervals; validated coverage vs nominal on skforecast simulated retail series and translated upper bounds into inventory service-level / fill-rate trade-offs in a Streamlit app.

## License

MIT · Data © skforecast simulated dataset (open for research/demo use).
