"""Conformal Demand Risk - Streamlit dashboard."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from conformal_demand.config import DEFAULT_ALPHAS, ITEM_COLS  # noqa: E402
from conformal_demand.data import load_wide  # noqa: E402
from conformal_demand.features import feature_matrix_for_item  # noqa: E402
from conformal_demand.conformal import SplitConformalRegressor  # noqa: E402
from conformal_demand.models import make_estimator  # noqa: E402
from conformal_demand.data import temporal_split_mask  # noqa: E402
from conformal_demand.config import TRAIN_FRAC, CALIB_FRAC, DEFAULT_LAGS, SEED  # noqa: E402
from conformal_demand.risk import InventoryPolicy, compare_stock_policies  # noqa: E402
from conformal_demand.metrics import coverage_curve, point_metrics  # noqa: E402


@st.cache_data
def load_metrics():
    for name in ("metrics_full.json", "metrics.json"):
        p = ROOT / "reports" / name
        if p.exists():
            return json.loads(p.read_text())
    return None


@st.cache_data
def load_data(source: str):
    return load_wide(source)


def find_run(metrics: dict, item: str, model: str):
    for r in metrics.get("runs", []):
        if r["item"] == item and r["model"] == model:
            return r
    return None


def live_fit(wide, item, model_name, alpha):
    X, y = feature_matrix_for_item(wide, item, lags=DEFAULT_LAGS)
    tr, ca, te = temporal_split_mask(len(X), TRAIN_FRAC, CALIB_FRAC)
    conf = SplitConformalRegressor(make_estimator(model_name, seed=SEED), alpha=alpha)
    conf.fit(X.iloc[tr], y.iloc[tr], X.iloc[ca], y.iloc[ca])
    res = conf.predict_interval(X.iloc[te])
    return X.iloc[te], y.iloc[te], conf, res


def main():
    st.set_page_config(
        page_title="Conformal Demand Risk",
        layout="wide",
        page_icon=":package:",
    )
    st.title("Conformal Demand Risk")
    st.markdown(
        "Upgrade point demand forecasts to **statistically honest prediction intervals** "
        "via **split conformal prediction**, then translate coverage into **inventory risk** "
        "(service level, fill rate, holding vs stockout cost)."
    )

    metrics = load_metrics()
    source = st.sidebar.selectbox("Data source", ["raw", "sample"], index=0)
    item = st.sidebar.selectbox("Item", list(ITEM_COLS))
    model_name = st.sidebar.selectbox("Model", ["ridge", "lightgbm"])
    alpha = st.sidebar.slider("alpha (miscoverage)", 0.05, 0.30, 0.10, 0.05)
    hold_c = st.sidebar.number_input("Holding cost / unit", 0.05, 2.0, 0.25, 0.05)
    stock_c = st.sidebar.number_input("Stockout cost / unit", 0.5, 10.0, 3.0, 0.5)

    try:
        wide = load_data(source)
    except FileNotFoundError as e:
        st.error(str(e))
        st.info("Run `python scripts/download_data.py` then `python scripts/train.py`.")
        st.stop()

    st.sidebar.markdown("### Dataset")
    st.sidebar.write(
        f"{wide.shape[0]} days - {wide.index.min().date()} to {wide.index.max().date()}"
    )

    run = None
    if metrics and abs(alpha - metrics["meta"].get("alpha", 0.1)) < 1e-9:
        run = find_run(metrics, item, model_name)

    needs_live = run is None or "y_true" not in (run or {})
    if needs_live:
        with st.spinner("Fitting split conformal model..."):
            X_te, y_te, conf, res = live_fit(wide, item, model_name, alpha)
        y_true = y_te.values
        y_pred, lower, upper = res.y_pred, res.lower, res.upper
        dates = X_te.index
        q_hat = res.q_hat
        pm = point_metrics(y_true, y_pred)
        curve = coverage_curve(
            y_true, y_pred, conf.calib_scores_, list(DEFAULT_ALPHAS) + [0.15, 0.25]
        )
        cov = float(((y_true >= lower) & (y_true <= upper)).mean())
    else:
        dates = pd.to_datetime(run["test_dates"])
        y_true = np.array(run["y_true"])
        y_pred = np.array(run["y_pred"])
        lower = np.array(run["lower"])
        upper = np.array(run["upper"])
        q_hat = run["q_hat"]
        pm = run["point_metrics"]
        curve = run["coverage_curve"]
        cov = run["interval_metrics"]["empirical_coverage"]

    risk = compare_stock_policies(
        y_true,
        y_pred,
        lower,
        upper,
        InventoryPolicy(holding_cost=hold_c, stockout_cost=stock_c),
    )

    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("MAE", f"{pm['mae']:.3f}")
    c2.metric("RMSE", f"{pm['rmse']:.3f}")
    c3.metric("R2", f"{pm['r2']:.3f}")
    c4.metric("Empirical coverage", f"{cov:.1%}", delta=f"target {(1-alpha):.0%}")
    c5.metric("q_hat (half-width)", f"{q_hat:.3f}")

    tab1, tab2, tab3, tab4 = st.tabs(
        ["Forecast + intervals", "Coverage calibration", "Inventory risk", "Method"]
    )

    with tab1:
        df_plot = pd.DataFrame(
            {
                "date": dates,
                "actual": y_true,
                "forecast": y_pred,
                "lower": lower,
                "upper": upper,
            }
        ).set_index("date")
        st.line_chart(df_plot[["actual", "forecast"]])
        st.caption("Shaded conformal band (lower / upper)")
        band = df_plot[["lower", "upper", "actual"]].copy()
        st.line_chart(band)
        st.dataframe(
            df_plot.tail(30).style.format("{:.2f}"),
            use_container_width=True,
        )

    with tab2:
        cal = pd.DataFrame(curve)
        st.subheader("Coverage vs nominal")
        st.dataframe(
            cal[
                [
                    "alpha",
                    "nominal_coverage",
                    "empirical_coverage",
                    "coverage_gap",
                    "mean_width",
                    "q_hat",
                ]
            ].style.format(
                {
                    "alpha": "{:.2f}",
                    "nominal_coverage": "{:.1%}",
                    "empirical_coverage": "{:.1%}",
                    "coverage_gap": "{:+.3f}",
                    "mean_width": "{:.3f}",
                    "q_hat": "{:.3f}",
                }
            ),
            use_container_width=True,
        )
        st.scatter_chart(
            cal.rename(
                columns={
                    "nominal_coverage": "nominal",
                    "empirical_coverage": "empirical",
                }
            ),
            x="nominal",
            y="empirical",
            size=None,
        )
        st.markdown(
            "If conformal calibration is honest, points hug the diagonal "
            "(empirical ~ nominal). Positive gaps = slightly conservative."
        )

    with tab3:
        st.subheader("Order-up-to policies from the interval")
        risk_df = pd.DataFrame(risk).T
        st.dataframe(
            risk_df[
                [
                    "service_level",
                    "fill_rate",
                    "avg_stock",
                    "avg_leftover",
                    "avg_stockout_units",
                    "total_holding_cost",
                    "total_stockout_cost",
                    "net_utility",
                ]
            ].style.format("{:.3f}"),
            use_container_width=True,
        )
        st.bar_chart(risk_df[["service_level", "fill_rate"]])
        st.markdown(
            """
**Interpretation**
- **conformal_upper** - stock to the upper PI: higher service level, more holding cost.
- **point_forecast** - naive stock-to-mean: more stockouts when demand is noisy.
- **conformal_lower** - understocks by design (risk illustration only).
            """
        )

    with tab4:
        st.markdown(
            f"""
### Split conformal regression

1. Fit a point model (**{model_name}**) on the **train** fold (60%).
2. Compute absolute residual scores on a held-out **calibration** fold (20%).
3. Set q_hat to the finite-sample (1-alpha) quantile of those scores.
4. Prediction interval: [y_hat - q_hat, y_hat + q_hat] (clipped at 0 for demand).

Under exchangeability of calibration and test residuals, coverage is at least
1-alpha in finite samples - **model-agnostic** and distribution-free.

**Features:** lags {list(DEFAULT_LAGS)}, rolling mean/std (7/14), day-of-week / month / weekend.
**Data:** skforecast `simulated_items_sales` (3 items, daily).
            """
        )
        if metrics:
            st.markdown("### Saved training summary")
            st.dataframe(pd.DataFrame(metrics["summary"]), use_container_width=True)

    st.caption("Conformal Demand Risk - Rohith Vairavel")


if __name__ == "__main__":
    main()
