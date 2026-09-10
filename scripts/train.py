#!/usr/bin/env python3
"""Train Ridge/LightGBM + split conformal intervals; save metrics & figures."""

from __future__ import annotations

import argparse
import base64
import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from conformal_demand.config import (  # noqa: E402
    CALIB_FRAC,
    DEFAULT_ALPHAS,
    DEFAULT_LAGS,
    FIGURES_DIR,
    ITEM_COLS,
    MODELS_DIR,
    REPORTS_DIR,
    SEED,
    TRAIN_FRAC,
)
from conformal_demand.conformal import SplitConformalRegressor  # noqa: E402
from conformal_demand.data import load_wide, temporal_split_mask  # noqa: E402
from conformal_demand.features import feature_matrix_for_item  # noqa: E402
from conformal_demand.metrics import coverage_curve, interval_metrics, point_metrics  # noqa: E402
from conformal_demand.models import make_estimator  # noqa: E402
from conformal_demand.plots import (  # noqa: E402
    save_coverage_plot,
    save_interval_plot,
    save_risk_bars,
    save_width_vs_alpha,
)
from conformal_demand.risk import InventoryPolicy, compare_stock_policies  # noqa: E402


def save_joblib_b64(obj, path: Path) -> Path:
    import io
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    buf = io.BytesIO()
    joblib.dump(obj, buf)
    raw = buf.getvalue()
    path.write_text(base64.b64encode(raw).decode("ascii"))
    if str(path).endswith(".joblib.b64"):
        Path(str(path)[: -len(".b64")]).write_bytes(raw)
    return path


def run_item(
    wide: pd.DataFrame,
    item: str,
    model_name: str,
    alpha: float,
    alphas_grid: list[float],
) -> dict:
    X, y = feature_matrix_for_item(wide, item, lags=DEFAULT_LAGS)
    n = len(X)
    tr, ca, te = temporal_split_mask(n, TRAIN_FRAC, CALIB_FRAC)

    X_tr, y_tr = X.iloc[tr], y.iloc[tr]
    X_ca, y_ca = X.iloc[ca], y.iloc[ca]
    X_te, y_te = X.iloc[te], y.iloc[te]

    est = make_estimator(model_name, seed=SEED)
    conf = SplitConformalRegressor(est, alpha=alpha, clip_nonneg=True)
    conf.fit(X_tr, y_tr, X_ca, y_ca)
    res = conf.predict_interval(X_te)

    pm = point_metrics(y_te, res.y_pred)
    im = interval_metrics(y_te, res.lower, res.upper, conf.coverage_target)
    curve = coverage_curve(y_te.values, res.y_pred, conf.calib_scores_, alphas_grid)

    risk = compare_stock_policies(
        y_te.values,
        res.y_pred,
        res.lower,
        res.upper,
        InventoryPolicy(unit_cost=1.0, holding_cost=0.25, stockout_cost=3.0),
    )

    artifact = {
        "item": item,
        "model": model_name,
        "alpha": alpha,
        "feature_cols": list(X.columns),
        "n_train": int(len(X_tr)),
        "n_calib": int(len(X_ca)),
        "n_test": int(len(X_te)),
        "q_hat": res.q_hat,
        "point_metrics": pm,
        "interval_metrics": im,
        "coverage_curve": curve,
        "inventory_risk": risk,
        "test_dates": [str(d.date()) for d in X_te.index],
        "y_true": y_te.values.tolist(),
        "y_pred": res.y_pred.tolist(),
        "lower": res.lower.tolist(),
        "upper": res.upper.tolist(),
    }

    fig_dir = FIGURES_DIR / f"{item}_{model_name}"
    fig_dir.mkdir(parents=True, exist_ok=True)
    save_interval_plot(
        X_te.index,
        y_te.values,
        res.y_pred,
        res.lower,
        res.upper,
        fig_dir / "intervals.png",
        title=f"{item} · {model_name} · {(1-alpha):.0%} conformal PI",
    )
    save_coverage_plot(curve, fig_dir / "coverage.png", title=f"{item} · {model_name}")
    save_width_vs_alpha(curve, fig_dir / "width_vs_alpha.png")
    save_risk_bars(risk, fig_dir / "inventory_risk.png")

    save_joblib_b64(
        {
            "conformal": conf,
            "feature_cols": list(X.columns),
            "item": item,
            "model": model_name,
            "alpha": alpha,
        },
        MODELS_DIR / f"{item}_{model_name}_a{int(alpha*100):02d}.joblib.b64",
    )
    return artifact


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--source", choices=["raw", "sample"], default="raw")
    p.add_argument("--models", nargs="+", default=["ridge", "lightgbm"])
    p.add_argument("--items", nargs="+", default=list(ITEM_COLS))
    p.add_argument("--alpha", type=float, default=0.10)
    p.add_argument(
        "--alphas-grid",
        nargs="+",
        type=float,
        default=list(DEFAULT_ALPHAS) + [0.15, 0.25],
    )
    args = p.parse_args()

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    wide = load_wide(args.source)
    results = []
    for item in args.items:
        for model_name in args.models:
            print(f"=== {item} / {model_name} ===")
            art = run_item(wide, item, model_name, args.alpha, args.alphas_grid)
            cov = art["interval_metrics"]["empirical_coverage"]
            nom = art["interval_metrics"]["nominal_coverage"]
            mae = art["point_metrics"]["mae"]
            print(
                f"  MAE={mae:.3f}  coverage={cov:.3f} (nominal {nom:.3f})  "
                f"width={art['interval_metrics']['mean_width']:.3f}  qhat={art['q_hat']:.3f}"
            )
            results.append(art)

    summary_rows = []
    for r in results:
        summary_rows.append(
            {
                "item": r["item"],
                "model": r["model"],
                "alpha": r["alpha"],
                "mae": r["point_metrics"]["mae"],
                "rmse": r["point_metrics"]["rmse"],
                "r2": r["point_metrics"]["r2"],
                "empirical_coverage": r["interval_metrics"]["empirical_coverage"],
                "nominal_coverage": r["interval_metrics"]["nominal_coverage"],
                "coverage_gap": r["interval_metrics"]["coverage_gap"],
                "mean_width": r["interval_metrics"]["mean_width"],
                "q_hat": r["q_hat"],
                "service_level_upper": r["inventory_risk"]["conformal_upper"]["service_level"],
                "fill_rate_upper": r["inventory_risk"]["conformal_upper"]["fill_rate"],
                "service_level_point": r["inventory_risk"]["point_forecast"]["service_level"],
            }
        )
    summary = pd.DataFrame(summary_rows)
    summary_path = REPORTS_DIR / "summary.csv"
    summary.to_csv(summary_path, index=False)

    payload = {
        "meta": {
            "source": args.source,
            "alpha": args.alpha,
            "alphas_grid": args.alphas_grid,
            "lags": list(DEFAULT_LAGS),
            "train_frac": TRAIN_FRAC,
            "calib_frac": CALIB_FRAC,
            "seed": SEED,
            "n_series_days": int(len(wide)),
            "date_start": str(wide.index.min().date()),
            "date_end": str(wide.index.max().date()),
        },
        "summary": summary_rows,
        "runs": results,
    }
    metrics_path = REPORTS_DIR / "metrics.json"
    metrics_path.write_text(json.dumps(payload, indent=2))
    print(f"\nWrote {metrics_path}")
    print(f"Wrote {summary_path}")
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
