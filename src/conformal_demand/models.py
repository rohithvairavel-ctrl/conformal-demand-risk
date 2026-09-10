"""Point forecast model factories: Ridge and LightGBM."""

from __future__ import annotations

from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


def make_ridge(alpha: float = 1.0, seed: int = 42) -> Pipeline:
    return Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            ("model", Ridge(alpha=alpha, random_state=seed)),
        ]
    )


def make_lightgbm(
    n_estimators: int = 200,
    learning_rate: float = 0.05,
    max_depth: int = 6,
    seed: int = 42,
):
    from lightgbm import LGBMRegressor

    return LGBMRegressor(
        n_estimators=n_estimators,
        learning_rate=learning_rate,
        max_depth=max_depth,
        subsample=0.9,
        colsample_bytree=0.9,
        random_state=seed,
        verbose=-1,
        n_jobs=2,
    )


def make_estimator(name: str, seed: int = 42):
    name = name.lower()
    if name in {"ridge", "linear"}:
        return make_ridge(seed=seed)
    if name in {"lgbm", "lightgbm"}:
        return make_lightgbm(seed=seed)
    raise ValueError(f"Unknown model '{name}'. Use ridge or lightgbm.")
