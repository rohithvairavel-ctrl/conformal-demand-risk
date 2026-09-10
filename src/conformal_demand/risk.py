"""Inventory risk translation from conformal intervals."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class InventoryPolicy:
    """
    Order-up-to using the conformal upper quantile as a service-level proxy.

    stock_target = upper bound (or y_pred + safety from q_hat).
    Underage when demand > stock; overage when demand < stock.
    """

    unit_cost: float = 1.0
    holding_cost: float = 0.2
    stockout_cost: float = 2.0


def simulate_inventory(
    demand: np.ndarray,
    stock_target: np.ndarray,
    policy: InventoryPolicy | None = None,
) -> dict[str, float]:
    policy = policy or InventoryPolicy()
    demand = np.asarray(demand, dtype=float)
    stock = np.asarray(stock_target, dtype=float)
    sold = np.minimum(demand, stock)
    leftover = np.maximum(stock - demand, 0.0)
    stockout_units = np.maximum(demand - stock, 0.0)
    revenue_proxy = sold.sum()
    hold = leftover.sum() * policy.holding_cost
    stockout = stockout_units.sum() * policy.stockout_cost
    purchase = stock.sum() * policy.unit_cost
    net = revenue_proxy - hold - stockout
    service_level = float((stockout_units == 0).mean())
    fill_rate = float(sold.sum() / max(demand.sum(), 1e-9))
    return {
        "service_level": service_level,
        "fill_rate": fill_rate,
        "avg_stock": float(stock.mean()),
        "avg_leftover": float(leftover.mean()),
        "avg_stockout_units": float(stockout_units.mean()),
        "total_holding_cost": float(hold),
        "total_stockout_cost": float(stockout),
        "total_purchase_cost": float(purchase),
        "net_utility": float(net),
        "n_days": int(len(demand)),
    }


def compare_stock_policies(
    demand: np.ndarray,
    y_pred: np.ndarray,
    lower: np.ndarray,
    upper: np.ndarray,
    policy: InventoryPolicy | None = None,
) -> dict[str, dict[str, float]]:
    """Point vs lower/upper conformal bounds as order-up-to levels."""
    policy = policy or InventoryPolicy()
    return {
        "point_forecast": simulate_inventory(demand, y_pred, policy),
        "conformal_lower": simulate_inventory(demand, lower, policy),
        "conformal_upper": simulate_inventory(demand, upper, policy),
        "conformal_mid": simulate_inventory(demand, 0.5 * (lower + upper), policy),
    }
