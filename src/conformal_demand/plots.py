"""Matplotlib figures for coverage, intervals, and inventory risk."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def save_interval_plot(
    dates,
    y_true,
    y_pred,
    lower,
    upper,
    path: Path,
    title: str = "Conformal prediction intervals",
    max_points: int = 120,
) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    n = len(y_true)
    if n > max_points:
        sl = slice(n - max_points, n)
    else:
        sl = slice(None)
    fig, ax = plt.subplots(figsize=(11, 4.5))
    ax.fill_between(dates[sl], lower[sl], upper[sl], alpha=0.25, label="Conformal PI", color="#4C78A8")
    ax.plot(dates[sl], y_true[sl], label="Actual", color="#333333", lw=1.4)
    ax.plot(dates[sl], y_pred[sl], label="Point forecast", color="#F58518", lw=1.2)
    ax.set_title(title)
    ax.set_ylabel("Sales")
    ax.legend(loc="upper left")
    ax.grid(True, alpha=0.3)
    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)
    return path


def save_coverage_plot(curve: list[dict], path: Path, title: str = "Coverage vs nominal") -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    nominal = [c["nominal_coverage"] for c in curve]
    empirical = [c["empirical_coverage"] for c in curve]
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot([0.7, 1.0], [0.7, 1.0], "--", color="gray", label="Perfect calibration")
    ax.scatter(nominal, empirical, s=60, color="#4C78A8", zorder=3)
    ax.plot(nominal, empirical, color="#4C78A8", label="Split conformal")
    ax.set_xlabel("Nominal coverage (1 - alpha)")
    ax.set_ylabel("Empirical coverage")
    ax.set_title(title)
    ax.set_xlim(0.7, 1.01)
    ax.set_ylim(0.7, 1.01)
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)
    return path


def save_width_vs_alpha(curve: list[dict], path: Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    alphas = [c["alpha"] for c in curve]
    widths = [c["mean_width"] for c in curve]
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(alphas, widths, marker="o", color="#E45756")
    ax.set_xlabel("alpha (miscoverage)")
    ax.set_ylabel("Mean interval width")
    ax.set_title("Interval width vs alpha")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)
    return path


def save_risk_bars(risk: dict[str, dict], path: Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    names = list(risk.keys())
    service = [risk[k]["service_level"] for k in names]
    fill = [risk[k]["fill_rate"] for k in names]
    x = np.arange(len(names))
    fig, ax = plt.subplots(figsize=(8, 4.5))
    w = 0.35
    ax.bar(x - w / 2, service, w, label="Service level", color="#4C78A8")
    ax.bar(x + w / 2, fill, w, label="Fill rate", color="#54A24B")
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=15, ha="right")
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Rate")
    ax.set_title("Inventory risk: stock policies from conformal bounds")
    ax.legend()
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)
    return path
