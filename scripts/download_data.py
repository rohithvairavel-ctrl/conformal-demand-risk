#!/usr/bin/env python3
"""Download skforecast simulated_items_sales and write a small sample CSV."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd
import requests

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from conformal_demand.config import (  # noqa: E402
    DATA_RAW,
    DATA_SAMPLE,
    DATA_URL,
    RAW_FILENAME,
    SAMPLE_FILENAME,
)


def download(url: str = DATA_URL, dest: Path | None = None) -> Path:
    dest = dest or (DATA_RAW / RAW_FILENAME)
    dest.parent.mkdir(parents=True, exist_ok=True)
    print(f"Fetching {url}")
    r = requests.get(url, timeout=60)
    r.raise_for_status()
    dest.write_bytes(r.content)
    print(f"Wrote {dest} ({dest.stat().st_size:,} bytes)")
    return dest


def write_sample(raw_path: Path, n_rows: int = 180) -> Path:
    df = pd.read_csv(raw_path)
    sample = df.head(n_rows)
    out = DATA_SAMPLE / SAMPLE_FILENAME
    out.parent.mkdir(parents=True, exist_ok=True)
    sample.to_csv(out, index=False)
    print(f"Wrote sample {out} ({len(sample)} rows)")
    return out


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--sample-rows", type=int, default=180)
    p.add_argument("--force", action="store_true")
    args = p.parse_args()

    dest = DATA_RAW / RAW_FILENAME
    if dest.exists() and not args.force:
        print(f"Already present: {dest} (use --force to re-download)")
    else:
        download(dest=dest)

    write_sample(dest, n_rows=args.sample_rows)
    df = pd.read_csv(dest, parse_dates=["date"])
    print(f"Shape: {df.shape} | date range: {df['date'].min().date()} -> {df['date'].max().date()}")


if __name__ == "__main__":
    main()
