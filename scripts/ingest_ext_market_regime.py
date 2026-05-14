"""
Download extended market and yield inputs for the daily oil-direction experiment.

This script intentionally writes separate raw files so the existing baseline raw
files remain unchanged.

Outputs:
  - data/raw/market_ext_regime.csv
  - data/raw/fred_ext_regime.csv
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import pandas as pd
import yfinance as yf


BASE_DIR = Path(__file__).resolve().parent.parent
RAW_DIR = BASE_DIR / "data" / "raw"
CACHE_DIR = BASE_DIR / ".cache" / "yfinance"

DEFAULT_START_DATE = "2015-01-01"
DEFAULT_END_DATE = "2026-03-20"

MARKET_TICKERS = {
    "ovx": "^OVX",
    "move": "^MOVE",
    "gold": "GC=F",
}

FRED_SERIES = {
    "DGS3": "dgs3",
    "DGS10": "dgs10",
}


def _flatten_yfinance_close(raw: pd.DataFrame, name: str) -> pd.DataFrame:
    if raw.empty:
        return pd.DataFrame(columns=["date", f"{name}_close"])

    close = raw["Close"]
    if isinstance(close, pd.DataFrame):
        close = close.iloc[:, 0]

    out = close.rename(f"{name}_close").reset_index()
    out.rename(columns={"Date": "date"}, inplace=True)
    out["date"] = pd.to_datetime(out["date"]).dt.tz_localize(None)
    return out[["date", f"{name}_close"]]


def download_market_ext(start: str, end: str) -> pd.DataFrame:
    frames = []
    for name, ticker in MARKET_TICKERS.items():
        print(f"Downloading {name} ({ticker})...")
        raw = yf.download(ticker, start=start, end=end, progress=False, auto_adjust=False)
        frames.append(_flatten_yfinance_close(raw, name))
        time.sleep(0.25)

    if not frames:
        return pd.DataFrame(columns=["date"])

    result = frames[0]
    for frame in frames[1:]:
        result = result.merge(frame, on="date", how="outer")
    return result.sort_values("date").reset_index(drop=True)


def fetch_fred_series(series_id: str, name: str, start: str, end: str) -> pd.DataFrame:
    url = f"https://fred.stlouisfed.org/graph/fredgraph.csv?id={series_id}"
    print(f"Downloading FRED {series_id} -> {name}...")
    df = pd.read_csv(url)
    df.rename(columns={"observation_date": "date", series_id: name}, inplace=True)
    df["date"] = pd.to_datetime(df["date"])
    df[name] = pd.to_numeric(df[name], errors="coerce")
    mask = (df["date"] >= pd.Timestamp(start)) & (df["date"] <= pd.Timestamp(end))
    return df.loc[mask, ["date", name]].reset_index(drop=True)


def download_fred_ext(start: str, end: str) -> pd.DataFrame:
    frames = []
    for series_id, name in FRED_SERIES.items():
        frames.append(fetch_fred_series(series_id, name, start, end))
        time.sleep(0.25)

    result = frames[0]
    for frame in frames[1:]:
        result = result.merge(frame, on="date", how="outer")
    return result.sort_values("date").reset_index(drop=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--start", default=DEFAULT_START_DATE)
    parser.add_argument("--end", default=DEFAULT_END_DATE)
    parser.add_argument("--market-output", default=str(RAW_DIR / "market_ext_regime.csv"))
    parser.add_argument("--fred-output", default=str(RAW_DIR / "fred_ext_regime.csv"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    if hasattr(yf, "set_tz_cache_location"):
        yf.set_tz_cache_location(str(CACHE_DIR))

    market = download_market_ext(args.start, args.end)
    market_path = Path(args.market_output)
    market_path.parent.mkdir(parents=True, exist_ok=True)
    market.to_csv(market_path, index=False)
    print(f"Saved market extension: {market_path} ({market.shape[0]} rows x {market.shape[1]} cols)")

    fred = download_fred_ext(args.start, args.end)
    fred_path = Path(args.fred_output)
    fred_path.parent.mkdir(parents=True, exist_ok=True)
    fred.to_csv(fred_path, index=False)
    print(f"Saved FRED extension: {fred_path} ({fred.shape[0]} rows x {fred.shape[1]} cols)")


if __name__ == "__main__":
    main()
