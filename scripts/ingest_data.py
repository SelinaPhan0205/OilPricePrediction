"""Download core market data.

Defaults preserve the original 2015-2026 raw file.  Long-history experiments can
pass --start/--end/--output to write a separate branch file.
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import pandas as pd
import yfinance as yf

# =========================
# CONFIG
# =========================

START_DATE = "2015-01-01"
END_DATE = "2026-03-20"

TICKERS = {
    "oil": "BZ=F",      # Brent oil
    "usd": "DX-Y.NYB",  # Dollar Index
    "sp500": "^GSPC",
    "vix": "^VIX"
}

BASE_DIR = Path(__file__).resolve().parent.parent
RAW_DIR = BASE_DIR / "data/raw"
RAW_DIR.mkdir(parents=True, exist_ok=True)
DEFAULT_OUTPUT = RAW_DIR / "market_data.csv"


# =========================
# DOWNLOAD MARKET DATA
# =========================

def flatten_close(raw: pd.DataFrame, name: str) -> pd.DataFrame:
    if raw.empty:
        return pd.DataFrame(columns=["date", f"{name}_close"])

    close = raw["Close"]
    if isinstance(close, pd.DataFrame):
        close = close.iloc[:, 0]

    out = close.rename(f"{name}_close").reset_index()
    out.rename(columns={"Date": "date"}, inplace=True)
    out["date"] = pd.to_datetime(out["date"]).dt.tz_localize(None)
    return out[["date", f"{name}_close"]]


def download_market_data(start: str, end: str):

    dfs = []

    for name, ticker in TICKERS.items():

        print(f"Downloading {name} ({ticker})...")

        raw = yf.download(
            ticker,
            start=start,
            end=end,
            progress=False,
            auto_adjust=False,
        )

        dfs.append(flatten_close(raw, name))
        time.sleep(0.25)

    if not dfs:
        return pd.DataFrame(columns=["date"])

    market_df = dfs[0]
    for frame in dfs[1:]:
        market_df = market_df.merge(frame, on="date", how="outer")

    return market_df.sort_values("date").reset_index(drop=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--start", default=START_DATE)
    parser.add_argument("--end", default=END_DATE)
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT))
    return parser.parse_args()


# =========================
# MAIN
# =========================

def main():
    args = parse_args()
    market = download_market_data(args.start, args.end)

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    market.to_csv(output, index=False)

    print(f"Data ingestion complete: {output} ({market.shape[0]} rows x {market.shape[1]} cols)")


if __name__ == "__main__":
    main()
