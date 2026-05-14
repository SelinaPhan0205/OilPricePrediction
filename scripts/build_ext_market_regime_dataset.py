"""
Build the daily_ext_market_regime_v1 dataset.

The builder starts from the canonical leakage-safe processed dataset and appends
deterministic extended market/yield/regime features. It does not overwrite the
current baseline dataset.

Inputs:
  - data/processed/dataset_final_noleak_processed.csv
  - data/raw/market_ext_regime.csv
  - data/raw/fred_ext_regime.csv

Outputs:
  - data/processed/dataset_final_noleak_ext_market_regime_v1.csv
  - data/processed/dataset_final_noleak_ext_market_regime_v1_report.json
  - data/processed/dataset_final_noleak_ext_market_regime_v1_features.csv
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


BASE_DIR = Path(__file__).resolve().parent.parent
RAW_DIR = BASE_DIR / "data" / "raw"
PROCESSED_DIR = BASE_DIR / "data" / "processed"

DEFAULT_BASE = PROCESSED_DIR / "dataset_final_noleak_processed.csv"
DEFAULT_MARKET_EXT = RAW_DIR / "market_ext_regime.csv"
DEFAULT_FRED_EXT = RAW_DIR / "fred_ext_regime.csv"
DEFAULT_OUTPUT = PROCESSED_DIR / "dataset_final_noleak_ext_market_regime_v1.csv"
DEFAULT_REPORT = PROCESSED_DIR / "dataset_final_noleak_ext_market_regime_v1_report.json"
DEFAULT_FEATURES = PROCESSED_DIR / "dataset_final_noleak_ext_market_regime_v1_features.csv"

TARGET_COLS = ["date", "oil_return_fwd1", "oil_return_fwd1_date"]
MARKET_CLOSE_COLS = ["ovx_close", "move_close", "gold_close"]
YIELD_COLS = ["dgs3", "dgs10"]


def signed_log1p(series: pd.Series) -> pd.Series:
    return np.sign(series) * np.log1p(np.abs(series))


def read_csv_dates(path: Path, required: bool = True) -> pd.DataFrame:
    if not path.exists():
        if required:
            raise FileNotFoundError(
                f"Missing {path}. Run scripts/ingest_ext_market_regime.py first."
            )
        return pd.DataFrame(columns=["date"])
    df = pd.read_csv(path)
    if "date" not in df.columns:
        raise ValueError(f"{path} must contain a date column")
    df["date"] = pd.to_datetime(df["date"])
    return df.sort_values("date").reset_index(drop=True)


def make_timeline(base: pd.DataFrame, market: pd.DataFrame, fred: pd.DataFrame) -> pd.DataFrame:
    min_date = base["date"].min()
    max_date = base["date"].max()
    dates = pd.DataFrame({"date": pd.date_range(min_date, max_date, freq="B")})
    out = dates.merge(market, on="date", how="left").merge(fred, on="date", how="left")

    for col in MARKET_CLOSE_COLS + YIELD_COLS:
        if col in out.columns:
            out[col] = pd.to_numeric(out[col], errors="coerce").ffill()
    return out


def add_market_extension_features(timeline: pd.DataFrame) -> tuple[pd.DataFrame, list[dict]]:
    features = pd.DataFrame({"date": timeline["date"]})
    rows: list[dict] = []

    for close_col in MARKET_CLOSE_COLS:
        if close_col not in timeline.columns:
            continue

        prefix = close_col.removesuffix("_close")
        close = pd.to_numeric(timeline[close_col], errors="coerce")
        ret = close.pct_change()
        level_lag1 = close.shift(1)
        ret_lag1 = ret.shift(1)

        new_cols = {
            f"{prefix}_return_slog1p": signed_log1p(ret),
            f"{prefix}_return_lag1_slog1p": signed_log1p(ret_lag1),
            f"{prefix}_level_lag1_log1p": np.log1p(level_lag1.clip(lower=0)),
        }

        if prefix in {"ovx", "move"}:
            q80 = level_lag1.rolling(252, min_periods=60).quantile(0.80).shift(1)
            q90 = level_lag1.rolling(252, min_periods=60).quantile(0.90).shift(1)
            new_cols[f"{prefix}_high_regime_lag1"] = (level_lag1 > q80).astype(float)
            new_cols[f"{prefix}_extreme_regime_lag1"] = (level_lag1 > q90).astype(float)

        for col, values in new_cols.items():
            features[col] = values
            rows.append(
                {
                    "feature": col,
                    "source": close_col,
                    "family": "extended_market",
                    "availability": "EOD T or lagged; no future target data",
                }
            )

    return features, rows


def add_yield_extension_features(timeline: pd.DataFrame) -> tuple[pd.DataFrame, list[dict]]:
    features = pd.DataFrame({"date": timeline["date"]})
    rows: list[dict] = []

    for col in YIELD_COLS:
        if col not in timeline.columns:
            continue
        value = pd.to_numeric(timeline[col], errors="coerce")
        lag1 = value.shift(1)
        change_lag1 = value.diff().shift(1)
        out_cols = {
            f"{col}_lag1": lag1,
            f"{col}_change_lag1_slog1p": signed_log1p(change_lag1),
        }
        for out_col, values in out_cols.items():
            features[out_col] = values
            rows.append(
                {
                    "feature": out_col,
                    "source": col,
                    "family": "yield_curve",
                    "availability": "lagged daily FRED series",
                }
            )

    if {"dgs10", "dgs3"}.issubset(timeline.columns):
        spread = pd.to_numeric(timeline["dgs10"], errors="coerce") - pd.to_numeric(
            timeline["dgs3"], errors="coerce"
        )
        features["yield_10y_3y_lag1"] = spread.shift(1)
        features["yield_10y_3y_change_lag1_slog1p"] = signed_log1p(spread.diff().shift(1))
        rows.extend(
            [
                {
                    "feature": "yield_10y_3y_lag1",
                    "source": "dgs10-dgs3",
                    "family": "yield_curve",
                    "availability": "lagged daily FRED series",
                },
                {
                    "feature": "yield_10y_3y_change_lag1_slog1p",
                    "source": "dgs10-dgs3",
                    "family": "yield_curve",
                    "availability": "lagged daily FRED series",
                },
            ]
        )

    return features, rows


def add_regime_features(base: pd.DataFrame) -> tuple[pd.DataFrame, list[dict]]:
    features = pd.DataFrame({"date": base["date"]})
    rows: list[dict] = []

    oil = pd.to_numeric(base["oil_return"], errors="coerce")
    oil_mom_5 = oil.rolling(5, min_periods=3).sum().shift(1)
    oil_mom_20 = oil.rolling(20, min_periods=10).sum().shift(1)
    oil_vol_5 = oil.rolling(5, min_periods=3).std().shift(1)
    oil_vol_20 = oil.rolling(20, min_periods=10).std().shift(1)
    oil_vol_q80 = oil_vol_20.rolling(252, min_periods=60).quantile(0.80).shift(1)

    regime_cols = {
        "oil_momentum_5d_lag1": oil_mom_5,
        "oil_momentum_20d_lag1": oil_mom_20,
        "oil_realized_vol_5d_lag1": oil_vol_5,
        "oil_realized_vol_20d_lag1": oil_vol_20,
        "oil_uptrend_5d_lag1": (oil_mom_5 > 0).astype(float),
        "oil_uptrend_20d_lag1": (oil_mom_20 > 0).astype(float),
        "oil_high_vol_regime_lag1": (oil_vol_20 > oil_vol_q80).astype(float),
    }

    if {"usd_return", "sp500_return"}.issubset(base.columns):
        usd_lag1 = pd.to_numeric(base["usd_return"], errors="coerce").shift(1)
        sp_lag1 = pd.to_numeric(base["sp500_return"], errors="coerce").shift(1)
        regime_cols["risk_off_cross_asset_lag1"] = ((usd_lag1 > 0) & (sp_lag1 < 0)).astype(float)

    for col, values in regime_cols.items():
        features[col] = values
        rows.append(
            {
                "feature": col,
                "source": "base_processed_returns",
                "family": "regime",
                "availability": "lagged rolling calculation",
            }
        )

    return features, rows


def merge_feature_frames(frames: list[pd.DataFrame]) -> pd.DataFrame:
    result = frames[0]
    for frame in frames[1:]:
        result = result.merge(frame, on="date", how="left")
    return result


def build_dataset(base_path: Path, market_path: Path, fred_path: Path) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    base = read_csv_dates(base_path)
    if "oil_return_fwd1_date" in base.columns:
        base["oil_return_fwd1_date"] = pd.to_datetime(base["oil_return_fwd1_date"])

    market = read_csv_dates(market_path)
    fred = read_csv_dates(fred_path)
    timeline = make_timeline(base, market, fred)

    market_features, market_rows = add_market_extension_features(timeline)
    yield_features, yield_rows = add_yield_extension_features(timeline)
    regime_features, regime_rows = add_regime_features(base)
    ext_features = merge_feature_frames([market_features, yield_features, regime_features])

    feature_cols = [c for c in ext_features.columns if c != "date"]
    missing_before_fill = ext_features[feature_cols].isna().sum().to_dict()

    # Deterministic cleanup: returns/regime flags that cannot be computed at the
    # start of the series are neutral-filled. No full-sample fitted imputer is used.
    ext_features[feature_cols] = ext_features[feature_cols].replace([np.inf, -np.inf], np.nan)
    ext_features[feature_cols] = ext_features[feature_cols].fillna(0.0)

    out = base.merge(ext_features, on="date", how="left")
    out[feature_cols] = out[feature_cols].fillna(0.0)

    feature_report = pd.DataFrame(market_rows + yield_rows + regime_rows)
    summary = {
        "base_file": str(base_path),
        "market_ext_file": str(market_path),
        "fred_ext_file": str(fred_path),
        "rows": int(out.shape[0]),
        "base_columns": int(base.shape[1]),
        "new_feature_count": int(len(feature_cols)),
        "output_columns": int(out.shape[1]),
        "date_min": str(out["date"].min().date()),
        "date_max": str(out["date"].max().date()),
        "target_date_min": str(pd.to_datetime(out["oil_return_fwd1_date"]).min().date()),
        "target_date_max": str(pd.to_datetime(out["oil_return_fwd1_date"]).max().date()),
        "missing_before_neutral_fill": missing_before_fill,
        "new_features": feature_cols,
    }
    return out, feature_report, summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", default=str(DEFAULT_BASE))
    parser.add_argument("--market-ext", default=str(DEFAULT_MARKET_EXT))
    parser.add_argument("--fred-ext", default=str(DEFAULT_FRED_EXT))
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT))
    parser.add_argument("--report", default=str(DEFAULT_REPORT))
    parser.add_argument("--features", default=str(DEFAULT_FEATURES))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output, feature_report, summary = build_dataset(
        Path(args.base), Path(args.market_ext), Path(args.fred_ext)
    )

    output_path = Path(args.output)
    report_path = Path(args.report)
    features_path = Path(args.features)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    output.to_csv(output_path, index=False)
    feature_report.to_csv(features_path, index=False)
    report_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(f"Saved dataset: {output_path}")
    print(f"Saved feature report: {features_path}")
    print(f"Saved summary report: {report_path}")
    print(
        f"Rows={summary['rows']} | Base cols={summary['base_columns']} | "
        f"New features={summary['new_feature_count']} | Output cols={summary['output_columns']}"
    )


if __name__ == "__main__":
    main()
