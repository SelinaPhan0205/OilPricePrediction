import numpy as np
import pandas as pd
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
RAW_DIR = BASE_DIR / "data/raw"
PROCESSED_DIR = BASE_DIR / "data/processed"

PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

MARKET_FILE = RAW_DIR / "market_data.csv"
FRED_FILE = RAW_DIR / "fred_data.csv"
EIA_FILE = RAW_DIR / "eia_data.csv"
GDELT_FILE = RAW_DIR / "gdelt_data.csv"
OUTPUT_FILE = PROCESSED_DIR / "dataset_preprocessed.csv"


def load_data() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    market = pd.read_csv(MARKET_FILE)
    fred = pd.read_csv(FRED_FILE)
    eia = pd.read_csv(EIA_FILE)
    gdelt = pd.read_csv(GDELT_FILE)

    for df in [market, fred, eia, gdelt]:
        df["date"] = pd.to_datetime(df["date"])

    return market, fred, eia, gdelt


def preprocess_gdelt(gdelt: pd.DataFrame) -> pd.DataFrame:
    gdelt = gdelt.sort_values("date").copy()

    # 1 = ngày thiếu dữ liệu gốc GDELT, sẽ được nội suy bằng ffill ngắn hạn.
    gdelt["gdelt_data_imputed"] = gdelt["gdelt_tone"].isna().astype(int)

    for col in ["gdelt_tone", "gdelt_goldstein", "gdelt_volume", "gdelt_events"]:
        gdelt[col] = gdelt[col].ffill(limit=3)

    gdelt["gdelt_tone_7d"] = gdelt["gdelt_tone"].rolling(7, min_periods=1).mean()
    gdelt["gdelt_tone_30d"] = gdelt["gdelt_tone"].rolling(30, min_periods=1).mean()
    gdelt["gdelt_volume_7d"] = gdelt["gdelt_volume"].rolling(7, min_periods=1).mean()
    gdelt["gdelt_goldstein_7d"] = gdelt["gdelt_goldstein"].rolling(7, min_periods=1).mean()
    gdelt["gdelt_tone_spike"] = (gdelt["gdelt_tone"] < gdelt["gdelt_tone_30d"] - 1).astype(int)

    gdelt["gdelt_volume_log"] = np.log1p(gdelt["gdelt_volume"])
    gdelt["gdelt_volume_7d_log"] = np.log1p(gdelt["gdelt_volume_7d"])

    return gdelt


def preprocess_fred(fred: pd.DataFrame) -> pd.DataFrame:
    fred = fred.sort_values("date").set_index("date")

    monthly_cols = ["fed_funds_rate", "cpi", "unemployment"]
    for col in monthly_cols:
        # Các series monthly đã được shift ở bước crawl để phản ánh publication lag.
        fred[f"{col}_lag"] = fred[col]

    fred = fred.reset_index()
    return fred


def compute_market_returns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["oil_return"] = df["oil_close"].pct_change()
    df["usd_return"] = df["usd_close"].pct_change()
    df["sp500_return"] = df["sp500_close"].pct_change()
    df["vix_return"] = df["vix_close"].pct_change()
    return df


def merge_datasets(
    market: pd.DataFrame,
    fred: pd.DataFrame,
    eia: pd.DataFrame,
    gdelt: pd.DataFrame,
) -> pd.DataFrame:
    df = market.sort_values("date").copy()

    fred_keep = [
        "date",
        "yield_spread",
        "wti_fred",
        "fed_funds_rate_lag",
        "cpi_lag",
        "unemployment_lag",
    ]

    df = df.merge(fred[fred_keep], on="date", how="left")
    df = df.merge(eia.sort_values("date"), on="date", how="left")
    df = df.merge(gdelt.sort_values("date"), on="date", how="left")

    return df


def post_merge_fill(df: pd.DataFrame) -> pd.DataFrame:
    df = df.sort_values("date").copy()

    fill_cols = [col for col in df.columns if col not in ["date", "oil_return", "usd_return", "sp500_return", "vix_return"]]
    df[fill_cols] = df[fill_cols].ffill(limit=3)

    if "gdelt_data_imputed" in df.columns:
        df["gdelt_data_imputed"] = df["gdelt_data_imputed"].fillna(1).astype(int)

    return df


def main():
    market, fred, eia, gdelt = load_data()

    fred = preprocess_fred(fred)
    gdelt = preprocess_gdelt(gdelt)

    df = merge_datasets(market, fred, eia, gdelt)
    df = compute_market_returns(df)
    df = post_merge_fill(df)

    df.to_csv(OUTPUT_FILE, index=False)

    total_missing = int(df.isna().sum().sum())
    print(f"Preprocessing complete. Shape={df.shape}, missing={total_missing}")


if __name__ == "__main__":
    main()