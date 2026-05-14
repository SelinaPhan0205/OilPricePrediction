"""
GDELT 1.0 crawler by raw ZIP download.

This script intentionally does not use the GDELT DOC/API endpoints.  It reads
the public tab-delimited event ZIP files from data.gdeltproject.org/events/.

GDELT 1.0 file layout matters:
- 1979-2005: yearly ZIP files, e.g. 2005.zip
- 2006-2013-03: monthly ZIP files, e.g. 200701.zip
- 2013-04-01 onward: daily ZIP files, e.g. 20130401.export.CSV.zip

The previous crawler only requested daily files, which made 2007-2012 look
empty even though the data exists in monthly files.
"""

from __future__ import annotations

import argparse
import io
import json
import os
import shutil
import time
import zipfile
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd
import requests


BASE_DIR = Path(__file__).resolve().parent.parent
RAW_DIR = BASE_DIR / "data/raw"
RAW_DIR.mkdir(parents=True, exist_ok=True)

PROGRESS_FILE = str(RAW_DIR / "gdelt_v4_progress.json")
OUTPUT_FILE = str(RAW_DIR / "gdelt_data.csv")

START_DATE = "2015-01-01"
TARGET_END_DATE = "2026-03-20"
END_DATE = min(datetime.today().strftime("%Y-%m-%d"), TARGET_END_DATE)
OUTPUT_START_DATE = START_DATE

EVENTS_BASE_URL = "http://data.gdeltproject.org/events"
GDELT_V2_BASE_URL = "http://data.gdeltproject.org/gdeltv2"
DAILY_START = pd.Timestamp("2013-04-01")
MONTHLY_START = pd.Timestamp("2006-01-01")
GDELT_V2_START = pd.Timestamp("2015-02-19")

# CAMEO actor country codes used in Actor1CountryCode / Actor2CountryCode.
MIDDLE_EAST_ACTOR_COUNTRIES = {
    "IRQ",  # Iraq
    "IRN",  # Iran
    "SAU",  # Saudi Arabia
    "SYR",  # Syria
    "YEM",  # Yemen
    "ISR",  # Israel
    "PSE",  # Palestine
    "LBN",  # Lebanon
    "KWT",  # Kuwait
    "ARE",  # United Arab Emirates
    "QAT",  # Qatar
    "BHR",  # Bahrain
    "OMN",  # Oman
    "JOR",  # Jordan
}

# FIPS-style geo country codes used in Actor*Geo_CountryCode and
# ActionGeo_CountryCode.  The old crawler compared these fields against the
# CAMEO 3-letter set, so geolocated Middle East events were undercounted.
MIDDLE_EAST_GEO_COUNTRIES = {
    "IZ",  # Iraq
    "IR",  # Iran
    "SA",  # Saudi Arabia
    "SY",  # Syria
    "YM",  # Yemen
    "IS",  # Israel
    "WE",  # West Bank
    "GZ",  # Gaza Strip
    "LE",  # Lebanon
    "KU",  # Kuwait
    "AE",  # United Arab Emirates
    "QA",  # Qatar
    "BA",  # Bahrain
    "MU",  # Oman
    "JO",  # Jordan
}

GDELT_COLS = [
    "GlobalEventID",
    "Day",
    "MonthYear",
    "Year",
    "FractionDate",
    "Actor1Code",
    "Actor1Name",
    "Actor1CountryCode",
    "Actor1KnownGroupCode",
    "Actor1EthnicCode",
    "Actor1Religion1Code",
    "Actor1Religion2Code",
    "Actor1Type1Code",
    "Actor1Type2Code",
    "Actor1Type3Code",
    "Actor2Code",
    "Actor2Name",
    "Actor2CountryCode",
    "Actor2KnownGroupCode",
    "Actor2EthnicCode",
    "Actor2Religion1Code",
    "Actor2Religion2Code",
    "Actor2Type1Code",
    "Actor2Type2Code",
    "Actor2Type3Code",
    "IsRootEvent",
    "EventCode",
    "EventBaseCode",
    "EventRootCode",
    "QuadClass",
    "GoldsteinScale",
    "NumMentions",
    "NumSources",
    "NumArticles",
    "AvgTone",
    "Actor1Geo_Type",
    "Actor1Geo_FullName",
    "Actor1Geo_CountryCode",
    "Actor1Geo_ADM1Code",
    "Actor1Geo_Lat",
    "Actor1Geo_Long",
    "Actor1Geo_FeatureID",
    "Actor2Geo_Type",
    "Actor2Geo_FullName",
    "Actor2Geo_CountryCode",
    "Actor2Geo_ADM1Code",
    "Actor2Geo_Lat",
    "Actor2Geo_Long",
    "Actor2Geo_FeatureID",
    "ActionGeo_Type",
    "ActionGeo_FullName",
    "ActionGeo_CountryCode",
    "ActionGeo_ADM1Code",
    "ActionGeo_Lat",
    "ActionGeo_Long",
    "ActionGeo_FeatureID",
    "DATEADDED",
    "SOURCEURL",
]

READ_USECOLS = [
    1,   # Day
    7,   # Actor1CountryCode
    17,  # Actor2CountryCode
    30,  # GoldsteinScale
    33,  # NumArticles
    34,  # AvgTone
    37,  # Actor1Geo_CountryCode
    44,  # Actor2Geo_CountryCode
    51,  # ActionGeo_CountryCode
]

GDELT_V2_COLS = [
    "GlobalEventID",
    "Day",
    "MonthYear",
    "Year",
    "FractionDate",
    "Actor1Code",
    "Actor1Name",
    "Actor1CountryCode",
    "Actor1KnownGroupCode",
    "Actor1EthnicCode",
    "Actor1Religion1Code",
    "Actor1Religion2Code",
    "Actor1Type1Code",
    "Actor1Type2Code",
    "Actor1Type3Code",
    "Actor2Code",
    "Actor2Name",
    "Actor2CountryCode",
    "Actor2KnownGroupCode",
    "Actor2EthnicCode",
    "Actor2Religion1Code",
    "Actor2Religion2Code",
    "Actor2Type1Code",
    "Actor2Type2Code",
    "Actor2Type3Code",
    "IsRootEvent",
    "EventCode",
    "EventBaseCode",
    "EventRootCode",
    "QuadClass",
    "GoldsteinScale",
    "NumMentions",
    "NumSources",
    "NumArticles",
    "AvgTone",
    "Actor1Geo_Type",
    "Actor1Geo_FullName",
    "Actor1Geo_CountryCode",
    "Actor1Geo_ADM1Code",
    "Actor1Geo_ADM2Code",
    "Actor1Geo_Lat",
    "Actor1Geo_Long",
    "Actor1Geo_FeatureID",
    "Actor2Geo_Type",
    "Actor2Geo_FullName",
    "Actor2Geo_CountryCode",
    "Actor2Geo_ADM1Code",
    "Actor2Geo_ADM2Code",
    "Actor2Geo_Lat",
    "Actor2Geo_Long",
    "Actor2Geo_FeatureID",
    "ActionGeo_Type",
    "ActionGeo_FullName",
    "ActionGeo_CountryCode",
    "ActionGeo_ADM1Code",
    "ActionGeo_ADM2Code",
    "ActionGeo_Lat",
    "ActionGeo_Long",
    "ActionGeo_FeatureID",
    "DATEADDED",
    "SOURCEURL",
]

READ_USECOLS_V2 = [
    1,   # Day
    7,   # Actor1CountryCode
    17,  # Actor2CountryCode
    30,  # GoldsteinScale
    33,  # NumArticles
    34,  # AvgTone
    37,  # Actor1Geo_CountryCode
    45,  # Actor2Geo_CountryCode
    53,  # ActionGeo_CountryCode
]

BASE_COLS = ["gdelt_tone", "gdelt_goldstein", "gdelt_volume", "gdelt_events"]
CHUNKSIZE = 200_000

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/120.0.0.0 Safari/537.36"
    )
}


@dataclass(frozen=True)
class DownloadUnit:
    unit_id: str
    url: str
    start: pd.Timestamp
    end: pd.Timestamp
    mode: str  # "yearly", "monthly", or "daily"


def get_dates(start: str, end: str) -> list[datetime]:
    start_dt = datetime.strptime(start, "%Y-%m-%d")
    end_dt = datetime.strptime(end, "%Y-%m-%d")

    current = start_dt
    all_dates = []
    while current <= end_dt:
        all_dates.append(current)
        current += timedelta(days=1)
    return all_dates


def month_end(ts: pd.Timestamp) -> pd.Timestamp:
    return ts + pd.offsets.MonthEnd(0)


def year_end(ts: pd.Timestamp) -> pd.Timestamp:
    return pd.Timestamp(year=ts.year, month=12, day=31)


def iter_download_units(start: str, end: str) -> list[DownloadUnit]:
    start_ts = pd.Timestamp(start)
    end_ts = pd.Timestamp(end)
    units: list[DownloadUnit] = []

    if start_ts < MONTHLY_START:
        year = start_ts.year
        while pd.Timestamp(year=year, month=1, day=1) <= min(end_ts, MONTHLY_START - pd.Timedelta(days=1)):
            unit_start = max(start_ts, pd.Timestamp(year=year, month=1, day=1))
            unit_end = min(end_ts, year_end(unit_start), MONTHLY_START - pd.Timedelta(days=1))
            units.append(
                DownloadUnit(
                    unit_id=f"yearly:{year}",
                    url=f"{EVENTS_BASE_URL}/{year}.zip",
                    start=unit_start,
                    end=unit_end,
                    mode="yearly",
                )
            )
            year += 1

    monthly_from = max(start_ts, MONTHLY_START)
    monthly_to = min(end_ts, DAILY_START - pd.Timedelta(days=1))
    current = pd.Timestamp(year=monthly_from.year, month=monthly_from.month, day=1)
    while current <= monthly_to:
        unit_start = max(start_ts, current)
        unit_end = min(end_ts, month_end(current), monthly_to)
        yyyymm = current.strftime("%Y%m")
        units.append(
            DownloadUnit(
                unit_id=f"monthly:{yyyymm}",
                url=f"{EVENTS_BASE_URL}/{yyyymm}.zip",
                start=unit_start,
                end=unit_end,
                mode="monthly",
            )
        )
        current = current + pd.offsets.MonthBegin(1)

    daily_from = max(start_ts, DAILY_START)
    current = daily_from
    while current <= end_ts:
        yyyymmdd = current.strftime("%Y%m%d")
        units.append(
            DownloadUnit(
                unit_id=f"daily:{yyyymmdd}",
                url=f"{EVENTS_BASE_URL}/{yyyymmdd}.export.CSV.zip",
                start=current,
                end=current,
                mode="daily",
            )
        )
        current += pd.Timedelta(days=1)

    return units


def expected_business_dates(unit: DownloadUnit, output_start: str, output_end: str) -> set[str]:
    start = max(unit.start, pd.Timestamp(output_start))
    end = min(unit.end, pd.Timestamp(output_end))
    if start > end:
        return set()
    return {d.strftime("%Y-%m-%d") for d in pd.date_range(start, end, freq="B")}


def unit_already_covered(unit: DownloadUnit, processed_dates: set[str], output_start: str, output_end: str) -> bool:
    expected = expected_business_dates(unit, output_start, output_end)
    return bool(expected) and expected.issubset(processed_dates)


def unit_has_output_dates(unit: DownloadUnit, output_start: str, output_end: str) -> bool:
    return bool(expected_business_dates(unit, output_start, output_end))


def load_progress() -> tuple[list[dict], set[str], set[str]]:
    if os.path.exists(PROGRESS_FILE):
        try:
            with open(PROGRESS_FILE, "r", encoding="utf-8") as f:
                payload = json.load(f)

            if isinstance(payload, list):
                rows = payload
                processed_dates = {
                    r["date"]
                    for r in rows
                    if isinstance(r, dict) and "date" in r and any(r.get(col) is not None for col in BASE_COLS)
                }
                return rows, processed_dates, set()

            if isinstance(payload, dict):
                rows = payload.get("rows", [])
                processed_dates = set(payload.get("processed_dates", []))
                if not processed_dates:
                    processed_dates = {
                        r["date"]
                        for r in rows
                        if isinstance(r, dict) and "date" in r and any(r.get(col) is not None for col in BASE_COLS)
                    }
                return rows, processed_dates, set(payload.get("processed_units", []))
        except json.JSONDecodeError:
            backup = PROGRESS_FILE + ".corrupt"
            try:
                os.replace(PROGRESS_FILE, backup)
                print(f"Progress file is corrupt; backed up to {backup}")
            except OSError:
                pass
    return [], set(), set()


def save_progress(rows: list[dict], processed_dates: set[str], processed_units: set[str]) -> None:
    tmp_file = PROGRESS_FILE + ".tmp"

    def json_default(obj):
        if hasattr(obj, "item"):
            return obj.item()
        raise TypeError(f"Object of type {type(obj).__name__} is not JSON serializable")

    payload = {
        "rows": rows,
        "processed_dates": sorted(processed_dates),
        "processed_units": sorted(processed_units),
    }
    with open(tmp_file, "w", encoding="utf-8") as f:
        json.dump(payload, f, default=json_default)
    os.replace(tmp_file, PROGRESS_FILE)


def rows_from_output_file(path: str) -> tuple[list[dict], set[str]]:
    if not path or not os.path.exists(path):
        return [], set()

    try:
        existing = pd.read_csv(path, parse_dates=["date"])
    except Exception:
        return [], set()

    if "date" not in existing.columns or any(col not in existing.columns for col in BASE_COLS):
        return [], set()

    existing = existing[["date"] + BASE_COLS].copy()
    existing = existing.dropna(subset=["date"])
    has_raw = existing[BASE_COLS].notna().any(axis=1)

    processed_dates = {
        date.strftime("%Y-%m-%d")
        for date in existing.loc[has_raw, "date"]
    }
    existing = existing.loc[has_raw]

    rows = []
    for _, row in existing.iterrows():
        rows.append(
            {
                "date": row["date"].strftime("%Y-%m-%d"),
                "gdelt_tone": float(row["gdelt_tone"]) if pd.notna(row["gdelt_tone"]) else None,
                "gdelt_goldstein": float(row["gdelt_goldstein"]) if pd.notna(row["gdelt_goldstein"]) else None,
                "gdelt_volume": int(row["gdelt_volume"]) if pd.notna(row["gdelt_volume"]) else None,
                "gdelt_events": int(row["gdelt_events"]) if pd.notna(row["gdelt_events"]) else None,
            }
        )

    return rows, processed_dates


def load_existing_rows_from_output() -> tuple[list[dict], set[str], datetime | None]:
    rows, processed_dates = rows_from_output_file(OUTPUT_FILE)
    if not rows:
        return [], set(), None
    last_date = pd.to_datetime([row["date"] for row in rows]).max()
    return rows, processed_dates, last_date


def normalize_country_cols(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    for col in cols:
        if col in df.columns:
            df[col] = df[col].astype("string").str.upper()
    return df


def filter_middle_east(df: pd.DataFrame) -> pd.DataFrame:
    actor_cols = ["Actor1CountryCode", "Actor2CountryCode"]
    geo_cols = ["Actor1Geo_CountryCode", "Actor2Geo_CountryCode", "ActionGeo_CountryCode"]
    df = normalize_country_cols(df, actor_cols + geo_cols)

    actor_mask = False
    for col in actor_cols:
        actor_mask = actor_mask | df[col].isin(MIDDLE_EAST_ACTOR_COUNTRIES)

    geo_mask = False
    for col in geo_cols:
        geo_mask = geo_mask | df[col].isin(MIDDLE_EAST_GEO_COUNTRIES)

    return df[actor_mask | geo_mask].copy()


def aggregate_chunk(df: pd.DataFrame, unit: DownloadUnit) -> pd.DataFrame:
    df_me = filter_middle_east(df)
    if df_me.empty:
        return pd.DataFrame()

    for col in ["AvgTone", "GoldsteinScale", "NumArticles"]:
        df_me[col] = pd.to_numeric(df_me[col], errors="coerce")

    if unit.mode in {"daily", "v2"}:
        df_me["_date"] = unit.start.strftime("%Y-%m-%d")
    else:
        parsed_day = pd.to_datetime(df_me["Day"].astype(str), format="%Y%m%d", errors="coerce")
        df_me["_date"] = parsed_day.dt.strftime("%Y-%m-%d")
        df_me = df_me[(parsed_day >= unit.start) & (parsed_day <= unit.end)]

    df_me = df_me.dropna(subset=["_date"])
    if df_me.empty:
        return pd.DataFrame()

    grouped = df_me.groupby("_date").agg(
        tone_sum=("AvgTone", "sum"),
        tone_count=("AvgTone", "count"),
        goldstein_sum=("GoldsteinScale", "sum"),
        goldstein_count=("GoldsteinScale", "count"),
        volume_sum=("NumArticles", "sum"),
        gdelt_events=("AvgTone", "size"),
    )
    return grouped


def rows_from_aggregates(aggregates: list[pd.DataFrame]) -> list[dict]:
    if not aggregates:
        return []

    combined = pd.concat(aggregates).groupby(level=0).sum(numeric_only=True)
    rows = []
    for date_str, row in combined.sort_index().iterrows():
        tone = row["tone_sum"] / row["tone_count"] if row["tone_count"] else None
        goldstein = row["goldstein_sum"] / row["goldstein_count"] if row["goldstein_count"] else None
        volume = row["volume_sum"] if pd.notna(row["volume_sum"]) else None
        rows.append(
            {
                "date": date_str,
                "gdelt_tone": float(tone) if tone is not None else None,
                "gdelt_goldstein": float(goldstein) if goldstein is not None else None,
                "gdelt_volume": int(volume) if volume is not None else None,
                "gdelt_events": int(row["gdelt_events"]),
            }
        )
    return rows


def download_and_filter_unit(unit: DownloadUnit) -> list[dict] | None:
    try:
        resp = requests.get(unit.url, headers=HEADERS, timeout=90, stream=True)
        if resp.status_code == 404:
            return []
        resp.raise_for_status()

        aggregates: list[pd.DataFrame] = []
        with zipfile.ZipFile(io.BytesIO(resp.content)) as zf:
            csv_names = [name for name in zf.namelist() if not name.endswith("/")]
            if unit.mode == "v2":
                names = GDELT_V2_COLS
                usecols = READ_USECOLS_V2
            else:
                names = GDELT_COLS if unit.mode == "daily" else GDELT_COLS[:-1]
                usecols = READ_USECOLS
            for csv_name in csv_names:
                with zf.open(csv_name) as f:
                    chunks = pd.read_csv(
                        f,
                        sep="\t",
                        header=None,
                        names=names,
                        usecols=usecols,
                        on_bad_lines="skip",
                        dtype=str,
                        chunksize=CHUNKSIZE,
                        low_memory=False,
                    )
                    for chunk in chunks:
                        aggregated = aggregate_chunk(chunk, unit)
                        if not aggregated.empty:
                            aggregates.append(aggregated)

        return rows_from_aggregates(aggregates)
    except zipfile.BadZipFile:
        return []
    except Exception as exc:
        print(f"\n      WARNING {type(exc).__name__}: {str(exc)[:120]}")
        return None


def download_and_filter(date: datetime) -> dict | None:
    """Backward-compatible helper for callers that expect a single daily row."""
    unit = DownloadUnit(
        unit_id=f"daily:{date.strftime('%Y%m%d')}",
        url=f"{EVENTS_BASE_URL}/{date.strftime('%Y%m%d')}.export.CSV.zip",
        start=pd.Timestamp(date.date()),
        end=pd.Timestamp(date.date()),
        mode="daily",
    )
    rows = download_and_filter_unit(unit)
    return rows[0] if rows else None


def has_gdelt_v2_files(day: pd.Timestamp) -> bool:
    # Two cheap probes avoid issuing 96 requests for known full-day outages.
    for hour in (0, 12):
        stamp = day.strftime("%Y%m%d") + f"{hour:02d}0000"
        url = f"{GDELT_V2_BASE_URL}/{stamp}.export.CSV.zip"
        try:
            resp = requests.head(url, headers=HEADERS, timeout=20, allow_redirects=True)
        except requests.RequestException:
            continue
        if resp.status_code == 200:
            return True
    return False


def download_and_filter_v2_day(day: pd.Timestamp) -> dict | None:
    if day < GDELT_V2_START:
        return None
    if not has_gdelt_v2_files(day):
        return None

    rows = []
    for minute_offset in range(0, 24 * 60, 15):
        stamp_dt = day + pd.Timedelta(minutes=minute_offset)
        stamp = stamp_dt.strftime("%Y%m%d%H%M%S")
        unit = DownloadUnit(
            unit_id=f"v2:{stamp}",
            url=f"{GDELT_V2_BASE_URL}/{stamp}.export.CSV.zip",
            start=day,
            end=day,
            mode="v2",
        )
        unit_rows = download_and_filter_unit(unit)
        if unit_rows:
            rows.extend(unit_rows)

    if not rows:
        return None

    total_events = sum(row["gdelt_events"] for row in rows)
    if total_events <= 0:
        return None

    tone_sum = sum(row["gdelt_tone"] * row["gdelt_events"] for row in rows if row["gdelt_tone"] is not None)
    goldstein_sum = sum(row["gdelt_goldstein"] * row["gdelt_events"] for row in rows if row["gdelt_goldstein"] is not None)
    volume_sum = sum(row["gdelt_volume"] for row in rows if row["gdelt_volume"] is not None)
    return {
        "date": day.strftime("%Y-%m-%d"),
        "gdelt_tone": float(tone_sum / total_events),
        "gdelt_goldstein": float(goldstein_sum / total_events),
        "gdelt_volume": int(volume_sum),
        "gdelt_events": int(total_events),
    }


def fill_missing_business_days_with_v2(
    row_by_date: dict[str, dict],
    processed_dates: set[str],
    output_start: str,
    end: str,
) -> int:
    expected = pd.date_range(output_start, end, freq="B")
    missing_days = [
        day for day in expected
        if day.strftime("%Y-%m-%d") not in processed_dates and day >= GDELT_V2_START
    ]

    filled = 0
    for day in missing_days:
        date_str = day.strftime("%Y-%m-%d")
        print(f"V2 fallback {date_str} ...", end=" ", flush=True)
        row = download_and_filter_v2_day(day)
        if row:
            row_by_date[date_str] = row
            processed_dates.add(date_str)
            filled += 1
            print(f"{row['gdelt_events']} events")
        else:
            print("no v2 file/match")
    return filled


def build_output(all_rows: list[dict], start: str, output_start: str, end: str, output_file: str) -> pd.DataFrame:
    if not all_rows:
        raise RuntimeError("No GDELT rows available.")

    row_by_date = {row["date"]: row for row in all_rows}
    daily_df = pd.DataFrame(row_by_date.values())
    daily_df["date"] = pd.to_datetime(daily_df["date"])
    daily_df = daily_df.set_index("date").sort_index()

    daily_index = pd.date_range(start, end, freq="B")
    result = pd.DataFrame(index=daily_index)
    result.index.name = "date"
    result = result.join(daily_df, how="left")

    if "gdelt_tone" in result.columns:
        result["gdelt_tone_7d"] = result["gdelt_tone"].rolling(7, min_periods=1).mean()
        result["gdelt_tone_30d"] = result["gdelt_tone"].rolling(30, min_periods=1).mean()
        result["gdelt_tone_spike"] = (result["gdelt_tone"] < result["gdelt_tone_30d"] - 1).astype(int)

    if "gdelt_volume" in result.columns:
        result["gdelt_volume_7d"] = result["gdelt_volume"].rolling(7, min_periods=1).mean()

    if "gdelt_goldstein" in result.columns:
        result["gdelt_goldstein_7d"] = result["gdelt_goldstein"].rolling(7, min_periods=1).mean()

    result = result.loc[pd.Timestamp(output_start):]

    if os.path.exists(output_file):
        backup_path = output_file + f".backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        shutil.copy2(output_file, backup_path)

    tmp_out = output_file + ".tmp"
    result.to_csv(tmp_out)
    os.replace(tmp_out, output_file)
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--start", default=START_DATE)
    parser.add_argument("--output-start", default=None)
    parser.add_argument("--end", default=END_DATE)
    parser.add_argument("--output", default=OUTPUT_FILE)
    parser.add_argument("--progress", default=PROGRESS_FILE)
    parser.add_argument("--seed-output", default=None)
    parser.add_argument("--build-only", action="store_true")
    parser.add_argument(
        "--force-refresh",
        action="store_true",
        help="Ignore existing output/progress rows and recrawl all selected source ZIP files.",
    )
    parser.add_argument(
        "--sleep",
        type=float,
        default=0.2,
        help="Pause between source ZIP downloads.",
    )
    parser.add_argument(
        "--v2-fill-missing",
        action="store_true",
        help="After GDELT 1.0 ZIP crawl, fill remaining business-day gaps with GDELT 2.0 15-minute export ZIP files.",
    )
    return parser.parse_args()


def main() -> None:
    global OUTPUT_FILE, PROGRESS_FILE

    args = parse_args()
    start = args.start
    output_start = args.output_start or args.start
    end = args.end
    OUTPUT_FILE = args.output
    PROGRESS_FILE = args.progress

    print("=" * 72)
    print("GDELT 1.0 crawler - raw ZIP/CSV download, no API")
    print(f"Source : {EVENTS_BASE_URL}/")
    print(f"Period : {start} -> {end}")
    print(f"Output : {output_start} -> {end}")
    print(f"File   : {OUTPUT_FILE}")
    print("=" * 72)

    if args.force_refresh:
        all_rows: list[dict] = []
        processed_dates: set[str] = set()
        processed_units: set[str] = set()
    else:
        saved_rows, processed_dates, processed_units = load_progress()
        all_rows = list(saved_rows)
        if saved_rows:
            print(
                f"Resume progress: {len(processed_units)} source files, "
                f"{len(processed_dates)} non-empty dates."
            )
        else:
            existing_rows, existing_processed_dates, last_existing_date = load_existing_rows_from_output()
            all_rows = existing_rows
            processed_dates = set(existing_processed_dates)
            processed_units = set()
            if last_existing_date is not None:
                print(
                    f"Seeded {len(existing_rows)} non-empty rows from existing output "
                    f"through {last_existing_date.strftime('%Y-%m-%d')}."
                )

        if args.seed_output:
            seed_rows, seed_processed_dates = rows_from_output_file(args.seed_output)
            row_by_date = {row["date"]: row for row in all_rows}
            for row in seed_rows:
                row_by_date.setdefault(row["date"], row)
            all_rows = list(row_by_date.values())
            processed_dates.update(seed_processed_dates)
            print(f"Seeded {len(seed_processed_dates)} non-empty dates from {args.seed_output}.")

    units = iter_download_units(start, end)
    if args.build_only:
        pending_units = []
    else:
        pending_units = [
            unit for unit in units
            if unit_has_output_dates(unit, output_start, end)
            and unit.unit_id not in processed_units
            and not unit_already_covered(unit, processed_dates, output_start, end)
        ]

    print(f"Source ZIP files total  : {len(units)}")
    print(f"Source ZIP files pending: {len(pending_units)}")

    row_by_date = {row["date"]: row for row in all_rows}
    for i, unit in enumerate(pending_units, 1):
        print(f"[{i:4d}/{len(pending_units)}] {unit.unit_id} ...", end=" ", flush=True)
        rows = download_and_filter_unit(unit)

        if rows is None:
            print("failed")
            continue

        for row in rows:
            row_date = pd.Timestamp(row["date"])
            if pd.Timestamp(start) <= row_date <= pd.Timestamp(end):
                row_by_date[row["date"]] = row
                processed_dates.add(row["date"])

        processed_units.add(unit.unit_id)
        all_rows = list(row_by_date.values())

        if rows:
            events = sum(row["gdelt_events"] for row in rows)
            print(f"{len(rows)} days, {events} events")
        else:
            print("no matching events or missing file")

        if i % 10 == 0:
            save_progress(all_rows, processed_dates, processed_units)
        time.sleep(args.sleep)

    if args.v2_fill_missing:
        filled = fill_missing_business_days_with_v2(row_by_date, processed_dates, output_start, end)
        all_rows = list(row_by_date.values())
        print(f"V2 fallback filled {filled} business days.")

    save_progress(all_rows, processed_dates, processed_units)

    print(f"\nBuilding output from {len(all_rows)} non-empty daily rows...")
    result = build_output(all_rows, start, output_start, end, OUTPUT_FILE)

    miss = int(result.isnull().sum().sum())
    total = result.shape[0] * result.shape[1]
    nonempty = int(result[BASE_COLS].notna().any(axis=1).sum())
    coverage = result.assign(year=result.index.year, has=result[BASE_COLS].notna().any(axis=1)).groupby("year")["has"].agg(["count", "sum"])

    print(f"\nSaved   : {OUTPUT_FILE}")
    print(f"Shape   : {result.shape[0]} rows x {result.shape[1]} cols")
    print(f"Nonempty: {nonempty}/{result.shape[0]} business days")
    print(f"Missing : {miss}/{total} ({miss / total * 100:.1f}%)")
    print("\nCoverage by year:")
    print(coverage.to_string())

    if os.path.exists(PROGRESS_FILE):
        os.remove(PROGRESS_FILE)
    print("\nDone.")


if __name__ == "__main__":
    main()
