"""
Fetch external datasets: Census ACS demographics and FRED macro indicators.

Census API: Public, no key required for basic queries.
FRED: Direct CSV download, no key required.

Usage:
    python src/data/fetch_external_data.py
"""

import logging
import time
from pathlib import Path

import pandas as pd
import requests

try:
    from src.utils.config import (
        CENSUS_ACS_YEAR,
        CENSUS_VARIABLES,
        FRED_BASE_CSV_URL,
        FRED_SERIES,
        DATA_EXTERNAL,
        ANALYSIS_START_YEAR,
    )
except ImportError:
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from src.utils.config import (
        CENSUS_ACS_YEAR,
        CENSUS_VARIABLES,
        FRED_BASE_CSV_URL,
        FRED_SERIES,
        DATA_EXTERNAL,
        ANALYSIS_START_YEAR,
    )

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s"
)
logger = logging.getLogger(__name__)


# ============================================================================
# CENSUS ACS
# ============================================================================


def fetch_census_acs_county(year: int = CENSUS_ACS_YEAR) -> pd.DataFrame:
    """
    Fetch county-level demographics from Census ACS 5-Year Data Profiles.

    Uses the public Census API (no key required for most queries).
    Returns ~3,200 rows (one per county) with demographic columns.
    """
    logger.info(f"Fetching Census ACS {year} 5-Year county data...")

    var_codes = list(CENSUS_VARIABLES.keys())
    var_string = ",".join(var_codes)

    url = (
        f"https://api.census.gov/data/{year}/acs/acs5/profile"
        f"?get=NAME,{var_string}"
        f"&for=county:*&in=state:*"
    )

    logger.info(f"URL: {url[:100]}...")

    resp = requests.get(url, timeout=120)
    resp.raise_for_status()
    data = resp.json()

    # First row is headers, rest is data
    headers = data[0]
    rows = data[1:]

    df = pd.DataFrame(rows, columns=headers)

    # Rename Census variable codes to readable names
    rename_map = {code: name for code, name in CENSUS_VARIABLES.items()}
    df = df.rename(columns=rename_map)

    # Construct 5-digit county FIPS
    df["county_fips"] = df["state"].str.zfill(2) + df["county"].str.zfill(3)

    # Convert numeric columns
    for col in CENSUS_VARIABLES.values():
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # Keep useful columns
    keep_cols = ["county_fips", "NAME"] + list(CENSUS_VARIABLES.values())
    df = df[[c for c in keep_cols if c in df.columns]]

    logger.info(f"Fetched {len(df):,} counties from Census ACS {year}")
    return df


# ============================================================================
# FRED MACRO INDICATORS
# ============================================================================


def fetch_fred_series(
    series_id: str,
    start_date: str = f"{ANALYSIS_START_YEAR}-01-01",
) -> pd.DataFrame:
    """
    Download a single FRED series as CSV.

    Returns DataFrame with columns: date, {series_id}
    """
    url = (
        f"{FRED_BASE_CSV_URL}"
        f"?bgcolor=%23e1e9f0&chart_type=line&drp=0"
        f"&fo=open%20sans&graph_bgcolor=%23ffffff"
        f"&id={series_id}&cosd={start_date}"
        f"&mode=fred&nber_recession_bars=on"
        f"&recession_bars=on&txtcolor=%23444444"
        f"&ts=12&tts=12&width=1168&height=450"
    )

    resp = requests.get(url, timeout=60)
    resp.raise_for_status()

    # Parse CSV from response text
    from io import StringIO
    df = pd.read_csv(StringIO(resp.text))

    # FRED CSV has columns: DATE, {series_id}
    if "DATE" in df.columns:
        df = df.rename(columns={"DATE": "date"})
    elif "observation_date" in df.columns:
        df = df.rename(columns={"observation_date": "date"})

    df["date"] = pd.to_datetime(df["date"])

    # Handle missing values (FRED uses "." for missing)
    if series_id in df.columns:
        df[series_id] = pd.to_numeric(df[series_id], errors="coerce")

    logger.info(
        f"FRED {series_id}: {len(df)} observations "
        f"({df['date'].min().date()} to {df['date'].max().date()})"
    )
    return df


def fetch_all_fred() -> pd.DataFrame:
    """
    Download all FRED series and merge into a single DataFrame.
    """
    logger.info("Fetching FRED macro indicators...")
    dfs = []

    for series_id, description in FRED_SERIES.items():
        logger.info(f"  Downloading {series_id} ({description})...")
        try:
            df = fetch_fred_series(series_id)
            dfs.append(df)
            time.sleep(0.5)  # Politeness delay
        except Exception as e:
            logger.warning(f"  Failed to fetch {series_id}: {e}")
            continue

    if not dfs:
        logger.error("No FRED series fetched successfully.")
        return pd.DataFrame()

    # Merge all series on date
    merged = dfs[0]
    for df in dfs[1:]:
        merged = merged.merge(df, on="date", how="outer")

    merged = merged.sort_values("date").reset_index(drop=True)

    # Forward-fill missing values (some series are weekly, some monthly)
    for col in merged.columns:
        if col != "date":
            merged[col] = merged[col].ffill()

    logger.info(
        f"FRED merged: {len(merged)} observations, "
        f"{len(merged.columns) - 1} series"
    )
    return merged


# ============================================================================
# MAIN
# ============================================================================


def main():
    """Fetch all external data and save."""
    DATA_EXTERNAL.mkdir(parents=True, exist_ok=True)

    # --- Census ACS ---
    logger.info("=" * 60)
    logger.info("CENSUS ACS DATA")
    logger.info("=" * 60)
    census_df = fetch_census_acs_county()
    output_path = DATA_EXTERNAL / f"census_acs_county_{CENSUS_ACS_YEAR}.csv"
    census_df.to_csv(output_path, index=False)
    logger.info(f"Saved {len(census_df):,} counties to {output_path}")

    # --- FRED ---
    logger.info("=" * 60)
    logger.info("FRED MACRO INDICATORS")
    logger.info("=" * 60)
    fred_df = fetch_all_fred()
    output_path = DATA_EXTERNAL / "fred_macro_indicators.csv"
    fred_df.to_csv(output_path, index=False)
    logger.info(f"Saved {len(fred_df):,} observations to {output_path}")

    logger.info("=" * 60)
    logger.info("External data collection complete!")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
