"""
Process raw FEMA disaster declarations data into analysis-ready datasets.

Inputs:  data/raw/disaster_declarations_2004_2024.csv
Outputs:
  - data/processed/disasters_cleaned.csv
  - data/processed/disasters_county_monthly.csv
  - data/processed/disasters_county_quarterly.csv
  - data/processed/disasters_state_monthly.csv
  - data/processed/disasters_national_monthly.csv

Usage:
    python src/data/process_disasters.py
"""

import logging
from pathlib import Path

import numpy as np
import pandas as pd

try:
    from src.utils.config import (
        DATA_RAW,
        DATA_PROCESSED,
        DISASTER_TYPE_MAP,
        ANALYSIS_START_YEAR,
        ANALYSIS_END_YEAR,
    )
except ImportError:
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from src.utils.config import (
        DATA_RAW,
        DATA_PROCESSED,
        DISASTER_TYPE_MAP,
        ANALYSIS_START_YEAR,
        ANALYSIS_END_YEAR,
    )

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
)
logger = logging.getLogger(__name__)

DATE_COLS = ["declarationDate", "incidentBeginDate", "incidentEndDate", "lastRefresh"]


def load_raw_declarations() -> pd.DataFrame:
    """Load the raw declarations CSV."""
    path = DATA_RAW / "disaster_declarations_2004_2024.csv"
    logger.info(f"Loading raw data from {path}")
    df = pd.read_csv(
        path, dtype={"fipsStateCode": str, "fipsCountyCode": str}
    )
    logger.info(f"Loaded {len(df):,} raw records")
    return df


def clean_declarations(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean and standardize the declarations dataframe.

    Steps:
    1. Parse date columns
    2. Construct 5-digit county FIPS code
    3. Map incidentType to standardized disaster categories
    4. Filter to analysis time range
    5. Deduplicate (same disaster-county can appear for multiple programs)
    6. Create derived columns
    """
    df = df.copy()

    # 1. Parse dates
    for col in DATE_COLS:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors="coerce", utc=True)

    # 2. Construct FIPS
    df["fipsStateCode"] = df["fipsStateCode"].fillna("00").str.zfill(2)
    df["fipsCountyCode"] = df["fipsCountyCode"].fillna("000").str.zfill(3)
    df["county_fips"] = df["fipsStateCode"] + df["fipsCountyCode"]
    df["is_statewide"] = df["fipsCountyCode"] == "000"

    # 3. Standardize disaster type
    df["disaster_category"] = (
        df["incidentType"].map(DISASTER_TYPE_MAP).fillna("Other")
    )

    # 4. Filter time range
    df["declaration_year"] = df["declarationDate"].dt.year
    df["declaration_month"] = df["declarationDate"].dt.month
    df["declaration_quarter"] = df["declarationDate"].dt.quarter
    df["year_month"] = df["declarationDate"].dt.to_period("M")

    mask = (df["declaration_year"] >= ANALYSIS_START_YEAR) & (
        df["declaration_year"] <= ANALYSIS_END_YEAR
    )
    df = df[mask].copy()
    logger.info(
        f"After time filter ({ANALYSIS_START_YEAR}-{ANALYSIS_END_YEAR}): "
        f"{len(df):,} records"
    )

    # 5. Deduplicate
    pre_dedup = len(df)
    df = df.drop_duplicates(subset=["disasterNumber", "county_fips"])
    logger.info(f"Deduplicated: {pre_dedup:,} -> {len(df):,} records")

    # 6. Derived columns
    df["incident_duration_days"] = (
        (df["incidentEndDate"] - df["incidentBeginDate"]).dt.total_seconds()
        / 86400
    ).round(1)

    return df


def aggregate_county_monthly(df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate to county-month level."""
    grouped = (
        df.groupby(
            [
                "county_fips",
                "fipsStateCode",
                "state",
                "declaration_year",
                "declaration_month",
                "disaster_category",
            ]
        )
        .agg(
            disaster_count=("disasterNumber", "count"),
            unique_disasters=("disasterNumber", "nunique"),
            avg_duration_days=("incident_duration_days", "mean"),
        )
        .reset_index()
        .rename(
            columns={"declaration_year": "year", "declaration_month": "month"}
        )
    )
    return grouped


def aggregate_county_quarterly(df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate to county-quarter level."""
    grouped = (
        df.groupby(
            [
                "county_fips",
                "fipsStateCode",
                "state",
                "declaration_year",
                "declaration_quarter",
                "disaster_category",
            ]
        )
        .agg(
            disaster_count=("disasterNumber", "count"),
            unique_disasters=("disasterNumber", "nunique"),
            avg_duration_days=("incident_duration_days", "mean"),
        )
        .reset_index()
        .rename(
            columns={
                "declaration_year": "year",
                "declaration_quarter": "quarter",
            }
        )
    )
    return grouped


def aggregate_state_monthly(df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate to state-month level."""
    grouped = (
        df.groupby(
            [
                "fipsStateCode",
                "state",
                "declaration_year",
                "declaration_month",
                "disaster_category",
            ]
        )
        .agg(
            disaster_count=("disasterNumber", "count"),
            unique_disasters=("disasterNumber", "nunique"),
        )
        .reset_index()
        .rename(
            columns={"declaration_year": "year", "declaration_month": "month"}
        )
    )
    return grouped


def aggregate_national_monthly(df: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregate to national monthly level.
    This is the primary input for national-level time series modeling.
    """
    grouped = (
        df.groupby(
            ["declaration_year", "declaration_month", "disaster_category"]
        )
        .agg(
            disaster_count=("disasterNumber", "count"),
            unique_disasters=("disasterNumber", "nunique"),
            states_affected=("state", "nunique"),
            counties_affected=("county_fips", "nunique"),
        )
        .reset_index()
        .rename(
            columns={"declaration_year": "year", "declaration_month": "month"}
        )
    )

    # Create proper date column for time series
    grouped["date"] = pd.to_datetime(
        grouped[["year", "month"]].assign(day=1)
    )
    return grouped


def main():
    """Run the full processing pipeline."""
    DATA_PROCESSED.mkdir(parents=True, exist_ok=True)

    # Load and clean
    raw_df = load_raw_declarations()
    clean_df = clean_declarations(raw_df)

    # Save cleaned row-level data
    clean_path = DATA_PROCESSED / "disasters_cleaned.csv"
    clean_df.to_csv(clean_path, index=False)
    logger.info(f"Saved cleaned data ({len(clean_df):,} rows) to {clean_path}")

    # Aggregations
    county_monthly = aggregate_county_monthly(clean_df)
    county_monthly.to_csv(
        DATA_PROCESSED / "disasters_county_monthly.csv", index=False
    )
    logger.info(f"Saved county-monthly ({len(county_monthly):,} rows)")

    county_quarterly = aggregate_county_quarterly(clean_df)
    county_quarterly.to_csv(
        DATA_PROCESSED / "disasters_county_quarterly.csv", index=False
    )
    logger.info(f"Saved county-quarterly ({len(county_quarterly):,} rows)")

    state_monthly = aggregate_state_monthly(clean_df)
    state_monthly.to_csv(
        DATA_PROCESSED / "disasters_state_monthly.csv", index=False
    )
    logger.info(f"Saved state-monthly ({len(state_monthly):,} rows)")

    national_monthly = aggregate_national_monthly(clean_df)
    national_monthly.to_csv(
        DATA_PROCESSED / "disasters_national_monthly.csv", index=False
    )
    logger.info(f"Saved national-monthly ({len(national_monthly):,} rows)")

    # Summary
    logger.info("\n" + "=" * 60)
    logger.info("PROCESSING SUMMARY")
    logger.info("=" * 60)
    logger.info(
        f"Date range: {clean_df['declarationDate'].min()} to "
        f"{clean_df['declarationDate'].max()}"
    )
    logger.info(f"Unique disasters: {clean_df['disasterNumber'].nunique()}")
    logger.info(f"Unique counties: {clean_df['county_fips'].nunique()}")
    logger.info(
        f"States represented: {clean_df['state'].nunique()}"
    )
    logger.info(
        f"\nDisaster categories:\n"
        f"{clean_df['disaster_category'].value_counts().to_string()}"
    )


if __name__ == "__main__":
    main()
