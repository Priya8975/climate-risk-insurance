"""
Process NFIP, Housing Assistance, Census, and FRED data into a
county-year panel dataset for GLM modeling.

Inputs:
  - data/raw/nfip_claims_2004_2024.csv
  - data/raw/nfip_policies_county_year.csv (if available)
  - data/raw/housing_assistance_owners.csv
  - data/processed/disasters_cleaned.csv
  - data/external/census_acs_county_2022.csv
  - data/external/fred_macro_indicators.csv

Outputs:
  - data/processed/nfip_claims_county_year.csv
  - data/processed/ha_county_year.csv
  - data/processed/disasters_county_year.csv
  - data/processed/county_year_panel.csv
  - data/processed/county_year_panel_glm_ready.csv

Usage:
    python src/data/process_insurance_data.py
"""

import logging
from pathlib import Path

import numpy as np
import pandas as pd

try:
    from src.utils.config import (
        DATA_RAW,
        DATA_PROCESSED,
        DATA_EXTERNAL,
        CENSUS_ACS_YEAR,
        FEMA_REGIONS,
        ANALYSIS_START_YEAR,
        ANALYSIS_END_YEAR,
    )
except ImportError:
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from src.utils.config import (
        DATA_RAW,
        DATA_PROCESSED,
        DATA_EXTERNAL,
        CENSUS_ACS_YEAR,
        FEMA_REGIONS,
        ANALYSIS_START_YEAR,
        ANALYSIS_END_YEAR,
    )

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s"
)
logger = logging.getLogger(__name__)


# ============================================================================
# STEP 1: CLEAN & AGGREGATE NFIP CLAIMS
# ============================================================================


def process_nfip_claims() -> pd.DataFrame:
    """
    Clean raw NFIP claims and aggregate to county-year level.

    Returns county-year DataFrame with claim metrics.
    """
    logger.info("Processing NFIP claims...")
    path = DATA_RAW / "nfip_claims_2004_2024.csv"
    df = pd.read_csv(path, dtype={"countyCode": str, "reportedZipCode": str})
    logger.info(f"Loaded {len(df):,} raw NFIP claims")

    # Standardize county FIPS to 5 digits
    df["countyCode"] = df["countyCode"].fillna("00000").str.zfill(5)

    # Convert dollar columns to numeric
    dollar_cols = [
        "amountPaidOnBuildingClaim",
        "amountPaidOnContentsClaim",
        "amountPaidOnIncreasedCostOfComplianceClaim",
        "totalBuildingInsuranceCoverage",
        "totalContentsInsuranceCoverage",
        "buildingDamageAmount",
        "contentsDamageAmount",
    ]
    for col in dollar_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)

    # Derived columns
    df["total_claim_paid"] = (
        df["amountPaidOnBuildingClaim"]
        + df["amountPaidOnContentsClaim"]
        + df.get("amountPaidOnIncreasedCostOfComplianceClaim", 0)
    )
    df["total_coverage"] = (
        df["totalBuildingInsuranceCoverage"]
        + df["totalContentsInsuranceCoverage"]
    )
    df["total_damage"] = (
        df["buildingDamageAmount"] + df["contentsDamageAmount"]
    )

    # Loss ratio (claim / coverage)
    df["loss_ratio"] = np.where(
        df["total_coverage"] > 0,
        df["total_claim_paid"] / df["total_coverage"],
        0,
    )

    # Filter valid records
    df = df[
        (df["countyCode"] != "00000")
        & (df["yearOfLoss"] >= ANALYSIS_START_YEAR)
        & (df["yearOfLoss"] <= ANALYSIS_END_YEAR)
    ].copy()
    logger.info(f"After filtering: {len(df):,} claims")

    # Aggregate to county-year
    agg = (
        df.groupby(["countyCode", "yearOfLoss"])
        .agg(
            claim_count=("total_claim_paid", "size"),
            total_claims_paid=("total_claim_paid", "sum"),
            avg_claim_severity=("total_claim_paid", "mean"),
            median_claim_severity=("total_claim_paid", "median"),
            max_claim_severity=("total_claim_paid", "max"),
            total_damage_reported=("total_damage", "sum"),
            avg_loss_ratio=("loss_ratio", "mean"),
            avg_coverage=("total_coverage", "mean"),
            pct_primary_residence=(
                "primaryResidenceIndicator",
                lambda x: (x == "Y").mean() if x.dtype == object else 0,
            ),
        )
        .reset_index()
        .rename(columns={"countyCode": "county_fips", "yearOfLoss": "year"})
    )

    logger.info(f"NFIP claims aggregated: {len(agg):,} county-year rows")
    return agg


# ============================================================================
# STEP 2: PROCESS HOUSING ASSISTANCE
# ============================================================================


def process_housing_assistance() -> pd.DataFrame:
    """
    Merge Housing Assistance with disaster declarations,
    then aggregate to county-year level.
    """
    logger.info("Processing Housing Assistance data...")

    # Load HA data
    ha = pd.read_csv(DATA_RAW / "housing_assistance_owners.csv")
    logger.info(f"Loaded {len(ha):,} HA records")

    # Load disasters for joining
    disasters = pd.read_csv(
        DATA_PROCESSED / "disasters_cleaned.csv",
        dtype={"county_fips": str, "fipsStateCode": str},
    )

    # Convert HA numeric columns
    ha_numeric = [
        "totalInspected",
        "totalDamage",
        "totalApprovedIhpAmount",
        "repairReplaceAmount",
        "rentalAmount",
        "otherNeedsAmount",
        "approvedForFemaAssistance",
        "validRegistrations",
    ]
    for col in ha_numeric:
        if col in ha.columns:
            ha[col] = pd.to_numeric(ha[col], errors="coerce").fillna(0)

    # Get unique disaster -> county_fips + year mapping
    disaster_map = (
        disasters[["disasterNumber", "state", "county_fips", "declaration_year"]]
        .drop_duplicates(subset=["disasterNumber", "state"])
    )

    # Join HA with disaster metadata
    ha_merged = ha.merge(
        disaster_map,
        on=["disasterNumber", "state"],
        how="inner",
    )
    logger.info(f"HA merged with disasters: {len(ha_merged):,} records")

    # Aggregate to county-year
    agg = (
        ha_merged.groupby(["county_fips", "declaration_year"])
        .agg(
            ha_total_inspected=("totalInspected", "sum"),
            ha_total_damage=("totalDamage", "sum"),
            ha_total_approved=("totalApprovedIhpAmount", "sum"),
            ha_repair_replace=("repairReplaceAmount", "sum"),
            ha_rental_amount=("rentalAmount", "sum"),
            ha_valid_registrations=("validRegistrations", "sum"),
            ha_approved_count=("approvedForFemaAssistance", "sum"),
        )
        .reset_index()
        .rename(columns={"declaration_year": "year"})
    )

    logger.info(f"HA aggregated: {len(agg):,} county-year rows")
    return agg


# ============================================================================
# STEP 3: AGGREGATE DISASTERS TO COUNTY-YEAR
# ============================================================================


def process_disasters_county_year() -> pd.DataFrame:
    """
    Create county-year disaster exposure features from Module 1 data.
    """
    logger.info("Aggregating disasters to county-year...")

    df = pd.read_csv(
        DATA_PROCESSED / "disasters_cleaned.csv",
        dtype={"county_fips": str, "fipsStateCode": str},
        parse_dates=["declarationDate"],
    )

    # Filter out statewide
    df = df[~df["is_statewide"]].copy()

    # Aggregate to county-year
    agg = (
        df.groupby(["county_fips", "fipsStateCode", "state", "declaration_year"])
        .agg(
            total_disasters=("disasterNumber", "nunique"),
            disaster_types=("disaster_category", "nunique"),
            avg_incident_duration=("incident_duration_days", "mean"),
            max_incident_duration=("incident_duration_days", "max"),
        )
        .reset_index()
        .rename(columns={"declaration_year": "year"})
    )

    # Add per-type counts
    type_counts = (
        df.groupby(["county_fips", "declaration_year", "disaster_category"])
        .size()
        .unstack(fill_value=0)
        .reset_index()
        .rename(columns={"declaration_year": "year"})
    )

    # Standardize type column names
    type_rename = {}
    for col in type_counts.columns:
        if col not in ["county_fips", "year"]:
            type_rename[col] = f"{col.lower().replace(' ', '_')}_count"
    type_counts = type_counts.rename(columns=type_rename)

    agg = agg.merge(type_counts, on=["county_fips", "year"], how="left")

    logger.info(f"Disaster county-year: {len(agg):,} rows")
    return agg


# ============================================================================
# STEP 4: LOAD EXTERNAL DATA
# ============================================================================


def load_census() -> pd.DataFrame:
    """Load Census ACS county demographics."""
    path = DATA_EXTERNAL / f"census_acs_county_{CENSUS_ACS_YEAR}.csv"
    df = pd.read_csv(path, dtype={"county_fips": str})
    logger.info(f"Loaded Census: {len(df):,} counties")
    return df


def load_fred_annual() -> pd.DataFrame:
    """
    Load FRED macro data and aggregate to annual averages.
    """
    path = DATA_EXTERNAL / "fred_macro_indicators.csv"
    df = pd.read_csv(path, parse_dates=["date"])
    df["year"] = df["date"].dt.year

    # Annual averages
    numeric_cols = [c for c in df.columns if c not in ["date", "year"]]
    annual = df.groupby("year")[numeric_cols].mean().reset_index()

    # Filter to analysis range
    annual = annual[
        (annual["year"] >= ANALYSIS_START_YEAR)
        & (annual["year"] <= ANALYSIS_END_YEAR)
    ]

    logger.info(f"FRED annual: {len(annual)} years, cols: {list(annual.columns)}")
    return annual


def load_nfip_policies() -> pd.DataFrame:
    """Load NFIP policies county-year aggregates if available."""
    path = DATA_RAW / "nfip_policies_county_year.csv"
    if not path.exists():
        logger.warning(
            "NFIP policies file not found. "
            "Panel will be built without premium data."
        )
        return pd.DataFrame()

    df = pd.read_csv(path, dtype={"countyCode": str})
    df = df.rename(columns={"countyCode": "county_fips", "propertyState": "state"})
    logger.info(f"Loaded NFIP policies: {len(df):,} county-year rows")
    return df


# ============================================================================
# STEP 5: BUILD PANEL
# ============================================================================


def build_panel(
    disasters: pd.DataFrame,
    claims: pd.DataFrame,
    policies: pd.DataFrame,
    ha: pd.DataFrame,
    census: pd.DataFrame,
    fred: pd.DataFrame,
) -> pd.DataFrame:
    """
    Merge all data into a single county-year panel.

    Spine: disaster county-year records.
    Left-join everything else onto it.
    """
    logger.info("Building county-year panel...")

    panel = disasters.copy()

    # Join NFIP claims
    panel = panel.merge(claims, on=["county_fips", "year"], how="left")

    # Join NFIP policies (if available)
    if not policies.empty:
        policy_cols = [
            "county_fips", "year", "avg_premium", "median_premium",
            "total_premium", "total_building_coverage",
            "total_contents_coverage", "policy_count", "n_records",
        ]
        available = [c for c in policy_cols if c in policies.columns]
        panel = panel.merge(
            policies[available], on=["county_fips", "year"], how="left"
        )

    # Join Housing Assistance
    panel = panel.merge(ha, on=["county_fips", "year"], how="left")

    # Join Census (static — same across years)
    census_cols = [c for c in census.columns if c != "NAME"]
    panel = panel.merge(census[census_cols], on="county_fips", how="left")

    # Join FRED (by year)
    panel = panel.merge(fred, on="year", how="left")

    # Fill NaN insurance/claims values with 0
    fill_zero_cols = [
        "claim_count", "total_claims_paid", "avg_claim_severity",
        "median_claim_severity", "ha_total_damage", "ha_total_approved",
    ]
    for col in fill_zero_cols:
        if col in panel.columns:
            panel[col] = panel[col].fillna(0)

    logger.info(
        f"Panel built: {len(panel):,} rows, {panel.shape[1]} columns, "
        f"{panel['county_fips'].nunique()} counties"
    )
    return panel


# ============================================================================
# STEP 6: FEATURE ENGINEERING
# ============================================================================


def engineer_features(panel: pd.DataFrame) -> pd.DataFrame:
    """
    Create derived features for GLM modeling.
    """
    logger.info("Engineering GLM features...")
    df = panel.copy()

    # --- FEMA region mapping ---
    state_to_region = {}
    for region, states in FEMA_REGIONS.items():
        for s in states:
            state_to_region[s] = region
    df["fema_region"] = df["fipsStateCode"].map(state_to_region).fillna(0).astype(int)

    # --- Disaster exposure features ---
    # Cumulative disasters (rolling 3-year window per county)
    df = df.sort_values(["county_fips", "year"])
    df["cum_disasters_3yr"] = (
        df.groupby("county_fips")["total_disasters"]
        .rolling(window=3, min_periods=1)
        .sum()
        .reset_index(level=0, drop=True)
    )

    # Disaster exposure index (weighted)
    df["disaster_exposure_index"] = (
        df["total_disasters"] * 1.0
        + df.get("flood_count", pd.Series(0)) * 0.5
        + df.get("hurricane_count", pd.Series(0)) * 1.5
        + df.get("wildfire_count", pd.Series(0)) * 1.0
    )

    # --- Insurance features ---
    if "avg_premium" in df.columns:
        # Premium change YoY
        df["prev_year_premium"] = df.groupby("county_fips")["avg_premium"].shift(1)
        df["premium_change_yoy"] = np.where(
            df["prev_year_premium"] > 0,
            (df["avg_premium"] - df["prev_year_premium"]) / df["prev_year_premium"],
            np.nan,
        )
        # Premium surge flag (>25% increase)
        df["premium_surge_flag"] = (df["premium_change_yoy"] > 0.25).astype(int)

        # Claims per policy
        if "policy_count" in df.columns:
            df["claims_per_policy"] = np.where(
                df["policy_count"] > 0,
                df["claim_count"] / df["policy_count"],
                0,
            )
    else:
        # No premium data yet — create placeholder columns
        df["premium_change_yoy"] = np.nan
        df["premium_surge_flag"] = np.nan
        df["claims_per_policy"] = 0

    # --- Demographic features ---
    if "total_population" in df.columns and "median_home_value" in df.columns:
        df["damage_per_capita"] = np.where(
            df["total_population"] > 0,
            df.get("ha_total_damage", 0) / df["total_population"],
            0,
        )

        df["claims_per_capita"] = np.where(
            df["total_population"] > 0,
            df["claim_count"] / df["total_population"],
            0,
        )

    # --- Log transforms for skewed features ---
    for col in ["avg_claim_severity", "total_claims_paid", "ha_total_damage"]:
        if col in df.columns:
            df[f"log_{col}"] = np.log1p(df[col])

    # --- Lagged features (for predictive modeling) ---
    lag_cols = ["total_disasters", "claim_count", "avg_claim_severity", "avg_loss_ratio"]
    for col in lag_cols:
        if col in df.columns:
            df[f"{col}_lag1"] = df.groupby("county_fips")[col].shift(1)

    logger.info(f"Features engineered: {df.shape[1]} total columns")
    return df


# ============================================================================
# MAIN
# ============================================================================


def main():
    """Run full processing pipeline."""
    DATA_PROCESSED.mkdir(parents=True, exist_ok=True)

    # Step 1: NFIP Claims
    claims = process_nfip_claims()
    claims.to_csv(DATA_PROCESSED / "nfip_claims_county_year.csv", index=False)

    # Step 2: Housing Assistance
    ha = process_housing_assistance()
    ha.to_csv(DATA_PROCESSED / "ha_county_year.csv", index=False)

    # Step 3: Disasters
    disasters = process_disasters_county_year()
    disasters.to_csv(DATA_PROCESSED / "disasters_county_year.csv", index=False)

    # Step 4: External data
    census = load_census()
    fred = load_fred_annual()
    policies = load_nfip_policies()

    # Step 5: Build panel
    panel = build_panel(disasters, claims, policies, ha, census, fred)
    panel.to_csv(DATA_PROCESSED / "county_year_panel.csv", index=False)

    # Step 6: Feature engineering
    glm_ready = engineer_features(panel)
    glm_ready.to_csv(
        DATA_PROCESSED / "county_year_panel_glm_ready.csv", index=False
    )

    # Summary
    logger.info("\n" + "=" * 60)
    logger.info("PANEL DATASET SUMMARY")
    logger.info("=" * 60)
    logger.info(f"Dimensions: {glm_ready.shape}")
    logger.info(f"Counties: {glm_ready['county_fips'].nunique()}")
    logger.info(f"Year range: {glm_ready['year'].min()}-{glm_ready['year'].max()}")
    logger.info(f"Columns: {list(glm_ready.columns)}")

    has_premium = "avg_premium" in glm_ready.columns and glm_ready["avg_premium"].notna().any()
    if has_premium:
        surge = glm_ready["premium_surge_flag"]
        logger.info(f"Premium surge distribution:\n{surge.value_counts()}")
    else:
        logger.info("Premium data not yet available (policies still downloading)")

    logger.info(f"\nClaim severity stats:")
    logger.info(
        glm_ready["avg_claim_severity"]
        .describe()
        .to_string()
    )


if __name__ == "__main__":
    main()
