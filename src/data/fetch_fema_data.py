"""
Fetch disaster data from OpenFEMA API.

Endpoints used:
  - DisasterDeclarationsSummaries (v2): declaration-level records with
    county FIPS, incident type, dates, and program flags.
  - HousingAssistanceOwners (v2): aggregated IA damage/assistance by
    county and zip code.

Both endpoints are public and require no API key.

Usage:
    python src/data/fetch_fema_data.py
"""

import json
import time
import logging
from pathlib import Path
from typing import Optional, List

import pandas as pd
import requests
from tqdm import tqdm

import numpy as np

try:
    from src.utils.config import (
        FEMA_DECLARATIONS_ENDPOINT,
        FEMA_HA_OWNERS_ENDPOINT,
        FEMA_NFIP_CLAIMS_ENDPOINT,
        FEMA_NFIP_POLICIES_ENDPOINT,
        FEMA_MAX_PAGE_SIZE,
        NFIP_CLAIMS_SELECT,
        NFIP_POLICIES_SELECT,
        DATA_RAW,
        ANALYSIS_START_YEAR,
    )
except ImportError:
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from src.utils.config import (
        FEMA_DECLARATIONS_ENDPOINT,
        FEMA_HA_OWNERS_ENDPOINT,
        FEMA_NFIP_CLAIMS_ENDPOINT,
        FEMA_NFIP_POLICIES_ENDPOINT,
        FEMA_MAX_PAGE_SIZE,
        NFIP_CLAIMS_SELECT,
        NFIP_POLICIES_SELECT,
        DATA_RAW,
        ANALYSIS_START_YEAR,
    )

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
)
logger = logging.getLogger(__name__)


def _request_with_retry(
    endpoint: str,
    params: dict,
    max_retries: int = 3,
    retry_delay: float = 2.0,
) -> dict:
    """Make GET request with exponential backoff retry."""
    for attempt in range(max_retries):
        try:
            resp = requests.get(endpoint, params=params, timeout=60)
            resp.raise_for_status()
            return resp.json()
        except (requests.RequestException, json.JSONDecodeError) as e:
            if attempt < max_retries - 1:
                wait = retry_delay * (2 ** attempt)
                logger.warning(
                    f"Request failed (attempt {attempt + 1}): {e}. "
                    f"Retrying in {wait}s..."
                )
                time.sleep(wait)
            else:
                logger.error(f"Request failed after {max_retries} attempts: {e}")
                raise


def fetch_paginated(
    endpoint: str,
    entity_name: str,
    filter_expr: Optional[str] = None,
    select_fields: Optional[List[str]] = None,
    page_size: int = FEMA_MAX_PAGE_SIZE,
    max_retries: int = 3,
    retry_delay: float = 2.0,
) -> pd.DataFrame:
    """
    Fetch all records from an OpenFEMA endpoint with pagination.

    Parameters
    ----------
    endpoint : str
        Full API URL.
    entity_name : str
        The JSON key that holds the data array.
    filter_expr : str, optional
        OData $filter expression.
    select_fields : list[str], optional
        List of field names to retrieve (reduces payload).
    page_size : int
        Records per request (max 10000).
    max_retries : int
        Number of retries on failure.
    retry_delay : float
        Base delay between retries (doubles each attempt).

    Returns
    -------
    pd.DataFrame
    """
    all_records = []
    skip = 0

    # Build initial params
    params = {
        "$top": page_size,
        "$skip": skip,
        "$format": "json",
        "$inlinecount": "allpages",
    }
    if filter_expr:
        params["$filter"] = filter_expr
    if select_fields:
        params["$select"] = ",".join(select_fields)

    # First request to get total count
    response = _request_with_retry(endpoint, params, max_retries, retry_delay)
    metadata = response.get("metadata", {})
    total_count = metadata.get("count", 0)

    if total_count == 0:
        logger.warning("API returned count=0. Will fetch until empty response.")
        total_count = float("inf")

    records = response.get(entity_name, [])
    all_records.extend(records)
    skip += page_size

    logger.info(f"Total records to fetch: {total_count}")

    # Paginate through remaining records
    with tqdm(
        total=total_count if total_count != float("inf") else None,
        initial=len(all_records),
        desc=f"Fetching {entity_name}",
    ) as pbar:
        while skip < total_count:
            params["$skip"] = skip
            params.pop("$inlinecount", None)

            response = _request_with_retry(
                endpoint, params, max_retries, retry_delay
            )
            records = response.get(entity_name, [])

            if not records:
                logger.info("No more records returned. Stopping.")
                break

            all_records.extend(records)
            pbar.update(len(records))
            skip += page_size

            # Politeness delay
            time.sleep(0.25)

    logger.info(f"Fetched {len(all_records)} total records from {entity_name}")
    return pd.DataFrame(all_records)


def fetch_disaster_declarations() -> pd.DataFrame:
    """
    Fetch FEMA Disaster Declarations Summaries from 2004 onward.

    Returns a DataFrame with one row per declaration-county combination.
    """
    filter_expr = (
        f"declarationDate gt '{ANALYSIS_START_YEAR}-01-01T00:00:00.000z'"
    )

    select_fields = [
        "disasterNumber",
        "state",
        "declarationType",
        "declarationDate",
        "fyDeclared",
        "incidentType",
        "declarationTitle",
        "incidentBeginDate",
        "incidentEndDate",
        "fipsStateCode",
        "fipsCountyCode",
        "placeCode",
        "designatedArea",
        "ihProgramDeclared",
        "iaProgramDeclared",
        "paProgramDeclared",
        "hmProgramDeclared",
        "lastRefresh",
    ]

    df = fetch_paginated(
        endpoint=FEMA_DECLARATIONS_ENDPOINT,
        entity_name="DisasterDeclarationsSummaries",
        filter_expr=filter_expr,
        select_fields=select_fields,
        page_size=FEMA_MAX_PAGE_SIZE,
    )

    return df


def fetch_housing_assistance_owners() -> pd.DataFrame:
    """
    Fetch Housing Assistance Program Data for Owners.

    This dataset uses county name strings and zip codes (no FIPS).
    We join on disasterNumber + state during processing.
    """
    df = fetch_paginated(
        endpoint=FEMA_HA_OWNERS_ENDPOINT,
        entity_name="HousingAssistanceOwners",
        page_size=FEMA_MAX_PAGE_SIZE,
    )
    return df


def fetch_nfip_claims() -> pd.DataFrame:
    """
    Fetch NFIP flood insurance claims from 2004 onward.

    ~1.5M records after filtering. Uses $select to reduce payload.
    Download takes ~15-20 minutes.
    """
    filter_expr = "yearOfLoss ge 2004"

    df = fetch_paginated(
        endpoint=FEMA_NFIP_CLAIMS_ENDPOINT,
        entity_name="FimaNfipClaims",
        filter_expr=filter_expr,
        select_fields=NFIP_CLAIMS_SELECT,
        page_size=FEMA_MAX_PAGE_SIZE,
    )
    return df


def fetch_nfip_policies_county_year_agg(
    years: range = range(2010, 2025),
) -> pd.DataFrame:
    """
    Fetch NFIP policies year-by-year and aggregate to county-year level.

    The full policies dataset has 72M+ records. Instead of downloading
    everything, we fetch one year at a time, immediately aggregate to
    county-year level (avg premium, policy count, total coverage),
    then free the raw records.

    Output: ~35K county-year rows (manageable).
    """
    all_agg = []

    for year in years:
        logger.info(f"Fetching NFIP policies for {year}...")
        filter_expr = (
            f"policyEffectiveDate ge '{year}-01-01T00:00:00.000z' "
            f"and policyEffectiveDate lt '{year + 1}-01-01T00:00:00.000z'"
        )

        try:
            df = fetch_paginated(
                endpoint=FEMA_NFIP_POLICIES_ENDPOINT,
                entity_name="FimaNfipPolicies",
                filter_expr=filter_expr,
                select_fields=NFIP_POLICIES_SELECT,
                page_size=FEMA_MAX_PAGE_SIZE,
            )
        except Exception as e:
            logger.error(f"Failed to fetch {year}: {e}. Skipping.")
            continue

        if df.empty:
            logger.warning(f"No data for {year}. Skipping.")
            continue

        # Convert numeric columns
        for col in [
            "totalInsurancePremiumOfThePolicy",
            "totalBuildingInsuranceCoverage",
            "totalContentsInsuranceCoverage",
            "policyCount",
        ]:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce")

        # Standardize county code
        if "countyCode" in df.columns:
            df["countyCode"] = df["countyCode"].astype(str).str.zfill(5)

        # Aggregate to county-year immediately
        agg = (
            df.groupby(["countyCode", "propertyState"])
            .agg(
                avg_premium=("totalInsurancePremiumOfThePolicy", "mean"),
                median_premium=("totalInsurancePremiumOfThePolicy", "median"),
                total_premium=("totalInsurancePremiumOfThePolicy", "sum"),
                total_building_coverage=(
                    "totalBuildingInsuranceCoverage", "sum"
                ),
                total_contents_coverage=(
                    "totalContentsInsuranceCoverage", "sum"
                ),
                policy_count=("policyCount", "sum"),
                n_records=("policyCount", "size"),
            )
            .reset_index()
        )
        agg["year"] = year
        all_agg.append(agg)

        logger.info(
            f"  {year}: {len(df):,} raw records -> "
            f"{len(agg):,} county aggregates"
        )
        del df  # Free memory

    if not all_agg:
        return pd.DataFrame()
    return pd.concat(all_agg, ignore_index=True)


def main():
    """Fetch all data and save to data/raw/."""
    DATA_RAW.mkdir(parents=True, exist_ok=True)

    # --- Disaster Declarations ---
    logger.info("=" * 60)
    logger.info("Fetching Disaster Declarations Summaries...")
    logger.info("=" * 60)
    declarations_df = fetch_disaster_declarations()
    output_path = DATA_RAW / "disaster_declarations_2004_2024.csv"
    declarations_df.to_csv(output_path, index=False)
    logger.info(f"Saved {len(declarations_df):,} declarations to {output_path}")

    # --- Housing Assistance Owners ---
    logger.info("=" * 60)
    logger.info("Fetching Housing Assistance Owners data...")
    logger.info("=" * 60)
    ha_owners_df = fetch_housing_assistance_owners()
    output_path = DATA_RAW / "housing_assistance_owners.csv"
    ha_owners_df.to_csv(output_path, index=False)
    logger.info(f"Saved {len(ha_owners_df):,} HA owner records to {output_path}")

    logger.info("=" * 60)
    logger.info("Data collection complete!")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
