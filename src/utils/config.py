"""Project-wide configuration constants."""

from pathlib import Path

# === PATHS ===
PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_RAW = PROJECT_ROOT / "data" / "raw"
DATA_PROCESSED = PROJECT_ROOT / "data" / "processed"
DATA_EXTERNAL = PROJECT_ROOT / "data" / "external"
DATA_INTERIM = PROJECT_ROOT / "data" / "interim"
REPORTS_FIGURES = PROJECT_ROOT / "reports" / "figures"
MODELS_DIR = PROJECT_ROOT / "models"

# === OPENFEMA API ===
FEMA_BASE_URL = "https://www.fema.gov/api/open/v2"
FEMA_DECLARATIONS_ENDPOINT = f"{FEMA_BASE_URL}/DisasterDeclarationsSummaries"
FEMA_HA_OWNERS_ENDPOINT = f"{FEMA_BASE_URL}/HousingAssistanceOwners"
FEMA_HA_RENTERS_ENDPOINT = f"{FEMA_BASE_URL}/HousingAssistanceRenters"

FEMA_NFIP_CLAIMS_ENDPOINT = f"{FEMA_BASE_URL}/FimaNfipClaims"
FEMA_NFIP_POLICIES_ENDPOINT = f"{FEMA_BASE_URL}/FimaNfipPolicies"

FEMA_PAGE_SIZE = 1000
FEMA_MAX_PAGE_SIZE = 10000

# === NFIP CLAIMS: COLUMNS TO SELECT ===
NFIP_CLAIMS_SELECT = [
    "yearOfLoss", "dateOfLoss", "state", "countyCode", "reportedZipCode",
    "occupancyType", "amountPaidOnBuildingClaim",
    "amountPaidOnContentsClaim", "amountPaidOnIncreasedCostOfComplianceClaim",
    "totalBuildingInsuranceCoverage", "totalContentsInsuranceCoverage",
    "buildingDamageAmount", "contentsDamageAmount",
    "ratedFloodZone", "postFIRMConstructionIndicator",
    "primaryResidenceIndicator", "numberOfFloorsInTheInsuredBuilding",
    "causeOfDamage", "policyCount",
]

# === NFIP POLICIES: COLUMNS TO SELECT ===
NFIP_POLICIES_SELECT = [
    "propertyState", "countyCode", "reportedZipCode",
    "totalInsurancePremiumOfThePolicy", "totalBuildingInsuranceCoverage",
    "totalContentsInsuranceCoverage", "policyEffectiveDate",
    "policyTerminationDate", "occupancyType",
    "ratedFloodZone", "postFIRMConstructionIndicator",
    "primaryResidenceIndicator", "policyCount",
]

# === CENSUS ACS ===
CENSUS_ACS_YEAR = 2022
CENSUS_VARIABLES = {
    "DP05_0001E": "total_population",
    "DP05_0018E": "median_age",
    "DP03_0062E": "median_household_income",
    "DP03_0063E": "mean_household_income",
    "DP04_0089E": "median_home_value",
    "DP04_0046E": "owner_occupied_units",
    "DP04_0047E": "renter_occupied_units",
    "DP02_0068PE": "pct_bachelors_degree",
    "DP03_0009PE": "unemployment_rate",
    "DP04_0002PE": "pct_occupied_housing",
}

# === FRED MACRO SERIES ===
FRED_BASE_CSV_URL = "https://fred.stlouisfed.org/graph/fredgraph.csv"
FRED_SERIES = {
    "FEDFUNDS": "Federal Funds Effective Rate",
    "CPIAUCSL": "Consumer Price Index (All Urban)",
    "CSUSHPINSA": "Case-Shiller Home Price Index (National)",
    "MORTGAGE30US": "30-Year Fixed Mortgage Rate",
}

# === ANALYSIS PARAMETERS ===
ANALYSIS_START_YEAR = 2004
ANALYSIS_END_YEAR = 2024
FORECAST_HORIZON_YEARS = 5

# === DISASTER TYPE MAPPING ===
DISASTER_TYPE_MAP = {
    "Flood": "Flood",
    "Hurricane": "Hurricane",
    "Severe Storm(s)": "Severe Storm",
    "Severe Storm": "Severe Storm",
    "Fire": "Wildfire",
    "Tornado": "Tornado",
    "Earthquake": "Earthquake",
    "Snow": "Winter Storm",
    "Ice Storm": "Winter Storm",
    "Severe Ice Storm": "Winter Storm",
    "Freezing": "Winter Storm",
    "Coastal Storm": "Coastal Storm",
    "Typhoon": "Hurricane",
    "Mud/Landslide": "Landslide",
    "Drought": "Drought",
    "Volcanic Eruption": "Volcanic",
    "Toxic Substances": "Other",
    "Dam/Levee Break": "Other",
    "Fishing Losses": "Other",
    "Biological": "Other",
    "Terrorist": "Other",
    "Human Cause": "Other",
    "Chemical": "Other",
    "Other": "Other",
}

# === FEMA REGIONS ===
FEMA_REGIONS = {
    1: ["09", "23", "25", "33", "44", "50"],
    2: ["34", "36", "72", "78"],
    3: ["10", "11", "24", "42", "51", "54"],
    4: ["01", "12", "13", "21", "28", "37", "45", "47"],
    5: ["17", "18", "26", "27", "39", "55"],
    6: ["05", "22", "35", "40", "48"],
    7: ["19", "20", "29", "31"],
    8: ["08", "30", "38", "46", "49", "56"],
    9: ["04", "06", "15", "32"],
    10: ["02", "16", "41", "53"],
}
