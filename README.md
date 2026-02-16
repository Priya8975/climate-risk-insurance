# Climate Risk & Property Insurance Affordability Crisis

**Predicting Insurance Market Instability: Identifying U.S. Counties at Risk of Becoming Uninsurable Due to Climate Disaster Exposure**

An end-to-end data science project analyzing 20 years of federal disaster data, 1.5M+ flood insurance claims, and macroeconomic indicators to quantify how climate disasters drive insurance costs — and predict which regions face affordability crises.

**Author:** Priya More

---

## Key Findings

- Natural disaster declarations have **increased 45%** over the 2004–2024 period, with clear seasonal patterns (June–November peak)
- Each additional flood event **increases claim severity by 28.6%** (Gamma GLM, p < 0.001)
- Inflation (CPI) roughly **doubles expected claim payouts** per unit increase
- FEMA Region 6 (TX, LA, AR, OK, NM) shows **154% higher claims** than the national baseline
- A logistic regression with lagged features predicts county-level claims surges with **AUC-ROC = 0.73**
- SARIMA(0,1,2)(0,1,1,12) outperformed Prophet for disaster forecasting (AIC = 3028)

---

## Modules

| Module | Focus Area | Methods | Status |
|--------|-----------|---------|--------|
| 1 | **Disaster Trend Analysis** | SARIMA, Prophet, Time Series Decomposition | Complete |
| 2 | **Insurance Claims Modeling** | Gamma GLM, Tweedie GLM, Logistic Regression | Complete |
| 3 | **Uninsurability Risk Classification** | Gradient Boosting, Random Forest, SHAP | In Progress |
| 4 | **Model Validation & Documentation** | Cross-validation, Sensitivity Analysis | Planned |

---

## Project Structure

```
├── data/
│   ├── raw/                → Original API downloads (not in repo)
│   ├── processed/          → Cleaned, merged datasets
│   └── external/           → Census ACS & FRED economic data
├── src/
│   ├── data/
│   │   ├── fetch_fema_data.py          → FEMA API data collection (disasters, claims, policies)
│   │   ├── fetch_external_data.py      → Census ACS + FRED downloads
│   │   ├── process_disasters.py        → Disaster data cleaning & aggregation
│   │   └── process_insurance_data.py   → County-year panel dataset construction
│   ├── models/
│   │   ├── disaster_time_series.py     → SARIMA & Prophet models
│   │   └── insurance_glms.py           → Gamma GLM, Tweedie GLM, Logistic Regression
│   └── utils/
│       └── config.py                   → Centralized configuration & constants
├── notebooks/
│   ├── 01_disaster_eda.ipynb           → Exploratory analysis & visualizations
│   ├── 02_disaster_time_series.ipynb   → Time series modeling results
│   └── 03_insurance_glm_results.ipynb  → GLM results & interpretation
├── models/                 → Model coefficients, metrics & diagnostics
├── reports/figures/        → Saved visualizations
└── requirements.txt
```

---

## Data Sources

| Source | Records | Description |
|--------|---------|-------------|
| [OpenFEMA — Disaster Declarations](https://www.fema.gov/api/open/v2/DisasterDeclarationsSummaries) | 41,363 | Federally declared disasters (2004–2024) |
| [OpenFEMA — NFIP Claims](https://www.fema.gov/api/open/v2/FimaNfipClaims) | 1,519,328 | Individual flood insurance claims |
| [OpenFEMA — NFIP Policies](https://www.fema.gov/api/open/v2/FimaNfipPolicies) | ~72M | Flood insurance policies (aggregated to county-year) |
| [OpenFEMA — Housing Assistance](https://www.fema.gov/api/open/v2/HousingAssistanceOwners) | 157,928 | FEMA individual assistance disbursements |
| [Census ACS 5-Year](https://api.census.gov/data/2022/acs/acs5/profile) | 3,222 counties | Demographics, income, housing characteristics |
| [FRED](https://fred.stlouisfed.org/) | 4 series | Fed Funds Rate, CPI, Case-Shiller HPI, 30-yr Mortgage Rate |

---

## Methodology

### Module 1: Disaster Trend Analysis

- Cleaned 45K disaster records → 41,363 analysis-ready records across 3,281 counties
- Created national, state, and county-level monthly aggregations
- Grid-searched **144 SARIMA parameter combinations**; best model: SARIMA(0,1,2)(0,1,1,12)
- Implemented expanding-window cross-validation for temporal forecasting
- Prophet model as baseline comparison

### Module 2: Insurance Claims Modeling

**Panel Dataset Construction:**
Built a county-year panel (25,415 rows × 65 features) by merging disaster counts, NFIP claims, housing assistance, Census demographics, and FRED economic indicators via sequential left joins.

**Claim Severity — Gamma GLM (log link):**
- Models positive claim amounts using the actuarial-standard Gamma distribution
- 10,907 county-years with positive claims; temporal train/test split (≤2021 / 2022–2024)
- 15 of 24 features statistically significant at p < 0.05
- Coefficients have multiplicative interpretation: exp(β) = effect multiplier

**Claim Severity — Tweedie GLM (log link):**
- Handles zero-inflated data (county-years with no claims) via compound Poisson-Gamma distribution
- Grid-searched variance power parameter; best converged model at p = 1.4
- Uses all 25,415 county-years without data exclusion

**Claims Surge Prediction — Logistic Regression:**
- Binary target: >50% year-over-year increase in county claim count
- All features lagged by 1 year for genuine out-of-sample prediction
- Standardized features for comparable odds ratios
- AUC-ROC: 0.687 (test), 0.728 ± 0.015 (5-fold stratified CV)
- Balanced class weights to handle class imbalance (~15% surge rate)

---

## Setup & Usage

### Prerequisites
- Python 3.9+
- ~500 MB disk space for data downloads

### Installation

```bash
git clone https://github.com/Priya8975/climate-risk-insurance.git
cd climate-risk-insurance
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### Running the Pipeline

```bash
# Module 1: Data Collection & Processing
python -m src.data.fetch_fema_data          # Fetch disaster declarations + housing assistance
python -m src.data.process_disasters        # Clean & aggregate disaster data

# Module 1: Time Series Modeling
python -m src.models.disaster_time_series   # Fit SARIMA + Prophet

# Module 2: Additional Data Collection
python -m src.data.fetch_external_data      # Fetch Census ACS + FRED data

# Module 2: Panel Construction & GLMs
python -m src.data.process_insurance_data   # Build county-year panel dataset
python -m src.models.insurance_glms         # Fit Gamma, Tweedie & Logistic models

# Explore Results
jupyter notebook notebooks/
```

---

## Selected Visualizations

Results notebooks include:
- Annual disaster trend analysis with seasonal decomposition
- State/county heatmaps and interactive choropleth maps
- GLM coefficient forest plots with confidence intervals
- FEMA region effect comparisons
- ROC and Precision-Recall curves for claims surge prediction
- Residual diagnostic panels (QQ, scale-location, residuals vs fitted)
- Top-risk county rankings

---

## Tech Stack

| Category | Tools |
|----------|-------|
| **Data Processing** | pandas, numpy, requests, tqdm |
| **Statistical Modeling** | statsmodels (GLMs), scikit-learn (Logistic Regression, CV) |
| **Time Series** | statsmodels SARIMAX, Prophet |
| **Visualization** | matplotlib, seaborn, plotly |
| **Environment** | Python 3.9, Jupyter, venv |

---

## License

This project is for educational and portfolio purposes. Data sourced from public U.S. government APIs (OpenFEMA, Census Bureau, FRED).
