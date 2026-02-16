"""
Generalized Linear Models for Insurance Claims Severity.

Models:
    1. Gamma GLM (log link) — positive claim amounts only
    2. Tweedie GLM (log link) — handles zeros + positives

Usage:
    python src/models/insurance_glms.py
"""

import logging
import pickle
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    mean_absolute_error, mean_squared_error, r2_score,
    roc_auc_score, average_precision_score, confusion_matrix,
    classification_report, roc_curve, precision_recall_curve,
)
from sklearn.model_selection import KFold, StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler

try:
    from src.utils.config import DATA_PROCESSED, MODELS_DIR, REPORTS_FIGURES
except ImportError:
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from src.utils.config import DATA_PROCESSED, MODELS_DIR, REPORTS_FIGURES

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s"
)
logger = logging.getLogger(__name__)


# ============================================================================
# FEATURE SELECTION
# ============================================================================

# Features for Gamma GLM (positive claims only)
GAMMA_FEATURES = [
    # Disaster exposure
    "total_disasters",
    "flood_count",
    "hurricane_count",
    "severe_storm_count",
    "wildfire_count",
    "cum_disasters_3yr",
    "avg_incident_duration",
    # Demographics
    "total_population",
    "median_household_income",
    "median_home_value",
    "pct_occupied_housing",
    # Macro indicators
    "MORTGAGE30US",
    "CSUSHPINSA",
    "CPIAUCSL",
]

# Features for Tweedie GLM (includes zeros)
TWEEDIE_FEATURES = GAMMA_FEATURES.copy()

# Features for FEMA region dummies (generated dynamically)
FEMA_REGION_DUMMIES = [f"fema_region_{i}" for i in range(1, 11)]


# ============================================================================
# DATA PREPARATION
# ============================================================================


def prepare_gamma_data(df: pd.DataFrame) -> tuple:
    """
    Prepare data for Gamma GLM: filter to positive claims, select features,
    drop rows with missing values, and split train/test.

    Returns: (X_train, X_test, y_train, y_test, feature_names)
    """
    logger.info("Preparing data for Gamma GLM...")

    # Filter to positive claim severity
    pos = df[df["avg_claim_severity"] > 0].copy()
    logger.info(f"  Positive claim rows: {len(pos):,}")

    # Create FEMA region dummies (as float for statsmodels)
    pos = pd.get_dummies(pos, columns=["fema_region"], prefix="fema_region", dtype=float)

    # Select features (use available ones)
    available_features = [f for f in GAMMA_FEATURES if f in pos.columns]
    available_dummies = [f for f in pos.columns if f.startswith("fema_region_") and f != "fema_region_0"]
    all_features = available_features + available_dummies
    logger.info(f"  Features: {len(all_features)}")

    # Target
    y = pos["avg_claim_severity"].values.astype(float)

    # Feature matrix — ensure all float
    X = pos[all_features].astype(float).copy()

    # Drop rows with any missing values
    mask = X.notna().all(axis=1)
    X = X[mask]
    y = y[mask.values]
    logger.info(f"  After dropping NaN: {len(X):,} rows")

    # Scale large features for numerical stability
    X["total_population"] = X["total_population"] / 10000  # per 10K people
    X["median_household_income"] = X["median_household_income"] / 10000
    X["median_home_value"] = X["median_home_value"] / 100000
    X["CPIAUCSL"] = X["CPIAUCSL"] / 100
    X["CSUSHPINSA"] = X["CSUSHPINSA"] / 100

    feature_names = list(X.columns)

    # Train/test split: use last 3 years as test
    years = pos.loc[mask, "year"].values
    train_mask = years <= 2021
    test_mask = years > 2021

    X_train = X.values[train_mask]
    X_test = X.values[test_mask]
    y_train = y[train_mask]
    y_test = y[test_mask]

    logger.info(f"  Train: {len(X_train):,}, Test: {len(X_test):,}")
    logger.info(f"  Train mean severity: ${y_train.mean():,.0f}")
    logger.info(f"  Test mean severity: ${y_test.mean():,.0f}")

    return X_train, X_test, y_train, y_test, feature_names


def prepare_tweedie_data(df: pd.DataFrame) -> tuple:
    """
    Prepare data for Tweedie GLM: includes zero and positive claims.
    Uses avg_claim_severity as target (zero when no claims filed).

    The Tweedie distribution (1<p<2) is a compound Poisson-Gamma that
    naturally handles the point mass at zero (no claims) plus the
    right-skewed positive values (actual claim amounts).

    Returns: (X_train, X_test, y_train, y_test, feature_names)
    """
    logger.info("Preparing data for Tweedie GLM...")

    data = df.copy()

    # Create FEMA region dummies (as float for statsmodels)
    data = pd.get_dummies(data, columns=["fema_region"], prefix="fema_region", dtype=float)

    # Select features
    available_features = [f for f in TWEEDIE_FEATURES if f in data.columns]
    available_dummies = [f for f in data.columns if f.startswith("fema_region_") and f != "fema_region_0"]
    all_features = available_features + available_dummies

    # Target: avg_claim_severity (0 when no claims, positive otherwise)
    y = data["avg_claim_severity"].fillna(0).values.astype(float)

    # Feature matrix — ensure all float
    X = data[all_features].astype(float).copy()

    # Drop rows with missing features
    mask = X.notna().all(axis=1)
    X = X[mask]
    y = y[mask.values]
    logger.info(f"  After dropping NaN: {len(X):,} rows")

    # Scale features
    X["total_population"] = X["total_population"] / 10000
    X["median_household_income"] = X["median_household_income"] / 10000
    X["median_home_value"] = X["median_home_value"] / 100000
    X["CPIAUCSL"] = X["CPIAUCSL"] / 100
    X["CSUSHPINSA"] = X["CSUSHPINSA"] / 100

    feature_names = list(X.columns)

    # Train/test split
    years = data.loc[mask, "year"].values
    train_mask = years <= 2021
    test_mask = years > 2021

    X_train = X.values[train_mask]
    X_test = X.values[test_mask]
    y_train = y[train_mask]
    y_test = y[test_mask]

    logger.info(f"  Train: {len(X_train):,}, Test: {len(X_test):,}")
    logger.info(f"  Train: {(y_train > 0).sum():,} positive, {(y_train == 0).sum():,} zero")
    logger.info(f"  Test: {(y_test > 0).sum():,} positive, {(y_test == 0).sum():,} zero")

    return X_train, X_test, y_train, y_test, feature_names


# ============================================================================
# GAMMA GLM
# ============================================================================


def fit_gamma_glm(
    X_train: np.ndarray,
    y_train: np.ndarray,
    feature_names: list,
) -> sm.GLM:
    """
    Fit Gamma GLM with log link for positive claim severity.

    Gamma distribution is ideal for:
    - Strictly positive continuous data
    - Right-skewed distributions (like insurance claims)
    - Variance proportional to mean squared

    Log link: E[Y|X] = exp(Xβ), so exp(β_j) is the multiplicative
    effect of a 1-unit increase in feature j.
    """
    logger.info("=" * 60)
    logger.info("FITTING GAMMA GLM")
    logger.info("=" * 60)

    X_const = sm.add_constant(X_train)

    gamma_family = sm.families.Gamma(link=sm.families.links.Log())

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model = sm.GLM(y_train, X_const, family=gamma_family)
        results = model.fit(maxiter=100)

    logger.info(f"  Converged: {results.converged}")
    logger.info(f"  AIC: {results.aic:,.1f}")
    logger.info(f"  BIC: {results.bic_deviance:,.1f}")
    logger.info(f"  Deviance: {results.deviance:,.1f}")
    logger.info(f"  Pearson chi2: {results.pearson_chi2:,.1f}")
    logger.info(f"  Scale (dispersion): {results.scale:.4f}")

    # Print coefficient summary
    params = pd.DataFrame({
        "feature": ["const"] + feature_names,
        "coefficient": results.params,
        "std_err": results.bse,
        "z_value": results.tvalues,
        "p_value": results.pvalues,
        "exp_coef": np.exp(results.params),
    })
    params["significant"] = params["p_value"] < 0.05

    logger.info("\nGamma GLM Coefficients:")
    logger.info(f"{'Feature':<30} {'Coef':>10} {'exp(β)':>10} {'p-value':>10} {'Sig':>5}")
    logger.info("-" * 70)
    for _, row in params.iterrows():
        sig = "***" if row["p_value"] < 0.001 else "**" if row["p_value"] < 0.01 else "*" if row["p_value"] < 0.05 else ""
        logger.info(
            f"{row['feature']:<30} {row['coefficient']:>10.4f} "
            f"{row['exp_coef']:>10.4f} {row['p_value']:>10.4f} {sig:>5}"
        )

    return results


def evaluate_glm(
    results,
    X_test: np.ndarray,
    y_test: np.ndarray,
    model_name: str,
) -> dict:
    """Evaluate GLM predictions on test set."""
    X_test_const = sm.add_constant(X_test)
    y_pred = results.predict(X_test_const)

    # Clip predictions to reasonable range
    y_pred = np.clip(y_pred, 1, 1e7)

    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    mae = mean_absolute_error(y_test, y_pred)

    # R-squared (can be negative for poor models)
    r2 = r2_score(y_test, y_pred)

    # Mean Absolute Percentage Error (for positive values)
    pos_mask = y_test > 0
    if pos_mask.sum() > 0:
        mape = np.mean(np.abs((y_test[pos_mask] - y_pred[pos_mask]) / y_test[pos_mask])) * 100
    else:
        mape = np.nan

    metrics = {
        "model": model_name,
        "aic": results.aic,
        "bic": results.bic_deviance,
        "deviance": results.deviance,
        "scale": results.scale,
        "test_rmse": rmse,
        "test_mae": mae,
        "test_r2": r2,
        "test_mape": mape,
        "n_train": results.nobs,
        "n_test": len(y_test),
    }

    logger.info(f"\n{model_name} Test Set Evaluation:")
    logger.info(f"  RMSE:  ${rmse:,.0f}")
    logger.info(f"  MAE:   ${mae:,.0f}")
    logger.info(f"  R²:    {r2:.4f}")
    logger.info(f"  MAPE:  {mape:.1f}%")

    return metrics


# ============================================================================
# TWEEDIE GLM
# ============================================================================


def fit_tweedie_glm(
    X_train: np.ndarray,
    y_train: np.ndarray,
    feature_names: list,
    power: float = 1.5,
) -> sm.GLM:
    """
    Fit Tweedie GLM with log link.

    Tweedie distribution with 1 < p < 2 is a compound Poisson-Gamma:
    - Handles exact zeros (no claims) and positive continuous values
    - p=1 is Poisson, p=2 is Gamma, p=1.5 is the "sweet spot"

    This is the standard for insurance pricing because real data
    has many zeros (no claims) mixed with right-skewed positives.
    """
    logger.info(f"\nFitting Tweedie GLM (power={power})...")

    X_const = sm.add_constant(X_train)

    tweedie_family = sm.families.Tweedie(
        link=sm.families.links.Log(),
        var_power=power,
    )

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model = sm.GLM(y_train, X_const, family=tweedie_family)
        results = model.fit(maxiter=100)

    logger.info(f"  Converged: {results.converged}")
    logger.info(f"  AIC: {results.aic:,.1f}")
    logger.info(f"  Deviance: {results.deviance:,.1f}")

    return results


def tweedie_grid_search(
    X_train: np.ndarray,
    y_train: np.ndarray,
    feature_names: list,
    powers: list = None,
) -> tuple:
    """
    Grid search over Tweedie power parameter to find best fit.

    Returns: (best_results, best_power, comparison_df)
    """
    if powers is None:
        powers = [1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8, 1.9]

    logger.info("=" * 60)
    logger.info("TWEEDIE POWER PARAMETER GRID SEARCH")
    logger.info("=" * 60)

    results_list = []
    all_fitted = {}  # power -> results

    for power in powers:
        try:
            res = fit_tweedie_glm(X_train, y_train, feature_names, power=power)
            results_list.append({
                "power": power,
                "aic": res.aic,
                "deviance": res.deviance,
                "converged": res.converged,
            })
            all_fitted[power] = res
        except Exception as e:
            logger.warning(f"  Power {power} failed: {e}")
            results_list.append({
                "power": power, "aic": np.nan,
                "deviance": np.nan, "converged": False,
            })

    comparison_df = pd.DataFrame(results_list)
    logger.info(f"\nTweedie Grid Search Results:")
    logger.info(comparison_df.to_string(index=False))

    # Prefer converged models; fall back to best AIC if none converged
    converged = comparison_df[comparison_df["converged"] == True]
    if len(converged) > 0:
        best_row = converged.loc[converged["aic"].idxmin()]
        best_power = best_row["power"]
        logger.info(f"\nBest converged power: {best_power} (AIC={best_row['aic']:,.1f})")
    else:
        best_row = comparison_df.loc[comparison_df["aic"].idxmin()]
        best_power = best_row["power"]
        logger.info(f"\nNo models converged. Best AIC power: {best_power}")

    best_results = all_fitted[best_power]
    best_aic = best_row["aic"]

    # Print best model coefficients
    if best_results is not None:
        params = pd.DataFrame({
            "feature": ["const"] + feature_names,
            "coefficient": best_results.params,
            "std_err": best_results.bse,
            "z_value": best_results.tvalues,
            "p_value": best_results.pvalues,
            "exp_coef": np.exp(best_results.params),
        })

        logger.info(f"\nBest Tweedie (p={best_power}) Coefficients:")
        logger.info(f"{'Feature':<30} {'Coef':>10} {'exp(β)':>10} {'p-value':>10}")
        logger.info("-" * 65)
        for _, row in params.iterrows():
            sig = "***" if row["p_value"] < 0.001 else "**" if row["p_value"] < 0.01 else "*" if row["p_value"] < 0.05 else ""
            logger.info(
                f"{row['feature']:<30} {row['coefficient']:>10.4f} "
                f"{row['exp_coef']:>10.4f} {row['p_value']:>10.4f} {sig}"
            )

    return best_results, best_power, comparison_df


# ============================================================================
# CROSS-VALIDATION
# ============================================================================


def cross_validate_gamma(
    X: np.ndarray,
    y: np.ndarray,
    n_folds: int = 5,
) -> dict:
    """
    K-fold cross-validation for Gamma GLM.
    Returns dict with mean and std of metrics across folds.
    """
    logger.info(f"\n{n_folds}-Fold Cross-Validation (Gamma GLM)...")

    kf = KFold(n_splits=n_folds, shuffle=True, random_state=42)
    fold_metrics = []

    for fold, (train_idx, val_idx) in enumerate(kf.split(X), 1):
        X_tr, X_val = X[train_idx], X[val_idx]
        y_tr, y_val = y[train_idx], y[val_idx]

        X_tr_const = sm.add_constant(X_tr)
        X_val_const = sm.add_constant(X_val)

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            model = sm.GLM(
                y_tr, X_tr_const,
                family=sm.families.Gamma(link=sm.families.links.Log()),
            )
            res = model.fit(maxiter=100)

        y_pred = np.clip(res.predict(X_val_const), 1, 1e7)
        rmse = np.sqrt(mean_squared_error(y_val, y_pred))
        mae = mean_absolute_error(y_val, y_pred)

        fold_metrics.append({"fold": fold, "rmse": rmse, "mae": mae})
        logger.info(f"  Fold {fold}: RMSE=${rmse:,.0f}, MAE=${mae:,.0f}")

    metrics_df = pd.DataFrame(fold_metrics)
    cv_results = {
        "mean_rmse": metrics_df["rmse"].mean(),
        "std_rmse": metrics_df["rmse"].std(),
        "mean_mae": metrics_df["mae"].mean(),
        "std_mae": metrics_df["mae"].std(),
    }

    logger.info(f"\n  CV Mean RMSE: ${cv_results['mean_rmse']:,.0f} ± ${cv_results['std_rmse']:,.0f}")
    logger.info(f"  CV Mean MAE:  ${cv_results['mean_mae']:,.0f} ± ${cv_results['std_mae']:,.0f}")

    return cv_results


# ============================================================================
# DIAGNOSTICS
# ============================================================================


def compute_diagnostics(results, X_test, y_test, model_name: str) -> pd.DataFrame:
    """
    Compute residual diagnostics for GLM evaluation.

    Returns DataFrame with: y_actual, y_predicted, deviance_residuals,
    pearson_residuals, working_residuals.
    """
    X_test_const = sm.add_constant(X_test)
    y_pred = np.clip(results.predict(X_test_const), 1, 1e7)

    # Deviance residuals
    dev_resid = results.resid_deviance if hasattr(results, "resid_deviance") else None

    # Pearson residuals on test set
    pearson_resid = (y_test - y_pred) / np.sqrt(y_pred * results.scale)

    diag_df = pd.DataFrame({
        "y_actual": y_test,
        "y_predicted": y_pred,
        "residual": y_test - y_pred,
        "pearson_residual": pearson_resid,
        "abs_pct_error": np.abs((y_test - y_pred) / np.where(y_test > 0, y_test, 1)) * 100,
    })

    logger.info(f"\n{model_name} Diagnostics:")
    logger.info(f"  Pearson residuals — mean: {pearson_resid.mean():.3f}, std: {pearson_resid.std():.3f}")
    logger.info(f"  Prediction ratio (pred/actual) — mean: {(y_pred / np.where(y_test > 0, y_test, 1)).mean():.3f}")

    return diag_df


# ============================================================================
# LOGISTIC REGRESSION: CLAIMS SURGE PREDICTION
# ============================================================================

LOGISTIC_FEATURES = [
    # Lagged features (t-1 data predicting t)
    "total_disasters_lag1",
    "claim_count_lag1",
    "avg_claim_severity_lag1",
    # Current disaster exposure
    "cum_disasters_3yr",
    "flood_count",
    "hurricane_count",
    "severe_storm_count",
    # Demographics
    "total_population",
    "median_household_income",
    "median_home_value",
    "pct_occupied_housing",
    # Macro
    "MORTGAGE30US",
    "CSUSHPINSA",
    "CPIAUCSL",
]


def prepare_logistic_data(df: pd.DataFrame, surge_threshold: float = 0.5) -> tuple:
    """
    Prepare data for claims surge logistic regression.

    Creates a binary target: 1 if claim_count increased >50% YoY, else 0.
    Uses lagged features so model is genuinely predictive.

    Returns: (X_train, X_test, y_train, y_test, feature_names, scaler)
    """
    logger.info("Preparing data for Logistic Regression...")

    data = df.copy()

    # Create claims surge target: >50% YoY increase in claim count
    data["claim_count_prev"] = data.groupby("county_fips")["claim_count"].shift(1)
    data["claims_change_yoy"] = (
        (data["claim_count"] - data["claim_count_prev"]) / data["claim_count_prev"]
    )
    data["claims_surge_flag"] = (data["claims_change_yoy"] > surge_threshold).astype(float)

    # Require: previous year had at least 1 claim (to compute YoY change)
    data = data[data["claim_count_prev"] > 0].copy()
    logger.info(f"  Rows with valid YoY change: {len(data):,}")
    logger.info(f"  Surge rate: {data['claims_surge_flag'].mean():.1%}")
    logger.info(f"  Surge=1: {int(data['claims_surge_flag'].sum()):,}, "
                f"Surge=0: {int((1 - data['claims_surge_flag']).sum()):,}")

    # Create FEMA region dummies
    data = pd.get_dummies(data, columns=["fema_region"], prefix="fema_region", dtype=float)

    # Select features
    available_features = [f for f in LOGISTIC_FEATURES if f in data.columns]
    available_dummies = [f for f in data.columns if f.startswith("fema_region_") and f != "fema_region_0"]
    all_features = available_features + available_dummies

    y = data["claims_surge_flag"].values
    X = data[all_features].astype(float).copy()

    # Drop NaN rows
    mask = X.notna().all(axis=1)
    X = X[mask]
    y = y[mask.values]
    logger.info(f"  After dropping NaN: {len(X):,} rows")

    feature_names = list(X.columns)

    # Train/test split by time
    years = data.loc[mask, "year"].values
    train_mask = years <= 2021
    test_mask = years > 2021

    X_train = X.values[train_mask]
    X_test = X.values[test_mask]
    y_train = y[train_mask]
    y_test = y[test_mask]

    # Standardize features for logistic regression
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    logger.info(f"  Train: {len(X_train):,} (surge rate: {y_train.mean():.1%})")
    logger.info(f"  Test: {len(X_test):,} (surge rate: {y_test.mean():.1%})")

    return X_train, X_test, y_train, y_test, feature_names, scaler


def fit_logistic_statsmodels(
    X_train: np.ndarray,
    y_train: np.ndarray,
    feature_names: list,
) -> sm.GLM:
    """
    Fit logistic regression via statsmodels GLM (Binomial + Logit link).
    This gives us p-values, confidence intervals, and odds ratios.
    """
    logger.info("=" * 60)
    logger.info("LOGISTIC REGRESSION (Statsmodels)")
    logger.info("=" * 60)

    X_const = sm.add_constant(X_train)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model = sm.GLM(
            y_train, X_const,
            family=sm.families.Binomial(link=sm.families.links.Logit()),
        )
        results = model.fit(maxiter=100)

    logger.info(f"  Converged: {results.converged}")
    logger.info(f"  AIC: {results.aic:,.1f}")
    logger.info(f"  Deviance: {results.deviance:,.1f}")

    # Coefficient table with odds ratios
    params = pd.DataFrame({
        "feature": ["intercept"] + feature_names,
        "coefficient": results.params,
        "std_err": results.bse,
        "z_value": results.tvalues,
        "p_value": results.pvalues,
        "odds_ratio": np.exp(results.params),
    })

    logger.info(f"\nLogistic Regression Coefficients (Odds Ratios):")
    logger.info(f"{'Feature':<30} {'Coef':>8} {'OR':>8} {'p-value':>10}")
    logger.info("-" * 60)
    for _, row in params.iterrows():
        sig = "***" if row["p_value"] < 0.001 else "**" if row["p_value"] < 0.01 else "*" if row["p_value"] < 0.05 else ""
        logger.info(
            f"{row['feature']:<30} {row['coefficient']:>8.4f} "
            f"{row['odds_ratio']:>8.4f} {row['p_value']:>10.4f} {sig}"
        )

    return results


def fit_logistic_sklearn(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    feature_names: list,
) -> dict:
    """
    Fit logistic regression via sklearn for cross-validated AUC-ROC.
    Uses balanced class weights to handle imbalanced surge events.
    """
    logger.info("\n" + "=" * 60)
    logger.info("LOGISTIC REGRESSION (Sklearn + CV)")
    logger.info("=" * 60)

    clf = LogisticRegression(
        class_weight="balanced",
        max_iter=1000,
        random_state=42,
        solver="lbfgs",
    )
    clf.fit(X_train, y_train)

    # Predictions
    y_prob = clf.predict_proba(X_test)[:, 1]
    y_pred = clf.predict(X_test)

    # Metrics
    auc_roc = roc_auc_score(y_test, y_prob)
    avg_precision = average_precision_score(y_test, y_prob)

    # Optimal threshold via Youden's J statistic
    fpr, tpr, thresholds = roc_curve(y_test, y_prob)
    j_scores = tpr - fpr
    optimal_idx = np.argmax(j_scores)
    optimal_threshold = thresholds[optimal_idx]
    y_pred_optimal = (y_prob >= optimal_threshold).astype(int)
    cm_optimal = confusion_matrix(y_test, y_pred_optimal)

    logger.info(f"\n  AUC-ROC:           {auc_roc:.4f}")
    logger.info(f"  Avg Precision:     {avg_precision:.4f}")
    logger.info(f"  Optimal threshold: {optimal_threshold:.4f}")

    logger.info(f"\n  Confusion Matrix (threshold={optimal_threshold:.3f}):")
    logger.info(f"  {'':>15} Pred=0  Pred=1")
    logger.info(f"  {'Actual=0':>15} {cm_optimal[0, 0]:>6}  {cm_optimal[0, 1]:>6}")
    logger.info(f"  {'Actual=1':>15} {cm_optimal[1, 0]:>6}  {cm_optimal[1, 1]:>6}")

    logger.info(f"\n  Classification Report (optimal threshold):")
    logger.info(classification_report(y_test, y_pred_optimal, target_names=["No Surge", "Surge"]))

    # Cross-validation on training data
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    cv_scores = cross_val_score(clf, X_train, y_train, cv=cv, scoring="roc_auc")
    logger.info(f"  5-Fold CV AUC-ROC: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")

    # Save curves for plotting
    precision_vals, recall_vals, _ = precision_recall_curve(y_test, y_prob)

    results = {
        "auc_roc": auc_roc,
        "avg_precision": avg_precision,
        "optimal_threshold": optimal_threshold,
        "cv_auc_mean": cv_scores.mean(),
        "cv_auc_std": cv_scores.std(),
        "confusion_matrix": cm_optimal,
        "fpr": fpr,
        "tpr": tpr,
        "precision_curve": precision_vals,
        "recall_curve": recall_vals,
        "y_prob": y_prob,
        "y_test": y_test,
        "model": clf,
    }

    return results


# ============================================================================
# EXTRACT RESULTS FOR NOTEBOOK
# ============================================================================


def extract_coefficient_table(results, feature_names: list, model_name: str) -> pd.DataFrame:
    """Extract tidy coefficient table for visualization."""
    params = pd.DataFrame({
        "feature": ["intercept"] + feature_names,
        "coefficient": results.params,
        "std_err": results.bse,
        "z_value": results.tvalues,
        "p_value": results.pvalues,
        "exp_coef": np.exp(results.params),
        "ci_lower": np.exp(results.conf_int()[:, 0]),
        "ci_upper": np.exp(results.conf_int()[:, 1]),
    })
    params["significant"] = params["p_value"] < 0.05
    params["model"] = model_name
    return params


# ============================================================================
# MAIN PIPELINE
# ============================================================================


def main():
    """Run full GLM pipeline."""
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    REPORTS_FIGURES.mkdir(parents=True, exist_ok=True)

    # --- Load panel data ---
    logger.info("Loading panel dataset...")
    df = pd.read_csv(
        DATA_PROCESSED / "county_year_panel_glm_ready.csv",
        dtype={"county_fips": str},
    )
    logger.info(f"Panel: {len(df):,} rows, {len(df.columns)} columns")

    # =================================================================
    # GAMMA GLM
    # =================================================================
    X_train, X_test, y_train, y_test, feature_names = prepare_gamma_data(df)

    # Fit Gamma GLM
    gamma_results = fit_gamma_glm(X_train, y_train, feature_names)

    # Evaluate on test set
    gamma_metrics = evaluate_glm(gamma_results, X_test, y_test, "Gamma GLM")

    # Cross-validation
    X_all = np.vstack([X_train, X_test])
    y_all = np.concatenate([y_train, y_test])
    gamma_cv = cross_validate_gamma(X_all, y_all, n_folds=5)

    # Diagnostics
    gamma_diag = compute_diagnostics(gamma_results, X_test, y_test, "Gamma GLM")

    # Coefficient table
    gamma_coefs = extract_coefficient_table(gamma_results, feature_names, "Gamma GLM")

    # =================================================================
    # TWEEDIE GLM
    # =================================================================
    X_train_tw, X_test_tw, y_train_tw, y_test_tw, tw_features = prepare_tweedie_data(df)

    # Grid search for best power
    best_tweedie, best_power, tweedie_comparison = tweedie_grid_search(
        X_train_tw, y_train_tw, tw_features
    )

    # Evaluate best Tweedie on test set
    tweedie_metrics = evaluate_glm(best_tweedie, X_test_tw, y_test_tw, f"Tweedie (p={best_power})")

    # Diagnostics
    tweedie_diag = compute_diagnostics(best_tweedie, X_test_tw, y_test_tw, "Tweedie GLM")

    # Coefficient table
    tweedie_coefs = extract_coefficient_table(best_tweedie, tw_features, f"Tweedie (p={best_power})")

    # =================================================================
    # LOGISTIC REGRESSION: CLAIMS SURGE
    # =================================================================
    X_train_lr, X_test_lr, y_train_lr, y_test_lr, lr_features, scaler = prepare_logistic_data(df)

    # Statsmodels (for p-values and odds ratios)
    logistic_sm = fit_logistic_statsmodels(X_train_lr, y_train_lr, lr_features)
    logistic_coefs = extract_coefficient_table(logistic_sm, lr_features, "Logistic (Claims Surge)")

    # Sklearn (for AUC-ROC and cross-validation)
    logistic_sk = fit_logistic_sklearn(X_train_lr, y_train_lr, X_test_lr, y_test_lr, lr_features)

    # =================================================================
    # SAVE RESULTS
    # =================================================================
    logger.info("\n" + "=" * 60)
    logger.info("SAVING RESULTS")
    logger.info("=" * 60)

    # Save model objects
    with open(MODELS_DIR / "gamma_glm.pkl", "wb") as f:
        pickle.dump(gamma_results, f)
    logger.info(f"  Saved Gamma GLM to {MODELS_DIR / 'gamma_glm.pkl'}")

    with open(MODELS_DIR / "tweedie_glm.pkl", "wb") as f:
        pickle.dump(best_tweedie, f)
    logger.info(f"  Saved Tweedie GLM to {MODELS_DIR / 'tweedie_glm.pkl'}")

    with open(MODELS_DIR / "logistic_claims_surge.pkl", "wb") as f:
        pickle.dump(logistic_sk["model"], f)
    logger.info(f"  Saved Logistic model")

    with open(MODELS_DIR / "logistic_scaler.pkl", "wb") as f:
        pickle.dump(scaler, f)

    # Save coefficient tables
    gamma_coefs.to_csv(MODELS_DIR / "gamma_glm_coefficients.csv", index=False)
    tweedie_coefs.to_csv(MODELS_DIR / "tweedie_glm_coefficients.csv", index=False)
    logistic_coefs.to_csv(MODELS_DIR / "logistic_coefficients.csv", index=False)
    logger.info("  Saved coefficient tables")

    # Save comparison metrics
    metrics_df = pd.DataFrame([gamma_metrics, tweedie_metrics])
    metrics_df.to_csv(MODELS_DIR / "glm_comparison_metrics.csv", index=False)
    logger.info("  Saved comparison metrics")

    # Save logistic metrics
    logistic_metrics = {
        "auc_roc": logistic_sk["auc_roc"],
        "avg_precision": logistic_sk["avg_precision"],
        "optimal_threshold": logistic_sk["optimal_threshold"],
        "cv_auc_mean": logistic_sk["cv_auc_mean"],
        "cv_auc_std": logistic_sk["cv_auc_std"],
    }
    pd.DataFrame([logistic_metrics]).to_csv(MODELS_DIR / "logistic_metrics.csv", index=False)

    # Save ROC/PR curve data for notebook
    curves_df = pd.DataFrame({
        "fpr": logistic_sk["fpr"],
        "tpr": logistic_sk["tpr"],
    })
    curves_df.to_csv(MODELS_DIR / "logistic_roc_curve.csv", index=False)

    # Save predicted probabilities
    pred_df = pd.DataFrame({
        "y_test": logistic_sk["y_test"],
        "y_prob": logistic_sk["y_prob"],
    })
    pred_df.to_csv(MODELS_DIR / "logistic_predictions.csv", index=False)

    # Save Tweedie grid search results
    tweedie_comparison.to_csv(MODELS_DIR / "tweedie_power_search.csv", index=False)
    logger.info("  Saved Tweedie power search results")

    # Save diagnostics
    gamma_diag.to_csv(MODELS_DIR / "gamma_glm_diagnostics.csv", index=False)
    tweedie_diag.to_csv(MODELS_DIR / "tweedie_glm_diagnostics.csv", index=False)
    logger.info("  Saved diagnostic data")

    # Save feature names for notebook
    pd.DataFrame({"feature": feature_names}).to_csv(
        MODELS_DIR / "glm_feature_names.csv", index=False
    )
    pd.DataFrame({"feature": lr_features}).to_csv(
        MODELS_DIR / "logistic_feature_names.csv", index=False
    )

    # =================================================================
    # SUMMARY
    # =================================================================
    logger.info("\n" + "=" * 60)
    logger.info("GLM PIPELINE COMPLETE")
    logger.info("=" * 60)
    logger.info(f"\nGamma GLM:")
    logger.info(f"  AIC: {gamma_metrics['aic']:,.1f}")
    logger.info(f"  Test RMSE: ${gamma_metrics['test_rmse']:,.0f}")
    logger.info(f"  Test MAE: ${gamma_metrics['test_mae']:,.0f}")
    logger.info(f"  Test R²: {gamma_metrics['test_r2']:.4f}")
    logger.info(f"  CV RMSE: ${gamma_cv['mean_rmse']:,.0f} ± ${gamma_cv['std_rmse']:,.0f}")
    logger.info(f"  Significant features: {gamma_coefs[gamma_coefs['significant'] & (gamma_coefs['feature'] != 'intercept')].shape[0]}/{len(feature_names)}")

    logger.info(f"\nTweedie GLM (power={best_power}):")
    logger.info(f"  AIC: {tweedie_metrics['aic']:,.1f}")
    logger.info(f"  Test RMSE: ${tweedie_metrics['test_rmse']:,.0f}")
    logger.info(f"  Test MAE: ${tweedie_metrics['test_mae']:,.0f}")
    logger.info(f"  Test R²: {tweedie_metrics['test_r2']:.4f}")

    logger.info(f"\nLogistic Regression (Claims Surge):")
    logger.info(f"  AUC-ROC: {logistic_sk['auc_roc']:.4f}")
    logger.info(f"  Avg Precision: {logistic_sk['avg_precision']:.4f}")
    logger.info(f"  5-Fold CV AUC: {logistic_sk['cv_auc_mean']:.4f} ± {logistic_sk['cv_auc_std']:.4f}")

    # Key findings
    logger.info("\n--- Key Coefficient Interpretations (Gamma GLM) ---")
    sig_coefs = gamma_coefs[
        gamma_coefs["significant"] & (gamma_coefs["feature"] != "intercept")
    ].sort_values("exp_coef", ascending=False)

    for _, row in sig_coefs.head(5).iterrows():
        effect = "increases" if row["exp_coef"] > 1 else "decreases"
        pct = abs(row["exp_coef"] - 1) * 100
        logger.info(f"  {row['feature']}: {effect} expected claims by {pct:.1f}% per unit")

    logger.info("\n--- Key Odds Ratios (Logistic Regression) ---")
    sig_or = logistic_coefs[
        logistic_coefs["significant"] & (logistic_coefs["feature"] != "intercept")
    ].sort_values("exp_coef", ascending=False)

    for _, row in sig_or.head(5).iterrows():
        or_val = row["exp_coef"]
        direction = "increases" if or_val > 1 else "decreases"
        logger.info(
            f"  {row['feature']}: OR={or_val:.3f} — 1 SD increase {direction} "
            f"surge odds by {abs(or_val - 1) * 100:.1f}%"
        )


if __name__ == "__main__":
    main()
