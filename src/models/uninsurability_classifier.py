"""
Uninsurability Risk Classification using Tree-Based ML.

Predicts which US counties face insurance affordability crises using a
composite risk target derived from claim severity, disaster exposure,
FEMA Housing Assistance damage, and aggregate claims.

Models:
    1. Gradient Boosting Classifier — primary model
    2. Random Forest Classifier — ensemble comparison

Explainability:
    - SHAP (TreeExplainer) for feature importance and interaction analysis

Usage:
    python -m src.models.uninsurability_classifier
"""

import logging
import pickle
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import shap
from sklearn.base import clone
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    confusion_matrix,
    classification_report,
    roc_curve,
    precision_recall_curve,
    f1_score,
    brier_score_loss,
)
from sklearn.model_selection import StratifiedKFold, GridSearchCV
from sklearn.utils.class_weight import compute_sample_weight

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

CLASSIFIER_FEATURES = [
    # Disaster exposure
    "total_disasters",
    "flood_count",
    "hurricane_count",
    "severe_storm_count",
    "wildfire_count",
    "winter_storm_count",
    "tornado_count",
    "cum_disasters_3yr",
    "disaster_exposure_index",
    "avg_incident_duration",
    # Claims
    "claim_count",
    "total_claims_paid",
    "avg_claim_severity",
    "log_avg_claim_severity",
    "log_total_claims_paid",
    "claims_per_capita",
    "damage_per_capita",
    # Housing Assistance
    "ha_total_damage",
    "ha_total_approved",
    "log_ha_total_damage",
    # Demographics
    "total_population",
    "median_household_income",
    "median_home_value",
    "pct_occupied_housing",
    "unemployment_rate",
    # Macro indicators
    "MORTGAGE30US",
    "CSUSHPINSA",
    "CPIAUCSL",
    "FEDFUNDS",
    # Lagged features
    "total_disasters_lag1",
    "claim_count_lag1",
    "avg_claim_severity_lag1",
]

# Predictive-only features (no current-year claims/HA — genuine forecasting)
PREDICTIVE_FEATURES = [
    "total_disasters",
    "flood_count",
    "hurricane_count",
    "severe_storm_count",
    "wildfire_count",
    "winter_storm_count",
    "tornado_count",
    "cum_disasters_3yr",
    "disaster_exposure_index",
    "avg_incident_duration",
    "total_population",
    "median_household_income",
    "median_home_value",
    "pct_occupied_housing",
    "unemployment_rate",
    "MORTGAGE30US",
    "CSUSHPINSA",
    "CPIAUCSL",
    "FEDFUNDS",
    "total_disasters_lag1",
    "claim_count_lag1",
    "avg_claim_severity_lag1",
]

# Thresholds for composite target (computed from training data only)
DEFAULT_SEVERITY_QUANTILE = 0.75
DEFAULT_DISASTER_QUANTILE = 0.75
DEFAULT_CLAIMS_PAID_QUANTILE = 0.75
MIN_RISK_SIGNALS = 2


# ============================================================================
# TARGET VARIABLE CONSTRUCTION
# ============================================================================


def create_uninsurability_target(
    df: pd.DataFrame,
    severity_quantile: float = DEFAULT_SEVERITY_QUANTILE,
    disaster_quantile: float = DEFAULT_DISASTER_QUANTILE,
    claims_paid_quantile: float = DEFAULT_CLAIMS_PAID_QUANTILE,
    min_signals: int = MIN_RISK_SIGNALS,
    train_years_max: int = 2021,
) -> tuple:
    """
    Create composite uninsurability risk target from multiple risk signals.

    Thresholds are computed from TRAINING data only (year <= train_years_max)
    to prevent data leakage.

    Risk signals:
        1. High claim severity (>= 75th pctile of positive claims)
        2. High cumulative disaster exposure (>= 75th pctile)
        3. Any FEMA Housing Assistance damage (ha_total_damage > 0)
        4. High total claims paid (>= 75th pctile)

    Target = 1 if county-year has >= min_signals active risk signals.

    Returns:
        (df_with_target, thresholds_dict)
    """
    logger.info("Creating composite uninsurability risk target...")

    data = df.copy()
    train_data = data[data["year"] <= train_years_max]

    # Compute thresholds from training data only
    positive_claims_train = train_data[train_data["avg_claim_severity"] > 0]
    severity_threshold = positive_claims_train["avg_claim_severity"].quantile(
        severity_quantile
    )
    disaster_threshold = train_data["cum_disasters_3yr"].quantile(disaster_quantile)
    claims_paid_threshold = train_data["total_claims_paid"].quantile(
        claims_paid_quantile
    )

    thresholds = {
        "severity_threshold": severity_threshold,
        "disaster_threshold": disaster_threshold,
        "claims_paid_threshold": claims_paid_threshold,
        "severity_quantile": severity_quantile,
        "disaster_quantile": disaster_quantile,
        "claims_paid_quantile": claims_paid_quantile,
        "min_signals": min_signals,
        "train_years_max": train_years_max,
    }

    logger.info(f"  Thresholds (from training data <= {train_years_max}):")
    logger.info(f"    Claim severity: >= ${severity_threshold:,.0f}")
    logger.info(f"    Cum disasters 3yr: >= {disaster_threshold:.1f}")
    logger.info(f"    Total claims paid: >= ${claims_paid_threshold:,.0f}")
    logger.info(f"    HA damage: > $0")

    # Apply signals across full dataset
    data["signal_high_severity"] = (
        data["avg_claim_severity"] >= severity_threshold
    ).astype(int)
    data["signal_high_disasters"] = (
        data["cum_disasters_3yr"] >= disaster_threshold
    ).astype(int)
    data["signal_ha_damage"] = (data["ha_total_damage"] > 0).astype(int)
    data["signal_high_claims_paid"] = (
        data["total_claims_paid"] >= claims_paid_threshold
    ).astype(int)

    # Composite target: >= min_signals active
    data["risk_signal_count"] = (
        data["signal_high_severity"]
        + data["signal_high_disasters"]
        + data["signal_ha_damage"]
        + data["signal_high_claims_paid"]
    )
    data["uninsurability_risk"] = (
        data["risk_signal_count"] >= min_signals
    ).astype(int)

    # Log distribution
    overall_rate = data["uninsurability_risk"].mean()
    train_rate = data[data["year"] <= train_years_max]["uninsurability_risk"].mean()
    test_rate = data[data["year"] > train_years_max]["uninsurability_risk"].mean()

    logger.info(f"\n  Target distribution:")
    logger.info(
        f"    Overall: {data['uninsurability_risk'].sum():,} / "
        f"{len(data):,} ({overall_rate:.1%})"
    )
    logger.info(f"    Train (<=  {train_years_max}): {train_rate:.1%}")
    logger.info(f"    Test  (> {train_years_max}): {test_rate:.1%}")

    logger.info(f"\n  Signal positive rates:")
    for col in [
        "signal_high_severity",
        "signal_high_disasters",
        "signal_ha_damage",
        "signal_high_claims_paid",
    ]:
        rate = data[col].mean()
        logger.info(f"    {col}: {rate:.1%}")

    return data, thresholds


# ============================================================================
# DATA PREPARATION
# ============================================================================


def prepare_classification_data(
    df: pd.DataFrame,
    feature_list: list = None,
    predictive_only: bool = True,
) -> tuple:
    """
    Prepare features and target for classification.

    By default uses PREDICTIVE_FEATURES (no current-year claims/HA data)
    to avoid target leakage, since the composite target is constructed
    from claims and HA signals. Set predictive_only=False to include
    all features for risk scoring (not genuine prediction).

    Steps:
        1. Create composite target via create_uninsurability_target()
        2. Create FEMA region dummies
        3. Select features
        4. Impute missing values (median strategy)
        5. Train/test split by time (<= 2021 train, > 2021 test)

    Returns:
        (X_train, X_test, y_train, y_test, feature_names, imputer,
         test_meta, thresholds)
        test_meta: DataFrame with county_fips, year, state for test rows
    """
    logger.info("Preparing data for classification...")

    # Create target
    data, thresholds = create_uninsurability_target(df)

    # Create FEMA region dummies
    data = pd.get_dummies(
        data, columns=["fema_region"], prefix="fema_region", dtype=float
    )

    # Select features — default to PREDICTIVE_FEATURES to avoid leakage
    if feature_list is not None:
        base_features = feature_list
    elif predictive_only:
        base_features = PREDICTIVE_FEATURES
    else:
        base_features = CLASSIFIER_FEATURES

    available_features = [f for f in base_features if f in data.columns]
    available_dummies = [
        f
        for f in data.columns
        if f.startswith("fema_region_") and f != "fema_region_0"
    ]
    all_features = available_features + available_dummies
    logger.info(f"  Features: {len(all_features)} ({len(available_features)} numeric + {len(available_dummies)} FEMA dummies)")

    # Target
    y = data["uninsurability_risk"].values

    # Feature matrix
    X = data[all_features].astype(float).copy()

    feature_names = list(X.columns)

    # Train/test split by time
    years = data["year"].values
    train_mask = years <= 2021
    test_mask = years > 2021

    # Save test metadata for risk scoring
    test_meta = data.loc[test_mask, ["county_fips", "year", "state"]].reset_index(
        drop=True
    )

    X_train = X.values[train_mask]
    X_test = X.values[test_mask]
    y_train = y[train_mask]
    y_test = y[test_mask]

    # Impute missing values (median, fitted on training only)
    imputer = SimpleImputer(strategy="median")
    X_train = imputer.fit_transform(X_train)
    X_test = imputer.transform(X_test)

    logger.info(f"  Train: {len(X_train):,} rows (risk rate: {y_train.mean():.1%})")
    logger.info(f"  Test:  {len(X_test):,} rows (risk rate: {y_test.mean():.1%})")

    return X_train, X_test, y_train, y_test, feature_names, imputer, test_meta, thresholds


# ============================================================================
# GRADIENT BOOSTING CLASSIFIER
# ============================================================================


def fit_gradient_boosting(
    X_train: np.ndarray,
    y_train: np.ndarray,
    feature_names: list,
) -> tuple:
    """
    Fit Gradient Boosting Classifier with balanced sample weights.

    Uses moderate regularization defaults; hyperparameter_search()
    provides the full grid search.

    Returns:
        (fitted_model, training_info_dict)
    """
    logger.info("=" * 60)
    logger.info("FITTING GRADIENT BOOSTING CLASSIFIER")
    logger.info("=" * 60)

    clf = GradientBoostingClassifier(
        n_estimators=300,
        max_depth=5,
        learning_rate=0.1,
        subsample=0.8,
        min_samples_leaf=20,
        min_samples_split=40,
        max_features="sqrt",
        random_state=42,
    )

    # Balanced sample weights for class imbalance
    sample_weights = compute_sample_weight("balanced", y_train)
    clf.fit(X_train, y_train, sample_weight=sample_weights)

    # Training metrics
    y_prob_train = clf.predict_proba(X_train)[:, 1]
    train_auc = roc_auc_score(y_train, y_prob_train)

    logger.info(f"  Training AUC-ROC: {train_auc:.4f}")

    # Feature importance (sklearn's impurity-based)
    importances = pd.DataFrame(
        {"feature": feature_names, "importance": clf.feature_importances_}
    ).sort_values("importance", ascending=False)

    logger.info(f"\n  Top 10 Feature Importances:")
    for _, row in importances.head(10).iterrows():
        logger.info(f"    {row['feature']:<35} {row['importance']:.4f}")

    info = {
        "train_auc": train_auc,
        "n_estimators": clf.n_estimators,
        "max_depth": clf.max_depth,
        "learning_rate": clf.learning_rate,
    }

    return clf, info


# ============================================================================
# RANDOM FOREST CLASSIFIER
# ============================================================================


def fit_random_forest(
    X_train: np.ndarray,
    y_train: np.ndarray,
    feature_names: list,
) -> tuple:
    """
    Fit Random Forest Classifier with balanced class weights.

    Returns:
        (fitted_model, training_info_dict)
    """
    logger.info("\n" + "=" * 60)
    logger.info("FITTING RANDOM FOREST CLASSIFIER")
    logger.info("=" * 60)

    clf = RandomForestClassifier(
        n_estimators=500,
        max_depth=10,
        min_samples_leaf=20,
        min_samples_split=40,
        max_features="sqrt",
        class_weight="balanced_subsample",
        random_state=42,
        n_jobs=-1,
    )
    clf.fit(X_train, y_train)

    # Training metrics
    y_prob_train = clf.predict_proba(X_train)[:, 1]
    train_auc = roc_auc_score(y_train, y_prob_train)

    logger.info(f"  Training AUC-ROC: {train_auc:.4f}")

    # Feature importance
    importances = pd.DataFrame(
        {"feature": feature_names, "importance": clf.feature_importances_}
    ).sort_values("importance", ascending=False)

    logger.info(f"\n  Top 10 Feature Importances:")
    for _, row in importances.head(10).iterrows():
        logger.info(f"    {row['feature']:<35} {row['importance']:.4f}")

    info = {
        "train_auc": train_auc,
        "n_estimators": clf.n_estimators,
        "max_depth": clf.max_depth,
    }

    return clf, info


# ============================================================================
# EVALUATION
# ============================================================================


def evaluate_classifier(
    model,
    X_test: np.ndarray,
    y_test: np.ndarray,
    model_name: str,
) -> dict:
    """
    Comprehensive evaluation of a binary classifier.

    Computes AUC-ROC, Average Precision, Brier Score, F1 (at optimal
    threshold via Youden's J), confusion matrix, ROC/PR curves.

    Returns:
        dict with all metrics and curve data.
    """
    logger.info(f"\n{model_name} — Test Set Evaluation:")

    y_prob = model.predict_proba(X_test)[:, 1]

    # Core metrics
    auc_roc = roc_auc_score(y_test, y_prob)
    avg_precision = average_precision_score(y_test, y_prob)
    brier = brier_score_loss(y_test, y_prob)

    # Optimal threshold via Youden's J statistic
    fpr, tpr, thresholds_roc = roc_curve(y_test, y_prob)
    j_scores = tpr - fpr
    optimal_idx = np.argmax(j_scores)
    optimal_threshold = thresholds_roc[optimal_idx]
    y_pred_optimal = (y_prob >= optimal_threshold).astype(int)

    # F1 at optimal threshold
    f1 = f1_score(y_test, y_pred_optimal)

    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred_optimal)

    logger.info(f"  AUC-ROC:           {auc_roc:.4f}")
    logger.info(f"  Avg Precision:     {avg_precision:.4f}")
    logger.info(f"  Brier Score:       {brier:.4f}")
    logger.info(f"  F1 Score:          {f1:.4f}")
    logger.info(f"  Optimal threshold: {optimal_threshold:.4f}")

    logger.info(f"\n  Confusion Matrix (threshold={optimal_threshold:.3f}):")
    logger.info(f"  {'':>15} Pred=0  Pred=1")
    logger.info(f"  {'Actual=0':>15} {cm[0, 0]:>6}  {cm[0, 1]:>6}")
    logger.info(f"  {'Actual=1':>15} {cm[1, 0]:>6}  {cm[1, 1]:>6}")

    logger.info(f"\n  Classification Report (optimal threshold):")
    logger.info(
        classification_report(
            y_test, y_pred_optimal, target_names=["Low Risk", "High Risk"]
        )
    )

    # Precision-Recall curve
    precision_vals, recall_vals, _ = precision_recall_curve(y_test, y_prob)

    results = {
        "model_name": model_name,
        "auc_roc": auc_roc,
        "avg_precision": avg_precision,
        "brier_score": brier,
        "f1_score": f1,
        "optimal_threshold": optimal_threshold,
        "confusion_matrix": cm,
        "fpr": fpr,
        "tpr": tpr,
        "precision_curve": precision_vals,
        "recall_curve": recall_vals,
        "y_prob": y_prob,
        "y_test": y_test,
    }

    return results


# ============================================================================
# CROSS-VALIDATION
# ============================================================================


def cross_validate_classifier(
    model,
    X: np.ndarray,
    y: np.ndarray,
    n_folds: int = 5,
    model_name: str = "Classifier",
) -> dict:
    """
    Stratified K-Fold cross-validation for a classifier.

    Reports AUC-ROC, Average Precision, and F1 per fold.

    Returns:
        dict with mean/std of metrics and per-fold results DataFrame.
    """
    logger.info(f"\n{n_folds}-Fold Stratified Cross-Validation ({model_name})...")

    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)
    fold_metrics = []

    for fold, (train_idx, val_idx) in enumerate(skf.split(X, y), 1):
        X_tr, X_val = X[train_idx], X[val_idx]
        y_tr, y_val = y[train_idx], y[val_idx]

        clf = clone(model)

        # Apply sample weights for GBC
        if isinstance(clf, GradientBoostingClassifier):
            sw = compute_sample_weight("balanced", y_tr)
            clf.fit(X_tr, y_tr, sample_weight=sw)
        else:
            clf.fit(X_tr, y_tr)

        y_prob = clf.predict_proba(X_val)[:, 1]
        auc = roc_auc_score(y_val, y_prob)
        ap = average_precision_score(y_val, y_prob)

        # F1 at 0.5 threshold for simplicity in CV
        y_pred = (y_prob >= 0.5).astype(int)
        f1 = f1_score(y_val, y_pred, zero_division=0)

        fold_metrics.append(
            {"fold": fold, "auc_roc": auc, "avg_precision": ap, "f1_score": f1}
        )
        logger.info(f"  Fold {fold}: AUC={auc:.4f}, AP={ap:.4f}, F1={f1:.4f}")

    metrics_df = pd.DataFrame(fold_metrics)
    cv_results = {
        "mean_auc": metrics_df["auc_roc"].mean(),
        "std_auc": metrics_df["auc_roc"].std(),
        "mean_ap": metrics_df["avg_precision"].mean(),
        "std_ap": metrics_df["avg_precision"].std(),
        "mean_f1": metrics_df["f1_score"].mean(),
        "std_f1": metrics_df["f1_score"].std(),
        "fold_results": metrics_df,
    }

    logger.info(
        f"\n  CV AUC-ROC: {cv_results['mean_auc']:.4f} +/- {cv_results['std_auc']:.4f}"
    )
    logger.info(
        f"  CV Avg Precision: {cv_results['mean_ap']:.4f} +/- {cv_results['std_ap']:.4f}"
    )
    logger.info(
        f"  CV F1: {cv_results['mean_f1']:.4f} +/- {cv_results['std_f1']:.4f}"
    )

    return cv_results


# ============================================================================
# HYPERPARAMETER SEARCH
# ============================================================================


def hyperparameter_search(
    X_train: np.ndarray,
    y_train: np.ndarray,
    model_type: str = "gradient_boosting",
) -> tuple:
    """
    Grid search over hyperparameters with stratified CV.

    Returns:
        (best_estimator, results_dataframe)
    """
    logger.info(f"\nHyperparameter search ({model_type})...")

    if model_type == "gradient_boosting":
        base_model = GradientBoostingClassifier(
            subsample=0.8,
            min_samples_leaf=20,
            max_features="sqrt",
            random_state=42,
        )
        param_grid = {
            "n_estimators": [200, 300, 500],
            "max_depth": [3, 5, 7],
            "learning_rate": [0.05, 0.1],
        }
    else:
        base_model = RandomForestClassifier(
            min_samples_leaf=20,
            class_weight="balanced_subsample",
            random_state=42,
            n_jobs=-1,
        )
        param_grid = {
            "n_estimators": [300, 500],
            "max_depth": [8, 10, 15],
            "max_features": ["sqrt", "log2"],
        }

    cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)

    # For GBC, use sample_weight in fit
    if model_type == "gradient_boosting":
        sample_weights = compute_sample_weight("balanced", y_train)
        grid_search = GridSearchCV(
            base_model,
            param_grid,
            scoring="roc_auc",
            cv=cv,
            n_jobs=-1,
            verbose=1,
        )
        grid_search.fit(X_train, y_train, sample_weight=sample_weights)
    else:
        grid_search = GridSearchCV(
            base_model,
            param_grid,
            scoring="roc_auc",
            cv=cv,
            n_jobs=-1,
            verbose=1,
        )
        grid_search.fit(X_train, y_train)

    results_df = pd.DataFrame(grid_search.cv_results_)[
        ["params", "mean_test_score", "std_test_score", "rank_test_score"]
    ].sort_values("rank_test_score")

    logger.info(f"\n  Best params: {grid_search.best_params_}")
    logger.info(f"  Best CV AUC-ROC: {grid_search.best_score_:.4f}")

    return grid_search.best_estimator_, results_df


# ============================================================================
# SHAP ANALYSIS
# ============================================================================


def compute_shap_values(
    model,
    X_test: np.ndarray,
    feature_names: list,
    max_samples: int = 1000,
) -> tuple:
    """
    Compute SHAP values using TreeExplainer for feature importance.

    Args:
        model: Fitted tree-based model (GBC or RFC).
        X_test: Test features.
        feature_names: Feature names.
        max_samples: Max samples for SHAP computation (performance).

    Returns:
        (shap_values_array, shap_explanation_object)
    """
    logger.info("\nComputing SHAP values...")

    explainer = shap.TreeExplainer(model)

    # Subsample for performance if test set is large
    if len(X_test) > max_samples:
        rng = np.random.RandomState(42)
        idx = rng.choice(len(X_test), max_samples, replace=False)
        X_explain = X_test[idx]
    else:
        X_explain = X_test

    logger.info(f"  Computing SHAP for {len(X_explain)} samples...")

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        shap_values = explainer.shap_values(X_explain)

    # For binary classifier, shap_values may be [class_0, class_1]
    if isinstance(shap_values, list):
        shap_values = shap_values[1]

    expected_value = explainer.expected_value
    if isinstance(expected_value, (list, np.ndarray)):
        expected_value = expected_value[1] if len(expected_value) > 1 else expected_value[0]

    explanation = shap.Explanation(
        values=shap_values,
        base_values=expected_value,
        data=X_explain,
        feature_names=feature_names,
    )

    # Log top SHAP features
    mean_abs_shap = np.abs(shap_values).mean(axis=0)
    shap_importance = pd.DataFrame(
        {"feature": feature_names, "mean_abs_shap": mean_abs_shap}
    ).sort_values("mean_abs_shap", ascending=False)

    logger.info(f"\n  Top 10 SHAP Feature Importances:")
    for _, row in shap_importance.head(10).iterrows():
        logger.info(f"    {row['feature']:<35} {row['mean_abs_shap']:.4f}")

    return shap_values, explanation


# ============================================================================
# RISK SCORE GENERATION
# ============================================================================


def generate_risk_scores(
    model,
    df: pd.DataFrame,
    feature_names: list,
    imputer: SimpleImputer,
) -> pd.DataFrame:
    """
    Generate risk scores for all county-years and identify highest-risk counties.

    Returns:
        DataFrame with county_fips, state, year, risk_probability, plus
        county-level aggregates (mean_risk, risk_rank).
    """
    logger.info("\nGenerating county-level risk scores...")

    data = df.copy()

    # Recreate FEMA dummies if needed
    if "fema_region" in data.columns:
        data = pd.get_dummies(
            data, columns=["fema_region"], prefix="fema_region", dtype=float
        )

    # Select features (use available ones)
    available = [f for f in feature_names if f in data.columns]
    missing = [f for f in feature_names if f not in data.columns]
    if missing:
        logger.warning(f"  Missing features (will be 0): {missing}")
        for col in missing:
            data[col] = 0.0

    X_all = data[feature_names].astype(float).values
    X_all = imputer.transform(X_all)

    # Predict risk probabilities
    risk_probs = model.predict_proba(X_all)[:, 1]

    scores = pd.DataFrame(
        {
            "county_fips": df["county_fips"].values,
            "state": df["state"].values,
            "year": df["year"].values,
            "risk_probability": risk_probs,
        }
    )

    # County-level aggregates (mean risk across all years)
    county_scores = (
        scores.groupby(["county_fips", "state"])
        .agg(
            mean_risk=("risk_probability", "mean"),
            max_risk=("risk_probability", "max"),
            n_years=("risk_probability", "count"),
        )
        .reset_index()
        .sort_values("mean_risk", ascending=False)
    )
    county_scores["risk_rank"] = range(1, len(county_scores) + 1)

    logger.info(f"  Scored {len(scores):,} county-years across {len(county_scores):,} counties")
    logger.info(f"\n  Top 10 Highest-Risk Counties:")
    logger.info(f"  {'Rank':<6} {'County FIPS':<12} {'State':<6} {'Mean Risk':>10} {'Max Risk':>10}")
    logger.info("  " + "-" * 50)
    for _, row in county_scores.head(10).iterrows():
        logger.info(
            f"  {row['risk_rank']:<6} {row['county_fips']:<12} "
            f"{row['state']:<6} {row['mean_risk']:>10.4f} {row['max_risk']:>10.4f}"
        )

    return county_scores


# ============================================================================
# MAIN PIPELINE
# ============================================================================


def main():
    """Run full uninsurability classification pipeline."""
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
    # DATA PREPARATION
    # =================================================================
    (
        X_train, X_test, y_train, y_test,
        feature_names, imputer, test_meta, thresholds
    ) = prepare_classification_data(df)

    # =================================================================
    # GRADIENT BOOSTING
    # =================================================================
    gb_model, gb_info = fit_gradient_boosting(X_train, y_train, feature_names)
    gb_eval = evaluate_classifier(gb_model, X_test, y_test, "Gradient Boosting")
    gb_cv = cross_validate_classifier(gb_model, X_train, y_train, model_name="Gradient Boosting")

    # =================================================================
    # RANDOM FOREST
    # =================================================================
    rf_model, rf_info = fit_random_forest(X_train, y_train, feature_names)
    rf_eval = evaluate_classifier(rf_model, X_test, y_test, "Random Forest")
    rf_cv = cross_validate_classifier(rf_model, X_train, y_train, model_name="Random Forest")

    # =================================================================
    # SHAP ANALYSIS (on best model by AUC-ROC)
    # =================================================================
    if gb_eval["auc_roc"] >= rf_eval["auc_roc"]:
        best_model = gb_model
        best_name = "Gradient Boosting"
    else:
        best_model = rf_model
        best_name = "Random Forest"

    logger.info(f"\nBest model: {best_name} (AUC-ROC: {max(gb_eval['auc_roc'], rf_eval['auc_roc']):.4f})")

    shap_values, shap_explanation = compute_shap_values(
        best_model, X_test, feature_names
    )

    # =================================================================
    # COUNTY RISK SCORES
    # =================================================================
    county_risk = generate_risk_scores(best_model, df, feature_names, imputer)

    # =================================================================
    # SAVE RESULTS
    # =================================================================
    logger.info("\n" + "=" * 60)
    logger.info("SAVING RESULTS")
    logger.info("=" * 60)

    # Model objects
    with open(MODELS_DIR / "gradient_boosting_classifier.pkl", "wb") as f:
        pickle.dump(gb_model, f)
    logger.info(f"  Saved Gradient Boosting model")

    with open(MODELS_DIR / "random_forest_classifier.pkl", "wb") as f:
        pickle.dump(rf_model, f)
    logger.info(f"  Saved Random Forest model")

    with open(MODELS_DIR / "classifier_imputer.pkl", "wb") as f:
        pickle.dump(imputer, f)

    # SHAP values
    with open(MODELS_DIR / "classifier_shap_values.pkl", "wb") as f:
        pickle.dump(shap_explanation, f)
    logger.info(f"  Saved SHAP values ({best_name})")

    # Comparison metrics
    comparison = pd.DataFrame(
        [
            {
                "model": "Gradient Boosting",
                "auc_roc": gb_eval["auc_roc"],
                "avg_precision": gb_eval["avg_precision"],
                "f1_score": gb_eval["f1_score"],
                "brier_score": gb_eval["brier_score"],
                "optimal_threshold": gb_eval["optimal_threshold"],
                "cv_auc_mean": gb_cv["mean_auc"],
                "cv_auc_std": gb_cv["std_auc"],
                "cv_ap_mean": gb_cv["mean_ap"],
                "cv_ap_std": gb_cv["std_ap"],
            },
            {
                "model": "Random Forest",
                "auc_roc": rf_eval["auc_roc"],
                "avg_precision": rf_eval["avg_precision"],
                "f1_score": rf_eval["f1_score"],
                "brier_score": rf_eval["brier_score"],
                "optimal_threshold": rf_eval["optimal_threshold"],
                "cv_auc_mean": rf_cv["mean_auc"],
                "cv_auc_std": rf_cv["std_auc"],
                "cv_ap_mean": rf_cv["mean_ap"],
                "cv_ap_std": rf_cv["std_ap"],
            },
        ]
    )
    comparison.to_csv(MODELS_DIR / "classifier_comparison_metrics.csv", index=False)
    logger.info("  Saved comparison metrics")

    # Feature importance (sklearn)
    fi_df = pd.DataFrame(
        {
            "feature": feature_names,
            "gb_importance": gb_model.feature_importances_,
            "rf_importance": rf_model.feature_importances_,
        }
    ).sort_values("gb_importance", ascending=False)
    fi_df.to_csv(MODELS_DIR / "classifier_feature_importance.csv", index=False)
    logger.info("  Saved feature importance")

    # SHAP importance
    mean_abs_shap = np.abs(shap_values).mean(axis=0)
    shap_df = pd.DataFrame(
        {"feature": feature_names, "mean_abs_shap": mean_abs_shap}
    ).sort_values("mean_abs_shap", ascending=False)
    shap_df.to_csv(MODELS_DIR / "classifier_shap_importance.csv", index=False)

    # ROC curves
    roc_df = pd.DataFrame(
        {
            "gb_fpr": pd.Series(gb_eval["fpr"]),
            "gb_tpr": pd.Series(gb_eval["tpr"]),
            "rf_fpr": pd.Series(rf_eval["fpr"]),
            "rf_tpr": pd.Series(rf_eval["tpr"]),
        }
    )
    roc_df.to_csv(MODELS_DIR / "classifier_roc_curves.csv", index=False)

    # Predictions
    pred_df = test_meta.copy()
    pred_df["y_test"] = y_test
    pred_df["y_prob_gb"] = gb_eval["y_prob"]
    pred_df["y_prob_rf"] = rf_eval["y_prob"]
    pred_df.to_csv(MODELS_DIR / "classifier_predictions.csv", index=False)
    logger.info("  Saved predictions")

    # Cross-validation fold results
    cv_all = pd.concat(
        [
            gb_cv["fold_results"].assign(model="Gradient Boosting"),
            rf_cv["fold_results"].assign(model="Random Forest"),
        ]
    )
    cv_all.to_csv(MODELS_DIR / "classifier_cv_results.csv", index=False)

    # County risk scores
    county_risk.to_csv(MODELS_DIR / "classifier_risk_scores.csv", index=False)
    logger.info("  Saved county risk scores")

    # Target thresholds for reproducibility
    pd.DataFrame([thresholds]).to_csv(
        MODELS_DIR / "classifier_target_thresholds.csv", index=False
    )

    # Feature names
    pd.DataFrame({"feature": feature_names}).to_csv(
        MODELS_DIR / "classifier_feature_names.csv", index=False
    )

    # =================================================================
    # SUMMARY
    # =================================================================
    logger.info("\n" + "=" * 60)
    logger.info("CLASSIFICATION PIPELINE COMPLETE")
    logger.info("=" * 60)

    logger.info(f"\nGradient Boosting:")
    logger.info(f"  AUC-ROC:       {gb_eval['auc_roc']:.4f}")
    logger.info(f"  Avg Precision:  {gb_eval['avg_precision']:.4f}")
    logger.info(f"  F1 Score:       {gb_eval['f1_score']:.4f}")
    logger.info(f"  CV AUC:         {gb_cv['mean_auc']:.4f} +/- {gb_cv['std_auc']:.4f}")

    logger.info(f"\nRandom Forest:")
    logger.info(f"  AUC-ROC:       {rf_eval['auc_roc']:.4f}")
    logger.info(f"  Avg Precision:  {rf_eval['avg_precision']:.4f}")
    logger.info(f"  F1 Score:       {rf_eval['f1_score']:.4f}")
    logger.info(f"  CV AUC:         {rf_cv['mean_auc']:.4f} +/- {rf_cv['std_auc']:.4f}")

    logger.info(f"\nBest Model: {best_name}")
    logger.info(f"  Counties scored: {len(county_risk):,}")
    logger.info(f"  Top risk county: {county_risk.iloc[0]['county_fips']} ({county_risk.iloc[0]['state']})")

    # Top SHAP features
    logger.info(f"\n--- Top Risk Drivers (SHAP, {best_name}) ---")
    for _, row in shap_df.head(5).iterrows():
        logger.info(f"  {row['feature']}: mean |SHAP| = {row['mean_abs_shap']:.4f}")


if __name__ == "__main__":
    main()
