"""
Model Validation & Documentation for Climate Risk Insurance project.

Performs sensitivity analysis, robustness checks, calibration analysis,
and cross-module summary across all three modeling modules.

Usage:
    python -m src.models.model_validation
"""

import logging
import pickle
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.calibration import calibration_curve
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    f1_score,
    brier_score_loss,
)
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.utils.class_weight import compute_sample_weight

try:
    from src.utils.config import (
        DATA_PROCESSED,
        MODELS_DIR,
        REPORTS_FIGURES,
        FEMA_REGIONS,
    )
except ImportError:
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from src.utils.config import (
        DATA_PROCESSED,
        MODELS_DIR,
        REPORTS_FIGURES,
        FEMA_REGIONS,
    )

from src.models.uninsurability_classifier import (
    create_uninsurability_target,
    PREDICTIVE_FEATURES,
)

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s"
)
logger = logging.getLogger(__name__)

# ============================================================================
# CONSTANTS
# ============================================================================

# Feature groups for ablation study
FEATURE_GROUPS = {
    "disaster": [
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
    ],
    "demographic": [
        "total_population",
        "median_household_income",
        "median_home_value",
        "pct_occupied_housing",
        "unemployment_rate",
    ],
    "economic": [
        "MORTGAGE30US",
        "CSUSHPINSA",
        "CPIAUCSL",
        "FEDFUNDS",
    ],
    "lagged": [
        "total_disasters_lag1",
        "claim_count_lag1",
        "avg_claim_severity_lag1",
    ],
}

# Reverse map: state FIPS -> FEMA region
STATE_TO_FEMA = {}
for _region, _states in FEMA_REGIONS.items():
    for _st in _states:
        STATE_TO_FEMA[_st] = _region

# GB hyperparameters (same as Module 3)
GB_PARAMS = dict(
    n_estimators=300,
    max_depth=5,
    learning_rate=0.1,
    subsample=0.8,
    random_state=42,
)


# ============================================================================
# HELPERS
# ============================================================================


def _prepare_data(
    df,
    severity_q=0.75,
    disaster_q=0.75,
    claims_paid_q=0.75,
    min_signals=2,
    train_years_max=2021,
    feature_list=None,
):
    """
    Prepare X_train, X_test, y_train, y_test for a given configuration.

    Calls create_uninsurability_target() with custom thresholds, then applies
    FEMA dummies, feature selection, median imputation, and temporal split.
    """
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        data, thresholds = create_uninsurability_target(
            df,
            severity_quantile=severity_q,
            disaster_quantile=disaster_q,
            claims_paid_quantile=claims_paid_q,
            min_signals=min_signals,
            train_years_max=train_years_max,
        )

    # FEMA region dummies
    data = pd.get_dummies(
        data, columns=["fema_region"], prefix="fema_region", dtype=float
    )

    # Feature selection
    base_features = feature_list if feature_list else PREDICTIVE_FEATURES
    available = [f for f in base_features if f in data.columns]
    dummies = [
        f
        for f in data.columns
        if f.startswith("fema_region_") and f != "fema_region_0"
    ]
    all_features = available + dummies

    y = data["uninsurability_risk"].values
    X = data[all_features].astype(float).copy()
    feature_names = list(X.columns)

    # Temporal split
    years = data["year"].values
    train_mask = years <= train_years_max
    test_mask = years > train_years_max

    X_train, X_test = X.values[train_mask], X.values[test_mask]
    y_train, y_test = y[train_mask], y[test_mask]

    # Median imputation
    imputer = SimpleImputer(strategy="median")
    X_train = imputer.fit_transform(X_train)
    X_test = imputer.transform(X_test)

    return X_train, X_test, y_train, y_test, feature_names


def _fit_and_eval_gb(X_train, y_train, X_test, y_test):
    """Fit GB classifier and return test metrics dict."""
    model = GradientBoostingClassifier(**GB_PARAMS)
    weights = compute_sample_weight("balanced", y_train)
    model.fit(X_train, y_train, sample_weight=weights)

    y_prob = model.predict_proba(X_test)[:, 1]

    # Guard against single-class test sets
    if len(np.unique(y_test)) < 2:
        return {"auc_roc": np.nan, "avg_precision": np.nan, "f1_score": np.nan}

    auc = roc_auc_score(y_test, y_prob)
    ap = average_precision_score(y_test, y_prob)
    y_pred = (y_prob >= 0.5).astype(int)
    f1 = f1_score(y_test, y_pred, zero_division=0)

    return {"auc_roc": auc, "avg_precision": ap, "f1_score": f1}


# ============================================================================
# SECTION 1: SENSITIVITY ANALYSIS
# ============================================================================


def sensitivity_target_thresholds(df):
    """
    Vary composite target quantiles and min_signals, refit GB for each.

    Grid: quantiles [0.50, 0.60, 0.75, 0.90] × min_signals [1, 2, 3, 4]
    """
    logger.info("1a. Target Threshold Sensitivity Analysis...")
    quantiles = [0.50, 0.60, 0.75, 0.90]
    min_signals_list = [1, 2, 3, 4]
    results = []

    for q in quantiles:
        for ms in min_signals_list:
            try:
                X_tr, X_te, y_tr, y_te, _ = _prepare_data(
                    df, severity_q=q, disaster_q=q, claims_paid_q=q, min_signals=ms
                )
                pos_train = y_tr.mean()
                pos_test = y_te.mean()

                # Skip degenerate cases
                if y_te.sum() < 30 or pos_test < 0.01 or pos_test > 0.50:
                    logger.warning(
                        f"  Skipping q={q}, ms={ms}: pos_rate_test={pos_test:.1%} "
                        f"(n_pos={int(y_te.sum())})"
                    )
                    results.append({
                        "quantile": q, "min_signals": ms,
                        "pos_rate_train": pos_train, "pos_rate_test": pos_test,
                        "auc_roc": np.nan, "avg_precision": np.nan,
                        "f1_score": np.nan, "n_train": len(y_tr), "n_test": len(y_te),
                    })
                    continue

                metrics = _fit_and_eval_gb(X_tr, y_tr, X_te, y_te)
                metrics.update({
                    "quantile": q, "min_signals": ms,
                    "pos_rate_train": pos_train, "pos_rate_test": pos_test,
                    "n_train": len(y_tr), "n_test": len(y_te),
                })
                results.append(metrics)
                logger.info(
                    f"  q={q}, ms={ms}: AUC={metrics['auc_roc']:.3f}, "
                    f"pos_rate={pos_test:.1%}"
                )
            except Exception as e:
                logger.warning(f"  Error at q={q}, ms={ms}: {e}")
                results.append({
                    "quantile": q, "min_signals": ms,
                    "pos_rate_train": np.nan, "pos_rate_test": np.nan,
                    "auc_roc": np.nan, "avg_precision": np.nan,
                    "f1_score": np.nan, "n_train": 0, "n_test": 0,
                })

    out = pd.DataFrame(results)
    out.to_csv(MODELS_DIR / "validation_sensitivity_thresholds.csv", index=False)
    logger.info(f"  Saved: validation_sensitivity_thresholds.csv ({len(out)} rows)")
    return out


def sensitivity_surge_threshold(df):
    """
    Vary the logistic regression YoY surge threshold (25%, 50%, 75%, 100%).
    """
    logger.info("1b. Claims Surge Threshold Sensitivity...")
    thresholds = [0.25, 0.50, 0.75, 1.00]
    results = []

    for t in thresholds:
        data = df.copy()

        # Recreate surge target with custom threshold
        data["claim_count_prev"] = data.groupby("county_fips")["claim_count"].shift(1)
        data["claims_change_yoy"] = (
            (data["claim_count"] - data["claim_count_prev"]) / data["claim_count_prev"]
        )
        data["claims_surge_flag"] = (data["claims_change_yoy"] > t).astype(float)
        data = data[data["claim_count_prev"] > 0].copy()

        # FEMA dummies
        data = pd.get_dummies(
            data, columns=["fema_region"], prefix="fema_region", dtype=float
        )

        # Lagged features for logistic regression
        logistic_features = [
            "total_disasters_lag1", "claim_count_lag1", "avg_claim_severity_lag1",
            "cum_disasters_3yr", "total_population", "median_household_income",
            "median_home_value", "unemployment_rate",
            "MORTGAGE30US", "CSUSHPINSA", "CPIAUCSL", "FEDFUNDS",
        ]
        available = [f for f in logistic_features if f in data.columns]
        dummies = [f for f in data.columns if f.startswith("fema_region_") and f != "fema_region_0"]
        all_feats = available + dummies

        y = data["claims_surge_flag"].values
        X = data[all_feats].astype(float)
        mask = X.notna().all(axis=1)
        X = X[mask]
        y = y[mask.values]
        years = data.loc[mask, "year"].values

        train_mask = years <= 2021
        test_mask = years > 2021
        X_tr, X_te = X.values[train_mask], X.values[test_mask]
        y_tr, y_te = y[train_mask], y[test_mask]

        scaler = StandardScaler()
        X_tr = scaler.fit_transform(X_tr)
        X_te = scaler.transform(X_te)

        if y_te.sum() < 10 or len(np.unique(y_te)) < 2:
            logger.warning(f"  Skipping threshold={t}: insufficient positives")
            results.append({
                "surge_threshold": t, "pos_rate": y_tr.mean(),
                "auc_roc": np.nan, "avg_precision": np.nan, "n_test": len(y_te),
            })
            continue

        model = LogisticRegression(
            class_weight="balanced", max_iter=1000, random_state=42
        )
        model.fit(X_tr, y_tr)
        y_prob = model.predict_proba(X_te)[:, 1]

        results.append({
            "surge_threshold": t,
            "pos_rate": y_tr.mean(),
            "auc_roc": roc_auc_score(y_te, y_prob),
            "avg_precision": average_precision_score(y_te, y_prob),
            "n_test": len(y_te),
        })
        logger.info(
            f"  threshold={t}: AUC={results[-1]['auc_roc']:.3f}, "
            f"surge_rate={results[-1]['pos_rate']:.1%}"
        )

    out = pd.DataFrame(results)
    out.to_csv(MODELS_DIR / "validation_sensitivity_surge.csv", index=False)
    logger.info(f"  Saved: validation_sensitivity_surge.csv ({len(out)} rows)")
    return out


def feature_ablation_study(df):
    """
    Remove each feature group and measure AUC-ROC drop vs baseline.
    Groups: disaster, demographic, economic, lagged, fema_dummies.
    """
    logger.info("1c. Feature Ablation Study...")

    # Baseline with all features
    X_tr, X_te, y_tr, y_te, feat_names = _prepare_data(df)
    baseline = _fit_and_eval_gb(X_tr, y_tr, X_te, y_te)
    baseline_auc = baseline["auc_roc"]

    results = [{
        "group_removed": "none (baseline)",
        "n_features_removed": 0,
        "n_features_remaining": len(feat_names),
        "auc_roc": baseline_auc,
        "auc_drop": 0.0,
        "pct_drop": 0.0,
    }]
    logger.info(f"  Baseline AUC: {baseline_auc:.3f} ({len(feat_names)} features)")

    # Ablate each group
    for group_name, group_features in FEATURE_GROUPS.items():
        reduced = [f for f in PREDICTIVE_FEATURES if f not in group_features]
        X_tr, X_te, y_tr, y_te, feat_names_r = _prepare_data(
            df, feature_list=reduced
        )
        metrics = _fit_and_eval_gb(X_tr, y_tr, X_te, y_te)
        drop = baseline_auc - metrics["auc_roc"]
        pct = (drop / baseline_auc) * 100 if baseline_auc > 0 else 0

        results.append({
            "group_removed": group_name,
            "n_features_removed": len(group_features),
            "n_features_remaining": len(feat_names_r),
            "auc_roc": metrics["auc_roc"],
            "auc_drop": drop,
            "pct_drop": pct,
        })
        logger.info(
            f"  Remove {group_name} ({len(group_features)} features): "
            f"AUC={metrics['auc_roc']:.3f} (drop={drop:.3f}, {pct:.1f}%)"
        )

    # Ablate FEMA dummies
    X_tr_no_fema, X_te_no_fema, y_tr, y_te, feat_names_nf = _prepare_data(df)
    # Remove FEMA columns from the prepared data
    fema_cols = [i for i, f in enumerate(feat_names_nf) if f.startswith("fema_region_")]
    non_fema = [i for i in range(len(feat_names_nf)) if i not in fema_cols]
    if non_fema:
        metrics = _fit_and_eval_gb(
            X_tr_no_fema[:, non_fema], y_tr,
            X_te_no_fema[:, non_fema], y_te,
        )
        drop = baseline_auc - metrics["auc_roc"]
        pct = (drop / baseline_auc) * 100 if baseline_auc > 0 else 0
        results.append({
            "group_removed": "fema_dummies",
            "n_features_removed": len(fema_cols),
            "n_features_remaining": len(non_fema),
            "auc_roc": metrics["auc_roc"],
            "auc_drop": drop,
            "pct_drop": pct,
        })
        logger.info(
            f"  Remove fema_dummies ({len(fema_cols)} features): "
            f"AUC={metrics['auc_roc']:.3f} (drop={drop:.3f}, {pct:.1f}%)"
        )

    out = pd.DataFrame(results)
    out.to_csv(MODELS_DIR / "validation_feature_ablation.csv", index=False)
    logger.info(f"  Saved: validation_feature_ablation.csv ({len(out)} rows)")
    return out


# ============================================================================
# SECTION 2: ROBUSTNESS CHECKS
# ============================================================================


def geographic_cv(df):
    """
    Leave-one-FEMA-region-out cross-validation (10 geographic folds).
    """
    logger.info("2a. Geographic (Leave-One-Region-Out) Cross-Validation...")

    # Create target and prepare full dataset
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        data, _ = create_uninsurability_target(df)

    data = pd.get_dummies(
        data, columns=["fema_region"], prefix="fema_region", dtype=float
    )

    # Assign FEMA region from state FIPS
    data["fema_region_id"] = data["fipsStateCode"].astype(str).str.zfill(2).map(
        STATE_TO_FEMA
    )

    # Use only training period for geographic CV
    data_train_period = data[data["year"] <= 2021].copy()

    available = [f for f in PREDICTIVE_FEATURES if f in data_train_period.columns]
    dummies = [
        f for f in data_train_period.columns
        if f.startswith("fema_region_") and f != "fema_region_0"
    ]
    all_features = available + dummies

    results = []
    for region in sorted(data_train_period["fema_region_id"].dropna().unique()):
        test_mask = data_train_period["fema_region_id"] == region
        train_mask = ~test_mask

        if test_mask.sum() < 50:
            logger.warning(f"  Region {region}: only {test_mask.sum()} rows, skipping")
            continue

        X_all = data_train_period[all_features].astype(float).values
        y_all = data_train_period["uninsurability_risk"].values

        X_tr = X_all[train_mask.values]
        X_te = X_all[test_mask.values]
        y_tr = y_all[train_mask.values]
        y_te = y_all[test_mask.values]

        # Impute
        imputer = SimpleImputer(strategy="median")
        X_tr = imputer.fit_transform(X_tr)
        X_te = imputer.transform(X_te)

        if len(np.unique(y_te)) < 2 or y_te.sum() < 10:
            logger.warning(
                f"  Region {region}: insufficient positive test samples "
                f"({int(y_te.sum())}), skipping"
            )
            continue

        metrics = _fit_and_eval_gb(X_tr, y_tr, X_te, y_te)
        metrics.update({
            "fema_region": int(region),
            "n_train": len(y_tr),
            "n_test": len(y_te),
            "pos_rate_test": y_te.mean(),
        })
        results.append(metrics)
        logger.info(
            f"  Region {int(region)}: AUC={metrics['auc_roc']:.3f}, "
            f"n_test={len(y_te)}, pos_rate={y_te.mean():.1%}"
        )

    out = pd.DataFrame(results)
    out.to_csv(MODELS_DIR / "validation_geographic_cv.csv", index=False)
    logger.info(f"  Saved: validation_geographic_cv.csv ({len(out)} rows)")
    return out


def temporal_stability(df):
    """
    Expanding-window temporal validation:
    Train 2004-2016 → test 2017, Train 2004-2017 → test 2018, ... through 2023.
    Recomputes target thresholds per window.
    """
    logger.info("2b. Temporal Stability (Expanding Window)...")
    results = []

    for test_year in range(2017, 2024):
        train_end = test_year - 1

        try:
            X_tr, X_te, y_tr, y_te, _ = _prepare_data(
                df, train_years_max=train_end
            )

            # Filter test to only the single test year
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                data_tmp, _ = create_uninsurability_target(
                    df, train_years_max=train_end
                )
            test_year_mask = data_tmp["year"] == test_year
            train_period_mask = data_tmp["year"] <= train_end

            # Re-prepare with correct masks
            data_tmp = pd.get_dummies(
                data_tmp, columns=["fema_region"], prefix="fema_region", dtype=float
            )
            avail = [f for f in PREDICTIVE_FEATURES if f in data_tmp.columns]
            dums = [
                f for f in data_tmp.columns
                if f.startswith("fema_region_") and f != "fema_region_0"
            ]
            all_f = avail + dums

            X_all = data_tmp[all_f].astype(float).values
            y_all = data_tmp["uninsurability_risk"].values

            X_tr = X_all[train_period_mask.values]
            X_te = X_all[test_year_mask.values]
            y_tr = y_all[train_period_mask.values]
            y_te = y_all[test_year_mask.values]

            imputer = SimpleImputer(strategy="median")
            X_tr = imputer.fit_transform(X_tr)
            X_te = imputer.transform(X_te)

            if len(np.unique(y_te)) < 2 or y_te.sum() < 10:
                logger.warning(
                    f"  Year {test_year}: insufficient positives ({int(y_te.sum())})"
                )
                results.append({
                    "train_end_year": train_end, "test_year": test_year,
                    "n_train": len(y_tr), "n_test": len(y_te),
                    "pos_rate_train": y_tr.mean(), "pos_rate_test": y_te.mean(),
                    "auc_roc": np.nan, "avg_precision": np.nan, "f1_score": np.nan,
                })
                continue

            metrics = _fit_and_eval_gb(X_tr, y_tr, X_te, y_te)
            metrics.update({
                "train_end_year": train_end,
                "test_year": test_year,
                "n_train": len(y_tr),
                "n_test": len(y_te),
                "pos_rate_train": y_tr.mean(),
                "pos_rate_test": y_te.mean(),
            })
            results.append(metrics)
            logger.info(
                f"  Train <=  {train_end}, Test {test_year}: "
                f"AUC={metrics['auc_roc']:.3f}, pos_rate={y_te.mean():.1%}"
            )
        except Exception as e:
            logger.warning(f"  Error for test_year={test_year}: {e}")
            results.append({
                "train_end_year": train_end, "test_year": test_year,
                "n_train": 0, "n_test": 0,
                "pos_rate_train": np.nan, "pos_rate_test": np.nan,
                "auc_roc": np.nan, "avg_precision": np.nan, "f1_score": np.nan,
            })

    out = pd.DataFrame(results)
    out.to_csv(MODELS_DIR / "validation_temporal_stability.csv", index=False)
    logger.info(f"  Saved: validation_temporal_stability.csv ({len(out)} rows)")
    return out


def distribution_comparison(df):
    """
    KS tests comparing train vs test distributions for each predictive feature.
    """
    logger.info("2c. Train/Test Distribution Comparison (KS Tests)...")

    train = df[df["year"] <= 2021]
    test = df[df["year"] > 2021]
    results = []

    for feat in PREDICTIVE_FEATURES:
        if feat not in df.columns:
            continue

        train_vals = train[feat].dropna().values
        test_vals = test[feat].dropna().values

        if len(train_vals) < 10 or len(test_vals) < 10:
            continue

        ks_stat, ks_pval = stats.ks_2samp(train_vals, test_vals)
        results.append({
            "feature": feat,
            "ks_statistic": ks_stat,
            "ks_pvalue": ks_pval,
            "train_mean": train_vals.mean(),
            "train_std": train_vals.std(),
            "test_mean": test_vals.mean(),
            "test_std": test_vals.std(),
            "significant_shift": ks_pval < 0.05,
        })

    out = pd.DataFrame(results).sort_values("ks_statistic", ascending=False)
    out.to_csv(MODELS_DIR / "validation_distribution_comparison.csv", index=False)

    n_sig = out["significant_shift"].sum()
    logger.info(
        f"  {n_sig}/{len(out)} features show significant distribution shift (p<0.05)"
    )
    logger.info(f"  Saved: validation_distribution_comparison.csv ({len(out)} rows)")
    return out


def prediction_stability(df):
    """
    5-fold stratified CV measuring per-fold prediction distribution consistency.
    """
    logger.info("2d. Prediction Stability Across CV Folds...")

    X_tr, X_te, y_tr, y_te, feat_names = _prepare_data(df)

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    results = []

    for fold_idx, (train_idx, val_idx) in enumerate(cv.split(X_tr, y_tr), 1):
        X_fold_tr = X_tr[train_idx]
        X_fold_val = X_tr[val_idx]
        y_fold_tr = y_tr[train_idx]
        y_fold_val = y_tr[val_idx]

        model = GradientBoostingClassifier(**GB_PARAMS)
        weights = compute_sample_weight("balanced", y_fold_tr)
        model.fit(X_fold_tr, y_fold_tr, sample_weight=weights)

        y_prob = model.predict_proba(X_fold_val)[:, 1]

        if len(np.unique(y_fold_val)) < 2:
            continue

        results.append({
            "fold": fold_idx,
            "auc_roc": roc_auc_score(y_fold_val, y_prob),
            "avg_precision": average_precision_score(y_fold_val, y_prob),
            "f1_score": f1_score(y_fold_val, (y_prob >= 0.5).astype(int), zero_division=0),
            "pred_mean": y_prob.mean(),
            "pred_std": y_prob.std(),
            "pred_median": np.median(y_prob),
            "n_val": len(y_fold_val),
            "pos_rate": y_fold_val.mean(),
        })
        logger.info(
            f"  Fold {fold_idx}: AUC={results[-1]['auc_roc']:.3f}, "
            f"pred_mean={y_prob.mean():.3f}"
        )

    out = pd.DataFrame(results)

    # Add summary row
    summary = pd.DataFrame([{
        "fold": "mean±std",
        "auc_roc": f"{out['auc_roc'].mean():.3f}±{out['auc_roc'].std():.3f}",
        "avg_precision": f"{out['avg_precision'].mean():.3f}±{out['avg_precision'].std():.3f}",
        "f1_score": f"{out['f1_score'].mean():.3f}±{out['f1_score'].std():.3f}",
        "pred_mean": out["pred_mean"].mean(),
        "pred_std": out["pred_std"].mean(),
        "pred_median": out["pred_median"].mean(),
        "n_val": "",
        "pos_rate": "",
    }])
    out = pd.concat([out, summary], ignore_index=True)

    out.to_csv(MODELS_DIR / "validation_prediction_stability.csv", index=False)
    logger.info(f"  Saved: validation_prediction_stability.csv ({len(out)} rows)")
    return out


# ============================================================================
# SECTION 3: CALIBRATION ANALYSIS
# ============================================================================


def calibration_analysis():
    """
    Calibration curves and Expected Calibration Error using saved predictions.
    """
    logger.info("3. Calibration Analysis...")

    preds = pd.read_csv(MODELS_DIR / "classifier_predictions.csv")
    results_curves = []
    results_ece = []

    for model_name, prob_col in [
        ("Gradient Boosting", "y_prob_gb"),
        ("Random Forest", "y_prob_rf"),
    ]:
        if prob_col not in preds.columns:
            logger.warning(f"  {prob_col} not found in predictions, skipping")
            continue

        y_true = preds["y_test"].values
        y_prob = preds[prob_col].values

        # Calibration curve (10 bins)
        fraction_pos, mean_pred = calibration_curve(
            y_true, y_prob, n_bins=10, strategy="uniform"
        )

        for i in range(len(fraction_pos)):
            results_curves.append({
                "model": model_name,
                "bin": i + 1,
                "fraction_positive": fraction_pos[i],
                "mean_predicted": mean_pred[i],
            })

        # Expected Calibration Error
        bin_edges = np.linspace(0, 1, 11)
        ece = 0.0
        for i in range(10):
            mask = (y_prob >= bin_edges[i]) & (y_prob < bin_edges[i + 1])
            if mask.sum() == 0:
                continue
            bin_acc = y_true[mask].mean()
            bin_conf = y_prob[mask].mean()
            ece += mask.sum() / len(y_prob) * abs(bin_acc - bin_conf)

        results_ece.append({
            "model": model_name,
            "ece": ece,
            "brier_score": brier_score_loss(y_true, y_prob),
        })
        logger.info(f"  {model_name}: ECE={ece:.4f}, Brier={results_ece[-1]['brier_score']:.4f}")

    curves_df = pd.DataFrame(results_curves)
    ece_df = pd.DataFrame(results_ece)
    curves_df.to_csv(MODELS_DIR / "validation_calibration_curves.csv", index=False)
    ece_df.to_csv(MODELS_DIR / "validation_calibration_ece.csv", index=False)
    logger.info(f"  Saved: validation_calibration_curves.csv, validation_calibration_ece.csv")
    return curves_df, ece_df


# ============================================================================
# SECTION 4: CROSS-MODULE SUMMARY
# ============================================================================


def cross_module_summary():
    """
    Consolidate metrics from all models into a unified report.
    """
    logger.info("4a. Cross-Module Summary...")
    rows = []

    # Module 1: Time Series
    ts_path = DATA_PROCESSED / "model_comparison.csv"
    if ts_path.exists():
        ts = pd.read_csv(ts_path)
        for _, r in ts.iterrows():
            rows.append({
                "module": 1,
                "model_name": r.get("model", "SARIMA/Prophet"),
                "primary_metric": "RMSE",
                "metric_value": r.get("rmse", r.get("RMSE", np.nan)),
                "cv_metric": np.nan,
                "cv_std": np.nan,
                "notes": "Time series forecasting",
            })
    else:
        logger.warning("  Module 1 model_comparison.csv not found")

    # Module 2: GLMs
    glm_path = MODELS_DIR / "glm_comparison_metrics.csv"
    if glm_path.exists():
        glm = pd.read_csv(glm_path)
        for _, r in glm.iterrows():
            rows.append({
                "module": 2,
                "model_name": r.get("model", "GLM"),
                "primary_metric": "Test RMSE",
                "metric_value": r.get("test_rmse", np.nan),
                "cv_metric": np.nan,
                "cv_std": np.nan,
                "notes": f"AIC={r.get('aic', 'N/A')}",
            })

    # Module 2: Logistic
    log_path = MODELS_DIR / "logistic_metrics.csv"
    if log_path.exists():
        log = pd.read_csv(log_path)
        if len(log) > 0:
            r = log.iloc[0]
            rows.append({
                "module": 2,
                "model_name": "Logistic Regression",
                "primary_metric": "AUC-ROC",
                "metric_value": r.get("auc_roc", np.nan),
                "cv_metric": r.get("cv_auc_mean", np.nan),
                "cv_std": r.get("cv_auc_std", np.nan),
                "notes": "Claims surge prediction",
            })

    # Module 3: Classifiers
    clf_path = MODELS_DIR / "classifier_comparison_metrics.csv"
    if clf_path.exists():
        clf = pd.read_csv(clf_path)
        for _, r in clf.iterrows():
            rows.append({
                "module": 3,
                "model_name": r.get("model", "Classifier"),
                "primary_metric": "AUC-ROC",
                "metric_value": r.get("auc_roc", np.nan),
                "cv_metric": r.get("cv_auc_mean", np.nan),
                "cv_std": r.get("cv_auc_std", np.nan),
                "notes": f"Brier={r.get('brier_score', 'N/A')}",
            })

    out = pd.DataFrame(rows)
    out.to_csv(MODELS_DIR / "validation_cross_module_summary.csv", index=False)
    logger.info(f"  Saved: validation_cross_module_summary.csv ({len(out)} rows)")
    return out


def performance_by_subset():
    """
    Module 3 classifier performance by FEMA region and by test year.
    Uses saved predictions from classifier_predictions.csv.
    """
    logger.info("4b. Performance by Subset...")

    preds = pd.read_csv(
        MODELS_DIR / "classifier_predictions.csv",
        dtype={"county_fips": str},
    )

    # Map county_fips to FEMA region (first 2 digits = state FIPS)
    if "county_fips" in preds.columns:
        preds["fema_region"] = (
            preds["county_fips"].astype(str).str.zfill(5).str[:2].map(STATE_TO_FEMA)
        )

    # By FEMA region
    region_results = []
    for region in sorted(preds["fema_region"].dropna().unique()):
        mask = preds["fema_region"] == region
        subset = preds[mask]
        y_true = subset["y_test"].values
        y_prob = subset["y_prob_gb"].values

        if len(np.unique(y_true)) < 2 or y_true.sum() < 5:
            continue

        region_results.append({
            "fema_region": int(region),
            "auc_roc": roc_auc_score(y_true, y_prob),
            "n_samples": len(y_true),
            "pos_rate": y_true.mean(),
        })

    region_df = pd.DataFrame(region_results)
    region_df.to_csv(MODELS_DIR / "validation_performance_by_region.csv", index=False)
    logger.info(f"  By region: {len(region_df)} FEMA regions")

    # By test year
    period_results = []
    if "year" in preds.columns:
        for year in sorted(preds["year"].unique()):
            mask = preds["year"] == year
            subset = preds[mask]
            y_true = subset["y_test"].values
            y_prob = subset["y_prob_gb"].values

            if len(np.unique(y_true)) < 2 or y_true.sum() < 5:
                continue

            period_results.append({
                "year": int(year),
                "auc_roc": roc_auc_score(y_true, y_prob),
                "n_samples": len(y_true),
                "pos_rate": y_true.mean(),
            })

    period_df = pd.DataFrame(period_results)
    period_df.to_csv(MODELS_DIR / "validation_performance_by_period.csv", index=False)
    logger.info(f"  By period: {len(period_df)} years")

    logger.info(
        f"  Saved: validation_performance_by_region.csv, "
        f"validation_performance_by_period.csv"
    )
    return region_df, period_df


# ============================================================================
# MAIN
# ============================================================================


def main():
    """Run full model validation pipeline."""
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    REPORTS_FIGURES.mkdir(parents=True, exist_ok=True)

    logger.info("=" * 70)
    logger.info("MODULE 4: MODEL VALIDATION & DOCUMENTATION")
    logger.info("=" * 70)

    # Load panel data
    panel_path = DATA_PROCESSED / "county_year_panel_glm_ready.csv"
    logger.info(f"Loading panel dataset from {panel_path}...")
    df = pd.read_csv(panel_path, dtype={"county_fips": str})
    logger.info(f"Panel: {len(df):,} rows, {len(df.columns)} columns")

    # Section 1: Sensitivity Analysis
    logger.info("\n" + "=" * 70)
    logger.info("SECTION 1: SENSITIVITY ANALYSIS")
    logger.info("=" * 70)

    threshold_results = sensitivity_target_thresholds(df)
    surge_results = sensitivity_surge_threshold(df)
    ablation_results = feature_ablation_study(df)

    # Section 2: Robustness Checks
    logger.info("\n" + "=" * 70)
    logger.info("SECTION 2: ROBUSTNESS CHECKS")
    logger.info("=" * 70)

    geo_results = geographic_cv(df)
    temporal_results = temporal_stability(df)
    dist_results = distribution_comparison(df)
    stability_results = prediction_stability(df)

    # Section 3: Calibration
    logger.info("\n" + "=" * 70)
    logger.info("SECTION 3: CALIBRATION ANALYSIS")
    logger.info("=" * 70)

    cal_curves, cal_ece = calibration_analysis()

    # Section 4: Cross-Module Summary
    logger.info("\n" + "=" * 70)
    logger.info("SECTION 4: CROSS-MODULE SUMMARY")
    logger.info("=" * 70)

    summary = cross_module_summary()
    region_perf, period_perf = performance_by_subset()

    # Final summary
    logger.info("\n" + "=" * 70)
    logger.info("VALIDATION PIPELINE COMPLETE")
    logger.info("=" * 70)
    logger.info(f"  Threshold sensitivity: {len(threshold_results)} configurations")
    logger.info(f"  Surge sensitivity: {len(surge_results)} thresholds")
    logger.info(f"  Feature ablation: {len(ablation_results)} groups")
    logger.info(f"  Geographic CV: {len(geo_results)} FEMA regions")
    logger.info(f"  Temporal stability: {len(temporal_results)} windows")
    logger.info(f"  Distribution tests: {len(dist_results)} features")
    logger.info(f"  Prediction stability: {len(stability_results)} folds")
    logger.info(f"  Calibration: {len(cal_ece)} models")
    logger.info(f"  Cross-module summary: {len(summary)} models")
    logger.info(f"\nAll outputs saved to: {MODELS_DIR}")


if __name__ == "__main__":
    main()
