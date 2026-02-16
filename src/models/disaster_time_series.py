"""
Time series modeling for disaster frequency forecasting.

Models:
  1. SARIMA (Seasonal ARIMA) via statsmodels
  2. Prophet via facebook/prophet

Both are fitted on monthly national disaster counts and optionally
on state-level data for the most-affected states.

Usage:
    python src/models/disaster_time_series.py
"""

import logging
import pickle
import warnings
from itertools import product
from pathlib import Path
from typing import Optional, Tuple, List, Dict

import numpy as np
import pandas as pd
from prophet import Prophet
from sklearn.model_selection import TimeSeriesSplit
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.stattools import adfuller

try:
    from src.utils.config import (
        DATA_PROCESSED,
        MODELS_DIR,
        REPORTS_FIGURES,
        FORECAST_HORIZON_YEARS,
    )
except ImportError:
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from src.utils.config import (
        DATA_PROCESSED,
        MODELS_DIR,
        REPORTS_FIGURES,
        FORECAST_HORIZON_YEARS,
    )

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s"
)
logger = logging.getLogger(__name__)
warnings.filterwarnings("ignore")


# ============================================================================
# DATA PREPARATION
# ============================================================================


def prepare_national_time_series(
    disaster_category: Optional[str] = None,
) -> pd.DataFrame:
    """
    Load and prepare a monthly time series of national disaster counts.

    Parameters
    ----------
    disaster_category : str, optional
        If provided, filter to this disaster category.

    Returns
    -------
    pd.DataFrame
        Columns: date (datetime), disaster_count (int)
    """
    df = pd.read_csv(
        DATA_PROCESSED / "disasters_national_monthly.csv", parse_dates=["date"]
    )

    if disaster_category:
        df = df[df["disaster_category"] == disaster_category]

    # Aggregate across categories
    ts = (
        df.groupby("date")
        .agg(disaster_count=("disaster_count", "sum"))
        .reset_index()
        .sort_values("date")
    )

    # Fill gaps (months with zero disasters)
    full_range = pd.date_range(
        start=ts["date"].min(), end=ts["date"].max(), freq="MS"
    )
    ts = ts.set_index("date").reindex(full_range, fill_value=0)
    ts.index.name = "date"
    ts = ts.reset_index()

    return ts


def train_test_split_temporal(
    ts: pd.DataFrame, test_months: int = 24
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Split time series: last `test_months` months become test set."""
    cutoff = ts["date"].max() - pd.DateOffset(months=test_months)
    train = ts[ts["date"] <= cutoff].copy()
    test = ts[ts["date"] > cutoff].copy()
    logger.info(
        f"Train: {train['date'].min().date()} to {train['date'].max().date()} "
        f"({len(train)} months)"
    )
    logger.info(
        f"Test:  {test['date'].min().date()} to {test['date'].max().date()} "
        f"({len(test)} months)"
    )
    return train, test


# ============================================================================
# STATIONARITY TESTING
# ============================================================================


def test_stationarity(series: pd.Series, name: str = "series") -> dict:
    """Run the Augmented Dickey-Fuller test for stationarity."""
    result = adfuller(series.dropna(), autolag="AIC")

    output = {
        "name": name,
        "adf_statistic": result[0],
        "p_value": result[1],
        "n_lags_used": result[2],
        "n_observations": result[3],
        "critical_values": result[4],
        "is_stationary": result[1] < 0.05,
    }

    logger.info(f"ADF Test for '{name}':")
    logger.info(f"  Test Statistic: {result[0]:.4f}")
    logger.info(f"  p-value: {result[1]:.4f}")
    logger.info(f"  Stationary (p<0.05): {output['is_stationary']}")

    return output


# ============================================================================
# SARIMA MODEL
# ============================================================================


def fit_sarima(
    train: pd.DataFrame,
    order: tuple = (1, 1, 1),
    seasonal_order: tuple = (1, 1, 1, 12),
):
    """
    Fit a SARIMA model on the training data.

    Parameters
    ----------
    train : pd.DataFrame
        Must have 'date' and 'disaster_count' columns.
    order : tuple
        (p, d, q) for non-seasonal component.
    seasonal_order : tuple
        (P, D, Q, m) for seasonal component.

    Returns
    -------
    SARIMAXResults
    """
    logger.info(f"Fitting SARIMA{order}x{seasonal_order}...")

    model = SARIMAX(
        train["disaster_count"],
        order=order,
        seasonal_order=seasonal_order,
        enforce_stationarity=False,
        enforce_invertibility=False,
    )
    results = model.fit(disp=False, maxiter=200)

    logger.info(f"AIC: {results.aic:.2f}, BIC: {results.bic:.2f}")
    return results


def sarima_grid_search(
    train: pd.DataFrame,
    p_range: range = range(0, 3),
    d_range: range = range(0, 2),
    q_range: range = range(0, 3),
    P_range: range = range(0, 2),
    D_range: range = range(0, 2),
    Q_range: range = range(0, 2),
    m: int = 12,
) -> pd.DataFrame:
    """
    Grid search over SARIMA parameters, selecting by AIC.
    Returns DataFrame of all tried combinations sorted by AIC.
    """
    results_list = []

    param_combinations = list(product(p_range, d_range, q_range))
    seasonal_combinations = list(product(P_range, D_range, Q_range))

    total = len(param_combinations) * len(seasonal_combinations)
    logger.info(f"Grid search: {total} combinations to try")

    tried = 0
    for p, d, q in param_combinations:
        for P, D, Q in seasonal_combinations:
            tried += 1
            try:
                model = SARIMAX(
                    train["disaster_count"],
                    order=(p, d, q),
                    seasonal_order=(P, D, Q, m),
                    enforce_stationarity=False,
                    enforce_invertibility=False,
                )
                result = model.fit(disp=False, maxiter=200)
                results_list.append(
                    {
                        "order": (p, d, q),
                        "seasonal_order": (P, D, Q, m),
                        "aic": result.aic,
                        "bic": result.bic,
                    }
                )
                if tried % 20 == 0:
                    logger.info(f"  Progress: {tried}/{total} combinations tried")
            except Exception:
                continue

    results_df = pd.DataFrame(results_list).sort_values("aic")
    if len(results_df) > 0:
        best = results_df.iloc[0]
        logger.info(
            f"Best SARIMA: {best['order']}x{best['seasonal_order']} "
            f"AIC={best['aic']:.2f}"
        )
    return results_df


def sarima_forecast(
    results,
    steps: int,
    test: Optional[pd.DataFrame] = None,
    last_train_date: Optional[pd.Timestamp] = None,
) -> pd.DataFrame:
    """Generate forecast from fitted SARIMA model."""
    forecast = results.get_forecast(steps=steps)
    forecast_mean = forecast.predicted_mean
    conf_int = forecast.conf_int(alpha=0.05)

    forecast_df = pd.DataFrame(
        {
            "forecast": forecast_mean.values,
            "lower_ci": conf_int.iloc[:, 0].values,
            "upper_ci": conf_int.iloc[:, 1].values,
        }
    )

    if test is not None and len(test) >= steps:
        forecast_df["date"] = test["date"].values[:steps]
        forecast_df["actual"] = test["disaster_count"].values[:steps]
    elif last_train_date is not None:
        forecast_df["date"] = pd.date_range(
            start=last_train_date + pd.DateOffset(months=1),
            periods=steps,
            freq="MS",
        )
    else:
        # Fallback: generate sequential dates starting from index
        forecast_df["date"] = pd.date_range(
            start="2025-01-01", periods=steps, freq="MS"
        )

    return forecast_df


# ============================================================================
# PROPHET MODEL
# ============================================================================


def fit_prophet(
    train: pd.DataFrame,
    yearly_seasonality: bool = True,
    changepoint_prior_scale: float = 0.05,
) -> Prophet:
    """
    Fit a Prophet model on training data.
    Prophet requires columns named 'ds' (date) and 'y' (value).
    """
    logger.info("Fitting Prophet model...")

    prophet_df = train.rename(columns={"date": "ds", "disaster_count": "y"})

    model = Prophet(
        yearly_seasonality=yearly_seasonality,
        weekly_seasonality=False,
        daily_seasonality=False,
        changepoint_prior_scale=changepoint_prior_scale,
        seasonality_mode="multiplicative",
    )
    model.fit(prophet_df)
    logger.info("Prophet model fitted.")
    return model


def prophet_forecast(
    model: Prophet,
    periods: int,
    test: Optional[pd.DataFrame] = None,
) -> pd.DataFrame:
    """Generate forecast from fitted Prophet model."""
    future = model.make_future_dataframe(periods=periods, freq="MS")
    forecast = model.predict(future)

    forecast_period = forecast.tail(periods).copy()

    result_df = pd.DataFrame(
        {
            "date": forecast_period["ds"].values,
            "forecast": forecast_period["yhat"].values,
            "lower_ci": forecast_period["yhat_lower"].values,
            "upper_ci": forecast_period["yhat_upper"].values,
        }
    )

    if test is not None and len(test) >= periods:
        result_df["actual"] = test["disaster_count"].values[:periods]

    return result_df


# ============================================================================
# EVALUATION METRICS
# ============================================================================


def calculate_metrics(actual: np.ndarray, predicted: np.ndarray) -> dict:
    """Calculate RMSE, MAE, and MAPE."""
    rmse = np.sqrt(np.mean((actual - predicted) ** 2))
    mae = np.mean(np.abs(actual - predicted))

    # MAPE with protection against division by zero
    nonzero_mask = actual != 0
    if nonzero_mask.sum() > 0:
        mape = (
            np.mean(
                np.abs(
                    (actual[nonzero_mask] - predicted[nonzero_mask])
                    / actual[nonzero_mask]
                )
            )
            * 100
        )
    else:
        mape = np.nan

    return {"rmse": rmse, "mae": mae, "mape": mape}


def time_series_cv(
    ts: pd.DataFrame,
    n_splits: int = 5,
    model_type: str = "sarima",
    **model_kwargs,
) -> List[Dict]:
    """
    Perform time series cross-validation using expanding window.
    Returns list of dicts with fold metrics (RMSE, MAE, MAPE).
    """
    tscv = TimeSeriesSplit(n_splits=n_splits)
    fold_metrics = []

    for fold, (train_idx, test_idx) in enumerate(tscv.split(ts)):
        train_fold = ts.iloc[train_idx]
        test_fold = ts.iloc[test_idx]

        if model_type == "sarima":
            order = model_kwargs.get("order", (1, 1, 1))
            seasonal_order = model_kwargs.get("seasonal_order", (1, 1, 1, 12))
            results = fit_sarima(
                train_fold, order=order, seasonal_order=seasonal_order
            )
            forecast_df = sarima_forecast(
                results, steps=len(test_fold), test=test_fold
            )
        elif model_type == "prophet":
            model = fit_prophet(train_fold, **model_kwargs)
            forecast_df = prophet_forecast(
                model, periods=len(test_fold), test=test_fold
            )
        else:
            raise ValueError(f"Unknown model_type: {model_type}")

        actual = forecast_df["actual"].values
        predicted = forecast_df["forecast"].values
        metrics = calculate_metrics(actual, predicted)

        fold_metrics.append(
            {
                "fold": fold,
                "train_size": len(train_fold),
                "test_size": len(test_fold),
                **metrics,
            }
        )
        logger.info(
            f"Fold {fold}: RMSE={metrics['rmse']:.2f}, "
            f"MAE={metrics['mae']:.2f}, MAPE={metrics['mape']:.1f}%"
        )

    return fold_metrics


def compare_models(
    sarima_metrics: List[Dict], prophet_metrics: List[Dict]
) -> pd.DataFrame:
    """Create comparison summary DataFrame."""
    sarima_summary = pd.DataFrame(sarima_metrics).mean(numeric_only=True)
    prophet_summary = pd.DataFrame(prophet_metrics).mean(numeric_only=True)

    comparison = pd.DataFrame(
        {
            "SARIMA": sarima_summary[["rmse", "mae", "mape"]],
            "Prophet": prophet_summary[["rmse", "mae", "mape"]],
        }
    )
    return comparison


# ============================================================================
# MODEL PERSISTENCE
# ============================================================================


def save_model(model_obj, name: str):
    """Serialize and save a model object."""
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    path = MODELS_DIR / f"{name}.pkl"
    with open(path, "wb") as f:
        pickle.dump(model_obj, f)
    logger.info(f"Model saved to {path}")


# ============================================================================
# MAIN PIPELINE
# ============================================================================


def main():
    """Full pipeline: prepare data, test stationarity, fit, evaluate, forecast."""
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    REPORTS_FIGURES.mkdir(parents=True, exist_ok=True)

    # --- Prepare data ---
    logger.info("=" * 60)
    logger.info("PREPARING TIME SERIES DATA")
    logger.info("=" * 60)
    ts = prepare_national_time_series()
    train, test = train_test_split_temporal(ts, test_months=24)

    # --- Stationarity tests ---
    logger.info("=" * 60)
    logger.info("STATIONARITY TESTS")
    logger.info("=" * 60)
    test_stationarity(train["disaster_count"], name="raw_monthly_disasters")
    diff1 = train["disaster_count"].diff().dropna()
    test_stationarity(diff1, name="first_differenced")

    # --- SARIMA grid search ---
    logger.info("=" * 60)
    logger.info("SARIMA GRID SEARCH")
    logger.info("=" * 60)
    grid_results = sarima_grid_search(
        train,
        p_range=range(0, 3),
        d_range=range(0, 2),
        q_range=range(0, 3),
        P_range=range(0, 2),
        D_range=range(0, 2),
        Q_range=range(0, 2),
        m=12,
    )
    grid_results.to_csv(
        DATA_PROCESSED / "sarima_grid_search_results.csv", index=False
    )

    # Fit best SARIMA
    best = grid_results.iloc[0]
    best_order = best["order"]
    best_seasonal = best["seasonal_order"]

    logger.info("=" * 60)
    logger.info(f"FITTING BEST SARIMA: {best_order}x{best_seasonal}")
    logger.info("=" * 60)
    sarima_results = fit_sarima(
        train, order=best_order, seasonal_order=best_seasonal
    )
    save_model(sarima_results, "sarima_best")

    # SARIMA test forecast
    sarima_forecast_df = sarima_forecast(
        sarima_results, steps=len(test), test=test
    )
    sarima_forecast_df.to_csv(
        DATA_PROCESSED / "sarima_test_forecast.csv", index=False
    )
    sarima_test_metrics = calculate_metrics(
        sarima_forecast_df["actual"].values,
        sarima_forecast_df["forecast"].values,
    )
    logger.info(f"SARIMA test metrics: {sarima_test_metrics}")

    # --- Prophet ---
    logger.info("=" * 60)
    logger.info("FITTING PROPHET MODEL")
    logger.info("=" * 60)
    prophet_model = fit_prophet(train)
    save_model(prophet_model, "prophet_model")

    prophet_forecast_df = prophet_forecast(
        prophet_model, periods=len(test), test=test
    )
    prophet_forecast_df.to_csv(
        DATA_PROCESSED / "prophet_test_forecast.csv", index=False
    )
    prophet_test_metrics = calculate_metrics(
        prophet_forecast_df["actual"].values,
        prophet_forecast_df["forecast"].values,
    )
    logger.info(f"Prophet test metrics: {prophet_test_metrics}")

    # --- Cross-validation ---
    logger.info("=" * 60)
    logger.info("CROSS-VALIDATION (5-fold TimeSeriesSplit)")
    logger.info("=" * 60)

    logger.info("--- SARIMA CV ---")
    sarima_cv = time_series_cv(
        ts,
        n_splits=5,
        model_type="sarima",
        order=best_order,
        seasonal_order=best_seasonal,
    )

    logger.info("--- Prophet CV ---")
    prophet_cv = time_series_cv(ts, n_splits=5, model_type="prophet")

    comparison = compare_models(sarima_cv, prophet_cv)
    comparison.to_csv(DATA_PROCESSED / "model_comparison.csv")
    logger.info(f"\nModel Comparison (CV averages):\n{comparison}")

    # --- Future forecast ---
    logger.info("=" * 60)
    logger.info(f"GENERATING {FORECAST_HORIZON_YEARS}-YEAR FUTURE FORECAST")
    logger.info("=" * 60)

    # Refit on ALL data
    full_sarima = fit_sarima(
        ts, order=best_order, seasonal_order=best_seasonal
    )
    future_steps = FORECAST_HORIZON_YEARS * 12
    last_date = ts["date"].max()
    future_forecast = sarima_forecast(
        full_sarima, steps=future_steps, last_train_date=last_date
    )
    future_forecast.to_csv(
        DATA_PROCESSED / "disaster_forecast_future.csv", index=False
    )
    save_model(full_sarima, "sarima_full_final")

    logger.info("=" * 60)
    logger.info("TIME SERIES MODELING PIPELINE COMPLETE")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
