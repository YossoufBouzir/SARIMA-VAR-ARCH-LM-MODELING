"""
sarima_model.py
===============
SARIMA Modeling Pipeline:
  1. Exploratory Data Analysis
  2. Stationarity Testing (ADF, KPSS, PP)
  3. Model Identification (ACF, PACF, Grid Search)
  4. Estimation & Diagnostics
  5. Forecasting
"""

import logging
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.stats.diagnostic import acorr_ljungbox
from scipy.stats import jarque_bera
from utils import run_stationarity_suite, descriptive_stats, ic_table

warnings.filterwarnings('ignore')
logger = logging.getLogger(__name__)


def exploratory_analysis(series: pd.Series) -> dict:
    """
    Phase 1: Exploratory Data Analysis
      - Descriptive statistics
      - Time plot
      - Seasonal decomposition
      - Distribution checks
    """
    logger.info(f"EDA for {series.name} ...")
    desc = descriptive_stats(series)
    print(f"\n=== Descriptive Stats: {series.name} ===")
    print(desc.to_string())

    # Seasonal decomposition
    from statsmodels.tsa.seasonal import seasonal_decompose
    decomp = seasonal_decompose(series, model='additive', period=12,
                                 extrapolate_trend='freq')

    return {
        'descriptive': desc,
        'decomposition': decomp,
        'series': series
    }


def identify_orders(series: pd.Series, max_p: int = 3, max_q: int = 3,
                    seasonal: bool = True) -> dict:
    """
    Phase 3a: Model Identification using ACF/PACF and Grid Search.

    Returns:
      - ACF/PACF plots
      - AIC/BIC grid
      - Recommended orders
    """
    logger.info(f"Order identification for {series.name} ...")

    # Differencing if needed (based on stationarity suite)
    # For simplicity, assume first difference is needed if not stationary
    series_diff = series.diff().dropna()

    # ACF/PACF
    fig, ax = plt.subplots(2, 1, figsize=(10, 6))
    plot_acf(series_diff, lags=36, ax=ax[0])
    ax[0].set_title(f'ACF: {series.name} (1st Diff)')
    plot_pacf(series_diff, lags=36, ax=ax[1])
    ax[1].set_title(f'PACF: {series.name} (1st Diff)')
    plt.tight_layout()
    plt.savefig(f'outputs/figures/{series.name}_acf_pacf.png', dpi=150)
    plt.close()

    # Grid search for non-seasonal ARIMA (p,1,q)
    aic_grid, bic_grid = {}, {}
    best_aic, best_order = np.inf, None

    logger.info("Running SARIMA grid search ...")
    for p in range(0, max_p + 1):
        for q in range(0, max_q + 1):
            order = (p, 1, q)
            seasonal_order = (1, 1, 1, 12) if seasonal else (0, 0, 0, 0)
            try:
                model = SARIMAX(series, order=order,
                                seasonal_order=seasonal_order,
                                enforce_stationarity=False,
                                enforce_invertibility=False)
                fit = model.fit(disp=False, maxiter=200)
                aic_grid[order] = fit.aic
                bic_grid[order] = fit.bic
                if fit.aic < best_aic:
                    best_aic = fit.aic
                    best_order = order
            except Exception as e:
                logger.debug(f"Order {order} failed: {e}")
                continue

    ic_df = ic_table(aic_grid, bic_grid)
    print(f"\n=== IC Table: {series.name} ===")
    print(ic_df.head(10).to_string())

    return {
        'ic_table': ic_df,
        'best_order': best_order,
        'best_aic': best_aic
    }


def estimate_sarima(series: pd.Series, order: tuple,
                    seasonal_order: tuple = (1, 1, 1, 12)) -> dict:
    """
    Phase 3b: Estimation & Diagnostics.

    Fits SARIMAX model, performs diagnostics:
      - Parameter significance
      - Ljung-Box test on residuals
      - Jarque-Bera normality test
      - Residual ACF
    """
    logger.info(f"Estimating SARIMA{order}x{seasonal_order} for {series.name} ...")

    model = SARIMAX(series, order=order, seasonal_order=seasonal_order,
                    enforce_stationarity=False, enforce_invertibility=False)
    fit = model.fit(disp=False, maxiter=500)

    print(f"\n=== SARIMA{order}x{seasonal_order} Summary: {series.name} ===")
    print(fit.summary())

    # Diagnostics
    residuals = fit.resid
    lb_stat = acorr_ljungbox(residuals, lags=[10, 20, 30], return_df=True)
    jb_stat, jb_p = jarque_bera(residuals)

    print("\n--- Ljung-Box Test (Residual Autocorrelation) ---")
    print(lb_stat.to_string())
    print(f"\n--- Jarque-Bera Normality Test ---")
    print(f"Statistic: {jb_stat:.4f}, p-value: {jb_p:.4f}")

    # Diagnostic plots
    fig, ax = plt.subplots(2, 2, figsize=(12, 8))
    residuals.plot(ax=ax[0, 0], title='Residuals Over Time')
    ax[0, 0].axhline(0, color='red', linestyle='--')

    residuals.plot(kind='hist', bins=30, ax=ax[0, 1], title='Residual Histogram')
    ax[0, 1].axvline(0, color='red', linestyle='--')

    plot_acf(residuals, lags=36, ax=ax[1, 0])
    ax[1, 0].set_title('Residual ACF')

    from scipy.stats import probplot
    probplot(residuals, dist="norm", plot=ax[1, 1])
    ax[1, 1].set_title('Q-Q Plot')

    plt.suptitle(f'SARIMA Diagnostics: {series.name}', fontsize=14, y=1.00)
    plt.tight_layout()
    plt.savefig(f'outputs/figures/{series.name}_sarima_diagnostics.png', dpi=150)
    plt.close()

    # Summary DataFrame
    summary_df = pd.DataFrame({
        'AIC': [fit.aic],
        'BIC': [fit.bic],
        'HQIC': [fit.hqic],
        'Log-Likelihood': [fit.llf],
        'JB Stat': [jb_stat],
        'JB p-value': [jb_p],
        'LB(10) p-value': [lb_stat.loc[10, 'lb_pvalue']],
        'LB(20) p-value': [lb_stat.loc[20, 'lb_pvalue']],
    }, index=[series.name])

    return {
        'model': model,
        'fit': fit,
        'residuals': residuals,
        'ljung_box': lb_stat,
        'jb_stat': jb_stat,
        'jb_p': jb_p,
        'summary_df': summary_df
    }


def forecast_sarima(fit, steps: int = 12) -> pd.DataFrame:
    """
    Phase 6: Generate forecasts from fitted SARIMA model.
    """
    logger.info(f"Generating {steps}-step forecast ...")
    forecast_result = fit.get_forecast(steps=steps)
    forecast_df = forecast_result.summary_frame()
    return forecast_df


def run_sarima_pipeline(df: pd.DataFrame) -> dict:
    """
    Run the complete SARIMA pipeline for each column (BTM, BTMP).

    Returns:
      results[col] = {
        'eda': {...},
        'stationarity': DataFrame,
        'identification': {...},
        'estimation': {...},
        'forecast_df': DataFrame
      }
    """
    results = {}
    for col in df.columns:
        logger.info(f"\n{'='*60}\nSARIMA Pipeline: {col}\n{'='*60}")
        series = df[col].copy()
        series.name = col

        # Phase 1: EDA
        eda_res = exploratory_analysis(series)

        # Phase 2: Stationarity
        stationarity_df = run_stationarity_suite(series)
        print(f"\n=== Stationarity Tests: {col} ===")
        print(stationarity_df.to_string(index=False))

        # Phase 3: Identification
        identification_res = identify_orders(series, max_p=3, max_q=3,
                                             seasonal=True)

        # Phase 3b: Estimation
        best_order = identification_res['best_order']
        seasonal_order = (1, 1, 1, 12)  # Fixed seasonal component
        estimation_res = estimate_sarima(series, order=best_order,
                                         seasonal_order=seasonal_order)

        # Phase 6: Forecasting
        forecast_df = forecast_sarima(estimation_res['fit'], steps=12)
        print(f"\n=== 12-Month Forecast: {col} ===")
        print(forecast_df.to_string())

        results[col] = {
            'eda': eda_res,
            'stationarity': stationarity_df,
            'identification': identification_res,
            'ic_table': identification_res['ic_table'],
            'estimation': estimation_res,
            'summary_df': estimation_res['summary_df'],
            'forecast_df': forecast_df,
            'fit': estimation_res['fit'],
            'residuals': estimation_res['residuals']
        }

    logger.info("SARIMA pipeline completed.")
    return results
