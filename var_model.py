"""
var_model.py
============
Vector AutoRegression (VAR) for joint BTM & BTMP modeling.
  - Lag order selection (AIC, BIC, HQIC)
  - Granger causality tests
  - Impulse Response Functions (IRF)
  - Forecast Error Variance Decomposition (FEVD)
  - Multi-step forecasting
"""

import logging
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.api import VAR
from statsmodels.tsa.stattools import grangercausalitytests

warnings.filterwarnings('ignore')
logger = logging.getLogger(__name__)


def select_var_lag(df: pd.DataFrame, maxlags: int = 12) -> dict:
    """
    Select VAR lag order using information criteria.
    """
    logger.info("VAR lag order selection ...")
    model = VAR(df)
    lag_order_results = model.select_order(maxlags=maxlags)

    print("\n=== VAR Lag Order Selection ===")
    print(lag_order_results.summary())

    # Extract selected lags
    selected = {
        'AIC': lag_order_results.aic,
        'BIC': lag_order_results.bic,
        'HQIC': lag_order_results.hqic,
        'FPE': lag_order_results.fpe,
    }

    lag_order_df = pd.DataFrame(selected, index=['Optimal Lag'])
    return {
        'lag_order_results': lag_order_results,
        'lag_order_df': lag_order_df,
        'optimal_aic': lag_order_results.aic,
        'optimal_bic': lag_order_results.bic
    }


def fit_var(df: pd.DataFrame, lags: int) -> dict:
    """
    Estimate VAR model with specified lag order.
    """
    logger.info(f"Estimating VAR({lags}) model ...")
    model = VAR(df)
    fit = model.fit(lags)

    print(f"\n=== VAR({lags}) Estimation Summary ===")
    print(fit.summary())

    # Extract coefficients
    coeffs = pd.DataFrame(fit.params, columns=df.columns)
    print("\n=== VAR Coefficient Matrix ===")
    print(coeffs.to_string())

    return {
        'model': model,
        'fit': fit,
        'coefficients': coeffs,
        'aic': fit.aic,
        'bic': fit.bic,
        'hqic': fit.hqic
    }


def granger_causality_tests_all(df: pd.DataFrame, maxlag: int = 6) -> pd.DataFrame:
    """
    Run pairwise Granger causality tests.
    Tests whether X Granger-causes Y for all pairs (X, Y).
    """
    logger.info("Running Granger causality tests ...")
    results = []

    for col_y in df.columns:
        for col_x in df.columns:
            if col_y == col_x:
                continue
            logger.debug(f"Testing: {col_x} -> {col_y}")
            try:
                gc_res = grangercausalitytests(df[[col_y, col_x]], maxlag=maxlag, verbose=False)
                # Extract p-values for each lag
                for lag in range(1, maxlag + 1):
                    p_val_ssr_ftest = gc_res[lag][0]['ssr_ftest'][1]
                    results.append({
                        'cause': col_x,
                        'effect': col_y,
                        'lag': lag,
                        'F-stat': gc_res[lag][0]['ssr_ftest'][0],
                        'p-value': p_val_ssr_ftest,
                        'significant': p_val_ssr_ftest < 0.05
                    })
            except Exception as e:
                logger.warning(f"Granger test failed for {col_x}->{col_y}: {e}")

    granger_df = pd.DataFrame(results)
    print("\n=== Granger Causality Test Results ===")
    print(granger_df.to_string(index=False))

    return granger_df


def compute_irf(fit, periods: int = 24) -> dict:
    """
    Compute Impulse Response Functions.
    """
    logger.info(f"Computing IRF for {periods} periods ...")
    irf = fit.irf(periods)

    # Plot IRF
    fig = irf.plot(orth=False, impulse=None, response=None)
    plt.suptitle('Impulse Response Functions', fontsize=14, y=0.98)
    plt.tight_layout()
    plt.savefig('outputs/figures/VAR_IRF.png', dpi=150)
    plt.close()

    # IRF summary table
    irf_data = irf.irfs  # shape: (periods, n_vars, n_vars)
    irf_summary = pd.DataFrame({
        'periods': list(range(periods)),
        'BTM->BTM': irf_data[:, 0, 0] if irf_data.shape[1] > 0 else np.nan,
        'BTM->BTMP': irf_data[:, 1, 0] if irf_data.shape[1] > 1 else np.nan,
        'BTMP->BTM': irf_data[:, 0, 1] if irf_data.shape[1] > 1 else np.nan,
        'BTMP->BTMP': irf_data[:, 1, 1] if irf_data.shape[1] > 1 else np.nan,
    })

    return {
        'irf': irf,
        'irf_summary': irf_summary
    }


def compute_fevd(fit, periods: int = 24) -> dict:
    """
    Compute Forecast Error Variance Decomposition.
    """
    logger.info(f"Computing FEVD for {periods} periods ...")
    fevd = fit.fevd(periods)

    # Plot FEVD
    fig = fevd.plot()
    plt.suptitle('Forecast Error Variance Decomposition', fontsize=14, y=0.98)
    plt.tight_layout()
    plt.savefig('outputs/figures/VAR_FEVD.png', dpi=150)
    plt.close()

    # FEVD summary table at selected horizons
    horizons = [1, 6, 12, 24]
    fevd_rows = []
    for h in horizons:
        if h <= periods:
            decomp = fevd.decomp[h - 1]  # 0-indexed
            fevd_rows.append({
                'horizon': h,
                'BTM_from_BTM': decomp[0, 0] * 100,
                'BTM_from_BTMP': decomp[0, 1] * 100 if decomp.shape[1] > 1 else 0,
                'BTMP_from_BTM': decomp[1, 0] * 100 if decomp.shape[0] > 1 else 0,
                'BTMP_from_BTMP': decomp[1, 1] * 100 if decomp.shape[0] > 1 and decomp.shape[1] > 1 else 0,
            })
    fevd_df = pd.DataFrame(fevd_rows)
    print("\n=== FEVD Summary (%) ===")
    print(fevd_df.to_string(index=False))

    return {
        'fevd': fevd,
        'fevd_df': fevd_df
    }


def forecast_var(fit, steps: int = 12) -> pd.DataFrame:
    """
    Generate VAR forecasts.
    """
    logger.info(f"Generating {steps}-step VAR forecast ...")
    forecast_result = fit.forecast(fit.endog[-fit.k_ar:], steps=steps)

    # Create forecast DataFrame
    forecast_index = pd.date_range(
        start=fit.endog_names[0] if hasattr(fit, 'endog_names') else pd.Timestamp.now(),
        periods=steps,
        freq='MS'
    )
    forecast_df = pd.DataFrame(forecast_result, columns=fit.names, index=forecast_index)
    print("\n=== VAR Forecast ===")
    print(forecast_df.to_string())

    return forecast_df


def run_var_pipeline(df: pd.DataFrame) -> dict:
    """
    Complete VAR modeling pipeline.
    """
    logger.info("\n" + "="*60)
    logger.info("VAR Pipeline Started")
    logger.info("="*60)

    # 1. Lag order selection
    lag_res = select_var_lag(df, maxlags=12)
    optimal_lag = lag_res['optimal_aic']  # Use AIC

    # 2. Estimate VAR
    var_res = fit_var(df, lags=optimal_lag)

    # 3. Granger causality
    granger_df = granger_causality_tests_all(df, maxlag=6)

    # 4. IRF
    irf_res = compute_irf(var_res['fit'], periods=24)

    # 5. FEVD
    fevd_res = compute_fevd(var_res['fit'], periods=24)

    # 6. Forecasting
    forecast_df = forecast_var(var_res['fit'], steps=12)

    logger.info("VAR pipeline completed.")

    return {
        'lag_selection': lag_res,
        'lag_order_df': lag_res['lag_order_df'],
        'optimal_lag': optimal_lag,
        'estimation': var_res,
        'coefficients': var_res['coefficients'],
        'granger': granger_df,
        'granger_df': granger_df,
        'irf': irf_res['irf'],
        'irf_summary': irf_res['irf_summary'],
        'fevd': fevd_res['fevd'],
        'fevd_df': fevd_res['fevd_df'],
        'forecast_df': forecast_df,
        'fit': var_res['fit']
    }
