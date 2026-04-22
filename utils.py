"""
utils.py
========
Shared utility functions: data loading, logging, stationarity tests,
information criteria, and result export.
"""

import logging
import warnings
import numpy as np
import pandas as pd
from statsmodels.tsa.stattools import adfuller, kpss
from statsmodels.tsa.stattools import zivot_andrews

warnings.filterwarnings('ignore')


# ------------------------------------------------------------------ #
# Logging
# ------------------------------------------------------------------ #
def setup_logging(level=logging.INFO):
    """Configure root logger with console and file handlers."""
    logging.basicConfig(
        level=level,
        format='%(asctime)s | %(levelname)-8s | %(name)s | %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S',
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler('outputs/pipeline.log', mode='w')
        ]
    )


# ------------------------------------------------------------------ #
# Data Loading
# ------------------------------------------------------------------ #
def load_data(filepath: str) -> pd.DataFrame:
    """
    Load the NAFTAL bitumen CSV and return a time-indexed DataFrame.

    Expected CSV columns: date, BTM, BTMP
      - date : YYYY-MM-DD (first of month)
      - BTM  : Standard bitumen sales (metric tons)
      - BTMP : Pure bitumen sales (metric tons)
    """
    df = pd.read_csv(filepath, comment='#')
    df.columns = [c.strip() for c in df.columns]

    # Accept flexible date column names
    date_col = [c for c in df.columns if c.lower() in ('date', 'time', 'month')]
    if not date_col:
        raise ValueError("No date column found. Expected 'date', 'time', or 'month'.")
    df['date'] = pd.to_datetime(df[date_col[0]])
    df = df.set_index('date').sort_index()
    df.index.freq = pd.infer_freq(df.index)
    if df.index.freq is None:
        df = df.asfreq('MS')  # Monthly start

    # Keep only numeric columns
    df = df.select_dtypes(include=[np.number])
    df = df.dropna()
    return df


# ------------------------------------------------------------------ #
# Descriptive Statistics
# ------------------------------------------------------------------ #
def descriptive_stats(series: pd.Series) -> pd.DataFrame:
    """Extended descriptive stats including skewness and kurtosis."""
    from scipy.stats import skew, kurtosis, jarque_bera
    jb_stat, jb_p = jarque_bera(series.dropna())
    stats = {
        'N'         : len(series.dropna()),
        'Mean'      : series.mean(),
        'Std'       : series.std(),
        'Min'       : series.min(),
        'Max'       : series.max(),
        'Skewness'  : skew(series.dropna()),
        'Kurtosis'  : kurtosis(series.dropna()),
        'JB Stat'   : jb_stat,
        'JB p-value': jb_p,
    }
    return pd.DataFrame(stats, index=[series.name])


# ------------------------------------------------------------------ #
# Stationarity Tests
# ------------------------------------------------------------------ #
def adf_test(series: pd.Series, maxlag: int = None,
             regression: str = 'ct') -> dict:
    """
    Augmented Dickey-Fuller Test.
    H0: Unit root (non-stationary)
    Returns dict with test statistic, p-value, lags, critical values.
    """
    result = adfuller(series.dropna(), maxlag=maxlag, regression=regression,
                      autolag='AIC')
    return {
        'test'        : 'ADF',
        'series'      : series.name,
        'regression'  : regression,
        'statistic'   : result[0],
        'p_value'     : result[1],
        'n_lags'      : result[2],
        'n_obs'       : result[3],
        'critical_1%' : result[4]['1%'],
        'critical_5%' : result[4]['5%'],
        'critical_10%': result[4]['10%'],
        'reject_H0'   : result[1] < 0.05,
        'conclusion'  : 'Stationary' if result[1] < 0.05 else 'Non-Stationary'
    }


def kpss_test(series: pd.Series, regression: str = 'ct',
              nlags: str = 'auto') -> dict:
    """
    KPSS Test.
    H0: Stationary
    """
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        stat, p_value, lags, crit = kpss(series.dropna(),
                                         regression=regression, nlags=nlags)
    return {
        'test'        : 'KPSS',
        'series'      : series.name,
        'regression'  : regression,
        'statistic'   : stat,
        'p_value'     : p_value,
        'n_lags'      : lags,
        'critical_1%' : crit['1%'],
        'critical_5%' : crit['5%'],
        'critical_10%': crit['10%'],
        'reject_H0'   : p_value < 0.05,
        'conclusion'  : 'Non-Stationary' if p_value < 0.05 else 'Stationary'
    }


def pp_test(series: pd.Series) -> dict:
    """
    Phillips-Perron Test via Zivot-Andrews (structural break-aware ADF).
    H0: Unit root
    """
    result = zivot_andrews(series.dropna(), autolag='AIC')
    return {
        'test'      : 'PP/ZA',
        'series'    : series.name,
        'statistic' : result[0],
        'p_value'   : result[1],
        'n_lags'    : result[2],
        'baselag'   : result[3],
        'reject_H0' : result[1] < 0.05,
        'conclusion': 'Stationary' if result[1] < 0.05 else 'Non-Stationary'
    }


def run_stationarity_suite(series: pd.Series) -> pd.DataFrame:
    """
    Run ADF, KPSS, and PP tests on both levels and first differences.
    Returns a summary DataFrame.
    """
    rows = []
    for level, s in [('Level', series), ('1st Diff', series.diff().dropna())]:
        s = s.copy()
        s.name = f"{series.name} ({level})"
        for test_fn in [adf_test, kpss_test, pp_test]:
            try:
                res = test_fn(s)
                res['transform'] = level
                rows.append(res)
            except Exception as e:
                rows.append({'test': str(test_fn.__name__), 'series': s.name,
                             'transform': level, 'error': str(e)})
    return pd.DataFrame(rows)


# ------------------------------------------------------------------ #
# Information Criteria Helper
# ------------------------------------------------------------------ #
def ic_table(aic_grid: dict, bic_grid: dict, hqic_grid: dict = None) -> pd.DataFrame:
    """
    Build an information-criteria comparison table from order->IC dicts.
    """
    df = pd.DataFrame({'AIC': aic_grid, 'BIC': bic_grid})
    if hqic_grid:
        df['HQIC'] = hqic_grid
    df.index.name = 'order'
    return df.sort_values('AIC')


# ------------------------------------------------------------------ #
# Excel Export
# ------------------------------------------------------------------ #
def save_results_to_excel(results: dict, path: str = 'outputs/results/all_results.xlsx'):
    """
    Save all model results to a multi-sheet Excel workbook.
    """
    import openpyxl
    from openpyxl.styles import Font, PatternFill, Alignment

    writer = pd.ExcelWriter(path, engine='openpyxl')

    # --- SARIMA Results ---
    if 'sarima' in results:
        sar = results['sarima']
        for col, res in sar.items():
            if 'summary_df' in res:
                sheet_name = f'SARIMA_{col}'[:31]
                res['summary_df'].to_excel(writer, sheet_name=sheet_name)
            if 'stationarity' in res:
                res['stationarity'].to_excel(
                    writer, sheet_name=f'Stat_{col}'[:31])
            if 'ic_table' in res:
                res['ic_table'].to_excel(
                    writer, sheet_name=f'IC_{col}'[:31])
            if 'forecast_df' in res:
                res['forecast_df'].to_excel(
                    writer, sheet_name=f'Forecast_{col}'[:31])

    # --- VAR Results ---
    if 'var' in results:
        var = results['var']
        if 'lag_order_df' in var:
            var['lag_order_df'].to_excel(writer, sheet_name='VAR_LagOrder')
        if 'coefficients' in var:
            var['coefficients'].to_excel(writer, sheet_name='VAR_Coefficients')
        if 'granger_df' in var:
            var['granger_df'].to_excel(writer, sheet_name='VAR_Granger')
        if 'irf_summary' in var:
            var['irf_summary'].to_excel(writer, sheet_name='VAR_IRF')
        if 'fevd_df' in var:
            var['fevd_df'].to_excel(writer, sheet_name='VAR_FEVD')
        if 'forecast_df' in var:
            var['forecast_df'].to_excel(writer, sheet_name='VAR_Forecast')

    # --- ARCH-LM Results ---
    if 'arch_lm' in results:
        arch = results['arch_lm']
        if 'arch_lm_df' in arch:
            arch['arch_lm_df'].to_excel(writer, sheet_name='ARCH_LM_Tests')
        if 'garch_summary' in arch:
            for col, summ in arch['garch_summary'].items():
                summ.to_excel(writer, sheet_name=f'GARCH_{col}'[:31])

    writer.close()
    print(f"Results saved to: {path}")
