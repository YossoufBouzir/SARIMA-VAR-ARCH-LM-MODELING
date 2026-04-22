"""
arch_lm_test.py
===============
ARCH-LM Volatility Tests on SARIMA Residuals.
  - Test for ARCH effects (heteroskedasticity)
  - Optional GARCH modeling if ARCH effects detected
"""

import logging
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.stats.diagnostic import het_arch

warnings.filterwarnings('ignore')
logger = logging.getLogger(__name__)


def arch_lm_test(residuals: pd.Series, lags: int = 12) -> dict:
    """
    ARCH-LM Test (Engle's Test for Autoregressive Conditional Heteroskedasticity).

    H0: No ARCH effects (homoskedasticity)
    H1: ARCH effects present (heteroskedasticity)
    """
    logger.info(f"Running ARCH-LM test (lags={lags}) on {residuals.name} residuals ...")

    lm_stat, lm_p, f_stat, f_p = het_arch(residuals.dropna(), nlags=lags)

    result = {
        'series': residuals.name,
        'lags': lags,
        'LM_statistic': lm_stat,
        'LM_p_value': lm_p,
        'F_statistic': f_stat,
        'F_p_value': f_p,
        'reject_H0': lm_p < 0.05,
        'conclusion': 'ARCH effects present' if lm_p < 0.05 else 'No ARCH effects'
    }

    print(f"\n=== ARCH-LM Test: {residuals.name} ===")
    print(f"Lags           : {lags}")
    print(f"LM Statistic   : {lm_stat:.4f}")
    print(f"LM p-value     : {lm_p:.4f}")
    print(f"F Statistic    : {f_stat:.4f}")
    print(f"F p-value      : {f_p:.4f}")
    print(f"Conclusion     : {result['conclusion']}")

    return result


def fit_garch(residuals: pd.Series, p: int = 1, q: int = 1) -> dict:
    """
    Fit GARCH(p,q) model to residuals if ARCH effects are present.
    Uses arch package.
    """
    try:
        from arch import arch_model
    except ImportError:
        logger.warning("arch package not installed. Skipping GARCH modeling.")
        return {'error': 'arch package not available'}

    logger.info(f"Fitting GARCH({p},{q}) to {residuals.name} residuals ...")

    # Scale residuals for numerical stability
    res_scaled = residuals * 100

    model = arch_model(res_scaled, vol='Garch', p=p, q=q, rescale=False)
    fit = model.fit(disp='off', show_warning=False)

    print(f"\n=== GARCH({p},{q}) Summary: {residuals.name} ===")
    print(fit.summary())

    # Extract key stats
    summary_df = pd.DataFrame({
        'AIC': [fit.aic],
        'BIC': [fit.bic],
        'Log-Likelihood': [fit.loglikelihood],
        'Alpha[1]': [fit.params.get('alpha[1]', np.nan)],
        'Beta[1]': [fit.params.get('beta[1]', np.nan)],
    }, index=[residuals.name])

    return {
        'model': model,
        'fit': fit,
        'summary_df': summary_df,
        'conditional_volatility': fit.conditional_volatility / 100  # rescale
    }


def run_arch_lm_pipeline(sarima_results: dict) -> dict:
    """
    Run ARCH-LM tests on all SARIMA residuals.

    If ARCH effects are detected, optionally fit GARCH models.
    """
    logger.info("\n" + "="*60)
    logger.info("ARCH-LM Testing Pipeline")
    logger.info("="*60)

    arch_lm_results = []
    garch_summaries = {}

    for col, res in sarima_results.items():
        residuals = res['residuals']
        residuals.name = col

        # ARCH-LM test
        arch_res = arch_lm_test(residuals, lags=12)
        arch_lm_results.append(arch_res)

        # If ARCH effects, fit GARCH
        if arch_res['reject_H0']:
            logger.info(f"ARCH effects detected in {col}. Fitting GARCH ...")
            garch_res = fit_garch(residuals, p=1, q=1)
            if 'error' not in garch_res:
                garch_summaries[col] = garch_res['summary_df']

                # Plot conditional volatility
                vol = garch_res['conditional_volatility']
                fig, ax = plt.subplots(figsize=(10, 4))
                ax.plot(residuals.index, vol, label='Conditional Volatility', color='red')
                ax.set_title(f'GARCH Conditional Volatility: {col}')
                ax.set_ylabel('Volatility')
                ax.legend()
                plt.tight_layout()
                plt.savefig(f'outputs/figures/{col}_garch_volatility.png', dpi=150)
                plt.close()
        else:
            logger.info(f"No ARCH effects in {col}. Skipping GARCH.")

    arch_lm_df = pd.DataFrame(arch_lm_results)
    print("\n=== ARCH-LM Summary Table ===")
    print(arch_lm_df.to_string(index=False))

    logger.info("ARCH-LM pipeline completed.")

    return {
        'arch_lm_df': arch_lm_df,
        'garch_summary': garch_summaries
    }
