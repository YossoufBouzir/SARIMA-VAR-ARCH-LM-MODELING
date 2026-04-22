"""
visualization.py
================
Comprehensive plotting for all model outputs.
"""

import logging
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

logger = logging.getLogger(__name__)
sns.set_style('whitegrid')


def plot_time_series(df: pd.DataFrame, title: str = 'Time Series Data'):
    """Plot original time series."""
    fig, ax = plt.subplots(figsize=(12, 5))
    for col in df.columns:
        ax.plot(df.index, df[col], label=col, marker='o', markersize=3)
    ax.set_title(title, fontsize=14)
    ax.set_xlabel('Date')
    ax.set_ylabel('Sales (metric tons)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('outputs/figures/time_series.png', dpi=150)
    plt.close()
    logger.info("Saved: time_series.png")


def plot_seasonal_decomposition(sarima_results: dict):
    """Plot seasonal decomposition for each series."""
    for col, res in sarima_results.items():
        decomp = res['eda']['decomposition']
        fig = decomp.plot()
        fig.set_size_inches(12, 8)
        plt.suptitle(f'Seasonal Decomposition: {col}', fontsize=14, y=0.995)
        plt.tight_layout()
        plt.savefig(f'outputs/figures/{col}_decomposition.png', dpi=150)
        plt.close()
        logger.info(f"Saved: {col}_decomposition.png")


def plot_sarima_forecasts(df: pd.DataFrame, sarima_results: dict):
    """Plot SARIMA forecasts vs actual."""
    fig, axes = plt.subplots(len(df.columns), 1, figsize=(12, 4*len(df.columns)))
    if len(df.columns) == 1:
        axes = [axes]

    for i, col in enumerate(df.columns):
        ax = axes[i]
        series = df[col]
        forecast_df = sarima_results[col]['forecast_df']

        # Plot historical
        ax.plot(series.index, series, label='Actual', color='blue', marker='o', markersize=3)

        # Plot forecast
        ax.plot(forecast_df.index, forecast_df['mean'], label='Forecast',
                color='red', linestyle='--', marker='s', markersize=4)

        # Confidence intervals
        ax.fill_between(forecast_df.index,
                        forecast_df['mean_ci_lower'],
                        forecast_df['mean_ci_upper'],
                        color='red', alpha=0.2, label='95% CI')

        ax.set_title(f'SARIMA Forecast: {col}', fontsize=12)
        ax.set_xlabel('Date')
        ax.set_ylabel('Sales')
        ax.legend()
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('outputs/figures/sarima_forecasts.png', dpi=150)
    plt.close()
    logger.info("Saved: sarima_forecasts.png")


def plot_var_forecasts(df: pd.DataFrame, var_results: dict):
    """Plot VAR forecasts."""
    forecast_df = var_results['forecast_df']

    fig, axes = plt.subplots(len(df.columns), 1, figsize=(12, 4*len(df.columns)))
    if len(df.columns) == 1:
        axes = [axes]

    for i, col in enumerate(df.columns):
        ax = axes[i]
        # Historical
        ax.plot(df.index, df[col], label='Actual', color='blue', marker='o', markersize=3)

        # VAR forecast
        if col in forecast_df.columns:
            ax.plot(forecast_df.index, forecast_df[col], label='VAR Forecast',
                    color='green', linestyle='--', marker='^', markersize=4)

        ax.set_title(f'VAR Forecast: {col}', fontsize=12)
        ax.set_xlabel('Date')
        ax.set_ylabel('Sales')
        ax.legend()
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('outputs/figures/var_forecasts.png', dpi=150)
    plt.close()
    logger.info("Saved: var_forecasts.png")


def plot_residuals_comparison(sarima_results: dict):
    """Compare residuals from all SARIMA models."""
    fig, axes = plt.subplots(len(sarima_results), 1, figsize=(12, 3*len(sarima_results)))
    if len(sarima_results) == 1:
        axes = [axes]

    for i, (col, res) in enumerate(sarima_results.items()):
        residuals = res['residuals']
        axes[i].plot(residuals.index, residuals, label=f'{col} Residuals', color='black', alpha=0.7)
        axes[i].axhline(0, color='red', linestyle='--')
        axes[i].set_title(f'SARIMA Residuals: {col}')
        axes[i].set_ylabel('Residual')
        axes[i].legend()
        axes[i].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('outputs/figures/residuals_comparison.png', dpi=150)
    plt.close()
    logger.info("Saved: residuals_comparison.png")


def plot_model_comparison(sarima_results: dict, var_results: dict):
    """Bar chart comparing AIC/BIC across models."""
    data = []
    for col, res in sarima_results.items():
        data.append({'Model': f'SARIMA_{col}', 'AIC': res['summary_df']['AIC'].values[0],
                     'BIC': res['summary_df']['BIC'].values[0]})

    data.append({'Model': 'VAR', 'AIC': var_results['estimation']['aic'],
                 'BIC': var_results['estimation']['bic']})

    df_comp = pd.DataFrame(data)

    fig, ax = plt.subplots(1, 2, figsize=(12, 5))
    df_comp.plot(x='Model', y='AIC', kind='bar', ax=ax[0], color='steelblue', legend=False)
    ax[0].set_title('AIC Comparison')
    ax[0].set_ylabel('AIC')
    ax[0].tick_params(axis='x', rotation=45)

    df_comp.plot(x='Model', y='BIC', kind='bar', ax=ax[1], color='coral', legend=False)
    ax[1].set_title('BIC Comparison')
    ax[1].set_ylabel('BIC')
    ax[1].tick_params(axis='x', rotation=45)

    plt.tight_layout()
    plt.savefig('outputs/figures/model_comparison.png', dpi=150)
    plt.close()
    logger.info("Saved: model_comparison.png")


def plot_all(df: pd.DataFrame, sarima_results: dict, var_results: dict, arch_results: dict):
    """
    Generate all visualization outputs.
    """
    logger.info("Generating all visualizations ...")

    # 1. Time series
    plot_time_series(df, title='NAFTAL Bitumen Sales (Monthly)')

    # 2. Seasonal decomposition
    plot_seasonal_decomposition(sarima_results)

    # 3. SARIMA forecasts
    plot_sarima_forecasts(df, sarima_results)

    # 4. VAR forecasts
    plot_var_forecasts(df, var_results)

    # 5. Residuals comparison
    plot_residuals_comparison(sarima_results)

    # 6. Model comparison
    plot_model_comparison(sarima_results, var_results)

    logger.info("All visualizations complete.")
