"""
Main Pipeline: SARIMA-VAR-ARCH-LM Modeling for NAFTAL Bitumen Sales
=====================================================================
Article: Modeling and Forecasting Bitumen Sales in Algeria Using
         SARIMA, VAR, and ARCH-LM Volatility Tests
Data: Monthly bitumen sales (BTM & BTMP) from NAFTAL, 2002-2010

Methodology:
  Phase 1 - Exploratory Data Analysis (EDA)
  Phase 2 - Stationarity Tests (ADF, KPSS, PP)
  Phase 3 - SARIMA Identification, Estimation & Diagnostics
  Phase 4 - VAR Modeling (BTM & BTMP jointly)
  Phase 5 - ARCH-LM Volatility Testing on Residuals
  Phase 6 - Forecasting & Visualization
"""

import os
import warnings
import logging
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from utils import load_data, setup_logging, save_results_to_excel
from sarima_model import run_sarima_pipeline
from var_model import run_var_pipeline
from arch_lm_test import run_arch_lm_pipeline
from visualization import plot_all

warnings.filterwarnings('ignore')


def main():
    setup_logging()
    logger = logging.getLogger(__name__)
    logger.info("========== SARIMA-VAR-ARCH-LM Pipeline Started ==========")

    # ------------------------------------------------------------------ #
    # 0. Setup output directories
    # ------------------------------------------------------------------ #
    os.makedirs('outputs/figures', exist_ok=True)
    os.makedirs('outputs/results', exist_ok=True)

    # ------------------------------------------------------------------ #
    # 1. Load Data
    # ------------------------------------------------------------------ #
    logger.info("Phase 0: Loading data ...")
    try:
        df = load_data('data/naftal_bitumen_data_template.csv')
        logger.info(f"Data loaded: {df.shape[0]} observations, columns: {list(df.columns)}")
    except FileNotFoundError:
        logger.warning("CSV not found — generating synthetic data ...")
        from data.generate_synthetic_data import generate_data
        df = generate_data(seed=42)
        df.to_csv('data/naftal_bitumen_data_template.csv')
        logger.info("Synthetic data generated and saved.")

    print("\n" + "="*60)
    print("DATA SUMMARY")
    print("="*60)
    print(df.describe().round(2).to_string())
    print(f"\nDate range : {df.index[0]} → {df.index[-1]}")
    print(f"Missing values : {df.isnull().sum().to_dict()}")

    results = {}

    # ------------------------------------------------------------------ #
    # 2. SARIMA Pipeline (BTM and BTMP individually)
    # ------------------------------------------------------------------ #
    logger.info("Phase 1-3: Running SARIMA pipeline ...")
    sarima_results = run_sarima_pipeline(df)
    results['sarima'] = sarima_results

    # ------------------------------------------------------------------ #
    # 3. VAR Pipeline (BTM & BTMP jointly)
    # ------------------------------------------------------------------ #
    logger.info("Phase 4: Running VAR pipeline ...")
    var_results = run_var_pipeline(df)
    results['var'] = var_results

    # ------------------------------------------------------------------ #
    # 4. ARCH-LM Tests on SARIMA Residuals
    # ------------------------------------------------------------------ #
    logger.info("Phase 5: Running ARCH-LM tests ...")
    arch_results = run_arch_lm_pipeline(sarima_results)
    results['arch_lm'] = arch_results

    # ------------------------------------------------------------------ #
    # 5. Full Visualization
    # ------------------------------------------------------------------ #
    logger.info("Phase 6: Generating all plots ...")
    plot_all(df, sarima_results, var_results, arch_results)

    # ------------------------------------------------------------------ #
    # 6. Save consolidated results
    # ------------------------------------------------------------------ #
    logger.info("Saving results to Excel ...")
    save_results_to_excel(results, path='outputs/results/all_results.xlsx')

    print("\n" + "="*60)
    print("PIPELINE COMPLETE")
    print("Figures saved  → outputs/figures/")
    print("Results saved  → outputs/results/all_results.xlsx")
    print("="*60)
    logger.info("========== Pipeline Finished Successfully ==========")


if __name__ == '__main__':
    main()
