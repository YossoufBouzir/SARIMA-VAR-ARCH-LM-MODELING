"""
Synthetic data generator that replicates the statistical properties
(mean, std, seasonality, SARIMA residuals) reported in the paper.
Replace with actual NAFTAL data when available.
"""

import numpy as np
import pandas as pd

def generate_data(seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2002-01", periods=108, freq="MS")

    # Seasonal pattern (construction peaks Apr-Aug)
    seasonal_btm  = np.array([0.70, 0.75, 0.90, 1.10, 1.25, 1.30,
                               1.25, 1.15, 0.95, 0.85, 0.75, 0.70])
    seasonal_btmp = np.array([0.72, 0.78, 0.92, 1.08, 1.22, 1.28,
                               1.20, 1.12, 0.96, 0.88, 0.78, 0.72])

    trend = np.linspace(18000, 28000, 108)

    btm_s  = np.tile(seasonal_btm,  9)
    btmp_s = np.tile(seasonal_btmp, 9)

    noise_btm  = rng.normal(0, 1800, 108)
    noise_btmp = rng.normal(0, 1600, 108)

    BTM  = trend * btm_s  + noise_btm
    BTMP = trend * btmp_s + noise_btmp

    # Ensure positivity
    BTM  = np.clip(BTM,  5000, None)
    BTMP = np.clip(BTMP, 5000, None)

    df = pd.DataFrame({"BTM": BTM, "BTMP": BTMP}, index=dates)
    df.index.name = "date"
    return df
