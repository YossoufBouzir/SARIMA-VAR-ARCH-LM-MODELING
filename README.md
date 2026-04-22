# SARIMA-VAR-ARCH-LM Modeling for NAFTAL Bitumen Sales

**Modeling and Forecasting Bitumen Sales in Algeria Using SARIMA, VAR, and ARCH-LM Volatility Tests**

---

## 📊 Project Overview

This repository implements a comprehensive time series analysis pipeline for modeling monthly bitumen sales from NAFTAL (Algeria's national fuel and petroleum products distributor) covering 2002-2010. The methodology combines:

- **SARIMA (Seasonal ARIMA)**: Univariate time series forecasting for BTM (Standard Bitumen) and BTMP (Pure Bitumen)
- **VAR (Vector AutoRegression)**: Multivariate modeling capturing interdependencies between BTM and BTMP
- **ARCH-LM Tests**: Volatility testing on residuals with optional GARCH modeling

**Research Context**: This code supports an academic article on bitumen demand forecasting using advanced econometric techniques.

---

## 🗂️ Repository Structure

```
SARIMA-VAR-ARCH-LM-MODELING/
│
├── main.py                     # Main orchestration pipeline
├── utils.py                    # Data loading, stationarity tests, logging, export
├── sarima_model.py             # SARIMA: EDA, identification, estimation, diagnostics
├── var_model.py                # VAR: lag selection, Granger, IRF, FEVD, forecasting
├── arch_lm_test.py             # ARCH-LM tests and GARCH modeling
├── visualization.py            # Comprehensive plotting functions
├── requirements.txt            # Python dependencies
│
├── data/
│   ├── naftal_bitumen_data_template.csv    # Data template (or real data)
│   ├── generate_synthetic_data.py          # Synthetic data generator
│   └── __init__.py
│
└── outputs/
    ├── figures/                # Generated plots
    └── results/                # Excel summaries
```

---

## 🔬 Methodology

### Phase 1: Exploratory Data Analysis (EDA)
- Descriptive statistics (mean, std, skewness, kurtosis, Jarque-Bera test)
- Time series plots
- Seasonal decomposition (trend, seasonal, residual)

### Phase 2: Stationarity Testing
- **ADF Test** (Augmented Dickey-Fuller): H0 = unit root (non-stationary)
- **KPSS Test**: H0 = stationary
- **Phillips-Perron / Zivot-Andrews**: Structural break-aware stationarity
- Applied to both levels and first differences

### Phase 3: SARIMA Modeling
1. **Identification**: ACF/PACF plots + grid search over (p,d,q)×(P,D,Q,s) orders
2. **Estimation**: MLE via `statsmodels.tsa.statespace.SARIMAX`
3. **Diagnostics**:
   - Ljung-Box test (residual autocorrelation)
   - Jarque-Bera normality test
   - Residual ACF, histograms, Q-Q plots
4. **Forecasting**: 12-month ahead forecasts with 95% confidence intervals

### Phase 4: VAR Modeling
1. **Lag Order Selection**: AIC, BIC, HQIC, FPE criteria
2. **Estimation**: VAR(p) model for joint [BTM, BTMP] dynamics
3. **Granger Causality Tests**: Does BTM Granger-cause BTMP? Vice versa?
4. **Impulse Response Functions (IRF)**: Dynamic response to shocks
5. **Forecast Error Variance Decomposition (FEVD)**: Variance attribution
6. **Forecasting**: 12-month joint forecasts

### Phase 5: ARCH-LM Volatility Tests
1. **ARCH-LM Test** (Engle's test): H0 = no ARCH effects (homoskedasticity)
2. **GARCH Modeling**: If ARCH effects detected, fit GARCH(1,1) to capture volatility clustering
3. **Conditional Volatility Plots**

### Phase 6: Visualization & Export
- Time series plots, seasonal decompositions
- SARIMA & VAR forecast comparison
- Residual diagnostics
- Model comparison (AIC/BIC bar charts)
- Excel export of all results (multi-sheet workbook)

---

## 🚀 Installation

### Prerequisites
- Python ≥ 3.8
- pip

### Setup

```bash
# Clone repository
git clone https://github.com/YossoufBouzir/SARIMA-VAR-ARCH-LM-MODELING.git
cd SARIMA-VAR-ARCH-LM-MODELING

# Create virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

---

## 💻 Usage

### Running the Full Pipeline

```bash
python main.py
```

**Outputs**:
- `outputs/figures/`: All diagnostic and forecast plots (PNG)
- `outputs/results/all_results.xlsx`: Consolidated Excel workbook with:
  - SARIMA summaries (AIC, BIC, diagnostics)
  - Stationarity test results
  - IC tables
  - VAR coefficients, Granger causality
  - ARCH-LM test results
  - GARCH summaries (if applicable)
  - Forecasts

### Data Format

The pipeline expects a CSV file at `data/naftal_bitumen_data_template.csv` with:

```csv
date,BTM,BTMP
2002-01-01,18500,19200
2002-02-01,19300,20100
...
```

- **date**: Monthly dates (YYYY-MM-DD, first of month)
- **BTM**: Standard Bitumen sales (metric tons)
- **BTMP**: Pure Bitumen sales (metric tons)

If the file is missing, the pipeline auto-generates synthetic data matching the reported statistical properties.

---

## 📈 Key Results (Expected)

### SARIMA Models
- **BTM**: SARIMA(p,1,q)(1,1,1)[12] — captures trend + seasonal pattern
- **BTMP**: SARIMA(p,1,q)(1,1,1)[12]
- Diagnostics: Ljung-Box p > 0.05 (no residual autocorrelation), Jarque-Bera assesses normality

### VAR Model
- Optimal lag: Determined by AIC (typically 2-6)
- Granger causality: Bidirectional causality between BTM ↔ BTMP expected
- IRF: Impulse in BTM → temporary increase in BTMP (and vice versa)
- FEVD: Variance mostly self-explained, with cross-effects

### ARCH-LM Tests
- Tests whether SARIMA residuals exhibit volatility clustering
- If detected (p < 0.05), GARCH(1,1) fitted to model conditional heteroskedasticity

---

## 🛠️ Module Documentation

### `main.py`
Orchestrates the entire pipeline: data loading → SARIMA → VAR → ARCH-LM → visualization → export.

### `utils.py`
- `load_data()`: CSV loader with date parsing
- `adf_test()`, `kpss_test()`, `pp_test()`: Stationarity tests
- `run_stationarity_suite()`: Runs all tests on levels & diffs
- `ic_table()`: Information criteria comparison
- `save_results_to_excel()`: Multi-sheet Excel export

### `sarima_model.py`
- `exploratory_analysis()`: EDA + seasonal decomposition
- `identify_orders()`: ACF/PACF + grid search
- `estimate_sarima()`: SARIMAX fitting + diagnostics
- `forecast_sarima()`: Multi-step forecasting
- `run_sarima_pipeline()`: Full SARIMA workflow

### `var_model.py`
- `select_var_lag()`: IC-based lag selection
- `fit_var()`: VAR estimation
- `granger_causality_tests_all()`: Pairwise Granger tests
- `compute_irf()`: Impulse Response Functions
- `compute_fevd()`: Forecast Error Variance Decomposition
- `forecast_var()`: VAR forecasting
- `run_var_pipeline()`: Full VAR workflow

### `arch_lm_test.py`
- `arch_lm_test()`: Engle's ARCH-LM test
- `fit_garch()`: GARCH(1,1) model (optional)
- `run_arch_lm_pipeline()`: Tests all SARIMA residuals

### `visualization.py`
- `plot_time_series()`: Original data plots
- `plot_seasonal_decomposition()`: Decomposition charts
- `plot_sarima_forecasts()`: SARIMA forecasts with CI
- `plot_var_forecasts()`: VAR forecast comparison
- `plot_residuals_comparison()`: Residual plots
- `plot_model_comparison()`: AIC/BIC bar charts
- `plot_all()`: Generates all plots

---

## 📦 Dependencies

```
numpy>=1.24
pandas>=2.0
matplotlib>=3.7
seaborn>=0.12
scipy>=1.10
statsmodels>=0.14
arch>=6.0         # For GARCH modeling (optional Phase 5)
openpyxl>=3.1     # For Excel export
```

See `requirements.txt` for full list.

---

## 📝 Citation

If you use this code for your research, please cite:

```bibtex
@article{yourname2026bitumen,
  title={Modeling and Forecasting Bitumen Sales in Algeria Using SARIMA, VAR, and ARCH-LM Volatility Tests},
  author={Your Name and Co-Authors},
  journal={Journal Name},
  year={2026},
  volume={XX},
  pages={YYY--ZZZ}
}
```

---

## 🤝 Contributing

Contributions welcome! Please:
1. Fork the repository
2. Create a feature branch (`git checkout -b feature/your-feature`)
3. Commit changes (`git commit -m 'Add feature'`)
4. Push (`git push origin feature/your-feature`)
5. Open a Pull Request

---

## 📄 License

MIT License — see [LICENSE](LICENSE) for details.

---

## 👤 Author

**Yossouf Bouzir**  
Email: youcefbouzir@gmail.com  
GitHub: [@YossoufBouzir](https://github.com/YossoufBouzir)

---

## 🙏 Acknowledgments

- **NAFTAL**: Data source (bitumen sales records)
- **statsmodels**: Time series modeling framework
- **arch**: GARCH/volatility modeling
- Academic supervisors and collaborators

---

**Last Updated**: April 2026
