# quantitative-stationarity-toolbox
A Python feature engineering pipeline to transform non-stationary financial time-series data into model-ready features for machine learning algorithms.

# Quantitative Stationarity & Feature Engineering Toolbox

**Machine Learning | Time-Series Preprocessing | Python**

### 🧠 Bridging Financial Data & Machine Learning
Standard machine learning algorithms (like XGBoost, Random Forests, and LSTMs) assume that input data is independent and identically distributed (IID). Financial time-series data naturally violates this assumption due to non-stationarity, shifting means, and volatility clustering.

This repository serves as an **Advanced Preprocessing and Feature Engineering Pipeline**. It automates the extraction of stationary features from raw financial data so they can be safely ingested by predictive ML models without causing data leakage or spurious regressions.

### ⚙️ Core Pipeline Components
* **Automated Stationarity Testing:** Pipeline integration for Augmented Dickey-Fuller (ADF) and KPSS tests to verify the integration order of assets.
* **Memory-Preserving Transformations:** Implementation of fractional differencing to stabilize variance while retaining long-term signal momentum.
* **Volatility Scaling:** Automated rolling $z$-score normalizations and Log-Return conversions for heteroskedastic datasets.
* **Walk-Forward Validation:** Custom cross-validation iterators designed strictly for time-series, ensuring zero look-ahead bias during model training.

### 📐 Mathematical Foundation
To ensure data is model-ready, the pipeline tests for the presence of a unit root using the ADF test:
$$\Delta y_t=\alpha+\beta t+\gamma y_{t-1}+\sum_{j=1}^{p}\delta_j\Delta y_{t-j}+\varepsilon_t$$

To achieve stationarity without losing market memory (which often happens with standard integer differencing), we apply fractional differencing using the binomial series expansion:
$$(1-B)^d=\sum_{k=0}^{\infty}\binom{d}{k}(-B)^k$$

### 🚀 Quick Start (Example Usage)
```python
from quant_toolbox.adf_tests import check_stationarity
from quant_toolbox.differencing import fractional_difference
import pandas as pd

# 1. Load raw, non-stationary financial data
df = pd.read_csv('data/asset_prices.csv')

# 2. Check initial stationarity for ML ingestion
is_stationary, p_value = check_stationarity(df['Close'])

# 3. Apply fractional differencing to create a stationary, model-ready feature
if not is_stationary:
    df['Stationary_Feature'] = fractional_difference(df['Close'], d=0.45)
