# Volatility Trading Analysis

A Python-based project for analyzing S&P 500 (SPX) options volatility, VIX estimation, and volatility trading strategies. This project implements various volatility estimators and regression models to understand the relationship between implied and realized volatility.

<p align="center">
  <img src="figures/realreal (1).png" alt="Comparison" width="60%">
  <br>
  <em>Figure 1: Comparison of Volatilities.</em>
</p>


## Authors

This project was developed by:
- **Maarten Stork** 
- **Lucas Keijzer** 
- **Pemmasani Prabakaran Rohith Saai**

## Project Overview

This project contains three main components:

1. **SPX-VIX Regression Analysis** - Examines the relationship between SPX returns and VIX levels
2. **VIX Estimation** - Calculates VIX from option chain data using CBOE methodology
3. **Volatility Estimators** - Implements classical and Parkinson volatility estimators

## Repository Structure

```
Volatility_Trading/
├── README.md                           # This file
├── data/                              # Option data files
│   ├── Call_option_data_2025-04-03_final.csv
│   └── Put_option_data_2025-04-03_final.csv
├── figures/                           # Generated plots and visualizations
└── FinDatSc_Models/                   # Core Python modules
    ├── spx_vix_regression.py          # SPX-VIX regression analysis
    ├── vix_estimator.py               # VIX calculation from options
    └── volatility_estimators.py       # Historical volatility estimators
```

## Features

### 1. SPX-VIX Regression Analysis (`spx_vix_regression.py`)

- **Data Download**: Automatically downloads SPX and VIX historical data from Yahoo Finance
- **Returns Calculation**: Computes daily SPX returns and realized variance
- **Regression Models**:
  - SPX Returns vs VIX levels
  - SPX Returns vs Realized Variance
  - SPX Returns vs ∆VIX (change in VIX)
- **Diagnostic Tests**: Breusch-Pagan heteroscedasticity test and Augmented Dickey-Fuller stationarity test
- **Visualization**: Scatter plots with fitted regression lines

### 2. VIX Estimation (`vix_estimator.py`)

- **Option Chain Retrieval**: Downloads real-time option data using yfinance
- **Forward Price Calculation**: Computes theoretical forward prices
- **VIX Calculation**: Implements CBOE VIX methodology (Equation 19)
- **Improved Estimator**: Enhanced version with mid-price calculation and strike filtering
- **Comparison Tools**: Correlates estimated VIX with actual VIX and tests for cointegration

### 3. Volatility Estimators (`volatility_estimators.py`)

- **Classical Volatility**: Standard historical volatility using returns (Equation 14)
- **Parkinson Estimator**: Range-based volatility using high-low prices (Equation 2)
- **Rolling Window Analysis**: Computes rolling estimates over time
- **MSFT Example**: Demonstrates estimators on Microsoft stock data

## Data Files

The project includes SPX option data from April 3, 2025:

### Call Options (`Call_option_data_2025-04-03_final.csv`)

- 126 call contracts with various strike prices
- Includes bid/ask prices, implied volatility, volume, and open interest
- Strikes ranging from $2,800 to above current market levels

### Put Options (`Put_option_data_2025-04-03_final.csv`)

- 177 put contracts with comprehensive strike coverage
- Similar data structure to call options
- Strikes ranging from $2,400 to current market levels

## Key Equations Implemented

### Classical Volatility (Equation 14)

```
σ̂² = (1/(N-1)) * Σ[1/(t_{k+1}-t_k)] * (r_k - dt*μ̂)²
```

### Parkinson Estimator (Equation 2)

```
σ_Parkinson = sqrt((1/(4ln2*T)) * Σ[ln(h_t/l_t)]²)
```

### VIX Calculation (Equation 19)

```
VIX² = (2*e^(rτ)/τ) * Σ(ΔK/K²) * Q(K)
```

## Requirements

```python
yfinance>=0.2.18
numpy>=1.21.0
pandas>=1.3.0
matplotlib>=3.5.0
statsmodels>=0.13.0
```

## Installation

1. Clone the repository:

```bash
git clone <repository-url>
cd Volatility_Trading
```

2. Install required packages:

```bash
pip install yfinance numpy pandas matplotlib statsmodels
```

## Usage

### Running SPX-VIX Regression Analysis

```python
from FinDatSc_Models.spx_vix_regression import *

# Download data and run analysis
spx_data, vix_data = download_spx_vix_data("2010-01-01", "2025-03-05")
spx_returns, realized_variance = compute_returns_and_realized_variance(spx_data)

# Run regression analysis
model_vix, spx_returns_vix, vix_close = regression_spx_returns_vs_vix(spx_returns, vix_data)
print(model_vix.summary())
```

### Estimating VIX from Option Data

```python
from FinDatSc_Models.vix_estimator import *
import yfinance as yf

# Download option data
ticker = yf.Ticker("^SPX")
expiry = "2025-04-03"
calls_df, puts_df = get_option_chain(ticker, expiry)

# Calculate VIX estimate
S0 = 5400  # Current SPX level
F0 = compute_forward_price(S0, r=0.02, T=30/365)
vix_estimate = estimate_vix_improved(calls_df, puts_df, F0)
print(f"VIX Estimate: {vix_estimate:.2f}%")
```

### Computing Historical Volatility

```python
from FinDatSc_Models.volatility_estimators import *

# Download MSFT data and compute rolling volatility
data = download_msft_data("2020-01-01", "2024-01-01")
rolling_vol = compute_rolling_estimators(data, window_size=30)

# Display results
print(rolling_vol.head())
```

