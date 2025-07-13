import yfinance as yf
import datetime
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm
from statsmodels.tsa.stattools import adfuller
from statsmodels.stats.diagnostic import het_breuschpagan

# =============================================================================
# 1) DOWNLOAD SPX AND VIX DATA (Historical)
# =============================================================================

def download_spx_vix_data(start_date="2010-01-01", end_date="2025-03-05"):
    """
    Download historical SPX and VIX data between start_date and end_date.
    
    Returns:
        spx_data: DataFrame with SPX historical prices (auto-adjusted)
        vix_data: DataFrame with VIX historical prices (auto-adjusted)
    """
    spx_data = yf.download("^SPX", start=start_date, end=end_date, auto_adjust=True)
    vix_data = yf.download("^VIX", start=start_date, end=end_date, auto_adjust=True)
    
    spx_data.dropna(inplace=True)
    vix_data.dropna(inplace=True)
    return spx_data, vix_data

# =============================================================================
# 2) COMPUTE DAILY SPX RETURNS & REALIZED VARIANCE ESTIMATOR
# =============================================================================

def compute_returns_and_realized_variance(spx_data, window=30):
    """
    Compute daily SPX returns and realized variance.
    
    - Returns are computed as percentage changes of the Close price.
    - Realized variance is computed as a 30-day rolling variance of daily returns, annualized.
    
    Returns:
        spx_returns: Series with daily returns (dropping NA's)
        realized_variance: Series with annualized rolling variance (window-day rolling)
    """
    spx_data['Return'] = spx_data['Close'].pct_change()
    spx_returns = spx_data['Return'].dropna()
    
    realized_variance = spx_returns.rolling(window=window).var() * 252
    realized_variance = realized_variance.dropna()
    return spx_returns, realized_variance

# =============================================================================
# 3) REGRESSION ANALYSIS: SPX RETURNS vs. VIX
# =============================================================================

def regression_spx_returns_vs_vix(spx_returns, vix_data):
    """
    Run OLS regression of SPX returns on VIX.
    
    Returns:
        model: Fitted OLS regression model.
        spx_returns_aligned: SPX returns aligned on common dates.
        vix_close: VIX closing prices aligned on common dates.
    """
    common_dates = spx_returns.index.intersection(vix_data.index)
    spx_returns_aligned = spx_returns.loc[common_dates]
    vix_close = vix_data['Close'].loc[common_dates]
    
    X = sm.add_constant(vix_close)
    model = sm.OLS(spx_returns_aligned, X).fit()
    return model, spx_returns_aligned, vix_close

def plot_regression_vix(spx_returns, vix_close, model):
    """
    Plot regression of SPX returns versus VIX using scatter plot and fitted line.
    """
    plt.figure(figsize=(10, 6))
    plt.scatter(vix_close, spx_returns, alpha=0.5, label="Data")
    x_range = np.linspace(vix_close.min(), vix_close.max(), 100)
    X_plot = sm.add_constant(x_range)
    y_fit = model.predict(X_plot)
    plt.plot(x_range, y_fit, color='red', label="Fitted Line")
    plt.xlabel("VIX (Implied Volatility)", fontsize=16)
    plt.ylabel("SPX Daily Return", fontsize=16)
    plt.title("Regression of SPX Returns on VIX", fontsize=18)
    plt.legend()
    plt.tight_layout()
    plt.show()

# =============================================================================
# 4) REGRESSION ANALYSIS: SPX RETURNS vs. REALIZED VARIANCE
# =============================================================================

def regression_spx_returns_vs_realized_variance(spx_returns, realized_variance):
    """
    Run OLS regression of SPX returns on Realized Variance.
    
    Returns:
        model: Fitted OLS regression model.
        spx_returns_aligned: SPX returns (aligned on common dates).
        realized_variance_aligned: Realized variance (aligned on common dates).
    """
    common_dates = spx_returns.index.intersection(realized_variance.index)
    spx_returns_aligned = spx_returns.loc[common_dates]
    realized_variance_aligned = realized_variance.loc[common_dates]
    
    X = sm.add_constant(realized_variance_aligned)
    model = sm.OLS(spx_returns_aligned, X).fit()
    return model, spx_returns_aligned, realized_variance_aligned

def plot_regression_realized_variance(spx_returns, realized_variance, model):
    """
    Plot regression of SPX returns versus Realized Variance (scatter and fitted line).
    """
    plt.figure(figsize=(10, 6))
    plt.scatter(realized_variance, spx_returns, alpha=0.5, label="Data", color='green')
    x_range = np.linspace(realized_variance.min(), realized_variance.max(), 100)
    X_plot = sm.add_constant(x_range)
    y_fit = model.predict(X_plot)
    plt.plot(x_range, y_fit, color='black', label="Fitted Line")
    plt.xlabel("Realized Variance (Annualized, 30-day Rolling)", fontsize=16)
    plt.ylabel("SPX Daily Return", fontsize=16)
    plt.title("Regression of SPX Returns on Realized Variance", fontsize=18)
    plt.legend()
    plt.tight_layout()
    plt.show()

# =============================================================================
# 5) ADDITIONAL DIAGNOSTIC TESTS FOR THE REGRESSIONS
# =============================================================================

def regression_diagnostics(model, X):
    """
    Perform diagnostic tests on regression residuals:
      - Breusch-Pagan for heteroscedasticity
      - Augmented Dickey-Fuller (ADF) test for stationarity
    
    Returns:
        diagnostics: Dictionary of test statistics and p-values.
    """
    resid = model.resid
    lm_stat, lm_pvalue, fvalue, f_pvalue = het_breuschpagan(resid, X)
    adf_stat, adf_pvalue, _, _, _, _ = adfuller(resid)
    diagnostics = {
        'LM Statistic': lm_stat,
        'LM p-value': lm_pvalue,
        'F-Statistic': fvalue,
        'F p-value': f_pvalue,
        'ADF Statistic': adf_stat,
        'ADF p-value': adf_pvalue
    }
    return diagnostics

def regression_spx_returns_vs_delta_vix(spx_returns, vix_data):
    """
    Run OLS regression of SPX returns on daily change in VIX (∆VIX).
    Returns the fitted model and the aligned series.
    """
    # compute ∆VIX
    vix = vix_data['Close']
    delta_vix = vix.diff().dropna()
    
    # align
    common = spx_returns.index.intersection(delta_vix.index)
    y = spx_returns.loc[common]
    x = delta_vix.loc[common]
    
    X = sm.add_constant(x)
    model = sm.OLS(y, X).fit()
    return model, y, x


def plot_regression_delta_vix(spx_returns, delta_vix, model):
    """
    Scatter + fitted line of SPX returns vs ∆VIX.
    """
    plt.figure(figsize=(10,6))
    plt.scatter(delta_vix, spx_returns, alpha=0.5, label="Data", color='purple')
    xx = np.linspace(delta_vix.min(), delta_vix.max(), 100)
    XX = sm.add_constant(xx)
    plt.plot(xx, model.predict(XX), color='black', lw=2, label="Fitted Line")
    plt.xlabel("∆VIX (daily change)", fontsize=16)
    plt.ylabel("SPX Daily Return", fontsize=16)
    plt.title("Regression of SPX Returns on ∆VIX", fontsize=18)
    plt.legend(fontsize=14)
    plt.tight_layout()
    plt.show()
    
# =============================================================================
# MAIN SCRIPT (for command-line or interactive run)
# =============================================================================

if __name__ == '__main__':
    # Set the date range
    start_date = "2010-01-01"
    end_date = "2025-03-05"
    
    # 1) Download data
    spx_data, vix_data = download_spx_vix_data(start_date, end_date)
    
    # 2) Compute returns and realized variance (30-day window)
    spx_returns, realized_variance = compute_returns_and_realized_variance(spx_data, window=30)
    
    # 3) Regression: SPX Returns vs. VIX
    model_vix, spx_returns_vix, vix_close = regression_spx_returns_vs_vix(spx_returns, vix_data)
    print("=== Regression: SPX Returns vs. VIX ===")
    print(model_vix.summary())
    plot_regression_vix(spx_returns_vix, vix_close, model_vix)
    
    # 4) Regression: SPX Returns vs. Realized Variance
    model_rv, spx_returns_rv, realized_variance_aligned = regression_spx_returns_vs_realized_variance(spx_returns, realized_variance)
    print("\n=== Regression: SPX Returns vs. Realized Variance ===")
    print(model_rv.summary())
    plot_regression_realized_variance(spx_returns_rv, realized_variance_aligned, model_rv)
    
    # 5) Diagnostics
    print("\n--- Diagnostic Tests for SPX Returns vs. VIX Model ---")
    X_vix = sm.add_constant(vix_close)
    diagnostics_vix = regression_diagnostics(model_vix, X_vix)
    for k, v in diagnostics_vix.items():
        print(f"{k}: {v:.4f}")
    
    print("\n--- Diagnostic Tests for SPX Returns vs. Realized Variance Model ---")
    X_rv = sm.add_constant(realized_variance_aligned)
    diagnostics_rv = regression_diagnostics(model_rv, X_rv)
    for k, v in diagnostics_rv.items():
        print(f"{k}: {v:.4f}")

