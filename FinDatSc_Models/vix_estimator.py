import yfinance as yf
import datetime
import numpy as np
import math
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm
import statsmodels.formula.api as smf
from statsmodels.tsa.stattools import coint

# =============================================================================
# 1) DATA DOWNLOAD & FORWARD PRICE CALCULATION
# =============================================================================

def download_spx_and_vix(today="2025-03-05"):
    """
    Downloads historical SPX and VIX data for the past year ending on `today`.
    
    Returns:
        spx_data: DataFrame with SPX historical prices
        vix_data: DataFrame with VIX close price on the last SPX business day
        last_bus_day: Timestamp of last available SPX trading day
    """
    end_date = today
    start_date = datetime.datetime.strptime(end_date, "%Y-%m-%d") - datetime.timedelta(days=365)
    
    spx_data = yf.download("^SPX", start=start_date.strftime("%Y-%m-%d"), end=end_date, auto_adjust=False)
    last_bus_day = spx_data.index[-1]
    
    vix_data = yf.download("^VIX", start=last_bus_day, end=last_bus_day + datetime.timedelta(days=1))
    
    return spx_data, vix_data, last_bus_day

def compute_forward_price(S0, r=0.02, T=1.0):
    """
    Computes forward price from spot using continuous compounding: F = S * exp(rT)
    
    Args:
        S0: Spot price
        r: Risk-free rate
        T: Time to maturity in years

    Returns:
        Forward price F0
    """
    return S0 * math.exp(r * T)

# =============================================================================
# 2) OPTION CHAIN RETRIEVAL
# =============================================================================

def get_option_chain(ticker, expiry):
    """
    Retrieves calls and puts DataFrames for a given ticker and expiry.
    
    Args:
        ticker: yfinance.Ticker object
        expiry: string date in "YYYY-MM-DD" format

    Returns:
        calls_df: DataFrame of call options
        puts_df: DataFrame of put options
    """
    chain = ticker.option_chain(expiry)
    return chain.calls, chain.puts

# =============================================================================
# 3) BASIC VIX ESTIMATION FROM EQUATION (19)
# =============================================================================

def estimate_vix_equation19(calls_df, puts_df, F0, r=0.02, tau=30/365):
    """
    Estimates VIX using Equation (19) with last prices.
    
    Args:
        calls_df, puts_df: DataFrames of call and put options
        F0: Forward price
        r: Risk-free rate
        tau: Time to maturity in years (30/365 default)

    Returns:
        VIX estimate (annualized, in percentage points)
    """
    F_strike = round(F0)

    puts_otm = puts_df[puts_df['strike'] < F_strike].copy()
    calls_otm = calls_df[calls_df['strike'] > F_strike].copy()

    all_strikes = sorted(set(puts_otm['strike']).union(calls_otm['strike']))
    if len(all_strikes) < 2:
        raise ValueError("Not enough strikes to compute dK.")
    dK = np.mean(np.diff(all_strikes))

    sum_puts = np.sum(puts_otm['lastPrice'] / puts_otm['strike']**2) * dK
    sum_calls = np.sum(calls_otm['lastPrice'] / calls_otm['strike']**2) * dK

    VIX_sq = (2 * np.exp(r * tau) / tau) * (sum_puts + sum_calls)
    return 100 * np.sqrt(VIX_sq)

# =============================================================================
# 4) IMPROVED VIX ESTIMATOR WITH MID-PRICES AND STRIKE FILTERING
# =============================================================================

def estimate_vix_improved(calls_df, puts_df, F0, r=0.02, tau=30/365):
    """
    Computes a refined VIX estimate using mid-prices and filtered strikes.
    
    - Filters to strikes within ±20% of F0
    - Uses mid = (bid + ask)/2 for prices
    - Drops invalid or nonpositive prices

    Returns:
        Improved VIX estimate (annualized, in percentage points)
    """
    calls_df['mid'] = (calls_df['bid'] + calls_df['ask']) / 2
    puts_df['mid'] = (puts_df['bid'] + puts_df['ask']) / 2

    calls_clean = calls_df.dropna(subset=["mid"])
    puts_clean = puts_df.dropna(subset=["mid"])
    calls_clean = calls_clean[calls_clean["mid"] > 0]
    puts_clean = puts_clean[puts_clean["mid"] > 0]

    strike_min = 0.8 * F0
    strike_max = 1.2 * F0
    calls_clean = calls_clean[
        (calls_clean["strike"] > F0) &
        (calls_clean["strike"] >= strike_min) &
        (calls_clean["strike"] <= strike_max)
    ]
    puts_clean = puts_clean[
        (puts_clean["strike"] < F0) &
        (puts_clean["strike"] >= strike_min) &
        (puts_clean["strike"] <= strike_max)
    ]

    all_strikes = sorted(set(puts_clean["strike"]).union(calls_clean["strike"]))
    if len(all_strikes) < 2:
        raise ValueError("Not enough clean strikes to compute dK.")
    dK = np.mean(np.diff(all_strikes))

    sum_puts = np.sum(puts_clean["mid"] / puts_clean["strike"]**2) * dK
    sum_calls = np.sum(calls_clean["mid"] / calls_clean["strike"]**2) * dK

    VIX_sq = (2 * np.exp(r * tau) / tau) * (sum_puts + sum_calls)
    return 100 * np.sqrt(VIX_sq)

# =============================================================================
# 5) COMPARISON FUNCTIONS: VIX vs HISTORICAL VOLATILITY
# =============================================================================

def download_vix_series(start_date="2010-01-01", end_date=None):
    """
    Download historical CBOE VIX daily close values.
    """
    if end_date is None:
        end_date = datetime.datetime.now().strftime("%Y-%m-%d")
    data = yf.download("^VIX", start=start_date, end=end_date)
    return data[['Close']].rename(columns={"Close": "VIX_close"}).dropna()

def merge_vol_series(hist_vol_df, vix_df):
    """
    Merge historical volatility estimates with VIX time series.
    Only rows with data in both are kept.
    """
    merged = pd.merge(hist_vol_df, vix_df, left_index=True, right_index=True, how="inner")
    return merged

def compute_correlation_cointegration(merged_df, col1, col2):
    """
    Compute correlation and cointegration between two volatility series.
    
    Args:
        merged_df: DataFrame with aligned series
        col1, col2: Column names of volatility series to compare

    Returns:
        corr: Pearson correlation coefficient
        coint_pval: p-value from cointegration test
    """
    corr = merged_df[col1].corr(merged_df[col2])
    _, pval, _ = coint(merged_df[col1], merged_df[col2])
    return corr, pval

def plot_volatility_comparison(merged_df, hist_col, vix_col, title="Volatility Comparison"):
    """
    Plot historical vs implied volatility over time.
    """
    plt.figure(figsize=(12, 5))
    plt.plot(merged_df.index, merged_df[hist_col], label=hist_col)
    plt.plot(merged_df.index, merged_df[vix_col], label=vix_col)
    plt.title(title)
    plt.xlabel("Date")
    plt.ylabel("Volatility (Annualized %)")
    plt.legend()
    plt.tight_layout()
    plt.show()