import yfinance as yf
import datetime
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import math
import statsmodels.api as sm
from statsmodels.tsa.stattools import coint

# =============================================================================
# 1) HELPER FUNCTIONS FOR EACH ESTIMATOR
# =============================================================================

def classical_mean_equation9(prices, dt):
    """
    Computes the classical historical mean (annualized) per Equation (9):
    
      mu_hat = (1/N) * sum_{k=0}^{N-1} [1/(t_{k+1} - t_k)] * [(S_{k+1} - S_k)/S_k].
    
    We assume each (t_{k+1} - t_k) = dt (in years, e.g. 1/252).
    """
    N = len(prices) - 1
    if N < 1:
        return np.nan
    returns = (prices[1:] - prices[:-1]) / prices[:-1]
    mu_hat = (1 / N) * np.sum(returns / dt)
    return float(mu_hat)

def classical_volatility_equation14(prices, mu_hat, dt):
    """
    Computes the classical historical volatility (annualized) per Equation (14):
    
      sigma_hat^2 = (1/(N-1)) * sum_{k=0}^{N-1} [1/(t_{k+1}-t_k)] * ( r_k - dt*mu_hat )^2,
      
    with r_k = (S_{k+1}-S_k)/S_k, and then sigma_hat = sqrt(sigma_hat^2).
    """
    N = len(prices) - 1
    if N < 2:
        return np.nan
    returns = (prices[1:] - prices[:-1]) / prices[:-1]
    sum_sq = 0.0
    for r in returns:
        sum_sq += (r - dt * mu_hat)**2 / dt
    sigma_squared_hat = (1 / (N - 1)) * sum_sq
    sigma_hat = np.sqrt(sigma_squared_hat)
    return float(sigma_hat)

def parkinson_estimator_equation2(highs, lows):
    """
    Computes the Parkinson volatility (annualized) per Equation (2):
    
      sigma_Parkinson = sqrt( (1 / (4 ln2 * T)) * sum_{t=1}^T [ln(h_t / l_t)]^2 ).
    """
    T = len(highs)
    if T < 1:
        return np.nan
    factor = 1.0 / (4.0 * math.log(2.0))
    sum_ln_sqr = 0.0
    for h, l in zip(highs, lows):
        h_scalar = float(h)
        l_scalar = float(l)
        if l_scalar > 0:
            sum_ln_sqr += (math.log(h_scalar / l_scalar))**2
    sigma_squared_park = factor * (sum_ln_sqr / T)
    return float(math.sqrt(sigma_squared_park))

# =============================================================================
# 2) DATA DOWNLOAD & PREPARATION FUNCTION
# =============================================================================

def download_msft_data(start_date="2010-01-01", end_date=None, adjust=False):
    """Download MSFT daily data (Open, High, Low, Close) from Yahoo Finance."""
    if end_date is None:
        end_date = datetime.datetime.now().strftime("%Y-%m-%d")
    data = yf.download("MSFT", start=start_date, end=end_date, auto_adjust=adjust)
    data = data[['Open', 'High', 'Low', 'Close']].dropna()
    return data

# =============================================================================
# 3) ROLLING WINDOW ESTIMATION FUNCTION
# =============================================================================

def compute_rolling_estimators(data, window_size=30, dt=1.0/252.0):
    """
    Compute rolling estimators (classical mean, classical vol, Parkinson vol)
    on the provided data and return a DataFrame with results.
    """
    prices_all = data['Close'].values
    highs_all = data['High'].values
    lows_all  = data['Low'].values
    
    rolling_mu = []
    rolling_sigma = []
    rolling_park = []
    idx_vals = []
    
    for i in range(window_size, len(data) + 1):
        window_prices = prices_all[i - window_size : i]
        window_highs  = highs_all[i - window_size : i]
        window_lows   = lows_all[i - window_size : i]
        
        mu_w = classical_mean_equation9(window_prices, dt)
        sigma_w = classical_volatility_equation14(window_prices, mu_w, dt)
        park_w = parkinson_estimator_equation2(window_highs, window_lows)
        
        rolling_mu.append(mu_w)
        rolling_sigma.append(sigma_w)
        rolling_park.append(park_w)
        idx_vals.append(data.index[i-1])
    
    df_roll = pd.DataFrame({
        'mu_rolling': rolling_mu,
        'sigma_rolling': rolling_sigma,
        'parkinson_rolling': rolling_park
    }, index=idx_vals)
    
    return df_roll