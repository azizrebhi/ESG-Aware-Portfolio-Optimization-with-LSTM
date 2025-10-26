import pandas as pd

def compute_returns(data, window=3):
    returns = data.pct_change().dropna()
    returns = returns.rolling(window=window).mean().dropna()
    return returns

def compute_covariance(returns, span=60):
    ewm_cov = returns.ewm(span=span).cov(pairwise=True)
    last_date = ewm_cov.index.get_level_values(0).max()
    Sigma = ewm_cov.loc[last_date]
    return Sigma
