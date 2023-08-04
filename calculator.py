import numpy as np
from functools import reduce

def calculate_sharpe_ratio(risk, returns, risk_free_rate):
    return (returns-risk_free_rate)/risk

def get_max_sharpe_ratio(df):
    return df.ix[df['SharpeRatio'].astype(float).idxmax()]

def get_min_risk(df):
    return df.ix[df['Risk'].astype(float).idxmin()]

def calculate_assets_expectedreturns(returns):        
    return returns.mean() * 252

def calculate_assets_covariance(returns):        
    return returns.cov() * 252
    
def calculate_portfolio_risk(allocations, cov):
    return np.sqrt(reduce(np.dot, [allocations, cov, allocations.T]))
 

def get_portfolio_return(weights, returns):
    return np.sum(weights * returns)

def get_portfolio_volatility(weights, cov_matrix):
    return np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))

def get_portfolio_sharpe_ratio(weights, returns, cov_matrix, risk_free_rate):
    portfolio_return = get_portfolio_return(weights, returns)
    portfolio_volatility = get_portfolio_volatility(weights, cov_matrix)
    sharpe_ratio = (portfolio_return - risk_free_rate) / portfolio_volatility
    return -sharpe_ratio