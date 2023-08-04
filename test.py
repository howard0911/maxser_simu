import pandas as pd
import numpy as np
import pandas_datareader as pdr
from pandas_datareader import data as wb
import yfinance as yf
yf.pdr_override()
from calculator import calculate_assets_expectedreturns, calculate_assets_covariance
from monte_carlo import mc_simu

# Parameters settings
sigma = 0.04 # variance constraint
T = 120 # sample size
replica = 1000 # replication times
# theoratical maximum SR = 1.882
N = 50 # num of stocks
n_stocks_available = 500

returns = pd.DataFrame({
        'AAPL': [0.01, 0.02, -0.03, 0.04, 0.01, -0.02],
        'GOOG': [0.02, 0.03, -0.1, 0.05, 0.03, -0.01],
        'AMZN': [0.03, 0.01, -0.01, 0.04, 0.02, -0.03],
        'FB': [0.02, 0.03, -0.01, 0.06, 0.04, -0.02],
        'NFLX': [0.04, 0.05, -0.02, 0.08, 0.02, -0.01],
        'MSFT': [0.01, 0.03, -0.01, 0.05, 0.02, -0.01]
    })
print(calculate_assets_expectedreturns(returns))
print(calculate_assets_covariance(returns))
meanReturns = calculate_assets_expectedreturns(returns)
covMatrix = calculate_assets_covariance(returns)

mc_sims = 100
#T = 100 # tiemframe in days
initialPortfolio = 10000
portfolios_allocations_df = mc_simu(meanReturns, covMatrix, mc_sims, T, initialPortfolio)
print(portfolios_allocations_df)

meanReturns = calculate_assets_expectedreturns(portfolios_allocations_df)        
covMatrix = calculate_assets_covariance(portfolios_allocations_df)
print(meanReturns, covMatrix)


temp = np.dot(np.transpose(np.array(meanReturns)), np.linalg.inv(np.array(covMatrix)))
theta_s = np.dot(temp, np.array(meanReturns))
theta_adj_hat = ((T-N-2)*theta_s-N)/T  #+ (2*theta_s**(N/2)*(1+theta_s)**(-(T-2)/2))/(T*B)
print(theta_adj_hat)


