import pandas as pd
import numpy as np
from sklearn.linear_model import Lasso
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import yfinance as yf 
from datetime import datetime
import matplotlib.pyplot as plt
import scipy.special as sc
import csv

from data_gene import GBMsimulator
from monte_carlo import mc_simu
from price_extractor import select_stock, price_extract, mean_cov  
from calculator import calculate_assets_expectedreturns, calculate_assets_covariance

## implementation steps(scenario 1)
# 1. conduct subpool (option)
# 2. estimate the square of maximum sharpe ratio by theta_hat in 2.32, and compute the response r_c_hat in 2.10
# 3. select lambda through CV according to 2.5.2, demoted by lambda_hat
# 4. set lambda in 2.12 to be lambda_hat and solve for the MAAXSER portfolio omega_hat_star in 2.12
# compare with benchmark portfolio 




if __name__ == '__main__':

    # Parameters settings
    sigma = 0.04 # variance constraint
    T = 120 # sample size
    replica = 1000 # replication times
    # theoratical maximum SR = 1.882
    N = 50 # num of stocks
    n_stocks_available = 500

    #1. Get companies
    #companies = select_stock((n_stocks_available, N))

    #2. Get company stock prices
    #price = price_extract()

    #3. Calculate Daily Returns, and Expected Mean Return & Covariance
    # meanReturns, covMatrix = mean_cov()
    #meanReturns = calculate_assets_expectedreturns(returns)
    #covMatrix = calculate_assets_covariance(returns)
 
    #4. Use Monte Carlo Simulation
    # Generate portfolios with allocations
    # num of simulations
    mc_sims = 100
    #T = 100 # timeframe in days
    initialPortfolio = 10000
    portfolios_allocations_df = mc_simu(meanReturns, covMatrix, mc_sims, T, initialPortfolio)
    meanReturns = calculate_assets_expectedreturns(portfolios_allocations_df)        
    covMatrix = calculate_assets_covariance(portfolios_allocations_df)

    #6. conduct subpool(optional)

    #7. estimate the square of maximum sharpe ratio theta_hat 2.32
    temp = np.dot(np.transpose(np.array(meanReturns)), np.linalg.inv(np.array(covMatrix)))
    theta_s = np.dot(temp, np.array(meanReturns))
    theta_adj_hat = ((T-N-2)*theta_s-N)/T + (2*theta_s**(N/2)*(1+theta_s)**(-(T-2)/2))/(T*sc.betainc(N/2,(T-N)/2, x=theta_s/(1+theta_s)))

    #8. compute r_c
    r_c = sigma*(1+theta_adj_hat)/np.sqrt(theta_adj_hat)

    #9. CV for select lambda_hat

    print(r_c)

    #10. solve w_hat for MAXSER



    


