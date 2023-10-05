import pandas as pd
import numpy as np
from sklearn.linear_model import Lasso
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import yfinance as yf 
from datetime import datetime
#import matplotlib.pyplot as plt
#import scipy.special as sc
import csv

from data_gene import GBMsimulator
from monte_carlo import mc_simu
from price_extractor import select_stock, price_extract, mean_cov  
from calculator import calculate_assets_expectedreturns, calculate_assets_covariance

start_date = "2007-01-01"
end_date = "2016-12-31"

sp500_tickers = ['AAPL', 'MSFT', 'AMZN', 'GOOGL', 'FB', 'TSLA', 'JPM', 'JNJ', 'V', 'PG',
                 'MA', 'NVDA', 'UNH', 'HD', 'VZ', 'DIS', 'PYPL', 'CMCSA', 'BAC', 'CSCO',
                 'INTC', 'T', 'PFE', 'ORCL', 'KO', 'MRK', 'XOM', 'NFLX', 'WMT', 'MCD',
                 'BA', 'TMO', 'CAT', 'ADBE', 'ABBV', 'NKE', 'CRM', 'PEP', 'GS', 'CVX',
                 'PM', 'UPS', 'COST', 'C', 'ABT', 'WFC', 'ACN', 'QCOM', 'AMGN', 'TXN',
                 'NEE', 'DHR', 'SO', 'ADP', 'HON', 'TMO', 'LMT', 'RTX', 'MO', 'MDT',
                 'AMAT', 'CME', 'GD', 'BKNG', 'CSX', 'BIIB', 'COF', 'CVS', 'STZ', 'AXP',
                 'GM', 'LRCX', 'MU', 'MMM', 'USB', 'FDX', 'ISRG', 'CCI', 'EQIX', 'CSX',
                 'SO', 'ADP', 'HON', 'TMO', 'LMT', 'RTX', 'MO', 'MDT', 'AMAT', 'CME',
                 'GD', 'BKNG', 'CSX', 'BIIB', 'COF', 'CVS', 'STZ', 'AXP', 'GM', 'LRCX']

stock_data = {}  # Create a dictionary to store data for each stock

for ticker in sp500_tickers:
    stock_data[ticker] = yf.download(ticker, start=start_date, end=end_date)

# Calculate daily returns for each stock
for ticker, data in stock_data.items():
    data['Daily_Return'] = data['Adj Close'].pct_change()


# Calculate Daily Returns, and Expected Mean Return & Covariance
meanReturns = calculate_assets_expectedreturns(data['Daily_Return'])
covMatrix = calculate_assets_covariance(data['Daily_Return'])

print(meanReturns, covMatrix)