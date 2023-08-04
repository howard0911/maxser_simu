import numpy as np 
import yfinance as yf

n_stocks_available=500 
n_stocks=100 

def select_stock(n_stocks_available, n_stocks):
    np.random.seed(1345) 
    x = np.random.choice(n_stocks_available, n_stocks)
    return x

def price_extract(stocks):
    price = stocks 

    return price 

def mean_cov(stocks, start, end):
    stockData = yf.download(stocks, start, end)
    stockData = stockData['Close']
    returns = stockData.pct_change()
    meanReturns = returns.mean()
    covMatrix = returns.cov()
    return meanReturns, covMatrix

stocks = select_stock(n_stocks_available, n_stocks)
print(select_stock(n_stocks_available, n_stocks))
print(mean_cov(stocks, start="2007-01-01", end="2016-12-31"))