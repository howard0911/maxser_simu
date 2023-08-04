import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import datetime as dt
import yfinance as yf
# yf.pdr_override()

# import data
def get_data(stocks, start, end):
    stockData = yf.download(stocks, start, end)
    stockData = stockData['Close']
    returns = stockData.pct_change()
    meanReturns = returns.mean()
    covMatrix = returns.cov()
    return meanReturns, covMatrix


stockList = ['CBA', 'BHP', 'TLS', 'NAB', 'WBC', 'STO']
stocks = [stock + '.AX' for stock in stockList]
endDate = dt.datetime.now()
startDate = endDate - dt.timedelta(days=300)

meanReturns, covMatrix = get_data(stocks, startDate, endDate)
# num of simulations
mc_sims = 100
T = 100 # tiemframe in days
initialPortfolio = 10000

def mc_simu(meanReturns, covMatrix, mc_sims, T, initialPortfolio):
    weights = np.random.random(len(meanReturns))
    weights /= np.sum(weights)
    
    meanM = np.full(shape=(T, len(weights)), fill_value=meanReturns)
    meanM = meanM.T

    portfolio_sims = np.full(shape=(T, mc_sims), fill_value=0.0)

    for m in range(0, mc_sims):
        # mc loops
        Z = np.random.normal(size=(T, len(weights)))
        L = np.linalg.cholesky(covMatrix)
        dailyReturns = meanM + np.inner(L, Z)
        portfolio_sims[:,m] = np.cumprod(np.inner(weights, dailyReturns.T)+1)*initialPortfolio

    portfolio_sims = pd.DataFrame(portfolio_sims)
    return portfolio_sims

#print(mc_simu(meanReturns, covMatrix, mc_sims, T, initialPortfolio))
#plt.plot(mc_simu(meanReturns, covMatrix, mc_sims, T, initialPortfolio))
#plt.ylabel('Portfolio Value ($)')
#plt.xlabel('Days')
#plt.title('MC simulations of a stock portfolio')
#plt.show()

