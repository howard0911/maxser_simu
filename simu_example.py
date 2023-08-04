import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.optimize as sco
import datetime as datetime

# Load S&P 500 data
#dateparse = lambda x: datetime.strptime(x, '%Y-%m-%d %H:%M:%S')
sp500 = pd.read_csv('sp500_3m.csv', parse_dates=['Date'], index_col='Date')

# Define portfolio weights
num_assets = 4
weights = np.array([0.25, 0.25, 0.25, 0.25])

# Calculate expected returns and covariance matrix
returns = sp500.pct_change().mean() * 252
cov_matrix = sp500.pct_change().cov() * 252

# Define objective function
def get_portfolio_return(weights, returns):
    return np.sum(weights * returns)

def get_portfolio_volatility(weights, cov_matrix):
    return np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))

def get_portfolio_sharpe_ratio(weights, returns, cov_matrix, risk_free_rate):
    portfolio_return = get_portfolio_return(weights, returns)
    portfolio_volatility = get_portfolio_volatility(weights, cov_matrix)
    sharpe_ratio = (portfolio_return - risk_free_rate) / portfolio_volatility
    return -sharpe_ratio

# Set optimization parameters
bounds = tuple((0,1) for i in range(num_assets))
constraints = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1})

# Optimize portfolio using MaxSharpe ratio method
risk_free_rate = 0.02
result = sco.minimize(get_portfolio_sharpe_ratio, weights, args=(returns, cov_matrix, risk_free_rate), method='SLSQP', bounds=bounds, constraints=constraints)
optimized_weights = result.x

# Simulate portfolio
initial_investment = 1000000
portfolio_values = [initial_investment]
for i in range(1, len(sp500)):
    returns_sample = sp500.iloc[i-1:i].pct_change().values
    portfolio_value = np.sum(portfolio_values[-1] * optimized_weights * (1 + returns_sample))
    portfolio_values.append(portfolio_value)

# Plot results
plt.plot(sp500.index, portfolio_values)
plt.title('Portfolio Value over Time (MaxSharpe)')
plt.xlabel('Date')
plt.ylabel('Value')
plt.show()
