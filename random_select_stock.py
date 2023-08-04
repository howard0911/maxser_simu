import numpy as np 

n_stocks_available=500 
n_stocks=100 

def select_stock(n_stocks_available, n_stocks):
    np.random.seed(1345) 
    x = np.random.choice(n_stocks_available, n_stocks)
    return x

print(select_stock(n_stocks_available, n_stocks))