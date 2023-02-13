import pandas as pd
import numpy as np
import pandas_datareader as pdr
from datetime import datetime
import matplotlib.pyplot as plt

from data_gene import GBMsimulator
from read_real_data import get_data
from monte_carlo import mc_simu
from random_select_stock import select_stock

# 1. conduct subpool 
# 2. estimate the square of macimum sharpe ratio by theta_hat in 2.32, and compute the response r_c_hat in 2.10
# 3. select lambda through CV according to 2.5.2, demoted by lambda_hat
# 4. set lambda in 2.12 to be lambda_hat and solve for the MAAXSER portfolio omega_hat_star in 2.12

# compare with benchmark portfolio 




if __name__ == '__main__':

    sigma = 0.04 # variance constraint
    T = 120 # sample size
    replica = 1000 # replication times
    # theoratical maximum SR = 1.882

    N = 100 # num of stocks



    


