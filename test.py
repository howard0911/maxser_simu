import pandas as pd
import pandas_datareader as pdr
from pandas_datareader import data as wb
import yfinance as yf
yf.pdr_override()

ticker='GOOGL'
start_date='2019-1-1'
data_source='yahoo'

ticker_data=wb.DataReader(ticker,data_source=data_source,start=start_date)
df=pd.DataFrame(ticker_data)

