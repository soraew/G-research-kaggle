# basics
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from datetime import datetime
import time

# plotting
import plotly.graph_objects as go
import seaborn as sns
import matplotlib.pyplot as plt

# ml shit
import scipy.stats as stats
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.multioutput import MultiOutputRegressor

data_root = "../data/"


def log_return(series, periods=1):
    return np.log(series).diff(periods=periods)[1:]


train = pd.read_csv(data_root+"g-research-crypto-forecasting/train.csv")
asset_d = pd.read_csv(data_root+"asset_details.csv")

train_Assets = train.groupby("Asset_ID")
btc = train_Assets.get_group(1).set_index("timestamp")
eth = train_Assets.get_group(6).set_index("timestamp")

btc = btc.reindex(range(eth.index[0], eth.index[-1]+60, 60), method='pad')
eth = eth.reindex(range(eth.index[0], eth.index[-1]+60, 60), method='pad')

totimestamp = lambda s: np.int32(time.mktime(datetime.strptime(s, "%d/%m/%Y").timetuple()))

# for only year 2021
# btc_mini_2021 = btc.loc[totimestamp('01/06/2021'):totimestamp('01/07/2021')]
# eth_mini_2021 = eth.loc[totimestamp('01/06/2021'):totimestamp('01/07/2021')]

# lret_btc = log_return(btc_mini_2021.Close)
# lret_eth = log_return(eth_mini_2021.Close)

# lret_btc.rename('lret_btc', inplace=True)
# lret_eth.rename('lret_eth', inplace=True)

lret_btc_long = log_return(btc.Close)[1:]
lret_eth_long = log_return(eth.Close)[1:]
lret_btc_long.rename("lret_btc", inplace=True)
lret_eth_long.rename("lret_eth", inplace=True)
two_assets = pd.concat([lret_btc_long, lret_eth_long], axis=1)


# Select some input features from the trading data: 
# 5 min log return, abs(5 min log return), upper shadow, and lower shadow.
upper_shadow = lambda asset: asset.High - np.maximum(asset.Close, asset.Open)
lower_shadow = lambda asset: -asset.Low + np.minimum(asset.Close, asset.Open)

X_btc = pd.concat([log_return(btc.VWAP, periods=5), log_return(btc.VWAP, periods=1).abs(),
                upper_shadow(btc), lower_shadow(btc)], axis=1)

y_btc = btc.Target

X_eth = pd.concat([log_return(eth.VWAP, periods=5),  log_return(eth.VWAP,periods=1).abs(), 
                upper_shadow(eth), lower_shadow(eth)], axis=1)

y_eth = eth.Target


# train, test split
train_window = [totimestamp("01/05/2021"), totimestamp("30/05/2021")]
test_window = [totimestamp("01/06/2021"), totimestamp("30/06/2021")]

X_btc_train = X_btc.loc[train_window[0]:train_window[1]].fillna(0).to_numpy()
y_btc_train = y_btc.loc[train_window[0]:train_window[1]].fillna(0).to_numpy()

X_btc_test = X_btc.loc[test_window[0]:test_window[1]].fillna(0).to_numpy()
y_btc_test = y_btc.loc[test_window[0]:test_window[1]].fillna(0).to_numpy() 

X_eth_train = X_eth.loc[train_window[0]:train_window[1]].fillna(0).to_numpy()  
y_eth_train = y_eth.loc[train_window[0]:train_window[1]].fillna(0).to_numpy()  

X_eth_test = X_eth.loc[test_window[0]:test_window[1]].fillna(0).to_numpy() 
y_eth_test = y_eth.loc[test_window[0]:test_window[1]].fillna(0).to_numpy() 


# scale variables
scaler = StandardScaler()

X_btc_train_scaled = scaler.fit_transform(X_btc_train)
X_btc_test_scaled = scaler.fit_transform(X_btc_test)

X_eth_train_scaled = scaler.fit_transform(X_eth_train)
X_eth_test_scaled = scaler.transform(X_eth_test)


# simple linear regression
lr = LinearRegression()

lr.fit(X_btc_train_scaled, y_btc_train)
y_pred_lr_btc = lr.predict(X_btc_test_scaled)

lr.fit(X_eth_train_scaled, y_eth_train)
y_pred_lr_eth = lr.predict(X_eth_test_scaled) 


# multi target regression
mlr = MultiOutputRegressor(LinearRegression())

X_both_train = np.concatenate((X_btc_train_scaled, X_eth_train_scaled), axis=1)
X_both_test = np.concatenate((X_btc_test_scaled, X_eth_test_scaled), axis=1)
y_both_train = np.column_stack((y_btc_train, y_eth_train))
y_both_test = np.column_stack((y_btc_test, y_eth_test))

lr.fit(X_both_train, y_both_train)
y_pred_lr_both = lr.predict((X_both_test))