# basics
from datetime import datetime
import time

#plotting
import matplotlib.pyplot as plt

# ml shit
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import scipy.stats as stats
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import make_scorer

# models
from sklearn.preprocessing import StandardScaler
from lightgbm import LGBMRegressor

# relative imports
from helperfuncs import get_time_fractions


# for now, just btc and eth 
# for some reason, it is not working properly
def get_features(data_root, nrows=None):
    df = pd.read_csv(data_root+"train.csv", nrows=nrows)


    assets = pd.read_csv(data_root+"asset_details.csv")
    df_grouped = df.groupby("Asset_ID")

    # we will work with other 12 assets
    btc = df_grouped.get_group(1)
    eth = df_grouped.get_group(6)


    btc.set_index("timestamp", inplace=True)
    btc = btc.drop(columns=["Target", "Asset_ID"])
    btc = btc.add_suffix("_btc")
    eth.set_index("timestamp", inplace=True)
    eth = eth.drop(columns=["Target", "Asset_ID"])
    eth = eth.add_suffix("_eth")

    btc_eth = pd.concat([btc, eth], axis=1)
    #filling nans for now(reindex and drop later)
    btc_eth.fillna(method="ffill", inplace=True)
    # btc_eth = btc_eth.fillna(method="ffill").fillna(method="bfill")

    df_features = btc_eth.copy()
    suffixes = ["_btc", "_eth"]
    for suffix in suffixes:
        df_features["logprice"+suffix] = np.log(df_features["Close"+suffix]) 
        df_features["Volatility"+suffix] = np.log(df_features["High"+suffix])\
            - np.log(df_features["Close"+suffix])
        df_features = df_features.drop(columns=["Close"+suffix, "High"+suffix,\
            "Low"+suffix, "Open"+suffix, "VWAP"+suffix])

    datetimes = pd.Series(df_features.index).astype("datetime64[s]")
    df_features["frac_day"], df_features["frac_week"], df_features['frac_of_month'], \
        df_features['frac_of_year'] = zip(*datetimes.map(get_time_fractions))


    # calculate 2-asset targets(not 14 asset target)
    df_logprices = df_features[["logprice_btc", "logprice_eth"]]
    # 𝑅𝑎(𝑡)=𝑙𝑜𝑔(𝑃𝑎(𝑡+16) / 𝑃𝑎(𝑡+1))=𝑙𝑜𝑔(𝑃𝑎(𝑡+16)−𝑙𝑜𝑔(𝑃𝑎(𝑡+1)
    df_returns = df_logprices.shift(-16) - df_logprices.shift(-1)
    for suffix in suffixes:
        df_returns.rename(columns={"logprice"+suffix : "R"+suffix}, inplace=True)

    # find a better way to write next line
    assets =  assets[(assets["Asset_ID"] == 1) | (assets["Asset_ID"] == 6)]
    assets = assets.sort_values(by=["Asset_ID"])
    weights = assets["Weight"].to_numpy()
    weights = weights.reshape(len(weights), 1)
    

    R = df_returns.to_numpy()# to array
    weights_sum = np.sum(weights)
    M = np.dot(R, weights) / weights_sum # weighted average => log_btc*w_btc + log_eth*w_eth
    df_M = pd.DataFrame(data=M, index=df_returns.index, columns=["M"])
    R.shape,weights.shape, M.shape


    df_R_M = df_returns.copy()
    for col in df_R_M.columns:
        df_R_M[col] = df_R_M[col] * df_M["M"] # calculated R・M here
    for suffix in suffixes:
        df_R_M.rename(columns={"R"+suffix:"R_M"+suffix}, inplace=True)
    df_R_M_rolling = df_R_M.rolling(window=3750).mean()


    # creating M^2 
    df_M2 = df_M ** 2
    df_M2.rename(columns={"M" : "M2"}, inplace = True)
    df_M2_rolling = df_M2.rolling(window=3750).mean()
    df_betas = df_R_M_rolling.copy()    
    for col in df_betas.columns: # columns = [R_M_btc	R_M_eth]   
        df_betas[col] = df_betas[col] / df_M2_rolling["M2"] # caculating <R・M>/<M^2> here
    for suffix in suffixes: # beta = <R・M>/<M^2> 
        df_betas.rename(columns={"R_M"+suffix : "beta"+suffix}, inplace = True)
    df_targets = df_returns.copy()
    for suffix in suffixes:
        df_targets["R"+suffix] -= df_betas["beta"+suffix] * df_M["M"] # R^a - β^a
        df_targets.rename(columns={"R"+suffix: "Target"+suffix}, inplace=True)

    df_features_targets = pd.concat([df_features, df_betas, df_targets], axis=1)
    df_features_targets = df_features_targets.iloc[3750:-16] # drop nan rows

    return df_features_targets

