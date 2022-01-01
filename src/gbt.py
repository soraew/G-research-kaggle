# basics
from datetime import datetime
import time
import lightgbm

#plotting
import matplotlib.pyplot as plt
import seaborn as sns
from seaborn_analyzer import regplot

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
from get_feats import get_features

SEED = 2021
data_root = "data/"



df_features_targets = get_features(data_root)

df_features_targets.drop(["Target_eth", "beta_eth"], inplace = True)

y = df_features_targets[["Target_btc"]].values()
X = df_features_targets.drop(["Target_btc"], inplace=False).values()

fit_params = {
    "verbose":0,
    "early_stopping":10,
    "eval_metric":"rmse",
    "eval_set":[(X, y)]
}
cv = TimeSeriesSplit(n_splits=5)
model = LGBMRegressor()

regplot.regression_heat_plot(model, )
