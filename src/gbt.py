# basics
from datetime import datetime
import time
import lightgbm
from lightgbm.basic import LightGBMError
from tqdm import tqdm


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
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import LinearRegression
from sklearn.multioutput import MultiOutputRegressor
from sklearn.svm import SVR
from sklearn.pipeline import make_pipeline
from sklearn.datasets import make_classification

# models
from sklearn.preprocessing import StandardScaler
from lightgbm import LGBMRegressor

# relative imports
from get_feats import get_features
from random_search import random_search_lgbm

SEED = 2021
data_root = "../data/"

print(lightgbm.__version__)

if __name__ == "__main__":
    # using only bitcoin and etherium data
    df_features_targets = get_features(data_root)
    df_features_targets.drop(columns=["Target_eth", "beta_eth"], inplace = True)

    y = df_features_targets[["Target_btc"]].to_numpy().squeeze()
    X =df_features_targets.drop(columns=["Target_btc"], inplace=False).to_numpy()

    # change if needed(copied from watanabe thesis of lab)
    params_space = {
        "max_depth":[4, 5, 6],
        "min_data_in_leaf":[15, 20, 25],
        "learning_rate":[0.01, 0.005],
        "num_leaves":[25, 30, 35, 40],
        "boosting_type":["gbdt"],
        "objective":["regression"],
        "random_state":[2021],
        "reg_alpha":[1, 102],
        "reg_lambda":[1, 1.2, 1.4]
        }

    random_search = False
    if random_search:
        # X and y here should be changed (grouped by KNN or something)
        # X_sm, y_sm = X[:90000], y[:90000]
        sorted_by_coef, sorted_by_rmse = random_search_lgbm(X_sm, y_sm, params_space)
        #rmse got lower than baseline(model=LGBMRegressor), however coef got worse
        sorted_by_coef_str = map(str, sorted_by_coef)
        sorted_by_rmse_str = map(str, sorted_by_rmse)
        print("sorted_by_rmse\n", "\n".join(sorted_by_rmse_str)) 
        print("sorted_by_rmse\n", "\n".join(sorted_by_coef_str))
    
    final_params = {'max_depth': 4,
    'min_data_in_leaf': 20,
    'learning_rate': 0.01,
    'num_leaves': 40,
    'boosting_type': 'gbdt',
    'objective': 'regression',
    'random_state': 2021,
    'reg_alpha': 1.2,
    'reg_lambda': 1.4}

    # for plotting tree
    X_sm, y_sm = X[:90000], y[:90000]
    model = LGBMRegressor(**final_params)
    model.fit(X_sm, y_sm)

    
