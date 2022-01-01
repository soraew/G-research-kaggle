# basics
from datetime import datetime
import time
import lightgbm
from tqdm import tqdm



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

# random search parameters
# the parameter co are a copy of 渡辺卒論
def random_search_lgbm(X, y, params_space):
    params_log = {}
    iteration = 10
    cv = TimeSeriesSplit()
    for i in tqdm(range(iteration)):
        params = {}
        for key in params_space.keys():
            param_list = params_space[key]
            length = len(param_list)
            idx =np.random.randint(0,length) 
            params.update({key:param_list[idx]})
            # fit model to data
        
        model = LGBMRegressor(**params)
        for train_idx, test_idx, in cv.split(X):
            X_train, X_test = X[train_idx], X[test_idx]
            y_train, y_test = y[train_idx], y[test_idx]
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test).squeeze()
            coef_score = np.corrcoef(y_pred, y_test)[0][1]
            rmse_score = np.sqrt(mean_squared_error(y_pred, y_test))
        params_log.update({i:[coef_score, rmse_score, params]})

    sorted_by_coef=sorted(params_log.items(), key = lambda item: item[1][0], reverse=True)
    sorted_by_rmse=sorted(params_log.items(), key = lambda item: item[1][1])
    
    return sorted_by_coef, sorted_by_rmse


#rmse got lower than baseline(model=LGBMRegressor), however coef got worse

# best for coef(first 900000 data points(46%))
# (rmse doesn't really change)
"""{'max_depth': 4,
    'min_data_in_leaf': 20,
    'learning_rate': 0.01,
    'num_leaves': 40,
    'boosting_type': 'gbdt',
    'objective': 'regression',
    'random_state': 2021,
    'reg_alpha': 1.2,
    'reg_lambda': 1.4}"""
