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
from get_feats import get_features


data_root = "data/"
df_features_targets = get_features(data_root)



