#basic functions
# basics
from datetime import datetime
import time
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)




#this one is prob redundant
to_datetime = lambda timestamp: datetime.strftime(datetime.fromtimestamp(timestamp),"%Y-%m-%d")

get_month_timestamp = lambda timestamp: to_datetime(timestamp).month

get_month = lambda datetime: datetime.month

get_month_day = lambda datetime: datetime.strftime("%m-%d")

############# Two new features from the competition tutorial
def log_return(series, periods=1):
    return np.log(series).diff(periods=periods)

def roll(array, shift):
    # this supposebly improves the performance of np.roll
    if not (isinstance(array, np.ndarray)):
        array = np.asarray(array)
    idx = shift%len(array)
    return np.concatenate([array[-idx:], array[:-idx]])


############# from lightGBT tutorial
def upper_shadow(df):
    return df['High'] - np.maximum(df['Close'], df['Open'])

def lower_shadow(df):
    return np.minimum(df['Close'], df['Open']) - df['Low']

############# realized here needs to use some data for calculation of initial values
def realized(close, N=240):
    rt = list(np.log(C_t / C_t_1) for C_t, C_t_1 in zip(close[1:], close[:-1]))
    rt_mean = sum(rt) / len(rt)
    return np.sqrt(sum((r_i - rt_mean) ** 2 for r_i in rt) * N / (len(rt) - 1))

########## function for calling all feature creating functions
def get_features(df, Lag=True):
    df_feat = df[["Count", "Open","High", "Low", "Close", "Volume","VWAP", "Target"]].copy()
    
    df_feat["Upper_shadow"] = upper_shadow(df_feat)
    df_feat["Lower_shadow"] = lower_shadow(df_feat)

    df_feat["Volume"] = log_return(df_feat["Volume"]) # maybe Volume is fine just like that(?)
    df_feat["Count"] = log_return(df_feat["Count"])
    df_feat["VWAP"] = log_return(df_feat["VWAP"])
    df_feat = df_feat[1:] # compensate
    if Lag:
        for lag in range(1, 6):
            roll_feature = "VWAP"
            df_feat["rolled_"+roll_feature+f"_{lag}"] = \
                roll(df_feat[roll_feature].values, lag)
    
    ########### for now, simple dropna()     ########### 
    ########### later we can use reindexing  ########### 
    df_feat.dropna(inplace=True)
    # df_feat = df_feat.reindex(range(btc.index[0],btc.index[-1]+60, 60), method="pad")

    return df_feat


def Xy_model_asset(train, asset_id, Lag=True):
    df = train[train["Asset_ID"]==asset_id]

    # todo : try different features here 
    #        also, scale the features
    df_proc = get_features(df, Lag)

    df_proc["y"] = df["Target"]
    # 念の為
    df_proc.dropna(how="any", inplace=True)
    X = df_proc.drop("y", axis=1)
    y = df_proc["y"]

    # todo : try different models here
    # model = LGBMRegressor()
    # model.fit(X, y)
    return X, y#, model



def get_time_fractions(date):

    def s(date):# returns seconds since epoch
        return time.mktime(date.timetuple())

    year = date.year
    month = date.month
    day = date.day
    dayofweek = date.dayofweek

    start_of_day = datetime(year = year, month = month, day = day, hour = 0, minute = 0, second = 0)
    day_sec = 60*60*24

    frac_day = (s(date) - s(start_of_day))/day_sec # how much time has passed during that day

    frac_week  = (dayofweek + frac_day) / 7 # how much time has passed during the week

    start_of_month = datetime(year=year, month=month, day=1)
    start_of_next_month = datetime(year=year, month=month+1, day=1) if month < 12 \
        else datetime(year=year+1, month=1, day=1)
    # here we use datetime function because we don't know how many days this next month is 
    frac_month = (s(date) - s(start_of_month)) / (s(start_of_next_month) - s(start_of_month))

    start_of_year = datetime(year=year, month=1, day=1)
    start_of_next_year = datetime(year=year+1, month=1, day=1)

    frac_year = (s(date) - s(start_of_year)) / (s(start_of_next_year) - s(start_of_year))

    return frac_day, frac_week, frac_month, frac_year




