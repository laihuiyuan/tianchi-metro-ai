# -*- coding: utf-8 -*-

import os
import time
import datetime
import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.model_selection import KFold, StratifiedKFold
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.filterwarnings('ignore')
from metro_tool import *
np.random.seed(1)

data = pd.read_csv('./feature/data_payType.csv')
weat = pd.read_csv('./feature/weather_features.csv')
road = pd.read_csv('./feature/roadMap_features.csv')
stat = pd.read_csv('./feature/station_features.csv')

data = data[data.day!=1]
data = data.merge(road,on=['stationID'],how='left')
#data = data.merge(stat,on=['stationID'],how='left')
data = data.merge(weat,on=['day'],how='left')

print(data.columns)
all_columns = [f for f in data.columns if f not in ['inNums','outNums','week_p', 'weekend_p','payType',
                                                    'inDID_h_max', 'inDID_h_min', 'inDID_h_mean',
                                                    'outDID_h_max', 'outDID_h_min', 'outDID_h_mean',
                                                    'inDID_hm_max', 'inDID_hm_min', 'inDID_hm_mean',
                                                    'outDID_hm_max', 'outDID_hm_min', 'outDID_hm_mean']]

all_train = data[data.day!=27]

train = data[data.day <26]
valid = data[data.day==26]
test  = data[data.day==27]

params = {
    'boosting_type': 'gbdt',
    'objective': 'regression',
    'metric': 'mae',
    'num_leaves': 65,
    'learning_rate': 0.01,
    'feature_fraction': 0.9,
    'bagging_fraction': 0.9,
    'bagging_seed':0,
    'bagging_freq': 1,
    'verbose': 1,
    'reg_alpha':1,
    'reg_lambda':2
}

######################################################inNums
valid_new=pd.DataFrame()
test_new=pd.DataFrame()

for type in range(4):
    train_data = train[train.payType==type]
    valid_data = valid[valid.payType==type]
    test_data  = test[test.payType==type]

    train_x = train_data[all_columns].values
    valid_x = valid_data[all_columns].values
    test_x  = test_data[all_columns].values

    train_y = train_data['inNums']
    valid_y = valid_data['inNums']

    all_train_data = all_train[all_train.payType==type]
    all_x = all_train_data[all_columns].values
    all_y = all_train_data['inNums']

    lgb_train = lgb.Dataset(train_x,train_y)
    lgb_evals = lgb.Dataset(valid_x,valid_y, reference=lgb_train)
    gbm = lgb.train(params,
                    lgb_train,
                    num_boost_round=10000,
                    valid_sets=[lgb_train, lgb_evals],
                    valid_names=['train', 'valid'],
                    early_stopping_rounds=200,
                    verbose_eval=1000,
                    )
    valid_data['inNums_pred']= gbm.predict(valid_x)

    lgb_train = lgb.Dataset(all_x, all_y)
    gbm = lgb.train(params,
                    lgb_train,
                    num_boost_round=gbm.best_iteration,
                    valid_sets=[lgb_train],
                    valid_names=['train'],
                    verbose_eval=1000,
                    )
    test_data['inNums_pred'] = gbm.predict(test_x)

    ######################################################outNums

    train_x = train_data[all_columns].values
    valid_x = valid_data[all_columns].values
    train_y = train_data['outNums']
    valid_y = valid_data['outNums']
    all_y = all_train_data['outNums']
    lgb_train = lgb.Dataset(train_x, train_y)
    lgb_evals = lgb.Dataset(valid_x, valid_y, reference=lgb_train)
    gbm = lgb.train(params,
                    lgb_train,
                    num_boost_round=10000,
                    valid_sets=[lgb_train, lgb_evals],
                    valid_names=['train', 'valid'],
                    early_stopping_rounds=200,
                    verbose_eval=1000,
                    )
    valid_data['outNums_pred']= gbm.predict(valid_x)

    lgb_train = lgb.Dataset(all_x, all_y)
    gbm = lgb.train(params,
                    lgb_train,
                    num_boost_round=gbm.best_iteration,
                    valid_sets=[lgb_train],
                    valid_names=['train'],
                    verbose_eval=1000,
                    )
    test_data['outNums_pred'] = gbm.predict(test_x)

    valid_new = pd.concat([valid_new,valid_data])
    test_new  = pd.concat([test_new,test_data])


valid = valid.merge(valid_new,on=['stationID', 'day', 'hour', 'minute','payType'],how='left')
test  = test.merge(test_new,on=['stationID', 'day', 'hour', 'minute','payType'],how='left')

valid = valid.groupby(['stationID',  'day', 'hour', 'minute'])['inNums_pred','outNums_pred'].sum().reset_index()
test = test.groupby(['stationID',  'day', 'hour', 'minute'])['inNums_pred','outNums_pred'].sum().reset_index()

df = pd.read_csv('./feature/data_B.csv')
df = df[df.day==26]
MAE(df['inNums'].values,valid['inNums_pred'].values)
MAE(df['outNums'].values,valid['outNums_pred'].values)

test = test.sort_values(['stationID', 'day', 'hour', 'minute'],ascending = True,inplace = False)
sub = pd.read_csv('./data/testB_submit_2019-01-27.csv')
sub['inNums']   = test['inNums_pred'].values
sub['outNums']  = test['outNums_pred'].values
# 结果修正
sub.loc[sub.inNums<0 , 'inNums']  = 0
sub.loc[sub.outNums<0, 'outNums'] = 0
sub[['stationID', 'startTime', 'endTime', 'inNums', 'outNums']].to_csv('submit.csv', index=False)