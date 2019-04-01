# -*- coding: utf-8 -*-

import os
import time
import datetime
import numpy as np
import pandas as pd
import xgboost as xgb
import lightgbm as lgb
from sklearn.model_selection import KFold, StratifiedKFold
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.filterwarnings('ignore')
from sklearn import metrics

data = pd.read_csv('./feature/data_B1.csv')
data['ratio'] = data['inNums_p']/data['outNums_p']
weat = pd.read_csv('./feature/weather_features.csv')
road = pd.read_csv('./feature/roadMap_features.csv')

data = data[data.day!=1]
data = data.merge(road,on=['stationID'],how='left')
data = data.merge(weat,on=['day'],how='left')

print(data.columns)
all_columns = [f for f in data.columns if f not in ['inNums','outNums','week_p', 'weekend_p','inNums_log','outNums_log'
                                                    'inDID_h_max', 'outDID_h_max',
                                                    'inDID_h_min', 'outDID_h_min',
                                                    'inDID_h_mean', 'outDID_h_mean',
                                                    'inDID_hm_max', 'inDID_hm_min', 'inDID_hm_mean',
                                                    'outDID_hm_max', 'outDID_hm_min', 'outDID_hm_mean',
                                                    'in_wkhm_max', 'in_wkhm_min', 'in_wkhm_mean',
                                                    'out_wkhm_max', 'out_wkhm_min', 'out_wkhm_mean',
                                                    'in_wkh_max', 'in_wkh_min', 'in_wkh_mean',
                                                    'out_wkh_max', 'out_wkh_min', 'out_wkh_mean']]

all_data = data[data.day!=27]
X_data = all_data[all_columns].values

train = data[data.day <26]
X_train = train[all_columns].values

valid = data[data.day==26]
X_valid = valid[all_columns].values

test  = data[data.day==27]
X_test = test[all_columns].values
print(X_train.shape,X_valid.shape, X_test.shape)

params = {
    'boosting_type': 'gbdt',
    'objective': 'reg:linear',
    'metric': 'mae',
    'max_depth' : 6,
    'learning_rate': 0.01,
    'n_estimators':500,
    'silent':1,
}

######################################################inNums
y_train = train['inNums']
y_valid = valid['inNums']
y_data  = all_data['inNums']
dtrain = xgb.DMatrix(X_train, y_train)
devals = xgb.DMatrix(X_valid, y_valid)
model = xgb.train(params, dtrain,
                  num_boost_round=5000,
                  evals=[(dtrain, 'Train'), (devals, 'valid')],
                  early_stopping_rounds=100,
                  verbose_eval=50,)
# print(pd.DataFrame({
#         'column': all_columns,
#         'importance': gbm.feature_importance(),
#     }).sort_values(by='importance'))

# all_data
# lgb_train = lgb.Dataset(X_data, y_data)
# gbm = lgb.train(params,
#                 lgb_train,
#                 num_boost_round=gbm.best_iteration,
#                 valid_sets=[lgb_train],
#                 valid_names=['train'],
#                 verbose_eval=1000,
#                 )
# test['inNums'] = gbm.predict(X_test)

######################################################outNums
y_train = train['outNums']
y_valid = valid['outNums']
y_data  = all_data['outNums']
lgb_train = lgb.Dataset(X_train, y_train)
lgb_evals = lgb.Dataset(X_valid, y_valid , reference=lgb_train)
gbm = lgb.train(params,
                lgb_train,
                num_boost_round=10000,
                valid_sets=[lgb_train,lgb_evals],
                valid_names=['train','valid'],
                early_stopping_rounds=200,
                verbose_eval=1000,
                )
# print(pd.DataFrame({
#         'column': all_columns,
#         'importance': gbm.feature_importance(),
#     }).sort_values(by='importance'))
# all_data
# lgb_train = lgb.Dataset(X_data, y_data)
# gbm = lgb.train(params,
#                 lgb_train,
#                 num_boost_round=gbm.best_iteration,
#                 valid_sets=[lgb_train],
#                 valid_names=['train'],
#                 verbose_eval=1000,
#                 )
# test['outNums'] = gbm.predict(X_test)

sub = pd.read_csv('./data/testB_submit_2019-01-27.csv')
sub['inNums']   = test['inNums'].values
sub['outNums']  = test['outNums'].values
# 结果修正
sub.loc[sub.inNums<0 , 'inNums']  = 0
sub.loc[sub.outNums<0, 'outNums'] = 0
sub[['stationID', 'startTime', 'endTime', 'inNums', 'outNums']].to_csv('submit.csv', index=False)