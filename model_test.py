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

data = pd.read_csv('./feature/data_B1.csv')
weat = pd.read_csv('./feature/weather_features.csv')
road = pd.read_csv('./feature/roadMap_features.csv')

data = data[data.day!=1]
data = data.merge(road,on=['stationID'],how='left')
data = data.merge(weat,on=['day'],how='left')

print(data.columns)
all_columns = [f for f in data.columns if f not in ['inNums','outNums','week_p', 'weekend_p',
                                                    'inDID_h_max', 'inDID_h_min', 'inDID_h_mean',
                                                    'outDID_h_max', 'outDID_h_min', 'outDID_h_mean',
                                                    'inDID_hm_max', 'inDID_hm_min', 'inDID_hm_mean',
                                                    'outDID_hm_max', 'outDID_hm_min', 'outDID_hm_mean']]

all_data = data[data.day!=27]
X_data = all_data[all_columns].values

train = data[data.day <26]
X_train = train[all_columns].values

test = data[data.day==26]
test_x = test[all_columns].values

# test  = data[data.day==27]
# X_test = test[all_columns].values
# print(X_train.shape,X_valid.shape, X_test.shape)

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
Y_train = train['inNums']
y_test = test['inNums'].values
kf = KFold(n_splits=5,shuffle=True)
test_preds = np.zeros(y_test.shape[0])
for tr_ind , te_ind in kf.split(Y_train):
    train_x = X_train[tr_ind]
    train_y = Y_train[tr_ind]

    valid_x = X_train[te_ind]
    valid_y = Y_train[te_ind]

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
    preds = gbm.predict(test_x,gbm.best_iteration)
    MAE(y_test, preds)
    test_preds += preds

MAE(y_test,test_preds/5)

######################################################outNums
Y_train = train['outNums']
y_test = test['outNums'].values
kf = KFold(n_splits=5,shuffle=True)
test_preds = np.zeros(y_test.shape[0])
for tr_ind , te_ind in kf.split(Y_train):
    train_x = X_train[tr_ind]
    train_y = Y_train[tr_ind]

    valid_x = X_train[te_ind]
    valid_y = Y_train[te_ind]

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

    preds = gbm.predict(test_x,gbm.best_iteration)
    MAE(y_test, preds)
    test_preds += preds

MAE(y_test,test_preds/5)