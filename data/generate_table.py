# -*- coding: utf-8 -*-

import os
import datetime
import warnings
import pandas as pd
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.filterwarnings('ignore')

from metro_tool import *
np.random.seed(1024)

path = '../data/'

def base_time(df,time='time'):

    # base time
    df['day'] = df[time].apply(lambda x: int(x[8:10]))
    df['week'] = pd.to_datetime(df[time]).dt.dayofweek + 1
    df['weekend'] = (pd.to_datetime(df[time]).dt.weekday >= 5).astype(int)
    df['hour'] = df[time].apply(lambda x: int(x[11:13]))
    df['minute'] = df[time].apply(lambda x: int(x[14:15] + '0'))

    return df

def base_features(df,sample_,flag=True):

    sample = sample_.copy()
    day = pd.to_datetime(sample.loc[0,'startTime']).day - pd.to_datetime(df.loc[0,'time']).day
    sample['startTime'] = pd.to_datetime(sample['startTime']).apply(lambda x: str(x - datetime.timedelta(days=day)))
    sample = base_time(sample,'startTime')
    sample = sample.drop(['startTime', 'endTime','inNums', 'outNums'], axis=1)

    # count,sum
    result = df.groupby(['stationID', 'week', 'weekend', 'day', 'hour', 'minute']).status.agg(
        ['count', 'sum']).reset_index()
    if flag:
        result = sample.merge(result, on=['stationID','week','weekend','day','hour','minute'],how='left')

    # nunique
    tmp = df.groupby(['stationID'])['deviceID'].nunique().reset_index(name='dID_of_sID')
    result = result.merge(tmp, on=['stationID'], how='left')

    tmp = df.groupby(['stationID', 'hour'])['deviceID'].nunique().reset_index(name='dID_of_sID_h')
    result = result.merge(tmp, on=['stationID', 'hour'], how='left')
    tmp = df.groupby(['stationID', 'hour', 'minute'])['deviceID'].nunique().reset_index(name='dID_of_sID_hm')
    result = result.merge(tmp, on=['stationID', 'hour', 'minute'], how='left')

    # tmp = df.groupby(['stationID', 'hour'])['payType'].nunique().reset_index(name='Pay_of_sID_h')
    # result = result.merge(tmp, on=['stationID', 'hour'], how='left')
    # tmp = df.groupby(['stationID', 'hour', 'minute'])['payType'].nunique().reset_index(name='pay_of_sID_hm')
    # result = result.merge(tmp, on=['stationID', 'hour', 'minute'], how='left')

    # in,out
    result['inNums'] = result['sum']
    result['outNums'] = result['count'] - result['sum']

    #
    result.fillna(0, inplace=True)
    del result['sum'], result['count']

    return result

def generate_features(df_):

    df = df_.copy
    # base time
    df['time'] = pd.to_datetime(sample['startTime']).apply(lambda x: str(x - datetime.timedelta(days=day)))
    df['day'] = df['time'].apply(lambda x: int(x[8:10]))
    df['week'] = pd.to_datetime(df['time']).dt.dayofweek + 1
    df['weekend'] = (pd.to_datetime(df.time).dt.weekday >= 5).astype(int)
    df['hour'] = df['time'].apply(lambda x: int(x[11:13]))
    df['minute'] = df['time'].apply(lambda x: int(x[14:15] + '0'))

    # count,sum
    result = df.groupby(['stationID', 'week', 'weekend', 'day', 'hour', 'minute']).status.agg(
        ['count', 'sum']).reset_index()

    # nunique
    tmp = df.groupby(['stationID'])['deviceID'].nunique().reset_index(name='nuni_deviceID_of_stationID')
    result = result.merge(tmp, on=['stationID'], how='left')
    tmp = df.groupby(['stationID', 'hour'])['deviceID'].nunique().reset_index(name='nuni_deviceID_of_stationID_hour')
    result = result.merge(tmp, on=['stationID', 'hour'], how='left')
    tmp = df.groupby(['stationID', 'hour', 'minute'])['deviceID'].nunique(). \
        reset_index(name='nuni_deviceID_of_stationID_hour_minute')
    result = result.merge(tmp, on=['stationID', 'hour', 'minute'], how='left')

    # in,out
    result['inNums'] = result['sum']
    result['outNums'] = result['count'] - result['sum']

    #
    result['day_since_first'] = result['day'] - 1
    result.fillna(0, inplace=True)
    del result['sum'], result['count']

    return result


df_al = pd.DataFrame()
df_28 = pd.read_csv(path+'testA_record_2019-01-28.csv')
df_te = pd.read_csv(path+'testA_submit_2019-01-29.csv')

df_28 = base_time(df_28)
data = base_features(df_28,df_te,False)
df_al = pd.concat([df_al,df_28],axis = 0)

data_list = os.listdir(path+'/Metro_train/')
for i in range(0, len(data_list)):
    t0 = time.time()
    if data_list[i].split('.')[-1] == 'csv':
        df = pd.read_csv(path+'/Metro_train/' + data_list[i])
        df = base_time(df)
        df_al = pd.concat([df_al, df], axis=0)
        df = base_features(df,df_te)
        data = pd.concat([data, df], axis=0, ignore_index=True)
    else:
        continue
    t1 = time.time()
    print(data_list[i], i, t1-t0)

df_te = base_time(df_te,'startTime')
test = df_te.drop(['startTime','endTime'], axis=1)
data = pd.concat([data,test], axis=0, ignore_index=True)

# df_fe = generate_features(df_al)
# data = data.merge(df_fe, on=['stationID', 'weekend', 'hour', 'minute'], how='left')

data.to_csv('data_A.csv',index=False)

#1678760