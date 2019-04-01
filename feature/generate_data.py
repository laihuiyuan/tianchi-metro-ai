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
    # df.loc[df.day == 1, 'weekend'] = 1

    return df

def base_features(df_,sample_=None,flag=True):

    df = df_.copy()
    # sample = sample_.copy()
    # day = pd.to_datetime(sample.loc[0, 'startTime']).day - pd.to_datetime(df.loc[0, 'time']).day
    # sample['startTime'] = pd.to_datetime(sample['startTime']).apply(lambda x: str(x - datetime.timedelta(days=day)))
    # sample = base_time(sample, 'startTime')
    # sample = sample.drop(['startTime', 'endTime', 'inNums', 'outNums'], axis=1)

    # count,sum
    result = df.groupby(['stationID', 'week', 'weekend', 'day', 'hour', 'minute']).status.agg(
        ['count', 'sum']).reset_index()
    # if flag:
    #     result = sample.merge(result, on=['stationID', 'week', 'weekend', 'day', 'hour', 'minute'], how='left')

    # nunique
    tmp = df.groupby(['stationID'])['deviceID'].nunique().reset_index(name='dID_of_sID')
    result = result.merge(tmp, on=['stationID'], how='left')
    tmp = df.groupby(['stationID', 'hour'])['deviceID'].nunique().reset_index(name='dID_of_sID_h')
    result = result.merge(tmp, on=['stationID', 'hour'], how='left')
    tmp = df.groupby(['stationID', 'hour', 'minute'])['deviceID'].nunique().reset_index(name='dID_of_sID_hm')
    result = result.merge(tmp, on=['stationID', 'hour', 'minute'], how='left')

    # in,out
    result['inNums'] = result['sum']
    result['outNums'] = result['count'] - result['sum']
    result['inNums_log'] = 1+ np.log(result['inNums'])/ np.log(2)
    result['outNums_log'] = 1+ np.log(result['outNums'])/ np.log(2)

    result['day_since_first'] = result['day'] - 1
    result.fillna(0, inplace=True)
    del result['sum'], result['count']

    temp = df.groupby(['stationID', 'hour', 'deviceID']).status.agg(
        ['count', 'sum']).reset_index()
    temp['inNums_dID'] = temp['sum']
    temp['outNums_dID'] = temp['count'] - temp['sum']

    count = temp.groupby(['stationID', 'hour'], as_index=False)['inNums_dID'].agg({
        'inDID_h_max': 'max', 'inDID_h_min': 'min', 'inDID_h_mean': 'mean'})
    result = result.merge(count, on=['stationID', 'hour'], how='left')

    count = temp.groupby(['stationID', 'hour'], as_index=False)['outNums_dID'].agg({
        'outDID_h_max': 'max', 'outDID_h_min': 'min', 'outDID_h_mean': 'mean'})
    result = result.merge(count, on=['stationID', 'hour'], how='left')

    # temp = df.groupby(['stationID', 'week', 'weekend', 'day', 'hour', 'minute','payType']).status.agg(
    #     ['count', 'sum']).reset_index()
    # temp['inNums_pay'] = temp['sum']
    # temp['outNums_pay'] = temp['count'] - temp['sum']
    # temp = pd.concat([temp[temp.payType==1],temp[temp.payType==2]])
    # del temp['count'], temp['sum'],temp['payType']
    # result = result.merge(temp, on=['stationID', 'week', 'weekend', 'day', 'hour', 'minute'], how='left')

    return result

def  generate_feature(data,df_):

    temp = df_[df_.day!=27][['stationID', 'week', 'weekend', 'day', 'hour', 'minute', 'inNums', 'outNums']]

    # add hour
    count = temp.groupby(['stationID', 'hour'], as_index=False)['inNums'].agg({
        'in_h_max': 'max', 'in_h_min': 'min', 'in_h_mean': 'mean'})
    data = data.merge(count, on=['stationID', 'hour'], how='left')

    count = temp.groupby(['stationID', 'hour'], as_index=False)['outNums'].agg({
        'out_h_max': 'max', 'out_h_min': 'min', 'out_h_mean': 'mean'})
    data = data.merge(count, on=['stationID', 'hour'], how='left')

    count = temp.groupby(['stationID', 'hour', 'minute'], as_index=False)['inNums'].agg({
        'in_hm_max': 'max', 'in_hm_min': 'min', 'in_hm_mean': 'mean'})
    data = data.merge(count, on=['stationID', 'hour', 'minute'], how='left')

    count = temp.groupby(['stationID', 'hour', 'minute'], as_index=False)['outNums'].agg({
        'out_hm_max': 'max', 'out_hm_min': 'min', 'out_hm_mean': 'mean'})
    data = data.merge(count, on=['stationID', 'hour', 'minute'], how='left')

    # week
    count = temp.groupby(['stationID', 'week', 'hour'], as_index=False)['inNums'].agg({
        'in_wh_max': 'max', 'in_wh_min': 'min', 'in_wh_mean': 'mean'})
    data = data.merge(count, on=['stationID', 'week', 'hour'], how='left')

    count = temp.groupby(['stationID', 'week', 'hour'], as_index=False)['outNums'].agg({
        'out_wh_max': 'max', 'out_wh_min': 'min', 'out_wh_mean': 'mean'})
    data = data.merge(count, on=['stationID', 'week', 'hour'], how='left')

    count = temp.groupby(['stationID', 'week', 'hour', 'minute'], as_index=False)['inNums'].agg({
        'in_whm_max': 'max', 'in_whm_min': 'min', 'in_whm_mean': 'mean'})
    data = data.merge(count, on=['stationID', 'week', 'hour', 'minute'], how='left')

    count = temp.groupby(['stationID', 'week', 'hour', 'minute'], as_index=False)['outNums'].agg({
        'out_whm_max': 'max', 'out_whm_min': 'min', 'out_whm_mean': 'mean'})
    data = data.merge(count, on=['stationID', 'week', 'hour', 'minute'], how='left')

    # # weekend
    # count = temp.groupby(['stationID', 'weekend', 'hour', 'minute'], as_index=False)['inNums'].agg({
    #     'in_wkhm_max': 'max', 'in_wkhm_min': 'min', 'in_wkhm_mean': 'mean'})
    # data = data.merge(count, on=['stationID', 'weekend', 'hour', 'minute'], how='left')
    #
    # count = temp.groupby(['stationID', 'weekend', 'hour', 'minute'], as_index=False)['outNums'].agg({
    #     'out_wkhm_max': 'max', 'out_wkhm_min': 'min', 'out_wkhm_mean': 'mean'})
    # data = data.merge(count, on=['stationID', 'weekend', 'hour', 'minute'], how='left')
    #
    # count = temp.groupby(['stationID', 'weekend', 'hour'], as_index=False)['inNums'].agg({
    #     'in_wkh_max': 'max', 'in_wkh_min': 'min', 'in_wkh_mean': 'mean'})
    # data = data.merge(count, on=['stationID', 'weekend', 'hour'], how='left')
    #
    # count = temp.groupby(['stationID', 'weekend', 'hour'], as_index=False)['outNums'].agg({
    #     'out_wkh_max': 'max', 'out_wkh_min': 'min', 'out_wkh_mean': 'mean'})
    # data = data.merge(count, on=['stationID', 'weekend', 'hour'], how='left')

    print(data.shape)
    return data


df_va = pd.read_csv(path+'testB_record_2019-01-26.csv')
df_te = pd.read_csv(path+'testB_submit_2019-01-27.csv')

data = pd.DataFrame()
all_data = pd.DataFrame()

data_list = os.listdir(path+'/Metro_train/')
for i in range(0, len(data_list)):
    t0 = time.time()
    if data_list[i].split('.')[-1] == 'csv':
        df = pd.read_csv(path+'/Metro_train/' + data_list[i])
        df = base_time(df)
        df = base_features(df)
        all_data = pd.concat([all_data, df], axis=0)
        df = generate_feature(df, all_data)
        data = pd.concat([data,df],axis=0)
        print(data.shape)
    else:
        continue
    t1 = time.time()
    print(data_list[i], i, t1-t0)

df_va = base_features(base_time(df_va))
all_data = pd.concat([all_data, df_va], axis=0)
df_va = generate_feature(df_va,all_data)
data = pd.concat([data, df_va], axis=0)

df_te = base_time(df_te,'startTime')
df_te = df_te.drop(['startTime','endTime'], axis=1)
df_te['day_since_first'] = df_te['day'] - 1
data = pd.concat([data,df_te], axis=0, ignore_index=True)

tmp = data.copy()
tmp_df = tmp[tmp.day == 1]
tmp_df['day'] = tmp_df['day'] - 1
tmp = pd.concat([tmp, tmp_df], axis=0, ignore_index=True)
tmp['day'] = tmp['day'].apply(lambda x: x+1)

for f in ['week','weekend','inNums','outNums']:
    tmp.rename(columns={f: f + '_p'}, inplace=True)

# tmp = tmp[['stationID', 'day', 'hour', 'minute', 'inNums_p', 'outNums_p']]
# data = data.merge(tmp, on=['stationID', 'day', 'hour', 'minute'], how='left')
data = data[['stationID','week', 'weekend', 'day', 'hour', 'minute', 'inNums', 'outNums']]

data = data.merge(tmp, on=['stationID','day', 'hour', 'minute'], how='left')
data.fillna(0, inplace=True)

# data = data[data.day!=1]
data.to_csv('data_B1.csv',index=False)

#1678760