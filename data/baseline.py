# -*- coding: utf-8 -*-

"""
Generate previous day's data as result.
"""

import datetime
import numpy as np
import pandas as pd

user_pay = pd.read_csv('./testA_record_2019-01-28.csv')

user_pay['time'] = pd.to_datetime(user_pay['time'])
# user_pay['date'] = pd.to_datetime(user_pay['time']).dt.date
# user_pay['hour'] = pd.to_datetime(user_pay['time']).dt.hour
# user_pay['minute'] = pd.to_datetime(user_pay['time']).dt.minute
#
# user_pay_new = user_pay.groupby(by =['lineID','stationID','date','hour','minute','status'],as_index = False).count()

sta=pd.to_datetime('2019-01-28 00:00:00')
end=pd.to_datetime('2019-01-28 00:00:00')
df = pd.DataFrame()
for x in range(145):
    print(x)
    sta = end
    end=pd.to_datetime(end) + datetime.timedelta(minutes=10)
    temp = user_pay[sta <= user_pay['time']]
    temp = temp[temp['time'] <= end]
    temp = temp.groupby(by=['stationID','status'],as_index=False).count()
    temp = temp.rename(columns={'time':'num'})
    temp = temp[['stationID','status', 'num']]
    temp = pd.pivot_table(temp, values=['num'], index=['stationID'], columns=['status'],aggfunc=np.sum, fill_value=0)
    temp = temp.reset_index()
    try:
        temp.columns = ['stationID', 'outNums', 'inNums']
        temp.loc[:, 'startTime'] = str(sta + datetime.timedelta(days=1))
        temp.loc[:, 'endTime'] = str(end + datetime.timedelta(days=1))
    except:
        continue
    df = pd.concat([df,temp],axis=0)

df = df[['stationID','startTime','endTime','inNums','outNums']]
sub = pd.read_csv('./submit_sample.csv')
sub = pd.merge(sub,df, on=['stationID','startTime','endTime'],how='left')
sub = sub.fillna(0)
sub.to_csv('./submit.csv',index=False)
