# -*- coding: utf-8 -*-

import datetime
import pandas as pd

sta=pd.to_datetime('2019-01-01')

data = []
for i in range(30):
    dow = sta.dayofweek
    if dow == 5 or dow ==6:
        data.append([i,1])
    else:
        data.append([i,0])
    sta+=datetime.timedelta(days=1)
df = pd.DataFrame(data,columns=['day','holi'])
df.to_csv('holi.csv',index=False)