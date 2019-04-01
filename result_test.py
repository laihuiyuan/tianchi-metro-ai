#-*- coding: utf-8 -*-

import datetime
import numpy as np
import pandas as pd

from metro_tool import *


a=pd.read_csv('submit.csv')
b=pd.read_csv('./data_new/train25.csv')

mae1=MAE(a.values[:,-2],b.values[:,-4])
mae2=MAE(a.values[:,-1],b.values[:,-3])

print((mae1+mae2)/2)

# MAE: 14.765432098765432
# MAE: 16.017746913580247
# 15.391589506172838