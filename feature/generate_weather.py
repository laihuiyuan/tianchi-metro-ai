# -*- coding: utf-8 -*-

import math

import numpy as np
import pandas as pd
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder

from metro_tool import *


def one_hot(data):
    # integer encode
    label_encoder = LabelEncoder()
    integer_encoded = label_encoder.fit_transform(data)

    # binary encode
    onehot_encoder = OneHotEncoder(sparse=False)
    integer_encoded = integer_encoded.reshape(len(integer_encoded), 1)
    onehot_encoded = onehot_encoder.fit_transform(integer_encoded)

    return onehot_encoded


def SSD(Temp, Velo=2, Humi=0.6):
    score = (1.818 * Temp + 18.18) * (0.88 + 0.002 * Humi) + 1.0 * (Temp - 32) / (45 - Temp) - 3.2 * Velo + 18.2
    return score

WEATHER = pd.read_csv('../data/weather_hangzhou.csv', )
WEATHER.columns = ['day','weat','temp','wind']

oh_weat = one_hot(WEATHER['weat'].values)
oh_wind = one_hot(WEATHER['wind'].values)
one_hot = np.concatenate([oh_weat,oh_wind],axis=1)

svd = TruncatedSVD(n_iter=10, random_state=1123, n_components=2)
data = svd.fit_transform(one_hot)
# wind = svd.fit_transform(oh_wind)
# data = np.concatenate([weat,wind],axis=1)

# temp = np.array([x.split('/') for x in WEATHER['temp']])
df = pd.DataFrame(data)
# data = [[idx for idx,i in enumerate(s) if i==1] for s in one_hot]
# data = pd.DataFrame(data)
# df = pd.concat([df,data],axis=1)
df['day'] = pd.DataFrame([i for i in range(1,32)])
df.columns = ['weatsvd1','weatsvd2','day']
df.to_csv('weather_features.csv', index=False)
