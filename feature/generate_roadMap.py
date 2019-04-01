# coding: utf-8

import numpy as np
import pandas as pd
from sklearn.decomposition import TruncatedSVD,NMF

df = pd.read_csv('../data/Metro_roadMap.csv')
data = df.values[:,1:]
svd = TruncatedSVD(n_iter=10, random_state=1123, n_components=2)
nmf = NMF(random_state=1123,n_components=2)
df = pd.DataFrame(svd.fit_transform(data))
nmf_data = pd.DataFrame(nmf.fit_transform(data))
df = pd.concat([df, nmf_data],axis=1)
df['stationID'] = [i for i in range(81)]
data = [[idx for idx,i in enumerate(s) if i==1] for s in data]
data = pd.DataFrame(data)
df = pd.concat([df,data],axis=1)
# df['Nums'] = [x.sum() for x in data]
df.columns = ['mapSVD1','mapSVD2','mapNMF2','mapNMF2','stationID']+['map'+str(i).zfill(2) for i in range(data.shape[1])]
print(df.shape)
df.to_csv('roadMap_features.csv',index=False)
