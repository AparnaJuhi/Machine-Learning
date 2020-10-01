# -*- coding: utf-8 -*-
"""
Created on Fri Apr  3 17:24:32 2020

@author: aparn
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
dataset=pd.read_csv('Mall_Customers.csv')
X=dataset.iloc[:,3:5].values

#finding optimal no of clusters using dendograms
import scipy.cluster.hierarchy as sch
dendrogram=sch.dendrogram(sch.linkage(X, method='ward'))
plt.xlabel('customers')
plt.ylabel('euclidean distance')
plt.title('dendograms')
plt.show()

#Fitting heirarchial clustering to the mall dataset
from sklearn.cluster import AgglomerativeClustering
hc=AgglomerativeClustering(n_clusters=5,affinity='euclidean', linkage='ward')
y_hc=hc.fit_predict(X)

#visualisng the clusters

plt.scatter(X[y_hc==0, 0],X[y_hc==0, 1],s=100,c='red', label='Cluster1')
plt.scatter(X[y_hc==1, 0],X[y_hc==1, 1],s=100,c='blue', label='Cluster2')
plt.scatter(X[y_hc==2, 0],X[y_hc==2, 1],s=100,c='green', label='Cluster3')
plt.scatter(X[y_hc==3, 0],X[y_hc==3, 1],s=100,c='grey', label='Cluster4')
plt.scatter(X[y_hc==4, 0],X[y_hc==4, 1],s=100,c='pink', label='Cluster5')


plt.legend()
plt.show()