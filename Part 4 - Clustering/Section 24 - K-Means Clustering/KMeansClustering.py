import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
dataset=pd.read_csv('Mall_Customers.csv')
X=dataset.iloc[:,[3,4]].values


#using elbow method to find optimal no of clusters
from sklearn.cluster import KMeans
wcss=[]
#obtaining the wcss value upto 10 clusters
for i in range(1,11):
    kmeans=KMeans(n_clusters=i,init='k-means++', max_iter=300, n_init=10, random_state=0)
    kmeans.fit(X)
    wcss.append(kmeans.inertia_)
plt.plot(range(1,11),wcss)
plt.title('The elbow method')
plt.xlabel('no of clusters')
plt.ylabel('WCSS')
plt.show()


#Applying kmeans to the mall dataset
kmeans=KMeans(n_clusters=5,init='k-means++', max_iter=300, n_init=10, random_state=0)
#fit predict method tells the cluster to which our pt belongs to
y_kmeans=kmeans.fit_predict(X)


#visualising the clusters

plt.scatter(X[y_kmeans==0, 0],X[y_kmeans==0, 1],s=100,c='red', label='Cluster1')
plt.scatter(X[y_kmeans==1, 0],X[y_kmeans==1, 1],s=100,c='blue', label='Cluster2')
plt.scatter(X[y_kmeans==2, 0],X[y_kmeans==2, 1],s=100,c='green', label='Cluster3')
plt.scatter(X[y_kmeans==3, 0],X[y_kmeans==3, 1],s=100,c='grey', label='Cluster4')
plt.scatter(X[y_kmeans==4, 0],X[y_kmeans==4, 1],s=100,c='pink', label='Cluster5')

#plotting the centroid of all the clusters

plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1],s=300,c='yellow',label='centroid')
plt.xlabel('Annual Income')
plt.ylabel('Spending score')
plt.show()
