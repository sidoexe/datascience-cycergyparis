import numpy as np  
from sklearn.cluster import KMeans  
from sklearn.cluster import AgglomerativeClustering  

# create a dataset of 1-dimensional points
Z = np.array([[11], [21], [22], [23], [26], [27], [28], [33], [35], [37], [47]])

# initialize the k-means model with 3 clusters and custom initial centers
kmeans = KMeans(n_clusters=3, init=np.array([[1], [2], [3]]), n_init=1)

# train the k-means model on the data
kmeans.fit(Z)

# get the coordinates of the cluster centers
print("kmeans cluster centers:", kmeans.cluster_centers_)

# get the cluster labels assigned to each data point
print("kmeans labels:", kmeans.labels_)

# initialize the agglomerative clustering model with 3 clusters using single linkage
cluster = AgglomerativeClustering(n_clusters=3, linkage='single')

# train the agglomerative clustering model on the data
cluster.fit(Z)

# get the cluster labels assigned to each data point
print("agglomerative labels:", cluster.labels_)