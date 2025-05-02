import numpy as np 
import matplotlib  
import matplotlib.pyplot as plt 
from scipy.cluster.hierarchy import dendrogram, linkage  

# create a dataset of 1-dimensional points
Y = np.array([[1], [2], [4], [7], [8], [10], [15], [17], [21]])

# perform hierarchical clustering using the single linkage method
linked = linkage(Y, 'single')

# create a list of labels corresponding to the data points
labelList = [[1], [2], [4], [7], [8], [10], [15], [17], [21]]

# set the figure size for the dendrogram
plt.figure(figsize=(10, 7))

# create the dendrogram with custom settings
dendrogram(
    linked,
    orientation='top',  # display the tree vertically
    labels=labelList,  # use the label list for the data points
    distance_sort='descending',  # sort clusters by distance in descending order
    show_leaf_counts=True  # show the number of points in each cluster
)

# display the dendrogram plot
plt.show()