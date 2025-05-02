import numpy as np
from sklearn.metrics import pairwise
from sklearn.datasets import load_iris
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split

# Part 1
def KNN(X, Y):
    distances = pairwise.euclidean_distances(X)
    np.fill_diagonal(distances, np.inf)
    nearest_indices = np.argmin(distances, axis=1)
    predictions = Y[nearest_indices]
    return predictions

# Part 2
def KNN_with_error(X, Y):
    predictions = KNN(X, Y)
    error = np.mean(predictions != Y)
    return predictions, error

# Part 3
iris = load_iris()
X, Y = iris.data, iris.target

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.4, random_state=80)

test_predictions, test_error = KNN_with_error(X_test, Y_test)
print("Test prediction error:", test_error)

# Part 4
knn_sklearn = KNeighborsClassifier(n_neighbors=1)
knn_sklearn.fit(X, Y)
print("Sklearn prediction error:", np.mean(knn_sklearn.predict(X) != Y))

# Part 5
def KNN_variable_K(X, Y, K):
    distances = pairwise.euclidean_distances(X)
    np.fill_diagonal(distances, np.inf)
    nearest_indices = np.argsort(distances, axis=1)[:, :K]
    predictions = np.array([np.bincount(Y[indices]).argmax() for indices in nearest_indices])
    error = np.mean(predictions != Y)
    return predictions, error

K = int(input("Enter the value of K: "))

predictions, error = KNN_variable_K(X, Y, K=K)
print(f"Prediction error for K={K}: {error}")
