from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import confusion_matrix, accuracy_score

# Step 1: Load the Iris dataset and split into learning and prediction subsets
iris = load_iris()
X, Y = iris.data, iris.target

# Split the dataset into 70% training and 30% testing
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.3, random_state=42)

# Step 2: Apply the algorithms

# Initialize the models
svm_model = SVC()
naive_bayes_model = GaussianNB()
decision_tree_model = DecisionTreeClassifier()

# Train the models
svm_model.fit(X_train, Y_train)
naive_bayes_model.fit(X_train, Y_train)
decision_tree_model.fit(X_train, Y_train)

# Predictions
svm_predictions = svm_model.predict(X_test)
naive_bayes_predictions = naive_bayes_model.predict(X_test)
decision_tree_predictions = decision_tree_model.predict(X_test)

# Compute prediction error (accuracy)
svm_error = 1 - accuracy_score(Y_test, svm_predictions)
naive_bayes_error = 1 - accuracy_score(Y_test, naive_bayes_predictions)
decision_tree_error = 1 - accuracy_score(Y_test, decision_tree_predictions)

# Confusion matrices
svm_confusion_matrix = confusion_matrix(Y_test, svm_predictions)
naive_bayes_confusion_matrix = confusion_matrix(Y_test, naive_bayes_predictions)
decision_tree_confusion_matrix = confusion_matrix(Y_test, decision_tree_predictions)

# Print the results
print("SVM Prediction Error:", svm_error)
print("Naive Bayes Prediction Error:", naive_bayes_error)
print("Decision Tree Prediction Error:", decision_tree_error)

print("\nSVM Confusion Matrix:")
print(svm_confusion_matrix)

print("\nNaive Bayes Confusion Matrix:")
print(naive_bayes_confusion_matrix)

print("\nDecision Tree Confusion Matrix:")
print(decision_tree_confusion_matrix)

# Fine-tuning the Naive Bayes model using GridSearchCV
param_grid_nb = {
    'priors': [None, [0.3, 0.3, 0.4], [0.2, 0.5, 0.3], [0.4, 0.4, 0.2]]  # Valid prior distributions summing to 1
}

grid_search_nb = GridSearchCV(GaussianNB(), param_grid_nb, cv=5, n_jobs=-1)
grid_search_nb.fit(X_train, Y_train)

# Best model and prediction error
best_nb_model = grid_search_nb.best_estimator_
nb_predictions = best_nb_model.predict(X_test)
nb_error = 1 - accuracy_score(Y_test, nb_predictions)

# Display results of the fine-tuned Naive Bayes
print("\nTuned Naive Bayes Prediction Error:", nb_error)
print("Tuned Naive Bayes Confusion Matrix:")
print(confusion_matrix(Y_test, nb_predictions))

