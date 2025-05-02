import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, KFold
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder

car_data = pd.read_csv('car.data', header=None)
ttt_data = pd.read_csv('tic-tac-toe.data', header=None)


# encode categorical features to numerical values
encoder = LabelEncoder()
for col in car_data.columns:
    car_data[col] = encoder.fit_transform(car_data[col])

# Encode categorical features to numerical values
for col in ttt_data.columns:
    ttt_data[col] = encoder.fit_transform(ttt_data[col])

# Split data into features  and labels
X_car = car_data.iloc[:, :-1]
y_car = car_data.iloc[:, -1]

X_ttt = ttt_data.iloc[:, :-1]
y_ttt = ttt_data.iloc[:, -1]

depths = range(3, 11)

def cross_validate_decision_tree(X, y, dataset_name):
    """Function to perform K-fold cross-validation and calculate average accuracy"""
    print("------------------")
    print(f"\nEvaluating {dataset_name} dataset")
    avg_accuracies = []
    for depth in depths:
        kf = KFold(n_splits=10, shuffle=True, random_state=42)
        accuracies = []

        for train_index, test_index in kf.split(X):
            X_train, X_test = X.iloc[train_index], X.iloc[test_index]
            y_train, y_test = y.iloc[train_index], y.iloc[test_index]

            # Train Decision Tree
            clf = DecisionTreeClassifier(max_depth=depth)
            clf.fit(X_train, y_train)
            y_pred = clf.predict(X_test)

            # Store accuracy
            accuracies.append(accuracy_score(y_test, y_pred))

        avg_accuracy = np.mean(accuracies)
        avg_accuracies.append(avg_accuracy)
        print(f"Max Depth: {depth}, Average Accuracy: {avg_accuracy:.2f}")

    # plot accuracy vs max depth
    plt.plot(depths, avg_accuracies, marker='o', label=dataset_name)
    plt.xlabel('Max Depth')
    plt.ylabel('Average Accuracy')
    plt.title(f'Decision Tree Performance for {dataset_name} Dataset')
    plt.legend()
    plt.grid(True)

cross_validate_decision_tree(X_car, y_car, "Car")
cross_validate_decision_tree(X_ttt, y_ttt, "Tic-Tac-Toe")

plt.show()
