# Loads the “Iris flower” dataset directly from “sklearn.datasets” package
import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score

# Load the Iris flower dataset
iris = load_iris()
X = iris.data
y = iris.target

# Split the dataset into training set (80%) and testing set (20%)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Train the MLP classifier with one hidden layer and 3 neurons
model = MLPClassifier(
    hidden_layer_sizes=(3,),
    activation="relu",
    solver="adam",
    learning_rate_init=0.01,
    max_iter=1000,
    batch_size=32,
    random_state=42
)

model.fit(X_train, y_train)

# Predict the training and testing datasets
train_pred = model.predict(X_train)
test_pred = model.predict(X_test)

# Compute the training and testing accuracy
train_acc = accuracy_score(y_train, train_pred) * 100
test_acc = accuracy_score(y_test, test_pred) * 100

print("Original MLP Classifier")
print(f"Training Accuracy = {train_acc:.2f}%")
print(f"Testing Accuracy = {test_acc:.2f}%")

# Change the hyperparameters without adding more layers or neurons
model2 = MLPClassifier(
    hidden_layer_sizes=(3,),
    activation="tanh",
    solver="adam",
    learning_rate_init=0.001,
    max_iter=2000,
    batch_size=32,
    random_state=42
)

model2.fit(X_train, y_train)

# Predict the training and testing datasets
train_pred2 = model2.predict(X_train)
test_pred2 = model2.predict(X_test)

# Compute the training and testing accuracy
train_acc2 = accuracy_score(y_train, train_pred2) * 100
test_acc2 = accuracy_score(y_test, test_pred2) * 100

print("\nMLP Classifier with Adjusted Hyperparameters")
print(f"Training Accuracy = {train_acc2:.2f}%")
print(f"Testing Accuracy = {test_acc2:.2f}%")


# Increase network complexity by adding more layers and neurons
model3 = MLPClassifier(
    hidden_layer_sizes=(10, 10),
    activation="tanh",
    solver="lbfgs",
    learning_rate_init=0.01,
    max_iter=1000,
    batch_size=32,
    random_state=42
)

model3.fit(X_train, y_train)

# Predict the training and testing datasets
train_pred3 = model3.predict(X_train)
test_pred3 = model3.predict(X_test)

# Compute the training and testing accuracy
train_acc3 = accuracy_score(y_train, train_pred3) * 100
test_acc3 = accuracy_score(y_test, test_pred3) * 100

print("\nMLP Classifier with Increased Complexity")
print(f"Training Accuracy = {train_acc3:.2f}%")
print(f"Testing Accuracy = {test_acc3:.2f}%")
