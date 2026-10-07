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