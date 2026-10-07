# Part 1
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

# Load the training dataset
X = np.array([-3.0, -2.5, -2.0, -1.5, -1.0, 0.0, 1.0, 1.5, 2.0, 2.5, 2.7])
Y = np.array([15.5, 12.9, 9.5, 6.2, 5.8, 5.5, 7.1, 9.7, 13.5, 18.4, 21.4])

# Reshape from shape (11,) to shape (11, 1)
X_2d = X.reshape(-1, 1)

# Displays a scatter plot for the dataset using “matplotlib” package for data visualization
plt.scatter(X, Y, color="blue", label="Training data")
plt.xlabel("x")
plt.ylabel("y")
plt.title("Training dataset")
plt.legend()
plt.show()

# Train using linear regression
model = LinearRegression()
model.fit(X_2d, Y)

# Show the coefficients and the equation
w0 = model.intercept_        # w0 is the intercept
w1 = model.coef_[0]          # w1 is the slope
 
print("Coefficient vector [w0, w1] =", [round(float(w0), 4), round(float(w1), 4)])
print(f"Equation: y = {w0:.4f} + {w1:.4f} * x")

# Function to compute the RMSE
def rmse(y_actual, y_predicted):
    errors = y_actual - y_predicted          # difference for every example
    return np.sqrt(np.mean(errors ** 2))     # square, average, square root

# Compute the training RMSE
Y_pred = model.predict(X_2d)
print(rmse(Y, Y_pred))
 
# Plot the data together with the line the model found
plt.scatter(X, Y, color="blue", label="Training data")
plt.plot(X, Y_pred, color="red", label="Linear regression line")
plt.xlabel("x")
plt.ylabel("y")
plt.title("Linear regression fit")
plt.legend()
plt.show()

#split the dataset into training set (80%) and testing set (20%) and perform the linear regression to find the coefficients using the least square method
import pandas as pd
from sklearn.model_selection import train_test_split

df = pd.read_csv("GasProperties.csv")

# T = temperature, P = pressure, TC = critical temperature, SV = specific volume
inputs = df[["T", "P", "TC", "SV"]].values
output = df["Idx"].values

# 80% training, 20% testing
X_train, X_test, y_train, y_test = train_test_split(
    inputs, output, test_size=0.2, random_state=2
)

# Add a column of 1s so the model also learns w0 (the intercept)
def add_bias_column(X):
    ones = np.ones((X.shape[0], 1))
    return np.hstack([ones, X])

X_train_b = add_bias_column(X_train)
X_test_b = add_bias_column(X_test)

# Least squares formula: w = (X^T X)^-1 X^T y
w = np.linalg.inv(X_train_b.T @ X_train_b) @ X_train_b.T @ y_train
print("Coefficients [w0, w1, w2, w3, w4] =", w)

# Predictions are  y_hat = X * w
train_pred = X_train_b @ w
test_pred = X_test_b @ w

# Uses the rmse(y_actual, y_predicted) function defined at the top of this file
print("Training RMSE =", rmse(y_train, train_pred))
print("Testing  RMSE =", rmse(y_test, test_pred))