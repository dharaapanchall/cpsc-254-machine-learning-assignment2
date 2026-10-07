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
print("Training RMSE =", round(rmse(Y, Y_pred), 4))
 
# Plot the data together with the line the model found
plt.scatter(X, Y, color="blue", label="Training data")
plt.plot(X, Y_pred, color="red", label="Linear regression line")
plt.xlabel("x")
plt.ylabel("y")
plt.title("Linear regression fit")
plt.legend()
plt.show()