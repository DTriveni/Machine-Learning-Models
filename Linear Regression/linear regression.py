#import NumPy and matplotlib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# import statments
from sklearn import datasets, linear_model
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

# Load the diabetes dataset
from sklearn.datasets import load_diabetes

# Load the diabetes dataset into a variable
# INPUT  : None (built-in dataset)
# OUTPUT : A Bunch object containing:
#          - data (features)
#          - target (labels)
#          - feature_names
#          - description
diabetes = load_diabetes()

# Print the shape of the feature matrix
# INPUT  : diabetes.data (NumPy array of features)
# OUTPUT : A tuple (number_of_samples, number_of_features)
print(diabetes.data.shape) 

# View the names of the features (columns)
# INPUT  : diabetes.feature_names
# OUTPUT : A list of feature (column) names
diabetes.feature_names

# Convert the feature data into a Pandas DataFrame
# INPUT  : diabetes.data (NumPy array)
#          diabetes.feature_names (column names)
# OUTPUT : x_data -> DataFrame of input features (X)
x_data = pd.DataFrame(diabetes.data, columns=diabetes.feature_names)

# # Display the feature DataFrame
x_data

# Convert the target values into a Pandas DataFrame
# INPUT  : diabetes.target (NumPy array)
# OUTPUT : y_data -> DataFrame containing the target variable (y)
y_data = pd.DataFrame(diabetes.target, columns=["Target"])

# Display the target DataFrame
y_data


# Split the dataset into training and testing sets
# INPUT  : 
#   x_data -> feature DataFrame
#   y_data -> target DataFrame
#   test_size = 0.2 -> 20% data for testing
#   random_state = 42 -> ensures reproducibility with same data points
#
# OUTPUT :
#   x_train -> 80% of features for training
#   x_test  -> 20% of features for testing
#   y_train -> 80% of target values for training
#   y_test  -> 20% of target values for testing
x_train, x_test, y_train, y_test = train_test_split(x_data, y_data, test_size = 0.2,random_state = 42)

# Add a column of ones to x_train for the intercept (bias) term
# INPUT  : x_train (DataFrame with only feature columns)
# OUTPUT : x_train is modified in-place by adding a new column called "Ones"
#          This column contains all 1s and represents the intercept term (θ₀)
x_train.insert(0,"Ones",1)

# Add a column of ones to x_test for the intercept (bias) term
# INPUT  : x_test (DataFrame with only feature columns)
# OUTPUT : x_test is modified in-place by adding a new column called "Ones"
#          This ensures x_test has the same structure as x_train
x_test.insert(0,"Ones",1)


# Display the updated training feature matrix and testing feature matrix
# INPUT  : x_train and x_test
# OUTPUT : DataFrame with an extra first column ("Ones")
x_train
x_test

# Calculate the regression coefficients (theta) using the normal equation
# θ = (XᵀX)⁻¹ Xᵀ y
# np.linalg.pinv computes the Moore–Penrose pseudo-inverse, which is more stable
# than a direct inverse and works even if XᵀX is not invertible.
#
# INPUT  :
#   x_train -> Feature matrix with intercept column (shape: [n_samples, n_features+1])
#   y_train -> Target values (shape: [n_samples, 1])
#
# PROCESS :
#   np.linalg.pinv(x_train) -> Pseudo-inverse of x_train
#   Matrix multiplication (@) with y_train
#
# OUTPUT :
#   c_theta_train -> Coefficient vector (theta), including intercept
c_theta_train = np.linalg.pinv(x_train)@y_train

# Generate predictions on the test set
# INPUT  :
#   x_test -> Feature matrix with intercept column
#   c_theta_train -> Learned coefficients (theta)
#
# OUTPUT :
#   y_pred -> Predicted target values
y_pred = x_test.values@c_theta_train
y_pred
# Calculate Mean Squared Error (MSE)
# INPUT  :
#   y_test.values -> Actual target values
#   y_pred        -> Predicted target values
# OUTPUT :
#   mse -> Average of squared prediction errors
mse = mean_squared_error(y_test.values, y_pred)

# Calculate Root Mean Squared Error (RMSE)
# INPUT  : mse
# OUTPUT : rmse -> Error in original target units
rmse = np.sqrt(mse)

# Calculate R-squared (R²)
# INPUT  :
#   y_test.values -> Actual values
#   y_pred        -> Predicted values
# OUTPUT :
#   r2 -> Proportion of variance explained by the model
r2 = r2_score(y_test.values, y_pred)

print("RMSE:",rmse)
print("R2:",r2)

# Instantiate the linear regression
from sklearn.linear_model import LinearRegression

# Instantiate the Linear Regression model
# INPUT  :
#   fit_intercept=False -> because x_train already contains a column of ones
# OUTPUT :
#   lr_model -> untrained LinearRegression object
lr_model = LinearRegression()

# Train (fit) the model on training data
# INPUT  :
#   x_train -> Feature matrix with intercept column
#   y_train -> Target values
# OUTPUT :
#   lr_model -> trained model with learned coefficients
lr_model.fit(x_train,y_train)

# Predict target values on the test set using the trained LinearRegression model
# INPUT  :
#   x_test -> Feature matrix (with intercept column)
# OUTPUT :
#   y_pred_sklearn -> Predicted target values from the sklearn model
lr_model.predict(x_test)

# Scatter plot: Predicted vs Actual values (using sklearn LinearRegression)
# INPUT  :
#   y_test -> Actual target values
#   y_pred_sklearn -> Predicted target values from sklearn model
# OUTPUT :
#   Scatter plot showing prediction accuracy
plt.scatter(
    y_test,
    y_pred,
    alpha=0.7,
    label="Predictions"
)

# y = x reference line
plt.plot(
    [y_test.min(), y_test.max()],
    [y_test.min(), y_test.max()],
    linestyle="--",
    label="Perfect prediction (y = x)"
)

plt.xlabel("Actual values (y_test)")
plt.ylabel("Predicted values (y_pred)")
plt.title("Predicted vs Actual Values")
plt.legend()
plt.grid(True)

plt.show()
