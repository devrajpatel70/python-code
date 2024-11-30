import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

# Load the dataset
df = pd.read_csv('friends.csv')

# Use Age as the feature and Salary as the target
X = df[['Age']]  # Feature
y = df['Salary']  # Target

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train a linear regression model
model = LinearRegression()
model.fit(X_train, y_train)

# Make predictions on the test set
y_pred = model.predict(X_test)

# Evaluate the model
mse = mean_squared_error(y_test, y_pred)
print(f"Mean Squared Error: {mse}")

# Generate a range of ages for the regression line
ages = np.linspace(df['Age'].min(), df['Age'].max(), 100).reshape(-1, 1)
predicted_salaries = model.predict(ages)

# Plot the data points
plt.scatter(df['Age'], df['Salary'], color='blue', label='Actual Data')

# Plot the regression line
plt.plot(ages, predicted_salaries, color='red', label='Regression Line')

# Add labels and legend
plt.xlabel('Age')
plt.ylabel('Salary')
plt.title('Linear Regression: Age vs Salary')
plt.legend()
plt.grid()

# Show the plot
plt.show()