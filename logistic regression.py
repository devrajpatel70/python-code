import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# Load the dataset
df = pd.read_csv('friends.csv')

# Create a binary target variable: HighSalary (1 if Salary > 75000, else 0)
df['HighSalary'] = (df['Salary'] > 75000).astype(int)

# Use Age as the only feature
X = df[['Age']]  # Feature
y = df['HighSalary']  # Target

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train a logistic regression model
model = LogisticRegression()
model.fit(X_train, y_train)

# Make predictions on the test set
y_pred = model.predict(X_test)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy}")

# Generate predictions for a range of ages
ages = np.linspace(df['Age'].min(), df['Age'].max(), 100).reshape(-1, 1)
predicted_labels = model.predict(ages)

# Plot the data points (blue: HighSalary=0, orange: HighSalary=1)
plt.scatter(df['Age'], df['HighSalary'], color='blue', label='Actual Data')

# Plot the decision boundary
plt.plot(ages, predicted_labels, color='red', label='Logistic Regression Boundary')

# Add labels and legend
plt.xlabel('Age')
plt.ylabel('High Salary (1=Yes, 0=No)')
plt.title('Logistic Regression: Age vs High Salary')
plt.legend()
plt.grid()

# Show the plot
plt.show()
