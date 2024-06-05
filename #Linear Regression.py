#linear regression model
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
import sklearn
from sklearn.datasets import load_iris
iris=load_iris()
x=iris.data
x
y=iris.target
y
# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=100)

# Train the linear regression model
model = LinearRegression()
model.fit(X_train, y_train)

# Make predictions on the testing set
y_pred = model.predict(X_test)

# Plot the data and the fitted line
# Loop through each feature in X_test
plt.scatter(X_test[:, 0], y_test, color='blue', label='Actual Values')
plt.plot(X_test[:, 0], y_pred, color='red', label='Predicted Values')
plt.xlabel('Feature 1 (Sepal Length)')  # Assuming the first feature is Sepal Length
plt.ylabel('Target Variable')
plt.title('Linear Regression - Actual vs. Predicted Values')
plt.legend()
plt.show() 
