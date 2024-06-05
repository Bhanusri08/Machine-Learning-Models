#logistic regression
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

import sklearn
from sklearn.datasets import load_iris
iris=load_iris()
x=iris.data
x
y=iris.target
y
# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=100)

# Create the logistic regression model
model = LogisticRegression()

# Train the model
model.fit(X_train, y_train)

# Make predictions on the testing set
y_pred = model.predict(X_test)

# Evaluate model performance (accuracy in this case)
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)
