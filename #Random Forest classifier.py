from sklearn import datasets
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
iris=datasets.load_iris()
x=iris.data
y=iris.target
rfc=RandomForestClassifier()
rfc.fit(x,y)
#prediction
print(rfc.predict(x[[0]]))
print(y[[0]])
#data split
X_train, X_test, Y_train, Y_test = train_test_split(x,y, test_size=0.2,random_state=100)
rfc.fit(X_train,Y_train)#training the classifier
#predict
y_pred=rfc.predict(X_test)
print(rfc.predict(X_test))
#actual class labels
print(Y_test)
#accuracy
print(accuracy_score(Y_test,y_pred))
