import pandas as pd
import sklearn.neighbors as ng
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import joblib
mydata=pd.read_csv("iris.csv")
x=mydata[["Id","SepalLengthCm","SepalWidthCm","PetalLengthCm","PetalWidthCm"]]
y=mydata[["Species"]]
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2)
model=ng.KNeighborsClassifier(n_neighbors=3)
model.fit(x_train , y_train)
print("training successful")
test_result=model.predict(x_test)
print("Accuracy score",accuracy_score(y_test,test_result))
joblib.dump(model,"iris.pkl")