import pandas as pd
import sklearn.neighbors as ng
from sklearn.metrics import r2_score,mean_squared_error
from sklearn.model_selection import train_test_split
import joblib
import math
mydata=pd.read_csv("heart.csv")
x=mydata[["age","sex","cp","trestbps","chol","fbs","restecg","thalach","exang","oldpeak","slope","ca","thal"]]
y=mydata[["target"]]
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2)
model=ng.KNeighborsClassifier(n_neighbors=3)
model.fit(x_train,y_train)
print("training successful")
joblib.dump(model,"heart.pkl")
test_result=model.predict(x_test)
print("MSE",mean_squared_error(y_test,test_result))
print("RMSE",math.sqrt(mean_squared_error(y_test,test_result)))
print("R2 score",r2_score(y_test,test_result))