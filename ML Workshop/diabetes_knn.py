import pandas as pd
import sklearn.neighbors as ng
mydata=pd.read_csv("diabetes.csv")
x=mydata[["Pregnancies","Glucose","BloodPressure","SkinThickness","Insulin","BMI","DiabetesPedigreeFunction","Age"]]
y=mydata[["Outcome"]]
model=ng.KNeighborsClassifier(n_neighbors=3)
model.fit(x,y)
x=model.predict([[2,78,65,38,1,33.9,32,22]])
if x[0]==1 :
    print("diabetic")
else:
    print("non diabetic")