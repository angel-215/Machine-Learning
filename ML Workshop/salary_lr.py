import pandas as pd
import matplotlib.pyplot as pt
from sklearn.preprocessing import LabelEncoder
import sklearn.linear_model as lr
mydata=pd.read_csv("salary.csv")
x=mydata[["qualification"]]
y=mydata[["salary"]]
le=LabelEncoder()
mydata["new_education"]=le.fit_transform(mydata[["qualification"]])
x_new=mydata[["new_education"]]
pt.scatter(x_new,y)
pt.show()
model=lr.LinearRegression()
model.fit(x_new,y)
print(model.predict([[0]]))