import pandas as pd
import sklearn.linear_model as lm
import matplotlib.pyplot as mt
mydata=pd.read_csv("fuel.csv")
x=mydata[["month"]]
y=mydata[["price"]]
mt.scatter(x,y)
mt.show()
model=lm.LinearRegression()
model.fit(x,y)
print(model.predict([[10]]))