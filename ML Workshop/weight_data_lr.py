import pandas as pd
from sklearn.linear_model import LinearRegression
mydata= pd.read_csv("weight_data.csv")
x=mydata[["height","age","bmi","muscle_mass","body_fat"]]
y=mydata[["weight"]]
model=LinearRegression()
model.fit(x,y)
print(model.predict([[172,25,23.5,36,22]]))