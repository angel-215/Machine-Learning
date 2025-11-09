import pandas as pd
import sklearn.neighbors as ng
mydata=pd.read_csv("height.csv")
x=mydata[["height"]]
y=mydata[["weight"]]
knn_model=ng.KNeighborsRegressor(n_neighbors=3)
knn_model.fit(x,y)
print(knn_model.predict([[160]]))
