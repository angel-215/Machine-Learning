import pandas as pd
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import joblib
mydata=pd.read_csv("red_wine.csv",sep=";")
x=mydata[["fixed acidity","volatile acidity","citric acid","residual sugar","chlorides","free sulfur dioxide","total sulfur dioxide","density","pH","sulphates","alcohol"]]
y=mydata[["quality"]]
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2)
model=Sequential()
model.add(Dense(10,activation="relu",input_shape=(11,)))
model.add(Dense(10,activation="relu"))
model.add(Dense(10,activation="relu"))
model.add(Dense(1))
model.compile(optimizer="adam",loss="mse")
model.fit(x_train,y_train,epochs=100)
joblib.dump(model,"red_wine.pkl")