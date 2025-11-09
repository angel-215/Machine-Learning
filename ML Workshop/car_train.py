import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import joblib
mydata=pd.read_csv("car.csv")
fe=LabelEncoder()
be=LabelEncoder()
mydata["fueltype_enc"]=fe.fit_transform(mydata[["FuelType"]])
mydata["brand_enc"]=be.fit_transform(mydata[["Brand"]])
x=mydata[["Year","brand_enc","fueltype_enc","Mileage"]]
y=mydata[["Price"]]
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2)
model=Sequential()
model.add(Dense(10,activation="relu",input_shape=(4,)))
model.add(Dense(10,activation="relu"))
model.add(Dense(10,activation="relu"))
model.add(Dense(1))
model.compile(optimizer="adam",loss="mse")
model.fit(x_train,y_train,epochs=100)
joblib.dump(model,"car.pkl")