import pandas as pd
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import load_model
from sklearn.metrics import mean_squared_error, mean_absolute_error
import numpy as np
import joblib
mydata=pd.read_csv("red_wine.csv",sep=";")
x=mydata[["fixed acidity","volatile acidity","citric acid","residual sugar","chlorides","free sulfur dioxide","total sulfur dioxide","density","pH","sulphates","alcohol"]]
y=mydata[["quality"]]
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2)
scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)
model=Sequential()
model.add(Dense(10,activation="relu",input_shape=(11,)))
model.add(Dense(10,activation="relu"))
model.add(Dense(10,activation="relu"))
model.add(Dense(1))
model.compile(optimizer="adam",loss="mse")
model.fit(x_train,y_train,epochs=100)
test_result = model.predict(x_test)
rmse = np.sqrt(mean_squared_error(y_test, test_result))
mae = mean_absolute_error(y_test, test_result)
print("RMSE:", rmse)
print("MAE:", mae)
model.save("red_wine_model.h5")
joblib.dump(scaler, "red_wine.pkl")