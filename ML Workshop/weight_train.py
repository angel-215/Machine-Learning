import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import joblib
mydata=pd.read_csv("weight.csv")
ge=LabelEncoder()
be=LabelEncoder()
mydata["gender_enc"]=ge.fit_transform(mydata[["Gender"]])
mydata["body_enc"]=be.fit_transform(mydata[["BodyType"]])
x=mydata[["Age","gender_enc","body_enc","Height"]]
y=mydata[["Weight"]]
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2)
model=Sequential()
model.add(Dense(10,activation="relu",input_shape=(4,)))
model.add(Dense(10,activation="relu"))
model.add(Dense(10,activation="relu"))
model.add(Dense(1))
model.compile(optimizer="adam",loss="mse")
model.fit(x_train,y_train,epochs=100)
joblib.dump(model,"weight.pkl")