import joblib
import numpy as np
model=joblib.load("weight.pkl")
age=int(input("enter age:"))
gender=int(input("enter gender:"))
bodytype=int(input("enter body type:"))
height=int(input("enter height:"))
print(model.predict(np.array([[age,gender,bodytype,height]])))