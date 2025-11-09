import joblib
import numpy as np
model=joblib.load("car.pkl")
year=int(input("enter year:"))
brand=int(input("enter brand:"))
fueltype=int(input("enter fuel type:"))
mileage=int(input("enter mileage:"))
print(model.predict(np.array([[year,brand,fueltype,mileage]])))