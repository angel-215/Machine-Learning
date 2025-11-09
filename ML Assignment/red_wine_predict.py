import joblib
import numpy as np
from tensorflow.keras.models import load_model
model=load_model("red_wine_model.h5", compile=False)
scaler = joblib.load("red_wine.pkl")
fa=float(input("enter fixed acidity:"))
va=float(input("enter volatile acidity:"))
ca=float(input("enter citric acid:"))
rs=float(input("enter residual sugar:"))
c=float(input("enter chlorides:"))
fsd=float(input("enter free sulphur dioxide:"))
tsd=float(input("enter total sulphur dioxide:"))
d=float(input("enter density:"))
ph=float(input("enter ph:"))
s=float(input("enter sulphates:"))
a=float(input("enter alcohol:"))
features=np.array([[fa,va,ca,rs,c,fsd,tsd,d,ph,s,a]])
feature_scale=scaler.transform(features)
print(model.predict(feature_scale))