import joblib
import numpy as np
import pandas as pd

aq_model = joblib.load("aq_model.pkl")
scaler_aq = joblib.load("scaler_aq.pkl")

hr_model = joblib.load("hr_model.pkl")
scaler_hr = joblib.load("scaler_hr.pkl")

print("✅ Models and scalers loaded successfully!")

def predict_air_quality(pm10, pm2_5, no2, so2, o3, temperature, humidity, windspeed):
    features = pd.DataFrame([[pm10, pm2_5, no2, so2, o3, temperature, humidity, windspeed]],
                        columns=["PM10","PM2_5","NO2","SO2","O3","Temperature","Humidity","WindSpeed"])
    scaled_features = scaler_aq.transform(features)
    prediction = aq_model.predict(scaled_features)
    return prediction[0]

def predict_health_risk(pm10, pm2_5, no2, so2, o3, temperature, humidity, windspeed,
                        respiratory_cases, cardiovascular_cases, hospital_admission, health_impact_score):
    air_quality = predict_air_quality(pm10,pm2_5, no2, so2, o3, temperature, humidity, windspeed)
    features = pd.DataFrame([[pm10, pm2_5, no2, so2, o3, temperature, humidity, windspeed,
                            respiratory_cases, cardiovascular_cases, hospital_admission, health_impact_score]],
                        columns=["PM10","PM2_5","NO2","SO2","O3","Temperature","Humidity","WindSpeed",
                                "RespiratoryCases","CardiovascularCases","HospitalAdmissions","HealthImpactScore"])
    if air_quality == "Unhealthy":
        final_risk = "High Risk"
    elif air_quality == "Moderate":
        final_risk = "Mild Risk"
    else:
        final_risk = "No Risk"
    return air_quality,final_risk

print("\nSample Predictions:")

pm10 = float(input("Enter PM10: "))
pm2_5 = float(input("Enter PM2.5: "))
no2 = float(input("Enter NO2: "))
so2 = float(input("Enter SO2: "))
o3 = float(input("Enter O3: "))
temperature = float(input("Enter Temperature: "))
humidity = float(input("Enter Humidity: "))
windspeed = float(input("Enter Wind Speed: "))
respiratory_cases = float(input("Enter Respiratory Cases: "))
cardiovascular_cases = float(input("Enter Cardiovascular Cases: "))
hospital_admission = float(input("Enter Hospital Admissions: "))
health_impact_score = float(input("Enter Health Impact Score: "))

air_quality, hr_result = predict_health_risk(pm10,pm2_5,no2,so2,o3,temperature,humidity,windspeed,respiratory_cases,
                        cardiovascular_cases,hospital_admission,health_impact_score)
print("Predicted Air Quality:", air_quality)
print("Predicted Health Risk:", hr_result)