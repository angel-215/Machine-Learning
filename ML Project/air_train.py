import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score,precision_score,recall_score,confusion_matrix
import joblib

data = pd.read_csv("air_quality_health_impact_data.csv")
data = data.dropna()

X_aq = data[["PM10", "PM2_5", "NO2", "SO2", "O3", "Temperature", "Humidity", "WindSpeed"]]
y_aq = data["AQI"]  

def categorize_aqi(aqi):
    if aqi <= 100:
        return "Good"
    elif aqi <= 300:
        return "Moderate"
    else:
        return "Unhealthy"

y_aq = y_aq.apply(categorize_aqi)

scaler_aq = StandardScaler()
X_aq_scaled = scaler_aq.fit_transform(X_aq)

X_train_aq, X_test_aq, y_train_aq, y_test_aq = train_test_split(X_aq_scaled, y_aq, test_size=0.2)

aq_model = DecisionTreeClassifier(random_state=42)
aq_model.fit(X_train_aq, y_train_aq)

y_pred_aq = aq_model.predict(X_test_aq)

print("Air Quality Model Metrics:")
print("Accuracy:", accuracy_score(y_test_aq, y_pred_aq))
print("Precision:", precision_score(y_test_aq, y_pred_aq, average='weighted'))
print("Recall:", recall_score(y_test_aq, y_pred_aq, average='weighted'))
print("Confusion Matrix:\n", confusion_matrix(y_test_aq, y_pred_aq))

joblib.dump(aq_model, "aq_model.pkl")
joblib.dump(scaler_aq, "scaler_aq.pkl")
print("✅ Air Quality model trained and saved!")

X_hr = data[["PM10", "PM2_5", "NO2", "SO2", "O3", "Temperature", "Humidity", "WindSpeed",
             "RespiratoryCases", "CardiovascularCases", "HospitalAdmissions", "HealthImpactScore"]]
y_hr = data["HealthImpactClass"]

scaler_hr = StandardScaler()
X_hr_scaled = scaler_hr.fit_transform(X_hr)

X_train_hr, X_test_hr, y_train_hr, y_test_hr = train_test_split(X_hr_scaled, y_hr, test_size=0.2)

hr_model =  DecisionTreeClassifier(random_state=42)
hr_model.fit(X_train_hr, y_train_hr)

y_pred_hr = hr_model.predict(X_test_hr)

print("\nHealth Risk Model Metrics:")
print("Accuracy:", accuracy_score(y_test_hr, y_pred_hr))
print("Precision:", precision_score(y_test_hr, y_pred_hr, average='weighted'))
print("Recall:", recall_score(y_test_hr, y_pred_hr, average='weighted'))
print("Confusion Matrix:\n", confusion_matrix(y_test_hr, y_pred_hr))

joblib.dump(hr_model, "hr_model.pkl")
joblib.dump(scaler_hr, "scaler_hr.pkl")
print("✅ Health Risk model trained and saved!")