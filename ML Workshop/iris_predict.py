import joblib
model=joblib.load("iris.pkl")
id=int(input("enter the id:"))
sepall=float(input("enter the sepal length in cm:"))
sepalw=float(input("enter the sepal width in cm:"))
petall=float(input("enter the petal length in cm:"))
petalw=float(input("enter the petal width in cm:"))
print(model.predict([[id,sepall,sepalw,petall,petalw]]))