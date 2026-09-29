"""Clean, top-to-bottom training script (replaces the modelling part of the notebook).
Run once:  python train.py   -> creates model.pkl
"""
import os
import joblib
import kagglehub
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

TARGET = "Performance Index"
FEATURES = [
    "Hours Studied",
    "Previous Scores",
    "Sleep Hours",
    "Sample Question Papers Practiced",
    "Extracurricular Activities_Yes",
]

path = kagglehub.dataset_download("nikhil7280/student-performance-multiple-linear-regression")
df = pd.read_csv(os.path.join(path, "Student_Performance.csv")).drop_duplicates()

X = pd.get_dummies(df.drop(columns=TARGET), drop_first=True, dtype=int)[FEATURES]
y = df[TARGET]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scaler + model in ONE object, so they can never get out of sync
pipe = Pipeline([("scaler", StandardScaler()), ("model", LinearRegression())])
pipe.fit(X_train, y_train)

y_pred = pipe.predict(X_test)
print(f"MAE : {mean_absolute_error(y_test, y_pred):.3f}")
print(f"RMSE: {mean_squared_error(y_test, y_pred) ** 0.5:.3f}")
print(f"R²  : {r2_score(y_test, y_pred):.4f}")

joblib.dump({"pipeline": pipe, "features": FEATURES}, "model.pkl")
print("Saved model.pkl")
