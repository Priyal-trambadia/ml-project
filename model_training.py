# model_training.py
import pandas as pd
from sklearn.linear_model import LinearRegression
import joblib
import os

# Ensure model directory exists
os.makedirs("model", exist_ok=True)

# Load small training dataset
df = pd.read_csv("data/sample.csv")  # file: data/sample.csv
X = df[["StudyHours"]]
y = df["ExamScore"]

# Train a simple linear regression model
model = LinearRegression()
model.fit(X, y)

# Save the trained model to disk
model_path = os.path.join("model", "student_model.pkl")
joblib.dump(model, model_path)
print(f"Saved model to: {model_path}")

