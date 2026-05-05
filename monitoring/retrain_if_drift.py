import os
import pandas as pd
import joblib
from sklearn.linear_model import LogisticRegression

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

LOG_FILE = os.path.join(BASE_DIR, "logs", "predictions.csv")
MODEL_PATH = os.path.join(BASE_DIR, "models", "churn_model.joblib")

FEATURES = ["tenure", "MonthlyCharges", "TotalCharges"]
TARGET = "prediction"


def retrain_model():
    print("Loading new data for retraining...")

    if not os.path.exists(LOG_FILE):
        print(" No prediction logs found. Skipping retraining.")
        return

    df = pd.read_csv(LOG_FILE)

    if df[TARGET].nunique() < 2:
        print(" Only one class present in data. Retraining skipped.")
        return

    X = df[FEATURES]
    y = df[TARGET]

    model = LogisticRegression(max_iter=1000)
    model.fit(X, y)

    joblib.dump(model, MODEL_PATH)
    print("Model retrained and saved successfully")


if __name__ == "__main__":
    retrain_model()