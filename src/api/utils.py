import os
import joblib
import pandas as pd
from datetime import datetime

# Base directory of project (root)
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../"))

# Paths
MODEL_PATH = os.path.join(BASE_DIR, "models", "churn_model.joblib")
LOG_FILE = os.path.join(BASE_DIR, "logs", "predictions.csv")

def load_model():
    """Load the trained model from models directory."""
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"Model not found at {MODEL_PATH}")
    return joblib.load(MODEL_PATH)

def log_prediction(data: dict, prediction: int):
    """Append prediction data to logs/predictions.csv"""
    os.makedirs(os.path.dirname(LOG_FILE), exist_ok=True)
    row = {
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        **data,
        "prediction": prediction
    }
    df = pd.DataFrame([row])
    if not os.path.exists(LOG_FILE):
        df.to_csv(LOG_FILE, index=False)
    else:
        df.to_csv(LOG_FILE, mode='a', header=False, index=False)

def make_prediction(model, data: dict):
    """Predict and log automatically"""
    df = pd.DataFrame([data])
    prediction = int(model.predict(df)[0])
    log_prediction(data, prediction)
    return prediction