import mlflow.sklearn
import pandas as pd
import joblib


def load_model():
    return joblib.load("models/churn_model.joblib")

def make_prediction(model, data: dict):
    df = pd.DataFrame([data])
    prediction = model.predict(df)
    return int(prediction[0])
