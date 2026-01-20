import pandas as pd
import joblib
import mlflow
import mlflow.sklearn
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# Load data
df = pd.read_csv("data/raw/churn.csv")

# Clean TotalCharges
df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
df.dropna(inplace=True)

# Select ONLY required features
X = df[["tenure", "MonthlyCharges", "TotalCharges"]]
y = df["Churn"].map({"Yes": 1, "No": 0})

# Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Train model
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

# Evaluate
preds = model.predict(X_test)
acc = accuracy_score(y_test, preds)

print("Accuracy:", acc)

# Save locally
joblib.dump(model, "models/churn_model.joblib")

# MLflow logging
mlflow.set_experiment("Churn_Prediction")

with mlflow.start_run():
    mlflow.log_param("features", "tenure, MonthlyCharges, TotalCharges")
    mlflow.log_metric("accuracy", acc)
    mlflow.sklearn.log_model(
        sk_model=model,
        artifact_path="model",
        registered_model_name="ChurnModel"
    )

# Save reference data for drift detection
X_train.to_csv("monitoring/reference_data.csv", index=False)