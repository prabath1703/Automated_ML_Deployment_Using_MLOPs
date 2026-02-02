import joblib
import os

import pandas as pd
import numpy as np
import mlflow
import mlflow.sklearn

from sklearn.model_selection import train_test_split, StratifiedKFold, cross_validate
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score



os.makedirs("models", exist_ok=True)

# =========================
# 1. Load & preprocess data
# =========================
EXPERIMENT_NAME = "Churn Prediction"

df = pd.read_csv("data/raw/churn.csv")

df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
df.dropna(inplace=True)

FEATURES = ["tenure", "MonthlyCharges", "TotalCharges"]
TARGET = "Churn"

X = df[FEATURES]
y = df[TARGET].map({"Yes": 1, "No": 0})


# =========================
# 2. Train-test split
# =========================
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)


# =========================
# 3. K-Fold setup
# =========================
kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

scoring = {
    "accuracy": "accuracy",
    "precision": "precision",
    "recall": "recall",
    "f1": "f1"
}


# =========================
# 4. Models
# =========================
models = {
    "Logistic Regression": LogisticRegression(max_iter=1000),
    "Random Forest Classifier": RandomForestClassifier(
        n_estimators=200,
        max_depth=8,
        random_state=42,
        n_jobs=-1
    )
}


# =========================
# 5. MLflow experiment
# =========================
mlflow.set_experiment(EXPERIMENT_NAME)

results = {}


for model_name, model in models.items():
    with mlflow.start_run(run_name=model_name):

        cv_results = cross_validate(
            model,
            X_train,
            y_train,
            cv=kfold,
            scoring=scoring
        )

        metrics = {
            "accuracy": np.mean(cv_results["test_accuracy"]),
            "precision": np.mean(cv_results["test_precision"]),
            "recall": np.mean(cv_results["test_recall"]),
            "f1_score": np.mean(cv_results["test_f1"])
        }

        # Train on full training set
        model.fit(X_train, y_train)

        # Log params
        mlflow.log_param("algorithm", model_name)
        mlflow.log_param("features", ", ".join(FEATURES))
        mlflow.log_param("cv_folds", 5)

        # Log metrics
        for k, v in metrics.items():
            mlflow.log_metric(k, v)

        # Log model
        mlflow.sklearn.log_model(model, artifact_path="model")

        results[model_name] = {
            "model": model,
            "f1_score": metrics["f1_score"]
        }

        print(f"\n{model_name} CV Metrics:")
        for k, v in metrics.items():
            print(f"{k}: {v:.4f}")


# =========================
# 6. Best model selection
# =========================
best_model_name = max(results, key=lambda x: results[x]["f1_score"])
best_model = results[best_model_name]["model"]

print(f"\n Best Model: {best_model_name}")


# =========================
# 7. Log Best Model run
# =========================
with mlflow.start_run(run_name="Best Model"):
    mlflow.log_param("selected_model", best_model_name)
    mlflow.sklearn.log_model(
        sk_model=best_model,
        artifact_path="model",
        registered_model_name="ChurnClassifier"
    )


# =========================
# 8. Save reference data
# =========================
X_train.to_csv(
    "monitoring/reference_data.csv",
    index=False
)


joblib.dump(best_model, "models/churn_model.joblib")

print("\nTraining completed.")
print("All experiments logged in MLflow.")
