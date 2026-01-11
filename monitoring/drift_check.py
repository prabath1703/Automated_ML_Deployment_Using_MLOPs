import pandas as pd
from evidently.report import Report
from evidently.metric_preset import DataDriftPreset

# File paths
REFERENCE_PATH = "monitoring/reference_data.csv"
CURRENT_PATH = "logs/predictions.csv"
REPORT_PATH = "logs/data_drift_report.html"

# Features used during training
FEATURES = ["tenure", "MonthlyCharges", "TotalCharges"]

# Load data
reference = pd.read_csv(REFERENCE_PATH)
current = pd.read_csv(CURRENT_PATH)

# 🔑 Select only numeric features
reference = reference[FEATURES]
current = current[FEATURES]

# 🔑 Ensure numeric dtype (very important)
reference = reference.apply(pd.to_numeric, errors="coerce")
current = current.apply(pd.to_numeric, errors="coerce")

# Drop rows with NaNs
reference.dropna(inplace=True)
current.dropna(inplace=True)

# Generate drift report
report = Report(metrics=[DataDriftPreset()])
report.run(reference_data=reference, current_data=current)

# Save report
report.save_html(REPORT_PATH)

print("Drift report generated successfully")
