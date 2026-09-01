import streamlit as st
import joblib
import pandas as pd

# -----------------------------
# Page Configuration
# -----------------------------
st.set_page_config(
    page_title="Customer Churn Prediction",
    page_icon="📊",
    layout="centered"
)

# -----------------------------
# Load Model
# -----------------------------
model = joblib.load("models/churn_model.joblib")

# -----------------------------
# Title
# -----------------------------
st.title("📊 Customer Churn Prediction")
st.write(
    "Predict whether a customer is likely to churn "
    "using a machine learning model."
)

st.divider()

# -----------------------------
# User Inputs
# -----------------------------
st.subheader("Customer Information")

tenure = st.number_input(
    "Tenure (months)",
    min_value=0,
    max_value=100,
    value=24
)

monthly_charges = st.number_input(
    "Monthly Charges",
    min_value=0.0,
    value=70.0
)

total_charges = st.number_input(
    "Total Charges",
    min_value=0.0,
    value=1500.0
)

# -----------------------------
# Prediction
# -----------------------------
if st.button("🔮 Predict Churn", use_container_width=True):

    input_data = pd.DataFrame(
        [[tenure, monthly_charges, total_charges]],
        columns=[
            "tenure",
            "MonthlyCharges",
            "TotalCharges"
        ]
    )

    prediction = model.predict(input_data)[0]

    st.divider()

    if prediction == 1:
        st.error("⚠️ Customer is likely to churn")
    else:
        st.success("✅ Customer is unlikely to churn")