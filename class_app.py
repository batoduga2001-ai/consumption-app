import streamlit as st
import pandas as pd
import joblib

# =========================
# LOAD MODEL
# =========================
model = joblib.load("telco_churn_model.pkl")

# =========================
# UI
# =========================
st.title("Telco Customer Churn Prediction")

st.write("Enter customer details below:")

tenure = st.number_input("Tenure", min_value=0)
monthly = st.number_input("Monthly Charges", min_value=0.0, format="%.2f")
total = st.number_input("Total Charges", min_value=0.0, format="%.2f")

# =========================
# PREDICT BUTTON
# =========================
if st.button("Predict"):
    input_data = pd.DataFrame(
        [[tenure, monthly, total]],
        columns=["tenure", "MonthlyCharges", "TotalCharges"]
    )

    result = model.predict(input_data)[0]

    if result == 1:
        st.error("⚠ Customer WILL CHURN")
    else:
        st.success("✅ Customer WILL STAY")