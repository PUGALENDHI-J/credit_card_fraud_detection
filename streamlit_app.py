import streamlit as st
import pandas as pd
import joblib

# Load model
model = joblib.load("models/model.pkl")

st.set_page_config(page_title="Credit Card Fraud Detection", layout="centered")

st.title("💳 Credit Card Fraud Detection")
st.write("Upload a CSV file to detect fraudulent transactions")

uploaded_file = st.file_uploader("Upload CSV file", type=["csv"])

if uploaded_file:
    data = pd.read_csv(uploaded_file)

    if "Class" in data.columns:
        data = data.drop("Class", axis=1)

    st.write("📄 Preview of Data")
    st.dataframe(data.head())

    if st.button("🔍 Predict Fraud"):
        prediction = model.predict(data)

        fraud_count = int(prediction.sum())
        legit_count = len(prediction) - fraud_count

        fraud_percent = round((fraud_count / len(prediction)) * 100, 2)
        legit_percent = 100 - fraud_percent

        st.subheader("📊 Results")
        st.success(f"Legitimate Transactions: {legit_count}")
        st.error(f"Fraud Transactions: {fraud_count}")

        st.progress(fraud_percent / 100)
        st.write(f"🚨 Fraud Probability: {fraud_percent}%")
