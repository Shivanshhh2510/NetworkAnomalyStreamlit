import streamlit as st
import pandas as pd
import numpy as np
import joblib
from sklearn.ensemble import IsolationForest

st.set_page_config(page_title="Anomaly Detection", layout="wide")

st.title("🚨 Network Traffic Anomaly Detection")
st.markdown("Upload your network traffic CSV file and detect anomalies using Isolation Forest.")

uploaded_file = st.file_uploader("Upload your CSV file", type=["csv"])

# Load Isolation Forest model and scaler
iso_model = joblib.load("iso_model.pkl")
scaler = joblib.load("scaler.pkl")

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    X = scaler.transform(df)

    # Predict anomalies
    iso_pred = iso_model.predict(X)
    iso_pred = np.where(iso_pred == 1, 0, 1)  # 1=normal → 0; -1=anomaly → 1

    df['anomaly'] = iso_pred
    st.write(df.head())

    st.subheader("📊 Anomaly Distribution")
    st.bar_chart(df['anomaly'].value_counts())

    csv = df.to_csv(index=False).encode()
    st.download_button("Download Results", csv, "predictions.csv", "text/csv")

    st.success("✅ Prediction complete!")
