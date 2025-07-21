import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest
from tensorflow.keras.models import load_model
from sklearn.preprocessing import MinMaxScaler
import joblib

st.set_page_config(page_title="Anomaly Detection", layout="wide")

st.title("🚨 Network Traffic Anomaly Detection")
st.markdown("Upload network traffic data (with same structure as training data) and select a model to detect anomalies.")

# Upload CSV
uploaded_file = st.file_uploader("Upload your CSV file", type=["csv"])

# Load models
iso_model = joblib.load("iso_model.pkl")
ae_model = load_model("autoencoder_model.h5")
scaler = joblib.load("scaler.pkl")

# Select model
model_choice = st.selectbox("Choose a model", ["Isolation Forest", "Autoencoder", "Hybrid"])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    X = scaler.transform(df)  # assumes same features
    
    # Isolation Forest
    iso_pred = iso_model.predict(X)
    iso_pred = np.where(iso_pred == 1, 0, 1)
    
    # Autoencoder
    ae_recon = ae_model.predict(X)
    ae_mse = np.mean(np.power(X - ae_recon, 2), axis=1)
    ae_pred = np.where(ae_mse > 0.0096, 1, 0)  # tuned threshold
    
    # Hybrid
    union_pred = np.logical_or(iso_pred, ae_pred).astype(int)
    
    # Choose output
    if model_choice == "Isolation Forest":
        final_pred = iso_pred
    elif model_choice == "Autoencoder":
        final_pred = ae_pred
    else:
        final_pred = union_pred
    
    df['anomaly'] = final_pred
    st.write(df.head())

    st.subheader("📊 Anomaly Distribution")
    st.bar_chart(df['anomaly'].value_counts())

    # Download predictions
    csv = df.to_csv(index=False).encode()
    st.download_button("Download Results", csv, "predictions.csv", "text/csv")

    st.success("✅ Prediction complete!")
