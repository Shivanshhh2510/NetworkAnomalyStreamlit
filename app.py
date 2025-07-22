import streamlit as st
import pandas as pd
import numpy as np
import joblib
from tensorflow.keras.models import load_model

st.set_page_config(page_title="Network Traffic Anomaly Detector", layout="wide")
st.title("🚨 Network Traffic Anomaly Detection")

st.sidebar.markdown("### Model & Threshold Settings")
model_choice = st.sidebar.selectbox(
    "Choose detection model:",
    ["Isolation Forest", "Autoencoder", "Hybrid – Union", "Hybrid – Intersection"]
)
ae_threshold = st.sidebar.number_input(
    "Autoencoder threshold",
    min_value=0.0, max_value=1.0, value=0.02, step=0.001
)

uploaded = st.file_uploader("Upload preprocessed feature CSV", type=["csv"])
if not uploaded:
    st.info("🔄 Upload your **scaled & one-hot-encoded** CSV to proceed.")
    st.stop()

df = pd.read_csv(uploaded)
X = df.values

@st.cache_resource
def load_models():
    iso = joblib.load("iso_model.pkl")
    ae  = load_model("autoencoder_model.h5")
    return iso, ae

iso_model, ae_model = load_models()

def predict_iso(X):
    p = iso_model.predict(X)
    return np.where(p == 1, 0, 1)

def predict_ae(X, thresh):
    X_rec = ae_model.predict(X)
    mse   = np.mean(np.power(X - X_rec, 2), axis=1)
    return mse, np.where(mse > thresh, 1, 0)

if model_choice == "Isolation Forest":
    preds = predict_iso(X); scores = None

elif model_choice == "Autoencoder":
    scores, preds = predict_ae(X, ae_threshold)

elif model_choice == "Hybrid – Union":
    iso_p = predict_iso(X)
    scores, ae_p = predict_ae(X, ae_threshold)
    preds = np.logical_or(iso_p, ae_p).astype(int)

else:  # Hybrid – Intersection
    iso_p = predict_iso(X)
    scores, ae_p = predict_ae(X, ae_threshold)
    preds = np.logical_and(iso_p, ae_p).astype(int)

out = df.copy()
out["anomaly"] = preds
if scores is not None:
    out["mse_score"] = scores

st.subheader("Preview Results")
st.write(out.head(10))

st.bar_chart(out["anomaly"].map({0:"Normal",1:"Anomaly"}).value_counts())

csv = out.to_csv(index=False).encode("utf-8")
st.download_button("⬇️ Download Results as CSV", csv, "anomaly_results.csv")
