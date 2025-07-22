import streamlit as st
import pandas as pd
import numpy as np
import joblib
from tensorflow.keras.models import load_model
import plotly.express as px

st.set_page_config(
    page_title="Network Traffic Anomaly Detection",
    layout="wide"
)

# ─── Load all artifacts ────────────────────────────────────────────────────────────
@st.cache_resource
def load_artifacts():
    iso = joblib.load("iso_model.pkl")
    # Load autoencoder without recompiling (avoids needing the 'mse' loss fn)
    ae  = load_model("autoencoder_model.h5", compile=False)
    lof = joblib.load("lof_model.pkl")
    scaler = joblib.load("scaler.pkl")
    cols = joblib.load("train_cols.pkl")
    # SHAP importances for the Explain tab
    shap_df = pd.read_csv("iso_shap_importances.csv", index_col=0)
    return iso, ae, lof, scaler, cols, shap_df

iso_model, ae_model, lof_model, scaler, train_cols, iso_shap_imp = load_artifacts()

# ─── Helper predict functions ─────────────────────────────────────────────────────
def predict_iso(X):
    p = iso_model.predict(X)
    return np.where(p == 1, 0, 1)

def predict_ae(X, thresh):
    rec = ae_model.predict(X)
    mse = np.mean((X - rec) ** 2, axis=1)
    preds = np.where(mse > thresh, 1, 0)
    return mse, preds

def predict_lof(X):
    p = lof_model.predict(X)
    return np.where(p == 1, 0, 1)

# ─── Multi-Tab Layout ─────────────────────────────────────────────────────────────
tabs = st.tabs(["🔍 Predict", "📊 EDA", "🧠 Explain"])

# ─── Tab 1: Predict ───────────────────────────────────────────────────────────────
with tabs[0]:
    st.sidebar.header("Model & Threshold Settings")
    model_choice = st.sidebar.selectbox(
        "Choose detection model:",
        [
            "Isolation Forest",
            "Autoencoder",
            "Local Outlier Factor",
            "Hybrid – Union",
            "Hybrid – Intersection"
        ],
    )
    ae_threshold = st.sidebar.slider(
        "Autoencoder threshold", 
        min_value=0.0, max_value=1.0, value=0.02, step=0.005
    )

    st.title("🚨 Network Traffic Anomaly Detection")
    uploaded = st.file_uploader(
        "Upload preprocessed feature CSV", 
        type=["csv"], 
        help="Scaled & one-hot–encoded CSV"
    )
    if uploaded:
        df = pd.read_csv(uploaded)
        # Ensure same column order & fill any missing
        X = df.reindex(columns=train_cols).fillna(0).values

        # Run selected model
        if model_choice == "Isolation Forest":
            preds = predict_iso(X); extra = None

        elif model_choice == "Autoencoder":
            mse, preds = predict_ae(X, ae_threshold)
            extra = mse

        elif model_choice == "Local Outlier Factor":
            preds = predict_lof(X); extra = None

        elif model_choice == "Hybrid – Union":
            iso_p = predict_iso(X)
            _, ae_p = predict_ae(X, ae_threshold)
            preds = np.logical_or(iso_p, ae_p).astype(int)
            extra = None

        else:  # Hybrid – Intersection
            iso_p = predict_iso(X)
            _, ae_p = predict_ae(X, ae_threshold)
            preds = np.logical_and(iso_p, ae_p).astype(int)
            extra = None

        # Attach results & display
        df["anomaly"] = preds
        st.subheader("Results Preview")
        st.dataframe(df.head(10))

        st.subheader("Anomaly Distribution")
        dist = df["anomaly"].map({0: "Normal", 1: "Attack"}).value_counts()
        st.bar_chart(dist)

        # Download button
        csv = df.to_csv(index=False).encode()
        st.download_button(
            "⬇️ Download Full Results",
            data=csv,
            file_name="anomaly_results.csv",
            mime="text/csv"
        )

# ─── Tab 2: EDA ──────────────────────────────────────────────────────────────────
with tabs[1]:
    st.header("📊 Quick Exploratory Data Analysis")
    st.markdown(
        """
        * Use the **Predict** tab to upload your CSV.
        * Here you’ll see basic previews once data is loaded.
        """
    )
    if uploaded:
        st.subheader("Data Sample")
        st.dataframe(df.sample(100))

        st.subheader("Feature Distributions (Attack vs Normal)")
        cols_to_plot = ["count", "srv_count"]
        for col in cols_to_plot:
            fig = px.histogram(
                df, x=col, color="anomaly", barmode="overlay",
                labels={"anomaly": "0=Normal,1=Attack"}, nbins=50
            )
            st.plotly_chart(fig, use_container_width=True)

# ─── Tab 3: Explain ─────────────────────────────────────────────────────────────
with tabs[2]:
    st.header("🧠 Explainability")
    st.markdown("**Global SHAP Importances** for Isolation Forest")
    # Prepare Plotly bar chart
    df_shap = iso_shap_imp.reset_index()
    df_shap.columns = ["feature", "importance"]
    fig = px.bar(
        df_shap,
        x="importance", y="feature",
        orientation="h",
        title="Top-10 SHAP Features – Isolation Forest",
        labels={"importance":"Mean |SHAP value|","feature":""},
        hover_data={"importance":":.3f"}
    )
    fig.update_layout(
        yaxis_categoryorder="total ascending",
        margin=dict(l=120,r=30,t=60,b=30),
        plot_bgcolor="white"
    )
    st.plotly_chart(fig, use_container_width=True)
