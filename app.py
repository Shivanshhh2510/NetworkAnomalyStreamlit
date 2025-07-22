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

# ─── Load Artifacts ───────────────────────────────────────────────────────────────
@st.cache_resource
def load_artifacts():
    iso = joblib.load("iso_model.pkl")
    ae  = load_model("autoencoder_model.h5", compile=False)
    lof = joblib.load("lof_model.pkl")
    scaler = joblib.load("scaler.pkl")
    train_cols = joblib.load("train_cols.pkl")
    shap_df = pd.read_csv("iso_shap_importances.csv", index_col=0)
    return iso, ae, lof, scaler, train_cols, shap_df

iso_model, ae_model, lof_model, scaler, train_cols, iso_shap_imp = load_artifacts()

# ─── Helpers ─────────────────────────────────────────────────────────────────────
@st.cache_data(show_spinner=False, max_entries=1)
def preprocess_raw_kdd(buf, nrows):
    # column names for KDD-Cup data
    cols = [
        "duration","protocol_type","service","flag","src_bytes","dst_bytes","land",
        "wrong_fragment","urgent","hot","num_failed_logins","logged_in","num_compromised",
        "root_shell","su_attempted","num_root","num_file_creations","num_shells",
        "num_access_files","num_outbound_cmds","is_host_login","is_guest_login",
        "count","srv_count","serror_rate","srv_serror_rate","rerror_rate",
        "srv_rerror_rate","same_srv_rate","diff_srv_rate","srv_diff_host_rate",
        "dst_host_count","dst_host_srv_count","dst_host_same_srv_rate",
        "dst_host_diff_srv_rate","dst_host_same_src_port_rate",
        "dst_host_srv_diff_host_rate","dst_host_serror_rate",
        "dst_host_srv_serror_rate","dst_host_rerror_rate","dst_host_srv_rerror_rate",
        "label"
    ]
    df = pd.read_csv(
        buf,
        names=cols,
        nrows=nrows,
        compression='infer'       # ← auto‐detect .gz, .zip, or plain CSV
    )
    df["attack_type"] = (df["label"] != "normal.").astype(int)
    df = df.drop(columns=["label","attack_type","num_outbound_cmds"])
    df = pd.get_dummies(df, columns=["protocol_type","service","flag"])
    for c in train_cols:
        if c not in df.columns:
            df[c] = 0
    return df[train_cols]

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

# ─── UI ───────────────────────────────────────────────────────────────────────────
tabs = st.tabs(["🔍 Predict", "📊 EDA", "🧠 Explain"])

# ─── Tab 1: Predict ───────────────────────────────────────────────────────────────
with tabs[0]:
    st.sidebar.header("Settings")
    upload_type = st.sidebar.radio(
        "Upload type:",
        ("Raw KDD data", "Preprocessed CSV")
    )
    sample_rows = st.sidebar.slider(
        "Rows to sample from raw file",
        min_value=10000, max_value=200000,
        value=50000, step=10000
    )
    iso_cont = st.sidebar.slider("IForest contamination", 0.01, 0.5, 0.1, 0.01)
    lof_cont = st.sidebar.slider("LOF contamination",    0.01, 0.5, 0.02, 0.01)
    ae_thresh= st.sidebar.slider("AE threshold",        0.0, 1.0, 0.02, 0.005)
    model_choice = st.sidebar.selectbox(
        "Model:",
        ["Isolation Forest","Autoencoder","Local Outlier Factor",
         "Hybrid – Union","Hybrid – Intersection"]
    )

    st.title("🚨 Network Traffic Anomaly Detection")
    uploaded = st.file_uploader(
        "Upload your dataset",
        type=["csv","gz","zip"],
        help="Either raw KDD data (.csv/.gz/.zip) or a preprocessed CSV"
    )
    if not uploaded:
        st.info("Please upload your dataset to begin.")
    else:
        # 1) Preprocess
        if upload_type == "Raw KDD data":
            st.warning(f"⚡ Processing first {sample_rows:,} rows of raw KDD upload…")
            df_proc = preprocess_raw_kdd(uploaded, sample_rows)
            X = scaler.transform(df_proc.values)
            df = df_proc.copy()
        else:
            df = pd.read_csv(uploaded)
            X = df.reindex(columns=train_cols).fillna(0).values

        # 2) Optionally refit with new contamination
        iso_model.set_params(contamination=iso_cont)
        iso_model.fit(X)
        lof_model.set_params(contamination=lof_cont)
        lof_model.fit(X)

        # 3) Predict
        if model_choice == "Isolation Forest":
            preds = predict_iso(X)
        elif model_choice == "Autoencoder":
            mse, preds = predict_ae(X, ae_thresh)
        elif model_choice == "Local Outlier Factor":
            preds = predict_lof(X)
        elif model_choice == "Hybrid – Union":
            iso_p = predict_iso(X); _, ae_p = predict_ae(X, ae_thresh)
            preds = np.logical_or(iso_p, ae_p).astype(int)
        else:
            iso_p = predict_iso(X); _, ae_p = predict_ae(X, ae_thresh)
            preds = np.logical_and(iso_p, ae_p).astype(int)

        # 4) Display
        df["anomaly"] = preds
        st.subheader("Sample Results")
        st.dataframe(df.head(10))

        if model_choice in ("Autoencoder","Hybrid – Union","Hybrid – Intersection"):
            rec = ae_model.predict(X)
            feat_err = pd.Series(np.mean((X-rec)**2,axis=0), index=train_cols)
            st.subheader("Top AE Reconstruction-Error Features")
            st.bar_chart(feat_err.nlargest(10))

        st.subheader("Anomaly Distribution")
        dist = df["anomaly"].map({0:"Normal",1:"Attack"}).value_counts()
        st.bar_chart(dist)

        csv = df.to_csv(index=False).encode()
        st.download_button(
            "⬇️ Download Results",
            csv,
            "anomaly_results.csv",
            "text/csv"
        )

# ─── Tab 2: EDA ──────────────────────────────────────────────────────────────────
with tabs[1]:
    st.header("📊 Quick EDA")
    if 'df' in locals():
        st.dataframe(df.sample(50))
        for col in ["count","srv_count"]:
            fig = px.histogram(df, x=col, color="anomaly", barmode="overlay", nbins=50)
            st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("Upload data to see EDA.")

# ─── Tab 3: Explain ─────────────────────────────────────────────────────────────
with tabs[2]:
    st.header("🧠 Explainability")
    st.markdown("**Global SHAP Importances** for Isolation Forest")
    df_shap = iso_shap_imp.reset_index().rename(columns={"index":"feature","0":"importance"})
    fig = px.bar(
        df_shap, x="importance", y="feature", orientation="h",
        title="Top-10 SHAP Features", hover_data={"importance":":.3f"}
    )
    fig.update_layout(
        yaxis_categoryorder="total ascending",
        margin=dict(l=120,r=30,t=60,b=30), plot_bgcolor="white"
    )
    st.plotly_chart(fig, use_container_width=True)
