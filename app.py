import os
import time
import streamlit as st
import pandas as pd
import numpy as np
import joblib
from tensorflow.keras.models import load_model
import plotly.express as px
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

# ─── Dark Theme & Layout Tweaks ─────────────────────────────────────────────────
st.markdown(
    """
    <style>
      /* Hide Streamlit header & footer */
      header, footer {visibility: hidden;}
      
      /* Sidebar width */
      div[data-testid="stSidebar"] {width: 260px;}
      
      /* Dark backgrounds and card styling */
      .block-container, .stTabs, .stButton, .stSelectbox, .stSlider, 
      .stFileUploader, .stDownloadButton {
        background-color: #0e1117 !important;
        color: #e1e1e1 !important;
      }
      .stMetric, .stDataFrame, .stBarChart, .stPlotlyChart {
        background-color: #1e1e1e !important;
        border-radius: 12px !important;
        box-shadow: 0 4px 8px rgba(0,0,0,0.5);
      }
    </style>
    """,
    unsafe_allow_html=True,
)

st.set_page_config(
    page_title="🚨 Network Traffic Anomaly Detection",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ─── Load Artifacts ───────────────────────────────────────────────────────────────
@st.cache_resource(show_spinner=False)
def load_artifacts():
    with st.spinner("Loading models & artifacts..."):
        iso = joblib.load("iso_model.pkl")
        ae  = load_model("autoencoder_model.h5", compile=False)
        lof = joblib.load("lof_model.pkl")
        scaler = joblib.load("scaler.pkl")
        train_cols = joblib.load("train_cols.pkl")
        shap_df = pd.read_csv("iso_shap_importances.csv", index_col=0)
    return iso, ae, lof, scaler, train_cols, shap_df

iso_model, ae_model, lof_model, scaler, train_cols, iso_shap_imp = load_artifacts()

# ─── Helpers ─────────────────────────────────────────────────────────────────────
def detect_compression(buf):
    name = getattr(buf, "name", "").lower()
    if name.endswith((".gz","gzip")): return "gzip"
    if name.endswith(".zip"):          return "zip"
    return None

@st.cache_data(show_spinner=False, max_entries=1)
def preprocess_raw_kdd(buf, nrows):
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
    comp = detect_compression(buf)
    df = pd.read_csv(buf, names=cols, nrows=nrows, compression=comp)
    df["attack_type"] = (df["label"] != "normal.").astype(int)
    df = df.drop(columns=["label","attack_type","num_outbound_cmds"])
    df = pd.get_dummies(df, columns=["protocol_type","service","flag"])
    for c in train_cols:
        if c not in df.columns:
            df[c] = 0
    return df[train_cols]

def predict_iso(X):    return np.where(iso_model.predict(X)==1, 0, 1)
def predict_ae(X, t):  rec = ae_model.predict(X); mse = np.mean((X-rec)**2,axis=1); return mse, np.where(mse>t,1,0)
def predict_lof(X):    return np.where(lof_model.predict(X)==1, 0, 1)
def lof_scores(X):     return lof_model.decision_function(X)

# ─── Top Header & Metrics ────────────────────────────────────────────────────────
header_col, desc_col = st.columns([2,3])
with header_col:
    st.title("🚨 Network Traffic Anomaly Detection")
    st.markdown("A real-time, interactive anomaly detection dashboard built with Isolation Forest, Autoencoder & LOF.")
with desc_col:
    st.write("""
      **Features**  
      - Upload raw KDD `.csv`/`.gz`/`.zip` or preprocessed CSV  
      - Tune contamination & threshold per model  
      - Hybrid union/intersection logic  
      - PCA embedding (2D/3D) & SHAP explainability  
      - Sleek dark theme & live spinners/metrics
    """)

# ─── Main Tabs ──────────────────────────────────────────────────────────────────
tabs = st.tabs(["🔍 Predict","📊 EDA","🧠 Explain","🔬 Embedding"])

# ─── Tab 1: Predict ───────────────────────────────────────────────────────────────
with tabs[0]:
    st.sidebar.header("Settings")
    upload_type  = st.sidebar.radio("Upload type:", ("Raw KDD data","Preprocessed CSV"))
    sample_rows  = st.sidebar.slider("Rows to sample (raw)",10000,200000,50000,10000)
    iso_cont     = st.sidebar.slider("IForest contamination",0.01,0.5,0.1,0.01)
    lof_cont     = st.sidebar.slider("LOF contamination",   0.01,0.5,0.02,0.01)
    ae_thresh    = st.sidebar.slider("AE threshold",       0.0,1.0,0.02,0.005)
    model_choice = st.sidebar.selectbox("Model:",[
        "Isolation Forest","Autoencoder","Local Outlier Factor",
        "Hybrid – Union","Hybrid – Intersection"
    ])

    uploaded = st.file_uploader("Upload dataset",type=["csv","gz","zip"])
    if uploaded:
        with st.spinner("Preprocessing data..."):
            if upload_type=="Raw KDD data":
                df_proc = preprocess_raw_kdd(uploaded, sample_rows)
                X = scaler.transform(df_proc.values); df = df_proc.copy()
            else:
                comp = detect_compression(uploaded)
                df = pd.read_csv(uploaded, compression=comp)
                X = df.reindex(columns=train_cols).fillna(0).values

        # refit
        iso_model.set_params(contamination=iso_cont); iso_model.fit(X)
        lof_model.set_params(contamination=lof_cont); lof_model.fit(X)

        # predict
        with st.spinner("Running anomaly detection..."):
            if model_choice=="Isolation Forest":
                preds = predict_iso(X)
            elif model_choice=="Autoencoder":
                mse, preds = predict_ae(X, ae_thresh)
            elif model_choice=="Local Outlier Factor":
                preds = predict_lof(X)
            elif model_choice=="Hybrid – Union":
                iso_p = predict_iso(X); _, ae_p = predict_ae(X, ae_thresh)
                preds = np.logical_or(iso_p,ae_p).astype(int)
            else:
                iso_p = predict_iso(X); _, ae_p = predict_ae(X, ae_thresh)
                preds = np.logical_and(iso_p,ae_p).astype(int)

        df["anomaly"] = preds
        # Metrics
        total = len(df); attacks = int(df["anomaly"].sum())
        m1, m2, m3 = st.columns(3)
        m1.metric("Total Samples", f"{total:,}")
        m2.metric("Detected Attacks", f"{attacks:,}")
        m3.metric("Current Model", model_choice)

        # Show table & charts
        st.subheader("Sample Results")
        st.dataframe(df.head(10), use_container_width=True)

        if model_choice in ("Autoencoder","Hybrid – Union","Hybrid – Intersection"):
            rec       = ae_model.predict(X)
            feat_err  = pd.Series(np.mean((X-rec)**2,axis=0), index=train_cols)
            st.subheader("Top AE Reconstruction Errors")
            st.bar_chart(feat_err.nlargest(10))

        st.subheader("Anomaly Distribution")
        dist = df["anomaly"].map({0:"Normal",1:"Attack"}).value_counts()
        st.bar_chart(dist)

        csv = df.to_csv(index=False).encode()
        st.download_button("⬇️ Download Results", csv, "anomaly_results.csv", "text/csv")

    else:
        st.info("Please upload your dataset to begin.")

# ─── Tab 2: EDA ──────────────────────────────────────────────────────────────────
with tabs[1]:
    st.header("📊 Exploratory Data Analysis")
    if "anomaly" in locals() or "df" in globals():
        df_eda = df.copy()
        st.subheader("Protocol vs Anomaly")
        fig = px.histogram(df_eda, x="protocol_type", color="anomaly", barmode="group")
        st.plotly_chart(fig, use_container_width=True)

        st.subheader("Feature Correlations")
        num_cols = ["duration","src_bytes","dst_bytes","count","srv_count"]
        corr = df_eda[num_cols].corr()
        fig2, ax2 = plt.subplots(figsize=(6,5))
        sns.heatmap(corr, annot=True, cmap="vlag", ax=ax2)
        st.pyplot(fig2)

        st.subheader("Byte Distributions by Class")
        fig3, axes3 = plt.subplots(1,2,figsize=(12,4))
        sns.boxplot(x=df_eda["anomaly"], y=df_eda["src_bytes"], ax=axes3[0])
        sns.boxplot(x=df_eda["anomaly"], y=df_eda["dst_bytes"], ax=axes3[1])
        st.pyplot(fig3)
    else:
        st.info("Run a prediction first to see EDA.")

# ─── Tab 3: Explain ─────────────────────────────────────────────────────────────
with tabs[2]:
    st.header("🧠 Explainability")
    choice = st.selectbox("Choose model:",[
      "Isolation Forest","Autoencoder","Local Outlier Factor"
    ])
    if choice=="Isolation Forest":
        st.write("Mean |SHAP value| per feature")
        df_shap = iso_shap_imp.reset_index().rename(columns={"index":"feature",0:"importance"})
        fig = px.bar(df_shap, x="importance", y="feature", orientation="h")
        st.plotly_chart(fig, use_container_width=True)

    elif choice=="Autoencoder":
        st.write("Top AE Reconstruction Errors")
        df_tmp = df.copy()
        Xp = scaler.transform(df_tmp[train_cols].values)
        rec, errs = ae_model.predict(Xp), np.mean((Xp - ae_model.predict(Xp))**2,axis=1)
        feat_err = pd.Series(np.mean((Xp-rec)**2,axis=0), index=train_cols)
        top = feat_err.nlargest(10).reset_index().rename(columns={"index":"feature",0:"error"})
        fig = px.bar(top, x="error", y="feature", orientation="h")
        st.plotly_chart(fig, use_container_width=True)

    else:
        st.write("LOF Score Distribution (lower→more anomalous)")
        Xp = scaler.transform(df[train_cols].values)
        scores = lof_scores(Xp)
        fig = px.histogram(scores, nbins=50, labels={"value":"LOF score"})
        st.plotly_chart(fig, use_container_width=True)

# ─── Tab 4: Embedding ────────────────────────────────────────────────────────────
with tabs[3]:
    st.header("🔬 PCA Embedding (2D/3D)")
    if "anomaly" in locals():
        Xp = scaler.transform(df[train_cols].values)
        n = min(len(Xp), 5000)
        idx = np.random.choice(len(Xp), n, replace=False)
        sample_df = df.iloc[idx].copy()
        samp_X    = Xp[idx]
        dim = st.radio("Dimension:",("2D","3D"))
        if dim=="2D":
            coords = PCA(2).fit_transform(samp_X)
            sample_df["PC1"], sample_df["PC2"] = coords[:,0], coords[:,1]
            fig = px.scatter(sample_df, x="PC1", y="PC2",
                             color=sample_df["anomaly"].map({0:"Normal",1:"Attack"}))
        else:
            coords = PCA(3).fit_transform(samp_X)
            sample_df["PC1"], sample_df["PC2"], sample_df["PC3"] = coords[:,0],coords[:,1],coords[:,2]
            fig = px.scatter_3d(sample_df, x="PC1", y="PC2", z="PC3",
                                color=sample_df["anomaly"].map({0:"Normal",1:"Attack"}))
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("Run a prediction first to see embedding.")
