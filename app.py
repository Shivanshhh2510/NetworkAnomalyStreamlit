import streamlit as st
import pandas as pd
import numpy as np
import joblib
from tensorflow.keras.models import load_model
import plotly.express as px
import seaborn as sns
import matplotlib.pyplot as plt

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
    iso_shap = pd.read_csv("iso_shap_importances.csv", index_col=0)
    return iso, ae, lof, scaler, train_cols, iso_shap

iso_model, ae_model, lof_model, scaler, train_cols, iso_shap_imp = load_artifacts()

# ─── Helpers ─────────────────────────────────────────────────────────────────────
def detect_compression(buf):
    name = getattr(buf, "name", "").lower()
    if name.endswith((".gz", ".gzip")): return "gzip"
    if name.endswith(".zip"):            return "zip"
    return None

@st.cache_data(show_spinner=False, max_entries=1)
def preprocess_raw_kdd(buf, nrows):
    cols = [  # KDD feature names + label
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
        if c not in df.columns: df[c] = 0
    return df[train_cols]

def predict_iso(X):
    p = iso_model.predict(X); return np.where(p==1,0,1)

def predict_ae(X, thresh):
    rec = ae_model.predict(X)
    mse = np.mean((X-rec)**2, axis=1)
    return mse, np.where(mse>thresh,1,0)

def predict_lof_scores(X):
    # LOF uses decision_function for outlier "normality" score
    return lof_model.decision_function(X)

def predict_lof(X):
    p = lof_model.predict(X); return np.where(p==1,0,1)

# ─── UI ───────────────────────────────────────────────────────────────────────────
tabs = st.tabs(["🔍 Predict", "📊 EDA", "🧠 Explain"])

# ─── Tab 1: Predict ───────────────────────────────────────────────────────────────
with tabs[0]:
    st.sidebar.header("Settings")
    upload_type = st.sidebar.radio("Upload type:",("Raw KDD data","Preprocessed CSV"))
    sample_rows = st.sidebar.slider("Rows to sample (raw)",10000,200000,50000,10000)
    iso_cont = st.sidebar.slider("IForest contamination",0.01,0.5,0.1,0.01)
    lof_cont = st.sidebar.slider("LOF contamination",0.01,0.5,0.02,0.01)
    ae_thresh= st.sidebar.slider("AE threshold",0.0,1.0,0.02,0.005)
    model_choice = st.sidebar.selectbox("Model:",[
        "Isolation Forest","Autoencoder","Local Outlier Factor","Hybrid – Union","Hybrid – Intersection"
    ])

    st.title("🚨 Network Traffic Anomaly Detection")
    uploaded = st.file_uploader(
        "Upload dataset", type=["csv","gz","zip"],
        help="Raw KDD (.csv/.gz/.zip) or preprocessed CSV"
    )
    if not uploaded:
        st.info("Please upload your dataset to begin.")
    else:
        # Preprocess
        if upload_type=="Raw KDD data":
            st.warning(f"Processing first {sample_rows:,} rows of raw upload…")
            df_proc = preprocess_raw_kdd(uploaded, sample_rows)
            X = scaler.transform(df_proc.values); df = df_proc.copy()
        else:
            comp = detect_compression(uploaded)
            df = pd.read_csv(uploaded, compression=comp)
            X = df.reindex(columns=train_cols).fillna(0).values

        # Refit models
        iso_model.set_params(contamination=iso_cont); iso_model.fit(X)
        lof_model.set_params(contamination=lof_cont); lof_model.fit(X)

        # Predict
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

        # Display
        df["anomaly"] = preds
        st.subheader("Sample of Results")
        st.dataframe(df.head(10))

        if model_choice in ("Autoencoder","Hybrid – Union","Hybrid – Intersection"):
            rec = ae_model.predict(X)
            feat_err = pd.Series(np.mean((X-rec)**2,axis=0), index=train_cols)
            st.subheader("Top 10 AE Reconstruction-Error Features")
            st.write("Features the autoencoder struggles with most:")
            st.bar_chart(feat_err.nlargest(10))

        st.subheader("Anomaly Distribution")
        st.write("Count of normal vs. attack instances in this sample")
        st.bar_chart(df["anomaly"].map({0:"Normal",1:"Attack"}).value_counts())

        csv = df.to_csv(index=False).encode()
        st.download_button("⬇️ Download Results",csv,"results.csv","text/csv")

# ─── Tab 2: EDA ──────────────────────────────────────────────────────────────────
with tabs[1]:
    st.header("📊 Exploratory Data Analysis")
    if 'df' not in locals():
        st.info("Upload data to see EDA.")
    else:
        st.subheader("Protocol Type Breakdown")
        st.write("Shows which network protocols (TCP/UDP/ICMP) are more common in anomalies vs normal:")
        fig1 = px.bar(df, x="protocol_type", color="anomaly", barmode="group")
        st.plotly_chart(fig1, use_container_width=True)

        st.subheader("Feature Correlation Heatmap")
        st.write("How numeric features relate—strong correlations indicate linked behaviors.")
        num_cols = ["duration","src_bytes","dst_bytes","count","srv_count"]
        corr = df[num_cols].corr()
        fig2, ax = plt.subplots(figsize=(6,5))
        sns.heatmap(corr, annot=True, cmap="vlag", ax=ax)
        st.pyplot(fig2)

        st.subheader("Data Distributions")
        st.write("Boxplots compare distributions of src_bytes and dst_bytes for anomalies vs. normal.")
        fig3, ax = plt.subplots(1,2, figsize=(12,4))
        sns.boxplot(x=df["anomaly"], y=df["src_bytes"], ax=ax[0])
        ax[0].set_title("src_bytes")
        sns.boxplot(x=df["anomaly"], y=df["dst_bytes"], ax=ax[1])
        ax[1].set_title("dst_bytes")
        st.pyplot(fig3)

# ─── Tab 3: Explain ─────────────────────────────────────────────────────────────
with tabs[2]:
    st.header("🧠 Explainability")
    model_explain = st.selectbox("Explain model:",[
        "Isolation Forest","Autoencoder","Local Outlier Factor"
    ])
    if model_explain=="Isolation Forest":
        st.write("Global SHAP importances show which features drive the forest’s anomaly decisions:")
        df_shap = iso_shap_imp.reset_index().rename(columns={"index":"feature","importance":"importance"})
        fig = px.bar(df_shap, x="importance", y="feature", orientation="h",
                     labels={"importance":"Mean |SHAP value|"})
        fig.update_layout(yaxis_categoryorder="total ascending",plot_bgcolor="white")
        st.plotly_chart(fig, use_container_width=True)

    elif model_explain=="Autoencoder":
        st.write("Features with highest reconstruction error—autoencoder’s blind spots:")
        rec = ae_model.predict(X)
        feat_err = pd.Series(np.mean((X-rec)**2,axis=0), index=train_cols)
        fig = px.bar(
            feat_err.nlargest(10).reset_index().rename(columns={"index":"feature",0:"error"}),
            x="error", y="feature", orientation="h",
            labels={"error":"Reconstruction MSE"}
        )
        fig.update_layout(yaxis_categoryorder="total ascending")
        st.plotly_chart(fig, use_container_width=True)

    else:  # LOF
        st.write("Histogram of LOF ‘normality’ scores—lower = more anomalous:")
        scores = predict_lof_scores(X)
        fig = px.histogram(scores, nbins=50, labels={"value":"LOF score"})
        st.plotly_chart(fig, use_container_width=True)
