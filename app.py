import os
import streamlit as st
import pandas as pd
import numpy as np
import joblib
from tensorflow.keras.models import load_model
import plotly.express as px
import seaborn as sns
import matplotlib.pyplot as plt

st.set_page_config(page_title="Network Traffic Anomaly Detection", layout="wide")

# ─── Load Artifacts ───────────────────────────────────────────────────────────────
@st.cache_resource
def load_artifacts():
    iso       = joblib.load("iso_model.pkl")
    ae        = load_model("autoencoder_model.h5", compile=False)
    lof       = joblib.load("lof_model.pkl")
    scaler    = joblib.load("scaler.pkl")
    train_cols= joblib.load("train_cols.pkl")
    iso_shap  = pd.read_csv("iso_shap_importances.csv", index_col=0)
    return iso, ae, lof, scaler, train_cols, iso_shap

iso_model, ae_model, lof_model, scaler, train_cols, iso_shap_imp = load_artifacts()

# ─── Feedback Persistence ────────────────────────────────────────────────────────
FEEDBACK_FILE = "feedback.csv"

@st.cache_resource
def load_feedback():
    if os.path.exists(FEEDBACK_FILE):
        return pd.read_csv(FEEDBACK_FILE)
    else:
        return pd.DataFrame(columns=[
            "session","index","model","pred_label","true_label","count","srv_count"
        ])

feedback_df = load_feedback()

def save_feedback(session, idx, model, pred_label, true_label, count, srv_count):
    row = {
        "session": session,
        "index": idx,
        "model": model,
        "pred_label": pred_label,
        "true_label": true_label,
        "count": count,
        "srv_count": srv_count
    }
    df_row = pd.DataFrame([row])
    if os.path.exists(FEEDBACK_FILE):
        df_row.to_csv(FEEDBACK_FILE, mode="a", header=False, index=False)
    else:
        df_row.to_csv(FEEDBACK_FILE, index=False)
    return load_feedback()

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
    raw = pd.read_csv(buf, names=cols, nrows=nrows, compression=comp)
    raw["attack_type"] = (raw["label"] != "normal.").astype(int)
    df = raw.drop(columns=["label","attack_type","num_outbound_cmds"])
    df = pd.get_dummies(df, columns=["protocol_type","service","flag"])
    for c in train_cols:
        if c not in df.columns: df[c] = 0
    return df[train_cols], raw[["count","srv_count"]]

def predict_iso(X):
    p = iso_model.predict(X); return np.where(p==1,0,1)

def predict_ae(X, thresh):
    rec = ae_model.predict(X)
    mse = np.mean((X-rec)**2, axis=1)
    return mse, np.where(mse>thresh,1,0)

def predict_lof(X):
    p = lof_model.predict(X); return np.where(p==1,0,1)

def predict_lof_scores(X):
    return lof_model.decision_function(X)

# ─── UI ───────────────────────────────────────────────────────────────────────────
tabs = st.tabs(["🔍 Predict", "📊 EDA", "🧠 Explain", "🔄 Feedback"])

# ─── Tab 1: Predict ───────────────────────────────────────────────────────────────
with tabs[0]:
    st.sidebar.header("Settings")
    upload_type = st.sidebar.radio("Upload type:", ("Raw KDD data","Preprocessed CSV"))
    sample_rows = st.sidebar.slider("Rows to sample (raw)",10000,200000,50000,10000)
    iso_cont    = st.sidebar.slider("IForest contamination",0.01,0.5,0.1,0.01)
    lof_cont    = st.sidebar.slider("LOF contamination",   0.01,0.5,0.02,0.01)
    ae_thresh   = st.sidebar.slider("AE threshold",       0.0,1.0,0.02,0.005)
    model_choice= st.sidebar.selectbox("Model:",[
        "Isolation Forest","Autoencoder","Local Outlier Factor",
        "Hybrid – Union","Hybrid – Intersection"
    ])

    st.title("🚨 Network Traffic Anomaly Detection")
    uploaded = st.file_uploader(
        "Upload dataset", type=["csv","gz","zip"],
        help="Raw KDD (.csv/.gz/.zip) or preprocessed CSV"
    )
    if not uploaded:
        st.info("Please upload your dataset to begin.")
    else:
        if upload_type=="Raw KDD data":
            st.warning(f"Processing first {sample_rows:,} rows of raw upload…")
            df_proc, raw_counts = preprocess_raw_kdd(uploaded, sample_rows)
            X = scaler.transform(df_proc.values)
            df = df_proc.copy()
            st.session_state["last_counts"] = raw_counts
        else:
            comp = detect_compression(uploaded)
            df = pd.read_csv(uploaded, compression=comp)
            X = df.reindex(columns=train_cols).fillna(0).values
            st.session_state["last_counts"] = df[["count","srv_count"]]

        # Refit & predict
        iso_model.set_params(contamination=iso_cont); iso_model.fit(X)
        lof_model.set_params(contamination=lof_cont); lof_model.fit(X)

        if model_choice=="Isolation Forest":
            preds = predict_iso(X)
        elif model_choice=="Autoencoder":
            mse, preds = predict_ae(X, ae_thresh)
        elif model_choice=="Local Outlier Factor":
            preds = predict_lof(X)
        elif model_choice=="Hybrid – Union":
            iso_p = predict_iso(X); _, ae_p = predict_ae(X,ae_thresh)
            preds = np.logical_or(iso_p,ae_p).astype(int)
        else:
            iso_p = predict_iso(X); _, ae_p = predict_ae(X,ae_thresh)
            preds = np.logical_and(iso_p,ae_p).astype(int)

        df["anomaly"] = preds
        st.session_state["last_df"] = df
        st.session_state["last_model"] = model_choice

        st.subheader("Sample Results")
        st.dataframe(df.head(10))

        if model_choice in ("Autoencoder","Hybrid – Union","Hybrid – Intersection"):
            rec = ae_model.predict(X)
            feat_err = pd.Series(np.mean((X-rec)**2,axis=0), index=train_cols)
            st.subheader("Top 10 AE Reconstruction-Error Features")
            st.write("Features the autoencoder struggles with most.")
            st.bar_chart(feat_err.nlargest(10))

        st.subheader("Anomaly Distribution")
        st.write("Count of normal vs. attack instances in this sample.")
        st.bar_chart(df["anomaly"].map({0:"Normal",1:"Attack"}).value_counts())

        csv = df.to_csv(index=False).encode()
        st.download_button("⬇️ Download Results", csv, "results.csv", "text/csv")

# ─── Tab 2: EDA ──────────────────────────────────────────────────────────────────
with tabs[1]:
    st.header("📊 Exploratory Data Analysis")
    if "last_df" not in st.session_state:
        st.info("Upload & predict to see EDA.")
    else:
        df = st.session_state["last_df"]
        raw_counts = st.session_state["last_counts"]

        if upload_type=="Raw KDD data":
            st.subheader("Protocol Type Breakdown")
            st.write("Which network protocols appear more often in anomalies vs normal.")
            proto_df = raw_counts.copy()
            proto_df["anomaly"] = df["anomaly"].map({0:"Normal",1:"Attack"})
            fig1 = px.bar(proto_df, x="protocol_type", color="anomaly", barmode="group",
                          labels={"anomaly":"0=Normal,1=Attack"})
            st.plotly_chart(fig1, use_container_width=True)

        st.subheader("Feature Correlation Heatmap")
        st.write("How key numeric features correlate with each other.")
        num_cols = ["duration","src_bytes","dst_bytes","count","srv_count"]
        corr = df[num_cols].corr()
        fig2, ax = plt.subplots(figsize=(6,5))
        sns.heatmap(corr, annot=True, cmap="vlag", ax=ax)
        st.pyplot(fig2)

        st.subheader("Data Distributions")
        st.write("Boxplots compare src_bytes and dst_bytes for anomalies vs normal.")
        fig3, axes = plt.subplots(1,2, figsize=(12,4))
        sns.boxplot(x=df["anomaly"], y=df["src_bytes"], ax=axes[0])
        axes[0].set_title("src_bytes")
        sns.boxplot(x=df["anomaly"], y=df["dst_bytes"], ax=axes[1])
        axes[1].set_title("dst_bytes")
        st.pyplot(fig3)

# ─── Tab 3: Explain ─────────────────────────────────────────────────────────────
with tabs[2]:
    st.header("🧠 Explainability")
    choice = st.selectbox("Explain model:", [
        "Isolation Forest","Autoencoder","Local Outlier Factor"
    ])
    if choice=="Isolation Forest":
        st.write("Global SHAP importances for Isolation Forest decisions.")
        df_shap = iso_shap_imp.reset_index().rename(
            columns={"index":"feature",0:"importance"}
        )
        fig = px.bar(df_shap, x="importance", y="feature", orientation="h",
                     labels={"importance":"Mean |SHAP value|"})
        fig.update_layout(yaxis_categoryorder="total ascending",
                          plot_bgcolor="white")
        st.plotly_chart(fig, use_container_width=True)

    elif choice=="Autoencoder":
        st.write("Top features by autoencoder reconstruction error.")
        df = st.session_state["last_df"]
        rec = ae_model.predict(scaler.transform(df[train_cols].values))
        feat_err = pd.Series(np.mean((scaler.transform(df[train_cols].values)-rec)**2,axis=0),
                             index=train_cols)
        df_err = feat_err.nlargest(10).reset_index().rename(
            columns={"index":"feature",0:"error"}
        )
        fig = px.bar(df_err, x="error", y="feature", orientation="h",
                     labels={"error":"Reconstruction MSE"})
        fig.update_layout(yaxis_categoryorder="total ascending")
        st.plotly_chart(fig, use_container_width=True)

    else:
        st.write("LOF ‘normality’ score distribution (lower = more anomalous).")
        df = st.session_state["last_df"]
        scores = predict_lof_scores(scaler.transform(df[train_cols].values))
        fig = px.histogram(scores, nbins=50, labels={"value":"LOF score"})
        st.plotly_chart(fig, use_container_width=True)

# ─── Tab 4: Feedback & Drift ─────────────────────────────────────────────────────
with tabs[3]:
    st.header("🔄 Feedback & Drift Monitoring")
    if "last_df" not in st.session_state:
        st.info("Run a prediction first to collect feedback.")
    else:
        df_last   = st.session_state["last_df"]
        raw_counts= st.session_state["last_counts"]
        model_last= st.session_state["last_model"]

        st.subheader("Label a Sample")
        idx = st.number_input("Row index", min_value=0, max_value=len(df_last)-1, step=1)
        true_label = st.radio("True Label:", ["Normal","Attack"])
        if st.button("Submit Feedback"):
            pred_label = "Attack" if df_last.loc[idx,"anomaly"]==1 else "Normal"
            session_id = int(pd.Timestamp.now().timestamp())
            feedback = save_feedback(
                session_id,
                idx,
                model_last,
                pred_label,
                true_label,
                int(raw_counts.loc[idx,"count"]),
                int(raw_counts.loc[idx,"srv_count"])
            )
            st.success("Feedback saved!")
        st.subheader("Collected Feedback")
        feedback = load_feedback()
        st.dataframe(feedback)

        if not feedback.empty:
            st.subheader("Drift on 'count'")
            fig_f = px.histogram(
                feedback,
                x="count",
                color="true_label",
                barmode="overlay",
                labels={"true_label":"True Label"}
            )
            st.plotly_chart(fig_f, use_container_width=True)
