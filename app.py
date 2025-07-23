import os
import json
import datetime
import requests
import streamlit as st
import pandas as pd
import numpy as np
import joblib
from tensorflow.keras.models import load_model
import plotly.express as px
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.metrics import (
    precision_recall_curve,
    roc_curve,
    auc as calc_auc,
    precision_score,
    recall_score,
    f1_score
)

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

# ─── Incident Logging ────────────────────────────────────────────────────────────
INCIDENT_FILE = "incidents.json"
def load_incidents():
    if os.path.exists(INCIDENT_FILE):
        with open(INCIDENT_FILE, "r") as f:
            return json.load(f)
    return []

def log_incident(model, count, top_samples):
    incident = {
        "timestamp": datetime.datetime.utcnow().isoformat()+"Z",
        "model": model,
        "anomaly_count": int(count),
        "top_samples": top_samples.to_dict(orient="records")
    }
    incidents = load_incidents()
    incidents.append(incident)
    with open(INCIDENT_FILE, "w") as f:
        json.dump(incidents, f, indent=2)
    return incident

def send_slack_alert(webhook_url, incident):
    text = (
        f"*Anomaly Alert* ({incident['model']})\n"
        f"> Time: {incident['timestamp']}\n"
        f"> Count: {incident['anomaly_count']}\n"
        f"> Top sample: {incident['top_samples'][0]}"
    )
    try:
        requests.post(webhook_url, json={"text": text}, timeout=3)
    except Exception:
        pass

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
    raw_meta = raw[["protocol_type","count","srv_count"]].copy()
    df = raw.drop(columns=["label","attack_type","num_outbound_cmds"])
    df = pd.get_dummies(df, columns=["protocol_type","service","flag"])
    for c in train_cols:
        if c not in df.columns:
            df[c] = 0
    return df[train_cols], raw_meta

def predict_iso(X, contamination=None):
    if contamination is not None:
        iso_model.set_params(contamination=contamination)
        iso_model.fit(X)
    p = iso_model.predict(X)
    return np.where(p == 1, 0, 1), iso_model.decision_function(X) * -1

def predict_ae(X, thresh=None):
    if thresh is None:
        thresh = st.session_state.get("ae_thresh", 0.02)
    rec = ae_model.predict(X)
    mse = np.mean((X - rec) ** 2, axis=1)
    return np.where(mse > thresh, 1, 0), mse

def predict_lof(X, contamination=None, n_neighbors=None):
    if contamination is not None or n_neighbors is not None:
        lof_model.set_params(
            contamination=contamination or lof_model.contamination,
            n_neighbors=n_neighbors or lof_model.n_neighbors
        )
        lof_model.fit(X)
    p = lof_model.predict(X)
    return np.where(p == 1, 0, 1), lof_model.decision_function(X) * -1

# ─── UI ───────────────────────────────────────────────────────────────────────────
tabs = st.tabs([
    "🔍 Predict",
    "📊 EDA",
    "🧠 Explain",
    "🔬 Embedding",
    "🚨 Incidents",
    "🛠️ Tuning Lab"
])

# ─── Tab 1: Predict ───────────────────────────────────────────────────────────────
with tabs[0]:
    st.sidebar.header("Settings")
    upload_type  = st.sidebar.radio("Upload type:", ("Raw KDD data","Preprocessed CSV"))
    sample_rows  = st.sidebar.slider("Rows to sample (raw)",10000,200000,50000,10000)
    iso_cont     = st.sidebar.slider("IForest contamination",0.01,0.5,0.1,0.01)
    lof_cont     = st.sidebar.slider("LOF contamination",   0.01,0.5,0.02,0.01)
    ae_thresh    = st.sidebar.slider("AE threshold",       0.0,1.0,0.02,0.005)
    alert_thresh = st.sidebar.number_input("Alert if ≥ anomalies", 1, 1000, 50, 1)
    slack_url    = st.sidebar.text_input("Slack Webhook URL", type="password")
    model_choice = st.sidebar.selectbox("Model:",[
        "Isolation Forest","Autoencoder","Local Outlier Factor",
        "Hybrid – Union","Hybrid – Intersection"
    ])
    st.session_state["ae_thresh"] = ae_thresh

    st.title("🚨 Network Traffic Anomaly Detection")
    uploaded = st.file_uploader(
        "Upload dataset", type=["csv","gz","zip"],
        help="Raw KDD (.csv/.gz/.zip) or preprocessed CSV"
    )
    if not uploaded:
        st.info("Please upload your dataset to begin.")
    else:
        # preprocess
        if upload_type=="Raw KDD data":
            st.warning(f"Processing first {sample_rows:,} rows…")
            df_proc, raw_meta = preprocess_raw_kdd(uploaded, sample_rows)
            X = scaler.transform(df_proc.values)
            df = df_proc.copy()
            st.session_state["last_meta"] = raw_meta
        else:
            comp = detect_compression(uploaded)
            df = pd.read_csv(uploaded, compression=comp)
            X = df.reindex(columns=train_cols).fillna(0).values
            st.session_state["last_meta"] = None

        # predict
        if model_choice=="Isolation Forest":
            preds, scores = predict_iso(X, contamination=iso_cont)
        elif model_choice=="Autoencoder":
            preds, scores = predict_ae(X, thresh=ae_thresh)
        elif model_choice=="Local Outlier Factor":
            preds, scores = predict_lof(X, contamination=lof_cont)
        elif model_choice=="Hybrid – Union":
            iso_p, iso_s = predict_iso(X, contamination=iso_cont)
            ae_p, ae_s   = predict_ae(X, thresh=ae_thresh)
            preds        = np.logical_or(iso_p, ae_p).astype(int)
            scores       = iso_s
        else:
            iso_p, iso_s = predict_iso(X, contamination=iso_cont)
            ae_p, ae_s   = predict_ae(X, thresh=ae_thresh)
            preds        = np.logical_and(iso_p, ae_p).astype(int)
            scores       = iso_s

        df["anomaly"] = preds
        st.session_state["last_df"]    = df
        st.session_state["last_model"] = model_choice
        st.session_state["last_scores"]= scores

        # display
        st.subheader("Sample Results")
        st.dataframe(df.head(10), use_container_width=True)

        if model_choice in ("Autoencoder","Hybrid – Union","Hybrid – Intersection"):
            st.subheader("Top AE Reconstruction Errors")
            rec      = ae_model.predict(X)
            feat_err = pd.Series(np.mean((X-rec)**2,axis=0), index=train_cols)
            st.bar_chart(feat_err.nlargest(10))

        st.subheader("Anomaly Distribution")
        st.bar_chart(df["anomaly"].map({0:"Normal",1:"Attack"}).value_counts())

        csv = df.to_csv(index=False).encode()
        st.download_button("⬇️ Download Results", csv, "results.csv", "text/csv")

        # check for alert
        total_anoms = int(preds.sum())
        if total_anoms >= alert_thresh:
            incident = log_incident(model_choice, total_anoms, df[preds==1].head(5))
            st.error(f"🚨 Alert: {total_anoms} anomalies detected by {model_choice}")
            if slack_url:
                send_slack_alert(slack_url, incident)

# ─── Tab 2: EDA ──────────────────────────────────────────────────────────────────
with tabs[1]:
    st.header("📊 Exploratory Data Analysis")
    if "last_df" not in st.session_state:
        st.info("Upload & predict to see EDA.")
    else:
        df       = st.session_state["last_df"]
        raw_meta = st.session_state["last_meta"]

        if raw_meta is not None:
            st.subheader("Protocol Breakdown")
            proto = raw_meta.copy()
            proto["anomaly"] = df["anomaly"].map({0:"Normal",1:"Attack"})
            fig1 = px.bar(proto, x="protocol_type", color="anomaly", barmode="group")
            st.plotly_chart(fig1, use_container_width=True)

        st.subheader("Numeric Correlations")
        num_cols = ["duration","src_bytes","dst_bytes","count","srv_count"]
        corr = df[num_cols].corr()
        fig2, ax = plt.subplots(figsize=(6,5))
        sns.heatmap(corr, annot=True, cmap="vlag", ax=ax)
        st.pyplot(fig2)

        st.subheader("Data Distributions")
        fig3, axes = plt.subplots(1,2,figsize=(12,4))
        sns.boxplot(x=df["anomaly"], y=df["src_bytes"], ax=axes[0])
        axes[0].set_title("src_bytes")
        sns.boxplot(x=df["anomaly"], y=df["dst_bytes"], ax=axes[1])
        axes[1].set_title("dst_bytes")
        st.pyplot(fig3)

# ─── Tab 3: Explain ─────────────────────────────────────────────────────────────
with tabs[2]:
    st.header("🧠 Explainability")
    choice = st.selectbox("Explain model:",[
        "Isolation Forest","Autoencoder","Local Outlier Factor"
    ])
    if choice=="Isolation Forest":
        st.write("Global SHAP importances for IForest.")
        shap_df = iso_shap_imp.reset_index().rename(columns={"index":"feature",0:"importance"})
        fig = px.bar(shap_df, x="importance", y="feature", orientation="h")
        fig.update_layout(yaxis_categoryorder="total ascending",plot_bgcolor="white")
        st.plotly_chart(fig, use_container_width=True)
    elif choice=="Autoencoder":
        st.write("Top AE reconstruction-error features.")
        df = st.session_state["last_df"]
        X  = scaler.transform(df[train_cols].values)
        rec= ae_model.predict(X)
        errs = pd.Series(np.mean((X-rec)**2,axis=0), index=train_cols)
        top = errs.nlargest(10).reset_index().rename(columns={"index":"feature",0:"error"})
        fig = px.bar(top, x="error", y="feature", orientation="h")
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.write("LOF score distribution (lower=more anomalous).")
        df     = st.session_state["last_df"]
        scores = st.session_state["last_scores"]
        fig    = px.histogram(scores, nbins=50, labels={"value":"LOF score"})
        st.plotly_chart(fig, use_container_width=True)

# ─── Tab 4: Embedding ────────────────────────────────────────────────────────────
with tabs[3]:
    st.header("🔬 PCA Embedding of Network Traffic")
    if "last_df" not in st.session_state:
        st.info("Upload & predict to see embedding.")
    else:
        df = st.session_state["last_df"]
        X  = scaler.transform(df[train_cols].values)
        n  = min(len(X), 5000)
        idxs = np.random.choice(len(X), n, replace=False)
        Xs = X[idxs]
        dfs= df.iloc[idxs].copy()

        dim = st.radio("Projection dimension:", ("2D","3D"))
        if dim=="2D":
            pca = PCA(n_components=2)
            coords = pca.fit_transform(Xs)
            dfs["PC1"], dfs["PC2"] = coords[:,0], coords[:,1]
            fig = px.scatter(dfs, x="PC1", y="PC2",
                             color=dfs["anomaly"].map({0:"Normal",1:"Attack"}),
                             title="PCA (2D)")
        else:
            pca = PCA(n_components=3)
            coords = pca.fit_transform(Xs)
            dfs["PC1"], dfs["PC2"], dfs["PC3"] = coords[:,0], coords[:,1], coords[:,2]
            fig = px.scatter_3d(dfs, x="PC1", y="PC2", z="PC3",
                                color=dfs["anomaly"].map({0:"Normal",1:"Attack"}),
                                title="PCA (3D)")
        st.plotly_chart(fig, use_container_width=True)

# ─── Tab 5: Incidents ────────────────────────────────────────────────────────────
with tabs[4]:
    st.header("🚨 Incident Dashboard")
    incidents = load_incidents()
    if not incidents:
        st.info("No incidents logged yet.")
    else:
        df_inc = pd.DataFrame(incidents)
        st.dataframe(df_inc[["timestamp","model","anomaly_count"]])
        for i,row in df_inc.iterrows():
            st.markdown(f"**Incident {i+1}** — {row['timestamp']} — *{row['model']}* detected {row['anomaly_count']} anomalies")
            st.table(row["top_samples"])

# ─── Tab 6: Tuning Lab ───────────────────────────────────────────────────────────
with tabs[5]:
    st.header("🛠️ Hyperparameter Tuning Lab")
    if "last_df" not in st.session_state:
        st.info("Run a prediction first to access tuning.")
    else:
        df        = st.session_state["last_df"]
        X_all     = scaler.transform(df[train_cols].values)
        # sample subset
        sample_n  = min(len(X_all), 1000)
        idxs      = np.random.choice(len(X_all), sample_n, replace=False)
        X_sample  = X_all[idxs]
        y_sample  = df["anomaly"].iloc[idxs].values

        st.subheader("Isolation Forest Tuning")
        iso_c = st.slider("contamination", 0.01, 0.5, 0.1, 0.01, key="tun_iso_cont")
        preds_iso, scores_iso = predict_iso(X_sample, contamination=iso_c)
        precision_iso = precision_score(y_sample, preds_iso, zero_division=0)
        recall_iso    = recall_score(y_sample, preds_iso, zero_division=0)
        f1_iso        = f1_score(y_sample, preds_iso, zero_division=0)
        fpr_iso,tpr_iso,_ = roc_curve(y_sample, scores_iso)
        auc_iso       = calc_auc(fpr_iso, tpr_iso)
        st.write(f"Precision: {precision_iso:.2f}, Recall: {recall_iso:.2f}, F1: {f1_iso:.2f}, AUC: {auc_iso:.2f}")
        fig_iso = px.area(
            x=fpr_iso, y=tpr_iso, title="ROC Curve (IForest)",
            labels={"x":"FPR","y":"TPR"},
            width=600, height=300
        )
        st.plotly_chart(fig_iso)

        st.subheader("Autoencoder Tuning")
        ae_t = st.slider("threshold", 0.0, 1.0, 0.02, 0.005, key="tun_ae_thresh")
        preds_ae, scores_ae = predict_ae(X_sample, thresh=ae_t)
        precision_ae = precision_score(y_sample, preds_ae, zero_division=0)
        recall_ae    = recall_score(y_sample, preds_ae, zero_division=0)
        f1_ae        = f1_score(y_sample, preds_ae, zero_division=0)
        fpr_ae,tpr_ae,_ = roc_curve(y_sample, scores_ae)
        auc_ae       = calc_auc(fpr_ae,tpr_ae)
        st.write(f"Precision: {precision_ae:.2f}, Recall: {recall_ae:.2f}, F1: {f1_ae:.2f}, AUC: {auc_ae:.2f}")
        fig_ae = px.area(x=fpr_ae, y=tpr_ae, title="ROC Curve (AE)", labels={"x":"FPR","y":"TPR"}, width=600, height=300)
        st.plotly_chart(fig_ae)

        st.subheader("LOF Tuning")
        lof_n  = st.slider("n_neighbors", 5, 100, 20, 1, key="tun_lof_nn")
        lof_c  = st.slider("contamination", 0.01, 0.5, 0.02, 0.01, key="tun_lof_cont")
        preds_lof, scores_lof = predict_lof(X_sample, contamination=lof_c, n_neighbors=lof_n)
        precision_lof = precision_score(y_sample, preds_lof, zero_division=0)
        recall_lof    = recall_score(y_sample, preds_lof, zero_division=0)
        f1_lof        = f1_score(y_sample, preds_lof, zero_division=0)
        fpr_lof,tpr_lof,_ = roc_curve(y_sample, scores_lof)
        auc_lof       = calc_auc(fpr_lof,tpr_lof)
        st.write(f"Precision: {precision_lof:.2f}, Recall: {recall_lof:.2f}, F1: {f1_lof:.2f}, AUC: {auc_lof:.2f}")
        fig_lof = px.area(x=fpr_lof, y=tpr_lof, title="ROC Curve (LOF)", labels={"x":"FPR","y":"TPR"}, width=600, height=300)
        st.plotly_chart(fig_lof)
