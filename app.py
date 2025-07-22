import os
import streamlit as st
import pandas as pd
import numpy as np
import joblib
from tensorflow.keras.models import load_model
import plotly.express as px
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import (
    confusion_matrix,
    classification_report,
    roc_curve,
    auc
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

def predict_iso(X):
    p = iso_model.predict(X)
    return np.where(p == 1, 0, 1)

def predict_ae(X, thresh):
    rec = ae_model.predict(X)
    mse = np.mean((X - rec) ** 2, axis=1)
    return mse, np.where(mse > thresh, 1, 0)

def predict_lof(X):
    p = lof_model.predict(X)
    return np.where(p == 1, 0, 1)

def lof_scores(X):
    return lof_model.decision_function(X)

# ─── UI ───────────────────────────────────────────────────────────────────────────
tabs = st.tabs(["🔍 Predict", "📊 EDA", "🧠 Explain", "📈 Performance"])

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

    st.title("🚨 Network Traffic Anomaly Detection")
    uploaded = st.file_uploader(
        "Upload dataset",
        type=["csv","gz","zip"],
        help="Raw KDD (.csv/.gz/.zip) or preprocessed CSV"
    )
    if not uploaded:
        st.info("Please upload your dataset to begin.")
    else:
        # 1) Preprocess
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

        # 2) Optional re-fit
        iso_model.set_params(contamination=iso_cont); iso_model.fit(X)
        lof_model.set_params(contamination=lof_cont); lof_model.fit(X)

        # 3) Predict
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
        st.session_state["last_df"]    = df
        st.session_state["last_model"] = model_choice
        st.session_state["ae_thresh"]  = ae_thresh

        # Display results
        st.subheader("Sample Results")
        st.dataframe(df.head(10))

        if model_choice in ("Autoencoder","Hybrid – Union","Hybrid – Intersection"):
            rec      = ae_model.predict(X)
            feat_err = pd.Series(np.mean((X-rec)**2,axis=0), index=train_cols)
            st.subheader("Top AE Reconstruction Errors")
            st.bar_chart(feat_err.nlargest(10))

        st.subheader("Anomaly Distribution")
        st.bar_chart(df["anomaly"].map({0:"Normal",1:"Attack"}).value_counts())

        csv = df.to_csv(index=False).encode()
        st.download_button("⬇️ Download Results", csv, "results.csv", "text/csv")

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
        df = st.session_state["last_df"]
        X  = scaler.transform(df[train_cols].values)
        scores = lof_scores(X)
        fig = px.histogram(scores, nbins=50, labels={"value":"LOF score"})
        st.plotly_chart(fig, use_container_width=True)

# ─── Tab 4: Performance ──────────────────────────────────────────────────────────
with tabs[3]:
    st.header("📈 Model Performance")
    if "last_df" not in st.session_state or "attack_type" not in st.session_state["last_df"].columns:
        st.info("To see performance, upload a CSV with an `attack_type` column.")
    else:
        df_perf = st.session_state["last_df"]
        y_true  = df_perf["attack_type"].values
        X       = scaler.transform(df_perf[train_cols].values)
        ae_thresh = st.session_state.get("ae_thresh", 0.02)

        eval_models = st.multiselect(
            "Select models to compare:",
            ["Isolation Forest","Autoencoder","Local Outlier Factor",
             "Hybrid – Union","Hybrid – Intersection"],
            default=["Isolation Forest"]
        )

        metrics = []
        roc_data = {}
        cms = {}

        for m in eval_models:
            if m=="Isolation Forest":
                y_pred = predict_iso(X)
                scores = iso_model.decision_function(X) * -1
            elif m=="Autoencoder":
                scores, y_pred = predict_ae(X, ae_thresh)
            elif m=="Local Outlier Factor":
                y_pred = predict_lof(X)
                scores = lof_scores(X) * -1
            elif m=="Hybrid – Union":
                iso_p = predict_iso(X); _, ae_p = predict_ae(X, ae_thresh)
                y_pred = np.logical_or(iso_p,ae_p).astype(int)
                scores = iso_model.decision_function(X)*-1
            else:
                iso_p = predict_iso(X); _, ae_p = predict_ae(X, ae_thresh)
                y_pred = np.logical_and(iso_p,ae_p).astype(int)
                scores = iso_model.decision_function(X)*-1

            cm      = confusion_matrix(y_true, y_pred)
            rpt     = classification_report(y_true, y_pred, output_dict=True, zero_division=0)
            fpr,tpr,_ = roc_curve(y_true, scores)
            roc_auc = auc(fpr, tpr)

            metrics.append({
                "Model": m,
                "Accuracy": rpt["accuracy"],
                "Precision(1)": rpt["1"]["precision"],
                "Recall(1)":    rpt["1"]["recall"],
                "F1(1)":        rpt["1"]["f1-score"],
                "ROC AUC":      roc_auc
            })
            roc_data[m] = (fpr,tpr)
            cms[m]      = cm

        perf_df = pd.DataFrame(metrics).set_index("Model")
        st.subheader("▶️ Summary Metrics")
        st.dataframe(perf_df.style.format("{:.3f}"))

        for m, cm in cms.items():
            st.subheader(f"Confusion Matrix — {m}")
            fig, ax = plt.subplots()
            sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax)
            ax.set_xlabel("Predicted"); ax.set_ylabel("True")
            st.pyplot(fig)

        st.subheader("📊 ROC Curves")
        fig, ax = plt.subplots()
        for m,(fpr,tpr) in roc_data.items():
            ax.plot(fpr, tpr, label=f"{m} (AUC={perf_df.loc[m,'ROC AUC']:.2f})")
        ax.plot([0,1],[0,1],"k--",alpha=0.3)
        ax.set_xlabel("False Positive Rate"); ax.set_ylabel("True Positive Rate")
        ax.legend(loc="lower right")
        st.pyplot(fig)
