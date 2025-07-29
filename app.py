import os
import time
import pandas as pd
import numpy as np
import joblib
import streamlit as st
import plotly.express as px
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from tensorflow.keras.models import load_model

# ─── Page config ────────────────────────────────────────────────────────────────
st.set_page_config(page_title="Network Traffic Anomaly Detection", layout="wide")

# ─── Custom UI Styling ─────────────────────────────────────────────────────────
st.markdown("""
    <style>
        html, body, [class*="css"]  {
            font-family: 'Segoe UI', sans-serif;
            background-color: #0f1117;
            color: #e1e1e1;
        }
        .stApp {
            background-color: #0f1117;
        }
        h1, h2, h3 {
            color: #00e676;
        }
        .block-container {
            padding-top: 2rem;
        }
        .stButton>button {
            background-color: #1f7a8c;
            color: white;
            border-radius: 8px;
            padding: 0.5em 1em;
        }
        .stDownloadButton>button {
            background: linear-gradient(to right, #00e676, #1de9b6);
            color: black;
            font-weight: bold;
            border-radius: 10px;
        }
        .stSidebar {
            background-color: #1e1e2d !important;
        }
        .stSlider > div[role='slider'] {
            background: #00e676 !important;
        }
    </style>
""", unsafe_allow_html=True)

# ─── Header ─────────────────────────────────────────────────────────────────────
st.markdown("""
    <h1 style='text-align: center; color: #00e676; font-weight: bold;'>
        🚨 Network Traffic Anomaly Detection Dashboard
    </h1>
    <p style='text-align: center; font-size: 16px; color: #ccc;'>
        Powered by Autoencoder • Isolation Forest • LOF • PCA • SHAP
    </p>
""", unsafe_allow_html=True)

# ─── Sidebar Expander ───────────────────────────────────────────────────────────
with st.sidebar.expander("📘 About This Tool"):
    st.markdown("""
    This dashboard uses **unsupervised ML** to detect intrusions in KDD'99 traffic data.

    Models:
    - 📊 Isolation Forest
    - 🧠 Autoencoder (Deep Learning)
    - 🗞 LOF (Local Density)

    Hybrid modes improve robustness. Built with **scikit-learn, TensorFlow, Streamlit**.
    """)

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
    return np.where(p==1, 0, 1)

def predict_ae(X, thresh):
    rec = ae_model.predict(X)
    mse = np.mean((X-rec)**2, axis=1)
    return mse, np.where(mse>thresh, 1, 0)

def predict_lof(X):
    p = lof_model.predict(X)
    return np.where(p==1, 0, 1)

def lof_scores(X):
    return lof_model.decision_function(X)

# ─── Initialize Session State ────────────────────────────────────────────────────
for key, val in {
    "last_df": None,
    "last_meta": None,
    "last_model": None,
    "ae_thresh": 0.02,
    "streaming": False,
    "stream_chart": None
}.items():
    if key not in st.session_state:
        st.session_state[key] = val

# ─── Main Tabs ────────────────────────────────────────────────────────────────────
tabs = st.tabs([
    "🔍 Predict", "📊 EDA", "🧠 Explain",
    "🔬 Embedding", "⚡ Live Feed", "📚 Education"
])

# ─── Tab 1: Predict ───────────────────────────────────────────────────────────────
with tabs[0]:
    st.sidebar.header("Settings")
    upload_type  = st.sidebar.radio("Upload type:", ("Raw KDD data","Preprocessed CSV"))
    sample_rows  = st.sidebar.slider("Rows to sample (raw)", 10000, 200000, 50000, 10000)
    iso_cont     = st.sidebar.slider("IForest contamination", 0.01, 0.5, 0.1, 0.01)
    lof_cont     = st.sidebar.slider("LOF contamination",    0.01, 0.5, 0.02, 0.01)
    ae_thresh    = st.sidebar.slider("AE threshold",         0.0, 1.0, 0.02, 0.005)
    model_choice = st.sidebar.selectbox("Model:", [
        "Isolation Forest","Autoencoder","Local Outlier Factor",
        "Hybrid – Union","Hybrid – Intersection"
    ])

    st.markdown("### 📂 Upload & Preprocess Your Data")
    uploaded = st.file_uploader(
        "Upload dataset", type=["csv","gz","zip"],
        help="Raw KDD (.csv/.gz/.zip) or preprocessed CSV"
    )
    if not uploaded:
        st.info("Please upload your dataset to begin.")
    else:
        if upload_type=="Raw KDD data":
            with st.spinner(f"🚀 Processing first {sample_rows:,} rows... Please wait..."):
                df_proc, raw_meta = preprocess_raw_kdd(uploaded, sample_rows)
            X = scaler.transform(df_proc.values)
            df = df_proc.copy()
            st.session_state.last_meta = raw_meta
        else:
            comp = detect_compression(uploaded)
            df = pd.read_csv(uploaded, compression=comp)
            X = df.reindex(columns=train_cols).fillna(0).values
            st.session_state.last_meta = None

        # Re-fit models
        iso_model.set_params(contamination=iso_cont); iso_model.fit(X)
        lof_model.set_params(contamination=lof_cont); lof_model.fit(X)

        # Make predictions
        if model_choice=="Isolation Forest":
            preds = predict_iso(X)
        elif model_choice=="Autoencoder":
            mse, preds = predict_ae(X, ae_thresh)
        elif model_choice=="Local Outlier Factor":
            preds = predict_lof(X)
        elif model_choice=="Hybrid – Union":
            iso_p = predict_iso(X); _, ae_p = predict_ae(X, ae_thresh)
            preds = np.logical_or(iso_p, ae_p).astype(int)
        else:
            iso_p = predict_iso(X); _, ae_p = predict_ae(X, ae_thresh)
            preds = np.logical_and(iso_p, ae_p).astype(int)

        df["anomaly"] = preds
        st.session_state.last_df    = df
        st.session_state.last_model = model_choice
        st.session_state.ae_thresh  = ae_thresh

        # Results
        st.markdown("### 📈 Anomaly Results")
        st.dataframe(df.head(10), use_container_width=True)

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
    st.markdown("### 📊 Exploratory Data Analysis")
    if st.session_state.last_df is None:
        st.info("Upload & predict to see EDA.")
    else:
        df       = st.session_state.last_df
        raw_meta = st.session_state.last_meta

        if raw_meta is not None:
            st.subheader("Protocol Breakdown")
            proto = raw_meta.copy()
            proto["anomaly"] = df["anomaly"].map({0:"Normal",1:"Attack"})
            fig1 = px.bar(proto, x="protocol_type", color="anomaly", barmode="group",
                          labels={"anomaly":"0=Normal,1=Attack"})
            fig1.update_layout(paper_bgcolor='white', plot_bgcolor='white', font_color='black')
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
    st.markdown("### 🧠 Explainability")
    choice = st.selectbox("Explain model:", [
        "Isolation Forest","Autoencoder","Local Outlier Factor"
    ])
    if choice=="Isolation Forest":
        st.write("Global SHAP importances for Isolation Forest.")
        shap_df = iso_shap_imp.reset_index().rename(columns={"index":"feature",0:"importance"})
        fig = px.bar(shap_df, x="importance", y="feature", orientation="h",
                     labels={"importance":"Mean |SHAP value|"})
        fig.update_layout(yaxis_categoryorder="total ascending", paper_bgcolor='white', plot_bgcolor='white', font_color='black')
        st.plotly_chart(fig, use_container_width=True)

    elif choice=="Autoencoder":
        st.write("Top features by autoencoder reconstruction error.")
        df = st.session_state.last_df
        X  = scaler.transform(df[train_cols].values)
        rec= ae_model.predict(X)
        errs = pd.Series(np.mean((X-rec)**2,axis=0), index=train_cols)
        top = errs.nlargest(10).reset_index().rename(columns={"index":"feature",0:"error"})
        fig = px.bar(top, x="error", y="feature", orientation="h",
                     labels={"error":"MSE"})
        fig.update_layout(paper_bgcolor='white', plot_bgcolor='white', font_color='black')
        st.plotly_chart(fig, use_container_width=True)

    else:
        st.write("LOF score distribution (lower = more anomalous).")
        df = st.session_state.last_df
        X  = scaler.transform(df[train_cols].values)
        scores = lof_scores(X)
        fig = px.histogram(scores, nbins=50, labels={"value":"LOF score"})
        fig.update_layout(paper_bgcolor='white', plot_bgcolor='white', font_color='black')
        st.plotly_chart(fig, use_container_width=True)

# ─── Tab 4: Embedding ────────────────────────────────────────────────────────────
with tabs[3]:
    st.markdown("### 🔬 PCA Embedding of Network Traffic")
    if st.session_state.last_df is None:
        st.info("Upload & predict to see embedding.")
    else:
        df = st.session_state.last_df
        X  = scaler.transform(df[train_cols].values)
        n  = min(len(X), 5000)
        idxs = np.random.choice(len(X), n, replace=False)
        Xs = X[idxs]; dfs = df.iloc[idxs].copy()

        dim = st.radio("Projection dimension:", ("2D","3D"))
        if dim=="2D":
            pca = PCA(2); coords = pca.fit_transform(Xs)
            dfs["PC1"], dfs["PC2"] = coords[:,0], coords[:,1]
            fig = px.scatter(dfs, x="PC1", y="PC2",
                             color=dfs["anomaly"].map({0:"Normal",1:"Attack"}),
                             title="PCA (2D)")
        else:
            pca = PCA(3); coords = pca.fit_transform(Xs)
            dfs["PC1"], dfs["PC2"], dfs["PC3"] = coords[:,0], coords[:,1], coords[:,2]
            fig = px.scatter_3d(dfs, x="PC1", y="PC2", z="PC3",
                                color=dfs["anomaly"].map({0:"Normal",1:"Attack"}),
                                title="PCA (3D)")
        fig.update_layout(margin=dict(l=0,r=0,t=40,b=0), paper_bgcolor='white', plot_bgcolor='white', font_color='black')
        st.plotly_chart(fig, use_container_width=True)

# ─── Tab 5: Live Feed ────────────────────────────────────────────────────────────
with tabs[4]:
    st.markdown("### ⚡ Real-Time Streaming Feed")
    interval = st.sidebar.slider("Update interval (sec)", 0.1, 5.0, 1.0, 0.1)
    if st.button("▶️ Start Streaming", key="start"):
        st.session_state.streaming = True
    if st.button("⏹ Stop Streaming", key="stop"):
        st.session_state.streaming = False

    placeholder = st.empty()
    if st.session_state.streaming and st.session_state.last_df is not None:
        chart = placeholder.line_chart(pd.DataFrame(columns=["anomaly"]))
        while st.session_state.streaming:
            df = st.session_state.last_df
            idx = np.random.randint(len(df))
            new = pd.DataFrame(
                {"anomaly":[ df.iloc[idx]["anomaly"] ]},
                index=[pd.Timestamp.now()]
            )
            chart.add_rows(new)
            time.sleep(interval)
    else:
        placeholder.write("Press ▶️ to stream anomaly flags over time.")

# ─── Tab 6: Education ────────────────────────────────────────────────────────────
with tabs[5]:
    st.markdown("### 📚 Educational Insights")
    st.markdown("""
    Dive deep into the mechanics of each anomaly detector and hybrid strategy.
    Each section below unfolds key intuition, workflows, pros & cons, and real-world tips.
    """)
    with st.expander("Isolation Forest"):
        st.write("""
        - **Idea**: Randomly partition features; anomalies are easier to isolate.
        - **Workflow**: Fit on data → anomaly score = path length in the random tree.
        - **Pros**: Fast, unsupervised, no distributional assumptions.
        - **Cons**: Sensitive to contamination setting; may miss subtle anomalies.
        """)
    with st.expander("Autoencoder"):
        st.write("""
        - **Idea**: Compress–reconstruct input; high reconstruction error ⇒ anomaly.
        - **Workflow**: Train on ‘normal’ → measure per-example MSE at inference.
        - **Pros**: Learns nonlinear representations; adaptive thresholds.
        - **Cons**: Needs enough normal data; MSE threshold tuning is crucial.
        """)
    with st.expander("Local Outlier Factor"):
        st.write("""
        - **Idea**: Compare local density to neighbors; low-density points ⇒ outliers.
        - **Workflow**: Fit on normal data → `decision_function` gives normality score.
        - **Pros**: Captures local structure; no training/inference divide.
        - **Cons**: O(n²) complexity; choice of `n_neighbors` impacts sensitivity.
        """)
    with st.expander("Hybrid Models"):
        st.write("""
        **Union** (🔁): Flag anomaly if **any** model says so  
        → **High recall**, catch more attacks, risk more false alarms.  

        **Intersection** (🔁): Flag anomaly only if **all** models agree  
        → **High precision**, fewer false alarms, risk missing subtle anomalies.  
        """)
    st.markdown("**Real-World Tip:** Test both strategies to find your best trade-off!")
