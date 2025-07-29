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

# ─── Page config & custom CSS ───────────────────────────────────────────────────
st.set_page_config(
    page_title="Network Traffic Anomaly Detection",
    page_icon="🚨",
    layout="wide"
)
st.markdown(
    """
    <style>
      /* Hide default Streamlit header & footer */
      #MainMenu {visibility: hidden;}
      footer {visibility: hidden;}
      header {visibility: hidden;}

      /* Restyle sidebar */
      [data-testid="stSidebar"] {
        background: #1e1e2d;
        color: #fff;
      }
      /* Style primary buttons */
      .stButton>button {
        background: linear-gradient(135deg, #1abc9c, #2ecc71);
        color: white;
        border: none;
        border-radius: 8px;
        padding: 8px 16px;
      }
      .stButton>button:hover {
        opacity: 0.9;
      }
      /* Tabs */
      .stTabs [data-baseweb="tab-list"] button {
        font-weight: bold;
        color: #f39c12;
      }
      .stTabs [data-baseweb="tab-list"] button[aria-selected="true"] {
        background: #f39c12;
        color: #1e1e2d;
      }
      /* Section headers */
      .section-header {
        font-size: 1.5rem;
        background: linear-gradient(135deg,#2980b9,#6dd5fa);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.5rem;
      }
    </style>
    """,
    unsafe_allow_html=True
)

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

# initialize state
for k in ("last_df","last_meta","last_model","ae_thresh","streaming"):
    if k not in st.session_state:
        st.session_state[k] = None
st.session_state.streaming = False

# ─── Main Tabs ───────────────────────────────────────────────────────────────────
tabs = st.tabs([
  "🔍 Predict","📊 EDA","🧠 Explain",
  "🔬 Embedding","⚡ Live Feed","📚 Education"
])

# ─── 1) Predict ──────────────────────────────────────────────────────────────────
with tabs[0]:
    st.markdown('<div class="section-header">🚨 Predict</div>', unsafe_allow_html=True)
    # sidebar controls
    st.sidebar.header("Settings")
    upload_type  = st.sidebar.radio("Upload type:",("Raw KDD data","Preprocessed CSV"))
    sample_rows  = st.sidebar.slider("Rows to sample (raw)",10000,200000,50000,10000)
    iso_cont     = st.sidebar.slider("IForest contamination",0.01,0.5,0.1,0.01)
    lof_cont     = st.sidebar.slider("LOF contamination",0.01,0.5,0.02,0.01)
    ae_thresh    = st.sidebar.slider("AE threshold",0.0,1.0,0.02,0.005)
    model_choice = st.sidebar.selectbox("Model:",[
        "Isolation Forest","Autoencoder","Local Outlier Factor",
        "Hybrid – Union","Hybrid – Intersection"
    ])

    uploaded = st.file_uploader("Upload dataset", type=["csv","gz","zip"])
    if uploaded:
        # preprocess
        if upload_type=="Raw KDD data":
            df_proc, raw_meta = preprocess_raw_kdd(uploaded, sample_rows)
            X = scaler.transform(df_proc.values); df = df_proc.copy()
            st.session_state.last_meta = raw_meta
        else:
            comp = detect_compression(uploaded)
            df = pd.read_csv(uploaded, compression=comp)
            X = df.reindex(columns=train_cols).fillna(0).values
            st.session_state.last_meta = None

        # refit models
        iso_model.set_params(contamination=iso_cont); iso_model.fit(X)
        lof_model.set_params(contamination=lof_cont); lof_model.fit(X)

        # predict
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
        st.session_state.last_df    = df
        st.session_state.last_model = model_choice
        st.session_state.ae_thresh  = ae_thresh

        # metrics cards
        total = len(preds)
        norm  = int((preds==0).sum())
        att   = int((preds==1).sum())
        c1, c2, c3 = st.columns(3)
        c1.metric("Total samples", total)
        c2.metric("Normal", norm)
        c3.metric("Anomalies", att)

        # sample table
        st.dataframe(df.head(10), use_container_width=True)

        # AE errors chart
        if model_choice in ("Autoencoder","Hybrid – Union","Hybrid – Intersection"):
            rec      = ae_model.predict(X)
            feat_err = pd.Series(np.mean((X-rec)**2,axis=0), index=train_cols)
            st.subheader("Top AE Reconstruction Errors")
            st.bar_chart(feat_err.nlargest(10))

        st.subheader("Anomaly Distribution")
        st.bar_chart(df["anomaly"].map({0:"Normal",1:"Attack"}).value_counts())

        csv = df.to_csv(index=False).encode()
        st.download_button("⬇️ Download results", csv, "results.csv", "text/csv")
    else:
        st.info("Please upload your dataset.")

# ─── 2) EDA ──────────────────────────────────────────────────────────────────────
with tabs[1]:
    st.markdown('<div class="section-header">📊 Exploratory Data Analysis</div>', unsafe_allow_html=True)
    if st.session_state.last_df is None:
        st.info("Run Predict first to populate data.")
    else:
        df = st.session_state.last_df
        norm = int((df.anomaly==0).sum())
        att  = int((df.anomaly==1).sum())
        m1, m2, m3 = st.columns([2,1,1])
        m1.metric("Records", len(df))
        m2.metric("Normal", norm)
        m3.metric("Attack", att)

        # protocol breakdown
        if st.session_state.last_meta is not None:
            st.subheader("Protocol Breakdown")
            meta = st.session_state.last_meta.copy()
            meta["anomaly"] = df["anomaly"].map({0:"Normal",1:"Attack"})
            fig = px.bar(meta, x="protocol_type", color="anomaly", barmode="group")
            st.plotly_chart(fig, use_container_width=True)

        # correlations
        st.subheader("Numeric Correlations")
        cols = ["duration","src_bytes","dst_bytes","count","srv_count"]
        corr = df[cols].corr()
        fig, ax = plt.subplots(figsize=(6,5))
        sns.heatmap(corr, annot=True, cmap="vlag", ax=ax)
        st.pyplot(fig)

        # distributions
        st.subheader("Data Distributions")
        fig3, (ax1,ax2) = plt.subplots(1,2,figsize=(12,4))
        sns.boxplot(x=df.anomaly, y=df.src_bytes, ax=ax1).set_title("src_bytes")
        sns.boxplot(x=df.anomaly, y=df.dst_bytes, ax=ax2).set_title("dst_bytes")
        st.pyplot(fig3)

# ─── 3) Explain ──────────────────────────────────────────────────────────────────
with tabs[2]:
    st.markdown('<div class="section-header">🧠 Explainability</div>', unsafe_allow_html=True)
    choice = st.selectbox("Choose model to explain:",[
        "Isolation Forest","Autoencoder","Local Outlier Factor"
    ])
    if choice=="Isolation Forest":
        st.write("Global SHAP importances for IForest.")
        dfsh = iso_shap_imp.reset_index().rename(columns={"index":"feature",0:"importance"})
        fig = px.bar(dfsh, x="importance", y="feature", orientation="h")
        fig.update_layout(margin=dict(l=0,r=0,t=40,b=0))
        st.plotly_chart(fig, use_container_width=True)
    elif choice=="Autoencoder":
        st.write("Top AE reconstruction-error features.")
        df = st.session_state.last_df
        X  = scaler.transform(df[train_cols].values)
        rec= ae_model.predict(X)
        errs = pd.Series(np.mean((X-rec)**2,axis=0), index=train_cols)
        top  = errs.nlargest(10).reset_index().rename(columns={"index":"feature",0:"error"})
        fig  = px.bar(top, x="error", y="feature", orientation="h")
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.write("LOF normality score distribution.")
        df = st.session_state.last_df
        X  = scaler.transform(df[train_cols].values)
        scores = lof_scores(X)
        fig = px.histogram(scores, nbins=50)
        st.plotly_chart(fig, use_container_width=True)

# ─── 4) Embedding ────────────────────────────────────────────────────────────────
with tabs[3]:
    st.markdown('<div class="section-header">🔬 PCA Embedding</div>', unsafe_allow_html=True)
    if st.session_state.last_df is None:
        st.info("Run Predict first.")
    else:
        df   = st.session_state.last_df
        X    = scaler.transform(df[train_cols].values)
        n    = min(len(X),5000)
        idxs = np.random.choice(len(X),n,replace=False)
        Xs   = X[idxs]; dfs = df.iloc[idxs].copy()

        dim = st.radio("Dimension:",("2D","3D"))
        if dim=="2D":
            pca    = PCA(2); coords = pca.fit_transform(Xs)
            dfs["PC1"],dfs["PC2"] = coords[:,0],coords[:,1]
            fig = px.scatter(dfs, x="PC1", y="PC2",
                             color=dfs.anomaly.map({0:"Normal",1:"Attack"}))
        else:
            pca    = PCA(3); coords = pca.fit_transform(Xs)
            dfs["PC1"],dfs["PC2"],dfs["PC3"] = coords[:,0],coords[:,1],coords[:,2]
            fig = px.scatter_3d(dfs, x="PC1",y="PC2",z="PC3",
                                color=dfs.anomaly.map({0:"Normal",1:"Attack"}))
        fig.update_layout(margin=dict(l=0,r=0,t=40,b=0))
        st.plotly_chart(fig, use_container_width=True)

# ─── 5) Live Feed ───────────────────────────────────────────────────────────────
with tabs[4]:
    st.markdown('<div class="section-header">⚡ Live Streaming Feed</div>', unsafe_allow_html=True)
    interval = st.slider("Update interval (sec)",0.1,5.0,1.0,0.1)
    if st.button("▶️ Start",key="s"):
        st.session_state.streaming = True
    if st.button("⏹ Stop",key="t"):
        st.session_state.streaming = False

    placeholder = st.empty()
    if st.session_state.streaming and st.session_state.last_df is not None:
        chart = placeholder.line_chart(pd.DataFrame(columns=["anomaly"]))
        while st.session_state.streaming:
            df = st.session_state.last_df
            i  = np.random.randint(len(df))
            new = pd.DataFrame({"anomaly":[df.iloc[i]["anomaly"]]},
                               index=[pd.Timestamp.now()])
            chart.add_rows(new)
            time.sleep(interval)
    else:
        placeholder.write("Click ▶️ to begin streaming anomalies over time.")

# ─── 6) Education ───────────────────────────────────────────────────────────────
with tabs[5]:
    st.markdown('<div class="section-header">📚 Educational Insights</div>', unsafe_allow_html=True)
    st.markdown("**Deep dives** into each detector, hybrid strategies, pros/cons & tips:")
    with st.expander("Isolation Forest"):
        st.write("""
          - **Core idea:** Random binary splits → short paths isolate outliers  
          - **When to use:** High-dim data, fast inference  
          - **Watch out:** Contamination hyperparam  
        """)
    with st.expander("Autoencoder"):
        st.write("""
          - **Core idea:** Train on normal → large reconstruction MSE signals anomalies  
          - **Pro tip:** Tune your threshold by plotting precision/recall vs MSE  
        """)
    with st.expander("Local Outlier Factor"):
        st.write("""
          - **Core idea:** Points with substantially lower local density are anomalies  
          - **Tip:** `novelty=True` lets you call `.predict` on new data  
        """)
    with st.expander("Hybrid: Union vs Intersection"):
        st.write("""
          - **Union:** flag if *any* model triggers → 🔍 *Higher recall*  
          - **Intersection:** flag only if *all* agree → 🎯 *Higher precision*  
        """)
    st.markdown("> Try both & compare on your dataset for best trade-off!")
