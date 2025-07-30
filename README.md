# 🚨 NetworkAnomalyStreamlit

**Real-time Network Traffic Anomaly Detection Dashboard**  
Built with **Streamlit + Scikit-Learn + TensorFlow/Keras + Plotly**

---

### What can you do?

| 🔎 Detect | 📊 Explore | 🧠 Explain | 🔬 Visualise | ⚡ Stream |
|-----------|-----------|-----------|-------------|----------|
| Isolation Forest · Deep Autoencoder · LOF · Hybrid (Union / Intersection) | Histograms · Heat-maps · Box-plots | SHAP bars · AE reconstruction-error ranks | 2-D / 3-D PCA (drag-rotate) | Live anomaly ticker for your SOC wall |

---

## ✨ Feature Highlights
1. **Zero-label friendly detectors**  
   *Isolation Forest* & *LOF* (contamination sliders) — *Deep Autoencoder* (live threshold tuning) — Hybrid modes for recall vs. precision.

2. **Rich EDA pane**  
   Protocol breakdown · Numeric heat-map · Attack/normal box-plots — all dark-mode.

3. **Explainability built-in**  
   SHAP importances, AE error bars, LOF score histogram.

4. **Interactive Embedding**  
   2-D / 3-D PCA scatter; colour-coded anomalies.

5. **⚡ Live Feed**  
   Real-time streaming chart (0.1-5 s refresh) ­— great for demo loops.

6. **📚 Education tab**  
   Bite-size cheat-sheets on each algorithm + hybrid strategy.

<img width="1600" height="836" alt="image" src="https://github.com/user-attachments/assets/3e1e76a8-3b31-489d-8059-d99068b19559" />
<img width="1600" height="790" alt="image" src="https://github.com/user-attachments/assets/cc5028ad-ff8f-48a9-a7c8-2a5214e9a97c" />
<img width="1600" height="549" alt="image" src="https://github.com/user-attachments/assets/a2468341-daef-45b5-9bd8-c0e412463f29" />
<img width="1600" height="562" alt="image" src="https://github.com/user-attachments/assets/231a54c7-3fa6-42ac-b09f-f44f9a43ad54" />
<img width="1600" height="684" alt="image" src="https://github.com/user-attachments/assets/c0024b31-7a21-4bb8-a5c0-b55b8f298e38" />


---

## ⚙️ Quick-start

```bash
git clone https://github.com/<your-username>/NetworkAnomalyStreamlit.git
cd NetworkAnomalyStreamlit
pip install -r requirements.txt          # create a venv first if you like
```
## Drop the pre-trained artefacts (or train your own) into the repo:
models/

  iso_model.pkl,
  lof_model.pkl,
  autoencoder_model.h5,
  scaler.pkl

data/

  train_cols.pkl,
  iso_shap_importances.csv

streamlit run app.py

## 🚀 Road-map
> NetFlow / Zeek / PCAP loaders

> Transformer-based sequence models

> Docker-compose deployment

> Alert web-hook & Slack sink

**Contributions & ideas welcome — open an issue or ping me on LinkedIn. 🚀**

