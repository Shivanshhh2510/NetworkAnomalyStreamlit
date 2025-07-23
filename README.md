# NetworkAnomalyStreamlit

🚨 **Interactive Network Traffic Anomaly Detection App**  
Built with Streamlit, Scikit-Learn, TensorFlow/Keras and Plotly, this app lets you upload raw KDD-Cup traffic logs (CSV/GZ/ZIP) or a preprocessed features-only CSV, then:

- 🔍 **Detect anomalies** with Isolation Forest, Autoencoder, LOF or hybrid ensembles  
- 📊 **Explore** your data via histograms, heatmaps & boxplots  
- 🧠 **Explain** model decisions with SHAP importances & reconstruction-error bar charts  
- 🔬 **Embed** high-dimensional flows into 2D/3D PCA for interactive visualization  

<img width="2535" height="678" alt="image" src="https://github.com/user-attachments/assets/94b81452-3786-4eba-a0ad-a23cda29cbac" />

<img width="2513" height="1308" alt="image" src="https://github.com/user-attachments/assets/21ce6b63-a30d-435b-8faf-6ed4f8b16bcf" />

<img width="2513" height="1265" alt="image" src="https://github.com/user-attachments/assets/3d8ff9e0-d109-4415-99f6-4bead457f9b1" />

<img width="1610" height="660" alt="image" src="https://github.com/user-attachments/assets/16a200d0-bda5-48ed-b872-d94f55d6bd3d" />

<img width="2489" height="658" alt="image" src="https://github.com/user-attachments/assets/a32953d6-950b-4565-88e7-1d246e933dde" />


---

## 🚀 Features

1. **Raw or preprocessed upload**  
   - Auto-detect `.csv`, `.gz` & `.zip`; sample first *N* rows for speed  
   - One-click one-hot encoding & scaling for KDD-Cup data  

2. **Multiple detectors & sliders**  
   - Isolation Forest & Local Outlier Factor with adjustable contamination  
   - Deep Autoencoder with real-time threshold tuning  
   - “Union” & “Intersection” hybrid modes  

3. **Rich EDA**  
   - Protocol-type breakdown bar charts  
   - Numeric feature correlations via heatmap  
   - Attack vs. normal boxplots  

4. **Explainability**  
   - Global SHAP bar chart for Isolation Forest  
   - Top-N reconstruction-error features for autoencoder  
   - LOF score distributions  

5. **Embedding**  
   - 2D & 3D PCA scatter plots (drag-rotate in 3D)  
   - Color-code anomalies vs. normals  

---

## 📥 Installation

1. **Clone**

   ```bash
   git clone https://github.com/<your-username>/NetworkAnomalyStreamlit.git
   cd NetworkAnomalyStreamlit
2. **Install dependencies**

    pip install -r requirements.txt

3. **Download artifacts**

Place the following files in the repo root:

iso_model.pkl

lof_model.pkl

autoencoder_model.h5

scaler.pkl

train_cols.pkl

iso_shap_importances.csv


4. **Run**:

streamlit run app.py

Then open http://localhost:8501 in your browser.


