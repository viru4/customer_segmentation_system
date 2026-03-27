import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

import sys
import os
# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.data_preprocessing import load_and_preprocess
from src.clustering import load_models
from src.pipeline import apply_pca, apply_autoencoder
from src.evaluation import evaluate_clustering

# ---------------- UI ----------------
st.title("🧠 Customer Segmentation System")

# Load data
X_scaled, df = load_and_preprocess()

# Sidebar
st.sidebar.header("⚙️ Settings")

dim_method = st.sidebar.selectbox(
    "Dimensionality Reduction",
    ["None", "PCA", "Autoencoder"]
)

model_name = st.sidebar.selectbox(
    "Clustering Algorithm",
    ["KMeans", "DBSCAN", "GMM", "Hierarchical"]
)

# Load models
models = load_models()
model = models[model_name]

# Apply dimensionality reduction
if dim_method == "PCA":
    X_processed = apply_pca(X_scaled)

elif dim_method == "Autoencoder":
    X_processed = apply_autoencoder(X_scaled)

else:
    X_processed = X_scaled

# Predict clusters
labels = model.fit_predict(X_processed)

# ---------------- Visualization ----------------
st.subheader("📊 Cluster Visualization")

plt.figure()
plt.scatter(X_processed[:, 0], X_processed[:, 1], c=labels)
plt.xlabel("Feature 1")
plt.ylabel("Feature 2")

st.pyplot(plt)

# ---------------- Evaluation ----------------
metrics = evaluate_clustering(X_processed, labels)

st.subheader("📈 Model Evaluation")

st.metric("Silhouette Score", round(metrics["Silhouette Score"], 3))
st.metric("Davies-Bouldin Index", round(metrics["Davies-Bouldin Index"], 3))
st.metric("Calinski-Harabasz Score", round(metrics["Calinski-Harabasz Score"], 3))

# ---------------- Business Insights ----------------
df["Cluster"] = labels

st.subheader("🧠 Customer Insights")

for cluster in sorted(df["Cluster"].unique()):
    segment = df[df["Cluster"] == cluster]

    st.write(f"### Cluster {cluster}")
    st.write(f"Avg Income: {segment['Annual Income (k$)'].mean():.2f}")
    st.write(f"Avg Spending Score: {segment['Spending Score (1-100)'].mean():.2f}")

# ---------------- Model Comparison ----------------
if st.button("Compare All Models"):

    comparison = []

    for name, m in models.items():
        labels = m.fit_predict(X_processed)
        scores = evaluate_clustering(X_processed, labels)
        scores["Model"] = name
        comparison.append(scores)

    df_comp = pd.DataFrame(comparison)

    st.subheader("📊 Model Comparison")
    st.dataframe(df_comp)

    # Best model
    best = df_comp.sort_values("Silhouette Score", ascending=False).iloc[0]
    st.success(f"🏆 Best Model: {best['Model']}")

# ---------------- Download ----------------
st.download_button(
    "⬇️ Download Clustered Data",
    df.to_csv(index=False),
    file_name="clustered_customers.csv"
)

st.markdown("---")

st.markdown(
    """
    <div style="text-align:center;">
        <h4>👨‍💻 Developed by Virendra Kumar</h4>
        <p>
            <a href="https://github.com/viru4">GitHub</a> |
            <a href="https://virendramlportfolio.netlify.app">Portfolio</a> |
            <a href="https://www.linkedin.com/in/virendra-kumar04/">LinkedIn</a>
        </p>
        <p>🚀 Machine Learning + Deep Learning Project</p>
    </div>
    """,
    unsafe_allow_html=True
)

st.sidebar.markdown("## 👨‍💻 Developer")
st.sidebar.markdown("Virendra Kumar")
st.sidebar.markdown("[GitHub](https://github.com/viru4)")
st.sidebar.markdown("[Portfolio](https://virendramlportfolio.netlify.app/)")