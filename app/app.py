import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.pipeline import predict_cluster

# Load data
df = pd.read_csv(ROOT_DIR / "data" / "Mall_Customers.csv")

# Load models from models folder (FIXED ✅)
scaler = joblib.load(ROOT_DIR / "models" / "scaler.pkl")
pca = joblib.load(ROOT_DIR / "models" / "pca.pkl")
model = joblib.load(ROOT_DIR / "models" / "kmeans_model.pkl")

st.title("🧠 Customer Segmentation Dashboard")
st.write("K-Means + PCA Clustering Project")

# Show dataset
if st.checkbox("Show Dataset"):
    st.write(df.head())

# Feature selection
X = df[['Annual Income (k$)', 'Spending Score (1-100)']]

# Step 1: Scaling
X_scaled = scaler.transform(X)

# Step 2: PCA transformation (IMPORTANT 🔥)
X_pca = pca.transform(X_scaled)

# Step 3: Clustering
clusters = model.predict(X_pca)

# Visualization (NOW CORRECT ✅)
fig, ax = plt.subplots()

scatter = ax.scatter(
    X_pca[:, 0],
    X_pca[:, 1],
    c=clusters
)

centroids = model.cluster_centers_

ax.scatter(
    centroids[:, 0],
    centroids[:, 1],
    s=200,
    marker='X'
)

ax.set_title("Customer Segments (PCA Reduced)")
ax.set_xlabel("Principal Component 1")
ax.set_ylabel("Principal Component 2")

st.pyplot(fig)

# Cluster insights
st.subheader("📌 Cluster Insights")
cluster_counts = pd.Series(clusters).value_counts()
st.write(cluster_counts)

# Prediction section
st.subheader("🔮 Predict Customer Segment")

income = st.slider("Annual Income", 0, 150, 50)
score = st.slider("Spending Score", 0, 100, 50)

if st.button("Predict Cluster"):
    
    # ✅ USE PIPELINE (VERY IMPORTANT)
    prediction = predict_cluster([[income, score]])
    
    def get_cluster_label(cluster):
        labels = {
            0: "Low Income - Low Spending",
            1: "High Income - High Spending",
            2: "Average Customers",
            3: "High Income - Low Spending",
            4: "Low Income - High Spending"
        }
        return labels.get(cluster, "Unknown")

    st.success(f"Cluster: {prediction} → {get_cluster_label(prediction)}")