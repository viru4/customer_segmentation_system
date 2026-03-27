import pickle
import numpy as np
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]

# Load models
with open(ROOT_DIR / "models" / "scaler.pkl", "rb") as f:
    scaler = pickle.load(f)
with open(ROOT_DIR / "models" / "pca.pkl", "rb") as f:
    pca = pickle.load(f)
with open(ROOT_DIR / "models" / "kmeans_model.pkl", "rb") as f:
    kmeans = pickle.load(f)

def predict_cluster(input_data):
    """
    input_data: [[income, spending_score]]
    """
    
    # Step 1: Scale
    scaled = scaler.transform(input_data)
    
    # Step 2: PCA
    reduced = pca.transform(scaled)
    
    # Step 3: Predict cluster
    cluster = kmeans.predict(reduced)
    
    return cluster[0]