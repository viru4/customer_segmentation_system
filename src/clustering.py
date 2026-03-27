"""
Central place for all clustering models
"""

import joblib

def load_models():
    """
    Load all trained clustering models
    """
    models = {
        "KMeans": joblib.load("models/kmeans_model.pkl"),
        "DBSCAN": joblib.load("models/dbscan_model.pkl"),
        "GMM": joblib.load("models/gmm_model.pkl"),
        "Hierarchical": joblib.load("models/hierarchical_model.pkl"),
    }
    return models


def predict_clusters(model, data):
    """
    Predict cluster labels
    """
    return model.fit_predict(data)