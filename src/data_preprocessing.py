import pandas as pd
from sklearn.preprocessing import StandardScaler
import joblib

def load_and_preprocess(data_path="data/Mall_Customers.csv"):
    """
    Load dataset and apply feature scaling
    """

    df = pd.read_csv(data_path)

    # Select important features
    X = df[['Annual Income (k$)', 'Spending Score (1-100)']]

    # Standardization
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Save scaler
    joblib.dump(scaler, "models/scaler.pkl")

    return X_scaled, df