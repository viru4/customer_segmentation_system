import pandas as pd
from sklearn.preprocessing import StandardScaler
import pickle

def load_data(path):
    return pd.read_csv(path)

def preprocess_data(df):
    X = df[['Annual Income (k$)', 'Spending Score (1-100)']]
    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    pickle.dump(scaler, open('models/scaler.pkl', 'wb'))

    return X_scaled