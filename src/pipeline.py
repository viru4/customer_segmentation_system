import joblib
import torch
from src.autoencoder import Autoencoder

def apply_pca(X):
    """
    Apply PCA dimensionality reduction
    """
    pca = joblib.load("models/pca.pkl")
    return pca.transform(X)


def apply_autoencoder(X):
    """
    Apply Autoencoder dimensionality reduction
    """

    model = Autoencoder(input_dim=X.shape[1])
    model.load_state_dict(torch.load("models/autoencoder.pth"))
    model.eval()

    X_tensor = torch.tensor(X, dtype=torch.float32)

    with torch.no_grad():
        encoded = model.encoder(X_tensor)

    return encoded.numpy()