import torch
import torch.nn as nn

class Autoencoder(nn.Module):
    """
    Autoencoder for dimensionality reduction
    
    Learns compressed representation (latent space)
    and reconstructs input from it.
    """

    def __init__(self, input_dim):
        super(Autoencoder, self).__init__()

        # Encoder: compress input
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 2)  # latent space (2D for visualization)
        )

        # Decoder: reconstruct input
        self.decoder = nn.Sequential(
            nn.Linear(2, 32),
            nn.ReLU(),
            nn.Linear(32, 64),
            nn.ReLU(),
            nn.Linear(64, input_dim)
        )

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded