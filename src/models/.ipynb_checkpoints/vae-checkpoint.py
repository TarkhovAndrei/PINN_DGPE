import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.func import jvp
from torch.utils.data import DataLoader, TensorDataset, Dataset
from tqdm.notebook import tqdm
from IPython.display import clear_output
from sklearn.model_selection import train_test_split


class VAE(nn.Module):
    """
    Variational Autoencoder for learning latent representations of initial conditions.
    
    Args:
        input_dim: dimension of the wavefunction (2*V for real and imaginary parts)
        latent_dim: dimension of the latent space
        hidden_dim: dimension of hidden layers
    """
    
    def __init__(self, input_dim, latent_dim=32, hidden_dim=256):
        super(VAE, self).__init__()
        
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        
        # Encoder
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        self.fc_mu = nn.Linear(hidden_dim, latent_dim)
        self.fc_logvar = nn.Linear(hidden_dim, latent_dim)
        
        # Decoder
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, input_dim),
        )
    
    def encode(self, x):
        """Encode input to latent distribution parameters."""
        h = self.encoder(x)
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        return mu, logvar
    
    def reparameterize(self, mu, logvar):
        """Reparameterization trick for sampling."""
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std
    
    def decode(self, z):
        """Decode latent vector to wavefunction."""
        return self.decoder(z)
    
    def forward(self, x):
        """Full forward pass: encode, sample, decode."""
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar
    
    def sample(self, num_samples=1, device='cpu'):
        """Sample new initial conditions from the prior."""
        z = torch.randn(num_samples, self.latent_dim, device=device)
        return self.decode(z)
    
    def loss_function(self, recon_x, x, mu, logvar, beta_vae=1.0):
        """VAE loss: reconstruction + KL divergence."""
        recon_loss = F.mse_loss(recon_x, x, reduction='sum')
        kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        return recon_loss + beta_vae * kl_loss, recon_loss, kl_loss

