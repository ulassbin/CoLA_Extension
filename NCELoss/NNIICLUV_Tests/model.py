import torch
import torch.nn as nn
import torch.nn.functional as F
import copy
import random

def reparametrize(mu,logvar, training=True):
    if(not training):
       return mu
    std = torch.exp(0.5*logvar)
    eps = torch.randn_like(std)
    return mu + eps * std

class VAE_Simple(nn.Module):
    def __init__(self, feature_dim=2048, latent_dim=128):
        super(VAE_Simple, self).__init__()
        self.fc = nn.Linear(feature_dim, 128)
        self.mu = nn.Linear(128, latent_dim)
        self.logvar = nn.Linear(128, latent_dim)
        self.activation = nn.LeakyReLU(0.2)

    def forward(self, x):
        h = self.activation(self.fc(x))
        mu = self.mu(h)
        logvar = self.logvar(h)
        return mu, logvar, reparametrize(mu, logvar, self.training) # just handle these here!


class NearestNeighborContrastiveI3D(nn.Module):
    def __init__(self, feature_dim=2048, projection_dim=128, time_steps=1000):
        super(NearestNeighborContrastiveI3D, self).__init__()

        # Intra-video projection head
        self.intra_projector = VAE_Simple(feature_dim, projection_dim)

        # Inter-video projection head
        #self.video_attention = nn.Linear(feature_dim, 1) # Attention to do this
        self.time_conv = nn.Conv1d(in_channels=(time_steps), out_channels=1, kernel_size=3, padding=1)
        self.inter_projector = VAE_Simple(feature_dim, projection_dim)

        # Now make a decoder for the intra and inter projections
        self.intra_decoder = nn.Sequential(
            nn.Linear(projection_dim, feature_dim),
            nn.LeakyReLU(),
            nn.Linear(feature_dim, feature_dim)
        )

        self.inter_time_embed = nn.Parameter(torch.randn(time_steps, projection_dim))
        self.layer_norm = nn.LayerNorm(projection_dim)
        self.inter_decoder = nn.Sequential(
            nn.Linear(projection_dim, feature_dim),
            nn.LeakyReLU(),
            nn.Linear(feature_dim, feature_dim)
        )

    def forward(self, features):
        # Pass features through projection heads
        # features are batchxtimexdim size
        mu_intra, logvar_intra, intra_embeddings = self.intra_projector(features)
        batch, time, dims = features.shape
        batch, time, latent_dims = mu_intra.shape

        pooled_features = self.time_conv(features).reshape(batch, dims) # batchxlatent_dimsx1
        mu_inter, logvar_inter, raw_inter_embeddings = self.inter_projector(pooled_features) # batchxdim
        # Repeat everything
        
        # Decoding...
        decoded_intra = self.intra_decoder(intra_embeddings)
        # Add pos encoding
        inter_embeddings = raw_inter_embeddings.unsqueeze(1).repeat(1, time, 1) + self.inter_time_embed.unsqueeze(0)  # batchxtimexlatent_dims
        inter_embeddings = self.layer_norm(inter_embeddings)
        decoded_inter = self.inter_decoder(inter_embeddings) # batchxtimexlatent_dims

        return intra_embeddings, raw_inter_embeddings, decoded_inter, decoded_intra, [mu_intra, logvar_intra], [mu_inter, logvar_inter]

    def from_latent_space(self, latent_features, both=False):
        # Pass features through projection heads
        # Given latent features, decode them to get the original features and original feature size.
        if both:
            decoded_intra = self.intra_decoder(latent_features[0])
            decoded_inter = self.inter_decoder(latent_features[1])
        else:
            decoded_intra = self.intra_decoder(latent_features)
            decoded_inter = decoded_intra
        return decoded_inter, decoded_intra
  
