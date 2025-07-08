import torch
import torch.nn as nn

class Autoencoder(nn.Module):
    def __init__(self, input_dim, hidden_dims=None, latent_dim=32):
        super(Autoencoder, self).__init__()
        
        if hidden_dims is None:
            hidden_dims = [128, 64]
        
        # Encoder
        encoder_layers = []
        encoder_layers.append(nn.Linear(input_dim, hidden_dims[0]))
        encoder_layers.append(nn.ReLU())
        
        for i in range(len(hidden_dims) - 1):
            encoder_layers.append(nn.Linear(hidden_dims[i], hidden_dims[i + 1]))
            encoder_layers.append(nn.ReLU())
        
        encoder_layers.append(nn.Linear(hidden_dims[-1], latent_dim))
        self.encoder = nn.Sequential(*encoder_layers)
        
        # Decoder
        decoder_layers = []
        decoder_layers.append(nn.Linear(latent_dim, hidden_dims[-1]))
        decoder_layers.append(nn.ReLU())
        
        for i in range(len(hidden_dims) - 1, 0, -1):
            decoder_layers.append(nn.Linear(hidden_dims[i], hidden_dims[i - 1]))
            decoder_layers.append(nn.ReLU())
        
        decoder_layers.append(nn.Linear(hidden_dims[0], input_dim))
        decoder_layers.append(nn.Sigmoid())  # Assuming data is normalized between 0 and 1
        self.decoder = nn.Sequential(*decoder_layers)
    
    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded, encoded
    
    def encode(self, x):
        return self.encoder(x)
    
    def decode(self, z):
        return self.decoder(z)


class VariationalAutoencoder(nn.Module):
    def __init__(self, input_dim, hidden_dims=None, latent_dim=32):
        super(VariationalAutoencoder, self).__init__()
        
        if hidden_dims is None:
            hidden_dims = [128, 64]
        
        # Encoder
        encoder_layers = []
        encoder_layers.append(nn.Linear(input_dim, hidden_dims[0]))
        encoder_layers.append(nn.ReLU())
        
        for i in range(len(hidden_dims) - 1):
            encoder_layers.append(nn.Linear(hidden_dims[i], hidden_dims[i + 1]))
            encoder_layers.append(nn.ReLU())
        
        self.encoder = nn.Sequential(*encoder_layers)
        self.mu = nn.Linear(hidden_dims[-1], latent_dim)
        self.log_var = nn.Linear(hidden_dims[-1], latent_dim)
        
        # Decoder
        decoder_layers = []
        decoder_layers.append(nn.Linear(latent_dim, hidden_dims[-1]))
        decoder_layers.append(nn.ReLU())
        
        for i in range(len(hidden_dims) - 1, 0, -1):
            decoder_layers.append(nn.Linear(hidden_dims[i], hidden_dims[i - 1]))
            decoder_layers.append(nn.ReLU())
        
        decoder_layers.append(nn.Linear(hidden_dims[0], input_dim))
        decoder_layers.append(nn.Sigmoid())  # Assuming data is normalized between 0 and 1
        self.decoder = nn.Sequential(*decoder_layers)
    
    def encode(self, x):
        h = self.encoder(x)
        return self.mu(h), self.log_var(h)
    
    def reparameterize(self, mu, log_var):
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return mu + eps * std
    
    def decode(self, z):
        return self.decoder(z)
    
    def forward(self, x):
        mu, log_var = self.encode(x)
        z = self.reparameterize(mu, log_var)
        return self.decode(z), mu, log_var 