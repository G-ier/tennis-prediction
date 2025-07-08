import torch
import torch.nn as nn
import torch.nn.functional as F


def mse_loss(outputs, targets):
    """
    Mean Squared Error loss function.
    
    Args:
        outputs: Model outputs
        targets: Target values
        
    Returns:
        MSE loss
    """
    return F.mse_loss(outputs, targets)


def bce_loss(outputs, targets):
    """
    Binary Cross-Entropy loss function.
    
    Args:
        outputs: Model outputs
        targets: Target values
        
    Returns:
        BCE loss
    """
    return F.binary_cross_entropy(outputs, targets)


def vae_loss(recon_x, x, mu, log_var):
    """
    Variational Autoencoder loss function: reconstruction loss + KL divergence.
    
    Args:
        recon_x: Reconstructed input
        x: Original input
        mu: Mean vector from the encoder
        log_var: Log variance vector from the encoder
        
    Returns:
        VAE loss
    """
    # Reconstruction loss
    recon_loss = F.binary_cross_entropy(recon_x, x, reduction='sum')
    
    # KL divergence
    kl_div = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())
    
    return recon_loss + kl_div


class CustomLoss(nn.Module):
    """
    Custom loss function for autoencoders that combines reconstruction loss 
    and a regularization term.
    """
    def __init__(self, alpha=1.0, beta=0.1):
        """
        Args:
            alpha: Weight for reconstruction loss
            beta: Weight for regularization term
        """
        super(CustomLoss, self).__init__()
        self.alpha = alpha
        self.beta = beta
        
    def forward(self, recon_x, x, latent=None):
        """
        Args:
            recon_x: Reconstructed input
            x: Original input
            latent: Latent representation from the encoder
            
        Returns:
            Custom loss value
        """
        # Reconstruction loss
        recon_loss = F.mse_loss(recon_x, x)
        
        # Regularization term (optional)
        reg_loss = 0.0
        if latent is not None and self.beta > 0:
            # L1 regularization on the latent space
            reg_loss = self.beta * torch.mean(torch.abs(latent))
        
        return self.alpha * recon_loss + reg_loss 