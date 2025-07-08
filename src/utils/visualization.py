import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.manifold import TSNE
import seaborn as sns
from sklearn.decomposition import PCA


def plot_reconstructions(original, reconstructed, n=10, figsize=(12, 6)):
    """
    Plot original images and their reconstructions side by side.
    
    Args:
        original: Original images
        reconstructed: Reconstructed images
        n: Number of images to plot
        figsize: Figure size
    """
    if isinstance(original, torch.Tensor):
        original = original.detach().cpu().numpy()
    if isinstance(reconstructed, torch.Tensor):
        reconstructed = reconstructed.detach().cpu().numpy()
    
    plt.figure(figsize=figsize)
    
    for i in range(min(n, len(original))):
        # Original
        ax = plt.subplot(2, n, i + 1)
        if original[i].shape[0] == 1 or len(original[i].shape) == 1:  # Grayscale or 1D
            if len(original[i].shape) > 1:  # Grayscale image
                plt.imshow(original[i].reshape(original[i].shape[1], original[i].shape[2]), cmap='gray')
            else:  # 1D data
                plt.plot(original[i])
        else:  # RGB
            plt.imshow(np.transpose(original[i], (1, 2, 0)))
        plt.title("Original")
        plt.axis('off')
        
        # Reconstructed
        ax = plt.subplot(2, n, i + 1 + n)
        if reconstructed[i].shape[0] == 1 or len(reconstructed[i].shape) == 1:  # Grayscale or 1D
            if len(reconstructed[i].shape) > 1:  # Grayscale image
                plt.imshow(reconstructed[i].reshape(reconstructed[i].shape[1], reconstructed[i].shape[2]), cmap='gray')
            else:  # 1D data
                plt.plot(reconstructed[i])
        else:  # RGB
            plt.imshow(np.transpose(reconstructed[i], (1, 2, 0)))
        plt.title("Reconstructed")
        plt.axis('off')
    
    plt.tight_layout()
    plt.show()


def plot_latent_space(latent_vectors, labels=None, method='tsne', perplexity=30, figsize=(10, 8)):
    """
    Visualize the latent space using dimensionality reduction techniques.
    
    Args:
        latent_vectors: Latent vectors from the encoder
        labels: Labels for the data points (if available)
        method: Dimensionality reduction method ('tsne' or 'pca')
        perplexity: Perplexity for t-SNE
        figsize: Figure size
    """
    if isinstance(latent_vectors, torch.Tensor):
        latent_vectors = latent_vectors.detach().cpu().numpy()
    
    plt.figure(figsize=figsize)
    
    # Apply dimensionality reduction
    if latent_vectors.shape[1] > 2:
        if method.lower() == 'tsne':
            reducer = TSNE(n_components=2, perplexity=perplexity, random_state=42)
            reduced_data = reducer.fit_transform(latent_vectors)
        elif method.lower() == 'pca':
            reducer = PCA(n_components=2)
            reduced_data = reducer.fit_transform(latent_vectors)
        else:
            raise ValueError(f"Unknown method: {method}. Use 'tsne' or 'pca'.")
    else:
        reduced_data = latent_vectors
    
    # Plot
    if labels is not None:
        if isinstance(labels, torch.Tensor):
            labels = labels.detach().cpu().numpy()
        
        scatter = plt.scatter(reduced_data[:, 0], reduced_data[:, 1], c=labels, cmap='viridis', alpha=0.8)
        plt.colorbar(scatter, label='Labels')
    else:
        plt.scatter(reduced_data[:, 0], reduced_data[:, 1], alpha=0.8)
    
    plt.title(f"Latent Space Visualization ({method.upper()})")
    plt.xlabel("Dimension 1")
    plt.ylabel("Dimension 2")
    plt.tight_layout()
    plt.show()


def plot_training_loss(train_losses, val_losses=None, figsize=(10, 6)):
    """
    Plot training and validation losses over epochs.
    
    Args:
        train_losses: List of training losses
        val_losses: List of validation losses (if available)
        figsize: Figure size
    """
    plt.figure(figsize=figsize)
    
    plt.plot(train_losses, label='Training Loss', color='blue')
    if val_losses is not None:
        plt.plot(val_losses, label='Validation Loss', color='red')
    
    plt.title('Loss Over Epochs')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.show()


def plot_feature_importance(model, feature_names=None, top_n=10, figsize=(12, 6)):
    """
    Plot feature importance based on the autoencoder's latent space.
    
    Args:
        model: Trained autoencoder model
        feature_names: Names of the input features
        top_n: Number of top features to display
        figsize: Figure size
    """
    # Get the weights of the encoder's first layer
    weights = model.encoder[0].weight.detach().cpu().numpy()
    
    # Calculate feature importance as the sum of absolute weights
    importance = np.sum(np.abs(weights), axis=0)
    
    # Create feature names if not provided
    if feature_names is None:
        feature_names = [f"Feature {i}" for i in range(len(importance))]
    
    # Sort by importance
    sorted_idx = np.argsort(importance)[::-1]
    top_idx = sorted_idx[:top_n]
    
    # Plot
    plt.figure(figsize=figsize)
    sns.barplot(x=importance[top_idx], y=[feature_names[i] for i in top_idx])
    plt.title(f"Top {top_n} Feature Importance")
    plt.xlabel("Importance")
    plt.ylabel("Feature")
    plt.tight_layout()
    plt.show() 