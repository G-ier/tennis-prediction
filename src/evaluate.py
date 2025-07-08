import os
import argparse
import torch
import numpy as np
from torch.utils.data import DataLoader
from tqdm import tqdm

from models.autoencoder import Autoencoder, VariationalAutoencoder
from data_processing.data_loader import get_data_loaders, load_tabular_data
from utils.losses import mse_loss, bce_loss, vae_loss, CustomLoss
from utils.visualization import plot_reconstructions, plot_latent_space, plot_feature_importance

# Set random seed for reproducibility
torch.manual_seed(42)
np.random.seed(42)


def evaluate_autoencoder(model, data_loader, loss_fn, device, is_variational=False):
    """
    Evaluate the autoencoder model.
    
    Args:
        model: Autoencoder model
        data_loader: Data loader
        loss_fn: Loss function
        device: Device to evaluate on
        is_variational: Whether the model is a variational autoencoder
        
    Returns:
        Loss, samples, reconstructions, latent vectors
    """
    model.eval()
    total_loss = 0
    samples = []
    reconstructions = []
    latent_vectors = []
    
    with torch.no_grad():
        for batch_idx, (data, _) in enumerate(tqdm(data_loader, desc='Evaluating')):
            # Move data to device
            if isinstance(data, list):
                data = [d.to(device) for d in data]
            else:
                data = data.to(device)
            
            # Forward pass
            if is_variational:
                recon_batch, mu, log_var = model(data)
                loss = vae_loss(recon_batch, data, mu, log_var)
                latent = mu  # Use mean as latent representation for VAE
            else:
                recon_batch, latent = model(data)
                if isinstance(loss_fn, CustomLoss):
                    loss = loss_fn(recon_batch, data, latent)
                else:
                    loss = loss_fn(recon_batch, data)
            
            # Update total loss
            total_loss += loss.item()
            
            # Store samples, reconstructions, and latent vectors
            samples.append(data.cpu())
            reconstructions.append(recon_batch.cpu())
            latent_vectors.append(latent.cpu())
    
    # Calculate average loss
    loss = total_loss / len(data_loader.dataset)
    
    # Concatenate samples, reconstructions, and latent vectors
    samples = torch.cat(samples)
    reconstructions = torch.cat(reconstructions)
    latent_vectors = torch.cat(latent_vectors)
    
    return loss, samples, reconstructions, latent_vectors


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Evaluate Autoencoder Model')
    
    # Data arguments
    parser.add_argument('--data_dir', type=str, default='./data', help='Data directory')
    parser.add_argument('--data_type', type=str, choices=['image', 'tabular'], default='image', help='Type of data')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size')
    
    # Model arguments
    parser.add_argument('--model_type', type=str, choices=['ae', 'vae'], default='ae', help='Type of autoencoder')
    parser.add_argument('--input_dim', type=int, default=784, help='Input dimension (for tabular data)')
    parser.add_argument('--hidden_dims', type=str, default='128,64', help='Hidden dimensions (comma-separated)')
    parser.add_argument('--latent_dim', type=int, default=32, help='Latent dimension')
    
    # Model loading arguments
    parser.add_argument('--model_path', type=str, required=True, help='Path to the model checkpoint')
    
    # Loss function
    parser.add_argument('--loss', type=str, choices=['mse', 'bce', 'custom'], default='mse', help='Loss function')
    
    # Output arguments
    parser.add_argument('--output_dir', type=str, default='./results', help='Directory to save evaluation results')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    
    return parser.parse_args()


def main():
    """Main function."""
    args = parse_args()
    
    # Set random seed
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')
    
    # Create output directory if it doesn't exist
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
    
    # Parse hidden dimensions
    hidden_dims = [int(dim) for dim in args.hidden_dims.split(',')]
    
    # Load data
    if args.data_type == 'image':
        _, _, test_loader = get_data_loaders(args.data_dir, args.batch_size)
        # Get input dimension from the first batch
        sample_batch, _ = next(iter(test_loader))
        input_dim = sample_batch[0].numel()
    else:  # 'tabular'
        data_path = os.path.join(args.data_dir, 'data.csv')  # Adjust as needed
        X_train, X_val, X_test = load_tabular_data(data_path)
        test_loader = DataLoader(torch.utils.data.TensorDataset(X_test, X_test), batch_size=args.batch_size)
        input_dim = args.input_dim
    
    # Create model
    if args.model_type == 'vae':
        model = VariationalAutoencoder(input_dim, hidden_dims, args.latent_dim)
        is_variational = True
    else:  # 'ae'
        model = Autoencoder(input_dim, hidden_dims, args.latent_dim)
        is_variational = False
    
    # Load model
    checkpoint = torch.load(args.model_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    
    # Create loss function
    if args.loss == 'mse':
        loss_fn = mse_loss
    elif args.loss == 'bce':
        loss_fn = bce_loss
    else:  # 'custom'
        loss_fn = CustomLoss(alpha=1.0, beta=0.1)
    
    # Evaluate model
    loss, samples, reconstructions, latent_vectors = evaluate_autoencoder(
        model, test_loader, loss_fn, device, is_variational
    )
    
    print(f'Test Loss: {loss:.6f}')
    
    # Plot reconstructions
    plot_reconstructions(samples[:10], reconstructions[:10])
    reconstructions_path = os.path.join(args.output_dir, 'reconstructions.png')
    plt.savefig(reconstructions_path)
    
    # Plot latent space
    plot_latent_space(latent_vectors)
    latent_space_path = os.path.join(args.output_dir, 'latent_space.png')
    plt.savefig(latent_space_path)
    
    # Plot feature importance (if applicable, for tabular data)
    if args.data_type == 'tabular':
        plot_feature_importance(model)
        feature_importance_path = os.path.join(args.output_dir, 'feature_importance.png')
        plt.savefig(feature_importance_path)
    
    # Save evaluation results
    results = {
        'loss': loss,
        'samples': samples.numpy(),
        'reconstructions': reconstructions.numpy(),
        'latent_vectors': latent_vectors.numpy()
    }
    torch.save(results, os.path.join(args.output_dir, 'evaluation_results.pt'))


if __name__ == '__main__':
    main() 