import os
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import numpy as np
from tqdm import tqdm
import pandas as pd
from models.autoencoder import Autoencoder, VariationalAutoencoder
from data_processing.data_loader import get_data_loaders, load_tabular_data
from utils.losses import mse_loss, bce_loss, vae_loss, CustomLoss
from utils.visualization import plot_reconstructions, plot_latent_space, plot_training_loss

# Install transformers --> Huggingface
# pip3 install transformers datasets tokenizers accelerate
from transformers import Trainer, TrainingArguments
from datasets import Dataset

# Set random seed for reproducibility
torch.manual_seed(42)
np.random.seed(42)

# Deepseeks Transformer Model
def train_transformer(model, train_loader, val_loader, optimizer, loss_fn, 
                      device, epochs=100, save_dir='./logs', model_name='transformer',
                      is_variational=False):
    
    df = pd.read_csv("your_tennis_data.csv")
    # Convert dataframe to datasets format
    feature_cols = [col for col in df.columns if col != 'Winner']
    label_col = 'Winner'

    # Split data into train and validation sets
    train_df = df.iloc[:int(0.8*len(df))]
    val_df = df.iloc[int(0.8*len(df)):]

    # Convert pandas DataFrames to HuggingFace Datasets
    train_dataset = Dataset.from_pandas(train_df)
    val_dataset = Dataset.from_pandas(val_df)
    
    # Preprocess function to separate features and labels
    def preprocess_function(examples):
        # Extract features and convert to appropriate format
        features = {col: examples[col] for col in feature_cols}
        # Extract and convert labels
        labels = examples[label_col]
        return {"features": features, "labels": labels}
    
    # Apply preprocessing to datasets
    train_dataset = train_dataset.map(preprocess_function, batched=True)
    val_dataset = val_dataset.map(preprocess_function, batched=True)

    # Setup training arguments
    training_args = TrainingArguments(
        output_dir="./results",
        num_train_epochs=3,
        per_device_train_batch_size=4,
        per_device_eval_batch_size=4,
        warmup_steps=500,
        weight_decay=0.01,
        logging_dir="./logs",
    )

    # Setup trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
    )

    # Fine-tune the model
    trainer.train()



def train_autoencoder(model, train_loader, val_loader, optimizer, loss_fn, 
                      device, epochs=100, save_dir='./logs', model_name='autoencoder',
                      is_variational=False):
    """
    Train the autoencoder model.
    
    Args:
        model: Autoencoder model
        train_loader: Training data loader
        val_loader: Validation data loader
        optimizer: Optimizer
        loss_fn: Loss function
        device: Device to train on
        epochs: Number of training epochs
        save_dir: Directory to save model checkpoints and logs
        model_name: Name of the model
        is_variational: Whether the model is a variational autoencoder
    
    Returns:
        Trained model and training history
    """
    # Create save directory if it doesn't exist
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    
    # Training history
    history = {
        'train_loss': [],
        'val_loss': []
    }
    
    # Move model to device
    model = model.to(device)
    
    # Training loop
    for epoch in range(epochs):
        # Training
        model.train()
        train_loss = 0
        
        for batch_idx, (data, _) in enumerate(tqdm(train_loader, desc=f'Epoch {epoch+1}/{epochs} [Train]')):
            # Move data to device
            if isinstance(data, list):
                data = [d.to(device) for d in data]
            else:
                data = data.to(device)
            
            # Zero gradients
            optimizer.zero_grad()
            
            # Forward pass
            if is_variational:
                recon_batch, mu, log_var = model(data)
                loss = vae_loss(recon_batch, data, mu, log_var)
            else:
                recon_batch, latent = model(data)
                if isinstance(loss_fn, CustomLoss):
                    loss = loss_fn(recon_batch, data, latent)
                else:
                    loss = loss_fn(recon_batch, data)
            
            # Backward pass and optimization
            loss.backward()
            optimizer.step()
            
            # Update total loss
            train_loss += loss.item()
        
        # Calculate average training loss
        train_loss /= len(train_loader.dataset)
        history['train_loss'].append(train_loss)
        
        # Validation
        model.eval()
        val_loss = 0
        
        with torch.no_grad():
            for batch_idx, (data, _) in enumerate(tqdm(val_loader, desc=f'Epoch {epoch+1}/{epochs} [Val]')):
                # Move data to device
                if isinstance(data, list):
                    data = [d.to(device) for d in data]
                else:
                    data = data.to(device)
                
                # Forward pass
                if is_variational:
                    recon_batch, mu, log_var = model(data)
                    loss = vae_loss(recon_batch, data, mu, log_var)
                else:
                    recon_batch, latent = model(data)
                    if isinstance(loss_fn, CustomLoss):
                        loss = loss_fn(recon_batch, data, latent)
                    else:
                        loss = loss_fn(recon_batch, data)
                
                # Update total loss
                val_loss += loss.item()
        
        # Calculate average validation loss
        val_loss /= len(val_loader.dataset)
        history['val_loss'].append(val_loss)
        
        # Print progress
        print(f'Epoch {epoch+1}/{epochs}, Train Loss: {train_loss:.6f}, Val Loss: {val_loss:.6f}')
        
        # Save model checkpoint
        if (epoch + 1) % 10 == 0:
            checkpoint_path = os.path.join(save_dir, f'{model_name}_epoch_{epoch+1}.pt')
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_loss': train_loss,
                'val_loss': val_loss,
            }, checkpoint_path)
    
    # Save final model
    final_model_path = os.path.join(save_dir, f'{model_name}_final.pt')
    torch.save({
        'epoch': epochs,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'train_loss': history['train_loss'][-1],
        'val_loss': history['val_loss'][-1],
    }, final_model_path)
    
    return model, history


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Train Autoencoder Model')
    
    # Data arguments
    parser.add_argument('--data_dir', type=str, default='./data', help='Data directory')
    parser.add_argument('--data_type', type=str, choices=['image', 'tabular'], default='image', help='Type of data')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size')
    
    # Model arguments
    parser.add_argument('--model_type', type=str, choices=['ae', 'vae', 'rf'], default='rf', help='Type of model')
    parser.add_argument('--input_dim', type=int, default=784, help='Input dimension (for tabular data)')
    parser.add_argument('--hidden_dims', type=str, default='128,64', help='Hidden dimensions (comma-separated)')
    parser.add_argument('--latent_dim', type=int, default=32, help='Latent dimension')
    
    # Training arguments
    parser.add_argument('--epochs', type=int, default=100, help='Number of epochs')
    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--loss', type=str, choices=['mse', 'bce', 'custom'], default='mse', help='Loss function')
    
    # Output arguments
    parser.add_argument('--save_dir', type=str, default='./logs', help='Directory to save model checkpoints and logs')
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
    
    # Create save directory if it doesn't exist
    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)
    
    # Parse hidden dimensions
    hidden_dims = [int(dim) for dim in args.hidden_dims.split(',')]
    
    # Load data
    if args.data_type == 'image':
        train_loader, val_loader, test_loader = get_data_loaders(args.data_dir, args.batch_size)
        # Get input dimension from the first batch
        sample_batch, _ = next(iter(train_loader))
        input_dim = sample_batch[0].numel()
    else:  # 'tabular'
        data_path = os.path.join(args.data_dir, 'atp_tennis.csv')  # Adjust as needed
        X_train, X_val, X_test = load_tabular_data(data_path)
        train_loader = DataLoader(torch.utils.data.TensorDataset(X_train, X_train), batch_size=args.batch_size, shuffle=True)
        val_loader = DataLoader(torch.utils.data.TensorDataset(X_val, X_val), batch_size=args.batch_size)
        test_loader = DataLoader(torch.utils.data.TensorDataset(X_test, X_test), batch_size=args.batch_size)
        input_dim = args.input_dim
    
    # Create model
    if args.model_type == 'vae':
        model = VariationalAutoencoder(input_dim, hidden_dims, args.latent_dim)
        model_name = 'vae'
        is_variational = True
    else:  # 'ae'
        model = Autoencoder(input_dim, hidden_dims, args.latent_dim)
        model_name = 'ae'
        is_variational = False
    
    # Create optimizer
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    
    # Create loss function
    if args.loss == 'mse':
        loss_fn = mse_loss
    elif args.loss == 'bce':
        loss_fn = bce_loss
    else:  # 'custom'
        loss_fn = CustomLoss(alpha=1.0, beta=0.1)
    
    # Train model
    model, history = train_autoencoder(
        model, train_loader, val_loader, optimizer, loss_fn,
        device, args.epochs, args.save_dir, model_name, is_variational
    )
    
    # Plot training history
    plot_training_loss(history['train_loss'], history['val_loss'])
    
    # Save training history
    history_path = os.path.join(args.save_dir, f'{model_name}_history.pt')
    torch.save(history, history_path)
    
    # Evaluate on test set
    model.eval()
    test_loss = 0
    test_samples = []
    reconstructions = []
    latent_vectors = []
    
    with torch.no_grad():
        for batch_idx, (data, _) in enumerate(test_loader):
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
            test_loss += loss.item()
            
            # Store samples, reconstructions, and latent vectors
            if len(test_samples) < 10:
                test_samples.append(data.cpu())
                reconstructions.append(recon_batch.cpu())
                latent_vectors.append(latent.cpu())
    
    # Calculate average test loss
    test_loss /= len(test_loader.dataset)
    print(f'Test Loss: {test_loss:.6f}')
    
    # Concatenate samples, reconstructions, and latent vectors
    test_samples = torch.cat(test_samples)
    reconstructions = torch.cat(reconstructions)
    latent_vectors = torch.cat(latent_vectors)
    
    # Plot reconstructions
    plot_reconstructions(test_samples, reconstructions)
    
    # Plot latent space
    plot_latent_space(latent_vectors)


if __name__ == '__main__':
    main() 