import os
import random
import numpy as np
import torch
import json
import matplotlib.pyplot as plt
from datetime import datetime


def set_seed(seed):
    """
    Set random seed for reproducibility.
    
    Args:
        seed: Random seed
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    os.environ['PYTHONHASHSEED'] = str(seed)


def save_config(config, save_dir, filename='config.json'):
    """
    Save configuration to a JSON file.
    
    Args:
        config: Configuration dictionary
        save_dir: Directory to save the file
        filename: Name of the file
    """
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    
    config_path = os.path.join(save_dir, filename)
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=4)


def load_config(config_path):
    """
    Load configuration from a JSON file.
    
    Args:
        config_path: Path to the configuration file
        
    Returns:
        Configuration dictionary
    """
    with open(config_path, 'r') as f:
        config = json.load(f)
    
    return config


def create_experiment_dir(base_dir='./logs'):
    """
    Create a new experiment directory with a timestamp.
    
    Args:
        base_dir: Base directory for experiments
        
    Returns:
        Path to the new experiment directory
    """
    if not os.path.exists(base_dir):
        os.makedirs(base_dir)
    
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    experiment_dir = os.path.join(base_dir, f'experiment_{timestamp}')
    os.makedirs(experiment_dir)
    
    return experiment_dir


def count_parameters(model):
    """
    Count the number of trainable parameters in the model.
    
    Args:
        model: PyTorch model
        
    Returns:
        Number of trainable parameters
    """
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def save_loss_plot(train_losses, val_losses=None, save_path=None):
    """
    Save a plot of training and validation losses.
    
    Args:
        train_losses: List of training losses
        val_losses: List of validation losses (optional)
        save_path: Path to save the plot
    """
    plt.figure(figsize=(10, 6))
    
    plt.plot(train_losses, label='Training Loss', color='blue')
    if val_losses is not None:
        plt.plot(val_losses, label='Validation Loss', color='red')
    
    plt.title('Loss Over Epochs')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path)
    else:
        plt.savefig('loss_plot.png')
    
    plt.close()


def save_model_summary(model, save_path=None):
    """
    Save a summary of the model architecture and parameters.
    
    Args:
        model: PyTorch model
        save_path: Path to save the summary
    """
    summary = []
    
    # Add model class name
    summary.append(f"Model: {model.__class__.__name__}")
    
    # Add number of trainable parameters
    n_params = count_parameters(model)
    summary.append(f"Trainable parameters: {n_params:,}")
    
    # Add model structure
    summary.append("\nModel structure:")
    
    # Print layers
    for name, module in model.named_children():
        summary.append(f"\n{name}:")
        if isinstance(module, torch.nn.Sequential):
            for idx, layer in enumerate(module):
                summary.append(f"  {idx}: {layer}")
        else:
            summary.append(f"  {module}")
    
    # Save or print summary
    if save_path:
        with open(save_path, 'w') as f:
            f.write('\n'.join(summary))
    else:
        with open('model_summary.txt', 'w') as f:
            f.write('\n'.join(summary))


def get_device():
    """
    Get the device to use for training.
    
    Returns:
        PyTorch device
    """
    if torch.cuda.is_available():
        device = torch.device('cuda')
        print(f"Using GPU: {torch.cuda.get_device_name(0)}")
    else:
        device = torch.device('cpu')
        print("Using CPU")
    
    return device 