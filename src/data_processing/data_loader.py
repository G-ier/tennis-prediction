import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image

class CustomDataset(Dataset):
    def __init__(self, data_dir, transform=None):
        """
        Args:
            data_dir (str): Directory with the data files
            transform (callable, optional): Optional transform to be applied on samples
        """
        self.data_dir = data_dir
        self.transform = transform
        self.samples = self._load_data()
        
    def _load_data(self):
        """
        Load the data from the data directory.
        Modify this method based on your specific data format.
        """
        # Example implementation for image files
        samples = []
        for file in os.listdir(self.data_dir):
            if file.endswith(('.jpg', '.png', '.jpeg')):
                samples.append(os.path.join(self.data_dir, file))
        return samples
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        # Example implementation for image files
        img_path = self.samples[idx]
        img = Image.open(img_path).convert('RGB')
        
        if self.transform:
            img = self.transform(img)
            
        return img, img  # Return input and target as the same for autoencoder


def get_data_loaders(data_dir, batch_size=32, transform=None, train_ratio=0.8, val_ratio=0.1):
    """
    Create train, validation, and test data loaders.
    
    Args:
        data_dir (str): Directory with the data files
        batch_size (int): Batch size for the data loaders
        transform (callable, optional): Transform to be applied on samples
        train_ratio (float): Ratio of training data
        val_ratio (float): Ratio of validation data
        
    Returns:
        train_loader, val_loader, test_loader
    """
    if transform is None:
        transform = transforms.Compose([
            transforms.Resize((128, 128)),
            transforms.ToTensor(),
        ])
    
    # Create dataset
    dataset = CustomDataset(data_dir, transform)
    
    # Split dataset
    total_size = len(dataset)
    train_size = int(total_size * train_ratio)
    val_size = int(total_size * val_ratio)
    test_size = total_size - train_size - val_size
    
    train_dataset, val_dataset, test_dataset = torch.utils.data.random_split(
        dataset, [train_size, val_size, test_size]
    )
    
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=4)
    
    return train_loader, val_loader, test_loader


def load_tabular_data(data_path, target_column=None, test_size=0.2, val_size=0.1, normalize=True):
    """
    Load and prepare tabular data for autoencoder.
    
    Args:
        data_path (str): Path to the CSV/Excel file
        target_column (str, optional): Name of the target column (if any)
        test_size (float): Ratio of test data
        val_size (float): Ratio of validation data
        normalize (bool): Whether to normalize the data
        
    Returns:
        X_train, X_val, X_test, (y_train, y_val, y_test) if target_column is not None
    """
    import pandas as pd
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import MinMaxScaler
    
    # Load data
    if data_path.endswith('.csv'):
        df = pd.read_csv(data_path)
    elif data_path.endswith(('.xls', '.xlsx')):
        df = pd.read_excel(data_path)
    else:
        raise ValueError("Unsupported file format. Use CSV or Excel files.")
    
    # Split features and target
    if target_column:
        X = df.drop(target_column, axis=1)
        y = df[target_column]
        has_target = True
    else:
        X = df
        has_target = False
    
    # Split into train, validation, and test sets
    if has_target:
        X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=test_size+val_size, random_state=42)
        val_ratio = val_size / (test_size + val_size)
        X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=1-val_ratio, random_state=42)
    else:
        X_train, X_temp = train_test_split(X, test_size=test_size+val_size, random_state=42)
        val_ratio = val_size / (test_size + val_size)
        X_val, X_test = train_test_split(X_temp, test_size=1-val_ratio, random_state=42)
    
    # Normalize data
    if normalize:
        scaler = MinMaxScaler()
        X_train = scaler.fit_transform(X_train)
        X_val = scaler.transform(X_val)
        X_test = scaler.transform(X_test)
    
    # Convert to PyTorch tensors
    X_train = torch.FloatTensor(X_train)
    X_val = torch.FloatTensor(X_val)
    X_test = torch.FloatTensor(X_test)
    
    if has_target:
        y_train = torch.FloatTensor(y_train.values)
        y_val = torch.FloatTensor(y_val.values)
        y_test = torch.FloatTensor(y_test.values)
        return X_train, X_val, X_test, y_train, y_val, y_test
    
    return X_train, X_val, X_test 