# Autoencoder-Based Model
Support my work: [Buy me a coffee](https://coff.ee/gier)

This project implements an autoencoder-based model for unsupervised learning and dimensionality reduction. It provides a flexible framework for training and evaluating different types of autoencoders on various datasets.

## Project Structure

```
.
├── data/                   # Data directory (add your data here)
├── logs/                   # Training logs and model checkpoints
├── notebooks/              # Jupyter notebooks for exploration and visualization
├── results/                # Evaluation results
├── scripts/                # Utility scripts for training and evaluation
│   ├── train_model.sh      # Script to train the model
│   └── evaluate_model.sh   # Script to evaluate the model
├── src/                    # Source code
│   ├── data_processing/    # Data processing modules
│   │   └── data_loader.py  # Data loaders for different data types
│   ├── models/             # Model definitions
│   │   └── autoencoder.py  # Autoencoder model architectures
│   ├── utils/              # Utility functions
│   │   ├── losses.py       # Loss functions
│   │   ├── utils.py        # General utility functions
│   │   └── visualization.py # Visualization utilities
│   ├── train.py            # Training script
│   └── evaluate.py         # Evaluation script
├── requirements.txt        # Project dependencies
└── README.md               # Project documentation
```

## Features

- Support for both standard Autoencoders and Variational Autoencoders (VAEs)
- Flexible model architecture with configurable hidden layers and latent dimensions
- Support for both image and tabular data
- Multiple loss functions (MSE, BCE, custom)
- Visualization utilities for reconstructions and latent space
- Model saving and loading functionality
- Command-line interface for training and evaluation

## Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd tn
```

2. Create and activate a virtual environment (optional):
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install the dependencies:
```bash
pip install -r requirements.txt
```

## Usage

### Data Preparation

Place your data in the `data/` directory. The framework supports:
- Image data: Place image files in subdirectories of `data/`
- Tabular data: Place CSV/Excel files in `data/`

### Training

To train a model, you can use the provided script:

```bash
bash scripts/train_model.sh [model_type] [data_type] [epochs] [batch_size] [latent_dim] [hidden_dims] [learning_rate] [loss]
```

Example:
```bash
bash scripts/train_model.sh ae image 100 32 32 "128,64" 0.001 mse
```

Alternatively, you can run the training script directly:

```bash
python src/train.py --model_type ae --data_type image --epochs 100 --batch_size 32 --latent_dim 32 --hidden_dims 128,64 --lr 0.001 --loss mse --save_dir logs --data_dir data
```

### Evaluation

To evaluate a trained model, use the provided script:

```bash
bash scripts/evaluate_model.sh [model_path] [model_type] [data_type] [batch_size] [latent_dim] [hidden_dims] [loss]
```

Example:
```bash
bash scripts/evaluate_model.sh logs/ae_final.pt ae image 32 32 "128,64" mse
```

Alternatively, you can run the evaluation script directly:

```bash
python src/evaluate.py --model_path logs/ae_final.pt --model_type ae --data_type image --batch_size 32 --latent_dim 32 --hidden_dims 128,64 --loss mse --output_dir results --data_dir data
```

## Model Types

The framework supports the following types of autoencoders:

1. **Standard Autoencoder (AE)**: A neural network that learns to encode the input data into a lower-dimensional latent space and then decode it back to the original space, minimizing reconstruction error.

2. **Variational Autoencoder (VAE)**: An extension of the standard autoencoder that adds a probabilistic element to the latent space, enabling generative capabilities.

## Support

If you find this project useful, please consider supporting me by buying me a coffee: https://coff.ee/gier