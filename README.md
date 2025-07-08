# Autoencoder-Based Model
[![Buy Me A Coffee](https://img.shields.io/badge/Buy%20Me%20a%20Coffee-FFDD00?style=flat-square&logo=buy-me-a-coffee&logoColor=000000)](https://coff.ee/gier)

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
- Baseline Random Forest classifier for supervised tennis match prediction

## Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd tennis-prediction
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

3. **Random Forest Classifier (RF)**: A supervised learning baseline that predicts match outcomes using engineered tennis features.

## Random Forest Model (Baseline)

While the core of this repository focuses on unsupervised representation learning with autoencoders, it also ships with a conventional supervised baseline built around a **Random Forest classifier**.

### Training the Random Forest

The training script ingests the prepared tennis dataset, splits it into train ⁄ test subsets, and fits a `sklearn.ensemble.RandomForestClassifier`.

```bash
# From the project root
python src/rf/rf_training.py
```

Key parameters such as the number of trees (`n_estimators`) or maximum tree depth (`max_depth`) can be edited at the top of `src/rf/rf_training.py` or adapted to a CLI if needed.  
By default the trained model is saved to `src/training/models/random_forest_model.joblib`.

### Generating Predictions

After a model has been trained (or if you have provided a pre-trained `random_forest_model.joblib`), predictions on fresh data can be generated with:

```bash
python src/rf/predict_with_rf.py
```

The script outputs two CSVs in the `data/` folder:

* `rf_predictions.csv` – row-wise winner predictions with confidence scores.

The prediction script conveniently merges predictions back with player names for easy human inspection.

Feel free to tweak either script to point to different datasets or model locations as required.

## Support

If you find this project useful, please consider supporting me by buying me a coffee:<br/><br/>
<a href="https://coff.ee/gier" target="_blank"><img src="https://cdn.buymeacoffee.com/buttons/v2/default-yellow.png" alt="Buy Me A Coffee" width="220" ></a>