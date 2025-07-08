#!/bin/bash

# Script to train autoencoder model

# Parse command line arguments
MODEL_TYPE=${1:-ae}  # ae or vae
DATA_TYPE=${2:-image}  # image or tabular
EPOCHS=${3:-100}
BATCH_SIZE=${4:-32}
LATENT_DIM=${5:-32}
HIDDEN_DIMS=${6:-"128,64"}
LEARNING_RATE=${7:-0.001}
LOSS=${8:-mse}  # mse, bce, or custom

# Create logs directory if it doesn't exist
mkdir -p logs

# Run training
echo "Starting training with the following parameters:"
echo "Model type: $MODEL_TYPE"
echo "Data type: $DATA_TYPE"
echo "Epochs: $EPOCHS"
echo "Batch size: $BATCH_SIZE"
echo "Latent dimension: $LATENT_DIM"
echo "Hidden dimensions: $HIDDEN_DIMS"
echo "Learning rate: $LEARNING_RATE"
echo "Loss function: $LOSS"

python src/train.py \
    --model_type $MODEL_TYPE \
    --data_type $DATA_TYPE \
    --epochs $EPOCHS \
    --batch_size $BATCH_SIZE \
    --latent_dim $LATENT_DIM \
    --hidden_dims $HIDDEN_DIMS \
    --lr $LEARNING_RATE \
    --loss $LOSS \
    --save_dir logs \
    --data_dir data 