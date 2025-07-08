#!/bin/bash

# Script to evaluate autoencoder model

# Parse command line arguments
MODEL_PATH=${1}  # Path to the model checkpoint
MODEL_TYPE=${2:-ae}  # ae or vae
DATA_TYPE=${3:-image}  # image or tabular
BATCH_SIZE=${4:-32}
LATENT_DIM=${5:-32}
HIDDEN_DIMS=${6:-"128,64"}
LOSS=${7:-mse}  # mse, bce, or custom

# Check if model path is provided
if [ -z "$MODEL_PATH" ]; then
    echo "Error: Model path must be provided"
    echo "Usage: ./evaluate_model.sh <model_path> [model_type] [data_type] [batch_size] [latent_dim] [hidden_dims] [loss]"
    exit 1
fi

# Create results directory if it doesn't exist
mkdir -p results

# Run evaluation
echo "Starting evaluation with the following parameters:"
echo "Model path: $MODEL_PATH"
echo "Model type: $MODEL_TYPE"
echo "Data type: $DATA_TYPE"
echo "Batch size: $BATCH_SIZE"
echo "Latent dimension: $LATENT_DIM"
echo "Hidden dimensions: $HIDDEN_DIMS"
echo "Loss function: $LOSS"

python src/evaluate.py \
    --model_path $MODEL_PATH \
    --model_type $MODEL_TYPE \
    --data_type $DATA_TYPE \
    --batch_size $BATCH_SIZE \
    --latent_dim $LATENT_DIM \
    --hidden_dims $HIDDEN_DIMS \
    --loss $LOSS \
    --output_dir results \
    --data_dir data 