#!/bin/bash

# Transformer Language Model Training Script
# Usage: ./scripts/run.sh [options]

set -e  # Exit on error

# Default parameters
EPOCHS=5
BATCH_SIZE=8
SEQ_LEN=64
LEARNING_RATE=0.0005
VOCAB_SIZE=3000
DATA_DIR="./data"
OUTPUT_DIR="./output"

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --epochs)
            EPOCHS="$2"
            shift 2
            ;;
        --batch-size)
            BATCH_SIZE="$2"
            shift 2
            ;;
        --seq-len)
            SEQ_LEN="$2"
            shift 2
            ;;
        --lr)
            LEARNING_RATE="$2"
            shift 2
            ;;
        --vocab-size)
            VOCAB_SIZE="$2"
            shift 2
            ;;
        --data-dir)
            DATA_DIR="$2"
            shift 2
            ;;
        --output-dir)
            OUTPUT_DIR="$2"
            shift 2
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

echo "Starting Transformer Language Model Training"
echo "============================================"
echo "Epochs: $EPOCHS"
echo "Batch size: $BATCH_SIZE"
echo "Sequence length: $SEQ_LEN"
echo "Learning rate: $LEARNING_RATE"
echo "Vocabulary size: $VOCAB_SIZE"
echo "Data directory: $DATA_DIR"
echo "Output directory: $OUTPUT_DIR"
echo ""

# Create directories
mkdir -p "$DATA_DIR"
mkdir -p "$OUTPUT_DIR"
mkdir -p "$OUTPUT_DIR/checkpoints"
mkdir -p "$OUTPUT_DIR/results"

# Run the training
python -c "
import sys
sys.path.append('src')
from dataset import download_wikitext2
from model import WikiText2Transformer
from trainer import WikiText2Trainer
from utils import save_training_plots
from torch.utils.data import DataLoader

# Download data
print('Downloading WikiText-2 dataset...')
download_wikitext2('$DATA_DIR')

# Create datasets
from dataset import WikiText2Dataset
train_dataset = WikiText2Dataset('$DATA_DIR/train.txt', seq_len=$SEQ_LEN, vocab_size=$VOCAB_SIZE, mode='train')
val_dataset = WikiText2Dataset('$DATA_DIR/valid.txt', seq_len=$SEQ_LEN, vocab_size=$VOCAB_SIZE, mode='valid')
test_dataset = WikiText2Dataset('$DATA_DIR/test.txt', seq_len=$SEQ_LEN, vocab_size=$VOCAB_SIZE, mode='test')

# Create data loaders
train_loader = DataLoader(train_dataset, batch_size=$BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=$BATCH_SIZE, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=$BATCH_SIZE, shuffle=False)

# Create model
model = WikiText2Transformer(
    vocab_size=len(train_dataset.word2idx),
    d_model=128,
    num_heads=4,
    d_ff=512,
    num_layers=3,
    dropout=0.1
)

# Create trainer and train
trainer = WikiText2Trainer(
    model, 
    train_loader, 
    val_loader, 
    test_loader, 
    learning_rate=$LEARNING_RATE
)

print('Starting training...')
train_losses, val_losses, test_loss, test_ppl = trainer.train(epochs=$EPOCHS)

# Save training plots
print('Generating training plots...')
chart_path = save_training_plots(trainer, '$OUTPUT_DIR/results')

# Save final model
import torch
torch.save({
    'model_state_dict': model.state_dict(),
    'train_losses': train_losses,
    'val_losses': val_losses,
    'vocab_size': len(train_dataset.word2idx),
    'config': {
        'd_model': 128,
        'num_heads': 4,
        'd_ff': 512,
        'num_layers': 3,
        'dropout': 0.1
    }
}, '$OUTPUT_DIR/checkpoints/final_model.pth')

print('Training completed successfully!')
print(f'Final Test Results - Loss: {test_loss:.4f}, Perplexity: {test_ppl:.2f}')
print(f'Results saved to: $OUTPUT_DIR')
"

echo ""
echo "Training completed!"
echo "Check $OUTPUT_DIR for results and checkpoints"