import torch
import os
import json
import matplotlib.pyplot as plt
from datetime import datetime
import numpy as np

def create_causal_mask(seq_len, device):
    mask = torch.triu(torch.ones(seq_len, seq_len, device=device), diagonal=1).bool()
    return mask.unsqueeze(0).unsqueeze(1)

def create_padding_mask(seq, pad_idx=0):
    mask = (seq != pad_idx).unsqueeze(1).unsqueeze(2)
    return mask

def save_training_plots(trainer, save_path='./results'):
    os.makedirs(save_path, exist_ok=True)
    
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
    
    epochs = range(1, len(trainer.train_losses) + 1)
    
    ax1.plot(epochs, trainer.train_losses, 'b-', label='Training Loss', linewidth=2)
    ax1.plot(epochs, trainer.val_losses, 'r-', label='Validation Loss', linewidth=2)
    ax1.set_title('Training and Validation Loss')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    ax2.plot(epochs, trainer.train_ppls, 'b-', label='Training Perplexity', linewidth=2)
    ax2.plot(epochs, trainer.val_ppls, 'r-', label='Validation Perplexity', linewidth=2)
    ax2.set_title('Training and Validation Perplexity')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Perplexity')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    steps = range(len(trainer.learning_rates))
    ax3.plot(steps, trainer.learning_rates, 'g-', linewidth=2)
    ax3.set_title('Learning Rate Schedule')
    ax3.set_xlabel('Training Step')
    ax3.set_ylabel('Learning Rate')
    ax3.grid(True, alpha=0.3)
    ax3.set_yscale('log')
    
    categories = ['Training', 'Validation']
    final_losses = [trainer.train_losses[-1], trainer.val_losses[-1]]
    final_ppls = [trainer.train_ppls[-1], trainer.val_ppls[-1]]
    
    x = np.arange(len(categories))
    width = 0.35
    
    bars1 = ax4.bar(x - width/2, final_losses, width, label='Loss', alpha=0.7)
    bars2 = ax4.bar(x + width/2, final_ppls, width, label='Perplexity', alpha=0.7)
    
    ax4.set_title('Final Results Comparison')
    ax4.set_xlabel('Dataset')
    ax4.set_ylabel('Values')
    ax4.set_xticks(x)
    ax4.set_xticklabels(categories)
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    chart_path = os.path.join(save_path, f'training_results_{timestamp}.png')
    plt.savefig(chart_path, dpi=300, bbox_inches='tight')
    plt.show()
    
    print(f"Training charts saved to: {chart_path}")
    return chart_path