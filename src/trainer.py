import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR
import math
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

def get_cosine_schedule_with_warmup(optimizer, num_warmup_steps, num_training_steps, num_cycles=0.5):
    def lr_lambda(current_step):
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        progress = float(current_step - num_warmup_steps) / float(max(1, num_training_steps - num_warmup_steps))
        return max(0.0, 0.5 * (1.0 + math.cos(math.pi * float(num_cycles) * 2.0 * progress)))
    
    return LambdaLR(optimizer, lr_lambda)

class WikiText2Trainer:
    def __init__(self, model, train_loader, val_loader, test_loader, learning_rate=1e-4):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        self.model.to(self.device)
        self.optimizer = AdamW(
            model.parameters(), 
            lr=learning_rate, 
            weight_decay=0.01,
            betas=(0.9, 0.98),
            eps=1e-9
        )
        
        self.num_training_steps = len(train_loader) * 10
        self.num_warmup_steps = len(train_loader)
        self.scheduler = get_cosine_schedule_with_warmup(
            self.optimizer, 
            self.num_warmup_steps, 
            self.num_training_steps
        )
        
        self.criterion = nn.CrossEntropyLoss(ignore_index=0)
        self.max_grad_norm = 1.0
        
        self.train_losses = []
        self.val_losses = []
        self.train_ppls = []
        self.val_ppls = []
        self.learning_rates = []
        
        print(f"Using device: {self.device}")
        print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    def train_epoch(self, epoch):
        self.model.train()
        total_loss = 0
        total_tokens = 0
        
        for batch_idx, (data, target) in enumerate(self.train_loader):
            data, target = data.to(self.device), target.to(self.device)
            
            max_allowed_idx = self.model.vocab_size - 1
            if data.max() > max_allowed_idx or target.max() > max_allowed_idx:
                continue
            
            batch_size, seq_len = data.shape
            causal_mask = create_causal_mask(seq_len, self.device)
            padding_mask = create_padding_mask(data)
            mask = causal_mask & padding_mask
            
            self.optimizer.zero_grad()
            
            try:
                output = self.model(data, mask)
                loss = self.criterion(output.reshape(-1, self.model.vocab_size), 
                                    target.reshape(-1))
                
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
                self.optimizer.step()
                self.scheduler.step()
                
                total_loss += loss.item() * target.numel()
                total_tokens += target.numel()
                
                current_lr = self.scheduler.get_last_lr()[0]
                self.learning_rates.append(current_lr)
                
                if batch_idx % 50 == 0:
                    current_loss = loss.item()
                    perplexity = math.exp(current_loss) if current_loss < 10 else float('inf')
                    print(f'Epoch {epoch} | Batch {batch_idx:4d} | Loss: {current_loss:.4f} | '
                          f'Perplexity: {perplexity:7.2f} | LR: {current_lr:.2e}')
                        
            except Exception as e:
                continue
        
        return total_loss / total_tokens if total_tokens > 0 else float('inf')
    
    def evaluate(self, loader, dataset_name="Validation Set"):
        self.model.eval()
        total_loss = 0
        total_tokens = 0
        
        with torch.no_grad():
            for data, target in loader:
                data, target = data.to(self.device), target.to(self.device)
                
                max_allowed_idx = self.model.vocab_size - 1
                if data.max() > max_allowed_idx or target.max() > max_allowed_idx:
                    continue
                
                batch_size, seq_len = data.shape
                causal_mask = create_causal_mask(seq_len, self.device)
                padding_mask = create_padding_mask(data)
                mask = causal_mask & padding_mask
                
                output = self.model(data, mask)
                loss = self.criterion(output.reshape(-1, self.model.vocab_size), 
                                    target.reshape(-1))
                
                total_loss += loss.item() * target.numel()
                total_tokens += target.numel()
        
        avg_loss = total_loss / total_tokens if total_tokens > 0 else float('inf')
        perplexity = math.exp(avg_loss) if avg_loss != float('inf') and avg_loss < 10 else float('inf')
        
        print(f"{dataset_name} Evaluation:")
        print(f"  Loss: {avg_loss:.4f}, Perplexity: {perplexity:.2f}")
        print('-' * 40)
        
        return avg_loss, perplexity
    
    def train(self, epochs=10):
        print("Starting training...")
        
        for epoch in range(epochs):
            print(f'\nEpoch {epoch+1}/{epochs}')
            print('-' * 50)
            
            train_loss = self.train_epoch(epoch + 1)
            val_loss, val_ppl = self.evaluate(self.val_loader)
            
            self.train_losses.append(train_loss)
            self.val_losses.append(val_loss)
            if train_loss != float('inf'):
                self.train_ppls.append(math.exp(train_loss) if train_loss < 10 else float('inf'))
            else:
                self.train_ppls.append(float('inf'))
            self.val_ppls.append(val_ppl)
            
            print(f'Epoch {epoch+1} Summary:')
            print(f'  Training Loss: {train_loss:.4f}')
            print(f'  Validation Loss: {val_loss:.4f}')
            if train_loss != float('inf'):
                train_ppl = math.exp(train_loss) if train_loss < 10 else float('inf')
                print(f'  Training Perplexity: {train_ppl:.2f}')
            print(f'  Validation Perplexity: {val_ppl:.2f}')
        
        print("\nFinal evaluation on test set...")
        test_loss, test_ppl = self.evaluate(self.test_loader, "Test Set")
        
        return self.train_losses, self.val_losses, test_loss, test_ppl