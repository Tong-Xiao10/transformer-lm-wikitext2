import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import os
from torch.utils.data import Dataset, DataLoader
from collections import Counter
import requests
import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime
import json
from torch.optim.lr_scheduler import LambdaLR

class WikiText2Dataset(Dataset):
    """WikiText-2数据集加载和预处理"""
    def __init__(self, file_path, seq_len=128, vocab_size=10000, mode='train'):
        self.seq_len = seq_len
        self.mode = mode
        self.vocab_size = vocab_size
        
        # 读取数据
        with open(file_path, 'r', encoding='utf-8') as f:
            text = f.read()
        
        # 构建词汇表
        self._build_vocab(text)
        
        # 分词和编码
        tokens = text.split()
        self.data = []
        for token in tokens:
            if token.strip():  # 跳过空token
                idx = self.word2idx.get(token, self.word2idx['<unk>'])
                self.data.append(idx)
        
        print(f"{mode} dataset loaded, total tokens: {len(self.data)}")
        print(f"Vocabulary size: {len(self.word2idx)}")
        print(f"Data index range: {min(self.data)} - {max(self.data)}")
    
    def _build_vocab(self, text):
        """构建词汇表"""
        tokens = text.split()
        tokens = [token for token in tokens if token.strip()]
        word_freq = Counter(tokens)
        
        # 选择最常见的词
        most_common = word_freq.most_common(self.vocab_size - 3)
        
        self.word2idx = {
            '<pad>': 0,
            '<unk>': 1,
            '<eos>': 2
        }
        self.idx2word = {v: k for k, v in self.word2idx.items()}
        
        # 添加常用词
        for i, (word, _) in enumerate(most_common):
            idx = i + 3
            if idx < self.vocab_size:
                self.word2idx[word] = idx
                self.idx2word[idx] = word
    
    def __len__(self):
        if len(self.data) <= self.seq_len + 1:
            return 0
        return (len(self.data) - 1) // self.seq_len
    
    def __getitem__(self, idx):
        start_idx = idx * self.seq_len
        end_idx = start_idx + self.seq_len + 1
        
        if end_idx > len(self.data):
            start_idx = max(0, len(self.data) - self.seq_len - 1)
            end_idx = len(self.data)
        
        sequence = self.data[start_idx:end_idx]
        
        if len(sequence) < self.seq_len + 1:
            padding_needed = self.seq_len + 1 - len(sequence)
            sequence.extend([self.word2idx['<pad>']] * padding_needed)
        
        input_seq = torch.tensor(sequence[:self.seq_len], dtype=torch.long)
        target_seq = torch.tensor(sequence[1:self.seq_len+1], dtype=torch.long)
        
        # 验证索引范围
        actual_vocab_size = len(self.word2idx)
        max_allowed_idx = actual_vocab_size - 1
        
        if input_seq.max() > max_allowed_idx:
            input_seq = torch.clamp(input_seq, 0, max_allowed_idx)
        if target_seq.max() > max_allowed_idx:
            target_seq = torch.clamp(target_seq, 0, max_allowed_idx)
        
        return input_seq, target_seq

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=2048, dropout=0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * 
                           (-math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        
        self.register_buffer('pe', pe)
        
    def forward(self, x):
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads, dropout=0.1):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        assert d_model % num_heads == 0
        self.d_k = d_model // num_heads
        
        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)
        
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, Q, K, V, mask=None):
        batch_size, seq_len = Q.size(0), Q.size(1)
        
        Q = self.W_q(Q).reshape(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)
        K = self.W_k(K).reshape(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)
        V = self.W_v(V).reshape(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)
        
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)
        
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
            
        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)
        attn_output = torch.matmul(attn_weights, V)
        
        attn_output = attn_output.transpose(1, 2).contiguous().reshape(
            batch_size, seq_len, self.d_model)
        output = self.W_o(attn_output)
        
        return output

class PositionwiseFFN(nn.Module):
    def __init__(self, d_model, d_ff, dropout=0.1):
        super().__init__()
        self.linear1 = nn.Linear(d_model, d_ff)
        self.linear2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        return self.linear2(self.dropout(F.gelu(self.linear1(x))))

class TransformerLayer(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, dropout=0.1):
        super().__init__()
        self.self_attn = MultiHeadAttention(d_model, num_heads, dropout)
        self.ffn = PositionwiseFFN(d_model, d_ff, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x, mask=None):
        residual = x
        x_normalized = self.norm1(x)
        attn_output = self.self_attn(x_normalized, x_normalized, x_normalized, mask)
        x = residual + self.dropout(attn_output)
        
        residual = x
        x_normalized = self.norm2(x)
        ffn_output = self.ffn(x_normalized)
        x = residual + self.dropout(ffn_output)
        
        return x

class WikiText2Transformer(nn.Module):
    def __init__(self, vocab_size, d_model=512, num_heads=8, d_ff=2048, 
                 num_layers=6, dropout=0.1, max_len=1024):
        super().__init__()
        self.vocab_size = vocab_size
        self.d_model = d_model
        
        self.token_embedding = nn.Embedding(vocab_size, d_model, padding_idx=0)
        self.pos_encoding = PositionalEncoding(d_model, max_len, dropout)
        
        self.layers = nn.ModuleList([
            TransformerLayer(d_model, num_heads, d_ff, dropout) 
            for _ in range(num_layers)
        ])
        
        self.output_norm = nn.LayerNorm(d_model)
        self.output_layer = nn.Linear(d_model, vocab_size)
        
        self._init_weights()
        
    def _init_weights(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
        nn.init.normal_(self.output_layer.weight, mean=0, std=0.02)
        
    def forward(self, x, mask=None):
        max_allowed_idx = self.vocab_size - 1
        if x.max() > max_allowed_idx:
            x = torch.clamp(x, 0, max_allowed_idx)
        
        x_embedded = self.token_embedding(x) * math.sqrt(self.d_model)
        x_embedded = self.pos_encoding(x_embedded)
        
        x = x_embedded
        for layer in self.layers:
            x = layer(x, mask)
        
        x = self.output_norm(x)
        output = self.output_layer(x)
        
        return output

def create_causal_mask(seq_len, device):
    mask = torch.triu(torch.ones(seq_len, seq_len, device=device), diagonal=1).bool()
    return mask.unsqueeze(0).unsqueeze(1)

def create_padding_mask(seq, pad_idx=0):
    mask = (seq != pad_idx).unsqueeze(1).unsqueeze(2)
    return mask

def get_cosine_schedule_with_warmup(optimizer, num_warmup_steps, num_training_steps, num_cycles=0.5):
    """创建带warmup的cosine学习率调度器"""
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
        
        # 使用AdamW优化器
        self.optimizer = torch.optim.AdamW(
            model.parameters(), 
            lr=learning_rate, 
            weight_decay=0.01,
            betas=(0.9, 0.98),
            eps=1e-9
        )
        
        # 学习率调度器
        self.num_training_steps = len(train_loader) * 10  # 假设10个epoch
        self.num_warmup_steps = len(train_loader)  # 1个epoch的warmup
        self.scheduler = get_cosine_schedule_with_warmup(
            self.optimizer, 
            self.num_warmup_steps, 
            self.num_training_steps
        )
        
        self.criterion = nn.CrossEntropyLoss(ignore_index=0)
        
        # 存储训练历史
        self.train_losses = []
        self.val_losses = []
        self.train_ppls = []
        self.val_ppls = []
        self.learning_rates = []
        
        # 梯度裁剪参数
        self.max_grad_norm = 1.0
        
        print(f"Using device: {self.device}")
        print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
        print(f"Vocabulary size: {model.vocab_size}")
        print(f"Training steps: {self.num_training_steps}")
        print(f"Warmup steps: {self.num_warmup_steps}")
        
        # 参数统计
        self._print_parameter_stats()
    
    def _print_parameter_stats(self):
        """打印模型参数统计"""
        total_params = 0
        trainable_params = 0
        
        print("\n" + "="*50)
        print("Model Parameter Statistics")
        print("="*50)
        
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                trainable_params += param.numel()
            total_params += param.numel()
            print(f"{name:50} | {str(param.shape):20} | {param.numel():10,} | {'Trainable' if param.requires_grad else 'Frozen'}")
        
        print("="*50)
        print(f"Total parameters: {total_params:,}")
        print(f"Trainable parameters: {trainable_params:,}")
        print(f"Non-trainable parameters: {total_params - trainable_params:,}")
        print("="*50 + "\n")
    
    def train_epoch(self, epoch):
        self.model.train()
        total_loss = 0
        total_tokens = 0
        batch_losses = []
        
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
                
                # 梯度裁剪
                grad_norm = torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), 
                    self.max_grad_norm
                )
                
                self.optimizer.step()
                self.scheduler.step()  # 更新学习率
                
                total_loss += loss.item() * target.numel()
                total_tokens += target.numel()
                batch_losses.append(loss.item())
                
                current_lr = self.scheduler.get_last_lr()[0]
                self.learning_rates.append(current_lr)
                
                if batch_idx % 50 == 0:
                    current_loss = loss.item()
                    perplexity = math.exp(current_loss) if current_loss < 10 else float('inf')
                    print(f'Epoch {epoch} | Batch {batch_idx:4d} | Loss: {current_loss:.4f} | '
                          f'Perplexity: {perplexity:7.2f} | Grad Norm: {grad_norm:.4f} | '
                          f'LR: {current_lr:.2e}')
                        
            except Exception as e:
                print(f"Error in batch {batch_idx}: {e}")
                continue
        
        avg_loss = total_loss / total_tokens if total_tokens > 0 else float('inf')
        return avg_loss, batch_losses
    
    def evaluate(self, loader, dataset_name="Validation Set"):
        """Evaluate on specified dataset"""
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
        
        print(f"{dataset_name} Evaluation Results:")
        print(f"  Loss: {avg_loss:.4f}")
        print(f"  Perplexity: {perplexity:.2f}")
        print('-' * 40)
        
        return avg_loss, perplexity
    
    def save_checkpoint(self, epoch, loss, save_dir='./checkpoints'):
        """保存模型检查点"""
        os.makedirs(save_dir, exist_ok=True)
        
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'loss': loss,
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
            'train_ppls': self.train_ppls,
            'val_ppls': self.val_ppls,
            'learning_rates': self.learning_rates,
            'vocab_size': self.model.vocab_size,
            'd_model': self.model.d_model,
        }
        
        checkpoint_path = os.path.join(save_dir, f'checkpoint_epoch_{epoch}.pth')
        torch.save(checkpoint, checkpoint_path)
        print(f"Checkpoint saved: {checkpoint_path}")
        
        # 同时保存最佳模型
        if loss == min(self.val_losses):
            best_model_path = os.path.join(save_dir, 'best_model.pth')
            torch.save(checkpoint, best_model_path)
            print(f"Best model saved: {best_model_path}")
    
    def load_checkpoint(self, checkpoint_path):
        """加载模型检查点"""
        if not os.path.exists(checkpoint_path):
            print(f"Checkpoint not found: {checkpoint_path}")
            return
        
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        
        self.train_losses = checkpoint.get('train_losses', [])
        self.val_losses = checkpoint.get('val_losses', [])
        self.train_ppls = checkpoint.get('train_ppls', [])
        self.val_ppls = checkpoint.get('val_ppls', [])
        self.learning_rates = checkpoint.get('learning_rates', [])
        
        print(f"Checkpoint loaded: {checkpoint_path}")
        print(f"Resuming from epoch {checkpoint['epoch']}")
        
        return checkpoint['epoch']
    
    def train(self, epochs=10, resume_checkpoint=None):
        print("Starting training...")
        
        start_epoch = 0
        if resume_checkpoint:
            start_epoch = self.load_checkpoint(resume_checkpoint)
        
        for epoch in range(start_epoch, epochs):
            print(f'\n{"="*60}')
            print(f'Starting epoch {epoch+1}/{epochs}')
            print(f'{"="*60}')
            
            train_loss, batch_losses = self.train_epoch(epoch + 1)
            val_loss, val_ppl = self.evaluate(self.val_loader, "Validation Set")
            
            # 存储历史数据
            self.train_losses.append(train_loss)
            self.val_losses.append(val_loss)
            if train_loss != float('inf'):
                self.train_ppls.append(math.exp(train_loss) if train_loss < 10 else float('inf'))
            else:
                self.train_ppls.append(float('inf'))
            self.val_ppls.append(val_ppl)
            
            print(f'\nEpoch {epoch+1}/{epochs} Summary:')
            print(f'  Training Loss: {train_loss:.4f}')
            print(f'  Validation Loss: {val_loss:.4f}')
            if train_loss != float('inf'):
                train_ppl = math.exp(train_loss) if train_loss < 10 else float('inf')
                print(f'  Training Perplexity: {train_ppl:.2f}')
            print(f'  Validation Perplexity: {val_ppl:.2f}')
            
            # 保存检查点
            self.save_checkpoint(epoch + 1, val_loss)
            
            print(f'{"="*60}')
        
        # Final evaluation on test set
        print("\nFinal evaluation on test set...")
        test_loss, test_ppl = self.evaluate(self.test_loader, "Test Set")
        
        return self.train_losses, self.val_losses, test_loss, test_ppl

    def plot_results(self, save_path='./results'):
        """Plot comprehensive training results charts"""
        os.makedirs(save_path, exist_ok=True)
        
        # 创建综合图表
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
        
        # 1. 损失曲线
        epochs = range(1, len(self.train_losses) + 1)
        ax1.plot(epochs, self.train_losses, 'b-', label='Training Loss', linewidth=2, marker='o')
        ax1.plot(epochs, self.val_losses, 'r-', label='Validation Loss', linewidth=2, marker='s')
        ax1.set_title('Training and Validation Loss', fontsize=14, fontweight='bold')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # 2. 困惑度曲线
        ax2.plot(epochs, self.train_ppls, 'b-', label='Training Perplexity', linewidth=2, marker='o')
        ax2.plot(epochs, self.val_ppls, 'r-', label='Validation Perplexity', linewidth=2, marker='s')
        ax2.set_title('Training and Validation Perplexity', fontsize=14, fontweight='bold')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Perplexity')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # 3. 学习率变化
        steps = range(len(self.learning_rates))
        ax3.plot(steps, self.learning_rates, 'g-', linewidth=2)
        ax3.set_title('Learning Rate Schedule', fontsize=14, fontweight='bold')
        ax3.set_xlabel('Training Step')
        ax3.set_ylabel('Learning Rate')
        ax3.grid(True, alpha=0.3)
        ax3.set_yscale('log')
        
        # 4. 最终结果对比
        categories = ['Training', 'Validation']
        final_losses = [self.train_losses[-1], self.val_losses[-1]]
        final_ppls = [self.train_ppls[-1], self.val_ppls[-1]]
        
        x = np.arange(len(categories))
        width = 0.35
        
        bars1 = ax4.bar(x - width/2, final_losses, width, label='Loss', alpha=0.7)
        bars2 = ax4.bar(x + width/2, final_ppls, width, label='Perplexity', alpha=0.7)
        
        ax4.set_title('Final Results Comparison', fontsize=14, fontweight='bold')
        ax4.set_xlabel('Dataset')
        ax4.set_ylabel('Values')
        ax4.set_xticks(x)
        ax4.set_xticklabels(categories)
        ax4.legend()
        ax4.grid(True, alpha=0.3)
        
        # 添加数值标签
        for bar, value in zip(bars1, final_losses):
            ax4.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1, 
                    f'{value:.2f}', ha='center', va='bottom', fontweight='bold')
        
        for bar, value in zip(bars2, final_ppls):
            ax4.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 5, 
                    f'{value:.1f}', ha='center', va='bottom', fontweight='bold')
        
        plt.tight_layout()
        
        # 保存图表
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        chart_path = os.path.join(save_path, f'training_results_{timestamp}.png')
        plt.savefig(chart_path, dpi=300, bbox_inches='tight')
        plt.show()
        
        print(f"Comprehensive charts saved to: {chart_path}")
        
        # 保存详细的训练结果
        self._save_training_report(save_path, timestamp)
        
        return chart_path

    def _save_training_report(self, save_path, timestamp):
        """保存详细的训练报告"""
        report_path = os.path.join(save_path, f'training_report_{timestamp}.json')
        
        report = {
            "training_info": {
                "total_epochs": len(self.train_losses),
                "final_train_loss": self.train_losses[-1] if self.train_losses else None,
                "final_val_loss": self.val_losses[-1] if self.val_losses else None,
                "final_train_perplexity": self.train_ppls[-1] if self.train_ppls else None,
                "final_val_perplexity": self.val_ppls[-1] if self.val_ppls else None,
                "model_parameters": sum(p.numel() for p in self.model.parameters()),
                "vocabulary_size": self.model.vocab_size,
                "training_time": datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            },
            "hyperparameters": {
                "learning_rate": self.optimizer.param_groups[0]['lr'],
                "weight_decay": self.optimizer.param_groups[0]['weight_decay'],
                "max_grad_norm": self.max_grad_norm,
                "warmup_steps": self.num_warmup_steps,
                "total_training_steps": self.num_training_steps
            },
            "epoch_results": {
                "train_losses": self.train_losses,
                "val_losses": self.val_losses,
                "train_perplexities": self.train_ppls,
                "val_perplexities": self.val_ppls
            }
        }
        
        with open(report_path, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
        
        print(f"Training report saved to: {report_path}")

def download_wikitext2(data_dir='./data'):
    """Download WikiText-2 dataset"""
    os.makedirs(data_dir, exist_ok=True)
    
    urls = {
        'train': 'https://raw.githubusercontent.com/pytorch/examples/master/word_language_model/data/wikitext-2/train.txt',
        'valid': 'https://raw.githubusercontent.com/pytorch/examples/master/word_language_model/data/wikitext-2/valid.txt',
        'test': 'https://raw.githubusercontent.com/pytorch/examples/master/word_language_model/data/wikitext-2/test.txt'
    }
    
    for split, url in urls.items():
        file_path = os.path.join(data_dir, f'{split}.txt')
        if not os.path.exists(file_path):
            print(f"Downloading {split} dataset...")
            try:
                response = requests.get(url, timeout=30)
                response.raise_for_status()
                with open(file_path, 'w', encoding='utf-8') as f:
                    f.write(response.text)
                print(f"{split} dataset downloaded successfully")
            except Exception as e:
                print(f"Failed to download {split} dataset: {e}")

def main():
    # 创建结果目录
    os.makedirs('./results', exist_ok=True)
    os.makedirs('./checkpoints', exist_ok=True)
    
    # 数据路径
    data_dir = './data'
    
    # 下载数据
    download_wikitext2(data_dir)
    
    try:
        # 创建所有数据集 - 使用小数据集进行验证
        vocab_size = 3000  # 更小的词汇表用于快速验证
        seq_len = 64      # 更短的序列长度
        batch_size = 8    # 更小的batch size
        
        print("Creating datasets...")
        train_dataset = WikiText2Dataset(
            os.path.join(data_dir, 'train.txt'), 
            seq_len=seq_len, vocab_size=vocab_size, mode='train'
        )
        val_dataset = WikiText2Dataset(
            os.path.join(data_dir, 'valid.txt'), 
            seq_len=seq_len, vocab_size=vocab_size, mode='valid'
        )
        test_dataset = WikiText2Dataset(
            os.path.join(data_dir, 'test.txt'), 
            seq_len=seq_len, vocab_size=vocab_size, mode='test'
        )
        
        actual_vocab_size = len(train_dataset.word2idx)
        print(f"Using actual vocabulary size: {actual_vocab_size}")
        
        if len(train_dataset) == 0:
            print("Training dataset is empty!")
            return
        
        # 创建数据加载器
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
        
        # 创建模型 - 使用小模型进行快速验证
        print("Initializing model...")
        model = WikiText2Transformer(
            vocab_size=actual_vocab_size,
            d_model=128,    # 更小的模型维度
            num_heads=4,
            d_ff=512,       # 更小的前馈网络维度
            num_layers=3,   # 更少的层数
            dropout=0.1
        )
        
        # 创建训练器并开始训练
        print("Starting training...")
        trainer = WikiText2Trainer(model, train_loader, val_loader, test_loader, learning_rate=5e-4)
        
        # 训练模型
        train_losses, val_losses, test_loss, test_ppl = trainer.train(epochs=3)
        
        # 绘制结果图表
        print("Generating results charts...")
        chart_path = trainer.plot_results()
        
        # 保存最终模型
        final_model_path = './checkpoints/final_model.pth'
        torch.save({
            'model_state_dict': model.state_dict(),
            'vocab_size': actual_vocab_size,
            'config': {
                'd_model': 128,
                'num_heads': 4,
                'd_ff': 512,
                'num_layers': 3,
                'dropout': 0.1
            }
        }, final_model_path)
        
        print(f"Final model saved to: {final_model_path}")
        
        print("\n" + "="*70)
        print("TRAINING COMPLETED SUCCESSFULLY!")
        print("="*70)
        print(f"Final Test Results:")
        print(f"  Loss: {test_loss:.4f}")
        print(f"  Perplexity: {test_ppl:.2f}")
        print(f"  Charts saved: {chart_path}")
        print(f"  Model saved: {final_model_path}")
        print("="*70)
        
    except Exception as e:
        print(f"Error during execution: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()