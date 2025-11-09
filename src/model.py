import torch
import torch.nn as nn
import torch.nn.functional as F
import math

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