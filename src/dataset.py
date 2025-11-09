import os
import torch
from torch.utils.data import Dataset, DataLoader
from collections import Counter
import requests

class WikiText2Dataset(Dataset):
    def __init__(self, file_path, seq_len=128, vocab_size=10000, mode='train'):
        self.seq_len = seq_len
        self.mode = mode
        self.vocab_size = vocab_size
        
        with open(file_path, 'r', encoding='utf-8') as f:
            text = f.read()
        
        self._build_vocab(text)
        
        tokens = text.split()
        self.data = []
        for token in tokens:
            if token.strip():
                idx = self.word2idx.get(token, self.word2idx['<unk>'])
                self.data.append(idx)
        
        print(f"{mode} dataset loaded, total tokens: {len(self.data)}")
        print(f"Vocabulary size: {len(self.word2idx)}")
    
    def _build_vocab(self, text):
        tokens = text.split()
        tokens = [token for token in tokens if token.strip()]
        word_freq = Counter(tokens)
        
        most_common = word_freq.most_common(self.vocab_size - 3)
        
        self.word2idx = {
            '<pad>': 0,
            '<unk>': 1,
            '<eos>': 2
        }
        self.idx2word = {v: k for k, v in self.word2idx.items()}
        
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
        
        actual_vocab_size = len(self.word2idx)
        max_allowed_idx = actual_vocab_size - 1
        
        if input_seq.max() > max_allowed_idx:
            input_seq = torch.clamp(input_seq, 0, max_allowed_idx)
        if target_seq.max() > max_allowed_idx:
            target_seq = torch.clamp(target_seq, 0, max_allowed_idx)
        
        return input_seq, target_seq

def download_wikitext2(data_dir='./data'):
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