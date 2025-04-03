import math

import torch
import torch.nn as nn
from torch.nn import functional as F
import torch.optim as optim

# Single headed attention mechanism 
class Attention(nn.Module):
    def __init__(self, d_embed, d_head, dropout):
        super().__init__()
        self.Query = nn.Linear(d_embed, d_head, bias=False)
        self.Key = nn.Linear(d_embed, d_head, bias=False)
        self.Value = nn.Linear(d_embed, d_head, bias=False)
        self.d_k = d_embed

        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        query = self.Query(x)
        key = self.Key(x)
        numerator = query @ key.transpose(-2, -1) * key.shape[-1]**-0.5
        y = F.softmax((numerator / math.sqrt(self.d_k)))
        value = self.Value(x)
        y = y @ value
        return y


class MultiHeadAttention(nn.Module):
    def __init__(self, n_heads, d_head, d_embed, dropout):
        super().__init__()
        self.heads = nn.ModuleList([Attention(d_embed, d_head, dropout) for _ in range(n_heads)])
        self.proj = nn.Linear(n_heads * d_head, d_embed)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        y = torch.cat([h(x) for h in self.heads], dim=-1)
        y = self.proj(y)
        y = self.dropout(y)
        return y

# Fully connected feed forward network layer 
class MultiLayerPerceptron(nn.Module):
    def __init__(self, d_embed, dropout):
        super().__init__()
        self.fc1 = nn.Linear(d_embed, 4 * d_embed)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(4 * d_embed, d_embed)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.dropout(x)
        return x

# Complete transformer block (attention + FFN)
class Transformer(nn.Module):
    def __init__(self, d_embed, n_heads, dropout=0.1):
        super().__init__()
        d_head = d_embed // n_heads 
        self.attention = MultiHeadAttention(n_heads, d_head, d_embed, dropout)
        self.mlp = MultiLayerPerceptron(d_embed, dropout)
        self.norm1 = nn.LayerNorm(d_embed)
        self.norm2 = nn.LayerNorm(d_embed)

    def forward(self, x):
        x = x + self.attention(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x

# Transformer language model which predicts next token
class TransPredict(nn.Module):
    def __init__(self, vocab_size, n_blocks, d_embed, n_heads, dropout, lr):
        super().__init__()
        
        self.blocks = nn.Sequential(*[Transformer(d_embed, n_heads, dropout) for _ in range(n_blocks)])

        self.token_embedding = nn.Embedding(vocab_size, d_embed)
        self.position_embedding = nn.Embedding(n_blocks, d_embed)
        self.head = nn.Linear(d_embed, vocab_size)

        self.criterion = nn.CrossEntropyLoss()
        self.optim = optim.Adam(self.parameters(), lr=lr)

    def forward(self, x):
        B, T = x.shape

        tok = self.token_embedding(x)
        pos = self.position_embedding((torch.arange(T)))
        x = tok + pos
        x = self.blocks(x)
        x = self.head(x)
        return x
        
