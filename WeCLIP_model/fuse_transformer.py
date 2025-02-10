import torch.nn as nn
import torch
import torch.nn.functional as F


class TextFusionTransformer_ver1(nn.Module):
    def __init__(self, embed_dim, heads, layers=2, dropout=0.1):
        super(TextFusionTransformer_ver1, self).__init__()
        self.multihead_attn = nn.MultiheadAttention(embed_dim, heads, dropout=dropout)
        self.layer_norm = nn.LayerNorm(embed_dim)
        self.dropout = nn.Dropout(dropout)
        self.layers = layers
        self.c = nn.Parameter(torch.tensor(1.0))
    
    def forward(self, query, key, value):
        for _ in range(self.layers):
            attn_output, _ = self.multihead_attn(query, key, value)
            attn_output = self.dropout(attn_output)
            query = self.layer_norm(query + self.c * attn_output)
        return query


class TextFusionTransformer_ver2(nn.Module):
    def __init__(self, embed_dim, heads, layers=2, dropout=0.0):
        super(TextFusionTransformer_ver2, self).__init__()
        self.multihead_attn = nn.MultiheadAttention(embed_dim, heads, dropout=dropout)
        self.layer_norm1 = nn.LayerNorm(embed_dim)
        self.layer_norm2 = nn.LayerNorm(embed_dim)
        self.layers = layers
        self.mlp = nn.Sequential([
            nn.Linear(embed_dim, embed_dim * 4),
            nn.GELU(),
            nn.Linear(embed_dim * 4, embed_dim)
        ])
    
    def forward(self, query, key, value):
        for _ in range(self.layers):
            query = self.layer_norm1(query)
            key = self.layer_norm1(key)
            value = self.layer_norm1(value)
            attn_output, _ = self.multihead_attn(query, key, value)
            query = query + attn_output
            query = query + self.mlp(self.layer_norm2(query))
        return query
    
class TextFusionTransformer_ver3(nn.Module):
    def __init__(self, embed_dim, heads, layers=2, dropout=0.0):
        super(TextFusionTransformer_ver3, self).__init__()
        self.multihead_attn = nn.MultiheadAttention(embed_dim, heads, dropout=dropout)
        self.layer_norm1 = nn.LayerNorm(embed_dim)
        self.layer_norm2 = nn.LayerNorm(embed_dim)
        self.layers = layers
        self.mlp = nn.Sequential([
            nn.Linear(embed_dim, embed_dim * 4),
            nn.GELU(),
            nn.Linear(embed_dim * 4, embed_dim)
        ])
    
    def forward(self, query, key, value):
        for _ in range(self.layers):
            query = self.layer_norm1(query)
            key = self.layer_norm1(key)
            value = self.layer_norm1(value)
            attn_output, _ = self.multihead_attn(query, key, value)
            query = query + self.mlp(self.layer_norm2(attn_output))
        return query