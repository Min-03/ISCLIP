import torch.nn as nn
import torch
import torch.nn.functional as F


class TextFusionTransformer_ver1(nn.Module):
    def __init__(self, embed_dim, heads, layers=2, dropout=0.1):
        super(TextFusionTransformer_ver1, self).__init__()
        self.layers = layers
        self.attention_layers = nn.ModuleList(nn.MultiheadAttention(embed_dim, heads, dropout=dropout) for _ in range(layers))
        self.layer_norm = nn.ModuleList(nn.LayerNorm(embed_dim) for _ in range(layers))
        self.dropout = nn.Dropout(dropout)
        self.c = nn.Parameter(torch.ones(self.layers))
    
    def forward(self, query, key, value):
        for i in range(self.layers):
            attn_output, _ = self.attention_layers[i](query, key, value)
            attn_output = self.dropout(attn_output)
            query = self.layer_norm[i](query + self.c[i] * attn_output)
        return query


class TextFusionTransformer_ver2(nn.Module):
    def __init__(self, embed_dim, heads, layers=2, dropout=0.0):
        super(TextFusionTransformer_ver2, self).__init__()
        self.multihead_attn = nn.MultiheadAttention(embed_dim, heads, dropout=dropout)
        self.layer_norm1 = nn.LayerNorm(embed_dim)
        self.layer_norm2 = nn.LayerNorm(embed_dim)
        self.layers = layers
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 4),
            nn.GELU(),
            nn.Linear(embed_dim * 4, embed_dim)
        )
    
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
        self.layers = layers
        self.attention_layers = nn.ModuleList([
            nn.MultiheadAttention(embed_dim, heads, dropout=dropout) for _ in range(layers)
        ])
        self.layer_norm1 = nn.ModuleList([nn.LayerNorm(embed_dim) for _ in range(layers)])
        self.layer_norm2 = nn.ModuleList([nn.LayerNorm(embed_dim) for _ in range(layers)])
        self.mlp_layers = nn.ModuleList([
            nn.Linear(embed_dim, embed_dim) for _ in range(layers)
        ])
    
    def forward(self, query, key, value):
        for i in range(self.layers):
            query = self.layer_norm1[i](query)
            key = self.layer_norm1[i](key)
            value = self.layer_norm1[i](value)
            attn_output, _ = self.attention_layers[i](query, key, value)
            query = query + self.mlp_layers[i](self.layer_norm2[i](attn_output))
        return query
    
class TextFusionTransformer_ver4(nn.Module):
    def __init__(self, embed_dim, heads, layers=2, dropout=0.0):
        super(TextFusionTransformer_ver4, self).__init__()
        self.layers = layers
        self.attention_layers = nn.ModuleList([
            nn.MultiheadAttention(embed_dim, heads, dropout=dropout) for _ in range(layers)
        ])
        self.layer_norm1 = nn.ModuleList([nn.LayerNorm(embed_dim) for _ in range(layers)])
        self.layer_norm2 = nn.ModuleList([nn.LayerNorm(embed_dim) for _ in range(layers)])
        self.mlp_layers = nn.ModuleList([
            nn.Sequential(
                nn.Linear(embed_dim, embed_dim),
                nn.GELU(),
                nn.Linear(embed_dim, embed_dim)
            ) for _ in range(layers)
        ])
    
    def forward(self, query, key, value):
        for i in range(self.layers):
            query = self.layer_norm1[i](query)
            key = self.layer_norm1[i](key)
            value = self.layer_norm1[i](value)
            attn_output, _ = self.attention_layers[i](query, key, value)
            query = query + self.mlp_layers[i](self.layer_norm2[i](attn_output))
        return query