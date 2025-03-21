import torch.nn as nn
import torch
import torch.nn.functional as F
    
class TextFusionTransformer(nn.Module):
    def __init__(self, embed_dim, heads, layers=2, dropout=0.0, **kwargs):
        super().__init__()
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
    
    def forward(self, query, key, value, **kwargs):
        for i in range(self.layers):
            query = self.layer_norm1[i](query)
            key = self.layer_norm1[i](key)
            value = self.layer_norm1[i](value)
            attn_output, _ = self.attention_layers[i](query, key, value)
            query = query + self.mlp_layers[i](self.layer_norm2[i](attn_output))
        return query

    
class TextFusionTransformer_ver2(nn.Module):
    def __init__(self, embed_dim, heads, layers=2, dropout=0.0, class_num=20, **kwargs):
        super().__init__()
        self.layers = layers
        self.attention_layers = nn.ModuleList([
            nn.MultiheadAttention(embed_dim, heads, dropout=dropout) for _ in range(layers)
        ])
        self.layer_norm1 = nn.ModuleList([nn.LayerNorm(embed_dim) for _ in range(layers)])
        self.layer_norm2 = nn.ModuleList([nn.LayerNorm(embed_dim) for _ in range(layers)])
        self.c = nn.ParameterList([nn.Parameter(torch.ones(class_num)) for _ in range(layers)])
        self.mlp_layers = nn.ModuleList([
            nn.Sequential(
                nn.Linear(embed_dim, embed_dim),
                nn.GELU(),
                nn.Linear(embed_dim, embed_dim)
            ) for _ in range(layers)
        ])
    
    def forward(self, query, key, value, class_idx=0, **kwargs):
        for i in range(self.layers):
            query = self.layer_norm1[i](query)
            key = self.layer_norm1[i](key)
            value = self.layer_norm1[i](value)
            attn_output, _ = self.attention_layers[i](query, key, value)
            query = query + self.c[i][class_idx] * self.mlp_layers[i](self.layer_norm2[i](attn_output))
        return query
    
class TextFusionTransformer_ver3(nn.Module):
    def __init__(self, embed_dim, heads, layers=2, dropout=0.0, class_num=20, **kwargs):
        super().__init__()
        self.layers = layers
        self.attention_layers = nn.ModuleList([
            nn.MultiheadAttention(embed_dim, heads, dropout=dropout) for _ in range(layers)
        ])
        self.layer_norm1 = nn.ModuleList([nn.LayerNorm(embed_dim) for _ in range(layers)])
        self.layer_norm2 = nn.ModuleList([nn.LayerNorm(embed_dim) for _ in range(layers)])
        self.c = nn.ParameterList([nn.Parameter(torch.ones(class_num)) for _ in range(layers)])
    
    def forward(self, query, key, value, class_idx=0, **kwargs):
        for i in range(self.layers):
            query = self.layer_norm1[i](query)
            key = self.layer_norm1[i](key)
            value = self.layer_norm1[i](value)
            attn_output, _ = self.attention_layers[i](query, key, value)
            query = query + self.c[i][class_idx] * self.layer_norm2[i](attn_output)
        return query
