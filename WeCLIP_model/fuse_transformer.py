import torch.nn as nn
import torch
import torch.nn.functional as F

class TextFusionTransformer(nn.Module):
    """
    Basic fusion transformer with attention module
    """
    def __init__(self, embed_dim, heads, layers=2, dropout=0.0, **kwargs):
        super().__init__()
        self.layers = layers
        self.attention_layers = nn.ModuleList([
            nn.MultiheadAttention(embed_dim, heads, dropout=dropout) for _ in range(layers)
        ])
        self.layer_norm1 = nn.ModuleList([nn.LayerNorm(embed_dim) for _ in range(layers)])
        self.layer_norm2 = nn.ModuleList([nn.LayerNorm(embed_dim) for _ in range(layers)])
    
    def forward(self, query, key, value, **kwargs):
        for i in range(self.layers):
            query = self.layer_norm1[i](query)
            key = self.layer_norm1[i](key)
            value = self.layer_norm1[i](value)
            attn_output, _ = self.attention_layers[i](query, key, value)
            query = query + self.layer_norm2[i](attn_output)
        return query

class TextFusionTransformer_ver2(nn.Module):
    """
    Basic fusion transformer with attention and additional mlp layers
    """
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
    

    
class TextFusionSubModule(nn.Module):
    """
    Module that reinforce the attribute by subtracting class prompt 
    from class specific caption.
    """
    def __init__(self, embed_dim, **kwargs):
        self.mlp_layer = nn.Linear(embed_dim, embed_dim)
        self.c = nn.Parameter(torch.ones(1,))
        
    def forward(self, org, cap, *args, **kwargs):
        diff = cap - org
        diff = self.mlp_layer(diff)
        diff *= self.c
        return org + diff
    
