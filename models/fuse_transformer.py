import torch.nn as nn
import torch
import torch.nn.functional as F


class TextFusionTransformer_ver1(nn.Module):
    def __init__(self, embed_dim, heads, layers=2, dropout=0.1, **kwargs):
        super(TextFusionTransformer_ver1, self).__init__()
        self.layers = layers
        self.attention_layers = nn.ModuleList(nn.MultiheadAttention(embed_dim, heads, dropout=dropout) for _ in range(layers))
        self.layer_norm = nn.ModuleList(nn.LayerNorm(embed_dim) for _ in range(layers))
        self.dropout = nn.Dropout(dropout)
        self.c = nn.Parameter(torch.ones(self.layers))
    
    def forward(self, query, key, value, **kwargs):
        for i in range(self.layers):
            attn_output, _ = self.attention_layers[i](query, key, value)
            attn_output = self.dropout(attn_output)
            query = self.layer_norm[i](query + self.c[i] * attn_output)
        return query


class TextFusionTransformer_ver2(nn.Module):
    def __init__(self, embed_dim, heads, layers=2, dropout=0.0, **kwargs):
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
    
    def forward(self, query, key, value, **kwargs):
        for _ in range(self.layers):
            query = self.layer_norm1(query)
            key = self.layer_norm1(key)
            value = self.layer_norm1(value)
            attn_output, _ = self.multihead_attn(query, key, value)
            query = query + attn_output
            query = query + self.mlp(self.layer_norm2(query))
        return query
    

    
class TextFusionTransformer_ver3(nn.Module):
    def __init__(self, embed_dim, heads, layers=2, dropout=0.0, **kwargs):
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
    
    def forward(self, query, key, value, **kwargs):
        for i in range(self.layers):
            query = self.layer_norm1[i](query)
            key = self.layer_norm1[i](key)
            value = self.layer_norm1[i](value)
            attn_output, _ = self.attention_layers[i](query, key, value)
            query = query + self.mlp_layers[i](self.layer_norm2[i](attn_output))
        return query
    
class TextFusionTransformer_ver4(nn.Module):
    def __init__(self, embed_dim, heads, layers=2, dropout=0.0, **kwargs):
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
    
    def forward(self, query, key, value, **kwargs):
        for i in range(self.layers):
            query = self.layer_norm1[i](query)
            key = self.layer_norm1[i](key)
            value = self.layer_norm1[i](value)
            attn_output, _ = self.attention_layers[i](query, key, value)
            query = query + self.mlp_layers[i](self.layer_norm2[i](attn_output))
        return query
    
class TextFusionTransformer_ver5(nn.Module):
    def __init__(self, embed_dim, heads, layers=2, dropout=0.0, class_num=20, **kwargs):
        super(TextFusionTransformer_ver5, self).__init__()
        self.layers = layers
        self.attention_layers = nn.ModuleList([
            nn.ModuleList([nn.MultiheadAttention(embed_dim, heads, dropout=dropout) for _ in range(layers)]) for _ in range(class_num)
        ])
        self.layer_norm1 = nn.ModuleList([
            nn.ModuleList([nn.LayerNorm(embed_dim) for _ in range(layers)]) for _ in range(class_num)
        ])
        self.layer_norm2 = nn.ModuleList([
            nn.ModuleList([nn.LayerNorm(embed_dim) for _ in range(layers)]) for _ in range(class_num)
        ])

        self.mlp_layers = nn.ModuleList([nn.ModuleList([
            nn.Sequential(
                nn.Linear(embed_dim, embed_dim),
                nn.GELU(),
                nn.Linear(embed_dim, embed_dim)
            ) for _ in range(layers)
        ]) for _ in range(class_num)])
    
    def forward(self, query, key, value, class_idx=0, **kwargs):
        for i in range(self.layers):
            query = self.layer_norm1[class_idx][i](query)
            key = self.layer_norm1[class_idx][i](key)
            value = self.layer_norm1[class_idx][i](value)
            attn_output, _ = self.attention_layers[class_idx][i](query, key, value)
            query = query + self.mlp_layers[class_idx][i](self.layer_norm2[class_idx][i](attn_output))
        return query

class TextFusionTransformer_ver6(nn.Module):
    def __init__(self, embed_dim, heads, layers=2, dropout=0.0, **kwargs):
        super(TextFusionTransformer_ver6, self).__init__()
        self.layers = layers
        self.attention_layers = nn.ModuleList([
            nn.MultiheadAttention(embed_dim, heads, dropout=dropout) for _ in range(layers)
        ])
        self.layer_norm1 = nn.ModuleList([nn.LayerNorm(embed_dim) for _ in range(layers)])
        self.layer_norm2 = nn.ModuleList([nn.LayerNorm(embed_dim) for _ in range(layers)])
        self.c = nn.ParameterList([nn.Parameter(torch.ones(1)) for _ in range(layers)])
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
            query = query + self.c[i] * self.mlp_layers[i](self.layer_norm2[i](attn_output))
        return query
    
class TextFusionTransformer_ver7(nn.Module):
    def __init__(self, embed_dim, heads, layers=2, dropout=0.0, class_num=20, **kwargs):
        super(TextFusionTransformer_ver7, self).__init__()
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

class TextFusionAvg(nn.Module):
    def __init__(self, embed_dim, **kwargs):
        super().__init__()
        self.avg_weight = nn.Parameter(torch.ones(embed_dim) / 2)
    
    def forward(self, org, cap, *args, **kwargs):
        return self.avg_weight * org + (1 - self.avg_weight) * cap
    
    
class TextFusionSubModule(nn.Module):
    """
    Module that reinforce the attribute by subtracting class prompt 
    from class specific caption.
    """
    def __init__(self, embed_dim, **kwargs):
        super().__init__()
        self.mlp_layer = nn.Linear(embed_dim, embed_dim)
        self.c = nn.Parameter(torch.ones(1,))
        
    def forward(self, org, cap, *args, **kwargs):
        diff = cap - org
        diff = self.mlp_layer(diff)
        diff *= self.c
        return org + diff