import torch
import torch.nn as nn
import math
from copy import deepcopy

class EventRGBFusionBlock(nn.Module):
    def __init__(self, dim, num_heads=8, attn_dropout=0.0, ff_hidden_mult=4):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.rgb_to_evt_attn = nn.MultiheadAttention(dim, num_heads, dropout=attn_dropout, batch_first=True)
        self.evt_to_rgb_attn = nn.MultiheadAttention(dim, num_heads, dropout=attn_dropout, batch_first=True)


        # small feed-forward blocks for each stream
        self.rgb_ff = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, dim * ff_hidden_mult),
            nn.GELU(),
            nn.Linear(dim * ff_hidden_mult, dim),
        )
        self.evt_ff = deepcopy(self.rgb_ff)

        # single layer to reduce dimension by half after fusion

        self.fusion_linear = nn.Linear(dim * 2, dim)


        self.rgb_norm = nn.LayerNorm(dim)
        self.evt_norm = nn.LayerNorm(dim)


    def forward(self, rgb_tokens, evt_tokens):
        # cross-attention: rgb queries event (rgb <- evt)
        rgb_q = self.rgb_norm(rgb_tokens)
        evt_kv = self.evt_norm(evt_tokens)
        rgb_attn_out, _ = self.evt_to_rgb_attn(rgb_q, evt_kv, evt_kv)
        rgb = rgb_tokens + rgb_attn_out
        rgb = rgb + self.rgb_ff(rgb)


        # cross-attention: event queries rgb (evt <- rgb)
        evt_q = self.evt_norm(evt_tokens)
        rgb_kv = self.rgb_norm(rgb_tokens)
        evt_attn_out, _ = self.rgb_to_evt_attn(evt_q, rgb_kv, rgb_kv)
        evt = evt_tokens + evt_attn_out
        evt = evt + self.evt_ff(evt)

        # concatenate along embedding dimension and reduce dimension
        combined = torch.cat((rgb, evt), dim=-1)  # B, N, 2*dim
        out = self.fusion_linear(combined)


        return out



class CrossAttentionFusion(nn.Module):
    """Simple cross-attention fusion block: query tokens are updated using kv tokens.
    Uses a residual + MLP like a transformer block.
    """
    def __init__(self, embed_dim, num_heads=8, mlp_ratio=4.0, dropout=0.0):
        super().__init__()
        self.attn = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout, batch_first=True)
        self.norm1 = nn.LayerNorm(embed_dim)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, int(embed_dim * mlp_ratio)),
            nn.GELU(),
            nn.Linear(int(embed_dim * mlp_ratio), embed_dim),
        )
        self.norm2 = nn.LayerNorm(embed_dim)

    def forward(self, query, key_value, pos_q=None, pos_k=None, attn_mask=None):
        # query: (B, Nq, C)
        # key_value: (B, Nk, C)
        # optional: add relative pos bias by adding to query/key (not implemented here)
        q = self.norm1(query)
        k_v = key_value  # already tokens; if you want pos you can add small proj to embed pos
        attn_out, _ = self.attn(q, k_v, k_v, key_padding_mask=attn_mask)
        query = query + attn_out
        # MLP + residual
        query = query + self.mlp(self.norm2(query))
        return query