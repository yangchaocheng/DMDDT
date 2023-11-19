import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math
from math import sqrt
import os


class TriangularCausalMask():
    def __init__(self, B, L, device="cpu"):
        mask_shape = [B, 1, L, L]
        with torch.no_grad():
            self._mask = torch.triu(torch.ones(mask_shape, dtype=torch.bool), diagonal=1).to(device)

    @property
    def mask(self):
        return self._mask


def create_diag_symmetric_mask(seq_len, diagonal, device="cpu"):
    # Create a mask tensor of shape (seq_len, seq_len)
    mask = torch.zeros(seq_len, seq_len).to(device)

    # Set the diagonal and upper triangle to 1
    for i in range(seq_len):
        for j in range(i, min(seq_len, i + diagonal)):
            mask[i, j] = 1

    # Set the lower triangle to 1
    mask += mask.t().triu(diagonal=1)

    # Set the upper triangle to 1
    mask += mask.t().tril(diagonal=-1)

    # Return the mask tensor
    return mask

class NMA(nn.Module):
    def __init__(self, mask_scale=0, mask_flag=False, scale=None, attention_dropout=0.0, output_attention=False):
        super(NMA, self).__init__()
        self.scale = scale
        self.mask_flag = mask_flag
        self.mask_scale = mask_scale
        self.output_attention = output_attention
        self.dropout = nn.Dropout(attention_dropout)

    def forward(self, queries, keys, values, attn_mask):
        B, L, H, E = queries.shape
        _, S, _, D = values.shape
        scale = self.scale or 1. / sqrt(E)
        scores = torch.einsum("blhe,bshe->bhls", queries, keys)
        mask = create_diag_symmetric_mask(L, self.mask_scale, device=queries.device)
        if self.mask_flag:
            if attn_mask is None:
                attn_mask = TriangularCausalMask(B, L, device=queries.device)
            scores.masked_fill_(attn_mask.mask, -np.inf)
        scores = scores.masked_fill(mask.bool(), float('-inf'))
        attn = scale * scores

        A = self.dropout(torch.softmax(attn, dim=-1))
        A = torch.softmax(attn, dim=-1)
        V = torch.einsum("bhls,bshd->blhd", A, values)

        if self.output_attention:
            return (V.contiguous(), A)
        else:
            return (V.contiguous(), None)

class AttentionLayer(nn.Module):
    def __init__(self, attention, d_model, n_heads, d_keys=None,
                 d_values=None):
        super(AttentionLayer, self).__init__()

        d_keys = d_keys or (d_model // n_heads)
        d_values = d_values or (d_model // n_heads)
        self.norm = nn.LayerNorm(d_model)
        self.inner_attention = attention
        self.query_projection = nn.Linear(d_model,
                                          d_keys * n_heads)
        self.key_projection = nn.Linear(d_model,
                                        d_keys * n_heads)
        self.value_projection = nn.Linear(d_model,
                                          d_values * n_heads)
        self.out_projection = nn.Linear(d_values * n_heads, d_model)

        self.n_heads = n_heads

    def forward(self, queries, keys, values, attn_mask):
        B, L, _ = queries.shape
        _, S, _ = keys.shape
        H = self.n_heads
        queries = self.query_projection(queries).view(B, L, H, -1)
        keys = self.key_projection(keys).view(B, S, H, -1)
        values = self.value_projection(values).view(B, S, H, -1)

        output = self.inner_attention(
            queries,
            keys,
            values,
            attn_mask
        )
        out, A = output
        out = out.view(B, L, -1)
        out = self.out_projection(out)

        return out
