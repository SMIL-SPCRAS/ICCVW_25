# coding: utf-8
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from torch.nn.functional import silu
from torch.nn.functional import softplus
from einops import rearrange, einsum

class PositionWiseFeedForward(nn.Module):
    def __init__(self, input_dim, hidden_dim, dropout=0.1):
        super().__init__()
        self.layer_1 = nn.Linear(input_dim, hidden_dim)
        self.layer_2 = nn.Linear(hidden_dim, input_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = self.layer_1(x)
        x = F.gelu(x)
        x = self.dropout(x)
        return self.layer_2(x)


class AddAndNorm(nn.Module):
    def __init__(self, input_dim, dropout=0.1):
        super().__init__()
        self.norm = nn.LayerNorm(input_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, residual):
        return self.norm(x + self.dropout(residual))


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, d_model)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        self.register_buffer("pe", pe)

    def forward(self, x):
        x = x + self.pe[: x.size(1)].detach()
        return self.dropout(x)


class TransformerEncoderLayer(nn.Module):
    def __init__(self, input_dim, num_heads, dropout=0.1, positional_encoding=False):
        super().__init__()
        self.input_dim = input_dim
        self.self_attention = nn.MultiheadAttention(input_dim, num_heads, dropout=dropout, batch_first=True)
        # self.self_attention = MHA(
        #     embed_dim=input_dim,
        #     num_heads=num_heads,
        #     dropout=dropout,
        #     # bias=True,
        #     use_flash_attn=True
        # )
        self.feed_forward = PositionWiseFeedForward(input_dim, input_dim, dropout=dropout)
        self.add_norm_after_attention = AddAndNorm(input_dim, dropout=dropout)
        self.add_norm_after_ff = AddAndNorm(input_dim, dropout=dropout)
        self.positional_encoding = PositionalEncoding(input_dim) if positional_encoding else None

    def forward(self, key, value, query):
        if self.positional_encoding:
            key = self.positional_encoding(key)
            value = self.positional_encoding(value)
            query = self.positional_encoding(query)

        attn_output, _ = self.self_attention(query, key, value, need_weights=False)
        # attn_output = self.self_attention(query, key, value)

        x = self.add_norm_after_attention(attn_output, query)

        ff_output = self.feed_forward(x)
        x = self.add_norm_after_ff(ff_output, x)

        return x

class MambaBlock(nn.Module):
    def __init__(self, d_input, d_model, d_state=16, d_discr=None, ker_size=4, dropout=0., device='cuda'):
        super().__init__()
        d_discr = d_discr if d_discr is not None else d_model // 16
        self.in_proj  = nn.Linear(d_input, 2 * d_model, bias=False)
        self.out_proj = nn.Linear(d_model, d_input, bias=False)
        self.s_B = nn.Linear(d_model, d_state, bias=False)
        self.s_C = nn.Linear(d_model, d_state, bias=False)
        self.s_D = nn.Sequential(nn.Linear(d_model, d_discr, bias=False), nn.Linear(d_discr, d_model, bias=False),)
        self.conv = nn.Conv1d(
            in_channels=d_model,
            out_channels=d_model,
            kernel_size=ker_size,
            padding=ker_size - 1,
            groups=d_model,
            bias=True,
        )
        self.A = nn.Parameter(torch.arange(1, d_state + 1, dtype=torch.float).repeat(d_model, 1))
        self.D = nn.Parameter(torch.ones(d_model, dtype=torch.float))
        self.add_norm = AddAndNorm(d_model, dropout=dropout)
        self.device = device

    def forward(self, seq, cache=None):
        b, l, d = seq.shape
        (prev_hid, prev_inp) = cache if cache is not None else (None, None)
        a, b = self.in_proj(seq).chunk(2, dim=-1)
        x = rearrange(a, 'b l d -> b d l')
        x = x if prev_inp is None else torch.cat((prev_inp, x), dim=-1)
        a = self.conv(x)[..., :l]
        a = rearrange(a, 'b d l -> b l d')
        a = silu(a)
        a, hid = self.ssm(a, prev_hid=prev_hid)
        b = silu(b)
        out = a * b
        out =  self.out_proj(out)
        if cache:
            cache = (hid.squeeze(), x[..., 1:])
        return out, cache

    def ssm(self, seq, prev_hid):
        A = -self.A
        D = +self.D
        B = self.s_B(seq)
        C = self.s_C(seq)
        s = softplus(D + self.s_D(seq))
        A_bar = einsum(torch.exp(A), s, 'd s,   b l d -> b l d s')
        B_bar = einsum(          B,  s, 'b l s, b l d -> b l d s')
        X_bar = einsum(B_bar, seq, 'b l d s, b l d -> b l d s')
        hid = self._hid_states(A_bar, X_bar, prev_hid=prev_hid)
        out = einsum(hid, C, 'b l d s, b l s -> b l d')
        # out = out + D * seq
        out = self.add_norm(D * seq, out)
        return out, hid

    def _hid_states(self, A, X, prev_hid=None):
        b, l, d, s = A.shape
        A = rearrange(A, 'b l d s -> l b d s')
        X = rearrange(X, 'b l d s -> l b d s')
        if prev_hid is not None:
            return rearrange(A * prev_hid + X, 'l b d s -> b l d s')
        h = torch.zeros(b, d, s, device=self.device)
        return torch.stack([h := A_t * h + X_t for A_t, X_t in zip(A, X)], dim=1)
