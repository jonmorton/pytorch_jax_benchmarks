import math

import torch
from torch import nn


class CausalSelfAttention(nn.Module):
    def __init__(self, dim, seq_len, dim_head=64):
        super().__init__()
        self.c_attn = nn.Linear(dim, 3 * dim, bias=False)
        self.c_proj = nn.Linear(dim, dim, bias=True)
        self.n_head = dim // dim_head
        self.n_embd = dim
        self.register_buffer(
            "bias",
            torch.tril(torch.ones(seq_len, seq_len)).view(1, 1, seq_len, seq_len),
        )

    def forward(self, x):
        (
            B,
            T,
            C,
        ) = x.size()

        q, k, v = nn.functional.linear(x, self.c_attn.weight.to(x.dtype)).split(
            self.n_embd, dim=2
        )
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(
            1, 2
        )  # (B, nh, T, hs)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(
            1, 2
        )  # (B, nh, T, hs)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(
            1, 2
        )  # (B, nh, T, hs)

        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
        att = att.masked_fill(self.bias[:, :, :T, :T] == 0, float("-inf"))
        att = nn.functional.softmax(att, dim=-1)
        y = att @ v  # (B, nh, T, T) x (B, nh, T, hs) -> (B, nh, T, hs)
        y = (
            y.transpose(1, 2).contiguous().view(B, T, C)
        )  # re-assemble all head outputs side by side

        # output projection
        y = nn.functional.linear(
            y, self.c_proj.weight.to(y.dtype), self.c_proj.bias.to(y.dtype)
        )
        return y
