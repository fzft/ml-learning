from torch import nn
import torch
import math


class Attention(nn.Module):

    def __init__(self, d_model, d_k):
        super().__init__()

        # d_model is the dimension of the input
        self.d_model = d_model

        # d_k is the dimension of the key and value
        self.d_k = d_k

        self.q = nn.Linear(self.d_model, self.d_k, bias=False)
        self.k = nn.Linear(self.d_model, self.d_k, bias=False)
        self.v = nn.Linear(self.d_model, self.d_k, bias=False)

    def self_attention(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, mask=None) -> torch.Tensor:
        kt = k.transpose(-2, -1)
        scores = torch.matmul(q, kt) / math.sqrt(self.dim)
        if mask is not None:
            scores = scores.masked_fill(mask == 0, float('-inf'))
        scores = torch.softmax(scores, dim=-1)
        return torch.matmul(scores, v)

    def forward(self, x: torch.Tensor, mask=None) -> torch.Tensor:
        q = self.q(x)
        k = self.k(x)
        v = self.v(x)
        return self.self_attention(q, k, v, mask=mask)


class MultiHeadAttention(nn.Module):

    def __init__(self, d_model, n_heads):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads
        self.multi_head = nn.ModuleList([Attention(self.d_model, self.d_k) for _ in range(n_heads)])
        self.o = nn.Linear(self.d_k, self.d_model)

    def forward(self, x: torch.Tensor, mask=None) -> torch.Tensor:
        heads = [head(x, mask=mask) for head in self.multi_head]
        return self.o(torch.cat(heads, dim=-1))
