import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from dataclasses import dataclass
from typing import Optional, Any, Tuple, List, Union


@dataclass
class ModelArgs:
    dim: int = 4096
    n_layers: int = 32
    n_heads: int = 32
    n_kv_heads: Optional[int] = None
    vocab_size: int = -1  # this will be set when we load the tokenizer
    multiple_of: int = 256
    ffn_dim_multiplier: Optional[int] = None
    norm_eps: float = 1e-6

    # Needed for KV cache
    max_batch_size: int = 32
    max_seq_len: int = 2048

    device: str = None


def precompute_theta_pos_freqs(head_dim: int, max_seq_len: int, device: str = "cpu", theta: float = 10000.0):
    # As written in the paper, the dimension of the embedding must be even
    assert head_dim % 2 == 0, "The dimension of the embedding must be even"
    # Build the theta parameter
    # According to the formula, theta_i = 10000 ^ (-2(i-1)/dim) for i in [1, dim/2]
    # Shape: (dim // 2)
    theta_numerator = torch.arange(0, head_dim, 2, device=device).float()
    theta = 1.0 / (theta ** (theta_numerator / head_dim)).to(device)
    m = torch.arange(0, max_seq_len, device=device).float()
    # Multiple each theta by each position using the out product
    freqs = torch.outer(m, theta).float()
    # we can now compute the complex numbers in the polar form c=R*exp(i*m*theta) where R=1
    # （seq_len, head_dim / 2) -> (seq_len, head_dim/2)
    freqs_complex = torch.polar(torch.ones_like(freqs), freqs)
    return freqs_complex


def apply_rotary_pos_emb(x: torch.Tensor, freqs_complex: torch.Tensor, device: str = "cpu"):
    # (B, T, H, head_dim) -> (B, T, H, head_dim // 2)
    x_complex = torch.view_as_complex(x.float().reshape(*x.shape[:-1], -1, 2))
    # (T, head_dim // 2) -> (1, T, H, head_dim // 2)
    freqs_complex = freqs_complex.unsqueeze(0).unsqueeze(2)
    # (B, T, H, head_dim // 2) * (1, T, H, head_dim // 2) -> (B, T, H, head_dim // 2)
    x_rotated = x_complex * freqs_complex
    # (B, T, H, head_dim // 2) -> (B, T, H, head_dim// 2, 2)
    x_out = torch.view_as_real(x_rotated)
    # (B, T, H, head_dim// 2, 2) -> (B, T, H, head_dim)
    x_out = x_out.reshape(*x.shape)
    return x_out.type_as(x).to(device)


class RMSNorm(nn.Module):

    def __init__(self, dim: int, eps: float = 1e-8):
        super().__init__()
        self.eps = eps
        # The gamma parameter is the same as the one in the LayerNorm
        self.weight = nn.Parameter(torch.ones(dim))

    def _norm(self, x: torch.Tensor, dim: int):
        # (B, T, C)
        return x * torch.rsqrt(torch.mean(x ** 2, dim=dim, keepdim=True) + self.eps)

    def forward(self, x: torch.Tensor):
        # (B, T, C)
        return self.weight * self._norm(x.float(), dim=-1).type_as(x)


class SelfAttention(nn.Module):

    def __init__(self, args: ModelArgs):
        super().__init__()
        # Indicate the number of heads for the key and value
        self.n_kv_heads = args.n_kv_heads if args.n_kv_heads is not None else args.n_heads

        # Indicate the number of heads for the query
        self.n_heads_q = args.n_heads

        # Indicate how many times the keys and values are repeated
        self.n_rep = self.n_heads_q // self.n_kv_heads
        # Indicate the dimension of the key and value
        self.head_dim = args.dim // args.n_heads

        self.wq = nn.Linear(args.dim, args.n_heads * self.head_dim, bias=False)
        self.wk = nn.Linear(args.dim, self.n_kv_heads * self.head_dim, bias=False)
        self.wv = nn.Linear(args.dim, self.n_kv_heads * self.head_dim, bias=False)
        self.wo = nn.Linear(args.n_heads * self.head_dim, args.dim, bias=False)

        self.cache_k = torch.zeros((args.max_batch_size, args.max_seq_len, self.n_kv_heads, self.head_dim))
        self.cache_v = torch.zeros((args.max_batch_size, args.max_seq_len, self.n_kv_heads, self.head_dim))

    def forward(self, x: torch.Tensor, start_pos: int, freqs_complex: torch.Tensor):
        # (B, 1, C)
        batch_size, seq_len, _ = x.shape

        # (B, T, C) -> (B, T, n_heads, head_dim)
        q = self.wq(x).view(batch_size, seq_len, self.n_heads_q, self.head_dim)
        # (B, T, C) -> (B, T, n_kv_heads, head_dim)
        k = self.wk(x).view(batch_size, seq_len, self.n_kv_heads, self.head_dim)
        # (B, T, C) -> (B, T, n_kv_heads, head_dim)
        v = self.wv(x).view(batch_size, seq_len, self.n_kv_heads, self.head_dim)

        xq = apply_rotary_pos_emb(q, freqs_complex, device=x.device)
        xk = apply_rotary_pos_emb(k, freqs_complex, device=x.device)

        self.cache_k.to(x.device)
        self.cache_v.to(x.device)

        # Replace the entry in the cache for this token
        self.cache_k[:batch_size, start_pos:start_pos + seq_len] = xk

        self.cache_v[:batch_size, start_pos:start_pos + seq_len] = v

        # Retrieve the cached keys and values
        keys = self.cache_k[:batch_size, :start_pos + seq_len]
        values = self.cache_v[:batch_size, :start_pos + seq_len]

        # Repeat the heads of the K and V to reach the same number of heads as the Q
        # (B, T, n_kv_heads, head_dim) -> (B, T, n_heads, head_dim)
        keys = keys.repeat_interleave(self.n_rep, dim=2).to(x.device)

        # (B, T, n_kv_heads, head_dim) -> (B, n_kv_heads, T, head_dim)
        values = values.repeat_interleave(self.n_rep, dim=2).to(x.device)

        # (B, T, n_heads, head_dim) -> (B, n_heads, 1, head_dim)
        xq = xq.transpose(1, 2)

        # (B, kv_seq_len, n_kv_heads, head_dim) -> (B, n_kv_heads, kv_seq_len, head_dim)
        keys = keys.transpose(1, 2)

        # (B, kv_seq_len, n_kv_heads, head_dim) -> (B, n_kv_heads, kv_seq_len, head_dim)
        values = values.transpose(1, 2)

        # (B, n_heads, 1, head_dim) * (B, n_heads, head_dim, kv_seq_len) -> (B, n_heads, 1, kv_seq_len)
        scores = torch.matmul(xq, keys.transpose(-2, -1)) / math.sqrt(self.head_dim)
        scores = F.softmax(scores, dim=-1).type_as(xq)

        # (B, n_heads, 1, kv_seq_len) * (B, n_heads, kv_seq_len, head_dim) -> (B, n_heads, 1, head_dim)
        out = torch.matmul(scores, values)

        # (B, n_heads, 1, head_dim) -> (B, 1, n_heads, head_dim) -> (B, 1, n_heads * head_dim)
        out = self.wo(out.transpose(1, 2).contiguous().view(batch_size, seq_len, -1))
        return out


class FeedForward(nn.Module):

    def __init__(self, args: ModelArgs):
        super().__init__()
        hidden_dim = args.dim * 4
        hidden_dim = int(2 * hidden_dim / 3)
        if args.ffn_dim_multiplier is not None:
            hidden_dim = args.ffn_dim_multiplier * args.dim
        hidden = args.multiple_of * ((hidden_dim + args.multiple_of - 1) // args.multiple_of)

        self.w1 = nn.Linear(args.dim, hidden, bias=False)
        self.w2 = nn.Linear(hidden, args.dim, bias=False)
        self.w3 = nn.Linear(args.dim, hidden, bias=False)


    def forward(self, x: torch.Tensor):
        # (B, T, C) -> (B, T, hidden)
        swish = F.silu(self.w1(x))

        # (B, T, hidden) -> (B, T, hidden)
        x_V = self.w3(x)

        # (B, T, hidden) * (B, T, hidden) -> (B, T, hidden)
        out = swish * x_V

        # (B, T, C) -> (B, T, C)
        return self.w2(out)


class EncoderBlock(nn.Module):

    def __init__(self, args: ModelArgs):
        super().__init__()

        self.args = args
        self.dim = args.dim
        self.head_dim = self.dim // args.n_heads

        self.attention = SelfAttention(args)
        self.feed_forward = FeedForward(args)

        # Norms BEFORE the self-attention
        self.attention_norm = RMSNorm(self.dim, eps=args.norm_eps)
        # Norms BEFORE the feed-forward
        self.ffn_norm = RMSNorm(self.dim, eps=args.norm_eps)

    def forward(self, x: torch.Tensor, start_pos: int, freqs_complex: torch.Tensor):
        # (B, T, C) + (B, T, C)
        h = x + self.attention.forward(self.attention_norm(x), start_pos, freqs_complex)
        out = h + self.feed_forward.forward(self.ffn_norm(h))
        return out


class Transformer(nn.Module):

    def __init__(self, args: ModelArgs):
        super().__init__()

        assert args.vocab_size > 0, "vocab_size must be set"

        self.args = args
        self.vocab_size = args.vocab_size
        self.n_layers = args.n_layers
        self.tok_embeddings = nn.Embedding(self.vocab_size, args.dim)

        self.layers = nn.ModuleList()
        for _ in range(self.n_layers):
            self.layers.append(EncoderBlock(args))

        self.norm = RMSNorm(args.dim, eps=args.norm_eps)
        self.output = nn.Linear(args.dim, self.vocab_size, bias=False)

        self.freqs_complex = precompute_theta_pos_freqs(self.args.dim // self.args.n_heads, args.max_seq_len * 2,
                                                        device=self.args.device)

    def forward(self, tokens: torch.Tensor, start_pos: int):
        # (B, T)
        batch_size, seq_len = tokens.shape
        assert seq_len == 1, "Only one token at a time"

        print("dtype", tokens.dtype, "batch_size", batch_size)

        # (B, T) -> (B, T, C)
        h = self.tok_embeddings(tokens)

        # retrieve the pairs (m, theta) for the positional encoding (start_pos, start_pos+seq_len)
        freqs_complex = self.freqs_complex[start_pos:start_pos + seq_len]

        # consecutively apply all the encoder layer
        for layer in self.layers:
            h = layer(h, start_pos, freqs_complex)

        output = self.output(self.norm(h)).float()
        return output

    def count_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


