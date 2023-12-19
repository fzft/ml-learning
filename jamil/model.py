import torch
import torch.nn as nn
import math
import torch.nn.functional as F


class TokenEmbeddings(nn.Module):

    def __init__(self, vocab_size, embedding_dim):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.vocab_size = vocab_size
        self.embedding = nn.Embedding(vocab_size, embedding_dim)

    def forward(self, x):
        return self.embedding(x) * math.sqrt(self.embedding_dim)


class PositionEmbeddings(nn.Module):

    def __init__(self, seq_len, embedding_dim, dropout: float):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.seq_len = seq_len
        self.dropout = nn.Dropout(dropout)

        # create a matrix of shape (seq_len, embedding_dim)
        pe = torch.zeros(seq_len, embedding_dim)
        # create a vector of shape (seq_len, 1)
        position = torch.arange(0, seq_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, embedding_dim, 2).float() * (-math.log(10000.0) / embedding_dim))
        # Apply the sin to even positions in the array; 2i
        pe[:, 0::2] = torch.sin(position * div_term)
        # Apply the cos to odd positions in the array; 2i+1
        pe[:, 1::2] = torch.cos(position * div_term)

        pe = pe.unsqueeze(0)  # (1, seq_len, embedding_dim)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + (self.pe[:, :x.shape[1], :]).requires_grad_(False)
        return self.dropout(x)


class LayerNormalization(nn.Module):

    def __init__(self, features: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.gamma = nn.Parameter(torch.ones(features))
        self.beta = nn.Parameter(torch.zeros(features))

    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)
        return self.gamma * (x - mean) / (std + self.eps) + self.beta


class FeedForwardBlock(nn.Module):

    def __init__(self, d_model, d_ff, dropout=0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
        )

    def forward(self, x):
        return self.net(x)


class MultiHeadAttentionBlock(nn.Module):
    def __init__(self, d_model, h, dropout: float):
        super().__init__()
        self.d_model = d_model
        self.h = h
        assert d_model % h == 0, "d_model must be a multiple of h"
        self.d_k = d_model // h
        self.w_q = nn.Linear(d_model, d_model, bias=False)
        self.w_k = nn.Linear(d_model, d_model, bias=False)
        self.w_v = nn.Linear(d_model, d_model, bias=False)
        self.out = nn.Linear(d_model, d_model, bias=False)
        self.dropout = nn.Dropout(dropout)

    @staticmethod
    def attention(q, k, v, mask, dropout: nn.Dropout):
        d_k = q.shape[-1]
        attention_scores = (q @ k.transpose(-2, -1)) / math.sqrt(d_k)
        # (B, h, T, d_k) * (B, h, d_k, T) -> (B, h, T, T)
        if mask is not None:
            attention_scores.masked_fill_(mask == 0, -1e9)
        attention_scores = F.softmax(attention_scores, dim=-1)
        if dropout is not None:
            attention_scores = dropout(attention_scores)
        return (attention_scores @ v), attention_scores  # (B, h, T, T) * (B, h, T, d_k) -> (B, h, T, d_k)

    def forward(self, q, k, v, mask=None):
        query = self.w_q(q)  # (B, T, d_model) -> (B, T, d_model)
        key = self.w_k(k)
        value = self.w_v(v)


        query = query.view(q.shape[0], q.shape[1], self.h, self.d_k).transpose(1,
                                                                               2)  # (B, T, d_model) -> (B, h, T, d_k)
        key = key.view(k.shape[0], k.shape[1], self.h, self.d_k).transpose(1, 2)  # (B, T, d_model) -> (B, h, T, d_k)
        value = value.view(v.shape[0], v.shape[1], self.h, self.d_k).transpose(1,
                                                                               2)  # (B, T, d_model) -> (B, h, T, d_k)

        x, self.scores = MultiHeadAttentionBlock.attention(query, key, value, mask, self.dropout)  # (B, h, T, d_k)

        x = x.transpose(1, 2).contiguous().view(x.shape[0], -1, self.h * self.d_k)  # (B, h, T, d_k) -> (B, T, d_model)

        return self.out(x)


class ResidualConnection(nn.Module):
    def __init__(self, features: int, dropout):
        super().__init__()
        self.norm = LayerNormalization(features)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, sublayer):
        return x + self.dropout(sublayer(self.norm(x)))


class EncoderBlock(nn.Module):
    def __init__(self, features: int, self_attention_block: MultiHeadAttentionBlock,
                 feed_forward_block: FeedForwardBlock, dropout):
        super().__init__()
        self.attention = self_attention_block
        self.feed_forward = feed_forward_block
        self.residual_connection = nn.ModuleList([ResidualConnection(features, dropout) for _ in range(2)])

    def forward(self, x, mask):
        x = self.residual_connection[0](x, lambda x: self.attention(x, x, x, mask))
        x = self.residual_connection[1](x, self.feed_forward)
        return x


class Encoder(nn.Module):
    def __init__(self, features: int, layers: nn.ModuleList):
        super().__init__()
        self.layers = layers
        self.norm = LayerNormalization(features)

    def forward(self, x, mask):
        for layer in self.layers:
            x = layer(x, mask)
        return self.norm(x)


class DecoderBlock(nn.Module):

    def __init__(self, features: int, self_attention_block: MultiHeadAttentionBlock,
                 cross_attention_block: MultiHeadAttentionBlock,
                 feed_forward_block: FeedForwardBlock, dropout):
        super().__init__()
        self.self_attention_block = self_attention_block
        self.cross_attention = cross_attention_block
        self.feed_forward = feed_forward_block
        self.residual_connection = nn.ModuleList([ResidualConnection(features, dropout) for _ in range(3)])

    def forward(self, x, encoder_output, src_mask, tgt_mask):
        x = self.residual_connection[0](x, lambda x: self.self_attention_block(x, x, x, tgt_mask))
        x = self.residual_connection[1](x, lambda x: self.cross_attention(x, encoder_output, encoder_output, src_mask))
        x = self.residual_connection[2](x, self.feed_forward)
        return x


class Decoder(nn.Module):

    def __init__(self, features: int, layers: nn.ModuleList):
        super().__init__()
        self.layers = layers
        self.norm = LayerNormalization(features)

    def forward(self, x, encoder_output, src_mask, tgt_mask):
        for layer in self.layers:
            x = layer(x, encoder_output, src_mask, tgt_mask)
        return self.norm(x)


class ProjectionLayer(nn.Module):

    def __init__(self, d_model, vocab_size):
        super().__init__()
        self.linear = nn.Linear(d_model, vocab_size)

    def forward(self, x):
        return self.linear(x)


class Transformer(nn.Module):

    def __init__(self, encoder: Encoder, decoder: Decoder, src_embed: TokenEmbeddings, tgt_embed: TokenEmbeddings,
                 src_pos: PositionEmbeddings, tgt_pos: PositionEmbeddings,
                 prj_layer: ProjectionLayer):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.src_embed = src_embed
        self.tgt_embed = tgt_embed
        self.src_pos = src_pos
        self.tgt_pos = tgt_pos
        self.prj_layer = prj_layer

    def encode(self, src, src_mask):
        src = self.src_embed(src)
        src = self.src_pos(src)
        return self.encoder(src, src_mask)

    def decode(self, encoder_output, src_mask, tgt, tgt_mask):
        # （B, T, C）
        tgt = self.tgt_embed(tgt)
        tgt = self.tgt_pos(tgt)
        return self.decoder(tgt, encoder_output, src_mask, tgt_mask)

    def project(self, x):
        return self.prj_layer(x)


def build_transformer(src_vocab_size, tgt_vocab_size, src_seq_len, tgt_seq_len, d_model: int = 512, n_layers: int = 6,
                      n_heads: int = 8, dropout: float = .1, d_ff: int = 2048):
    # create the token embeddings
    src_embed = TokenEmbeddings(src_vocab_size, d_model)
    tgt_embed = TokenEmbeddings(tgt_vocab_size, d_model)

    # create the position embeddings
    src_pos = PositionEmbeddings(src_seq_len, d_model, dropout)
    tgt_pos = PositionEmbeddings(tgt_seq_len, d_model, dropout)

    # create the encoder
    encoder_blocks = []
    for _ in range(n_layers):
        encoder_blocks.append(EncoderBlock(d_model, MultiHeadAttentionBlock(d_model, n_heads, dropout),
                                           FeedForwardBlock(d_model, d_ff, dropout), dropout))

    # create the decoder
    decoder_blocks = []
    for _ in range(n_layers):
        decoder_blocks.append(DecoderBlock(d_model, MultiHeadAttentionBlock(d_model, n_heads, dropout),
                                           MultiHeadAttentionBlock(d_model, n_heads, dropout),
                                           FeedForwardBlock(d_model, d_ff, dropout), dropout))

    encoder = Encoder(d_model, nn.ModuleList(encoder_blocks))
    decoder = Decoder(d_model, nn.ModuleList(decoder_blocks))

    # create the projection layer
    prj_layer = ProjectionLayer(d_model, tgt_vocab_size)

    # create the transformer
    transformer = Transformer(encoder, decoder, src_embed, tgt_embed, src_pos, tgt_pos, prj_layer)

    # initialize the parameters
    for p in transformer.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform_(p)

    print("model parameters number: ", sum(p.numel() for p in transformer.parameters() if p.requires_grad))

    return transformer
