import torch
from torch import nn
from torch.nn import functional as F
from attention import SelfAttention


class CLIPEmbedding(nn.Module):
    def __init__(self, n_vocab: int, n_embd: int, n_tokens: int):
        super.__init__()

        self.token_embedding = nn.Embedding(n_vocab, n_embd)
        self.position_embedding = nn.Parameter(torch.zeros(n_tokens, n_embd))

    def forward(self, tokens):
        # (bs, seq_len) -> (bs, seq_len, dim)
        x = self.token_embedding(tokens)

        x += self.position_embedding

        return x


class CLIP(nn.module):
    def __init__(self):
        self.embedding = CLIPEmbedding(
            49408,
            768,
            77,
        )
        self.layers = nn.Module([CLIPLayer(12, 768) for i in range(12)])
        self.layernorm = nn.LayerNorm(768)

    def forward(self, tokens: torch.LongTensor) -> torch.FloatTensor:
        tokens = tokens.type(torch.long)

        # (bs, seq_len) -> (bs, seq_len, dim)
        state = self.embedding(tokens)

        for layer in self.layers:
            state = layer(state)

        # (bs, seq_len, dim)
        output = self.layernorm(state)

        return output
