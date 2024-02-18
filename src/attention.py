import torch
from torch import nn
import torch.nn.functional as F


class SelfAttention(nn.Module):
    def __init__(self, n_heads, d_embed, in_proj_bias=True, out_proj_bias=True):
        super().__init__()

        self.n_heads = n_heads
        self.d_head = d_embed // n_heads  # Dim of each attention head

        # Combines the Wq, Wk and Wv matrices into one matrix
        self.in_proj = nn.Linear(d_embed, 3 * d_embed, bias=in_proj_bias)

        # This represents the Wo matrix
        self.out_proj = nn.Linear(d_embed, d_embed, bias=out_proj_bias)

    def forward(self, x: torch.Tensor, causal_mask=False) -> torch.Tensor:
        # x = (batch_size, sequence_length, dim)
        input_shape = x.shape
        batch_size, sequence_length, d_embed = input_shape

        # (batch_size, seq_len, n_heads, dim / n_heads)
        interim_shape = (batch_size, sequence_length, self.n_heads, self.d_head)

        # (batch_size, seq_len, dim) -> (batch_size, seq_len, dim * 3) -> 3 * (batch_size, seq_len, dim )
        q, k, v = self.in_proj(x).chunk(3, dim=-1)

        # (batch_size, seq_len, dim) -> (batch_size, seq_len, heads, dim / heads) ->(transpose) (batch_size, heads, seq_len, dim / heads)
        q = q.view(interim_shape).transpose(1, 2)
        k = k.view(interim_shape).transpose(1, 2)
        v = v.view(interim_shape).transpose(1, 2)

        # matrix mult Q.Kt
        weight = q @ k.transpose(-1, -2)

        if causal_mask:
            # Mask where the upper triangle (above the principal diagonal) is made up of 1
            mask = torch.ones_like(weight, dtype=torch.bool).triu(1)
            weight.masked_fill_(mask, -torch.inf)

        weight /= math.sqrt(self.d_head)
        weight = F.softmax(weight, dim=-1)

        # (batch_size, h, seq_len, seq_len) @ (batch_size, h, seq_len, dim/h) -> (batch_size, h, seq_len, dim/h)
        output = weight @ v

        # (batch_size, h, seq_len, dim/h) -> (batch_size, seq_len, h,  dim/h)
        output = output.transpose(1, 2)

        # (batch_size, seq_len, h,  dim/h) -> (batch_size, seq_len, dim)
        output = output.reshape(input_shape)

        output = self.out_proj(output)

        # (batch_size, seq_len, dim)
        return output
