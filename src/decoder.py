import torch
from torch import nn
from torch.nn import functional as F
from attention import SelfAttention


class VAE_ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super.__init__()
        self.groupnorm1 = nn.GroupNorm(32, in_channels)
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)

        self.groupnorm2 = nn.GroupNorm(32, out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)

        if in_channels == out_channels:
            self.residual_layer = nn.Identity()
        else:
            self.residual_layer = nn.Conv2d(
                in_channels, out_channels, kernel_size=1, padding=0
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x = (batch_size, in_channels, height, width)

        # save for later
        residue = x

        x = self.groupnorm1(x)
        x = F.silu(x)
        x = self.conv1(x)

        x = self.groupnorm2(x)
        x = F.silu(x)
        x = self.conv2(x)

        return x + self.residual_layer(residue)
