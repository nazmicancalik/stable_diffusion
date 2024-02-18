import torch
from torch import nn
from torch.nn import functional as F
from attention import SelfAttention


class VAE_AttentionBlock(nn.Module):
    def __init__(self, channels):
        super.__init__()
        self.groupnorm = nn.GroupNorm(32, channels)
        self.attention = SelfAttention(1, channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x = (batch_size, features, height, width)

        residue = x

        # (batch_size, features, height, width) -> (batch_size, features, height, width)
        x = self.groupnorm(x)

        b, c, h, w = x.shape

        # (batch_size, features, height, width) -> (batch_size, features, height * width)
        # For attention make the input 1D, pixelwise self attention
        x = x.view((b, c, h * w))

        # Features must be the last dimension before feeding it to attention
        # (batch_size, features, height*width) -> (batch_size, height*width, features)
        x = x.transpose(-1, -2)

        # Self attention without MASK
        # (batch_size, height*width, features) -> (batch_size, height*width, features)
        # query, key and value are the same input hence self.
        # Features are at the last dimension
        x = self.attention(x)

        # Transpose back to features being second dim
        # (batch_size, height*width, features) -> (batch_size, features, height*width)
        x = x.transpose(-1, -2)

        # Go back to image look
        # (batch_size, features, height*width) -> (batch_size, features, height, width)
        x = x.view((b, c, h, w))

        # Add residual connection
        x += residue

        # (batch_size, features, height, width)
        return x


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


class VAE_Decoder(nn.Sequential):

    def __init__(self):
        super.__init__(
            nn.Conv2d(4, 4, kernel_size=1, padding=0),
            nn.Conv2d(4, 512, kernel_size=3, padding=1),
            VAE_ResidualBlock(512, 512),
            VAE_AttentionBlock(512),
            VAE_ResidualBlock(512, 512),
            VAE_ResidualBlock(512, 512),
            VAE_ResidualBlock(512, 512),
            # (bs, 512, height/8, width/8) -> (bs, 512, height/8, width/8)
            VAE_ResidualBlock(512, 512),
            # (bs, 512, height/8, width/8) -> (bs, 512, height/4, width/4)
            nn.Upsample(scale_factor=2),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            VAE_ResidualBlock(512, 512),
            VAE_ResidualBlock(512, 512),
            VAE_ResidualBlock(512, 512),
            # (bs, 512, height/4, width/4) -> (bs, 512, height/2, width/2)
            nn.Upsample(scale_factor=2),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            VAE_ResidualBlock(512, 256),
            VAE_ResidualBlock(256, 256),
            VAE_ResidualBlock(256, 256),
            # (bs, 256, height/2, width/2) -> (bs, 512, height, width)
            nn.Upsample(scale_factor=2),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            VAE_ResidualBlock(256, 128),
            VAE_ResidualBlock(128, 128),
            VAE_ResidualBlock(128, 128),
            nn.GroupNorm(32, 128),
            nn.SiLU(),
            # (bs, 128, height, width) -> (bs, 3, height, width)
            nn.Conv2d(128, 3, kernel_size=3, padding=1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x : (bs, 4, height/8, width/8)
        x /= 0.18215

        for module in self:
            x = module(x)

        # (bs, 3, height, width)
        return x
