import torch
from torch import nn
from torch.nn import functional as F
from decoder import VAE_AttentionBlock, VAE_ResidualBlock


class VAE_Encoder(nn.Sequential):
    def __init__(self):
        super.__init__(
            # (b, channel, h, w) -> (b, 128, h, w)
            nn.Conv2d(3, 128, kernel_size=3, padding=1),
            # (b, 128, h, w) -> (b, 128, h, w)
            VAE_ResidualBlock(128, 128),
            # (b, 128, h, w) -> (b, 128, h, w)
            VAE_ResidualBlock(128, 128),
            # (b, 128, h, w) -> (b, 128, h/2, w/2)
            nn.Conv2d(128, 128, kernel_size=3, stride=2, padding=0),
            # (b, 128, h/2, w/2) -> (b, 256, h/2, w/2)
            VAE_ResidualBlock(128, 256),
            # (b, 256, h/2, w/2) -> (b, 256, h/2, w/2)
            VAE_ResidualBlock(256, 256),
            # (b, 256, h/2, w/2) -> (b, 256, h/2, w/2)
            VAE_ResidualBlock(256, 256),
            # (b, 256, h/2, w/2) -> (b, 256, h/4, w/4)
            nn.Conv2d(256, 256, kernel_size=3, stride=2, padding=0),
            # (b, 256, h/4, w/4) -> (b, 512, h/4, w/4)
            VAE_ResidualBlock(256, 512),
            # (b, 512, h/4, w/4) -> (b, 256, h/4, w/4)
            VAE_ResidualBlock(512, 512),
            # (b, 512, h/4, w/4) -> (b, 256, h/8, w/8)
            nn.Conv2d(512, 512, kernel_size=3, stride=2, padding=0),
            VAE_ResidualBlock(512, 512),
            VAE_ResidualBlock(512, 512),
            # (b, 512, h/8, w/8) -> (b, 512, h/8, w/8)
            VAE_ResidualBlock(512, 512),
            # (b, 512, h/8, w/8) -> (b, 512, h/8, w/8)
            VAE_AttentionBlock(512),
            # (b, 512, h/8, w/8) -> (b, 512, h/8, w/8)
            VAE_ResidualBlock(512, 512),
            # (b, 512, h/8, w/8) -> (b, 512, h/8, w/8)
            nn.GroupNorm(32, 512),
            # (b, 512, h/8, w/8) -> (b, 512, h/8, w/8)
            nn.SiLU(),
            # (b, 512, h/8, w/8) -> (b, 8, h/8, w/8)
            nn.Conv2d(512, 8, kernel_size=3, padding=1),
            # (b, 8, h/8, w/8) -> (b, 8, h/8, w/8)
            nn.Conv2d(8, 8, kernel_size=1, padding=0),
        )

    def forward(self, x: torch.Tensor, noise: torch.Tensor) -> torch.Tensor:
        # x = (batch_size, channel, height, width)
        # noise = (batch_size, out_channels, height/8, width/8)

        for module in self:
            if getattr(module, "stride", None) == (2, 2):
                # (padding_left, padding_right, padding_top, padding_bottom)
                x = F.pad(x, (0, 1, 0, 1))
            x = module(x)

        # (b, 8, h/8, w/8 ) -> 2x (b, 4, h/8, w/8)
        mean, log_variance = torch.chunk(x, 2, dim=1)

        # If the log variance too big or small then clamp it to region
        # (b, 4, h/8, w/8 ) -> 2x (b, 4, h/8, w/8)
        log_variance = torch.clamp(log_variance, -30, 20)

        # (b, 4, h/8, w/8 ) -> 2x (b, 4, h/8, w/8)
        variance = log_variance.exp()

        # (b, 4, h/8, w/8 ) -> 2x (b, 4, h/8, w/8)
        stdev = variance.sqrt()

        # Z=N(0,1) -> X = N(mean, variance) Z -> X transform of gaussians
        # X = mean + stdev * Z
        x = mean + stdev * noise

        # Scale the output by a constant.
        # This magic number is taken from orig paper.
        x *= 0.18215

        return x
