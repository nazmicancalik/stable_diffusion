import torch
from torch.nn import nn
from torch.nn import functional as F
from attention import SelfAttention, CrossAttention


class Diffusion(nn.Module):

    def __init__(self):
        self.time_embedding = TimeEmbedding(320)
        self.unet = UNET()
        self.final = UNET_OutputLayer(320, 4)

    def forward(self, latent: torch.Tensor, context: torch.Tensor, time: torch.Tensor):
        # latent = (bs, 4 , height/8, width/8)
        # context = (bs, seq_len, dim)
        # time = (1, 320)

        # (1,320) -> (1, 1280)
        time = self.time_embedding(time)

        # (bs, 4, height/8, width/8) -> (bs, 320, height/8, width/8)
        output = self.unet(latent, context, time)

        # (bs, 320, height/8, width/8) -> (bs, 4, height/8, width/8) Go back to original size
        output = self.final(output)

        return output
