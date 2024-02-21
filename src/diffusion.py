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

        time = self.time_embedding(time)
