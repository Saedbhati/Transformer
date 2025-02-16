from .Attention import MultiHeadAttention
from .FeedForward import FeedForward
import torch
from torch import nn

class EncoderLayer(nn.Module):
    def __init__(self, embed_size, num_heads, ff_dim):
        super(EncoderLayer, self).__init__()
        self.self_attention = MultiHeadAttention(embed_size, num_heads)
        self.FeedForward=FeedForward(embed_size,ff_dim)

    def forward(self, x):
        x = self.self_attention(x)
        x = self.FeedForward(x)
        return x