from .Attention import MultiHeadAttention, MaskedSelfAttention
from .FeedForward import FeedForward
import torch
from torch import nn

class DecoderLayer(nn.Module):
  def __init__(self, embed_size, num_heads, ff_dim):
        super(DecoderLayer, self).__init__()
        self.masked_self_attention = MaskedSelfAttention(embed_size)
        self.self_attention = MultiHeadAttention(embed_size, num_heads)
        self.feed_forward = FeedForward(embed_size, ff_dim)

  def forward(self, x, encoder_output):
        x = self.masked_self_attention(x)
        
        src_mask = None  
        x = self.self_attention.cross_attention(x, encoder_output, src_mask)
        x = self.feed_forward(x)
        return x

