import torch
from torch import nn
import math

class PositionalEncoding(nn.Module):
    def __init__(self, d_model=768, max_len=5000):
        super(PositionalEncoding, self).__init__()


        pe = torch.zeros(max_len, d_model)

        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)

        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        pe = pe.unsqueeze(0)

        self.register_buffer('pe', pe)

    def forward(self, x):
        """
        x: Tensor of shape (batch_size, seq_len, d_model)
        Returns positional encoding of shape (batch_size, seq_len, d_model)
        """
        # Get the sequence length from the input tensor
        seq_len = x.size(1)
        # Slice the pre-computed positional encodings to match the input sequence length
        # and expand the batch dimension to match the input batch size.
        return self.pe[:, :seq_len].expand(x.size(0), -1, -1) #This line is changed to correctly broadcast the positional embeddings
