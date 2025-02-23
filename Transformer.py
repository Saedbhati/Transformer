import torch
from torch import nn
import math
from .Encoder import EncoderLayer
from .Decoder import DecoderLayer
from .PositionalEncoding import PositionalEncoding

class Transformer(nn.Module):
    def __init__(self, embed_size, num_heads, ff_dim, num_layers, vocab_size):
        super(Transformer, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_size)
        self.positional_encoding = PositionalEncoding(embed_size)
        self.encoder_layers = nn.ModuleList([EncoderLayer(embed_size, num_heads, ff_dim) for _ in range(num_layers)])
        self.decoder_layers = nn.ModuleList([DecoderLayer(embed_size, num_heads, ff_dim) for _ in range(num_layers)])
        self.fc = nn.Linear(embed_size, vocab_size)
        

    def forward(self, x, y=None):
      x = self.embedding(x)
      x = self.positional_encoding(x)
      
      y = self.embedding(y.type(torch.int64)) # Cast y to LongTensor if it's not None
      y = self.positional_encoding(y)


      for layer in self.encoder_layers:
          x = layer(x)
      for layer in self.decoder_layers:
          y = layer(y, x)

      y = self.fc(y)




      return y
    
    def generate(self, src, start_token, seq_length, end_token):
        # Ensure the source has a batch dimension
        if src.dim() == 1:
            src = src.unsqueeze(0)  # shape: (1, seq_length)
        x = self.embedding(src)
        x = self.positional_encoding(x)
        for layer in self.encoder_layers:
            x = layer(x)

        # Start with start_token and add batch dimension
        output = torch.tensor([[start_token]], dtype=torch.long)  # shape: (1, 1)
        for i in range(seq_length):
            y = self.embedding(output)
            y = self.positional_encoding(y)
            for layer in self.decoder_layers:
                y = layer(y, x)
            y = self.fc(y)
            # Pick the token with the highest probability from the last time step
            y_token = y[:, -1].max(dim=1)[1].unsqueeze(1)  # shape: (1, 1)
            output = torch.cat((output, y_token), dim=1)
            if y_token.item() == end_token:
                break
        # Return output without the start token
        return output[:, 1:]
