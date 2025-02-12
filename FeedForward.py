import torch
from torch import nn

class FeedForward(nn.Module):
    def __init__(self, embed_size, ff_dim):
        super(FeedForward, self).__init__()
        self.fc1 = nn.Linear(embed_size, ff_dim)
        self.fc2 = nn.Linear(ff_dim, embed_size)
        self.relu = nn.ReLU()
        self.norm = nn.LayerNorm(embed_size)

    def forward(self, x):
        out = self.fc1(x)
        out = self.relu(out)
        out = self.fc2(out)
        return self.norm(out + x)