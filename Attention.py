import torch
from torch import nn
import math

class MultiHeadAttention(nn.Module):
    def __init__(self, embed_size, num_heads):
        super(MultiHeadAttention, self).__init__()
        assert embed_size % num_heads == 0

        self.num_heads = num_heads
        self.head_dim = embed_size // num_heads
        self.embed_size = embed_size


        self.Wq = nn.Linear(embed_size, embed_size,bias=False)
        self.Wk = nn.Linear(embed_size, embed_size,bias=False)
        self.Wv = nn.Linear(embed_size, embed_size,bias=False)
        self.Wout = nn.Linear(embed_size, embed_size,bias=False)

        self.softmax = nn.Softmax(dim=-1)
        self.norm = nn.LayerNorm(embed_size)

    def forward(self, x):
        batch_size, seq_length, embed_size = x.shape


        Q = self.Wq(x)
        K = self.Wk(x)
        V = self.Wv(x)

        Q = Q.view(batch_size, seq_length, self.num_heads, self.head_dim).transpose(1, 2)
        K = K.view(batch_size, seq_length, self.num_heads, self.head_dim).transpose(1, 2)
        V = V.view(batch_size, seq_length, self.num_heads, self.head_dim).transpose(1, 2)

        attention_scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.head_dim)
        attention_scores = self.softmax(attention_scores)

        attention_output = torch.matmul(attention_scores, V)

        attention_output = attention_output.transpose(1, 2).contiguous().view(batch_size, seq_length, embed_size)

        attention_output = self.Wout(attention_output)

        return self.norm(attention_output + x)


    def cross_attention(self, x, encoder_output, src_mask=None):
        batch_size, tgt_len, embed_size = x.shape
        src_len = encoder_output.shape[1]


        Q = self.Wq(x)
        K = self.Wk(encoder_output)
        V = self.Wv(encoder_output)
        Q = Q.view(batch_size, tgt_len, self.num_heads, self.head_dim).transpose(1, 2)
        K = K.view(batch_size, src_len, self.num_heads, self.head_dim).transpose(1, 2)
        V = V.view(batch_size, src_len, self.num_heads, self.head_dim).transpose(1, 2)

        attention_scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.head_dim)

        if src_mask is not None:
            attention_scores = attention_scores.masked_fill(~src_mask, float('-inf'))

        attention_scores = self.softmax(attention_scores)
        attention_output = torch.matmul(attention_scores, V)

        attention_output = attention_output.transpose(1, 2).contiguous().view(batch_size, tgt_len, embed_size)

        return self.norm(attention_output + x)
    

class MaskedSelfAttention(nn.Module):
    def __init__(self, embed_size):
        super(MaskedSelfAttention, self).__init__()
        self.embed_size = embed_size

        self.Wq = nn.Linear(embed_size, embed_size)
        self.Wk = nn.Linear(embed_size, embed_size)
        self.Wv = nn.Linear(embed_size, embed_size)
        self.softmax = nn.Softmax(dim=-1)
        self.norm = nn.LayerNorm(embed_size)

    def forward(self, x, mask=None):
        q = self.Wq(x)
        k = self.Wk(x)
        v = self.Wv(x)

        attention_scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.embed_size)

        # Create a default mask if none is provided
        if mask is None:
            mask = torch.ones_like(attention_scores, dtype=torch.bool) # create a mask of all True values

        # Apply mask: set masked positions to a very large negative number
        attention_scores = attention_scores.masked_fill(~mask, float('-inf')) # Invert the mask for masked_fill

        attention_probs = self.softmax(attention_scores)
        attention_output = torch.matmul(attention_probs, v)
        return self.norm(attention_output + x)