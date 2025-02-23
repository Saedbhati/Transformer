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


    def cross_attention(self, x, encoder_output):
        batch_size, tgt_len, embed_size = x.shape
        src_len = encoder_output.shape[1]


        Q = self.Wq(x)
        K = self.Wk(encoder_output)
        V = self.Wv(encoder_output)
        Q = Q.view(batch_size, tgt_len, self.num_heads, self.head_dim).transpose(1, 2)
        K = K.view(batch_size, src_len, self.num_heads, self.head_dim).transpose(1, 2)
        V = V.view(batch_size, src_len, self.num_heads, self.head_dim).transpose(1, 2)

        attention_scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.head_dim)

    

        attention_scores = self.softmax(attention_scores)
        attention_output = torch.matmul(attention_scores, V)

        attention_output = attention_output.transpose(1, 2).contiguous().view(batch_size, tgt_len, embed_size)

        return self.norm(attention_output + x)
    

class MaskedSelfAttention(nn.Module):
    def __init__(self, embed_size, num_heads):
        super(MaskedSelfAttention, self).__init__()
        self.embed_size = embed_size
        self.num_heads = num_heads
        self.head_dim = embed_size // num_heads
        self.Wq = nn.Linear(embed_size, embed_size)
        self.Wk = nn.Linear(embed_size, embed_size)
        self.Wv = nn.Linear(embed_size, embed_size)
        self.fc_out = nn.Linear(embed_size, embed_size)

        self.softmax = nn.Softmax(dim=-1)
        self.norm = nn.LayerNorm(embed_size)

    def forward(self, x, mask=None):
        N, seq_len, _ = x.shape
        
        # Create Q, K, V matrices
        Q = self.Wq(x)
        K = self.Wk(x)
        V = self.Wv(x)

        
        # Reshape into [N, num_heads, seq_len, head_dim]
        Q = Q.view(N, seq_len, self.num_heads, self.head_dim).transpose(1,2)
        K = K.view(N, seq_len, self.num_heads, self.head_dim).transpose(1,2)
        V = V.view(N, seq_len, self.num_heads, self.head_dim).transpose(1,2)
        
        # Compute scaled dot-product attention scores
        energy = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.head_dim)
        
        # Generate a causal mask and adjust its dimensions
        # The mask is True for allowed positions (current and past tokens)
        mask = torch.tril(torch.ones(seq_len, seq_len, device=x.device)).bool()
        mask = mask.unsqueeze(0).unsqueeze(0)  # shape becomes [1, 1, seq_len, seq_len]
        
        # Apply the mask: set positions that are False in mask to -inf
        energy = energy.masked_fill(~mask, float('-inf'))
        print(energy)
        
        # Softmax over the last dimension to obtain attention weights
        attention = torch.softmax(energy, dim=-1)
        
        # Compute the final attention output
        out = torch.matmul(attention, V)
        out = out.transpose(1,2).contiguous().view(N, seq_len, self.embed_size)
        out = self.fc_out(out)
        return self.norm(out + x)