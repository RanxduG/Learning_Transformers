import torch
import torch.nn as nn
import torch.nn.functional as F
from models.encoder import MultiHeadAttention, LayerNormalization, PositionwiseFeedForward, PositionalEncoding

def generate_subsequent_mask(size):
    mask = torch.tril(torch.ones(size, size)).unsqueeze(0).unsqueeze(1)  # [1, 1, size, size]
    return mask  # shape: (1, 1, tgt_len, tgt_len)

class DecoderLayer(nn.Module):
    def __init__(self, d_model, hidden, num_heads, dropout):
        super().__init__()
        self.self_attn = MultiHeadAttention(d_model, num_heads)
        self.norm1 = LayerNormalization(d_model)

        self.cross_attn = MultiHeadAttention(d_model, num_heads)
        self.norm2 = LayerNormalization(d_model)

        self.ff = PositionwiseFeedForward(d_model, hidden, dropout)
        self.norm3 = LayerNormalization(d_model)

        self.dropout = nn.Dropout(dropout)


    def forward(self, x, memory, tgt_mask=None):
        x = self.norm1(x + self.dropout(self.self_attn(x, mask=tgt_mask)))
        x = self.norm2(x + self.dropout(self.cross_attn(query=x, mask=None, key_value_input=memory)))
        x = self.norm3(x + self.dropout(self.ff(x)))
        return x

class Decoder(nn.Module):
    def __init__(self, d_model, hidden, num_heads, num_layers, dropout, max_seq_len):
        super().__init__()
        self.pos = PositionalEncoding(d_model, max_seq_len)
        self.layers = nn.ModuleList([
            DecoderLayer(d_model, hidden, num_heads, dropout)
            for _ in range(num_layers)
        ])
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, memory, tgt_mask=None):
        x = self.dropout(self.pos(x))
        for layer in self.layers:
            x = layer(x, memory, tgt_mask)
        return x